# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job Workflow modules"""
import os
import threading
import functools
import time
import uuid
import logging

from queue import PriorityQueue

from dataclasses import dataclass, field, asdict, fields
from datetime import datetime, timezone

from nvidia_tao_core.microservices.utils.handler_utils import JobContext
from nvidia_tao_core.microservices.utils.timeout_utils import (
    get_all_running_jobs,
    get_all_running_automl_experiments,
)
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    get_all_pending_jobs,
    update_job_status,
    update_job_message,
    get_handler_type,
    get_handler_metadata,
    get_automl_controller_info,
    get_automl_experiment_job_id,
    delete_dnn_status
)
from nvidia_tao_core.microservices.utils.job_utils.timeout_monitor import (
    check_job_timeout,
    terminate_timed_out_job
)
# AutoMLHandler import moved to function level to avoid circular imports
from nvidia_tao_core.microservices.utils.mongo_utils import MongoHandler
from nvidia_tao_core.microservices.utils.core_utils import read_network_config
from .dependencies import dependency_type_map, dependency_check_default

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)
logger.info(f"Logging configured at level: {TAO_LOG_LEVEL}")


def synchronized(wrapped):
    """Decorator function for synchronizing threaded functions"""
    lock = threading.Lock()

    @functools.wraps(wrapped)
    def _wrap(*args, **kwargs):
        with lock:
            return wrapped(*args, **kwargs)
    return _wrap


@dataclass
class IdedItem:
    """Base class for representing id's in uuid"""

    id: uuid.UUID = field(default=uuid.uuid4())


@dataclass(order=True)
class PrioritizedItem:
    """Base class for prioritizing items"""

    priority: int = field(default=1)
    created_on: str = field(default=datetime.now(tz=timezone.utc))


@dataclass
class Dependency:
    """Base class for representing dependecies"""

    type: str = field(default=None)
    name: str = field(default=None)
    num: int = field(default=1)


@dataclass
class Job(PrioritizedItem, IdedItem):
    """Class for representing jobs"""

    last_modified: str = field(compare=False, default=datetime.now(tz=timezone.utc))
    action: str = field(compare=False, default=None)
    dependencies: list = field(compare=False, default=None)
    # More parameters for Job from JobContext
    parent_id: uuid.UUID = field(compare=False, default=uuid.uuid4())
    network: str = field(compare=False, default=None)
    handler_id: uuid.UUID = field(compare=False, default=uuid.uuid4())
    user_id: uuid.UUID = field(compare=False, default=uuid.uuid4())
    org_name: str = field(compare=False, default=None)
    kind: str = field(compare=False, default=None)
    specs: dict = field(compare=False, default=None)
    num_gpu: int = field(compare=False, default=0)
    backend_details: dict = field(compare=False, default=None)
    workflow_status: str = field(compare=False, default=None)
    retain_checkpoints_for_resume: bool = field(compare=False, default=False)
    early_stop_epoch: int = field(compare=False, default=None)
    timeout_minutes: int = field(compare=False, default=None)
    backend_type: str = field(compare=False, default=None)


def dependency_check(job_context, dependency):
    """Checks if depencies for the job are met"""
    dependency_check_fn = dependency_type_map.get(dependency.type, dependency_check_default)
    dependency_met = dependency_check_fn(job_context, dependency)
    return dependency_met


def execute_job(job_context):
    """Starts a thread on pipelines present in actions.py

    Returns:
        bool: True if job can be dequeued (successfully started or doesn't need GPUs)
              False if job should stay in queue (GPU assignment failed)
    """
    # CRITICAL FIX: Pre-assign GPUs BEFORE starting async thread
    # This prevents race condition where job is dequeued before we discover GPUs aren't available
    # NOTE: Skip GPU pre-assignment for SLURM jobs as SLURM manages its own GPU scheduling
    from nvidia_tao_core.microservices.utils.stateless_handler_utils import BACKEND
    from nvidia_tao_core.microservices.enum_constants import Backend

    # Check if this job is using a SLURM workspace
    is_slurm_job = False
    try:
        from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
            get_handler_job_metadata
        )
        job_metadata = get_handler_job_metadata(str(job_context.id))
        if job_metadata:
            # Try to get workspace_id - could be in either field
            workspace_id = job_metadata.get("workspace_id") or job_metadata.get("platform_id")

            # If not in job metadata, check handler/experiment metadata
            if not workspace_id:
                handler_id = job_metadata.get("experiment_id") or job_metadata.get("handler_id")
                kind = job_metadata.get("kind", "experiment")
                if handler_id:
                    handler_metadata = get_handler_metadata(handler_id, kind + "s")
                    if handler_metadata:
                        workspace_id = handler_metadata.get("workspace") or handler_metadata.get("platform_id")

            if workspace_id:
                workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
                if workspace_metadata:
                    # Check both top-level and nested cloud_type (for robustness)
                    cloud_type = workspace_metadata.get("cloud_type")
                    if not cloud_type:
                        # Fallback: check inside cloud_specific_details
                        cloud_type = workspace_metadata.get("cloud_specific_details", {}).get("cloud_type")

                    if cloud_type == "slurm":
                        is_slurm_job = True
                        logger.debug(
                            f"[WORKFLOW] Job {job_context.id} is using SLURM workspace, "
                            f"skipping GPU pre-assignment (SLURM manages its own scheduling)"
                        )
    except Exception as e:
        logger.debug(f"[WORKFLOW] Could not check workspace cloud_type for job {job_context.id}: {e}")

    if BACKEND == Backend.LOCAL_DOCKER and not is_slurm_job:
        # Check if this job needs GPUs
        gpu_dependency = None
        for dep in job_context.dependencies:
            if dep.type == "gpu":
                gpu_dependency = dep
                break

        if gpu_dependency and gpu_dependency.num > 0:
            from nvidia_tao_core.microservices.utils.job_utils.gpu_manager import gpu_manager
            logger.debug(
                f"[WORKFLOW] Pre-assigning {gpu_dependency.num} GPU(s) for job {job_context.id} "
                f"to prevent dequeuing before we know GPUs are available"
            )
            gpu_ids = gpu_manager.assign_gpus(str(job_context.id), gpu_dependency.num)

            if not gpu_ids:
                logger.warning(
                    f"[WORKFLOW] Job {job_context.id}: GPU assignment FAILED - "
                    f"requested {gpu_dependency.num} GPU(s) but none available. "
                    f"Job will stay in queue and retry in next scan cycle."
                )
                return False  # Don't dequeue - job stays in queue for retry

            logger.debug(f"[WORKFLOW] Successfully pre-assigned GPUs {gpu_ids} to job {job_context.id}")
            # Store assigned GPU IDs in MongoDB so start_container can retrieve them
            from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
                get_handler_job_metadata,
                write_job_metadata
            )
            job_metadata = get_handler_job_metadata(str(job_context.id))
            if job_metadata:
                job_metadata["pre_assigned_gpu_ids"] = gpu_ids
                write_job_metadata(str(job_context.id), job_metadata)
                logger.debug(f"[WORKFLOW] Stored pre-assigned GPU IDs in job metadata for {job_context.id}")

    isautoml = False
    for dep in job_context.dependencies:
        if dep.type == "automl":
            isautoml = True
            break

    if not isautoml:
        # Get action, network
        action = job_context.action
        network = job_context.network
        # Get the correct ActionPipeline:
        # - build specs
        # - build run command
        # - launch K8s job
        # - monitor status
        # - run post-job steps
        network_config = read_network_config(network)
        action_pipeline_name = network_config["api_params"]["actions_pipe"].get(action, "")
        if action == "validate_images":
            action_pipeline_name = "data_services"
        if action_pipeline_name:
            from nvidia_tao_core.microservices.handlers.actions import ACTIONS_TO_FUNCTIONS
            action_pipeline = ACTIONS_TO_FUNCTIONS[action_pipeline_name]
            _Actionpipeline = action_pipeline(job_context)
            # Thread this!
            job_run_thread = threading.Thread(
                target=_Actionpipeline.run,
                args=(),
                name=f'tao-job-thread-{job_context.id}'
            )
            job_run_thread.start()
        else:
            logger.error("Action pipeline couldn't be found: %s %s %s", network_config, network, job_context)
            return False
    else:
        # AUTOML Job
        # TODO: At test time, sequentially run it and not as a thread to catch errors
        from nvidia_tao_core.microservices.handlers.actions import AutoMLPipeline
        _AutoMLPipeline = AutoMLPipeline(job_context)
        job_run_thread = threading.Thread(target=_AutoMLPipeline.run, args=(), name=f'tao-job-thread-{job_context.id}')
        job_run_thread.start()
        # AutoMLPipeline(job_context)
    return True


@synchronized
def still_exists(job_to_check):
    """Checks if the the job is yet to be executed/queued or not"""
    job_metadata = get_handler_metadata(job_to_check.id, "jobs")
    if job_metadata and job_metadata.get('workflow_status', 'unknown') == 'enqueued':
        return True
    return False


@synchronized
def report_healthy(message, clear=False):
    """Writes healthy message with timestamp"""
    mongo_health = MongoHandler("tao", "health")
    now = datetime.now(tz=timezone.utc)
    health_id = str(uuid.uuid4())
    health_message = f"Healthy at {now.isoformat()}"
    if message:
        health_message += f"\n{message}"
    logger.info(health_message)
    mongo_health.upsert({'id': health_id}, {'id': health_id, 'created_on': now,
                        'message': f"Healthy at {now.isoformat()}"})
    mongo_health.create_ttl_index('created_on', 86400)


def write_job(job):
    """Writes job into DB"""
    mongo_jobs = MongoHandler("tao", "jobs")
    mongo_jobs.upsert({'id': job.id}, asdict(job))


def check_for_timed_out_jobs():
    """Check all running jobs and AutoML experiments for timeouts and terminate timed out ones

    Note: SLURM status syncing happens automatically in get_job_status() for each job,
    so no separate sync loop is needed here.
    """
    # Check if timeout monitoring is enabled
    timeout_monitoring_enabled = os.getenv("JOB_TIMEOUT_MONITORING_ENABLED", "true").lower() in ("true", "1")
    if not timeout_monitoring_enabled:
        return []

    terminated_jobs = []

    try:
        # Get all running jobs (regular jobs)
        running_jobs = get_all_running_jobs()

        # Get all running AutoML experiments
        running_automl_experiments = get_all_running_automl_experiments()

        # Combine both lists
        all_jobs_to_check = running_jobs + running_automl_experiments

        # Count brain jobs separately for logging
        brain_job_count = sum(1 for job in running_jobs if job.get('is_automl_brain', False))
        regular_job_count = len(running_jobs) - brain_job_count

        logger.info(
            f"Checking {regular_job_count} regular jobs, {brain_job_count} AutoML brain jobs, "
            f"and {len(running_automl_experiments)} AutoML experiments for timeouts"
        )

        for job_info in all_jobs_to_check:
            try:
                job_id = job_info.get("job_id")
                is_automl = job_info.get("is_automl", False)
                experiment_number = job_info.get("experiment_number", "0")

                if not job_id:
                    logger.warning(f"Skipping job with missing ID: {job_info}")
                    continue

                # Check if job is timed out
                if check_job_timeout(job_info):
                    # Terminate the job
                    if terminate_timed_out_job(job_info):
                        job_description = (
                            f"AutoML experiment {experiment_number} for job {job_id}" if is_automl else job_id
                        )
                        terminated_jobs.append(job_description)

            except Exception as e:
                logger.error(f"Error processing job {job_info}: {e}")
                continue

    except Exception as e:
        import traceback
        logger.info(traceback.print_exc())
        logger.error(f"Error checking running jobs for timeouts: {e}")

    if terminated_jobs:
        logger.info(f"Terminated {len(terminated_jobs)} timed out jobs/experiments: {terminated_jobs}")
    else:
        logger.info("No jobs or AutoML experiments terminated due to timeout")

    return terminated_jobs


def get_all_queued_jobs():
    """Gets all jobs with workflow_status of enqueued"""
    mongo_jobs = MongoHandler("tao", "jobs")
    jobs = []
    jobs_raw = mongo_jobs.find({'workflow_status': 'enqueued'})
    for job in jobs_raw:
        properties = {field.name for field in fields(Job)}
        job_keys = list(job.keys())
        for key in job_keys:
            if key not in properties:
                job.pop(key, None)
        j = Job(**job)
        j.dependencies = []
        for d in job.get('dependencies'):
            j.dependencies.append(Dependency(**d))
        jobs.append(j)

    return jobs


@synchronized
def scan_for_jobs():
    """Scans for new jobs and queues them if dependencies are met"""
    while True:
        report_healthy("Workflow has waken up", clear=True)

        # Check for inference microservice auto-deletion requests
        process_inference_microservice_auto_deletions()

        # Create global queue
        queue = PriorityQueue()
        for j in get_all_queued_jobs():
            queue.put(j)
        len_q = len(queue.queue)
        report_healthy(f"Found {len_q} pending jobs")
        # Parse to dequeue
        jobs_to_dequeue = []
        list.sort(queue.queue)
        for i in range(len(queue.queue)):
            # check dependencies
            job = queue.queue[i]
            report_healthy(f"{job.id} with action {job.action}: Checking dependencies")
            report_healthy(f"Total dependencies: {len(job.dependencies)}")
            logger.debug(
                f"[WORKFLOW] Processing job {i + 1}/{len(queue.queue)}: {job.id} "
                f"({job.network}/{job.action}) - checking {len(job.dependencies)} dependencies"
            )
            all_met = True
            pending_reason_message = ""

            # Check if this is an AutoML experiment job (not brain job) in a single pass
            automl_experiment_job_id = None
            for dep in job.dependencies:
                if dep.type == "automl":
                    recommendation_id = dep.name
                    # Map recommendation_id to actual job_id using parent_id (brain job ID)
                    brain_job_id = job.parent_id if job.parent_id else job.id
                    automl_experiment_job_id = get_automl_experiment_job_id(brain_job_id, recommendation_id)
                    break

            for dep in job.dependencies:
                logger.debug(
                    f"[WORKFLOW] Job {job.id}: Checking dependency "
                    f"type={dep.type}, name={dep.name}, num={dep.num}"
                )
                dependency_met, message = dependency_check(job, dep)
                if not dependency_met:
                    pending_reason_message += f"{message} and, "
                    report_healthy(f"Unmet dependency for job {job.id}: {dep.type} {pending_reason_message}")
                    logger.debug("job details: %s", job.__dict__)
                    logger.debug("dependency details: %s", dep.__dict__)
                    all_met = False
                # Handle permanent failures that should error out the job
                if "Parent job " in message and "errored out" in message:
                    jobs_to_dequeue.append(job)
                    break
                # Handle GPU validation failures (e.g., requested GPUs > available GPUs)
                if "GPUs requested count" in message and "is greater than" in message:
                    update_job_status(job.handler_id, job.id, status="Error", kind=job.kind + "s")
                    detailed_status_message = {
                        "status": "FAILURE",
                        "message": message
                    }
                    if automl_experiment_job_id:
                        brain_job_id = job.parent_id if job.parent_id else job.id
                        update_job_message(
                            job.handler_id, brain_job_id, kind=job.kind + "s", message=detailed_status_message,
                            automl_expt_job_id=automl_experiment_job_id, update_automl_expt=True
                        )
                    update_job_message(job.handler_id, job.id, kind=job.kind + "s", message=detailed_status_message)
                    jobs_to_dequeue.append(job)
                    break

            # Update detailed status message in response when appropriate message is available
            pending_reason_message = ''.join(pending_reason_message.rsplit(" and, ", 1))
            if automl_experiment_job_id:
                brain_job_id = job.parent_id if job.parent_id else job.id
                if pending_reason_message:
                    pending_status = {"status": "PENDING", "message": pending_reason_message}
                else:
                    pending_status = pending_reason_message
                update_job_message(
                    job.handler_id, brain_job_id, kind=job.kind + "s", message=pending_status,
                    automl_expt_job_id=automl_experiment_job_id, update_automl_expt=True
                )
            else:
                # For regular jobs and AutoML brain jobs
                update_job_message(job.handler_id, job.id, kind=job.kind + "s", message=pending_reason_message)

            # if all dependencies are met
            if all_met and still_exists(job):
                # execute job
                # check if job is still enqueued in the DB
                report_healthy(f"{job.id} with action {job.action}: All dependencies met")
                logger.debug(f"[WORKFLOW] Job {job.id}: All dependencies met, calling execute_job()")
                if execute_job(job):
                    # dequeue job
                    logger.debug(f"[WORKFLOW] Job {job.id}: execute_job() returned True, will be dequeued")
                    jobs_to_dequeue.append(job)
                else:
                    logger.warning(f"[WORKFLOW] Job {job.id}: execute_job() returned False, will NOT be dequeued")
                    report_healthy(f"{job.id} with action {job.action}: Job execution failed")

        logger.debug(f"[WORKFLOW] Dequeuing {len(jobs_to_dequeue)} job(s)")
        for job in jobs_to_dequeue:
            logger.debug(f"[WORKFLOW] Dequeuing job {job.id}")
            Workflow.dequeue(job)

        # Check for timed out jobs and terminate them
        try:
            terminated_jobs = check_for_timed_out_jobs()
            if terminated_jobs:
                report_healthy(f"Terminated {len(terminated_jobs)} timed out jobs")
        except Exception as e:
            logger.error(f"Error checking for timed out jobs: {e}")

        report_healthy("Workflow going to sleep")
        time.sleep(15)


def process_inference_microservice_auto_deletions():
    """Process pending inference microservice auto-deletion requests

    This function monitors job_statuses for AUTO_DELETION_REQUESTED status from
    inference microservice pods and handles the actual deletion (requires DB access).
    """
    try:
        # Check job_statuses collection for AUTO_DELETION_REQUESTED status
        mongo_status_table = MongoHandler("tao", "job_statuses")

        # Find all jobs with AUTO_DELETION_REQUESTED in their status array
        all_status_records = mongo_status_table.find({})

        # Import here to avoid circular dependency
        from nvidia_tao_core.microservices.handlers.inference_microservice_handler import (
            InferenceMicroserviceHandler
        )

        jobs_to_delete = []

        for status_record in all_status_records:
            job_id = status_record.get("id", "")
            status_list = status_record.get("status", [])

            # Check if any status entry has AUTO_DELETION_REQUESTED
            for status_entry in status_list:
                if isinstance(status_entry, dict) and status_entry.get("status") == "AUTO_DELETION_REQUESTED":
                    # Check if we haven't processed this yet
                    if not status_entry.get("auto_deletion_processed", False):
                        jobs_to_delete.append({
                            "job_id": job_id,
                            "idle_time_minutes": status_entry.get("idle_time_minutes", 0),
                            "reason": status_entry.get("reason", "unknown")
                        })
                        break

        if not jobs_to_delete:
            return

        logger.info(f"Found {len(jobs_to_delete)} inference microservices requesting auto-deletion")

        for job_info in jobs_to_delete:
            job_id = job_info["job_id"]
            try:
                logger.info(
                    f"Processing auto-deletion for inference microservice job {job_id}. "
                    f"Reason: {job_info.get('reason', 'unknown')}, "
                    f"Idle time: {job_info.get('idle_time_minutes', 0):.2f} minutes"
                )

                result = InferenceMicroserviceHandler.stop_inference_microservice(
                    job_id,
                    auto_deletion=True,
                    reason=job_info.get("reason", "unknown"),
                )

                if result.code == 200:
                    logger.info(f"Auto-deletion successful for job {job_id}")
                    # Note: Job status is already updated to Done in stop_inference_microservice

                    # Mark status as processed to avoid re-processing
                    status_record = mongo_status_table.find_one({"id": job_id})
                    if status_record:
                        status_list = status_record.get("status", [])
                        for status_entry in status_list:
                            if (isinstance(status_entry, dict) and
                                    status_entry.get("status") == "AUTO_DELETION_REQUESTED"):
                                status_entry["auto_deletion_processed"] = True
                                status_entry["processed_at"] = datetime.now(tz=timezone.utc).isoformat()
                        mongo_status_table.upsert({"id": job_id}, {"id": job_id, "status": status_list})
                else:
                    logger.error(f"Auto-deletion failed for job {job_id}: {result.data}")
                    # Mark as processed even on failure to avoid retry loops
                    status_record = mongo_status_table.find_one({"id": job_id})
                    if status_record:
                        status_list = status_record.get("status", [])
                        for status_entry in status_list:
                            if (isinstance(status_entry, dict) and
                                    status_entry.get("status") == "AUTO_DELETION_REQUESTED"):
                                status_entry["auto_deletion_processed"] = True
                                status_entry["failed"] = True
                                status_entry["error"] = str(result.data)
                        mongo_status_table.upsert({"id": job_id}, {"id": job_id, "status": status_list})

            except Exception as e:
                logger.error(f"Error processing auto-deletion for job {job_id}: {e}")
                # Mark as processed with error to avoid retry loops
                status_record = mongo_status_table.find_one({"id": job_id})
                if status_record:
                    status_list = status_record.get("status", [])
                    for status_entry in status_list:
                        if isinstance(status_entry, dict) and status_entry.get("status") == "AUTO_DELETION_REQUESTED":
                            status_entry["auto_deletion_processed"] = True
                            status_entry["failed"] = True
                            status_entry["error"] = str(e)
                    mongo_status_table.upsert({"id": job_id}, {"id": job_id, "status": status_list})

    except Exception as e:
        logger.error(f"Error in process_inference_microservice_auto_deletions: {e}")


class Workflow:
    """Workflow is an abstraction that can run on multiple threads.

    Its use is to be able to perform dependency checks and spawn off K8s jobs
    Currently, jobs are packaged inside the ActionPipeline that runs as a thread.
    On application restart, it will check if there were any pending job
    monitoring threads that were interrupted and restart them.
    """

    @staticmethod
    def restart_threads():
        """Method used to restart unfinished job monitoring threads"""
        jobs = get_all_pending_jobs()
        automl_brain_restarted = False
        for job_dict in jobs:
            parent_job_id = job_dict.get("parent_id")
            action = job_dict.get("action")
            job_id = job_dict.get("id")
            name = job_dict.get("name")
            org_name = job_dict.get("org_name")
            specs = job_dict.get("specs")
            backend_details = job_dict.get("backend_details")
            retain_checkpoints_for_resume = job_dict.get("retain_checkpoints_for_resume", False)
            early_stop_epoch = job_dict.get("early_stop_epoch", None)
            timeout_minutes = job_dict.get("timeout_minutes", None)
            if 'experiment_id' in job_dict:
                kind = 'experiment'
                handler_id = job_dict['experiment_id']
            elif 'dataset_id' in job_dict:
                kind = 'dataset'
                handler_id = job_dict['dataset_id']
            elif 'workspace_id' in job_dict:
                kind = 'workspace'
                handler_id = job_dict['workspace_id']
            else:
                logger.error(
                    "Warning: Job %s monitoring unable to be restarted, "
                    "cannot determine handler kind",
                    job_id
                )
                continue

            handler_metadata = get_handler_metadata(handler_id, kind)
            if not handler_metadata:
                logger.error(
                    "Warning: Job %s monitoring unable to be restarted, "
                    "cannot find %s %s",
                    job_id,
                    kind,
                    handler_id
                )
                continue
            network = get_handler_type(handler_metadata)
            user_id = handler_metadata.get("user_id")
            if not org_name:
                if "org_name" not in handler_metadata:
                    logger.error(
                        "Warning: Job %s monitoring unable to be restarted, "
                        "cannot determine org name",
                        job_id
                    )
                    continue
                org_name = handler_metadata.get("org_name")
            num_gpu = handler_metadata.get("num_gpu", -1)
            isautoml = handler_metadata.get("automl_settings", {}).get("automl_enabled", False)

            job_context = JobContext(
                job_id,
                parent_job_id,
                network,
                action,
                handler_id,
                user_id,
                org_name,
                kind,
                name=name,
                num_gpu=num_gpu,
                specs=specs,
                retain_checkpoints_for_resume=retain_checkpoints_for_resume,
                early_stop_epoch=early_stop_epoch,
                timeout_minutes=timeout_minutes,
                backend_details=backend_details
            )
            # If job has yet to be executed, skip monitoring
            if still_exists(job_context):
                continue
            logger.error(
                "Found unfinished monitoring thread for job %s, restarting job thread now",
                job_id
            )

            if not isautoml:
                # Reset timeout timer by clearing old status history for restarted jobs
                delete_dnn_status(job_id, automl=False)
                logger.info(f"Cleared status history for restarted job {job_id} to reset timeout timer")
                # Get the correct ActionPipeline and monitor status
                network_config = read_network_config(network)
                action_pipeline_name = network_config["api_params"]["actions_pipe"].get(action, "")
                if action_pipeline_name:
                    from nvidia_tao_core.microservices.handlers.actions import ACTIONS_TO_FUNCTIONS
                    action_pipeline = ACTIONS_TO_FUNCTIONS[action_pipeline_name]

                    _Actionpipeline = action_pipeline(job_context)
                    # Thread this!
                    job_run_thread = threading.Thread(
                        target=_Actionpipeline.monitor_job,
                        args=(),
                        name=f'tao-monitor-job-thread-{job_context.id}'
                    )
                    job_run_thread.start()
                    logger.info("Monitoring thread for job %s restarted", job_id)
                else:
                    logger.error("Action pipeline couldn't be found: %s %s %s", network_config, network, job_dict)
            else:
                # Restart AutoML job monitoring threads
                recommendations = get_automl_controller_info(job_id)
                handler_metadata = get_handler_metadata(handler_id, kind + "s")
                if handler_metadata:
                    if not automl_brain_restarted:
                        from nvidia_tao_core.microservices.handlers.automl_handler import AutoMLHandler
                        AutoMLHandler.resume(user_id, org_name, handler_id, job_id, handler_metadata, name=name)
                        automl_brain_restarted = True
                    for recommendation in recommendations:
                        if (recommendation.get("status", None) in ("pending", "running", "started") and
                                recommendation.get("id", None)):
                            rec_id = recommendation["id"]
                            deps = [Dependency(type="automl", name=str(rec_id))]
                            # Get retain_checkpoints_for_resume from job metadata (same as parent job)
                            automl_context = JobContext(
                                job_id,
                                parent_job_id,
                                network,
                                action,
                                handler_id,
                                user_id,
                                org_name,
                                kind,
                                name=name,
                                num_gpu=num_gpu,
                                retain_checkpoints_for_resume=retain_checkpoints_for_resume,
                                early_stop_epoch=early_stop_epoch,
                                timeout_minutes=timeout_minutes,
                                backend_details=backend_details
                            )
                            automl_context.dependencies = deps
                            from nvidia_tao_core.microservices.handlers.actions import AutoMLPipeline
                            _AutoMLPipeline = AutoMLPipeline(automl_context)
                            job_run_thread = threading.Thread(
                                target=_AutoMLPipeline.monitor_job,
                                args=(),
                                name=f'tao-monitor-job-thread-{automl_context.id}'
                            )
                            job_run_thread.start()
                            logger.info(
                                f"Restarted AutoML monitoring thread for job {job_id} "
                                f"and recommendation {rec_id}"
                            )

    @staticmethod
    def start():
        """Method used to initialize the workflow. Starts a thread if thread is not there from before"""
        # Make sure there is no other Workflow thread
        for thread in threading.enumerate():
            if thread.name == "WorkflowThreadTAO":
                return False
        # Restart unfinished monitoring threads, if any
        Workflow.restart_threads()
        t = threading.Thread(target=scan_for_jobs)
        t.name = 'WorkflowThreadTAO'
        t.daemon = True
        t.start()
        return True

    @staticmethod
    def enqueue(job):
        """Method used from outside to put a job into the workflow"""
        # Called only by on_new_job()
        job.workflow_status = 'enqueued'
        write_job(job)

    @staticmethod
    def dequeue(job):
        """Method used from outside to remove a job from the workflow"""
        # Simply remove the job from the filename
        # Read all jobs
        job.workflow_status = 'dequeued'
        write_job(job)

    @staticmethod
    def healthy():
        """Method used to see if the workflow thread is running"""
        try:
            mongo_health = MongoHandler("tao", "health")
            health_record = mongo_health.find_latest()
            if not health_record.get("created_on"):
                return False
            last_updated_time = datetime.now(tz=timezone.utc) - health_record.get("created_on")
            total_seconds = last_updated_time.total_seconds()
            if total_seconds > 3600:
                logger.error("Health file was updated %s ago which is > 3600", total_seconds)
            return total_seconds <= 3600
        except Exception as e:
            logger.error(str(e))
            return False
