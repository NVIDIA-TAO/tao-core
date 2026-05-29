# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timeout monitoring utilities for detecting and managing running jobs"""
import os
import logging

from .mongo_utils import MongoHandler
from .stateless_handler_utils import (
    BACKEND,
    get_handler_metadata,
    get_automl_controller_info,
    is_request_automl
)
from nvidia_tao_core.microservices.enum_constants import Backend

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


def _create_orphaned_job_info(job_id, job_metadata, source, is_automl_brain=None):
    """Helper function to create job info dict for orphaned pods

    Args:
        job_id: Job ID
        job_metadata: Job metadata from MongoDB
        source: Source of detection (orphaned_docker, orphaned_k8s_statefulset, orphaned_k8s_job)
        is_automl_brain: Override for automl_brain detection (None to auto-detect)

    Returns:
        dict: Job info dictionary for timeout monitoring
    """
    if is_automl_brain is None:
        is_automl_brain = is_request_automl(
            job_metadata.get('handler_id'),
            job_metadata.get('action'),
            job_metadata.get('kind', 'experiments')
        )

    return {
        'job_id': job_id,
        'handler_id': job_metadata.get('handler_id'),
        'kind': job_metadata.get('kind', ''),
        'status': 'Running',  # Override status since pod is actually running
        'user_id': job_metadata.get('user_id'),
        'org_name': job_metadata.get('org_name'),
        'action': job_metadata.get('action'),
        'network': job_metadata.get('network'),
        'last_modified': job_metadata.get('last_modified'),
        'is_automl': False,
        'is_automl_brain': is_automl_brain,
        'experiment_number': '0',
        'timeout_minutes': job_metadata.get('timeout_minutes'),
        'source': source
    }


def _find_orphaned_automl_experiment(experiment_job_id, source):
    """Helper function to find and create info for orphaned AutoML experiment pods

    Args:
        experiment_job_id: Job ID of the experiment
        source: Source of detection (orphaned_docker or orphaned_k8s_statefulset)

    Returns:
        tuple: (automl_exp_info dict or None, experiment_job_id)
    """
    mongo_jobs = MongoHandler("tao", "jobs")
    all_brain_jobs = mongo_jobs.find({'status': {'$exists': True}})

    logger.debug(f"_find_orphaned: Searching for experiment {experiment_job_id}")
    brain_count = 0
    automl_brain_count = 0
    total_experiments = 0

    for brain_job_data in all_brain_jobs:
        brain_job_id = brain_job_data.get('id')
        handler_id = brain_job_data.get('handler_id')
        brain_count += 1

        if not brain_job_id or not handler_id:
            continue

        # Check if this brain job has AutoML settings
        handler_metadata = get_handler_metadata(
            handler_id, brain_job_data.get('kind', '') + 's'
        )
        if (handler_metadata and
                handler_metadata.get("automl_settings", {}).get("automl_enabled", False)):
            automl_brain_count += 1
            controller_info = get_automl_controller_info(brain_job_id)
            brain_status = brain_job_data.get('status', 'Unknown')
            logger.debug(
                f"  Checking brain job {brain_job_id} (status={brain_status}), "
                f"has {len(controller_info) if isinstance(controller_info, list) else 0} experiments"
            )

            if isinstance(controller_info, list):
                for recommendation in controller_info:
                    if isinstance(recommendation, dict):
                        rec_job_id = recommendation.get("job_id", "")
                        rec_id = str(recommendation.get("id", ""))
                        rec_status = recommendation.get("status", "")
                        total_experiments += 1

                        if (rec_job_id == experiment_job_id and
                                rec_status not in ("pending", "running", "started")):
                            logger.warning(
                                f"Found orphaned {source} for AutoML experiment {rec_id} "
                                f"(job_id: {experiment_job_id}, brain: {brain_job_id}) with status "
                                f"{rec_status} - adding to timeout monitoring"
                            )

                            automl_exp_info = {
                                'job_id': experiment_job_id,
                                'handler_id': handler_id,
                                'kind': brain_job_data.get('kind', ''),
                                'status': 'Running',  # Override since pod is running
                                'user_id': brain_job_data.get('user_id'),
                                'org_name': brain_job_data.get('org_name'),
                                'action': brain_job_data.get('action'),
                                'network': brain_job_data.get('network'),
                                'last_modified': brain_job_data.get('last_modified'),
                                'is_automl': True,
                                'brain_job_id': brain_job_id,
                                'experiment_number': str(rec_id),
                                'timeout_minutes': brain_job_data.get('timeout_minutes'),
                                'source': source
                            }
                            return (automl_exp_info, experiment_job_id)

    logger.debug(
        f"  No match found. Checked {brain_count} jobs "
        f"({automl_brain_count} AutoML brains, {total_experiments} total experiments)"
    )
    return (None, None)


def get_all_running_jobs():
    """Returns a list of all jobs with status of Running or Pending for timeout monitoring

    This includes:
    1. Jobs with Running/Pending status in MongoDB
    2. Jobs with Done/Success status in MongoDB but still have running pods in K8s/Docker
    """
    mongo_jobs = MongoHandler("tao", "jobs")

    # First, get jobs with Running or Pending status from DB
    job_query = {
        'status': {
            '$in': ['Running', 'Pending']
        }
    }
    jobs = mongo_jobs.find(job_query)

    # Extract necessary information for timeout monitoring
    running_jobs = []
    job_ids_from_db = set()

    for job in jobs:
        automl_brain = False
        job_id = job.get('id')
        job_ids_from_db.add(job_id)

        # Check if this is an AutoML brain job by looking at the job's own metadata
        job_details = job.get('job_details', {})
        if job_id and job_id in job_details:
            # If job has automl_brain_info or automl_result, it's an AutoML brain job
            if 'automl_brain_info' in job_details[job_id] or 'automl_result' in job_details[job_id]:
                automl_brain = True

        # Fallback: check handler's current settings (for backwards compatibility)
        if not automl_brain and is_request_automl(
            job.get('handler_id'), job.get('action'), job.get('kind', 'experiments')
        ):
            automl_brain = True

        # Get workspace info to determine if this is a SLURM job
        workspace_id = job.get('workspace')
        workspace_metadata = None
        cloud_type = None
        if workspace_id:
            try:
                workspace_metadata = get_handler_metadata(workspace_id, 'workspaces')
                cloud_type = workspace_metadata.get('cloud_type') if workspace_metadata else None
            except Exception:
                pass  # Ignore errors getting workspace metadata

        # AutoML brain jobs store experiment_id instead of handler_id, and may omit kind.
        resolved_handler_id = job.get('handler_id') or job.get('experiment_id')
        resolved_kind = job.get('kind') or 'experiment'

        job_info = {
            'job_id': job_id,
            'handler_id': resolved_handler_id,
            'kind': resolved_kind,
            'status': job.get('status'),
            'user_id': job.get('user_id'),
            'org_name': job.get('org_name'),
            'action': job.get('action'),
            'network': job.get('network'),
            'last_modified': job.get('last_modified'),
            'is_automl': False,
            'is_automl_brain': automl_brain,
            'experiment_number': '0',
            'timeout_minutes': job.get('timeout_minutes'),
            'source': 'db',
            'workspace_id': workspace_id,
            'workspace_metadata': workspace_metadata,
            'cloud_type': cloud_type
        }
        running_jobs.append(job_info)

    # Now check K8s/Docker for orphaned pods
    # (pods that are running but job status is Done/Success)
    # Note: We exclude AutoML experiment job IDs - they're handled separately
    # by get_all_running_automl_experiments()
    # Collect all AutoML experiment job IDs to exclude them
    automl_experiment_job_ids = set()
    try:
        all_brain_jobs = mongo_jobs.find({'status': {'$exists': True}})
        for brain_job_data in all_brain_jobs:
            brain_job_id = brain_job_data.get('id')
            handler_id = brain_job_data.get('handler_id')
            if brain_job_id and handler_id:
                handler_metadata = get_handler_metadata(
                    handler_id, brain_job_data.get('kind', '') + 's'
                )
                if (handler_metadata and
                        handler_metadata.get("automl_settings", {}).get("automl_enabled", False)):
                    controller_info = get_automl_controller_info(brain_job_id)
                    if isinstance(controller_info, list):
                        for recommendation in controller_info:
                            if isinstance(recommendation, dict):
                                rec_job_id = recommendation.get("job_id", "")
                                if rec_job_id:
                                    automl_experiment_job_ids.add(rec_job_id)
    except Exception as e:
        logger.debug(f"Error collecting AutoML experiment job IDs: {e}")

    if BACKEND == Backend.LOCAL_DOCKER:
        # Import at function level to avoid circular imports
        from ..handlers.execution_handlers.docker_handler import get_all_docker_running_containers
        docker_containers = get_all_docker_running_containers()
        for container in docker_containers:
            job_id = container.get('job_id')
            # Skip if this is an AutoML experiment - handled separately
            if (job_id and job_id not in job_ids_from_db and
                    job_id not in automl_experiment_job_ids):
                # This container is running but not in our Running/Pending list
                # Check both jobs and automl_jobs tables
                job_metadata = mongo_jobs.find_one({'id': job_id})
                if not job_metadata:
                    # Try automl_jobs table
                    mongo_automl_jobs = MongoHandler("tao", "automl_jobs")
                    job_metadata = mongo_automl_jobs.find_one({'id': job_id})

                if job_metadata:
                    logger.warning(
                        f"Found orphaned Docker container {job_id} with status "
                        f"{job_metadata.get('status')} - adding to timeout monitoring"
                    )
                    job_info = _create_orphaned_job_info(
                        job_id, job_metadata, 'orphaned_docker'
                    )
                    running_jobs.append(job_info)
    else:
        # Check K8s resources
        # Import at function level to avoid circular imports
        from .executor_utils import get_all_k8s_running_resources
        k8s_resources = get_all_k8s_running_resources()

        # Check StatefulSets
        for ss in k8s_resources.get('statefulsets', []):
            job_id = ss.get('job_id')
            # Skip if this is an AutoML experiment - handled separately
            if (job_id and job_id not in job_ids_from_db and
                    job_id not in automl_experiment_job_ids):
                # This StatefulSet is running but not in our Running/Pending list
                logger.info(
                    f"Orphaned StatefulSet candidate: {job_id} "
                    f"(not in Running/Pending, not in automl_experiments)"
                )
                # Check both jobs and automl_jobs tables
                job_metadata = mongo_jobs.find_one({'id': job_id})
                if not job_metadata:
                    # Try automl_jobs table
                    mongo_automl_jobs = MongoHandler("tao", "automl_jobs")
                    job_metadata = mongo_automl_jobs.find_one({'id': job_id})
                    if job_metadata:
                        logger.debug("Found in automl_jobs table")

                if job_metadata:
                    logger.debug(
                        f"  Found in MongoDB with status: {job_metadata.get('status')}"
                    )
                    logger.warning(
                        f"Found orphaned StatefulSet {job_id} with DB status "
                        f"{job_metadata.get('status')} - adding to timeout monitoring"
                    )
                    job_info = _create_orphaned_job_info(
                        job_id, job_metadata, 'orphaned_k8s_statefulset'
                    )
                    running_jobs.append(job_info)
                else:
                    logger.warning(
                        f"StatefulSet {job_id} running but NOT found in jobs OR automl_jobs! "
                        f"True ghost experiment - cannot monitor"
                    )

        # Check K8s Jobs (AutoML brain jobs)
        for k8s_job in k8s_resources.get('jobs', []):
            job_id = k8s_job.get('job_id')
            if job_id and job_id not in job_ids_from_db:
                # This K8s Job is running but not in our Running/Pending list
                job_metadata = mongo_jobs.find_one({'id': job_id})
                if job_metadata:
                    logger.warning(
                        f"Found orphaned K8s Job {job_id} with DB status "
                        f"{job_metadata.get('status')} - adding to timeout monitoring"
                    )
                    # K8s Jobs are always AutoML brain jobs
                    job_info = _create_orphaned_job_info(
                        job_id, job_metadata, 'orphaned_k8s_job', is_automl_brain=True
                    )
                    running_jobs.append(job_info)

    return running_jobs


def get_all_running_automl_experiments():
    """Returns a list of all running AutoML experiment jobs for timeout monitoring

    This includes:
    1. AutoML experiments with running/pending/started status in DB
    2. AutoML experiment pods that are running in K8s/Docker but marked as success/done in DB
    """
    running_automl_experiments = []
    tracked_experiment_job_ids = set()

    try:
        # Get all running regular jobs first to find AutoML brain jobs
        regular_jobs = get_all_running_jobs()
        logger.debug(f"Checking {len(regular_jobs)} running jobs for AutoML brain jobs")

        automl_brain_count = 0
        for job in regular_jobs:
            job_id = job.get('job_id')
            handler_id = job.get('handler_id')

            if not job_id:
                continue

            # AutoML brain jobs store experiment_id (not handler_id) and may omit kind.
            # Fall back to reading experiment_id from the raw job document if handler_id is missing.
            if not handler_id:
                raw_job = MongoHandler("tao", "jobs").find_one({"id": job_id})
                if raw_job:
                    handler_id = raw_job.get("experiment_id") or raw_job.get("handler_id")
                    if handler_id:
                        job['handler_id'] = handler_id
                    if not job.get('kind'):
                        job['kind'] = raw_job.get("kind", "experiment")
                if not handler_id:
                    continue

            # Check if this is an AutoML job by looking at handler metadata
            kind_key = job.get('kind', '') or 'experiment'
            try:
                handler_metadata = get_handler_metadata(
                    handler_id, kind_key + 's'
                )
                if (handler_metadata and
                        handler_metadata.get("automl_settings", {}).get("automl_enabled", False)):
                    automl_brain_count += 1
                    # This is an AutoML brain job, get its running experiments
                    controller_info = get_automl_controller_info(job_id)
                    logger.debug(
                        f"Found AutoML brain job {job_id}, "
                        f"has {len(controller_info) if isinstance(controller_info, list) else 0} experiments"
                    )

                    if isinstance(controller_info, list):
                        for recommendation in controller_info:
                            if isinstance(recommendation, dict):
                                rec_status = recommendation.get("status", "")
                                rec_id = str(recommendation.get("id", ""))
                                rec_job_id = recommendation.get("job_id", "")
                                is_running = rec_status in ("pending", "running", "started") and rec_id

                                logger.debug(
                                    f"  Experiment {rec_id}: job_id={rec_job_id}, "
                                    f"status={rec_status}, is_running={is_running}"
                                )

                                # Track this experiment job_id
                                if rec_job_id:
                                    tracked_experiment_job_ids.add(rec_job_id)

                                # Check if this recommendation/experiment is running
                                if is_running:
                                    automl_exp_info = {
                                        'job_id': rec_job_id,
                                        'handler_id': handler_id,
                                        'kind': job.get('kind', ''),
                                        'status': rec_status,
                                        'user_id': job.get('user_id'),
                                        'org_name': job.get('org_name'),
                                        'action': job.get('action'),
                                        'network': job.get('network'),
                                        'last_modified': job.get('last_modified'),
                                        'is_automl': True,
                                        'brain_job_id': job_id,
                                        'experiment_number': str(rec_id),
                                        'timeout_minutes': job.get('timeout_minutes'),
                                        'workspace_metadata': job.get('workspace_metadata'),  # For SLURM detection
                                        'cloud_type': job.get('cloud_type'),  # For SLURM detection
                                        'source': 'db'
                                    }
                                    running_automl_experiments.append(automl_exp_info)
            except Exception as e:
                logger.error(f"Error checking AutoML status for job {job_id}: {e}")
                continue

        logger.debug(
            f"Found {automl_brain_count} AutoML brain jobs with "
            f"{len(running_automl_experiments)} running experiments"
        )

        # Now check K8s/Docker for orphaned AutoML experiment pods
        # These are pods that have completed (success/failure) in DB but are still running
        if BACKEND == Backend.LOCAL_DOCKER:
            # Import at function level to avoid circular imports
            from ..handlers.execution_handlers.docker_handler import get_all_docker_running_containers
            docker_containers = get_all_docker_running_containers()
            for container in docker_containers:
                container_job_id = container.get('job_id')
                if container_job_id and container_job_id not in tracked_experiment_job_ids:
                    automl_exp_info, exp_job_id = _find_orphaned_automl_experiment(
                        container_job_id, 'Docker container'
                    )
                    if automl_exp_info:
                        running_automl_experiments.append(automl_exp_info)
                        tracked_experiment_job_ids.add(exp_job_id)
        else:
            # Check K8s resources for orphaned AutoML experiment StatefulSets
            # Import at function level to avoid circular imports
            from .executor_utils import get_all_k8s_running_resources
            k8s_resources = get_all_k8s_running_resources()

            logger.debug(
                f"Tracked experiment job IDs (already added): {tracked_experiment_job_ids}"
            )
            for ss in k8s_resources.get('statefulsets', []):
                ss_job_id = ss.get('job_id')
                logger.debug(f"Checking StatefulSet job_id: {ss_job_id}")
                if ss_job_id and ss_job_id not in tracked_experiment_job_ids:
                    logger.debug(
                        f"Job {ss_job_id} not in tracked list, "
                        f"checking if it's an orphaned experiment"
                    )
                    automl_exp_info, exp_job_id = _find_orphaned_automl_experiment(
                        ss_job_id, 'K8s StatefulSet'
                    )
                    if automl_exp_info:
                        logger.info(
                            f"Found orphaned experiment: {exp_job_id}, adding to monitoring"
                        )
                        running_automl_experiments.append(automl_exp_info)
                        tracked_experiment_job_ids.add(exp_job_id)
                    else:
                        logger.debug(
                            f"Job {ss_job_id} is not an orphaned experiment "
                            f"(either not an AutoML experiment or still running normally)"
                        )
                else:
                    if ss_job_id:
                        logger.debug(
                            f"Job {ss_job_id} already tracked, skipping orphan check"
                        )

    except Exception as e:
        logger.error(f"Error getting running AutoML experiments: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return running_automl_experiments
