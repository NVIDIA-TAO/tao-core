# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job timeout monitoring and termination utilities"""
import os
import logging
from datetime import datetime, timezone

from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    get_dnn_status,
    get_handler_job_metadata,
    update_job_status,
    save_automl_controller_info,
    get_automl_controller_info,
    internal_job_status_update,
    get_health_beat
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

# Track when we first detected jobs with no timestamp (for timeout purposes)
# Format: {job_id: first_detection_time}
_no_timestamp_job_tracker = {}


def get_last_status_timestamp(job_id, automl=False, experiment_number="0"):
    """Get the timestamp of the last status update for a job"""
    try:
        status_data = get_dnn_status(job_id, automl=automl, experiment_number=experiment_number)
        if not status_data:
            logger.info(f"No status data found for job {job_id}")
            return None

        # Find the most recent timestamp in the status data
        latest_timestamp = None
        for status_entry in status_data:
            if isinstance(status_entry, dict) and 'timestamp' in status_entry:
                try:
                    timestamp_str = status_entry['timestamp']
                    if isinstance(timestamp_str, str):
                        # Try different timestamp formats
                        for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S.%f']:
                            try:
                                timestamp = datetime.strptime(timestamp_str, fmt).replace(tzinfo=timezone.utc)
                                break
                            except ValueError:
                                continue
                        else:
                            # If no format matches, try parsing as ISO format
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        continue

                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                except (ValueError, TypeError) as e:
                    logger.info(f"Failed to parse timestamp {status_entry.get('timestamp')} for job {job_id}: {e}")
                    continue

        return latest_timestamp

    except Exception as e:
        logger.error(f"Error getting last status timestamp for job {job_id}: {e}")
        return None


def check_pod_liveness(job_id):
    """Check if a pod/container is alive by hitting its liveness endpoint

    Supports both Kubernetes and Docker Compose backends.

    Returns:
        bool: True if pod/container is alive and responding, False otherwise
    """
    try:
        import requests
        from nvidia_tao_core.microservices.utils.stateless_handler_utils import BACKEND

        port = 8000

        if BACKEND == Backend.LOCAL_DOCKER:
            # Docker Compose: use container name directly
            # Container name is the same as job_id
            liveness_url = f"http://{job_id}:{port}/api/v1/health/liveness"
            logger.info(f"Checking liveness for Docker container {job_id} at {liveness_url}")
        else:
            # Kubernetes: use service name with namespace
            from nvidia_tao_core.microservices.utils.handler_utils import get_statefulset_service_name
            service_name = get_statefulset_service_name(job_id)
            namespace = os.getenv('NAMESPACE', 'default')

            # Format: http://service-name.namespace.svc.cluster.local:8000/api/v1/health/liveness
            liveness_url = f"http://{service_name}.{namespace}.svc.cluster.local:{port}/api/v1/health/liveness"
            logger.info(f"Checking liveness for K8s job {job_id} at {liveness_url}")

        # Make request with short timeout
        response = requests.get(liveness_url, timeout=5)

        if response.status_code == 200:
            logger.info(f"Job {job_id} is alive and responding to liveness checks (backend: {BACKEND})")
            return True

        logger.info(f"Job {job_id} liveness check returned status {response.status_code}")
        return False

    except requests.exceptions.RequestException as e:
        logger.info(f"Job {job_id} liveness check failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking liveness for job {job_id}: {e}")
        return False


def check_brain_job_timeout(job_info):
    """Check if an AutoML brain job has timed out based on health beats

    Brain jobs are K8s jobs (not StatefulSets), so they don't have liveness endpoints.
    Instead, we rely on health beats reported by the controller during its execution loop.

    Args:
        job_info: Dictionary containing job information including job_id

    Returns:
        bool: True if brain job has timed out, False otherwise
    """
    job_id = job_info.get('job_id')

    if not job_id:
        return False

    try:
        # Use per-job timeout if available, otherwise use global timeout
        timeout_minutes = job_info.get('timeout_minutes') or 60
        timeout_seconds = timeout_minutes * 60

        # Get the last health beat timestamp
        health_beat = get_health_beat(job_id)

        if health_beat is None:
            # No health beat found - brain might not have started yet
            # Check job creation time
            job_metadata = get_handler_job_metadata(job_id)
            if job_metadata:
                created_on = job_metadata.get("created_on")
                last_modified = job_metadata.get("last_modified")

                job_start_time = None
                for timestamp_field in [last_modified, created_on]:
                    if timestamp_field:
                        try:
                            if isinstance(timestamp_field, str):
                                job_start_time = datetime.fromisoformat(timestamp_field.replace('Z', '+00:00'))
                            elif isinstance(timestamp_field, datetime):
                                job_start_time = timestamp_field
                            break
                        except (ValueError, TypeError):
                            continue

                if job_start_time:
                    current_time = datetime.now(tz=timezone.utc)
                    time_since_start = current_time - job_start_time

                    if time_since_start.total_seconds() > timeout_seconds:
                        timeout_message = (
                            f"AutoML brain job timed out: no health beats received for "
                            f"{time_since_start.total_seconds():.0f}s (timeout: {timeout_seconds}s). "
                            "Controller may have failed to start. Terminating."
                        )
                        logger.warning(f"Brain job {job_id}: {timeout_message}")

                        # Update job status
                        internal_job_status_update(
                            job_id=job_id,
                            automl=False,
                            message=timeout_message,
                            status="FAILURE",
                            handler_id=job_info.get('handler_id'),
                            kind=job_info.get('kind')
                        )
                        return True

                    logger.info(
                        f"Brain job {job_id} has no health beats yet but started only "
                        f"{time_since_start.total_seconds():.0f}s ago. Giving it time to initialize."
                    )
                    return False

            logger.info(f"Brain job {job_id} has no health beats and cannot determine start time. Being conservative.")
            return False

        # We have a health beat - check if it's recent
        last_beat_time = health_beat.get('last_beat')
        if last_beat_time:
            if isinstance(last_beat_time, str):
                last_beat_time = datetime.fromisoformat(last_beat_time.replace('Z', '+00:00'))

            current_time = datetime.now(tz=timezone.utc)
            time_since_beat = current_time - last_beat_time

            if time_since_beat.total_seconds() > timeout_seconds:
                timeout_message = (
                    f"AutoML brain job timed out: last health beat was {time_since_beat.total_seconds():.0f}s ago "
                    f"(timeout: {timeout_seconds}s). Controller appears to be stuck. Terminating."
                )
                logger.warning(f"Brain job {job_id}: {timeout_message}")

                # Update job status
                internal_job_status_update(
                    job_id=job_id,
                    automl=False,
                    message=timeout_message,
                    status="FAILURE",
                    handler_id=job_info.get('handler_id'),
                    kind=job_info.get('kind')
                )
                return True

            logger.info(
                f"Brain job {job_id} last health beat: {time_since_beat.total_seconds():.0f}s ago "
                f"(timeout: {timeout_seconds}s). Message: {health_beat.get('message', 'N/A')}"
            )
            return False

        return False

    except Exception as e:
        logger.error(f"Error checking brain job timeout for {job_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def check_job_timeout(job_info):
    """Check if a job has timed out based on last status update

    For AutoML brain jobs, uses health beats instead of pod liveness checks
    since brain jobs are K8s jobs (not StatefulSets with services).
    """
    job_id = job_info.get('job_id')
    is_automl = job_info.get('is_automl', False)
    is_automl_brain = job_info.get('is_automl_brain', False)
    experiment_number = job_info.get('experiment_number', '0')
    brain_job_id = job_info.get('brain_job_id', None)
    lookup_job_id = job_id if not brain_job_id else brain_job_id

    if not job_id:
        return False

    try:
        # Special handling for AutoML brain jobs - use health beats
        if is_automl_brain:
            return check_brain_job_timeout(job_info)

        # Use per-job timeout if available, otherwise use global timeout
        timeout_minutes = job_info.get('timeout_minutes') or 60
        timeout_seconds = timeout_minutes * 60
        last_timestamp = get_last_status_timestamp(lookup_job_id, automl=is_automl, experiment_number=experiment_number)

        if is_automl:
            job_description = f"AutoML experiment {experiment_number} for job {job_id} with brain job {brain_job_id}"
        else:
            job_description = f"Job {lookup_job_id}"

        # CASE 1: We have status updates - check if they're recent
        if last_timestamp is not None:
            # Job is sending status updates - clean up from no-timestamp tracker if present
            actual_job_id = job_id
            _no_timestamp_job_tracker.pop(actual_job_id, None)

            current_time = datetime.now(tz=timezone.utc)
            time_since_update = current_time - last_timestamp
            is_timed_out = time_since_update.total_seconds() > timeout_seconds

            if is_timed_out:
                timeout_message = (
                    f"Job timed out: last status update was {time_since_update.total_seconds():.0f}s ago "
                    f"(timeout: {timeout_seconds}s). Terminating due to inactivity."
                )
                logger.warning(f"{job_description} {timeout_message}")

                # Update job status before terminating
                internal_job_status_update(
                    job_id=lookup_job_id,
                    automl=is_automl,
                    automl_experiment_number=experiment_number,
                    message=timeout_message,
                    status="FAILURE",
                    handler_id=job_info.get('handler_id'),
                    kind=job_info.get('kind')
                )
            else:
                logger.info(
                    f"{job_description} last status update: {time_since_update.total_seconds():.0f}s ago "
                    f"(timeout: {timeout_seconds}s)"
                )

            return is_timed_out

        # CASE 2: No status updates - check if pod is alive via liveness endpoint
        # EXCEPT for SLURM and Lepton jobs which are batch jobs with no pods to check
        cloud_type = job_info.get('cloud_type')

        if cloud_type in ('slurm', 'lepton'):
            # SLURM/Lepton batch jobs have no pods and no liveness checks
            # We rely purely on job status and status.json updates
            logger.info("=" * 80)
            logger.info(f"{cloud_type.upper()} JOB TIMEOUT CHECK: {job_description}")
            logger.info(f"  Job ID: {job_id}")
            logger.info(f"  Cloud type: {cloud_type}")
            logger.info("  Status: No status updates found yet")
            logger.info(f"  ℹ {cloud_type.upper()} batch jobs have no pods/liveness checks")
            logger.info("  ℹ Job might be queued, starting, or not writing status yet")
            logger.info(f"  ℹ {cloud_type.upper()} job status will detect actual failures")
            logger.info("  ✓ Skipping timeout - relying on status.json sync")
            logger.info("=" * 80)
            # Don't timeout SLURM/Lepton jobs without status updates - they might just be starting
            # or in queue. The job status check will handle actual failures.
            return False

        logger.info(
            f"No status updates found for {job_description}. "
            "Checking if pod is alive via liveness endpoint..."
        )

        # Use actual_job_id for tracking (same as job_id for regular jobs)
        actual_job_id = job_id

        # Check if the pod is responding to liveness checks
        pod_is_alive = check_pod_liveness(job_id)

        if pod_is_alive:
            # Pod is running and responding to liveness but NOT sending status updates
            # Check how long the job has been in this state before timing out
            logger.info(
                f"{job_description} pod is alive (liveness check passed) but has sent no status updates. "
                "Checking job age to determine if timeout applies..."
            )

            # Get job creation/start time from job metadata
            # For AutoML experiments, timestamps are stored in controller info, not handler_job_metadata
            last_modified = None
            created_on = None

            if is_automl and brain_job_id:
                # AutoML experiments: fetch from controller info
                logger.debug(
                    f"{job_description} Fetching timestamps from controller info "
                    f"(brain_job_id={brain_job_id}, experiment_number={experiment_number})"
                )

                controller_info = get_automl_controller_info(brain_job_id)

                if isinstance(controller_info, list):
                    # Find the specific experiment in the controller info list
                    try:
                        experiment_num = int(experiment_number)
                        if 0 <= experiment_num < len(controller_info):
                            experiment_data = controller_info[experiment_num]
                            if isinstance(experiment_data, dict):
                                last_modified = experiment_data.get("last_modified")
                                created_on = experiment_data.get("created_on")
                                logger.debug(
                                    f"{job_description} Found timestamps in controller info: "
                                    f"last_modified={last_modified}, created_on={created_on}"
                                )
                            else:
                                logger.debug(
                                    f"{job_description} Experiment data at index {experiment_num} "
                                    f"is not a dict: {type(experiment_data)}"
                                )
                        else:
                            logger.error(
                                f"{job_description} Experiment number {experiment_num} out of range "
                                f"(controller_info has {len(controller_info)} experiments)"
                            )
                    except (ValueError, TypeError) as e:
                        logger.error(
                            f"{job_description} Error parsing experiment_number {experiment_number}: {e}"
                        )
                else:
                    logger.error(
                        f"{job_description} Controller info is not a list: {type(controller_info)}"
                    )
            else:
                # Regular jobs: fetch from handler_job_metadata
                job_metadata = get_handler_job_metadata(job_id)

                logger.debug(
                    f"{job_description} Fetching timestamps from handler_job_metadata "
                    f"(job_id={job_id})"
                )

                if job_metadata:
                    last_modified = job_metadata.get("last_modified")
                    created_on = job_metadata.get("created_on")
                    logger.debug(
                        f"{job_description} Metadata timestamps: "
                        f"last_modified={last_modified}, created_on={created_on}"
                    )
                else:
                    logger.warning(f"{job_description} No handler_job_metadata found")

            # Use the most recent timestamp (applies to both AutoML and regular jobs)
            job_start_time = None
            for timestamp_field in [last_modified, created_on]:
                if timestamp_field:
                    try:
                        if isinstance(timestamp_field, str):
                            job_start_time = datetime.fromisoformat(timestamp_field.replace('Z', '+00:00'))
                        elif isinstance(timestamp_field, datetime):
                            job_start_time = timestamp_field
                        break
                    except (ValueError, TypeError):
                        continue

            if job_start_time:
                logger.debug(
                    f"{job_description} Using job_start_time={job_start_time} "
                    f"for timeout calculation"
                )
                # Job has proper metadata - clean up from no-timestamp tracker
                _no_timestamp_job_tracker.pop(actual_job_id, None)

                current_time = datetime.now(tz=timezone.utc)
                time_since_start = current_time - job_start_time

                if time_since_start.total_seconds() > timeout_seconds:
                    timeout_message = (
                        f"Job timed out: pod is alive but has sent no status updates for "
                        f"{time_since_start.total_seconds():.0f}s (timeout: {timeout_seconds}s). "
                        "Job may be stuck or failing to report progress. Terminating."
                    )
                    logger.debug(
                        f"{job_description} {timeout_message}\n"
                        f"  Current time: {current_time}\n"
                        f"  Job start time: {job_start_time}\n"
                        f"  Elapsed: {time_since_start.total_seconds():.0f}s\n"
                        f"  Timeout threshold: {timeout_seconds}s"
                    )

                    # Update job status before terminating
                    internal_job_status_update(
                        job_id=lookup_job_id,
                        automl=is_automl,
                        automl_experiment_number=experiment_number,
                        message=timeout_message,
                        status="FAILURE",
                        handler_id=job_info.get('handler_id'),
                        kind=job_info.get('kind')
                    )
                    return True  # Timeout this job

                logger.info(
                    f"{job_description} pod is alive but no status updates yet. "
                    f"Job started {time_since_start.total_seconds():.0f}s ago. "
                    f"Giving it more time (timeout: {timeout_seconds}s)."
                )
                return False  # Still within grace period

            # If we can't determine job start time, track when we first detected this
            # and apply timeout after grace period
            current_time = datetime.now(tz=timezone.utc)

            # Check if we've been tracking this job
            if actual_job_id not in _no_timestamp_job_tracker:
                # First time detecting this condition - start tracking
                _no_timestamp_job_tracker[actual_job_id] = current_time
                logger.info(
                    f"{job_description} pod is alive but no status updates and no job timestamp. "
                    f"Starting timeout tracking. Will terminate after {timeout_seconds}s."
                )
                return False  # Give it the grace period

            # We've been tracking this job - check how long
            first_detection_time = _no_timestamp_job_tracker[actual_job_id]
            time_in_limbo = current_time - first_detection_time

            if time_in_limbo.total_seconds() > timeout_seconds:
                # Been in this state too long - timeout
                timeout_message = (
                    f"Job timed out: pod is alive but has sent no status updates for "
                    f"{time_in_limbo.total_seconds():.0f}s and job age cannot be determined. "
                    f"This indicates a stuck or orphaned job. Terminating (timeout: {timeout_seconds}s)."
                )
                logger.warning(f"{job_description} {timeout_message}")

                # Clean up tracker
                _no_timestamp_job_tracker.pop(actual_job_id, None)

                # Update job status before terminating
                internal_job_status_update(
                    job_id=lookup_job_id,
                    automl=is_automl,
                    automl_experiment_number=experiment_number,
                    message=timeout_message,
                    status="FAILURE",
                    handler_id=job_info.get('handler_id'),
                    kind=job_info.get('kind')
                )
                return True  # Timeout this job

            # Still within grace period
            logger.info(
                f"{job_description} pod is alive but no status updates/timestamp. "
                f"In limbo for {time_in_limbo.total_seconds():.0f}s. "
                f"Will timeout after {timeout_seconds}s."
            )
            return False

        # Pod is not responding yet - likely still waiting for resources or starting up
        logger.info(
            f"{job_description} pod is not responding to liveness checks yet. "
            "Likely waiting for GPU resources or still initializing. No timeout applied."
        )
        return False  # Don't timeout - job is still initializing

    except Exception as e:
        logger.error(f"Error checking timeout for job {job_id} (AutoML: {is_automl}): {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def terminate_timed_out_job(job_info):
    """Terminate a timed out job"""
    from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import ExecutionHandler
    job_id = job_info.get('job_id')
    handler_id = job_info.get('handler_id')
    kind = job_info.get('kind', '')
    is_automl = job_info.get('is_automl', False)
    is_automl_brain = job_info.get('is_automl_brain', False)
    brain_job_id = job_info.get('brain_job_id', None)
    experiment_number = job_info.get('experiment_number', '0')
    source = job_info.get('source', '')
    # Enhanced logging for termination with full context
    logger.debug(
        f"{'-' * 80}\n"
        f"TERMINATING TIMED OUT JOB\n"
        f"job_id: {job_id}\n"
        f"handler_id: {handler_id}\n"
        f"kind: {kind}\n"
        f"is_automl: {is_automl}\n"
        f"is_automl_brain: {is_automl_brain}\n"
        f"brain_job_id: {brain_job_id}\n"
        f"experiment_number: {experiment_number}\n"
        f"source: {source}\n"
        f"full_job_info: {job_info}\n"
        f"{'-' * 80}"
    )

    if not job_id:
        logger.error(f"Cannot terminate job: missing job_id in {job_info}")
        return False

    # Check if this is an orphaned job without metadata
    is_orphaned = not handler_id or source.startswith('orphaned_')

    try:
        # Handle AutoML brain jobs differently - they're K8s jobs, not StatefulSets
        if is_automl_brain:
            if is_orphaned:
                logger.warning(
                    f"Terminating orphaned AutoML brain job {job_id} (source: {source}). "
                    f"No handler_id available, will only delete K8s Job."
                )
            else:
                logger.info(f"Terminating timed out AutoML brain job {job_id}")
                # Update job status to indicate timeout (only if we have handler_id)
                if handler_id:
                    update_job_status(handler_id, job_id, status="Error", kind=kind)

            # Delete the K8s Job (not StatefulSet)
            ExecutionHandler.delete_job_with_handler(job_id)

            logger.info(f"Deletion request sent for timed out AutoML brain job {job_id}")
            return True

        if is_automl and brain_job_id:
            logger.info(f"Terminating timed out AutoML experiment {experiment_number} for job {brain_job_id}")

            # For AutoML experiments, we need to update the specific experiment status
            # and potentially terminate the StatefulSet for that experiment
            controller_info = get_automl_controller_info(brain_job_id)
            logger.info(f"Controller info for job {brain_job_id}: {controller_info}")

            if isinstance(controller_info, list):
                updated_controller = []
                experiment_found = False

                for recommendation in controller_info:
                    if isinstance(recommendation, dict):
                        if str(recommendation.get("id", "")) == experiment_number:
                            # Mark this experiment as failed due to timeout
                            recommendation["status"] = "failure"
                            recommendation["message"] = "Terminated due to timeout - no status updates received"
                            experiment_found = True
                            logger.info(f"Marked AutoML experiment {experiment_number} as error due to timeout")
                        updated_controller.append(recommendation)

                if experiment_found:
                    # Save the updated controller info
                    logger.info(
                        f"Saving updated controller info for AutoML experiment {experiment_number} "
                        f"for job {job_id}"
                    )
                    logger.info(f"Updated controller info: {updated_controller}")
                    logger.info(f"Brain job id: {brain_job_id}")
                    save_automl_controller_info(brain_job_id, updated_controller)
                    get_automl_controller_info(brain_job_id)

                    # Try to terminate the StatefulSet for this specific experiment
                    # The StatefulSet name for AutoML experiments typically includes the experiment number
                    success = ExecutionHandler.delete_with_handler(job_id)

                    if success:
                        logger.info(
                            f"Successfully terminated timed out AutoML experiment {experiment_number} for job {job_id}"
                        )
                    else:
                        logger.error(
                            f"Failed to terminate StatefulSet for AutoML experiment {experiment_number} "
                            f"for job {job_id}"
                        )

                    return success
                logger.warning(f"AutoML experiment {experiment_number} not found in controller info for job {job_id}")
                return False
            logger.error(f"Invalid controller info format for AutoML job {job_id}")
            return False

        # Regular job or orphaned job termination
        if is_orphaned:
            logger.warning(
                f"Terminating orphaned job {job_id} (source: {source}). "
                f"No handler_id available, will only delete K8s/Docker resources."
            )
        else:
            logger.info(f"Terminating timed out job {job_id}")
            # Update job status to indicate timeout (only if we have handler_id)
            if handler_id:
                update_job_status(handler_id, job_id, status="Error", kind=kind)

        # Delete the StatefulSet (works for both orphaned and regular jobs)
        success = ExecutionHandler.delete_with_handler(job_id)

        if success:
            if is_orphaned:
                logger.info(f"Successfully deleted orphaned job resources for {job_id}")
            else:
                logger.info(f"Successfully terminated timed out job {job_id}")
        else:
            logger.error(f"Failed to terminate job {job_id}")

        return success

    except Exception as e:
        logger.error(f"Error terminating timed out job {job_id} (AutoML: {is_automl}): {e}")
        return False
