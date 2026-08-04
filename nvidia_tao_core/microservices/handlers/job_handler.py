# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job handler module for managing job operations"""
import re
import os
import glob
import tarfile
import time
import traceback
import uuid
import logging
from datetime import datetime, timezone

from nvidia_tao_core.microservices.constants import (
    MISSING_EPOCH_FORMAT_NETWORKS
)
from nvidia_tao_core.microservices.handlers.automl_handler import AutoMLHandler
from nvidia_tao_core.microservices.utils.automl_utils import apply_automl_custom_param_ranges
from nvidia_tao_core.microservices.utils.log_streaming_utils import get_job_logs_from_backend
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    check_read_access,
    check_write_access,
    get_automl_controller_info,
    get_handler_status,
    get_automl_current_rec,
    get_automl_best_rec_info,
    get_handler_job_metadata,
    get_handler_metadata,
    write_handler_metadata,
    get_handler_log_root,
    get_jobs_root,
    get_jobs_for_handler,
    is_request_automl,
    update_job_status,
    save_dnn_status,
    get_dnn_status,
    save_job_specs,
    resolve_metadata,
    resolve_existence
)
from nvidia_tao_core.microservices.utils.handler_utils import (
    Code,
    download_log_from_cloud,
    get_files_from_cloud,
    send_microservice_request
)
from nvidia_tao_core.microservices.utils.job_utils.workflow_driver import create_job_context, on_delete_job, on_new_job
from nvidia_tao_core.microservices.utils.automl_job_utils import on_delete_automl_job
from nvidia_tao_core.microservices.utils.core_utils import (
    check_and_convert
)
from nvidia_tao_core.microservices.utils.deduplication_utils import find_duplicate_job

if os.getenv("BACKEND"):
    from .mongo_handler import MongoHandler
else:
    MongoHandler = None  # type: ignore

from ..utils.basic_utils import (
    get_job,
    resolve_metadata_with_jobs,
    get_job_logs as get_job_logs_util
)

# Configure logging
logger = logging.getLogger(__name__)


class JobHandler:
    """Handles job operations including running, canceling, pausing, and retrieving job information."""

    @staticmethod
    def job_run(
        org_name,
        handler_id,
        parent_job_id,
        action,
        kind,
        specs=None,
        name=None,
        description=None,
        num_gpu=-1,
        backend_details=None,
        job_id=None,
        from_ui=False,
        retain_checkpoints_for_resume=False,
        early_stop_epoch=None,
        timeout_minutes=60,
        automl_settings=None,
        force_create=False
    ):
        """Runs a job based on the specified parameters.

        This method initiates a job based on the given organization, experiment,
        dataset details, and other job specifications. It handles different job
        scenarios including AutoML and regular jobs, and validates all necessary
        conditions before scheduling the job.

        Args:
            org_name (str): The organization name.
            handler_id (str): UUID corresponding to experiment or dataset.
            parent_job_id (str): UUID of the parent job.
            action (str): The action to be performed.
            kind (str): The type of resource ("experiment" or "dataset").
            specs (dict, optional): Specifications for the job.
            name (str, optional): The job's name.
            description (str, optional): The job's description.
            num_gpu (int, optional): The number of GPUs to allocate.
            backend_details (dict, optional): Backend-specific execution details
                (partition, cluster_name, platform_id, etc.).
            job_id (str, optional): The ID of the job to be created (auto-generated if not provided).
            from_ui (bool, optional): Indicates whether the job call is from the UI.
            retain_checkpoints_for_resume (bool, optional): Whether to retain .pth checkpoints for training resume.
            early_stop_epoch (int, optional): The epoch number to early stop training.
            timeout_minutes (int, optional): The job-specific timeout in minutes. Defaults to 60 minutes.
            force_create (bool, optional): If False, return existing job with same params. Defaults to False.

        Returns:
            Code: A response code object containing the status and job ID or error details:
                  - 200: Job successfully queued or duplicate found.
                  - 400: If job execution was unsuccessful.
                  - 404: If dataset/experiment/action not found or access is denied.
        """
        # Extract platform_id from backend_details (stored but not used in this function)
        if backend_details and backend_details.get('backend_type') == "lepton":
            _ = backend_details.get('platform_id')

        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{handler_id} {kind} doesn't exist")

        user_id = handler_metadata.get("user_id")

        # Check for duplicate job unless force_create is True
        if not force_create:
            job_params = {
                "kind": kind,
                "handler_id": handler_id,
                "action": action,
                "specs": specs,
                "parent_job_id": parent_job_id,
                "base_experiment_ids": handler_metadata.get("base_experiment_ids", []),
                "network_arch": handler_metadata.get("network_arch", ""),
                "automl_settings": automl_settings  # Include AutoML settings for proper deduplication
            }
            duplicate_job_id = find_duplicate_job(user_id, org_name, job_params)
            if duplicate_job_id:
                logger.info(
                    "Returning existing job %s for %s %s with matching params",
                    duplicate_job_id, kind, handler_id
                )
                return Code(200, duplicate_job_id, "Job with same parameters already exists")
        if not handler_metadata:
            return Code(404, [], f"{handler_id} {kind} doesn't exist")

        user_id = handler_metadata.get("user_id")
        network_arch = handler_metadata.get("network_arch", "")
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{org_name} has no write access to {handler_id}")

        if not action:
            return Code(404, [], "action not sent")

        if action not in handler_metadata.get("actions", []):
            if not (kind == "dataset" and action == "validate_images"):
                return Code(
                    404, {},
                    f"Action {action} requested not in {','.join(handler_metadata.get('actions', []))}"
                )

        if not user_id:
            return Code(
                404, [],
                "User ID couldn't be found in the experiment metadata. "
                "Try creating the experiment again"
            )

        if parent_job_id:
            parent_job_metadata = get_handler_job_metadata(parent_job_id)
            if not parent_job_metadata:
                return Code(404, [], f"Parent job {parent_job_id} not found")

            if kind != "dataset":
                parent_handler_id = parent_job_metadata.get("dataset_id")
                parent_kind = "dataset"
                if not parent_handler_id:
                    parent_kind = "experiment"
                    parent_handler_id = parent_job_metadata.get("experiment_id")
                if not parent_handler_id:
                    return Code(404, [], f"Unable to identify {parent_kind} id for parent job {parent_job_id}")

        try:
            job_id = job_id or str(uuid.uuid4())
            if specs:
                from .spec_handler import SpecHandler
                spec_schema_response = SpecHandler.get_spec_schema(user_id, org_name, handler_id, action, kind)
                if spec_schema_response.code == 200:
                    spec_schema = spec_schema_response.data
                    default_spec = spec_schema["default"]
                    check_and_convert(specs, default_spec)

            # Inject TAO_CLIENT_TYPE=api for jobs triggered via API
            if "docker_env_vars" not in handler_metadata:
                handler_metadata["docker_env_vars"] = {}
            if "TAO_CLIENT_TYPE" not in handler_metadata["docker_env_vars"]:
                handler_metadata["docker_env_vars"]["TAO_CLIENT_TYPE"] = "api"

            msg = ""
            if is_request_automl(handler_id, action, kind):
                logger.info("Creating AutoML job %s", job_id)

                # Apply custom AutoML parameter ranges from automl_settings if provided
                if automl_settings:
                    automl_range_override = automl_settings.get("automl_range_override")
                    if automl_range_override:
                        success, error_msg = apply_automl_custom_param_ranges(
                            job_id, network_arch, automl_range_override
                        )
                        if not success:
                            return Code(400, [], f"Failed to apply custom AutoML parameter ranges: {error_msg}")

                AutoMLHandler.start(
                    user_id,
                    org_name,
                    handler_id,
                    job_id,
                    handler_metadata,
                    name=name,
                    backend_details=backend_details,
                    retain_checkpoints_for_resume=retain_checkpoints_for_resume,
                    timeout_minutes=timeout_minutes,
                )
                msg = "AutoML "
            else:
                logger.info("Creating job %s", job_id)
                job_context = create_job_context(
                    parent_job_id,
                    action,
                    job_id,
                    handler_id,
                    user_id,
                    org_name,
                    kind,
                    handler_metadata=handler_metadata,
                    specs=specs,
                    name=name,
                    description=description,
                    num_gpu=num_gpu,
                    backend_details=backend_details,
                    retain_checkpoints_for_resume=retain_checkpoints_for_resume,
                    early_stop_epoch=early_stop_epoch,
                    timeout_minutes=timeout_minutes,
                )
                on_new_job(job_context)
            if specs:
                save_job_specs(job_id, specs)
            return Code(200, job_id, f"{msg}Job scheduled")
        except Exception as e:
            logger.error("Exception thrown in job_run is %s", str(e))
            logger.error(traceback.format_exc())
            try:
                handler_kind = "experiments" if kind == "experiment" else kind + "s"
                update_job_status(handler_id, job_id, status="Error", kind=handler_kind)
                from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
                    write_job_metadata
                )
                err_metadata = get_handler_job_metadata(job_id)
                if err_metadata:
                    err_metadata["status"] = "Error"
                    err_metadata.setdefault("job_details", {})[job_id] = {
                        "detailed_status": {
                            "message": f"Job creation failed: {e}"
                        }
                    }
                    write_job_metadata(job_id, err_metadata)
            except Exception as status_err:
                logger.error("Failed to update job %s status to Error: %s", job_id, status_err)
            return Code(500, [], f"Job creation failed: {e}")

    @staticmethod
    def job_retry(org_name, handler_id, kind, job_id, from_ui=False):
        """Retries a job based on its status and metadata.

        Args:
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            kind (str): Type of resource, either 'experiment' or 'dataset'.
            job_id (str): UUID of the job to retry.
            from_ui (bool): Whether the retry is triggered from the UI (default is False).

        Returns:
            Code: A response indicating the result of the retry action:
                - 200 with the new job UUID if successfully queued.
                - 400 if the job status prevents retrying.
                - 404 if the job or related resources are not found.
        """
        job_metadata = get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, [], f"Job {job_id} doesn't exist")
        status = job_metadata.get('status', 'Unknown')
        if status in ("Done", "Running", "Pending", "Canceling", "Resuming", "Pausing", "Paused"):
            return Code(400, [], f"Unable to retry job with {status} status")

        parent_job_id = job_metadata.get('parent_id')
        action = job_metadata.get('action')
        specs = job_metadata.get('specs')
        name = job_metadata.get('name', "Job") + " Retry"
        description = job_metadata.get('description')
        backend_details = job_metadata.get('backend_details')
        job_response = JobHandler.job_run(org_name, handler_id, parent_job_id, action, kind, specs, name, description,
                                          backend_details=backend_details, from_ui=from_ui)
        return job_response

    @staticmethod
    def job_get_epoch_numbers(user_id, org_name, handler_id, job_id, kind):
        """Retrieves the epoch numbers associated with a given job.

        Args:
            user_id (str): UUID of the user.
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            job_id (str): UUID of the job.
            kind (str): Type of resource, either 'experiment' or 'dataset'.

        Returns:
            Code: A response indicating the result:
                - 200 with a list of epoch numbers if found.
                - 404 if no epoch numbers are found or an error occurs.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        handler_metadata = get_handler_metadata(handler_id, kind + "s")
        if not handler_metadata or "user_id" not in handler_metadata:
            return Code(404, [], "job trying to update not found")

        job = get_job(job_id)
        if not job:
            return Code(404, [], "job trying to update not found")
        try:
            job_files, _, _, _ = get_files_from_cloud(handler_metadata, job_id)
            epoch_numbers = []
            for job_file in job_files:
                # Extract numbers before the extension using regex
                match = re.search(r'(\d+)(?=\.(pth|hdf5|tlt)$)', job_file)
                if match:
                    epoch_numbers.append(match.group(1))
            return Code(200, {"data": epoch_numbers}, "Job status updated")
        except Exception:
            logger.error(traceback.format_exc())
            return Code(404, [], "Exception caught during getting epoch numbers")

    @staticmethod
    def job_status_update(org_name, handler_id, job_id, kind, callback_data):
        """Updates the status of a given job.

        Args:
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            job_id (str): UUID of the job.
            kind (str): Type of resource, either 'experiment' or 'dataset'.
            callback_data (dict): Data containing the new status information.

        Returns:
            Code: A response indicating the result:
                - 200 if the job status was successfully updated.
                - 404 if the job or related resources are not found.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id", None)
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        job = get_job(job_id)
        action = job.get("action", "")
        if not job or not user_id:
            return Code(404, [], "job trying to update not found")

        automl = False
        experiment_number = "0"
        if is_request_automl(handler_id, action, kind) and action == "train":
            automl = True
            # Extract experiment_number from callback_data for automl jobs
            experiment_number = callback_data.get("experiment_number", "0")
        save_dnn_status(
            job_id, automl, callback_data, experiment_number=experiment_number, handler_id=handler_id, kind=kind
        )
        return Code(200, [], "Job status updated")

    @staticmethod
    def job_log_update(org_name, handler_id, job_id, kind, callback_data):
        """Appends log contents to the job's log file.

        Args:
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            job_id (str): UUID of the job.
            kind (str): Type of resource, either 'experiment' or 'dataset'.
            callback_data (dict): Data containing the log contents to append.

        Returns:
            Code: A response indicating the result:
                - 200 if the log was successfully updated.
                - 404 if the job or related resources are not found.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id", None)
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        job = get_job(job_id)
        action = job.get("action", "")
        if not job or not user_id:
            return Code(404, [], "job trying to update not found")

        handler_log_root = get_handler_log_root(user_id, org_name, handler_id)
        log_file = os.path.join(handler_log_root, job_id + ".txt")
        if is_request_automl(handler_id, action, kind):
            job_root = os.path.join(get_jobs_root(user_id, org_name), job_id)
            experiment_number = callback_data.get("experiment_number", "0")
            handler_log_root = f"{job_root}/experiment_{experiment_number}"
            log_file = f"{job_root}/experiment_{experiment_number}/log.txt"
        if not os.path.exists(handler_log_root):
            os.makedirs(handler_log_root, exist_ok=True)
        with open(log_file, "a", encoding='utf-8') as file_ptr:
            file_ptr.write(callback_data["log_contents"])
            file_ptr.write("\n")
        return Code(200, [], "Job status updated")

    @staticmethod
    def job_list(user_id, org_name, handler_id, kind):
        """Retrieves a list of jobs associated with a given handler.

        Args:
            user_id (str): UUID of the user.
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            kind (str): Type of resource, either 'experiment' or 'dataset'.

        Returns:
            Code: A response indicating the result:
                - 200 with a list of jobs if found.
                - 404 if the handler or jobs are not found.
        """
        if handler_id not in ("*", "all") and not resolve_existence(kind, handler_id):
            return Code(404, [], f"{kind} not found")

        if handler_id not in ("*", "all") and not check_read_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        return_metadata = resolve_metadata_with_jobs(user_id, org_name, kind, handler_id).get("jobs", [])

        return Code(200, return_metadata, "Jobs retrieved")

    @staticmethod
    def augment_automl_job_info(job_id, job_meta):
        """Augment job metadata with AutoML-specific information.

        This function enriches the job_meta for AutoML jobs by:
        1. Adding best experiment ID to automl_brain_info
        2. Adding best metric value to automl_result for completed jobs
        3. Adding specs to each experiment in job_details

        Parameters:
        job_id (str): UUID of the AutoML brain job
        job_meta (dict): Job metadata dictionary to augment

        Returns:
        None: Modifies job_meta in place
        """
        try:
            # Get controller data
            automl_controller_data = get_automl_controller_info(job_id)

            if not automl_controller_data:
                return

            job_details = job_meta.get("job_details", {})
            brain_job_details = job_details.get(job_id, {})
            automl_brain_info = brain_job_details.get("automl_brain_info", [])
            automl_result = brain_job_details.get("automl_result", [])

            # Get metric name for later use
            metric_name = automl_controller_data[0].get("metric")

            # Add best experiment id to automl_brain_info
            best_rec_number, _, _ = get_automl_best_rec_info(job_id)
            if best_rec_number and best_rec_number != "-1":
                # Check if best experiment id is already in the list (added by controller)
                has_best_exp_id = any(item.get("metric") == "Best experiment id" for item in automl_brain_info)
                if not has_best_exp_id:
                    # Add it if not already present (value must be string for automl_brain_info)
                    automl_brain_info.append({
                        "metric": "Best experiment id",
                        "value": str(best_rec_number)
                    })

                # Add best metric value to automl_result if not already present
                best_exp_id = int(best_rec_number)
                if best_exp_id < len(automl_controller_data):
                    best_exp_data = automl_controller_data[best_exp_id]
                    best_metric_value = best_exp_data.get("result")
                    if best_metric_value is not None and metric_name:
                        best_metric_key = f"best_{metric_name}"
                        has_best_metric = any(item.get("metric") == best_metric_key for item in automl_result)
                        if not has_best_metric:
                            automl_result.append({
                                "metric": best_metric_key,
                                "value": best_metric_value
                            })

            # Update brain_job_details with modified automl_brain_info and automl_result
            if automl_brain_info:
                brain_job_details["automl_brain_info"] = automl_brain_info
            if automl_result:
                brain_job_details["automl_result"] = automl_result

            # Add specs to each experiment job in job_details
            for experiment_details in automl_controller_data:
                exp_job_id = experiment_details.get("job_id", "")
                if exp_job_id and exp_job_id in job_details:
                    job_details[exp_job_id]["specs"] = experiment_details.get("specs", {})
        except Exception as e:
            logger.error(f"Error adding AutoML information to job_retrieve: {e}")
            logger.error(traceback.format_exc())

    @staticmethod
    def job_retrieve(org_name, handler_id, job_id, kind, return_specs=False):
        """Retrieve the specified job based on its ID and kind (experiment or dataset).

        Parameters:
        org_name (str): The name of the organization.
        handler_id (str): UUID corresponding to the experiment or dataset.
        job_id (str): UUID corresponding to the job to be retrieved.
        kind (str): The type of job, either "experiment" or "dataset".
        return_specs (bool): Flag indicating whether to return specs.

        Returns:
        Code: A response code (200 if the job is found, 404 if not found).
              - 200: Returns a dictionary following the JobResultSchema.
              - 404: Returns an empty dictionary if the job or handler is not found.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, {}, f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, {}, f"{kind} not found")

        job_meta = get_job(job_id)
        if not job_meta:
            return Code(404, {}, "Job trying to retrieve not found")
        job_meta.pop('num_gpu', None)

        # Check if this is an AutoML job
        is_automl_job = (
            job_meta.get("action") == "train" and
            handler_metadata.get("automl_settings", {}).get("automl_enabled", False)
        )

        # Remove top-level specs (not needed in response)
        # For AutoML, specs will be added per-experiment in job_details
        if not return_specs:
            if "specs" in job_meta:
                _ = job_meta.pop("specs")

        # For AutoML jobs, add AutoML-specific information
        if is_automl_job:
            JobHandler.augment_automl_job_info(job_id, job_meta)

        return Code(200, job_meta, "Job retrieved")

    @staticmethod
    def job_cancel(org_name, handler_id, job_id, kind):
        """Cancels a job based on its current status.

        Args:
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            job_id (str): UUID of the job to cancel.
            kind (str): Type of resource, either 'experiment' or 'dataset'.

        Returns:
            Code: A response indicating the result:
                - 200 if the job was successfully canceled or if the job cannot be canceled due to its current status.
                - 404 if the job or related resources are not found.
        """
        logger.debug(
            f"[CANCEL] Starting cancel operation for job_id={job_id}, handler_id={handler_id}, "
            f"kind={kind}, org_name={org_name}"
        )

        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            logger.debug(f"[CANCEL] Handler not found: handler_id={handler_id}, kind={kind}")
            return Code(404, [], f"{kind} not found")

        if job_id not in handler_metadata.get("jobs", {}).keys():
            logger.debug(f"[CANCEL] Job not found in handler's job list: job_id={job_id}, handler_id={handler_id}")
            return Code(404, [], f"Job to cancel not found in the {kind}.")

        workspace_id = handler_metadata.get("workspace")
        workspace_metadata = get_handler_metadata(workspace_id, kind="workspaces")

        user_id = handler_metadata.get("user_id")
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            logger.debug(f"[CANCEL] Write access denied: user_id={user_id}, handler_id={handler_id}")
            return Code(404, [], f"{kind} not found")

        job_metadata = get_handler_job_metadata(job_id)
        if not job_metadata:
            logger.debug(f"[CANCEL] Job metadata not found: job_id={job_id}")
            return Code(404, [], "job trying to cancel not found")
        action = job_metadata.get("action", "")

        logger.debug(
            f"[CANCEL] Job metadata retrieved: job_id={job_id}, action={action}, "
            f"status={job_metadata.get('status', 'Unknown')}"
        )

        if is_request_automl(handler_id, action, kind):
            logger.debug(f"[CANCEL] Processing AutoML cancel: job_id={job_id}, handler_id={handler_id}")
            update_job_status(handler_id, job_id, status="Canceling", kind=kind + "s")
            logger.debug(f"[CANCEL] Job status updated to Canceling: job_id={job_id}")
            automl_response = AutoMLHandler.stop(user_id, org_name, handler_id, job_id)
            if automl_response.code != 200:
                logger.warning(
                    "[CANCEL] AutoML cancellation remains pending: job_id=%s response=%s",
                    job_id, automl_response.code,
                )
                return automl_response
            # Remove any pending jobs from Workflow queue
            try:
                logger.debug(f"[CANCEL] Removing pending AutoML jobs from workflow: job_id={job_id}")
                on_delete_automl_job(job_id)
                logger.debug(f"[CANCEL] Successfully removed pending AutoML jobs: job_id={job_id}")
            except Exception as e:
                logger.error(f"[CANCEL] Exception thrown in automl_job_cancel: job_id={job_id}, error={str(e)}")
                return Code(200, {"message": f"job {job_id} cancelled, and no pending recommendations"})
            update_job_status(handler_id, job_id, status="Canceled", kind=kind + "s")
            logger.debug(f"[CANCEL] Job status updated to Canceled: job_id={job_id}")
            return automl_response

        # If job is error / done, then cancel is NoOp
        job_status = job_metadata.get("status", "Error")
        logger.debug(f"[CANCEL] Processing non-AutoML cancel: job_id={job_id}, current_status={job_status}")

        if job_status in ["Error", "Done", "Canceled", "Canceling", "Pausing", "Paused"]:
            logger.debug(f"[CANCEL] Job cannot be canceled due to status: job_id={job_id}, status={job_status}")
            return Code(
                200,
                {
                    f"Job {job_id} with current status {job_status} can't be attemped to cancel. "
                    "Current status should be one of Running, Pending, Resuming"
                }
            )
        specs = job_metadata.get("specs", None)
        use_ngc = not (specs and "cluster" in specs and specs["cluster"] == "local")
        logger.debug(f"[CANCEL] Cluster configuration: job_id={job_id}, use_ngc={use_ngc}")

        if job_status == "Pending":
            logger.debug(f"[CANCEL] Canceling pending job: job_id={job_id}")
            update_job_status(handler_id, job_id, status="Canceling", kind=kind + "s")
            logger.debug(f"[CANCEL] Deleting job from workflow queue: job_id={job_id}")
            on_delete_job(job_id)
            from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import ExecutionHandler
            ExecutionHandler.delete_with_handler(job_id)
            update_job_status(handler_id, job_id, status="Canceled", kind=kind + "s")
            logger.debug(f"[CANCEL] Pending job successfully canceled: job_id={job_id}")
            return Code(200, {"message": f"Pending job {job_id} cancelled"})

        if job_status == "Running":
            logger.debug(f"[CANCEL] Canceling running job: job_id={job_id}")
            try:
                # Delete job (K8s pod or Docker container)
                update_job_status(handler_id, job_id, status="Canceling", kind=kind + "s")
                logger.debug(f"[CANCEL] Job status updated to Canceling: job_id={job_id}")
                logger.debug(f"[CANCEL] Deleting statefulset for running job: job_id={job_id}, use_ngc={use_ngc}")
                from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import ExecutionHandler
                ExecutionHandler.delete_with_handler(job_id)
                logger.debug(f"[CANCEL] Waiting for K8s job termination: job_id={job_id}")
                k8s_status = ExecutionHandler.get_job_status_with_handler(
                    job_name=job_id,
                    workspace_metadata=workspace_metadata,
                    handler_id=handler_id,
                    handler_kind=kind + "s",
                    network="",
                    action="",
                    specs=None,
                    docker_env_vars={},
                    results_dir="",
                    automl_exp_job=False,
                    org_name=org_name
                )
                logger.debug(f"[CANCEL] Initial status after delete: job_id={job_id}, status={k8s_status}")
                while k8s_status in ("Done", "Error", "Running", "Pending"):
                    if k8s_status in ("Done", "Error"):
                        logger.debug(f"[CANCEL] Job terminated: job_id={job_id}, final_status={k8s_status}")
                        break
                    k8s_status = ExecutionHandler.get_job_status_with_handler(
                        job_name=job_id,
                        workspace_metadata=workspace_metadata,
                        handler_id=handler_id,
                        handler_kind=kind + "s",
                        network="",
                        action="",
                        specs=None,
                        docker_env_vars={},
                        results_dir="",
                        automl_exp_job=False,
                        org_name=org_name
                    )
                    logger.debug(f"[CANCEL] Polling status: job_id={job_id}, status={k8s_status}")
                    time.sleep(5)
                update_job_status(handler_id, job_id, status="Canceled", kind=kind + "s")
                logger.debug(f"[CANCEL] Running job successfully canceled: job_id={job_id}")
                return Code(200, {"message": f"Running job {job_id} cancelled"})
            except Exception as e:
                logger.error(f"[CANCEL] Exception thrown in job_cancel: job_id={job_id}, error={str(e)}")
                logger.error(f"[CANCEL] Cancel traceback: {traceback.format_exc()}")
                return Code(404, [], "job not found in platform")
        else:
            logger.debug(f"[CANCEL] Unexpected job status: job_id={job_id}, status={job_status}")
            return Code(404, [], "job status not found")

    @staticmethod
    def job_pause(org_name, handler_id, job_id, kind, graceful=True):
        """Pauses a job based on its current status.

        Args:
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            job_id (str): UUID of the job to pause.
            kind (str): Type of resource, either 'experiment' or 'dataset'.
            graceful (bool): If True, performs graceful pause by signaling job to terminate
                           and upload checkpoints before shutting down. Default is False.

        Returns:
            Code: A response indicating the result:
                - 200 if the job was successfully paused or if the job cannot be paused due to its current status.
                - 404 if the job or related resources are not found.
        """
        logger.debug(
            f"[PAUSE] Starting pause operation: job_id={job_id}, handler_id={handler_id}, "
            f"kind={kind}, org_name={org_name}, graceful={graceful}"
        )

        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            logger.debug(f"[PAUSE] Handler not found: handler_id={handler_id}, kind={kind}")
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            logger.debug(f"[PAUSE] Write access denied: user_id={user_id}, handler_id={handler_id}")
            return Code(404, [], f"{kind} not found")

        workspace_id = handler_metadata.get("workspace")
        workspace_metadata = get_handler_metadata(workspace_id, kind="workspaces")

        job_metadata = get_handler_job_metadata(job_id)
        if not job_metadata:
            logger.debug(f"[PAUSE] Job metadata not found: job_id={job_id}")
            return Code(404, [], "job trying to pause not found")
        action = job_metadata.get("action", "")

        logger.debug(
            f"[PAUSE] Job metadata retrieved: job_id={job_id}, action={action}, "
            f"status={job_metadata.get('status', 'Unknown')}"
        )

        if is_request_automl(handler_id, action, kind):
            logger.debug(f"[PAUSE] Processing AutoML pause: job_id={job_id}, handler_id={handler_id}")
            update_job_status(handler_id, job_id, status="Pausing", kind=kind + "s")
            logger.debug(f"[PAUSE] Job status updated to Pausing: job_id={job_id}")
            automl_response = AutoMLHandler.stop(
                user_id, org_name, handler_id, job_id, cleanup_checkpoints=False
            )
            if automl_response.code != 200:
                logger.warning(
                    "[PAUSE] AutoML pause remains pending: job_id=%s response=%s",
                    job_id, automl_response.code,
                )
                return automl_response
            # Remove any pending jobs from Workflow queue
            try:
                logger.debug(f"[PAUSE] Removing pending AutoML jobs from workflow: job_id={job_id}")
                on_delete_automl_job(job_id)
                logger.debug(f"[PAUSE] Successfully removed pending AutoML jobs: job_id={job_id}")
            except Exception as e:
                logger.error(f"[PAUSE] Exception thrown in automl job_pause: job_id={job_id}, error={str(e)}")
                return Code(200, {"message": f"job {job_id} paused, and no pending recommendations"})
            update_job_status(handler_id, job_id, status="Paused", kind=kind + "s")
            logger.debug(f"[PAUSE] Job status updated to Paused: job_id={job_id}")
            return automl_response

        job_action = job_metadata.get("action", "")
        if job_action not in ("train", "distill", "quantize", "retrain"):
            logger.debug(f"[PAUSE] Action not pausable: job_id={job_id}, action={job_action}")
            return Code(
                404, [],
                f"Only train, distill, quantize or retrain jobs can be paused. The current action is {job_action}"
            )
        job_status = job_metadata.get("status", "Error")
        logger.debug(f"[PAUSE] Processing non-AutoML pause: job_id={job_id}, current_status={job_status}")

        # If job is error / done, or one of cancel or pause states then pause is NoOp
        if job_status in ["Error", "Done", "Canceled", "Canceling", "Pausing", "Paused"]:
            logger.debug(f"[PAUSE] Job cannot be paused due to status: job_id={job_id}, status={job_status}")
            return Code(
                200,
                {
                    "message": (
                        f"Job {job_id} with current status {job_status} can't be attemped to pause. "
                        f"Current status should be one of Running, Pending, Resuming"
                    )
                }
            )
        specs = job_metadata.get("specs", None)
        use_ngc = not (specs and "cluster" in specs and specs["cluster"] == "local")
        logger.debug(f"[PAUSE] Cluster configuration: job_id={job_id}, use_ngc={use_ngc}")

        if job_status == "Pending":
            logger.debug(f"[PAUSE] Pausing pending job: job_id={job_id}")
            update_job_status(handler_id, job_id, status="Pausing", kind=kind + "s")
            logger.debug(f"[PAUSE] Deleting job from workflow queue: job_id={job_id}")
            on_delete_job(job_id)
            from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import ExecutionHandler
            ExecutionHandler.delete_with_handler(job_id)
            update_job_status(handler_id, job_id, status="Paused", kind=kind + "s")
            logger.debug(f"[PAUSE] Pending job successfully paused: job_id={job_id}")
            return Code(200, {"message": f"Pending job {job_id} paused"})

        if job_status == "Running":
            logger.debug(f"[PAUSE] Pausing running job: job_id={job_id}")
            try:
                update_job_status(handler_id, job_id, status="Pausing", kind=kind + "s")
                logger.debug(f"[PAUSE] Job status updated to Pausing: job_id={job_id}")

                # Try graceful pause if requested
                if graceful:
                    network = job_metadata.get("network", "")
                    action = job_metadata.get("action", "")
                    logger.debug(
                        f"[PAUSE] Attempting graceful pause: job_id={job_id}, "
                        f"network={network}, action={action}"
                    )
                    if network and action:
                        try:
                            logger.debug(
                                f"[PAUSE] Sending microservice pause request: job_id={job_id}, "
                                f"network={network}, action={action}"
                            )
                            response = send_microservice_request(
                                api_endpoint="pause_job",
                                network=network,
                                action=action,
                                specs={},
                                job_id=job_id
                            )
                            if response.status_code == 200:
                                logger.debug(
                                    f"[PAUSE] Graceful pause request successful: job_id={job_id}, "
                                    f"status_code={response.status_code}"
                                )
                                return Code(200, {
                                    "message": f"Job {job_id} is being paused gracefully. "
                                               f"Checkpoints will be uploaded and pod will terminate automatically."
                                })
                            logger.debug(
                                f"[PAUSE] Graceful pause request failed: job_id={job_id}, "
                                f"status_code={response.status_code}"
                            )
                        except Exception as e:
                            logger.error(f"[PAUSE] Graceful pause failed for job {job_id}: {str(e)}")

                    logger.warning(f"[PAUSE] Graceful pause unavailable for job {job_id}, using abrupt pause")

                # Abrupt pause (or fallback if graceful pause failed)
                from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import ExecutionHandler
                logger.debug(f"[PAUSE] Performing abrupt pause: job_id={job_id}")
                logger.debug(f"[PAUSE] Deleting statefulset for running job: job_id={job_id}, use_ngc={use_ngc}")
                ExecutionHandler.delete_with_handler(job_id)
                logger.debug(f"[PAUSE] Waiting for K8s job termination: job_id={job_id}")
                k8s_status = ExecutionHandler.get_job_status_with_handler(
                    job_name=job_id,
                    workspace_metadata=workspace_metadata,
                    handler_id=handler_id,
                    handler_kind=kind + "s",
                    network="",
                    action="",
                    specs=None,
                    docker_env_vars={},
                    results_dir="",
                    automl_exp_job=False,
                    org_name=org_name
                )
                logger.debug(f"[PAUSE] Initial status after delete: job_id={job_id}, status={k8s_status}")
                while k8s_status in ("Done", "Error", "Running", "Pending"):
                    if k8s_status in ("Done", "Error"):
                        logger.debug(f"[PAUSE] Job terminated: job_id={job_id}, final_status={k8s_status}")
                        break
                    k8s_status = ExecutionHandler.get_job_status_with_handler(
                        job_name=job_id,
                        workspace_metadata=workspace_metadata,
                        handler_id=handler_id,
                        handler_kind=kind + "s",
                        network="",
                        action="",
                        specs=None,
                        docker_env_vars={},
                        results_dir="",
                        automl_exp_job=False,
                        org_name=org_name
                    )
                    logger.debug(f"[PAUSE] Polling status: job_id={job_id}, status={k8s_status}")
                    time.sleep(5)
                update_job_status(handler_id, job_id, status="Paused", kind=kind + "s")
                return Code(200, {"message": f"Running job {job_id} paused"})
            except Exception as e:
                logger.error(f"[PAUSE] Exception thrown in job_pause: job_id={job_id}, error={str(e)}")
                logger.error(f"[PAUSE] Pause traceback: {traceback.format_exc()}")
                return Code(404, [], "job not found in platform")

        logger.debug(f"[PAUSE] Unexpected job status: job_id={job_id}, status={job_status}")
        return Code(404, [], "job status not found")

    @staticmethod
    def all_job_cancel(user_id, org_name, handler_id, kind):
        """Cancels all jobs associated with a given handler.

        Args:
            user_id (str): UUID of the user.
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            kind (str): Type of resource, either 'experiment' or 'dataset'.

        Returns:
            Code: A response indicating the result:
                - 200 if all jobs within the experiment can be canceled.
                - 404 if the handler or jobs are not found.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        def cancel_jobs_within_handler(cancel_handler_id, cancel_kind):
            cancel_success = True
            cancel_message = ""
            jobs = get_jobs_for_handler(cancel_handler_id, cancel_kind)
            for job_metadata in jobs:
                job_id = job_metadata.get("id")
                job_status = job_metadata.get("status", "Error")
                if job_status not in ["Error", "Done", "Canceled", "Canceling", "Pausing", "Paused"] and job_id:
                    cancel_response = JobHandler.job_cancel(org_name, cancel_handler_id, job_id, cancel_kind)
                    if cancel_response.code != 200:
                        if (type(cancel_response.data) is dict and
                                cancel_response.data.get("error_desc", "") != "incomplete job not found"):
                            cancel_success = False
                            cancel_message += f"Cancelation for job {job_id} failed due to {str(cancel_response.data)} "
            return cancel_success, cancel_message

        if handler_metadata.get("all_jobs_cancel_status") == "Canceling":
            return Code(200, {"message": "Canceling all jobs is already triggered"})

        try:
            handler_metadata["all_jobs_cancel_status"] = "Canceling"
            write_handler_metadata(handler_id, handler_metadata, kind)

            appended_message = ""
            for train_dataset in handler_metadata.get("train_datasets", []):
                jobs_cancel_sucess, message = cancel_jobs_within_handler(train_dataset, "dataset")
                appended_message += message

            eval_dataset = handler_metadata.get("eval_dataset", None)
            if eval_dataset:
                jobs_cancel_sucess, message = cancel_jobs_within_handler(eval_dataset, "dataset")
                appended_message += message

            inference_dataset = handler_metadata.get("inference_dataset", None)
            if inference_dataset:
                jobs_cancel_sucess, message = cancel_jobs_within_handler(inference_dataset, "dataset")
                appended_message += message

            jobs_cancel_sucess, message = cancel_jobs_within_handler(handler_id, kind)
            appended_message += message

            handler_metadata = resolve_metadata(kind, handler_id)
            if jobs_cancel_sucess:
                handler_metadata["all_jobs_cancel_status"] = "Canceled"
                write_handler_metadata(handler_id, handler_metadata, kind)
                return Code(200, {"message": "All jobs within experiment canceled"})
            handler_metadata["all_jobs_cancel_status"] = "Error"
            write_handler_metadata(handler_id, handler_metadata, kind)
            return Code(404, [], appended_message)
        except Exception as e:
            logger.error("Exception thrown in all_job_cancel is %s", str(e))
            logger.error(traceback.format_exc())
            handler_metadata["all_jobs_cancel_status"] = "Error"
            write_handler_metadata(handler_id, handler_metadata, kind)
            return Code(404, [], "Runtime exception caught during deleting a job")

    @staticmethod
    def job_delete(org_name, handler_id, job_id, kind):
        """Delete the specified job.

        Parameters:
        org_name (str): The name of the organization.
        handler_id (str): UUID corresponding to the experiment or dataset.
        job_id (str): UUID corresponding to the job to be deleted.
        kind (str): The type of job, either "experiment" or "dataset".

        Returns:
        Code: A response code:
                 - 200 if the job is successfully deleted
                 - 404 if not found
                 - 400 if deletion is not allowed
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        job_metadata = get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, [], "job trying to delete not found")

        try:
            # If job is running, cannot delete
            if job_metadata.get("status", "Error") in ["Running", "Pending"]:
                return Code(400, [], "job cannot be deleted")
            # Delete job metadata
            mongo_jobs = MongoHandler("tao", "jobs")
            mongo_jobs.delete_one({'id': job_id})
            # Delete job from handler metadata
            if "jobs" in handler_metadata and job_id in handler_metadata["jobs"]:
                del handler_metadata["jobs"][job_id]
                handler_metadata["status"] = get_handler_status(handler_metadata)
                handler_metadata["last_modified"] = datetime.now(tz=timezone.utc)
                write_handler_metadata(handler_id, handler_metadata, kind)
            # Delete job logs
            job_log_path = os.path.join(
                get_handler_log_root(user_id, org_name, handler_id),
                job_id + ".txt"
            )
            if os.path.exists(job_log_path):
                os.remove(job_log_path)
            return Code(200, job_metadata, "job deleted")
        except Exception as e:
            logger.error("Exception thrown in job_delete is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(400, [], "job cannot be deleted")

    @staticmethod
    def job_download(
        org_name,
        handler_id,
        job_id,
        kind,
        file_lists=None,
        best_model=None,
        latest_model=None,
        tar_files=True,
        export_type="tao"
    ):
        """Download files associated with the specified job.

        Parameters:
        org_name (str): The name of the organization.
        handler_id (str): UUID corresponding to the experiment or dataset.
        job_id (str): UUID corresponding to the job.
        kind (str): The type of job, either "experiment" or "dataset".
        file_lists (list, optional): List of files to download (defaults to None).
        best_model (bool, optional): Whether to download the best model (defaults to None).
        latest_model (bool, optional): Whether to download the latest model (defaults to None).
        tar_files (bool, optional): Whether to compress the downloaded files into a tarball (defaults to True).
        export_type (str, optional): The type of export (defaults to "tao").

        Returns:
        Code: A response code (200 if the files are successfully downloaded, 404 if not found, 400 for errors).
              - 200: Successfully downloaded the files.
              - 404: If the job or handler is not found.
              - 400: If there are errors during file download.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, None, f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, None, f"{kind} not found")

        handler_job_metadata = get_handler_job_metadata(job_id)
        if not handler_job_metadata:
            return Code(404, None, "job trying to download not found")

        try:
            root = get_jobs_root(user_id, org_name)

            # Following is for `if export_type == "tao":`
            # Copy job logs from root/logs/<job_id>.txt to root/<job_id>/logs_from_toolkit.txt
            out_tar = os.path.join(root, job_id + ".tar.gz")
            files = [os.path.join(root, job_id)]
            if file_lists or best_model or latest_model:
                files = []
                for file in file_lists:
                    if os.path.exists(os.path.join(root, file)):
                        files.append(os.path.join(root, file))
                action = handler_job_metadata.get("action", "")
                epoch_number_dictionary = handler_metadata.get("checkpoint_epoch_number", {})
                best_checkpoint_epoch_number = epoch_number_dictionary.get(f"best_model_{job_id}", 0)
                latest_checkpoint_epoch_number = epoch_number_dictionary.get(f"latest_model_{job_id}", 0)
                if (not best_model) and latest_model:
                    best_checkpoint_epoch_number = latest_checkpoint_epoch_number
                network = handler_metadata.get("network_arch", "")
                if network in MISSING_EPOCH_FORMAT_NETWORKS:
                    format_epoch_number = str(best_checkpoint_epoch_number)
                else:
                    format_epoch_number = f"{best_checkpoint_epoch_number:03}"
                if best_model or latest_model:
                    job_root = os.path.join(root, job_id)
                    if (handler_metadata.get("automl_settings", {}).get("automl_enabled") is True and
                       action in ("train", "distill", "quantize")):
                        job_root = os.path.join(job_root, "best_model")
                    find_trained_tlt = (
                        glob.glob(f"{job_root}/*{format_epoch_number}.tlt") +
                        glob.glob(f"{job_root}/train/*{format_epoch_number}.tlt") +
                        glob.glob(f"{job_root}/weights/*{format_epoch_number}.tlt")
                    )
                    find_trained_pth = (
                        glob.glob(f"{job_root}/*{format_epoch_number}.pth") +
                        glob.glob(f"{job_root}/train/*{format_epoch_number}.pth") +
                        glob.glob(f"{job_root}/weights/*{format_epoch_number}.pth")
                    )
                    find_trained_hdf5 = (
                        glob.glob(f"{job_root}/*{format_epoch_number}.hdf5") +
                        glob.glob(f"{job_root}/train/*{format_epoch_number}.hdf5") +
                        glob.glob(f"{job_root}/weights/*{format_epoch_number}.hdf5")
                    )
                    if find_trained_tlt:
                        files.append(find_trained_tlt[0])
                    if find_trained_pth:
                        files.append(find_trained_pth[0])
                    if find_trained_hdf5:
                        files.append(find_trained_hdf5[0])
                if not files:
                    return Code(404, None, "Atleast one of the requested files not present")

            log_root = get_handler_log_root(user_id, org_name, handler_id)
            log_file = glob.glob(f"{log_root}/**/*{job_id}.txt", recursive=True)
            files = list(set(files))

            if tar_files or (not tar_files and len(files) > 1) or files == [os.path.join(root, job_id)]:
                if files == [os.path.join(root, job_id)]:
                    files += log_file
                    files = list(set(files))

                def get_files_recursively(directory):
                    return [
                        file for file in glob.glob(os.path.join(directory, '**'), recursive=True)
                        if os.path.isfile(file) and not file.endswith(".lock")
                    ]
                all_files = []
                for file in files:
                    if os.path.isdir(file):
                        all_files.extend(get_files_recursively(file))
                    elif os.path.isfile(file):
                        all_files.append(file)

                # Appending UUID to not overwrite the tar file created at end of job complete
                out_tar = out_tar.replace(
                    ".tar.gz",
                    str(uuid.uuid4()) + ".tar.gz"
                )
                with tarfile.open(out_tar, "w:gz") as tar:
                    for file_path in all_files:
                        tar.add(file_path, arcname=file_path.replace(root, "", 1).replace(log_root, "", 1))
                return Code(200, out_tar, "selective files of job downloaded")

            if files and os.path.exists(os.path.join(root, files[0])):
                return Code(200, os.path.join(root, files[0]), "single file of job downloaded")
            return Code(404, None, "job output not found")

        except Exception as e:
            logger.error("Exception thrown in job_download is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(404, None, "job output not found")

    @staticmethod
    def job_list_files(org_name, handler_id, job_id, kind):
        """Lists the files associated with a specific job.

        Args:
            org_name (str): The name of the organization.
            handler_id (str): The UUID corresponding to the experiment or dataset.
            job_id (str): The UUID of the job whose files need to be listed.
            kind (str): The type of handler, either 'experiment' or 'dataset'.

        Returns:
            Code: A response object indicating the result of the operation.
                - 200 with a list of file paths if files are found.
                - 404 with an error message if no files are found or access is denied.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        job = get_job(job_id)
        if not job:
            return Code(404, None, "job trying to view not found")

        files, _, _, _ = get_files_from_cloud(handler_metadata, job_id)
        if files:
            return Code(200, files, "Job files retrieved")
        return Code(200, files, "No downloadable files for this job is found")

    @staticmethod
    def get_job_logs(org_name, handler_id, job_id, kind, automl_experiment_index=None):
        """Retrieves real-time logs for a specific job.

        Args:
            org_name (str): The name of the organization.
            handler_id (str): The UUID corresponding to the experiment or dataset.
            job_id (str): The UUID of the job for which logs are being retrieved.
            kind (str): The type of handler, either 'experiment' or 'dataset'.
            automl_experiment_index (int, optional): The index of the AutoML experiment, if applicable.

        Returns:
            Code: A response object indicating the result of the operation.
                - 200 with the log content or a detailed message if logs are not yet available.
                - 404 if the log file is not found.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, {}, f"{kind} not found.")

        # Get log file path
        # Normal action log is saved at /orgs/<org_name>/users/<user_id>/logs/<job_id>.txt
        # AutoML train  log is saved at:
        # /orgs/<org_name>/users/<user_id>/jobs/<job_id>/experiment_<recommendation_index>/log.txt
        user_id = handler_metadata.get("user_id")
        log_file_path = os.path.join(get_handler_log_root(user_id, org_name, handler_id), str(job_id) + ".txt")
        job_metadata = get_handler_job_metadata(job_id)
        automl_index = None
        lookup_job_id = job_id
        is_brain_job = False

        logger.debug(
            f"[JOB_LOGS] get_job_logs called: org_name={org_name}, handler_id={handler_id}, "
            f"job_id={job_id}, kind={kind}, automl_experiment_index={automl_experiment_index}"
        )
        if (handler_metadata.get("automl_settings", {}).get("automl_enabled", False) and
                job_metadata.get("action", "") == "train"):
            logger.debug(f"[JOB_LOGS] AutoML job detected for job_id={job_id}")

            # Handle experiment_number=-1 for brain job logs
            if automl_experiment_index is not None and int(automl_experiment_index) == -1:
                logger.info(f"[JOB_LOGS] Fetching AutoML brain job logs for job_id={job_id}")
                is_brain_job = True
                lookup_job_id = job_id  # Brain job uses the main job_id
                log_file_path = os.path.join(get_handler_log_root(user_id, org_name, handler_id), str(job_id) + ".txt")
                logger.info(f"[JOB_LOGS] Brain job: lookup_job_id={lookup_job_id}, log_file_path={log_file_path}")
            else:
                logger.debug(
                    f"[JOB_LOGS] Regular AutoML recommendation, "
                    f"provided experiment_index={automl_experiment_index}"
                )
                # Regular AutoML experiment logs
                automl_index = get_automl_current_rec(job_id)
                logger.debug(f"[JOB_LOGS] Current recommendation from DB: {automl_index}")
                if automl_experiment_index is not None:
                    logger.info(f"[JOB_LOGS] Overriding with provided experiment_index: {automl_experiment_index}")
                    automl_index = int(automl_experiment_index)
                logger.info(f"[JOB_LOGS] Using automl_index={automl_index} for log retrieval")
                logger.debug(f"[JOB_LOGS] Experiment log_file_path: {log_file_path}")

            # For AutoML, need to find the actual job_id for the recommendation
            if automl_index is not None:
                logger.debug(f"[JOB_LOGS] Looking up AutoML recommendation job_id for experiment {automl_index}")
                controller_list = get_automl_controller_info(job_id)
                logger.debug(f"[JOB_LOGS] Found {len(controller_list)} recommendations in controller list")
                logger.debug(
                    f"[JOB_LOGS] Controller list: "
                    f"{[{k: v for k, v in rec.items() if k in ['id', 'job_id']} for rec in controller_list]}"
                )
                # Get list of valid experiment IDs
                valid_experiment_ids = sorted([rec.get('id') for rec in controller_list if rec.get('id') != ""])
                logger.debug(f"[JOB_LOGS] Valid experiment IDs: {valid_experiment_ids}")
                # Search for the requested experiment
                found = False
                for rec_info in controller_list:
                    rec_id = rec_info.get("id", "")
                    rec_job_id = rec_info.get("job_id", "")
                    logger.debug(
                        f"[JOB_LOGS] Checking rec: id={rec_id}, job_id={rec_job_id}, "
                        f"looking_for={automl_index}"
                    )
                    if rec_id != "" and int(rec_id) == int(automl_index):
                        lookup_job_id = rec_info.get("job_id", job_id)
                        log_file_path = os.path.join(
                            get_handler_log_root(user_id, org_name, handler_id),
                            f"{lookup_job_id}.txt"
                        )
                        logger.info(
                            f"[JOB_LOGS] AutoML: Found match! Using job_id={lookup_job_id} "
                            f"for experiment {automl_index}"
                        )
                        found = True
                        break
                # If experiment index was explicitly provided but not found, return error with valid IDs
                if not found and automl_experiment_index is not None:
                    logger.error(
                        f"[JOB_LOGS] AutoML: Experiment index {automl_index} not found. "
                        f"Available experiments: {valid_experiment_ids}"
                    )

                    if valid_experiment_ids:
                        error_msg = (
                            f"AutoML experiment index '{automl_index}' does not exist. "
                            f"Available experiment indices: {', '.join(map(str, valid_experiment_ids))}. "
                            "Use experiment_number=-1 to get AutoML brain job logs."
                        )
                    else:
                        error_msg = (
                            "No AutoML experiments have been created yet. "
                            "Use experiment_number=-1 to get AutoML brain job logs."
                        )

                    return Code(404, {}, error_msg)

                # If not found and using default (current rec), just log warning
                if not found:
                    logger.warning(
                        f"[JOB_LOGS] AutoML: Could not find job_id for experiment {automl_index} "
                        f"in controller list. Available: {valid_experiment_ids}"
                    )

        # For K8s and Docker backends, get logs from the most appropriate source
        job_status = job_metadata.get("status", "")
        backend = os.getenv("BACKEND", "local-k8s")
        logger.debug(
            f"[JOB_LOGS] job_status={job_status}, backend={backend}, "
            f"log_file_exists={os.path.exists(log_file_path)}, log_file_path={log_file_path}"
        )
        # Determine log retrieval strategy based on job status and backend
        is_completed = job_status in ("Done", "Error", "Canceled", "Paused")
        if is_completed:
            # For completed jobs with streaming backends:
            # 1. Try cloud storage first (log monitor uploads complete logs there)
            # 2. Fall back to cached file if cloud download fails
            logger.info(
                f"[JOB_LOGS] Job completed (status={job_status}), attempting to retrieve "
                f"complete logs from cloud storage for job_id={lookup_job_id}"
            )
            workspace_id = handler_metadata.get("workspace", "")
            if workspace_id:
                try:
                    download_log_from_cloud(handler_metadata, lookup_job_id, log_file_path, automl_index)
                    logger.info("[JOB_LOGS] Successfully downloaded logs from cloud storage")
                except Exception as e:
                    logger.warning(
                        f"[JOB_LOGS] Failed to download from cloud storage: {type(e).__name__}: {e}, "
                        "will use cached file if available"
                    )
            else:
                logger.warning(
                    "[JOB_LOGS] No workspace assigned, cannot download from cloud. "
                    "Will use cached file if available."
                )
        elif not is_completed:
            # For running jobs with streaming backends:
            # Try to get real-time logs from backend (K8s/Docker), then fall back to cloud
            log_type = "brain job" if is_brain_job else f"job_id={lookup_job_id}"
            logger.info(f"[JOB_LOGS] Job running, attempting real-time logs from {backend} for {log_type}")
            direct_logs = get_job_logs_from_backend(lookup_job_id)
            if direct_logs:
                log_desc = "brain job" if is_brain_job else "job"
                log_size_kb = len(direct_logs) / 1024
                logger.info(
                    f"[JOB_LOGS] Successfully retrieved {log_desc} logs directly from {backend} "
                    f"for job_id={lookup_job_id} ({log_size_kb:.1f} KB)"
                )
                # Cache the logs to file for future use
                try:
                    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                    logger.debug(f"[JOB_LOGS] Caching logs to {log_file_path}")
                    with open(log_file_path, 'w', encoding='utf-8') as f:
                        f.write(direct_logs)
                    logger.info(f"[JOB_LOGS] Successfully cached logs to {log_file_path}")
                except Exception as e:
                    logger.warning(f"[JOB_LOGS] Failed to cache logs to file: {type(e).__name__}: {e}")
                    logger.debug("[JOB_LOGS] Cache exception details:", exc_info=True)

                # Return logs directly
                logger.debug(f"[JOB_LOGS] Returning {len(direct_logs.splitlines())} lines of logs")
                return Code(200, iter(direct_logs.splitlines(keepends=True)))
            logger.warning(f"[JOB_LOGS] Could not retrieve real-time logs from {backend} for job_id={lookup_job_id}")
            logger.info(f"[JOB_LOGS] Falling back to cloud storage download for job_id={lookup_job_id}")
            # Fallback to cloud storage
            workspace_id = handler_metadata.get("workspace", "")
            if workspace_id:
                try:
                    download_log_from_cloud(handler_metadata, lookup_job_id, log_file_path, automl_index)
                except Exception as e:
                    logger.warning(f"[JOB_LOGS] Cloud download failed: {type(e).__name__}: {e}")
            else:
                logger.warning("[JOB_LOGS] No workspace assigned, cannot download from cloud")
        # File not present - Use detailed message or job status
        if not os.path.exists(log_file_path):
            detailed_result_msg = (
                job_metadata.get("job_details", {})
                .get(job_id, {})
                .get("detailed_status", {})
                .get("message", "")
            )
            if detailed_result_msg:
                return Code(200, detailed_result_msg)

            if (handler_metadata.get("automl_settings", {}).get("automl_enabled", False) and
                    job_metadata.get("action", "") == "train"):
                if handler_metadata.get("status") in ["Canceled", "Canceling"]:
                    return Code(200, "AutoML training has been canceled.")
                if handler_metadata.get("status") in ["Paused", "Pausing"]:
                    return Code(200, "AutoML training has been paused.")
                if handler_metadata.get("status") == "Resuming":
                    return Code(200, "AutoML training is resuming.")
                if handler_metadata.get("status") == "Running":
                    return Code(200, "Generating new recommendation for AutoML experiment.")
            return Code(404, {}, "Logs for the job are not available yet.")
        return Code(200, get_job_logs_util(log_file_path))

    @staticmethod
    def get_job_events(org_name, handler_id, job_id, kind, automl_experiment_number="0"):
        """Retrieves all status events for a specific job.

        Args:
            org_name (str): The name of the organization.
            handler_id (str): The UUID corresponding to the experiment or dataset.
            job_id (str): The UUID of the job for which events are being retrieved.
            kind (str): The type of handler, either 'experiment' or 'dataset'.
            automl_experiment_number (str, optional): The experiment number for AutoML jobs.

        Returns:
            Code: A response object indicating the result of the operation.
                - 200 with the list of all status events.
                - 404 if the job or handler is not found, or if events are not found for requested IDs.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, {"error_desc": f"{kind} not found.", "error_code": 1}, f"{kind} not found.")

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, {"error_desc": f"{kind} not found.", "error_code": 1}, f"{kind} not found.")

        job_metadata = get_handler_job_metadata(job_id)
        if not job_metadata:
            error_desc = (
                "Job not found. If you changed config (API URL) or restarted containers, ensure the client "
                "uses the same API server that created the job and that the server uses the same MongoDB."
            )
            return Code(404, {"error_desc": error_desc, "error_code": 1}, "Job not found.")

        # Check if this is an AutoML job
        automl_enabled = handler_metadata.get("automl_settings", {}).get("automl_enabled", False)
        is_automl = automl_enabled and job_metadata.get("action", "") == "train"

        # Validate AutoML experiment number if this is an AutoML job
        if is_automl and automl_experiment_number != "0":
            # Check if the requested AutoML experiment exists
            controller_list = get_automl_controller_info(job_id)
            valid_experiment_ids = [str(rec.get('id')) for rec in controller_list if rec.get('id') != ""]

            # Special case: -1 is for brain job logs
            if automl_experiment_number != "-1" and automl_experiment_number not in valid_experiment_ids:
                valid_exp_list = ', '.join(valid_experiment_ids)
                error_msg = (f"Events not found for the requested AutoML experiment number "
                             f"'{automl_experiment_number}'. Valid experiment numbers: {valid_exp_list}")
                logger.warning(f"AutoML experiment {automl_experiment_number} not found for job {job_id}")
                return Code(404, {"error_desc": error_msg, "error_code": 1}, error_msg)

        # Get all status events
        status_events = get_dnn_status(job_id, automl=is_automl, experiment_number=automl_experiment_number)

        if status_events is None or len(status_events) == 0:
            if is_automl and automl_experiment_number != "0":
                error_msg = (f"Events not found for the requested job ID '{job_id}' and AutoML experiment "
                             f"number '{automl_experiment_number}'.")
            else:
                error_msg = f"Events not found for the requested job ID '{job_id}'."
            logger.info(f"No events found for job {job_id}, automl_experiment_number={automl_experiment_number}")
            return Code(404, {"error_desc": error_msg, "error_code": 1}, error_msg)

        logger.info(f"Retrieved {len(status_events)} status events for job {job_id}")
        return Code(200, status_events, "Job events retrieved successfully")
