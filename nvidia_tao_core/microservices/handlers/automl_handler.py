# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoML handler modules"""
import ast
import os
import json
import traceback
from copy import deepcopy
from datetime import datetime, timezone
import sysconfig
import logging
import time

from nvidia_tao_core.microservices.utils.automl_utils import update_automl_details_metadata
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    BACKEND,
    get_handler_metadata,
    get_handler_type,
    get_jobs_root,
    save_automl_controller_info,
    serialize_object,
    write_job_metadata,
    get_handler_job_metadata,
    get_automl_child_job_ids,
    request_automl_child_cancellation,
    update_handler_with_jobs_info,
    get_automl_controller_info,
    set_automl_checkpoint_cleanup_state,
)
from nvidia_tao_core.microservices.enum_constants import Backend
from nvidia_tao_core.microservices.utils.handler_utils import Code, decrypt_handler_metadata
from .docker_images import DOCKER_IMAGE_MAPPER
from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import ExecutionHandler
from nvidia_tao_core.microservices.utils.log_monitor_service import start_monitoring_job, stop_monitoring_job
from nvidia_tao_core.microservices.utils.automl_job_utils import cancel_automl_child_job

# TODO Make sure the image name is current docker tag of the API
image = DOCKER_IMAGE_MAPPER["API"]

logger = logging.getLogger(__name__)


def get_automl_param(automl_settings, param_name, default_value=None):
    """Get AutoML parameter with backward compatibility for nested structure.

    Args:
        automl_settings (dict): AutoML settings dictionary
        param_name (str): Parameter name to retrieve
        default_value: Default value if parameter not found

    Returns:
        Parameter value from either flat structure or nested algorithm_specific_params
    """
    # First check flat structure (backward compatibility)
    if param_name in automl_settings:
        return automl_settings.get(param_name, default_value)

    # Then check nested structure
    algo_params = automl_settings.get('algorithm_specific_params', {})
    if isinstance(algo_params, dict) and param_name in algo_params:
        return algo_params.get(param_name, default_value)

    return default_value


def _normalize_automl_hyperparameters(automl_hyperparameters):
    """Normalize automl_hyperparameters to JSON format for shell-safe passing.

    Handles both SDK format "['param1', 'param2']" and CLI format "[param1, param2]".
    Returns a JSON string that can be safely passed through shell and parsed with json.loads().
    """
    if not isinstance(automl_hyperparameters, str):
        return json.dumps(automl_hyperparameters)

    try:
        # Try ast.literal_eval first (works for SDK format with quoted elements)
        params_list = ast.literal_eval(automl_hyperparameters)
        return json.dumps(params_list)
    except (ValueError, SyntaxError):
        # Fallback for CLI format (unquoted elements due to shell processing)
        params_str = automl_hyperparameters.strip('[]').strip()
        if params_str:
            params_list = [p.strip() for p in params_str.split(',')]
            return json.dumps(params_list)
        return "[]"


class AutoMLHandler:
    """Handles AutoML job operations including starting, stopping, resuming, deleting, and retrieving job metadata.

    - **Start**: Launches an AutoML job as a Kubernetes job.
    - **Stop**: Terminates an ongoing AutoML job and cancels any pending recommendations.
    - **Resume**: Restarts a previously stopped AutoML job with restored settings.
    - **Delete**: Deletes an AutoML job (same as AppHandler behavior).
    - **Download**: Downloads artifacts from an AutoML job (same as AppHandler behavior).
    - **Retrieve**: Constructs and returns job metadata based on the job's status.
    """

    @staticmethod
    def start(user_id, org_name, experiment_id, job_id, handler_metadata, name="",
              backend_details=None, retain_checkpoints_for_resume=False, timeout_minutes=60):
        """Starts an AutoML job by executing `automl_start.py` with the provided parameters.

        Args:
            user_id (str): ID of the user initiating the job.
            org_name (str): Name of the organization.
            experiment_id (str): ID of the associated experiment.
            job_id (str): Unique identifier for the AutoML job.
            handler_metadata (dict): Metadata containing AutoML configuration settings.
            name (str, optional): Name of the job. Defaults to "automl train job".
            backend_details (dict, optional): Backend-specific execution details. Defaults to None.
            retain_checkpoints_for_resume (bool, optional): Whether to retain .pth
                checkpoints for training resume. Defaults to False.
            timeout_minutes (int, optional): The job-specific timeout in minutes. If not specified, uses global timeout.
            backend_type (str, optional): The backend type to use for the job. Defaults to None.
        """
        network = get_handler_type(handler_metadata)
        metric = handler_metadata.get("metric", "map")
        automl_settings = handler_metadata.get("automl_settings", {})
        automl_metric = automl_settings.get("metric")
        if automl_metric:
            metric = automl_metric
        automl_algorithm = automl_settings.get("automl_algorithm", "Bayesian")
        # Hyperband-like algorithms need checkpoint retention for resume
        if automl_algorithm.lower() in ("hyperband", "h", "bohb", "asha", "dehb", "hyperband_es", "hes"):
            retain_checkpoints_for_resume = True

        job_metadata = {
            "name": name,
            "id": job_id,
            "org_name": org_name,
            "parent_id": None,
            "backend_details": backend_details,
            "action": "train",
            "created_on": datetime.now(tz=timezone.utc),
            "experiment_id": experiment_id,
            "status": "Pending",
            "job_details": {},
            "retain_checkpoints_for_resume": retain_checkpoints_for_resume,
            "timeout_minutes": timeout_minutes,
        }
        root = os.path.join(get_jobs_root(user_id, org_name), job_id)
        if not os.path.exists(root):
            os.makedirs(root)

        if not name:
            name = "automl train job"
        automl_delete_intermediate_ckpt = automl_settings.get("automl_delete_intermediate_ckpt", True)
        automl_hyperparameters = automl_settings.get("automl_hyperparameters", "[]")
        override_automl_disabled_params = automl_settings.get("override_automl_disabled_params", False)

        # Get algorithm-specific parameters as JSON
        algorithm_specific_params = json.dumps(automl_settings.get("algorithm_specific_params", {}))

        write_job_metadata(job_id, job_metadata)
        update_handler_with_jobs_info(job_metadata, experiment_id, job_id, "experiments")

        workspace_id = handler_metadata.get("workspace")
        workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
        decrypted_workspace_metadata = deepcopy(workspace_metadata)
        decrypt_handler_metadata(decrypted_workspace_metadata)
        decrypted_workspace_metadata.pop('_id', None)

        # Call the script
        logger.info("Starting automl %s", job_id)
        python_lib_path = sysconfig.get_path("purelib")
        automl_script = os.path.join(python_lib_path, "nvidia_tao_core/microservices/automl_start.py")

        run_command = (
            f'umask 0 && python3 {automl_script} '
            f'--user_id={user_id} '
            f'--org_name={org_name} '
            f'--name="{name}" '
            f'--root={root} '
            f'--automl_job_id={job_id} '
            f'--network={network} '
            f'--experiment_id={experiment_id} '
            f'--resume=False '
            f'--automl_algorithm={automl_algorithm} '
            f'--automl_delete_intermediate_ckpt={automl_delete_intermediate_ckpt} '
            f'--metric={metric} '
            f"--algorithm_specific_params='{algorithm_specific_params}' "
            f"--automl_hyperparameters='{_normalize_automl_hyperparameters(automl_hyperparameters)}' "
            f'--override_automl_disabled_params={override_automl_disabled_params} '
            f'--retain_checkpoints_for_resume={retain_checkpoints_for_resume} '
            f'--timeout_minutes={timeout_minutes} '
            f"--decrypted_workspace_metadata='{json.dumps(decrypted_workspace_metadata, default=str)}'"
        )
        if backend_details:
            run_command = f"{run_command} --backend_details='{json.dumps(backend_details)}'"

        cluster_num_gpus = int(os.getenv('NUM_GPU_PER_NODE', default='0'))
        docker_env_vars = {
            "NUM_GPU_PER_NODE": str(cluster_num_gpus),
            "TAO_LOG_LEVEL": os.getenv('TAO_LOG_LEVEL', default='DEBUG')
        }

        # Pass SSH configuration to brain job for SLURM access
        # SSH keys are mounted at /root/.ssh in the brain container
        docker_env_vars["SSH_KEY_PATH"] = "/root/.ssh/id_ed25519"

        docker_env_vars.update(handler_metadata.get("docker_env_vars", {}))
        logger.debug(
            f"[AUTOML-START] Creating brain job {job_id}: "
            f"Setting NUM_GPU_PER_NODE={cluster_num_gpus} in brain container env (no actual GPUs assigned)"
        )

        try:
            ExecutionHandler.create_job_with_handler(
                org_name=org_name,
                job_name=job_id,
                image=image,
                command=run_command,
                workspace_metadata=decrypted_workspace_metadata,
                num_gpu=0,
                docker_env_vars=docker_env_vars,
                automl_brain=True,
                automl_exp_job=False,
                backend_details=backend_details
            )
        except Exception as e:
            logger.error(f"[AUTOML-START] Failed to create brain job {job_id}: {e}")
            from nvidia_tao_core.microservices.utils.stateless_handler_utils import update_job_status
            update_job_status(experiment_id, job_id, status="Error", kind="experiments")
            job_metadata["status"] = "Error"
            job_metadata["job_details"] = {
                job_id: {
                    "detailed_status": {
                        "message": f"AutoML brain job creation failed: {e}"
                    }
                }
            }
            write_job_metadata(job_id, job_metadata)
            raise

        # Start log monitoring for AutoML brain job (server-side)
        backend = os.getenv("BACKEND", "local-k8s")
        logger.debug(f"[AUTOML] Checking if log monitoring should start for brain job {job_id}, backend={backend}")
        if backend in ("local-k8s", "local-docker"):
            try:
                logger.debug(f"[AUTOML] Starting log monitoring setup for brain job {job_id}")
                # Get namespace for K8s
                namespace = None
                if backend == "local-k8s":
                    namespace = os.getenv("NAMESPACE")
                    logger.debug(f"[AUTOML] K8s namespace from env: {namespace}")
                    if not namespace:
                        try:
                            namespace_file = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
                            logger.debug(f"[AUTOML] Reading namespace from {namespace_file}")
                            with open(namespace_file, 'r', encoding='utf-8') as f:
                                namespace = f.read().strip()
                            logger.debug(f"[AUTOML] Got namespace from service account: {namespace}")
                        except Exception as e:
                            namespace = "default"
                            logger.debug(f"[AUTOML] Using default namespace (error: {e})")

                # Start monitoring for brain job
                logger.info(f"[AUTOML] Starting log monitoring for AutoML brain job {job_id}, namespace={namespace}")
                start_monitoring_job(
                    job_id,
                    callback_url=None,  # Brain job doesn't need callbacks (server-side only)
                    namespace=namespace,
                    metadata={
                        'handler_id': experiment_id,
                        'handler_kind': 'experiment',
                        'action': 'automl_brain',
                        'network': network,
                        'is_brain_job': True
                    }
                )
                logger.info(f"[AUTOML] Successfully started log monitoring for AutoML brain job {job_id}")
            except Exception as e:
                logger.warning(
                    f"[AUTOML] Failed to start log monitoring for AutoML brain job {job_id}: "
                    f"{type(e).__name__}: {e}"
                )
                logger.debug("[AUTOML] Exception details:", exc_info=True)
        else:
            logger.debug(f"[AUTOML] Skipping log monitoring for backend {backend}")

    @staticmethod
    def stop(user_id, org_name, experiment_id, job_id, cleanup_checkpoints=True):
        """Stops a running AutoML job and cancels any active recommendations.

        Args:
            user_id (str): ID of the user.
            org_name (str): Name of the organization.
            experiment_id (str): ID of the associated experiment.
            job_id (str): Unique identifier for the AutoML job.

        Returns:
            Code: Status code and message indicating job cancellation success or failure.
        """
        logger.debug(
            f"[AUTOML-STOP] Starting AutoML stop operation: job_id={job_id}, experiment_id={experiment_id}, "
            f"org_name={org_name}, user_id={user_id}"
        )

        try:
            recommendations = get_automl_controller_info(job_id) or []
            child_job_ids = {
                recommendation.get("job_id")
                for recommendation in recommendations
                if recommendation.get("job_id")
            }
            child_job_ids.update(get_automl_child_job_ids(job_id))

            for recommendation in recommendations:
                status = str(recommendation.get("status", "")).lower()
                if status not in ("done", "success", "failure", "error", "canceled"):
                    recommendation["status"] = "canceling"
            if recommendations:
                save_automl_controller_info(job_id, recommendations)
                update_automl_details_metadata(job_id, experiment_id, "experiments")

            cancellation_intents_persisted = all(
                request_automl_child_cancellation(child_job_id)
                for child_job_id in sorted(child_job_ids)
            )
            if not cancellation_intents_persisted:
                return Code(409, [], "child cancellation intent could not be persisted")

            if cleanup_checkpoints and not set_automl_checkpoint_cleanup_state(
                job_id, "pending"
            ):
                return Code(409, [], "checkpoint cleanup intent could not be persisted")

            timeout_seconds = int(os.getenv("AUTOML_CANCEL_TIMEOUT_SECONDS", "180"))
            deadline = time.monotonic() + timeout_seconds
            while True:
                child_results = [
                    cancel_automl_child_job(child_job_id) is True
                    for child_job_id in sorted(child_job_ids)
                ]
                children_quiescent = all(child_results)
                cleanup_complete = not cleanup_checkpoints
                if cleanup_checkpoints:
                    metadata = get_handler_job_metadata(job_id) or {}
                    cleanup_complete = (
                        metadata.get("checkpoint_cleanup_pending") is False and
                        metadata.get("checkpoint_cleanup_status") in ("complete", "skipped")
                    )

                if children_quiescent and cleanup_complete:
                    break
                if time.monotonic() >= deadline:
                    logger.error(
                        "AutoML cancellation barrier timed out for %s "
                        "(children_quiescent=%s, cleanup_complete=%s)",
                        job_id, children_quiescent, cleanup_complete,
                    )
                    return Code(
                        409,
                        {"message": f"job {job_id} remains Canceling"},
                        "child quiescence or checkpoint cleanup is still pending",
                    )
                time.sleep(1)

            latest_recommendations = get_automl_controller_info(job_id) or recommendations
            for recommendation in latest_recommendations:
                status = str(recommendation.get("status", "")).lower()
                if status not in ("done", "success", "failure", "error", "canceled"):
                    recommendation["status"] = "canceled"
            if latest_recommendations:
                save_automl_controller_info(job_id, latest_recommendations)
                update_automl_details_metadata(job_id, experiment_id, "experiments")

            # The brain remains alive until it has acknowledged durable cleanup.
            if ExecutionHandler.delete_job_with_handler(job_id) is not True:
                return Code(409, [], "AutoML brain termination was not accepted")
            handler = ExecutionHandler.create_handler(backend=BACKEND, job_id=job_id)
            if not handler or not handler.wait_for_job_termination(
                job_id, timeout_seconds=120
            ):
                return Code(409, [], "AutoML brain termination was not confirmed")

            if BACKEND in (Backend.LOCAL_K8S, Backend.LOCAL_DOCKER):
                try:
                    stop_monitoring_job(job_id)
                except Exception as err:
                    logger.warning(
                        "Failed to stop log monitoring for AutoML brain %s: %s",
                        job_id, err,
                    )
        except Exception as e:
            logger.error(f"[AUTOML-STOP] Exception thrown in AutoMLHandler stop: job_id={job_id}, error={str(e)}")
            logger.error(f"[AUTOML-STOP] Traceback: {traceback.format_exc()}")
            return Code(404, [], "job cannot be stopped in platform")

        return Code(200, {"message": f"job {job_id} cancelled"})

    @staticmethod
    def resume(user_id, org_name, experiment_id, job_id, handler_metadata, name="",
               backend_details=None, timeout_minutes=60):
        """Resumes a previously stopped AutoML job by re-running `automl_start.py` with the resume flag.

        Args:
            user_id (str): ID of the user.
            org_name (str): Name of the organization.
            experiment_id (str): ID of the associated experiment.
            job_id (str): Unique identifier for the AutoML job.
            handler_metadata (dict): Metadata containing AutoML configuration settings.
            name (str, optional): Name of the job. Defaults to "automl train job".
            backend_details (dict, optional): Backend-specific execution details. Defaults to None.
        """
        logger.debug(
            f"[AUTOML-RESUME] Starting AutoML resume operation: job_id={job_id}, "
            f"experiment_id={experiment_id}, org_name={org_name}, user_id={user_id}"
        )

        root = os.path.join(get_jobs_root(user_id, org_name), job_id)
        if not os.path.exists(root):
            logger.debug(f"[AUTOML-RESUME] Creating job root directory: path={root}")
            os.makedirs(root)
        else:
            logger.debug(f"[AUTOML-RESUME] Job root directory exists: path={root}")

        if not name:
            name = "automl train job"
        network = get_handler_type(handler_metadata)
        metric = handler_metadata.get("metric", "map")
        automl_settings = handler_metadata.get("automl_settings", {})
        automl_metric = automl_settings.get("metric")
        if automl_metric:
            metric = automl_metric
        automl_algorithm = automl_settings.get("automl_algorithm", "Bayesian")
        automl_delete_intermediate_ckpt = automl_settings.get("automl_delete_intermediate_ckpt", True)
        automl_hyperparameters = automl_settings.get("automl_hyperparameters", "[]")
        override_automl_disabled_params = automl_settings.get("override_automl_disabled_params", False)

        logger.debug(
            f"[AUTOML-RESUME] AutoML settings: job_id={job_id}, network={network}, "
            f"algorithm={automl_algorithm}, metric={metric}, "
            f"automl_settings={automl_settings}, timeout_minutes={timeout_minutes}"
        )
        # Get algorithm-specific parameters as JSON
        algorithm_specific_params = json.dumps(automl_settings.get("algorithm_specific_params", {}))

        workspace_id = handler_metadata.get("workspace")
        logger.debug(f"[AUTOML-RESUME] Loading workspace metadata: job_id={job_id}, workspace_id={workspace_id}")
        workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
        decrypted_workspace_metadata = deepcopy(workspace_metadata)
        decrypt_handler_metadata(decrypted_workspace_metadata)
        decrypted_workspace_metadata.pop('_id', None)

        job_metadata = get_handler_job_metadata(job_id)
        retain_checkpoints_for_resume = (
            job_metadata.get("retain_checkpoints_for_resume", False) if job_metadata else False
        )
        # Hyperband-like algorithms need checkpoint retention for resume
        if automl_algorithm.lower() in ("hyperband", "h", "bohb", "asha", "dehb", "hyperband_es", "hes"):
            retain_checkpoints_for_resume = True
            logger.debug(
                f"[AUTOML-RESUME] Hyperband algorithm detected, "
                f"forcing retain_checkpoints_for_resume=True: job_id={job_id}"
            )

        logger.debug(
            f"[AUTOML-RESUME] Job settings: job_id={job_id}, "
            f"retain_checkpoints_for_resume={retain_checkpoints_for_resume}"
        )

        # Call the script
        python_lib_path = sysconfig.get_path("purelib")
        automl_script = os.path.join(python_lib_path, "nvidia_tao_core/microservices/automl_start.py")
        logger.debug(f"[AUTOML-RESUME] AutoML script path: {automl_script}")
        run_command = (
            f"umask 0 && python3 {automl_script} "
            f'--user_id={user_id} '
            f'--org_name={org_name} '
            f'--name="{name}" '
            f'--root={root} '
            f'--automl_job_id={job_id} '
            f'--network={network} '
            f'--experiment_id={experiment_id} '
            f'--resume=True '
            f'--automl_algorithm={automl_algorithm} '
            f'--automl_delete_intermediate_ckpt={automl_delete_intermediate_ckpt} '
            f'--metric={metric} '
            f"--algorithm_specific_params='{algorithm_specific_params}' "
            f"--automl_hyperparameters='{_normalize_automl_hyperparameters(automl_hyperparameters)}' "
            f'--override_automl_disabled_params={override_automl_disabled_params} '
            f'--retain_checkpoints_for_resume={retain_checkpoints_for_resume} '
            f'--timeout_minutes={timeout_minutes} '
            f"--decrypted_workspace_metadata='{json.dumps(decrypted_workspace_metadata, default=serialize_object)}'"
        )
        if backend_details:
            run_command = f"{run_command} --backend_details='{json.dumps(backend_details)}'"
        # CRITICAL: Pass NUM_GPU_PER_NODE to resumed brain job for GPU validation
        cluster_num_gpus = int(os.getenv('NUM_GPU_PER_NODE', default='0'))
        docker_env_vars = {
            "NUM_GPU_PER_NODE": str(cluster_num_gpus),
            "TAO_LOG_LEVEL": os.getenv('TAO_LOG_LEVEL', default='INFO')
        }

        # Pass SSH configuration to brain job for SLURM access
        # SSH keys are mounted at /root/.ssh in the brain container
        docker_env_vars["SSH_KEY_PATH"] = "/root/.ssh/id_ed25519"

        docker_env_vars.update(handler_metadata.get("docker_env_vars", {}))
        logger.debug(
            f"[AUTOML-RESUME] Creating K8s job for AutoML brain: job_id={job_id}, num_gpu=0, "
            f"NUM_GPU_PER_NODE={cluster_num_gpus} in env"
        )
        try:
            ExecutionHandler.create_job_with_handler(
                org_name=org_name,
                job_name=job_id,
                image=image,
                command=run_command,
                workspace_metadata=decrypted_workspace_metadata,
                num_gpu=0,
                docker_env_vars=docker_env_vars,
                automl_brain=True,
                automl_exp_job=False,
                backend_details=backend_details
            )
        except Exception as e:
            logger.error(f"[AUTOML-RESUME] Failed to create brain job {job_id}: {e}")
            from nvidia_tao_core.microservices.utils.stateless_handler_utils import update_job_status
            update_job_status(experiment_id, job_id, status="Error", kind="experiments")
            resumed_metadata = get_handler_job_metadata(job_id) or {}
            resumed_metadata["status"] = "Error"
            resumed_metadata["job_details"] = {
                job_id: {
                    "detailed_status": {
                        "message": f"AutoML brain job creation failed on resume: {e}"
                    }
                }
            }
            write_job_metadata(job_id, resumed_metadata)
            raise
        logger.debug(f"[AUTOML-RESUME] AutoML resume operation completed: job_id={job_id}")
