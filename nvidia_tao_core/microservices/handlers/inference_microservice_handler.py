# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference Microservice handler using StatefulSets for long-lived inference"""
import base64
import json
import logging
import requests
import shlex
from datetime import datetime, timezone
from typing import Dict, Any
import os

from .docker_images import DOCKER_IMAGE_MAPPER
from nvidia_tao_core.microservices.utils.handler_utils import (
    Code, get_model_results_path
)
from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_handler_metadata, BACKEND
from nvidia_tao_core.microservices.utils.core_utils import read_network_config
from nvidia_tao_core.microservices.enum_constants import Backend
from nvidia_tao_core.microservices.handlers.execution_handlers.kubernetes_handler import KubernetesHandler


# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


class InferenceMicroserviceHandler:
    """Handler class for long-lived Inference Microservice jobs using StatefulSets

    This handler manages:
    - Starting long-lived Inference Microservice using StatefulSets for persistence
    - Running both HTTP server (for file operations) and container_handler.py (for TAO jobs)
    - Managing service lifecycle (start, update, stop)
    - Following standard TAO container and job processing patterns
    """

    @staticmethod
    def start_inference_microservice(org_name: str, experiment_id: str, job_id: str,
                                     job_config: Dict[str, Any],
                                     replicas: int = 1, api_port: int = 8080) -> Code:
        """Starts a long-lived Inference Microservice using StatefulSet

        The network architecture is automatically determined from the experiment metadata.
        If hf_model is provided, uses the HuggingFace inference microservice instead.
        """
        from nvidia_tao_core.microservices.utils.stateless_handler_utils import write_job_metadata
        from nvidia_tao_core.microservices.utils import get_admin_key

        logger.info("Starting Inference Microservice %s for experiment %s", job_id, experiment_id)

        # Check if HuggingFace model is specified - this takes precedence over network_arch
        hf_model = job_config.get("hf_model")
        use_huggingface = hf_model is not None and hf_model.strip() != ""

        if use_huggingface:
            logger.info("HuggingFace model specified: %s - using HuggingFace inference microservice", hf_model)
            network_arch = "huggingface"  # Virtual network arch for HuggingFace models
        else:
            # Get experiment metadata to determine network architecture
            experiment_metadata = get_handler_metadata(experiment_id, kind="experiments")
            network_arch = job_config.get("network_arch")
            if hasattr(network_arch, 'value'):
                network_arch = network_arch.value
            if not network_arch:
                network_arch = experiment_metadata.get("network_arch", "vila")
            logger.info("Network architecture from experiment metadata: %s", network_arch)

        # Get experiment metadata (used for model-path resolution etc.).
        experiment_metadata = get_handler_metadata(experiment_id, kind="experiments")

        folder_path_function = "parent_model"

        # Determine Docker image based on model type
        if use_huggingface:
            # Use TAO_PYTORCH image for HuggingFace models (has transformers installed)
            image = job_config.get("docker_image") or DOCKER_IMAGE_MAPPER.get(
                "TAO_PYTORCH", "nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt"
            )
            logger.info("Using Docker image for HuggingFace model: %s", image)
        else:
            # Read network config to get docker image name
            try:
                network_config = read_network_config(network_arch.lower())

                if network_config:
                    image_key = network_config.get('api_params', {}).get('image', network_arch.upper())
                    image = DOCKER_IMAGE_MAPPER.get(image_key, "nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt")
                    logger.info("Using Docker image: %s (from network_arch: %s)", image, network_arch)
                    folder_path_function = (
                        network_config.get('spec_params', {})
                        .get('inference', {})
                        .get('model_path', "")
                    )
                else:
                    # Fallback if network config is empty
                    image = DOCKER_IMAGE_MAPPER.get(network_arch.upper(), "nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt")
                    logger.info("Using fallback Docker image: %s", image)
            except Exception as e:
                logger.warning("Could not read network config for %s: %s. Using default image.", network_arch, str(e))
                image = DOCKER_IMAGE_MAPPER.get(network_arch.upper(), "nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt")
                logger.info("Using fallback Docker image: %s", image)

        # Build command for Inference Microservice integrated into TAO container
        parent_id = job_config.get("parent_job_id", job_config.get("parent_id", ""))

        if parent_id:
            from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
                get_handler_job_metadata
            )
            parent_job_metadata = get_handler_job_metadata(parent_id)
            if parent_job_metadata:
                parent_status = parent_job_metadata.get("status", "")
                if parent_status != "Done":
                    error_msg = (
                        f"Cannot start inference microservice: parent job {parent_id} "
                        f"is not in 'Done' state (current status: {parent_status})"
                    )
                    logger.error(error_msg)
                    return Code(400, {}, error_msg)
            else:
                error_msg = (
                    f"Cannot start inference microservice: parent job {parent_id} not found"
                )
                logger.error(error_msg)
                return Code(400, {}, error_msg)

        # Determine model path
        if use_huggingface:
            # For HuggingFace models, model_path is the HuggingFace model name
            model_path = hf_model
            logger.info("Using HuggingFace model: %s", model_path)
        else:
            folder_path = "folder" in folder_path_function
            model_path = job_config.get(
                "model_path",
                get_model_results_path(experiment_metadata, parent_id, folder_path)
            )
            logger.info("Using model path: %s", model_path)

        if not model_path:
            return Code(400, {}, "Model path or hf_model is required for Inference Microservice")

        # cli_args = convert_dict_to_cli_args(job_config)
        # cli_args = " ".join(cli_args)
        # logger.info("Using CLI args: %s", cli_args)

        docker_env_vars = dict(job_config.get("docker_env_vars", {}))

        docker_env_vars["TAO_EXECUTION_BACKEND"] = BACKEND.value
        docker_env_vars["TAO_API_JOB_ID"] = job_id

        # Set up environment variables for status callbacks (auto-deletion)
        docker_env_vars["CLOUD_BASED"] = "True"
        host_base_url = os.getenv("HOSTBASEURL", "no_url")
        if BACKEND == Backend.LOCAL_K8S:
            from nvidia_tao_core.microservices.utils.executor_utils import get_cluster_ip
            cluster_ip, cluster_port = get_cluster_ip()
            if cluster_ip and cluster_port:
                host_base_url = f"http://{cluster_ip}:{cluster_port}"
        status_url = f"{host_base_url}/api/v1/orgs/{org_name}/experiments/{experiment_id}/jobs/{job_id}"
        docker_env_vars["TAO_LOGGING_SERVER_URL"] = status_url
        docker_env_vars["TAO_ADMIN_KEY"] = get_admin_key()

        # Inference microservice does not depend on cloud storage:
        # no workspace_id / cloud_metadata / results_dir is propagated.
        specs = {
            "model_path": model_path,
        }

        # Propagate additional parameters from job_config to specs
        # These include enable_lora, base_model_path, torch_dtype, device_map, and any other user-provided configs
        # NOTE: docker_env_vars must be excluded - it's handled separately and contains URLs that would
        # be incorrectly parsed as cloud storage paths by download_files_from_spec
        excluded_keys = ["parent_id", "parent_job_id", "model_path", "hf_model", "network_arch", "docker_env_vars"]
        for key, value in job_config.items():
            if key not in excluded_keys:
                # Convert enum values to their string representation
                if hasattr(value, 'value'):
                    value = value.value
                specs[key] = value
                logger.info(f"Propagating parameter to specs: {key} = {value}")

        # For HuggingFace models, ensure HF-specific parameters are in specs
        if use_huggingface:
            specs["hf_model"] = hf_model
            # Set default torch_dtype and device_map if not provided
            if "torch_dtype" not in specs:
                specs["torch_dtype"] = "auto"
            if "device_map" not in specs:
                specs["device_map"] = "auto"

        job_metadata = {
            "job_id": job_id,
            "specs": specs,
            "neural_network_name": network_arch,
        }

        # Base64 encode custom function strings to avoid shell escaping issues
        # These contain Python code with quotes, parentheses, etc. that break bash parsing
        if specs.get("custom_pipeline_loader"):
            specs["custom_pipeline_loader"] = base64.b64encode(
                specs["custom_pipeline_loader"].encode()
            ).decode()
            specs["custom_pipeline_loader_encoded"] = True
        if specs.get("custom_inference_fn"):
            specs["custom_inference_fn"] = base64.b64encode(
                specs["custom_inference_fn"].encode()
            ).decode()
            specs["custom_inference_fn_encoded"] = True

        # Determine the inference microservice command based on model type
        if use_huggingface:
            # Use the HuggingFace inference microservice from tao-core
            inference_command = (
                "python -m nvidia_tao_core.microservices.handlers"
                ".huggingface_inference_microservice_server"
            )
            logger.info("Using HuggingFace inference microservice")
        else:
            # Use the network-specific inference microservice
            inference_command = f"{network_arch}-inference-microservice"
            logger.info("Using network-specific inference microservice: %s", inference_command)

        # Serialize job_metadata to JSON and shell-escape it for safe bash execution
        job_metadata_json = json.dumps(job_metadata)
        docker_env_vars_json = json.dumps(docker_env_vars)

        # Clean TAO-compliant StatefulSet setup: Pure container_handler.py approach
        run_command = f"""
umask 0 &&


{inference_command} --job {shlex.quote(job_metadata_json)} --docker_env_vars {shlex.quote(docker_env_vars_json)}
        """
        logger.info("Using run command: %s", run_command)
        run_command = ["/bin/bash", "-c", run_command]

        try:
            # Create long-lived inference service StatefulSet
            # IMPORTANT: This overrides the default container entrypoint (e.g., "flask run")
            # with our custom command that starts the persistent model server + container_handler.py
            from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import ExecutionHandler
            execution_handler = ExecutionHandler.create_handler(
                backend=BACKEND,
                container_image=image,
                job_id=job_id
            )
            # Get number of GPUs from job config (default to 1)
            num_gpus = job_config.get("num_gpus", 1)
            if num_gpus == 0:
                num_gpus = 1  # Inference requires at least 1 GPU

            try:
                success = execution_handler.create_microservice(
                    job_id=job_id,
                    image=image,
                    custom_command=run_command,
                    api_port=api_port,
                    num_gpu=num_gpus,
                    inference_microservice=True,
                    docker_env_vars=docker_env_vars  # Pass env vars to container
                )
            except RuntimeError as e:
                error_msg = str(e)
                try:
                    from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import (
                        ExecutionHandler,
                    )
                    ExecutionHandler.delete_job_with_handler(job_id, inference_microservice=True)
                    logger.info("Cleaned up failed IMS resources for job %s", job_id)
                except Exception as cleanup_err:
                    logger.warning("Failed to clean up IMS resources for job %s: %s", job_id, cleanup_err)
                if "GPU" in error_msg or "gpu" in error_msg:
                    logger.error("GPU allocation failed: %s", error_msg)
                    return Code(503, {}, f"Insufficient GPU resources: {error_msg}")
                logger.error("Microservice creation failed: %s", error_msg)
                return Code(500, {}, f"Failed to create Inference Microservice: {error_msg}")

            if not success:
                return Code(500, {}, "Failed to create Inference Microservice StatefulSet")

            # Wait for service to be ready before returning success
            service_id = f"ims-svc-{job_id}"
            logger.info("Waiting for Inference Microservice service %s to be ready", service_id)

            if BACKEND == Backend.LOCAL_K8S:
                kubernetes_handler = KubernetesHandler()
                service_status = kubernetes_handler.wait_for_service(job_id, service_name=service_id)
                if service_status != "Running":
                    logger.error("Inference Microservice service failed to become ready. Status: %s", service_status)
                    try:
                        from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import (
                            ExecutionHandler,
                        )
                        ExecutionHandler.delete_job_with_handler(job_id, inference_microservice=True)
                        logger.info("Cleaned up failed IMS resources for job %s", job_id)
                    except Exception as cleanup_err:
                        logger.warning("Failed to clean up IMS resources for job %s: %s", job_id, cleanup_err)
                    return Code(500, {}, f"Inference Microservice service failed to become ready: {service_status}")

            # For Kubernetes services, we typically use cluster IP for internal communication
            service_url = InferenceMicroserviceHandler.get_inference_microservice_url(job_id, None, api_port)

            logger.info("Inference Microservice created at %s", service_url)

            # Save job metadata to database so status callbacks can find it

            job_metadata = {
                "id": job_id,
                "action": "inference_microservice",  # Special action for inference microservices
                "status": "Running",
                "created_on": datetime.now(tz=timezone.utc),
                "last_modified": datetime.now(tz=timezone.utc),
                "experiment_id": experiment_id,
                "org_name": org_name,
                "user_id": experiment_metadata.get("user_id"),
                "network": network_arch,
                "parent_id": job_config.get("parent_id", ""),
                "num_gpu": num_gpus,
                "platform_id": None,
                "kind": "experiment",
                "specs": {},
                "workflow_status": "Running",  # Not enqueued since it's already running
            }

            write_job_metadata(job_id, job_metadata)
            logger.info("Saved inference microservice job metadata for %s", job_id)

            return Code(200, {
                "service_id": service_id,
                "service_url": service_url,
                "status": "Running",
                "endpoints": {
                    "inference": f"{service_url}/api/v1/inference",
                    "health": f"{service_url}/api/v1/health/liveness",
                    "readiness": f"{service_url}/api/v1/health/readiness",
                    "status": f"{service_url}/api/v1/status"
                },
                "job_id": job_id,
                "api_port": api_port
            }, "Inference Microservice started successfully")

        except Exception as e:
            logger.error("Error starting Inference Microservice: %s", str(e))
            return Code(500, {}, f"Failed to start Inference Microservice: {str(e)}. Try again")

    @staticmethod
    def stop_inference_microservice(
        job_id: str, auto_deletion: bool = False, reason: str = ""
    ) -> Code:
        """Stop a running Inference Microservice

        Args:
            job_id: Job ID for the microservice to stop
            auto_deletion: True if called due to auto-deletion, False if manual stop
            reason: Reason for auto-deletion (e.g. "idle_timeout_exceeded",
                    "initialization_failed", "model_loading_failed")

        Returns:
            Code object with status and result information
        """
        action = "Auto-deleting" if auto_deletion else "Stopping"
        reason_desc = f"(reason: {reason})" if reason else "manually"

        logger.info("%s Inference Microservice %s %s", action, job_id, reason_desc)

        try:
            from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import ExecutionHandler
            success = ExecutionHandler.delete_job_with_handler(job_id, inference_microservice=True)

            if success:
                is_failure = reason in (
                    "initialization_failed", "model_loading_failed"
                )
                job_status = "Error" if is_failure else "Done"
                success_message = (
                    f"auto-deleted ({reason})" if auto_deletion
                    else "stopped successfully"
                )
                logger.info(
                    "Successfully %s Inference Microservice %s",
                    "auto-deleted" if auto_deletion else "stopped", job_id
                )

                from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
                    update_job_status, get_handler_job_metadata
                )
                job_metadata = get_handler_job_metadata(job_id)
                if job_metadata:
                    experiment_id = job_metadata.get("experiment_id")
                    if experiment_id:
                        update_job_status(
                            experiment_id,
                            job_id,
                            status=job_status,
                            kind="experiments"
                        )
                        logger.info(
                            "Updated job status to %s for %s", job_status, job_id
                        )

                result = {
                    "status": "success",
                    "message": f"Inference microservice for job {job_id} {success_message}",
                    "job_id": job_id,
                    "timestamp": datetime.now().isoformat(),
                    "auto_deletion": auto_deletion,
                    "reason": reason,
                }
                return Code(200, result, f"Inference Microservice {success_message}")

            error_msg = f"Failed to {action.lower()} inference microservice for job {job_id}"
            logger.error(error_msg)
            return Code(500, {"error": error_msg}, f"Failed to {action.lower()} Inference Microservice")

        except Exception as e:
            logger.error("Error %s Inference Microservice %s: %s", action.lower(), job_id, str(e))
            return Code(500, {"error": str(e)}, f"Error {action.lower()} Inference Microservice")

    @staticmethod
    def get_inference_microservice_status(job_id: str) -> Code:
        """Gets the status of a Inference Microservice StatefulSet"""
        statefulset_name = f"ims-{job_id}"

        try:
            stat_dict = KubernetesHandler().get_statefulset_status(
                statefulset_name, replicas=1, resource_type="Inference Microservice"
            )
            status = stat_dict.get("status", "Unknown")

            return Code(200, {
                "job_id": job_id,
                "service_name": statefulset_name,
                "status": status,
                "replicas": stat_dict.get("replicas", {}),
                "pods": []
            }, f"Inference Microservice status: {status}")

        except Exception as e:
            logger.error("Error getting Inference Microservice status: %s", str(e))
            return Code(500, {}, f"Failed to get service status: {str(e)}")

    @staticmethod
    def check_inference_microservice_model_readiness(job_id: str, api_port: int = 8080) -> dict:
        """Check if Inference Microservice model is ready in StatefulSet containers

        Args:
            job_id: Job ID for the microservice
            api_port: Port number for the microservice

        Returns:
            Dictionary with readiness status and progress information
        """
        try:
            statefulset_name = f"ims-{job_id}"

            # Check if StatefulSet pods exist and are running
            try:
                stat_dict = KubernetesHandler().get_statefulset_status(
                    statefulset_name, replicas=1, resource_type="Inference Microservice"
                )
                statefulset_status = stat_dict.get("status", "Unknown")

                # If StatefulSet is running, get detailed status from the microservice
                if statefulset_status == "Running":
                    try:
                        # Get detailed status including progress
                        status_response = InferenceMicroserviceHandler.get_inference_microservice_status_direct(
                            job_id, api_port
                        )
                        return {
                            "job_id": job_id,
                            "status": "ready" if status_response.get("model_loaded") else "loading",
                            "loaded": status_response.get("model_loaded", False),
                            "loading": status_response.get("model_loading", False),
                            "initializing": status_response.get("server_initializing", False),
                            "statefulset_status": statefulset_status,
                            "progress": status_response.get("progress", {})
                        }
                    except Exception as status_err:
                        logger.warning(f"Could not get detailed status for {job_id}: {status_err}")
                        # Fallback to basic response
                        return {
                            "job_id": job_id,
                            "status": "ready",
                            "loaded": True,
                            "statefulset_status": statefulset_status
                        }
                return {
                    "job_id": job_id,
                    "status": "not_ready",
                    "loaded": False,
                    "statefulset_status": statefulset_status
                }
            except Exception:
                return {
                    "job_id": job_id,
                    "status": "not_found",
                    "loaded": False,
                    "statefulset_status": "NotFound"
                }

        except Exception as e:
            logger.error(f"Error checking Inference Microservice model readiness: {e}")
            return {"status": "error", "error": str(e), "loaded": False}

    @staticmethod
    def get_inference_microservice_url(job_id: str, endpoint: str = "inference", api_port: int = 8080) -> str:
        """Get the URL for inference microservice requests

        Args:
            job_id: Job ID for the microservice
            endpoint: Endpoint to call (inference, health, status)
            api_port: Port number for the microservice (default: 8080)

        Returns:
            Full URL for the request (always uses simple service name)
        """
        service_name = f"ims-svc-{job_id}"

        # Always use simple service name for both Kubernetes and docker-compose
        # This works reliably for intra-cluster communication and avoids DNS issues
        if BACKEND == Backend.LOCAL_DOCKER:
            url = f"http://{job_id}:{api_port}"
        else:
            url = f"http://{service_name}:{api_port}"

        if endpoint:
            url = f"{url}/api/v1/{endpoint}"

        logger.info(f"Inference microservice URL: {url}")
        return url

    @staticmethod
    def process_inference_microservice_request(
            job_id: str, request_data: Dict[str, Any], api_port: int = 8080
    ) -> Dict[str, Any]:
        """Process inference request directly to the StatefulSet microservice

        Args:
            job_id: Job ID for the microservice
            request_data: Request data to send to the microservice

        Returns:
            Response from the microservice
        """
        try:
            logger.info(f"Processing inference request for job {job_id}")

            # Get the inference URL
            inference_url = InferenceMicroserviceHandler.get_inference_microservice_url(
                job_id, "inference", api_port
            )

            # Make request to the microservice
            timeout = 300  # 5 minutes timeout for inference
            response = requests.post(
                inference_url,
                json=request_data,
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Inference request completed successfully for job {job_id}")
                return result
            if response.status_code in [202, 503]:
                # Server is initializing or loading - return appropriate response
                result = response.json()
                logger.info(
                    f"Inference microservice not ready for job {job_id}: "
                    f"{result.get('message', 'Unknown status')}"
                )
                return result
            error_msg = f"Inference request failed with status {response.status_code}"
            logger.error(f"{error_msg} for job {job_id}")
            try:
                error_detail = response.json()
                error_msg += f": {error_detail.get('error', 'Unknown error')}"
            except (ValueError, KeyError):
                error_msg += f": {response.text}"

            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat()
            }

        except requests.exceptions.Timeout:
            error_msg = f"Inference request timed out after {timeout} seconds for job {job_id}"
            logger.error(error_msg)
            # Try to get current progress to include in error response
            try:
                status_response = (
                    InferenceMicroserviceHandler.get_inference_microservice_status_direct(
                        job_id, api_port
                    )
                )
                progress_info = status_response.get("progress", {})
            except Exception:
                progress_info = {}
            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "progress": progress_info
            }
        except requests.exceptions.ConnectionError:
            error_msg = f"Could not connect to inference microservice for job {job_id}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "progress": {
                    "stage": "error",
                    "message": "Connection failed - microservice may not be running",
                    "remaining_steps": [],
                    "details": {"error": "ConnectionError"}
                }
            }
        except Exception as e:
            error_msg = f"Unexpected error during inference request for job {job_id}: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "progress": {
                    "stage": "error",
                    "message": f"Unexpected error: {str(e)}",
                    "remaining_steps": [],
                    "details": {"error": str(e)}
                }
            }

    @staticmethod
    def get_inference_microservice_status_direct(job_id: str, api_port: int = 8080) -> Dict[str, Any]:
        """Get status directly from the StatefulSet microservice

        Args:
            job_id: Job ID for the microservice

        Returns:
            Status response from the microservice including progress information
        """
        try:
            logger.info(f"Getting status for inference microservice job {job_id}")

            # Get the status URL
            status_url = InferenceMicroserviceHandler.get_inference_microservice_url(
                job_id, "status", api_port
            )

            # Make request to the microservice
            response = requests.get(status_url, timeout=30)

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Status retrieved successfully for job {job_id}")
                # Progress information is already included in the result from the server
                return result
            error_msg = f"Status request failed with status {response.status_code}"
            logger.error(f"{error_msg} for job {job_id}")
            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "progress": {
                    "stage": "error",
                    "message": error_msg,
                    "remaining_steps": [],
                    "details": {}
                }
            }

        except Exception as e:
            error_msg = f"Failed to get status for job {job_id}: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "progress": {
                    "stage": "error",
                    "message": f"Failed to connect to microservice: {str(e)}",
                    "remaining_steps": [],
                    "details": {"error": str(e)}
                }
            }

    @staticmethod
    def get_inference_microservice_status_detailed(job_id: str) -> dict:
        """Get Inference Microservice service status with model readiness information"""
        try:
            statefulset_name = f"ims-{job_id}"
            stat_dict = KubernetesHandler().get_statefulset_status(
                statefulset_name, replicas=1, resource_type="Inference Microservice"
            )

            # Check model readiness
            model_state = InferenceMicroserviceHandler.check_inference_microservice_model_readiness(job_id)

            return {
                "job_id": job_id,
                "service_name": statefulset_name,
                "status": stat_dict.get("status", "Unknown"),
                "replicas": stat_dict.get("replicas", {}),
                "model_loaded": model_state.get("loaded", False),
                "model_status": model_state.get("status", "unknown")
            }

        except Exception as e:
            logger.error(f"Error getting Inference Microservice service status: {e}")
            return {
                "job_id": job_id,
                "status": "error",
                "error": str(e)
            }
