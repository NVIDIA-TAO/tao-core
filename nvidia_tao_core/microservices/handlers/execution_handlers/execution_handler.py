# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base execution handler class for all backends"""

from abc import ABC
import base64
import json
import logging
import os
import time
import traceback
import uuid
import requests
from nvidia_tao_core.microservices.constants import NETWORK_CONTAINER_MAPPING
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    BACKEND,
    get_dnn_status,
    get_handler_id,
    get_handler_kind,
    get_handler_metadata,
    get_handler_job_metadata,
    get_automl_controller_info,
    save_dnn_status,
    update_job_message,
    update_job_status,
    get_job_specs,
    internal_job_status_update
)
from nvidia_tao_core.microservices.handlers.mongo_handler import MongoHandler
from nvidia_tao_core.microservices.enum_constants import Backend
from nvidia_tao_core.microservices.utils.cloud_utils import create_cs_instance
from nvidia_tao_core.microservices.handlers.container_handler import ContainerJobHandler
from datetime import datetime, timezone

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


class ExecutionHandler(ABC):
    """Base class for ALL execution handlers (K8s, Docker, Slurm, Lepton)

    This provides a unified interface for job execution across all backends.
    Both K8s executors and cloud handlers inherit from this class.

    All execution backends (K8s, Docker, Slurm, Lepton) are treated equally
    and share this common base class.
    """

    def __init__(self, backend_type):
        """Initialize base execution handler with logging"""
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.backend_type = backend_type

    # ============================================================================
    # Image Pull Status Utilities
    # ============================================================================

    @staticmethod
    def format_bytes_human_readable(size_bytes):
        """Convert bytes to human readable string.

        Args:
            size_bytes: Size in bytes

        Returns:
            str: Human readable size string (e.g., "5.2 GB")
        """
        if size_bytes is None or size_bytes < 0:
            return "unknown size"

        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    @staticmethod
    def parse_docker_image_string(image_string):
        """Parse a Docker image string into registry, repository, and tag.

        Args:
            image_string (str): Docker image string (e.g., "nvcr.io/nvidia/tao/tao-toolkit:5.0.0-pyt")

        Returns:
            tuple: (registry, repository, tag)

        Examples:
            >>> ExecutionHandler.parse_docker_image_string("nvcr.io/nvidia/tao/tao-toolkit:5.0.0-pyt")
            ("nvcr.io", "nvidia/tao/tao-toolkit", "5.0.0-pyt")
            >>> ExecutionHandler.parse_docker_image_string("ubuntu:20.04")
            ("docker.io", "ubuntu", "20.04")
        """
        if not image_string:
            raise ValueError("Image string cannot be empty")

        # Split by tag (after last colon)
        if ':' in image_string:
            parts = image_string.split(':')
            if len(parts) == 2:
                repository_part, tag = parts
            else:
                # Complex case: registry:port/repository:tag
                for i in range(len(parts) - 1, 0, -1):
                    if not parts[i].isdigit():
                        repository_part = ':'.join(parts[:i])
                        tag = ':'.join(parts[i:])
                        break
                else:
                    repository_part = ':'.join(parts[:-1])
                    tag = parts[-1]
        else:
            repository_part = image_string
            tag = "latest"

        # Split repository by first slash to separate registry from image path
        if '/' in repository_part:
            first_part = repository_part.split('/')[0]
            if '.' in first_part or first_part == 'localhost':
                registry = first_part
                repository = '/'.join(repository_part.split('/')[1:])
            else:
                registry = "docker.io"
                repository = repository_part
        else:
            registry = "docker.io"
            repository = repository_part

        return registry, repository, tag

    @staticmethod
    def get_ngc_image_size(registry, repository, tag, ngc_api_key=None):
        """Get the compressed image size from NGC registry.

        Queries the NGC registry API to get the manifest and calculate the total
        compressed size of all layers.

        Args:
            registry (str): Registry URL (e.g., "nvcr.io")
            repository (str): Image repository (e.g., "nvidia/tao/tao-toolkit")
            tag (str): Image tag (e.g., "5.0.0-pyt")
            ngc_api_key (str, optional): NGC API key for authentication

        Returns:
            dict: Dictionary containing:
                - compressed_size: Total compressed size in bytes
                - layer_count: Number of layers
                - error: Error message if failed, None otherwise
        """
        logger = logging.getLogger(__name__)
        result = {
            "compressed_size": None,
            "layer_count": 0,
            "error": None
        }

        if registry != "nvcr.io":
            result["error"] = f"Image size query only supported for nvcr.io registry, got: {registry}"
            return result

        try:
            # Get NGC API key if not provided
            if not ngc_api_key:
                from nvidia_tao_core.microservices.utils.core_utils import get_admin_key
                ngc_api_key = get_admin_key()

            if not ngc_api_key:
                result["error"] = "NGC API key not available"
                return result

            # NGC Registry API endpoint for manifest
            # Format: https://nvcr.io/v2/{repository}/manifests/{tag}
            manifest_url = f"https://{registry}/v2/{repository}/manifests/{tag}"

            headers = {
                "Accept": "application/vnd.docker.distribution.manifest.v2+json, "
                          "application/vnd.oci.image.manifest.v1+json"
            }

            # First, get an auth token using Basic auth with $oauthtoken
            # NGC uses Docker registry v2 auth flow
            auth_string = base64.b64encode(f"$oauthtoken:{ngc_api_key}".encode()).decode()
            auth_url = f"https://authn.nvidia.com/token?scope=repository:{repository}:pull&service=registry"
            auth_headers = {"Authorization": f"Basic {auth_string}"}

            auth_response = requests.get(auth_url, headers=auth_headers, timeout=30)
            if auth_response.status_code != 200:
                result["error"] = f"Failed to authenticate with NGC: {auth_response.status_code}"
                logger.warning(f"NGC auth failed: {auth_response.status_code} - {auth_response.text[:200]}")
                return result

            auth_json = auth_response.json()
            token = auth_json.get("token") or auth_json.get("access_token")
            if not token:
                result["error"] = "Failed to get authentication token from NGC"
                logger.warning("NGC auth response had no token")
                return result

            # Now get the manifest with the token
            headers["Authorization"] = f"Bearer {token}"
            manifest_response = requests.get(manifest_url, headers=headers, timeout=30)

            if manifest_response.status_code != 200:
                result["error"] = f"Failed to get manifest: {manifest_response.status_code}"
                logger.warning(f"NGC manifest failed: {manifest_response.status_code} - {manifest_response.text[:200]}")
                return result

            manifest = manifest_response.json()

            # Calculate total compressed size from layers
            total_size = 0
            layers = manifest.get("layers", [])

            # Handle manifest list (multi-arch images)
            if manifest.get("mediaType") == "application/vnd.docker.distribution.manifest.list.v2+json":
                # For multi-arch, get the first amd64 manifest
                manifests = manifest.get("manifests", [])
                for m in manifests:
                    if m.get("platform", {}).get("architecture") == "amd64":
                        # Fetch this specific manifest
                        digest = m.get("digest")
                        if digest:
                            arch_manifest_url = f"https://{registry}/v2/{repository}/manifests/{digest}"
                            arch_response = requests.get(arch_manifest_url, headers=headers, timeout=30)
                            if arch_response.status_code == 200:
                                manifest = arch_response.json()
                                layers = manifest.get("layers", [])
                        break

            for layer in layers:
                layer_size = layer.get("size", 0)
                total_size += layer_size

            # Also add config size if present
            config = manifest.get("config", {})
            total_size += config.get("size", 0)

            result["compressed_size"] = total_size
            result["layer_count"] = len(layers)
            logger.info(f"NGC image size calculated: {total_size} bytes, {len(layers)} layers")

        except requests.exceptions.Timeout:
            result["error"] = "Timeout while querying NGC registry"
        except requests.exceptions.RequestException as e:
            result["error"] = f"Network error querying NGC registry: {str(e)}"
        except Exception as e:
            logger.warning(f"Failed to get image size from NGC: {e}")
            result["error"] = f"Failed to get image size: {str(e)}"

        return result

    @classmethod
    def get_image_pull_status_message(cls, image_string, pull_phase, ngc_api_key=None, error_message=None):
        """Generate user-friendly image pull status message.

        Args:
            image_string (str): Full Docker image string
            pull_phase (str): Current phase - "checking", "pulling", "extracting",
                             "complete", "error", "not_found", "already_exists",
                             "auth_error", "not_exists_in_registry"
            ngc_api_key (str, optional): NGC API key for size queries
            error_message (str, optional): Error message if phase is "error"

        Returns:
            dict: Message dictionary with 'status' and 'message' keys
        """
        try:
            registry, repository, tag = cls.parse_docker_image_string(image_string)
        except ValueError:
            registry, repository, tag = "unknown", image_string, "latest"

        image_short = f"{repository}:{tag}"
        if pull_phase == "checking":
            return {
                "status": "PENDING",
                "message": f"Checking if Docker image '{image_short}' exists locally..."
            }

        if pull_phase == "not_found":
            message = f"Docker image '{image_short}' not found locally. Preparing to pull from registry..."
            if registry == "nvcr.io":
                size_info = cls.get_ngc_image_size(registry, repository, tag, ngc_api_key)
                if size_info["compressed_size"]:
                    compressed_size = cls.format_bytes_human_readable(size_info["compressed_size"])
                    estimated_uncompressed = cls.format_bytes_human_readable(size_info["compressed_size"] * 2)
                    message = (
                        f"Docker image '{image_short}' not found locally. "
                        f"Starting download from NGC registry. "
                        f"Compressed size: ~{compressed_size} ({size_info['layer_count']} layers). "
                        f"Note: After download, layers will be extracted. "
                        f"Estimated size after extraction: ~{estimated_uncompressed}. "
                        f"This process may take several minutes depending on network speed."
                    )
            return {
                "status": "PENDING",
                "message": message
            }

        if pull_phase == "pulling":
            message = f"Pulling Docker image '{image_short}' from {registry}. This may take several minutes..."
            if registry == "nvcr.io":
                size_info = cls.get_ngc_image_size(registry, repository, tag, ngc_api_key)
                if size_info["compressed_size"]:
                    compressed_size = cls.format_bytes_human_readable(size_info["compressed_size"])
                    estimated_uncompressed = cls.format_bytes_human_readable(size_info["compressed_size"] * 2)
                    message = (
                        f"Pulling Docker image '{image_short}' from NGC registry. "
                        f"Downloading {size_info['layer_count']} compressed layers (~{compressed_size}). "
                        f"After download completes, layers will be extracted to ~{estimated_uncompressed}. "
                        f"Please wait, this process may take 5-15 minutes on first run."
                    )
            return {
                "status": "PENDING",
                "message": message
            }

        if pull_phase == "extracting":
            message = f"Download complete. Extracting Docker image layers for '{image_short}'..."
            if registry == "nvcr.io":
                size_info = cls.get_ngc_image_size(registry, repository, tag, ngc_api_key)
                if size_info["compressed_size"]:
                    compressed_size = cls.format_bytes_human_readable(size_info["compressed_size"])
                    estimated_uncompressed = cls.format_bytes_human_readable(size_info["compressed_size"] * 2)
                    message = (
                        f"Image size: ~{compressed_size}. Download complete. "
                        f"Extracting {size_info['layer_count']} layers for '{image_short}'. "
                        f"Estimated extracted size: ~{estimated_uncompressed}. "
                        f"Extraction may take a few minutes..."
                    )
            return {
                "status": "PENDING",
                "message": message
            }

        if pull_phase == "complete":
            return {
                "status": "PENDING",
                "message": f"Docker image '{image_short}' is ready. Starting container..."
            }

        if pull_phase == "already_exists":
            return {
                "status": "PENDING",
                "message": f"Docker image '{image_short}' found locally. Starting container..."
            }

        if pull_phase == "error":
            error_msg = error_message or "Unknown error"
            return {
                "status": "FAILURE",
                "message": f"Failed to pull Docker image '{image_short}': {error_msg}"
            }

        if pull_phase == "auth_error":
            return {
                "status": "FAILURE",
                "message": (
                    f"Authentication failed for Docker image '{image_short}'. "
                    f"Please ensure your NGC API key is valid and has access to this image."
                )
            }

        if pull_phase == "not_exists_in_registry":
            return {
                "status": "FAILURE",
                "message": (
                    f"Docker image '{image_short}' does not exist in the registry. "
                    f"Please verify the image name and tag are correct."
                )
            }

        return {
            "status": "PENDING",
            "message": f"Processing Docker image '{image_short}'..."
        }

    def update_image_pull_status(self, job_id, image, pull_phase, error_message=None):
        """Update job message with image pull status.

        This is an instance method that uses self.logger and can be called by
        subclasses (DockerHandler, KubernetesHandler) to update job status
        during image pull operations.

        Args:
            job_id (str): Job ID to update
            image (str): Docker image string
            pull_phase (str): Current pull phase
            error_message (str, optional): Error message if phase is error
        """
        try:
            self.logger.debug(
                f"[IMAGE_PULL_STATUS] Entry: job_id={job_id}, image={image}, "
                f"pull_phase={pull_phase}, error={error_message}"
            )

            from nvidia_tao_core.microservices.utils.core_utils import get_admin_key
            ngc_api_key = get_admin_key()
            status_msg = self.get_image_pull_status_message(
                image,
                pull_phase,
                ngc_api_key=ngc_api_key,
                error_message=error_message
            )

            self.logger.debug(
                f"[IMAGE_PULL_STATUS] Generated status message: status={status_msg['status']}, "
                f"message={status_msg['message'][:100]}..."
            )

            # Detect if this is an AutoML experiment job
            # AutoML experiment jobs have a parent_id that points to the brain job
            is_automl_experiment = False
            brain_job_id = None
            experiment_number = "0"

            self.logger.debug(f"[IMAGE_PULL_STATUS] Looking up job metadata for job_id={job_id}")
            job_metadata = get_handler_job_metadata(job_id)

            if not job_metadata:
                self.logger.debug(
                    f"[IMAGE_PULL_STATUS] No job metadata found for job_id={job_id}, "
                    "treating as regular job"
                )
            else:
                self.logger.debug(
                    f"[IMAGE_PULL_STATUS] Found job metadata for job_id={job_id}, "
                    f"keys={list(job_metadata.keys())}"
                )

                parent_id = job_metadata.get('parent_id')
                self.logger.debug(
                    f"[IMAGE_PULL_STATUS] Job parent_id={parent_id} "
                    f"({'found' if parent_id else 'not found'})"
                )

                if parent_id:
                    # This job has a parent, check if parent is an AutoML brain job
                    # by looking for it in the automl_jobs collection
                    self.logger.debug(
                        f"[IMAGE_PULL_STATUS] Checking if parent_id={parent_id} is an AutoML brain job"
                    )

                    controller_info = get_automl_controller_info(parent_id)

                    if not controller_info:
                        self.logger.debug(
                            f"[IMAGE_PULL_STATUS] Parent {parent_id} is NOT an AutoML brain job "
                            "(no controller_info found), treating as regular job"
                        )
                    else:
                        self.logger.debug(
                            f"[IMAGE_PULL_STATUS] Parent {parent_id} IS an AutoML brain job! "
                            f"Found {len(controller_info)} experiments in controller_info"
                        )

                        # This is an AutoML experiment job!
                        # Find the experiment number by matching job_id in controller_info
                        self.logger.debug(
                            f"[IMAGE_PULL_STATUS] Searching for job_id={job_id} in controller_info "
                            f"to find experiment number"
                        )

                        for idx, rec in enumerate(controller_info):
                            rec_job_id = rec.get('job_id')
                            self.logger.debug(
                                f"[IMAGE_PULL_STATUS] Checking experiment {idx}: "
                                f"rec_job_id={rec_job_id}, status={rec.get('status')}, "
                                f"match={rec_job_id == job_id}"
                            )

                            if rec_job_id == job_id:
                                is_automl_experiment = True
                                brain_job_id = parent_id
                                experiment_number = str(idx)
                                self.logger.info(
                                    f"[IMAGE_PULL_STATUS] ✓ DETECTED AutoML experiment job: "
                                    f"job_id={job_id}, brain_job_id={brain_job_id}, "
                                    f"experiment_number={experiment_number}"
                                )
                                break

                        if not is_automl_experiment:
                            self.logger.warning(
                                f"[IMAGE_PULL_STATUS] Job {job_id} has AutoML parent {parent_id} "
                                f"but was NOT found in controller_info recommendations. "
                                f"This might indicate the controller hasn't saved state yet. "
                                f"Treating as regular job for now."
                            )
                else:
                    self.logger.debug(
                        f"[IMAGE_PULL_STATUS] Job {job_id} has no parent_id, treating as regular job"
                    )

            # Use internal_job_status_update for consistent status handling
            if is_automl_experiment:
                # For AutoML experiment jobs, use the brain job ID and experiment number
                self.logger.debug(
                    f"[IMAGE_PULL_STATUS] Calling internal_job_status_update with: "
                    f"job_id={brain_job_id}, automl=True, experiment_number={experiment_number}"
                )

                internal_job_status_update(
                    job_id=brain_job_id,
                    automl=True,
                    automl_experiment_number=experiment_number,
                    message=status_msg["message"],
                    status=status_msg["status"]
                )

                self.logger.info(
                    f"[IMAGE_PULL_STATUS] ✓ Updated AutoML experiment: job_id={job_id}, "
                    f"brain={brain_job_id}, exp={experiment_number}, phase={pull_phase}"
                )
            else:
                # For regular jobs, use the job_id directly
                self.logger.debug(
                    f"[IMAGE_PULL_STATUS] Calling internal_job_status_update with: "
                    f"job_id={job_id}, automl=False"
                )

                internal_job_status_update(
                    job_id=job_id,
                    automl=False,
                    message=status_msg["message"],
                    status=status_msg["status"]
                )

                self.logger.info(
                    f"[IMAGE_PULL_STATUS] ✓ Updated regular job: job_id={job_id}, "
                    f"phase={pull_phase}"
                )

        except Exception as e:
            # Don't fail the operation if status update fails
            self.logger.error(
                f"[IMAGE_PULL_STATUS] ✗ FAILED to update image pull status for job {job_id}: "
                f"{type(e).__name__}: {e}"
            )
            self.logger.debug(
                f"[IMAGE_PULL_STATUS] Exception traceback:\n{traceback.format_exc()}"
            )

    # ============================================================================
    # End Image Pull Status Utilities
    # ============================================================================

    def get_job_name(self, job_id):
        """Generate standardized job name

        Args:
            job_id: Job identifier

        Returns:
            str: Formatted job name
        """
        return f"tao-job-{job_id}"

    def cancel_job(self, job_id):
        """Cancel a running job

        Args:
            job_id: Job identifier

        Returns:
            bool: True if cancellation successful

        Raises:
            NotImplementedError: If backend doesn't support cancellation
        """
        raise NotImplementedError(f"{self.backend_type} does not support job cancellation")

    def get_job_status(self, job_id, **kwargs):
        """Get status of a job

        This is the unified interface for all handlers to implement.
        Each handler should implement this method with their specific logic.

        Args:
            job_name: Job identifier/name
            **kwargs: Additional handler-specific parameters

        Returns:
            str: Job status (Pending, Running, Done, Error, etc.)

        Raises:
            NotImplementedError: If backend doesn't implement status queries
        """
        raise NotImplementedError(f"{self.backend_type} does not implement get_job_status")

    def create_microservice(
        self,
        job_id,
        api_port=8000,
        num_gpu=1,
        num_nodes=1,
        image=None,
        accelerator=None,
        custom_command=None,
        inference_microservice=False,
        docker_env_vars={}
    ):
        """Create a microservice (Docker container or K8s StatefulSet)

        This method should create a microservice that exposes an API endpoint.
        For Docker: creates a container
        For K8s: creates a StatefulSet with service
        For cloud backends (Slurm/Lepton): Not applicable

        Args:
            job_id: Job/microservice identifier
            api_port: Port for the API service
            num_gpu: Number of GPUs to assign
            num_nodes: Number of nodes/replicas
            image: Container image (optional, may be set during init)
            accelerator: Accelerator type for K8s
            custom_command: Custom command to run
            inference_microservice: Whether this is an inference microservice
        Returns:
            bool: True if microservice created successfully, False otherwise

        Raises:
            NotImplementedError: If backend doesn't support microservices
        """
        raise NotImplementedError(f"{self.backend_type} does not support create_microservice")

    def send_request_to_microservice(
        self,
        api_endpoint,
        network,
        action,
        cloud_metadata={},
        specs={},
        job_id="",
        docker_env_vars={},
        port=8000,
        statefulset_replicas=1
    ):
        """Send a request to a created microservice

        This method sends a request to an already-created microservice.

        Args:
            api_endpoint: The API endpoint to call
            network: Neural network name
            action: Action to perform
            cloud_metadata: Cloud-specific metadata
            specs: Job specifications
            job_id: Job identifier
            docker_env_vars: Docker environment variables
            port: Port for the API service
            statefulset_replicas: Number of statefulset replicas

        Returns:
            requests.Response: Response from the microservice

        Raises:
            NotImplementedError: If backend doesn't support microservices
        """
        raise NotImplementedError(f"{self.backend_type} does not support send_request_to_microservice")

    def delete(self, job_id, **kwargs):
        """Delete/clean up a microservice or job

        Args:
            job_id: Job/microservice identifier

        Raises:
            NotImplementedError: If backend doesn't support deletion
        """
        raise NotImplementedError(f"{self.backend_type} does not support delete")

    def send_microservice_request(
        self,
        base_url,
        api_endpoint,
        network,
        action,
        cloud_metadata={},
        specs={},
        job_id="",
        docker_env_vars={},
        cloud_based=True,
        statefulset_replicas=1,
        max_retries=3,
        retry_delay=5
    ):
        """Send a request to an endpoint with retry logic

        Args:
            base_url: The base URL for the microservice
            api_endpoint: The API endpoint to call
            network: Neural network name
            action: Action to perform
            cloud_metadata: Cloud-specific metadata
            specs: Job specifications
            job_id: Job identifier
            docker_env_vars: Docker environment variables
            cloud_based: Whether this is a cloud-based execution
            statefulset_replicas: Number of statefulset replicas
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay in seconds between retries (default: 5)

        Returns:
            requests.Response: The response from the microservice

        Raises:
            Exception: If all retry attempts fail
        """
        if not docker_env_vars.get("TAO_API_SERVER"):
            docker_env_vars["TAO_API_SERVER"] = "https://nvidia.com"
        if not docker_env_vars.get("TAO_LOGGING_SERVER_URL"):
            docker_env_vars["TAO_LOGGING_SERVER_URL"] = "https://nvidia.com"

        if action == "retrain":
            action = "train"

        if cloud_based:
            docker_env_vars["CLOUD_BASED"] = "True"
        else:
            docker_env_vars["CLOUD_BASED"] = "False"  # Disable status callback

        request_metadata = {
            "neural_network_name": network,
            "action_name": action,
            "specs": specs,
            "cloud_metadata": cloud_metadata,
            "docker_env_vars": docker_env_vars,
        }

        if job_id:
            request_metadata["job_id"] = job_id
            request_metadata["docker_env_vars"]["JOB_ID"] = job_id

        endpoint = f"{base_url}/api/v1/internal/container_job"

        if api_endpoint == "get_job_status":
            endpoint = f"{base_url}/api/v1/internal/container_job:status"
            request_metadata = {"results_dir": specs.get("results_dir", "")}
        elif api_endpoint == "post_action" and statefulset_replicas > 1:
            request_metadata["statefulset_replicas"] = statefulset_replicas

        # Send request with retry logic
        for attempt in range(max_retries):
            try:
                if os.getenv("DEBUG_MODE", "false").lower() == "true":
                    self.logger.info(
                        f"Sending request to {endpoint} with request_metadata {request_metadata} "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                else:
                    self.logger.info(f"Sending request to {endpoint} (attempt {attempt + 1}/{max_retries})")

                if api_endpoint == "get_job_status":
                    response = requests.get(endpoint, params=request_metadata, timeout=120)
                else:
                    data = json.dumps(request_metadata)
                    response = requests.post(endpoint, data=data, timeout=120)

                if response.status_code not in [200, 201]:
                    self.logger.error(f"Error response from {endpoint}: {response.status_code} {response.text}")

                return response

            except Exception as e:
                self.logger.error(
                    f"Exception caught during sending a microservice request "
                    f"(attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                raise e

        raise Exception(f"All {max_retries} retry attempts failed for microservice request to {endpoint}")

    @staticmethod
    def get_cloud_metadata_from_workspace(workspace_metadata):
        """Get the cloud metadata from the workspace metadata"""
        cloud_type = workspace_metadata.get('cloud_type', '')
        cloud_metadata = {}
        cloud_specific_details = workspace_metadata.get('cloud_specific_details', {})
        bucket_name = cloud_specific_details.get('cloud_bucket_name', '')
        access_key = cloud_specific_details.get('access_key', '')
        secret_key = cloud_specific_details.get('secret_key', '')
        cloud_region = cloud_specific_details.get('cloud_region', '')
        endpoint_url = cloud_specific_details.get('endpoint_url', '')
        lepton_workspace_id = cloud_specific_details.get('lepton_workspace_id')
        lepton_auth_token = cloud_specific_details.get('lepton_auth_token')
        slurm_user = cloud_specific_details.get('slurm_user')
        slurm_hostname = cloud_specific_details.get('slurm_hostname')
        base_results_dir = cloud_specific_details.get('base_results_dir')
        if lepton_workspace_id and lepton_auth_token:
            cloud_metadata["lepton_workspace_id"] = lepton_workspace_id
            cloud_metadata["lepton_auth_token"] = lepton_auth_token
        if slurm_user and slurm_hostname:
            cloud_metadata["slurm_user"] = slurm_user
            cloud_metadata["slurm_hostname"] = slurm_hostname
            if base_results_dir:
                cloud_metadata["base_results_dir"] = base_results_dir
            else:
                cloud_metadata["base_results_dir"] = f"/lustre/fsw/portfolios/edgeai/users/{slurm_user}"

        if cloud_type not in cloud_metadata:
            cloud_metadata[cloud_type] = {}
        cloud_metadata[cloud_type][bucket_name] = {
            "cloud_region": cloud_region,
            "access_key": access_key,
            "secret_key": secret_key,
            "endpoint_url": endpoint_url,
        }
        return cloud_metadata

    @staticmethod
    def get_results_dir_for_workspace(workspace_metadata, job_id, specs=None):
        """Determine the results directory based on workspace type and specs.

        This function provides consistent results directory handling across all execution backends
        including Slurm, Lepton, K8s, and Docker.

        Args:
            workspace_metadata (dict): Workspace metadata containing cloud_type and cloud_specific_details
            job_id (str): Job ID for constructing the path
            specs (dict, optional): Job specs that may contain a pre-determined results_dir

        Returns:
            str: The results directory path (e.g., /results, /lustre/..., s3://..., etc.)
        """
        # First priority: Use results_dir from specs if provided (e.g., by infer_output_dir)
        if specs and specs.get("results_dir"):
            return specs["results_dir"]

        # Second priority: Construct based on workspace cloud_type
        cloud_type = workspace_metadata.get("cloud_type") if workspace_metadata else None

        if cloud_type == "slurm":
            # Slurm: Use Lustre filesystem paths
            cloud_specific_details = workspace_metadata.get("cloud_specific_details", {})
            base_results_dir = cloud_specific_details.get("base_results_dir")
            slurm_user = cloud_specific_details.get("slurm_user")

            if not base_results_dir:
                # Default SLURM path construction
                base_results_dir = f"/lustre/fsw/portfolios/edgeai/users/{slurm_user}/results"
            elif not base_results_dir.endswith("/results"):
                # Ensure it has /results suffix
                base_results_dir = f"{base_results_dir}/results"

            return f"{base_results_dir}/{job_id}"

        # Use cloud storage paths (S3/GCS) from workspace metadata
        cloud_specific_details = workspace_metadata.get("cloud_specific_details", {})
        bucket_name = cloud_specific_details.get("cloud_bucket_name", "")

        if bucket_name:
            # Construct cloud storage path
            # Format: s3://bucket/results/job_id or gs://bucket/results/job_id
            if cloud_type == "aws":
                return f"s3://{bucket_name}/results/{job_id}"
            if cloud_type == "gcp":
                return f"gs://{bucket_name}/results/{job_id}"
            # Generic cloud storage format
            return f"{cloud_type}://{bucket_name}/results/{job_id}"

        # Fallback to standard path if no bucket specified
        return f"/results/{job_id}"

    def get_available_instances(self, **kwargs):
        """Get the available instances for the backend"""
        return {}

    @staticmethod
    def create_handler(
        cloud_metadata=None,
        backend=None,
        container_image=None,
        job_id=None,
        workspace_metadata=None,
        automl_brain=False
    ):
        """Factory method to create the appropriate handler based on cloud metadata or backend

        Args:
            cloud_metadata: Dictionary containing cloud-specific credentials and info
            backend: String indicating the backend type ('local-docker', 'local-k8s', 'slurm', 'lepton')
            container_image: Container image to use (required for Docker handler)

        Returns:
            ExecutionHandler: An instance of the appropriate handler subclass

        Raises:
            ValueError: If unable to determine the appropriate handler
        """
        if workspace_metadata and cloud_metadata is None:
            cloud_metadata = ExecutionHandler.get_cloud_metadata_from_workspace(workspace_metadata)

        # Check for cloud platforms (Slurm, Lepton)
        if cloud_metadata:
            # Priority 1: Check for Slurm credentials in cloud_metadata
            slurm_hostname = cloud_metadata.get('slurm_hostname')
            slurm_user = cloud_metadata.get('slurm_user')
            if slurm_hostname and slurm_user and not automl_brain:
                from .slurm_handler import SlurmHandler
                ssh_key_path = cloud_metadata.get('ssh_key_path', "/home/www-data/.ssh/id_ed25519")
                return SlurmHandler(slurm_user, slurm_hostname, ssh_key_path)

            # Priority 2: Check for Lepton credentials in cloud_metadata
            lepton_workspace_id = cloud_metadata.get('lepton_workspace_id')
            lepton_auth_token = cloud_metadata.get('lepton_auth_token')
            if lepton_workspace_id and lepton_auth_token and not automl_brain:
                from .lepton_handler import LeptonHandler
                return LeptonHandler(lepton_workspace_id, lepton_auth_token)

        # Check for local backends (Docker, K8s)
        backend_type = backend or BACKEND

        if backend_type == Backend.LOCAL_DOCKER:
            # Import here to avoid circular dependencies
            from .docker_handler import DockerHandler
            if container_image:
                return DockerHandler(container_image)
            if job_id:
                return DockerHandler.get_handler_for_container(job_id)
            logger.error("Docker backend requires container_image or job_id")
            return None
        if backend_type == Backend.LOCAL_K8S:
            from .kubernetes_handler import KubernetesHandler
            return KubernetesHandler()
        logger.error(f"Unable to determine appropriate handler for backend '{backend_type}' and cloud_metadata")
        return None

    @staticmethod
    def get_job_status_with_handler(
        job_name,
        workspace_metadata=None,
        handler_id="",
        handler_kind="",
        network="",
        action="",
        specs=None,
        docker_env_vars=None,
        results_dir="",
        automl_exp_job=False,
        automl_experiment_id="0",
        authorized_party_nca_id="",
        org_name="",
        backend=None
    ):
        """Factory method to get job status using the appropriate handler

        This method creates the correct handler based on workspace metadata/backend
        and calls get_job_status with the appropriate kwargs for that handler.

        Args:
            job_name: Job identifier
            workspace_metadata: Workspace metadata (for cloud handlers)
            handler_id: Handler ID
            handler_kind: Handler kind
            network: Network name
            action: Action name
            specs: Job specifications
            docker_env_vars: Docker environment variables
            results_dir: Results directory path
            automl_exp_job: Whether this is an AutoML experiment job
            authorized_party_nca_id: Authorized party NCA ID
            org_name: Organization name
            backend: Backend type override (optional)

        Returns:
            str: Job status (Pending, Running, Done, Error, etc.)
        """
        # Get cloud metadata from workspace
        job_backend = backend or BACKEND
        cloud_metadata = {}
        if workspace_metadata:
            cloud_metadata = {
                'workspace_id': workspace_metadata.get('id'),
                'cloud_type': workspace_metadata.get('cloud_type'),
                'org_name': workspace_metadata.get('org_name', org_name)
            }
            cloud_specific_details = workspace_metadata.get('cloud_specific_details', {})
            cloud_metadata.update(cloud_specific_details)

        # Create the appropriate handler
        try:
            handler = ExecutionHandler.create_handler(
                cloud_metadata=cloud_metadata,
                backend=job_backend,
                job_id=job_name
            )
            if not handler:
                logger.error(f"Unable to determine appropriate handler for backend {job_backend}")
                return "Error"

            # Prepare kwargs based on handler type
            kwargs = {}

            if handler and handler.backend_type in [Backend.SLURM, Backend.LEPTON]:
                # Cloud handlers need: results_dir, workspace_metadata
                kwargs = {
                    'results_dir': results_dir,
                    'workspace_metadata': workspace_metadata
                }
            elif handler and handler.backend_type in [Backend.LOCAL_DOCKER, Backend.LOCAL_K8S]:
                # Local handlers need: network, action, specs
                if not specs:
                    specs = get_job_specs(job_name, automl=automl_exp_job, automl_experiment_id=automl_experiment_id)
                kwargs = {
                    'network': network,
                    'action': action,
                    'specs': specs or {}
                }

            # Call get_job_status with the appropriate kwargs
            status = handler.get_job_status(job_name, **kwargs)
            return status

        except Exception as e:
            logger.error(f"Error getting job status for {job_name}: {e}")
            logger.error(traceback.format_exc())
            return "Error"

    @staticmethod
    def create_job_with_handler(
        org_name,
        job_name,
        image,
        command,
        workspace_metadata=None,
        num_gpu=-1,
        num_nodes=1,
        accelerator=None,
        docker_env_vars=None,
        nv_job_metadata=None,
        automl_brain=False,
        automl_exp_job=False,
        local_cluster=False,
        backend=None,
        backend_details=None
    ):
        """Factory method to create a job using the appropriate handler

        This method creates the correct handler based on workspace metadata/backend
        and calls the appropriate job creation method.

        Args:
            org_name: Organization name
            job_name: Job name/identifier
            image: Container image to use
            command: Command to run
            workspace_metadata: Workspace metadata (for cloud handlers)
            num_gpu: Number of GPUs (-1 for default)
            num_nodes: Number of nodes
            accelerator: Accelerator type
            docker_env_vars: Docker environment variables
            nv_job_metadata: Job metadata for cloud backends
            automl_brain: Whether this is an AutoML brain job
            automl_exp_job: Whether this is an AutoML experiment job
            local_cluster: Whether this is a local cluster
            backend: Backend type override (optional)

        Returns:
            None or job creation result
        """
        # Get cloud metadata from workspace
        cloud_metadata = {}
        if workspace_metadata:
            cloud_metadata = {
                'workspace_id': workspace_metadata.get('id'),
                'cloud_type': workspace_metadata.get('cloud_type'),
                'org_name': workspace_metadata.get('org_name', org_name)
            }
            cloud_specific_details = workspace_metadata.get('cloud_specific_details', {})
            cloud_metadata.update(cloud_specific_details)

        backend_type = backend or BACKEND

        try:
            # For cloud backends (Slurm, Lepton)
            handler = ExecutionHandler.create_handler(
                cloud_metadata=cloud_metadata,
                backend=backend_type,
                automl_brain=automl_brain,
                container_image=image,
                job_id=job_name,
            )
            if not handler:
                logger.error(f"Unable to determine appropriate handler for backend '{backend_type}' and cloud_metadata")
                return
            if handler.backend_type in [Backend.SLURM, Backend.LEPTON]:
                # Cloud handlers implement create_job with different signature
                if hasattr(handler, 'create_job'):
                    partition = None
                    if backend_details and backend_details.get('backend_type') == Backend.SLURM.value:
                        partition = backend_details.get('partition', "")
                    handler.create_job(
                        image=image,
                        network="",  # Will be extracted from command
                        action="",   # Will be extracted from command
                        cloud_metadata=cloud_metadata,
                        specs={},
                        job_id=job_name,
                        docker_env_vars=docker_env_vars or {},
                        num_gpus=num_gpu,
                        num_nodes=num_nodes,
                        partition=partition
                    )
                return

            # For local-docker backend
            if backend_type == Backend.LOCAL_DOCKER:
                from .docker_handler import DockerHandler
                from nvidia_tao_core.microservices.utils.mongo_utils import mongo_secret

                docker_handler = DockerHandler(image)
                env_vars = {
                    "BACKEND": backend_type.value,
                    "HOST_PLATFORM": "local-docker",
                    "MONGOSECRET": mongo_secret,
                    "DOCKER_HOST": os.getenv("DOCKER_HOST", default="unix:///var/run/docker.sock"),
                    "DOCKER_NETWORK": os.getenv("DOCKER_NETWORK", default="tao_default")
                }
                if docker_env_vars:
                    env_vars.update(docker_env_vars)

                volumes = None
                if automl_brain:
                    # AutoML brain needs Docker socket and SSH keys to sync status from SLURM
                    host_ssh_path = os.getenv('HOST_SSH_PATH')
                    if host_ssh_path:
                        volumes = [
                            '/var/run/docker.sock:/var/run/docker.sock',
                            f'{host_ssh_path}:/root/.ssh:ro'
                        ]
                    else:
                        volumes = ['/var/run/docker.sock:/var/run/docker.sock']

                docker_handler.start_container(
                    job_name,
                    command=["/bin/bash", "-c", command],
                    num_gpus=num_gpu,
                    volumes=volumes,
                    docker_env_vars=env_vars
                )
                return

            # For local-k8s backend (default)
            from nvidia_tao_core.microservices.handlers.execution_handlers.kubernetes_handler import KubernetesHandler
            k8s_handler = KubernetesHandler()
            k8s_handler.create_job(
                org_name=org_name,
                job_name=job_name,
                image=image,
                command=command,
                num_gpu=num_gpu,
                num_nodes=num_nodes,
                accelerator=accelerator,
                docker_env_vars=docker_env_vars,
                nv_job_metadata=nv_job_metadata,
                automl_brain=automl_brain,
                automl_exp_job=automl_exp_job,
                local_cluster=local_cluster
            )

        except Exception as e:
            logger.error(f"Error creating job {job_name}: {e}")
            logger.error(traceback.format_exc())
            raise

    @staticmethod
    def delete_with_handler(job_id, workspace_metadata=None):
        """Helper method to delete a job using the appropriate handler

        This provides a convenient abstraction for deleting jobs without needing to
        instantiate handlers manually. It determines the correct handler based on
        workspace metadata or backend configuration.

        Args:
            job_id: Job/microservice identifier to delete
            workspace_metadata: Dictionary containing workspace metadata (for cloud backends)

        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            # Create appropriate handler based on workspace metadata
            handler = ExecutionHandler.create_handler(
                workspace_metadata=workspace_metadata,
                backend=BACKEND,
                job_id=job_id
            )
            if not handler:
                logger.error(f"Unable to determine appropriate handler for backend '{BACKEND}' and job_id '{job_id}'")
                return False
            result = handler.delete(job_id)
            # Backends that can confirm termination return a boolean. Preserve
            # compatibility with legacy handlers that return None on success,
            # but never discard an explicit cancellation failure.
            return result is not False

        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def create_job(self, **kwargs):
        """Create a job"""
        raise NotImplementedError(f"{self.backend_type} does not support create_job")

    def create_microservice_and_send_request(
        self,
        api_endpoint,
        network,
        action,
        cloud_metadata={},
        specs={},
        microservice_pod_id="",
        num_gpu=-1,
        microservice_container="",
        resource_shape=None,
        dedicated_node_group=None,
        backend_details={},
        docker_env_vars={},
        num_nodes=1,
        accelerator=None,
        **kwargs
    ):
        """Create a microservice and send request - unified across all backends

        This method orchestrates the creation of a microservice across different backends
        (Docker, K8s, Slurm, Lepton) and sends a request to it.

        Args:
            api_endpoint: The API endpoint to call
            network: Neural network name
            action: Action to perform
            cloud_metadata: Cloud-specific metadata
            specs: Job specifications
            microservice_pod_id: Microservice/job identifier
            num_gpu: Number of GPUs to assign
            microservice_container: Container image to use
            docker_env_vars: Docker environment variables
            num_nodes: Number of nodes for distributed jobs
            resource_shape: Resource shape for cloud backends
            dedicated_node_group: Dedicated node group for cloud backends
            accelerator: Accelerator type for K8s

        Returns:
            requests.Response: Response from the microservice, or None on failure
        """
        try:
            # Generate microservice ID if not provided
            if not microservice_pod_id:
                microservice_pod_id = str(uuid.uuid4())

            # Determine number of GPUs
            if num_gpu == -1:
                num_gpu = int(os.getenv('NUM_GPU_PER_NODE', default='1'))

            # Determine container image
            if not microservice_container:
                microservice_container = os.getenv(f'IMAGE_{NETWORK_CONTAINER_MAPPING[network]}')
                if action == "gen_trt_engine":
                    microservice_container = os.getenv('IMAGE_TAO_DEPLOY')

            # Normalize action
            if action == "retrain":
                action = "train"

            port = 8000
            response = None

            # Get or create handler instance
            # If self is already a specialized handler (Slurm/Lepton), use it
            # Otherwise, create appropriate handler via factory
            valid_backends = [Backend.SLURM, Backend.LEPTON, Backend.LOCAL_DOCKER, Backend.LOCAL_K8S]
            if hasattr(self, 'backend_type') and self.backend_type in valid_backends:
                handler = self
            else:
                handler = self.create_handler(cloud_metadata, BACKEND, container_image=microservice_container)
            # Handle job-based execution (Slurm or Lepton)
            if handler.backend_type in [Backend.SLURM, Backend.LEPTON]:
                if docker_env_vars is None:
                    docker_env_vars = {}
                docker_env_vars["CLOUD_BASED"] = "False"
                if backend_details is None:
                    backend_details = {}
                if handler.backend_type == Backend.LEPTON:
                    self.logger.info(f"Resource shape: {resource_shape}")
                    self.logger.info(f"Dedicated node group: {dedicated_node_group}")
                # Create job using the handler
                if hasattr(handler, 'create_job'):
                    partition = None
                    if backend_details and backend_details.get('backend_type') == Backend.SLURM.value:
                        partition = backend_details.get('partition', "")
                    output = handler.create_job(
                        image=microservice_container,
                        network=network,
                        action=action,
                        cloud_metadata=cloud_metadata,
                        specs=specs,
                        job_id=microservice_pod_id,
                        docker_env_vars=docker_env_vars,
                        num_gpus=num_gpu,
                        num_nodes=num_nodes,
                        resource_shape=resource_shape,
                        dedicated_node_group=dedicated_node_group,
                        partition=partition
                    )
                    job_id = microservice_pod_id
                    if handler.backend_type == Backend.SLURM and output:
                        job_id = handler.get_slurm_job_id_from_output(output)
                    message = f"Job submitted to {handler.backend_type} (Job ID: {job_id}). Waiting for resources..."
                    self.update_job_status(microservice_pod_id, "PENDING", message)
                    return response  # Cloud jobs don't return immediate responses

            elif handler.backend_type in [Backend.LOCAL_DOCKER, Backend.LOCAL_K8S]:
                # Local microservice execution (Docker or K8s)
                if handler.create_microservice(
                    job_id=microservice_pod_id,
                    api_port=port,
                    num_gpu=num_gpu,
                    num_nodes=num_nodes,
                    image=microservice_container,
                    accelerator=accelerator,
                    docker_env_vars=docker_env_vars
                ):
                    response = handler.send_request_to_microservice(
                        api_endpoint=api_endpoint,
                        network=network,
                        action=action,
                        cloud_metadata=cloud_metadata,
                        specs=specs,
                        job_id=microservice_pod_id,
                        docker_env_vars=docker_env_vars,
                        port=port,
                        statefulset_replicas=num_nodes
                    )

                    if response and response.status_code != 200 and response.text:
                        self.logger.error(f"Error when sending microservice request {response.text}")
                        internal_job_status_update(
                            microservice_pod_id,
                            message=f"Error when sending microservice request {response.text}"
                        )
                        handler.delete(microservice_pod_id)
                        return None

                    # Clean up if not a long-running action
                    if api_endpoint != "post_action":
                        handler.delete(microservice_pod_id)

                    return response

                # Microservice creation failed
                internal_job_status_update(
                    microservice_pod_id,
                    message=f"Error when creating microservice pod {microservice_pod_id}"
                )
                return None

            return None

        except Exception as e:
            exc_type = type(e).__name__
            exc_msg = str(e)
            self.logger.error(f"Exception in create_microservice_and_send_request: {exc_msg}")
            self.logger.error(traceback.format_exc())

            status_message = (
                f"Error when creating microservice pod {microservice_pod_id}: "
                f"{exc_type}: {exc_msg}"
            )
            max_msg_len = 2000
            if len(status_message) > max_msg_len:
                status_message = status_message[:max_msg_len] + "..."
            try:
                handler_id = get_handler_id(microservice_pod_id)
                handler_metadata = get_handler_metadata(microservice_pod_id)
                handler_kind = get_handler_kind(handler_metadata)
                internal_job_status_update(
                    microservice_pod_id,
                    message=status_message,
                    handler_id=handler_id,
                    kind=handler_kind
                )
                update_job_status(handler_id, microservice_pod_id, "Error", kind=handler_kind)
            except Exception as cleanup_error:
                self.logger.error(f"Error during cleanup: {str(cleanup_error)}")
                try:
                    internal_job_status_update(
                        microservice_pod_id,
                        message=f"Error when creating microservice pod {microservice_pod_id}"
                    )
                except Exception:
                    pass

            return None

    @staticmethod
    def delete_job_with_handler(job_name, inference_microservice=False):
        """Delete a job using the appropriate handler

        Args:
            job_name: Job/microservice identifier
            inference_microservice: If True, deletes IMS StatefulSet + service
                instead of a regular K8s Job
        """
        try:
            handler = ExecutionHandler.create_handler(backend=BACKEND, job_id=job_name)
            if not handler:
                logger.error(f"Unable to determine appropriate handler for backend '{BACKEND}' and job_id '{job_name}'")
                return False
            if handler.backend_type == Backend.LOCAL_K8S:
                from .kubernetes_handler import KubernetesHandler
                k8s_handler = KubernetesHandler()
                if inference_microservice:
                    k8s_handler.delete_statefulset(
                        job_name, use_ngc=False,
                        resource_type="inference_microservice"
                    )
                    k8s_handler.delete_service(
                        service_name=f"ims-svc-{job_name}"
                    )
                else:
                    k8s_handler.delete_service(job_id=job_name, service_type="flask")
                    k8s_handler.delete_job(job_name)
                # Kubernetes deletion is asynchronous; callers that need a
                # quiescence barrier must follow this accepted request with
                # ``wait_for_job_termination``.
                return True
            result = handler.delete(job_name)
            return result is not False
        except Exception as e:
            logger.error(f"Failed to delete job {job_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def wait_for_job_termination(self, job_id, timeout_seconds=120):
        """Wait for a Job and its containers/pods to be fully terminated.

        Args:
            job_id: The job ID to wait for termination
            timeout_seconds: Maximum time to wait (default 120 seconds)

        Returns:
            bool: True if job terminated, False if timeout
        """
        poll_interval = 5
        max_polls = timeout_seconds // poll_interval
        poll_count = 0
        job_terminated = False

        self.logger.debug(
            f"Waiting for job termination: job_id={job_id}, backend={BACKEND}, timeout={timeout_seconds}s"
        )

        # Handle docker-compose backend
        if BACKEND == Backend.LOCAL_DOCKER:
            from .docker_handler import DockerHandler

            while poll_count < max_polls:
                docker_handler = DockerHandler.get_handler_for_container(job_id)
                if docker_handler and docker_handler._container:
                    # Reload container status
                    try:
                        docker_handler._container.reload()
                        container_status = docker_handler._container.status
                        if container_status in ("exited", "dead", "removing", "removed"):
                            self.logger.debug(
                                f"Docker container terminated: job_id={job_id}, status={container_status}"
                            )
                            job_terminated = True
                            break
                        self.logger.debug(
                            f"Docker container still running: job_id={job_id}, status={container_status}, "
                            f"poll={poll_count}/{max_polls}"
                        )
                    except Exception as e:
                        self.logger.debug(f"Docker container no longer exists: job_id={job_id}, error={str(e)}")
                        job_terminated = True
                        break
                else:
                    # Container not found
                    self.logger.debug(f"Docker container not found: job_id={job_id}")
                    job_terminated = True
                    break

                time.sleep(poll_interval)
                poll_count += 1

        # Handle local-k8s backend
        elif BACKEND == Backend.LOCAL_K8S:
            from kubernetes import client, config
            if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                config.load_kube_config()
            else:
                config.load_incluster_config()

            batch_v1 = client.BatchV1Api()
            core_v1 = client.CoreV1Api()
            namespace = os.getenv("NAMESPACE", "default")

            while poll_count < max_polls:
                try:
                    # Check if K8s Job still exists
                    batch_v1.read_namespaced_job(job_id, namespace)
                    self.logger.debug(
                        f"K8s Job still exists, waiting for deletion: job_id={job_id}, "
                        f"poll={poll_count}/{max_polls}"
                    )
                except client.exceptions.ApiException as e:
                    if e.status == 404:
                        self.logger.debug(f"K8s Job deleted: job_id={job_id}")
                        # Also check if pods are gone
                        try:
                            pods = core_v1.list_namespaced_pod(
                                namespace,
                                label_selector=f"job-name={job_id}"
                            )
                            if len(pods.items) == 0:
                                self.logger.debug(f"All K8s job pods terminated: job_id={job_id}")
                                job_terminated = True
                                break
                            # Still waiting for pods to terminate
                            self.logger.debug(
                                f"K8s job pods still terminating: job_id={job_id}, count={len(pods.items)}"
                            )
                        except Exception as pod_err:
                            self.logger.warning(f"Error checking pods: {str(pod_err)}")
                            job_terminated = True  # Assume terminated if we can't check
                            break
                    self.logger.error(f"Error checking K8s job: {str(e)}")
                    break

                time.sleep(poll_interval)
                poll_count += 1
        else:
            self.logger.warning(f"Unknown BACKEND: {BACKEND}, assuming job terminated")
            return True

        if not job_terminated and poll_count >= max_polls:
            self.logger.warning(f"Timeout waiting for job termination: job_id={job_id}, backend={BACKEND}")
            return False

        self.logger.debug(f"Job termination confirmed: job_id={job_id}, backend={BACKEND}")
        return True

    def wait_for_termination(self, job_id, timeout_seconds=120):
        """Wait for a StatefulSet and its containers/pods to be fully terminated.

        Args:
            job_id: The job ID to wait for termination
            timeout_seconds: Maximum time to wait (default 120 seconds)

        Returns:
            bool: True if StatefulSet terminated, False if timeout
        """
        poll_interval = 5
        max_polls = timeout_seconds // poll_interval
        poll_count = 0
        sts_terminated = False

        self.logger.debug(
            f"Waiting for StatefulSet termination: job_id={job_id}, backend={BACKEND}, "
            f"timeout={timeout_seconds}s"
        )

        # Handle docker-compose backend
        if self.backend_type == Backend.LOCAL_DOCKER:
            from .docker_handler import DockerHandler

            while poll_count < max_polls:
                docker_handler = DockerHandler.get_handler_for_container(job_id)
                if docker_handler and docker_handler._container:
                    # Reload container status
                    try:
                        docker_handler._container.reload()
                        container_status = docker_handler._container.status
                        if container_status in ("exited", "dead", "removing", "removed"):
                            self.logger.debug(
                                f"Docker container terminated: job_id={job_id}, status={container_status}"
                            )
                            sts_terminated = True
                            break
                        self.logger.debug(
                            f"Docker container still running: job_id={job_id}, status={container_status}, "
                            f"poll={poll_count}/{max_polls}"
                        )
                    except Exception as e:
                        self.logger.debug(f"Docker container no longer exists: job_id={job_id}, error={str(e)}")
                        sts_terminated = True
                        break
                else:
                    # Container not found
                    self.logger.debug(f"Docker container not found: job_id={job_id}")
                    sts_terminated = True
                    break

                time.sleep(poll_interval)
                poll_count += 1

        # Handle local-k8s backend
        elif self.backend_type == Backend.LOCAL_K8S:
            from kubernetes import client, config
            if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                config.load_kube_config()
            else:
                config.load_incluster_config()

            apps_v1 = client.AppsV1Api()
            core_v1 = client.CoreV1Api()
            namespace = os.getenv("NAMESPACE", "default")
            statefulset_name = f"tao-api-sts-{job_id}"

            while poll_count < max_polls:
                try:
                    # Check if K8s StatefulSet still exists
                    apps_v1.read_namespaced_stateful_set(statefulset_name, namespace)
                    self.logger.debug(
                        f"K8s StatefulSet still exists, waiting for deletion: job_id={job_id}, "
                        f"poll={poll_count}/{max_polls}"
                    )
                except client.exceptions.ApiException as e:
                    if e.status == 404:
                        self.logger.debug(f"K8s StatefulSet deleted: job_id={job_id}")
                        # Also check if pods are gone
                        try:
                            pods = core_v1.list_namespaced_pod(
                                namespace,
                                label_selector=f"job={job_id}"
                            )
                            if len(pods.items) == 0:
                                self.logger.debug(f"All K8s StatefulSet pods terminated: job_id={job_id}")
                                sts_terminated = True
                                break
                            self.logger.debug(
                                f"K8s StatefulSet pods still terminating: job_id={job_id}, "
                                f"count={len(pods.items)}"
                            )
                        except Exception as pod_err:
                            self.logger.warning(f"Error checking pods: {str(pod_err)}")
                            sts_terminated = True  # Assume terminated if we can't check
                            break
                    self.logger.error(f"Error checking K8s StatefulSet: {str(e)}")
                    break

                time.sleep(poll_interval)
                poll_count += 1
        else:
            self.logger.warning(f"Unknown BACKEND: {BACKEND}, assuming StatefulSet terminated")
            return True

        if not sts_terminated and poll_count >= max_polls:
            self.logger.warning(f"Timeout waiting for StatefulSet termination: job_id={job_id}, backend={BACKEND}")
            return False

        self.logger.info(f"StatefulSet termination confirmed: job_id={job_id}, backend={BACKEND}")
        return True

    def get_automl_brain_job_id(self, job_id):
        """Get the AutoML brain job ID for a given job ID"""
        # Find brain job ID by searching all brain jobs for this experiment
        mongo_jobs = MongoHandler("tao", "jobs")
        all_jobs = mongo_jobs.find({'status': {'$exists': True}})
        brain_job_id = None

        for job_data in all_jobs:
            potential_brain_id = job_data.get('id')
            if potential_brain_id:
                controller_info = get_automl_controller_info(potential_brain_id)
                if isinstance(controller_info, list):
                    for idx, rec in enumerate(controller_info):
                        if rec.get("job_id") == job_id:
                            brain_job_id = potential_brain_id
                            experiment_number = str(idx)
                            self.logger.info(
                                f"Found brain job {brain_job_id} for "
                                f"experiment {experiment_number}"
                            )
                            break
                if brain_job_id:
                    break

        if not brain_job_id:
            self.logger.error(f"No brain job found for job {job_id}")

        return brain_job_id

    def update_job_status(self, job_name, status, job_message):
        """Update the status of a job"""
        brain_job_id = self.get_automl_brain_job_id(job_name)
        lookup_id_for_handler = brain_job_id if brain_job_id else job_name
        handler_id = get_handler_id(lookup_id_for_handler)
        handler_metadata = get_handler_metadata(lookup_id_for_handler)
        handler_kind = get_handler_kind(handler_metadata)

        # Map SLURM status to TAO status and update descriptive message
        # For AutoML experiments, we need to pass brain_job_id and experiment job_id separately
        update_params = {
            "handler_id": handler_id,
            "job_id": lookup_id_for_handler,
            "kind": handler_kind
        }
        if brain_job_id:
            # AutoML experiment: add experiment-specific params
            update_params["automl_expt_job_id"] = job_name
            update_params["update_automl_expt"] = True

        current_time = datetime.now(timezone.utc)
        update_job_message(
            **update_params,
            message={
                "date": current_time.strftime("%m/%d/%Y"),
                "time": current_time.strftime("%H:%M:%S"),
                "status": status,
                "message": job_message
            }
        )
        self.logger.info(f"Updated job status: {job_name} - {status} - {job_message}")

    def get_automl_aware_handler_params(self, job_id):
        """Get handler parameters that are AutoML-aware.

        For AutoML experiments, returns brain job ID for handler lookups
        and experiment job ID for status updates.

        Args:
            job_id: TAO job ID (could be regular job, brain job, or experiment job)

        Returns:
            dict: {
                'handler_lookup_id': ID to use for handler lookups,
                'is_automl_experiment': boolean,
                'experiment_job_id': experiment job_id if AutoML, None otherwise,
                'brain_job_id': brain job_id if AutoML, None otherwise
            }
        """
        # Try to find if this job_id is an AutoML experiment
        mongo_jobs = MongoHandler("tao", "jobs")
        all_jobs = mongo_jobs.find({'status': {'$exists': True}})

        for job_data in all_jobs:
            potential_brain_id = job_data.get('id')
            if potential_brain_id:
                controller_info = get_automl_controller_info(potential_brain_id)
                if isinstance(controller_info, list):
                    for rec in controller_info:
                        if rec.get("job_id") == job_id:
                            # This is an AutoML experiment!
                            return {
                                'handler_lookup_id': potential_brain_id,
                                'is_automl_experiment': True,
                                'experiment_job_id': job_id,
                                'brain_job_id': potential_brain_id
                            }

        # Not an AutoML experiment - regular job or brain job
        return {
            'handler_lookup_id': job_id,
            'is_automl_experiment': False,
            'experiment_job_id': job_id,
            'brain_job_id': job_id
        }

    def sync_status_to_database(self, job_id, results_dir, workspace_metadata):
        """Sync the status of a job to the database"""
        # Detect if this is an AutoML experiment
        automl_params = self.get_automl_aware_handler_params(job_id)

        is_automl_experiment = automl_params['is_automl_experiment']
        if is_automl_experiment:
            # For AutoML experiments, find the experiment number from the brain's controller info
            brain_job_id = automl_params['brain_job_id']
            experiment_job_id = automl_params['experiment_job_id']
            controller_info = get_automl_controller_info(brain_job_id)
            experiment_number = None
            for exp in controller_info:
                if exp.get('job_id') == job_id:
                    experiment_number = str(exp.get('id'))
                    break

            if experiment_number is None:
                self.logger.warning(f"Could not find experiment number for AutoML job {job_id}")
                experiment_number = "0"

            self.logger.info("Detected AutoML experiment")
            self.logger.info(f"Brain Job ID: {brain_job_id}")
            self.logger.info(f"Experiment Number: {experiment_number}")
            self.logger.info(f"Experiment Job ID: {experiment_job_id}")
        else:
            self.logger.info("Not an AutoML experiment")
            brain_job_id = job_id
            experiment_number = "0"

        try:
            # Fetch status lines from Lustre
            self.logger.info("Step 1: Fetching status.json...")
            status_lines = self.fetch_status_json(results_dir, workspace_metadata)

            if not status_lines:
                self.logger.info(f"No status updates found for job {job_id}")
                self.logger.info("This is expected if job hasn't started or hasn't written status yet")
                self.logger.info("=" * 80)
                return False

            self.logger.info(f"Fetched {len(status_lines)} status entries")

            # Get existing status from database to avoid duplicates
            self.logger.info("Step 2: Checking for existing status in database...")
            existing_status = get_dnn_status(
                brain_job_id,
                automl=is_automl_experiment,
                experiment_number=experiment_number
            )
            existing_count = len(existing_status) if existing_status else 0

            self.logger.info(f"Database has {existing_count} existing status entries")

            # Only save new status lines (ones we haven't saved yet)
            new_status_count = len(status_lines) - existing_count

            if new_status_count <= 0:
                self.logger.info(
                    "  No new status updates to sync "
                    f"(DB has {existing_count}, file has {len(status_lines)})"
                )
                self.logger.info("Status is already up to date")
                self.logger.info("=" * 80)
                return True

            self.logger.info(f"Found {new_status_count} NEW status entries to sync")

            # Save only the new status lines
            self.logger.info("Step 3: Saving new status entries to database...")

            saved_count = 0
            for idx, status_entry in enumerate(status_lines[existing_count:], start=1):
                try:
                    # Ensure timestamp exists
                    if 'timestamp' not in status_entry:
                        status_entry['timestamp'] = datetime.now(tz=timezone.utc).isoformat()

                    # Create callback_data structure that save_dnn_status expects
                    callback_data = {
                        'status': json.dumps(status_entry),
                        'experiment_number': experiment_number
                    }

                    self.logger.debug(f"Saving status entry {idx}/{new_status_count}: {str(status_entry)[:100]}...")

                    save_dnn_status(
                        job_id=brain_job_id,
                        automl=is_automl_experiment,
                        callback_data=callback_data,
                        experiment_number=experiment_number,
                    )
                    saved_count += 1

                except Exception as save_error:
                    self.logger.error(f"Failed to save status entry {idx}: {save_error}")
                    # Continue with next entry
                    continue

            if saved_count == new_status_count:
                self.logger.info(f"Successfully saved all {saved_count} new status entries to database")
            else:
                self.logger.warning(f"Saved {saved_count}/{new_status_count} status entries (some failed)")

            self.logger.info(f"Database now has {existing_count + saved_count} total status entries")
            self.logger.info("SYNC COMPLETE")
            self.logger.info("=" * 80)
            return True

        except Exception as e:
            self.logger.error(f"Error syncing status to database for job {job_id}: {e}")
            self.logger.error(traceback.format_exc())
            self.logger.info("=" * 80)
            return False

    def fetch_status_json(self, results_dir, workspace_metadata):
        """Fetch status.json from the cloud storage"""
        status_lines = []
        if "://" in results_dir:
            bucket_name = results_dir.split("//")[1].split("/")[0]
            results_dir = results_dir[results_dir.find(bucket_name) + len(bucket_name):]
        if workspace_metadata:
            cs_instance, _ = create_cs_instance(workspace_metadata)
            if cs_instance:
                cs_instance.download_folder(results_dir, results_dir, extensions=[".json"])
            status_file = ContainerJobHandler.get_status_file(results_dir)
            if status_file and os.path.exists(status_file):
                with open(status_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            status_entry = json.loads(line)
                            status_lines.append(status_entry)
                        except json.JSONDecodeError:
                            continue
        return status_lines

    def get_job_logs(self, job_id, tail_lines=None):
        """Get the logs of a job"""
        self.logger.info(f"[LOG_BACKEND] {self.backend_type} does not implement get_job_logs")
