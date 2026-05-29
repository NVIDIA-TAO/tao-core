# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Handler to execute jobs on Lepton"""

import json
import re
import time
import traceback
import uuid
from leptonai.api.v1.types.deployment import (
    LeptonDeployment,
    LeptonDeploymentUserSpec,
    LeptonContainer,
    ContainerPort,
    ResourceRequirement,
    LeptonLog,
    EnvVar
)
from leptonai.api.v1.types.job import (
    LeptonJob,
    LeptonJobState,
    LeptonJobUserSpec
)
from leptonai.api.v1.types.common import Metadata, LeptonVisibility
from leptonai.api.v2.client import APIClient
from leptonai.api.v1.types.affinity import LeptonResourceAffinity
import logging
import requests
from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import ExecutionHandler
from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_handler_metadata
from nvidia_tao_core.microservices.utils import get_admin_key
from nvidia_tao_core.microservices.enum_constants import Backend
from datetime import datetime, timezone
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

LEPTON_API_BASE_URL = "https://gateway.dgxc-lepton.nvidia.com"


def get_lepton_handler_from_workspace(workspace_id):
    """Get the Lepton handler from the workspace"""
    workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
    cloud_specific_details = workspace_metadata.get("cloud_specific_details", {})
    lepton_workspace_id = cloud_specific_details.get("lepton_workspace_id")
    lepton_auth_token = cloud_specific_details.get("lepton_auth_token")
    if lepton_workspace_id and lepton_auth_token:
        try:
            logger.info(f"Instantiating Lepton Handler for workspace {workspace_id}")
            return LeptonHandler(lepton_workspace_id, lepton_auth_token)
        except Exception as e:
            logger.error(f"Exception instantiating Lepton Handler: {e}")
    return None


class LeptonHandler(ExecutionHandler):
    """Handler to execute jobs on Lepton"""

    def __init__(self, workspace_id, auth_token):
        """Initialize the Lepton Handler"""
        super().__init__(backend_type=Backend.LEPTON)
        self.workspace_id = workspace_id
        self.auth_token = auth_token
        self.api_client = APIClient(workspace_id=workspace_id, auth_token=auth_token)

    def get_job_name(self, job_id):
        """Get the Lepton job name (limited to 36 characters)"""
        # Lepton job name is limited to 36 characters and must start with alphabetical character
        job_name = super().get_job_name(job_id)
        return job_name[:36]

    def get_shapes_for_node_group(self, node_group):
        """Get the possible shapes for a dedicated node group"""
        try:
            endpoint = f"{LEPTON_API_BASE_URL}/api/v2/workspaces/{self.workspace_id}/shapes"
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            params = {"node_groups": node_group}
            response = requests.get(endpoint, headers=headers, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            self.logger.error(f"Failed to get shapes for node group {node_group}: {response.json()}")
        except Exception as e:
            self.logger.error(f"Error in get_shapes_for_node_group for {node_group}: {str(e)}")
        return []

    def list_image_pull_secrets(self):
        """List the image pull secrets"""
        try:
            endpoint = f"{LEPTON_API_BASE_URL}/api/v2/workspaces/{self.workspace_id}/imagepullsecrets"
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            response = requests.get(endpoint, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.json()
            self.logger.error(f"Failed to list image pull secrets: {response.json()}")
        except Exception as e:
            self.logger.error(f"Error in list_image_pull_secrets: {str(e)}")
        return []

    def get_tao_image_pull_secret(self):
        """Get the TAO image pull secret"""
        secrets = self.list_image_pull_secrets()
        for secret in secrets:
            if secret.get("metadata", {}).get("id") == "tao-image-pull-secret":
                return "tao-image-pull-secret"
        self.logger.info("No TAO image pull secret found, creating one")
        ptm_key = get_admin_key(legacy_key=True)
        return self.create_image_pull_secret(ptm_key)

    def create_image_pull_secret(self, api_key, username="$oauthtoken",
                                 secret_name="tao-image-pull-secret",
                                 registry_server="https://nvcr.io"):
        """Create a image pull secret"""
        try:
            endpoint = f"{LEPTON_API_BASE_URL}/api/v2/workspaces/{self.workspace_id}/imagepullsecrets"
            headers = {
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "origin": LEPTON_API_BASE_URL,
            }
            data = {
                "metadata": {
                    "name": secret_name,
                    "id": secret_name
                },
                "spec": {
                    "registry_server": registry_server,
                    "username": username,
                    "password": api_key
                }
            }
            response = requests.post(endpoint, headers=headers, json=data, timeout=30)
            if response.status_code == 201:
                self.logger.info("Image pull secret created successfully")
                return "tao-image-pull-secret"
            self.logger.error(f"Response: {response.status_code}, {response.text}")
        except Exception as e:
            self.logger.error(f"Error in create_image_pull_secret: {str(e)}")
        return None

    def get_available_instances(self, **kwargs):
        """Get the available Lepton instances"""
        available_lepton_instances = {}
        node_groups = self.api_client.nodegroup.list_all()
        for node_group in node_groups:
            resource_shapes = self.get_shapes_for_node_group(node_group.metadata.id_)
            for resource_shape in resource_shapes:
                spec = resource_shape.get('spec', {})
                metadata = resource_shape.get('metadata', {})
                accelerator_type = spec.get('accelerator_type')
                listable_in = spec.get('listable_in', [])
                if not listable_in or 'deployment' not in listable_in or not accelerator_type:
                    continue
                instance_id = metadata.get('id')
                platform_id = str(uuid.uuid5(
                    uuid.NAMESPACE_X500,
                    f"{node_group.metadata.id_}_{instance_id}_{accelerator_type}"))
                available_lepton_instances[platform_id] = {
                    "cluster": node_group.metadata.id_,
                    "gpu_type": instance_id,
                    "instance_type": accelerator_type,
                    "gpu_count": spec.get('accelerator_num'),
                    "cpu_cores": spec.get('cpu'),
                    "system_memory": f"{spec.get('memory_in_mb', 0) / 1024}GB",
                    "gpu_memory": f"{spec.get('accelerator_memory_in_mb', 0) / 1024}GB",
                    "backend_type": "lepton"
                }
        return available_lepton_instances

    def create_deployment(self, job_id, image, command, container_port=8000,
                          resource_shape=None, dedicated_node_group=None):
        """Create a Lepton deployment"""
        secret = self.get_tao_image_pull_secret()
        if secret:
            image_pull_secrets = [secret]
        else:
            self.logger.error("Failed to get TAO image pull secret, attempting deployment without it")
            image_pull_secrets = []
        if dedicated_node_group:
            self.logger.info(
                f"Creating deployment with dedicated node group {dedicated_node_group} "
                f"and resource shape {resource_shape}"
            )
            affinity = LeptonResourceAffinity(allowed_dedicated_node_groups=[dedicated_node_group])
        else:
            affinity = None
        lepton_deployment = LeptonDeployment(
            metadata=Metadata(name=self.get_job_name(job_id)),
            spec=LeptonDeploymentUserSpec(
                container=LeptonContainer(
                    image=image,
                    command=command,
                    ports=[ContainerPort(container_port=container_port)]),
                resource_requirement=ResourceRequirement(
                    resource_shape=resource_shape,
                    min_replicas=1,
                    max_replicas=1,
                    affinity=affinity
                ),
                image_pull_secrets=image_pull_secrets
            )
        )
        self.api_client.deployment.create(lepton_deployment)
        self.logger.info(f"Lepton Deployment {job_id} created successfully")

    def get_deployment(self, job_id):
        """Get deployment details by job_id."""
        deployment = self.api_client.deployment.get(self.get_job_name(job_id))
        return deployment

    def delete_deployment(self, job_id):
        """Delete deployment by job_id."""
        self.api_client.deployment.delete(self.get_job_name(job_id))
        self.logger.info(f"Lepton Deployment {job_id} deleted successfully")

    def scale_down_deployment(self, job_id, min_replicas=0):
        """Scale down deployment to minimum replicas."""
        spec = LeptonDeployment(
            spec=LeptonDeploymentUserSpec(
                resource_requirement=ResourceRequirement(
                    min_replicas=min_replicas,
                    max_replicas=min_replicas
                )
            )
        )
        self.api_client.deployment.update(self.get_job_name(job_id), spec)
        self.logger.info(f"Lepton Deployment {job_id} scaled down successfully")

    def scale_up_deployment(self, job_id, max_replicas=1):
        """Scale up deployment to specified max replicas."""
        spec = LeptonDeployment(
            spec=LeptonDeploymentUserSpec(
                resource_requirement=ResourceRequirement(
                    min_replicas=1,
                    max_replicas=max_replicas
                )
            )
        )
        self.api_client.deployment.update(self.get_job_name(job_id), spec)
        self.logger.info(f"Lepton Deployment {job_id} scaled up successfully")

    def wait_for_deployment(self, job_id):
        """Wait for deployment to become ready."""
        timeout = 300  # 5 minutes
        while True:
            deployment = self.get_deployment(job_id)
            deployment_status = deployment.status.state
            if deployment_status == "Ready":
                break
            time.sleep(5)
            timeout -= 5
            if timeout <= 0:
                self.logger.error(f"Deployment {job_id} timed out waiting for status: {deployment_status}")
                return None
        return deployment

    def get_node_group(self, node_group_id):
        """Get a node group"""
        node_groups = self.api_client.nodegroup.list_all()
        for node_group in node_groups:
            if node_group.metadata.id_ == node_group_id:
                return node_group
        return None

    def create_job(
            self,
            image,
            network,
            action,
            cloud_metadata={},
            specs={},
            job_id="",
            docker_env_vars={},
            resource_shape=None,
            dedicated_node_group=None,
            num_nodes=1,
            **kwargs
    ):
        """Create a job on Lepton"""
        command = []
        if num_nodes > 1:
            command += [
                "wget -O init.sh https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh",
                "chmod +x init.sh",
                "source init.sh",
            ]
        command.append(
            "python -m nvidia_tao_core.microservices.handlers.container_handler "
            f"--neural-network-name {network} "
            f"--action-name {action} "
            f"--job-id {job_id} "
            f"--specs '{json.dumps(specs)}' "
            f"--docker-env-vars '{json.dumps(docker_env_vars)}'"
        )
        command = "; ".join(command)

        secret = self.get_tao_image_pull_secret()
        if secret:
            image_pull_secrets = [secret]
        else:
            self.logger.error("Failed to get TAO image pull secret, attempting deployment without it")
            image_pull_secrets = []
        if dedicated_node_group:
            self.logger.info(
                f"Creating job with dedicated node group {dedicated_node_group} "
                f"and resource shape {resource_shape}"
            )
            affinity = LeptonResourceAffinity(allowed_dedicated_node_groups=[dedicated_node_group])
        else:
            affinity = None
        envs = [EnvVar(name='CLOUD_METADATA', value=json.dumps(cloud_metadata))]
        if num_nodes > 1:
            envs += [EnvVar(name="WORLD_SIZE", value=str(num_nodes))]
            envs += [EnvVar(name="SAVE_ON_EACH_NODE", value="True")]

        shared_memory_size = None
        if "LEPTON_SHARED_MEMORY_SIZE" in docker_env_vars:
            try:
                shared_memory_size = int(docker_env_vars["LEPTON_SHARED_MEMORY_SIZE"])
                self.logger.info(f"Setting lepton job shared memory size to {shared_memory_size} MiB")
            except Exception as e:
                self.logger.error(f"Failed to parse LEPTON_SHARED_MEMORY_SIZE: {e}")
                shared_memory_size = None
        if not shared_memory_size:
            shared_memory_size = 131072  # Default to 128GB

        lepton_job = LeptonJob(
            metadata=Metadata(
                name=self.get_job_name(job_id),
                visibility=LeptonVisibility.PRIVATE
            ),
            spec=LeptonJobUserSpec(
                container=LeptonContainer(
                    image=image,
                    command=["/bin/bash", "-c", command],
                ),
                shared_memory_size=shared_memory_size,
                affinity=affinity,
                intra_job_communication=True if num_nodes > 1 else None,
                parallelism=num_nodes if num_nodes > 1 else None,
                completions=num_nodes,
                resource_shape=resource_shape,
                ttl_seconds_after_finished=259200,
                image_pull_secrets=image_pull_secrets,
                log=LeptonLog(enable_collection=True,
                              save_termination_logs=True),
                envs=envs
            )
        )
        self.api_client.job.create(lepton_job)
        self.logger.info(f"Lepton Job {job_id} created successfully")
        return lepton_job

    def delete_job(self, job_id):
        """Delete a job on Lepton"""
        try:
            self.api_client.job.delete(self.get_job_name(job_id))
            self.logger.info(f"Lepton Job {job_id} deleted successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete Lepton job {job_id}: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def delete(self, job_id):
        """Delete a job on Lepton"""
        job = self.get_lepton_job(self.get_job_name(job_id))
        success = True
        if job:
            success = self.delete_job(job_id)
        else:
            self.logger.error(f"Lepton Job {job_id} not found, skipping deletion")
        return success

    def get_job_status(self, job_id, **kwargs):
        """Get job status - unified interface for ExecutionHandler

        Args:
            job_id: Job identifier
            **kwargs: Additional parameters (results_dir and workspace_metadata)

        Returns:
            str: Job status (Pending, Running, Done, Error, etc.)
        """
        self.logger.info(f"Checking Lepton job {job_id} status")
        lepton_status = self.get_tao_job_status(job_id)
        self.logger.info(f"Lepton job {job_id} status: {lepton_status}")

        from nvidia_tao_core.microservices.handlers.container_handler import ContainerJobHandler
        results_dir = kwargs.get("results_dir")
        workspace_metadata = kwargs.get("workspace_metadata")
        if not results_dir or not workspace_metadata:
            self.logger.error(f"No results_dir or workspace_metadata found for Lepton job {job_id}")
            return "Error"
        if lepton_status is not None:
            return lepton_status
        try:
            self.logger.info(f"Lepton job {job_id} status is Completed, getting final status from status.json")
            container_job_status = ContainerJobHandler.get_current_job_status(
                results_dir,
                workspace_metadata,
                job_id=job_id
            )
            self.logger.info(f"Lepton job {job_id} status from status.json: {container_job_status}")
            return container_job_status
        except Exception as e:
            self.logger.error(f"Failed to get Lepton job status for {job_id}: {e}")
            return "Pending"

    def get_lepton_job(self, job_name):
        """Get a Lepton job"""
        try:
            jobs = self.api_client.job.list_all(job_query_mode="alive_and_archive", q=job_name)
            for job in jobs:
                if job.metadata.name == job_name:
                    return job
            return None
        except Exception as e:
            self.logger.error(f"Failed to get Lepton job for {job_name}: {e}")
            return None

    def get_lepton_job_status(self, job_name):
        """Get a Lepton job"""
        try:
            lepton_job_name = self.get_job_name(job_name)
            self.logger.info(f"Getting Lepton job status for {lepton_job_name}")
            job = self.get_lepton_job(lepton_job_name)
            if not job:
                return LeptonJobState.Unknown
            if not job.status or not job.status.state:
                return LeptonJobState.Awaiting
            return job.status.state
        except Exception as e:
            self.logger.error(f"Failed to get Lepton job status for {job_name}: {e}")
            return LeptonJobState.Unknown

    def get_tao_job_status(self, job_name):
        """Get the status of a TAO job"""
        job_state = self.get_lepton_job_status(job_name)
        self.logger.info(f"Lepton job {job_name} status: {job_state}")

        if job_state == LeptonJobState.Unknown:
            self.logger.warning(f"Lepton job {job_name} not found")
            return "Pending"
        if job_state == LeptonJobState.Queueing:
            message = f"Lepton job {job_name} is queueing"
            self.update_job_status(job_name, "RUNNING", message)
            return "Pending"
        if job_state == LeptonJobState.Starting:
            message = f"Lepton job {job_name} is starting"
            self.update_job_status(job_name, "RUNNING", message)
            return "Pending"
        if job_state == LeptonJobState.Running:
            message = f"Lepton job {job_name} is running"
            self.update_job_status(job_name, "RUNNING", message)
            return "Running"
        if job_state == LeptonJobState.Completed:
            return None
        if job_state == LeptonJobState.Stopped:
            self.update_job_status(job_name, "PAUSED", f"Lepton job {job_name} was stopped (status: {job_state})")
            return "Paused"
        if job_state == LeptonJobState.Deleted:
            self.update_job_status(job_name, "CANCELLED", f"Lepton job {job_name} was deleted (status: {job_state})")
            return "Canceled"
        if job_state == LeptonJobState.Archived:
            self.update_job_status(job_name, "FAILURE", f"Lepton job {job_name} was archived (status: {job_state})")
            return "Error"
        if job_state == LeptonJobState.Failed:
            self.update_job_status(job_name, "FAILURE", f"Lepton job {job_name} failed with status: {job_state}")
            return "Error"

        self.logger.warning(f"Unknown Lepton job status '{job_state}' for job {job_name}")
        return "Pending"

    def send_deployment_request(
        self,
        api_endpoint,
        network,
        action,
        cloud_metadata={},
        specs={},
        job_id="",
        docker_env_vars={},
        max_retries=3,
        retry_delay=5,
    ):
        """Send a request to the Lepton microservice.

        This method waits for the deployment to be ready and then sends a request
        to the microservice. The retry logic is handled by the base class
        send_microservice_request method.
        """
        deployment = self.wait_for_deployment(job_id)
        if deployment and deployment.status and deployment.status.endpoint:
            base_url = deployment.status.endpoint.external_endpoint
            if base_url:
                return self.send_microservice_request(
                    base_url=base_url,
                    api_endpoint=api_endpoint,
                    network=network,
                    action=action,
                    cloud_metadata=cloud_metadata,
                    specs=specs,
                    job_id=job_id,
                    docker_env_vars=docker_env_vars,
                    cloud_based=False,
                    max_retries=max_retries,
                    retry_delay=retry_delay)
        self.logger.error(f"Failed to send deployment request for {job_id}")
        return None

    def get_job_logs(self, job_id, tail_lines=None):
        """Get the logs of a job"""
        try:
            lepton_job_name = self.get_job_name(job_id)
            job = self.get_lepton_job(lepton_job_name)
            if not job or not job.status:
                return None
            creation_time = job.status.creation_time or "now"
            end_time = job.status.completion_time or "now"
            start_ns = self.preprocess_time_to_nanoseconds(creation_time)
            end_ns = self.preprocess_time_to_nanoseconds(end_time)
            response = self.api_client.log.get_log(name_or_job=job, start=start_ns, end=end_ns)
            log_lines = []
            if response and response.get("data", {}).get("result", []):
                results = response.get("data", {}).get("result", [])
                for result in results:
                    values = result.get("values", [])
                    for value in values:
                        idx = 0
                        if len(value) > 1:
                            idx = 1
                        log_line = value[idx]
                        log_lines.append(log_line)
            if not log_lines:
                return None
            if tail_lines is not None:
                log_lines = log_lines[-tail_lines:]
            log_lines = reversed(log_lines)
            return "\n".join(log_lines)
        except Exception as e:
            self.logger.error(f"Failed to get Lepton job logs for {job_id}: {e}")
            return None

    def preprocess_time_to_nanoseconds(self, input_time):
        """Convert various time formats to nanosecond epoch timestamp for Lepton API.

        Supported formats:
        - "now" - current time
        - "YYYY-MM-DD HH:MM:SS[.ffffff]" or "YYYY/MM/DD HH:MM:SS[.ffffff]"
        - Existing epoch timestamp (seconds, milliseconds, microseconds, or nanoseconds)

        Args:
            input_time: Time string or epoch timestamp

        Returns:
            int: Nanosecond epoch timestamp
        """
        # Handle existing epoch timestamps
        if isinstance(input_time, (int, float)) or (
            isinstance(input_time, str) and re.fullmatch(r"-?\d+", input_time.strip())
        ):
            epoch_int = int(input_time)
            abs_val = abs(epoch_int)
            if abs_val < 100_000_000_000:  # seconds
                return epoch_int * 1_000_000_000
            if abs_val < 100_000_000_000_000:  # milliseconds
                return epoch_int * 1_000_000
            if abs_val < 100_000_000_000_000_000:  # microseconds
                return epoch_int * 1_000
            # nanoseconds
            return epoch_int

        # Get current time
        now = datetime.now(timezone.utc)

        # Replace / with -
        input_time = input_time.replace("/", "-")

        # Pad microseconds to 6 digits
        input_time = re.sub(
            r"(\.\d{1,5})(?!\d)", lambda m: m.group(1).ljust(7, "0"), input_time
        )

        str_time_format = "%Y-%m-%d %H:%M:%S.%f"
        if input_time.lower() == "now":
            input_time = now.strftime(str_time_format)

        # Parse the time
        try:
            parsed_time = datetime.fromisoformat(input_time)
            parsed_time = parsed_time.replace(tzinfo=timezone.utc)
        except ValueError as e:
            self.logger.error(f"Error: Invalid time format '{input_time}': {e}")
            return None

        # Convert to nanoseconds
        return int(parsed_time.timestamp() * 1_000_000_000)
