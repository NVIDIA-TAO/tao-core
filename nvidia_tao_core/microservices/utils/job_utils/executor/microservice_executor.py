# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Microservice executor for microservice operations"""
from nvidia_tao_core.microservices.utils.stateless_handler_utils import BACKEND
import logging
import os

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


class MicroserviceExecutor():
    """Handles microservice operations"""

    def create_microservice_and_send_request(
            self, api_endpoint, network, action, cloud_metadata={}, specs={},
            microservice_pod_id="", num_gpu=-1,
            microservice_container="", org_name="", handler_id="",
            handler_kind="", accelerator=None, docker_env_vars={}, num_nodes=1,
            resource_shape=None, dedicated_node_group=None, backend_details=None):
        """Create a DNN container microservice pod and send request to the POD IP"""
        from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import ExecutionHandler
        # Use the base class implementation
        handler = ExecutionHandler.create_handler(
            cloud_metadata=cloud_metadata,
            job_id=microservice_pod_id,
            backend=BACKEND,
            container_image=microservice_container)
        if not handler:
            logger.error(
                f"Unable to determine appropriate handler for backend '{BACKEND}'"
                f"and job_id '{microservice_pod_id}'"
            )
            return None
        return handler.create_microservice_and_send_request(
            api_endpoint=api_endpoint,
            network=network,
            action=action,
            cloud_metadata=cloud_metadata,
            specs=specs,
            microservice_pod_id=microservice_pod_id,
            num_gpu=num_gpu,
            microservice_container=microservice_container,
            docker_env_vars=docker_env_vars,
            num_nodes=num_nodes,
            accelerator=accelerator,
            resource_shape=resource_shape,
            dedicated_node_group=dedicated_node_group,
            backend_details=backend_details
        )
