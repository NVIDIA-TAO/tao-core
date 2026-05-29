# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions and shared components for executor module"""
import os
import uuid
import logging
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from .stateless_handler_utils import get_toolkit_status
from nvidia_tao_core.microservices.enum_constants import Backend
from .stateless_handler_utils import BACKEND

# Global constants
release_name = os.getenv("RELEASE_NAME", 'tao-api')

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


def get_namespace():
    """Returns the namespace of the environment"""
    if BACKEND == Backend.LOCAL_DOCKER:
        return "default"
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        name_space = os.getenv('NAMESPACE', default="default")
        config.load_kube_config()
    else:
        with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace', 'r', encoding='utf-8') as f:
            current_name_space = f.read()
        name_space = os.getenv('NAMESPACE', default=current_name_space)
        config.load_incluster_config()
    return name_space


def get_service_in_cluster_ip(service_name, namespace="default"):
    """Get the cluster IP of a service"""
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    service = v1.read_namespaced_service(namespace=namespace, name=service_name)
    return service.spec.cluster_ip


def get_available_local_k8s_gpus():
    """Construct a dictionary where key is a UUID and value contains the gpu type and node it belongs to"""
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        config.load_kube_config()
    else:
        config.load_incluster_config()
    v1 = client.CoreV1Api()
    nodes = v1.list_node().items
    available_local_k8s_gpus = {}
    for node in nodes:
        node_name = node.metadata.name
        labels = node.metadata.labels
        label_value = labels.get("accelerator")
        if label_value:
            for gpu_type in label_value.split(","):
                platform_id = str(uuid.uuid5(uuid.NAMESPACE_X500, f"local_k8s_{gpu_type}"))
                available_local_k8s_gpus[platform_id] = {"node": node_name,
                                                         "gpu_type": gpu_type}
    return available_local_k8s_gpus


def get_owner_reference():
    """Get the owner reference for K8s resources"""
    api_instance = client.AppsV1Api()
    name_space = get_namespace()
    workflow_deployment = api_instance.read_namespaced_deployment(
        name=f"{release_name}-workflow-pod",
        namespace=name_space)
    owner_reference = client.V1OwnerReference(
        api_version=workflow_deployment.api_version,
        kind=workflow_deployment.kind,
        controller=True,
        name=workflow_deployment.metadata.name,
        uid=workflow_deployment.metadata.uid
    )
    return owner_reference


def dependency_check(num_gpu=-1, accelerator=None, job_id=None):
    """Checks for GPU dependency

    Args:
        num_gpu: Number of GPUs required
        accelerator: Accelerator type
        job_id: Job ID to check (for PBT/Hyperband resume with reused IDs)

    Returns:
        tuple: (is_available, gpu_count) where:
            - is_available (bool): True if requested GPUs are available
            - gpu_count (int): Maximum number of available GPUs on any single node
                              (-1 for non-local backends where count is unknown)
    """
    # Check if this job is using a SLURM workspace
    # SLURM manages its own GPU scheduling, so we don't check MongoDB GPU availability
    if job_id:
        try:
            from .stateless_handler_utils import get_handler_job_metadata, get_handler_metadata
            job_metadata = get_handler_job_metadata(job_id)
            if job_metadata:
                # Try to get workspace_id from job metadata
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
                            logger.debug(
                                f"[GPU_CHECK] Job {job_id} is using SLURM workspace, "
                                f"skipping GPU availability check (SLURM manages its own scheduling)"
                            )
                            return True, -1
        except Exception as e:
            logger.debug(f"[GPU_CHECK] Could not check workspace cloud_type for job {job_id}: {e}")

    if os.getenv("BACKEND", "") not in ("local-k8s", "local-docker"):
        return True, -1
    if num_gpu == -1:
        num_gpu = int(os.getenv('NUM_GPU_PER_NODE', default='1'))
    if BACKEND == Backend.LOCAL_DOCKER:
        from .job_utils.gpu_manager import gpu_manager
        logger.debug(
            f"[GPU_CHECK] Checking GPU availability for job {job_id}: "
            f"requesting {num_gpu} GPU(s)"
        )

        # Check if this job already has GPUs assigned (PBT/Hyperband resume)
        gpus_already_assigned = []
        if job_id:
            from .mongo_utils import MongoHandler
            mongo_handler = MongoHandler("tao", "gpus")
            assigned_gpus = mongo_handler.find({"job_id": job_id, "status": "assigned"})
            gpus_already_assigned = [gpu.get("id") for gpu in assigned_gpus if gpu.get("id") is not None]
            if gpus_already_assigned:
                logger.debug(
                    f"[GPU_CHECK] Job {job_id} already has {len(gpus_already_assigned)} "
                    f"GPU(s) assigned: {gpus_already_assigned} (PBT/Hyperband resume)"
                )

        # Trigger lazy GC to get accurate availability
        reclaimed = gpu_manager._reclaim_stale_gpus()
        if reclaimed > 0:
            logger.debug(f"[GPU_CHECK] Lazy GC reclaimed {reclaimed} GPU(s) from stopped containers")

        # Get available GPUs + GPUs already assigned to this job
        available_gpus = gpu_manager.get_available_gpus()
        total_available = list(available_gpus) + gpus_already_assigned
        gpu_count = len(total_available) if total_available else 0

        # DEBUG: Log GPU table state
        from .mongo_utils import MongoHandler
        mongo_handler = MongoHandler("tao", "gpus")
        all_gpus = mongo_handler.find({})
        logger.debug("[GPU_CHECK_DEBUG] GPU table state:")
        for gpu in all_gpus:
            logger.debug(
                f"[GPU_CHECK_DEBUG]   GPU {gpu.get('id')}: "
                f"status={gpu.get('status')}, job_id={gpu.get('job_id')}"
            )

        is_available = gpu_count >= num_gpu
        logger.debug(
            f"[GPU_CHECK] Result: {gpu_count} available, {num_gpu} requested, sufficient={is_available}"
        )
        return is_available, gpu_count
    label_selector = 'accelerator=' + str(accelerator)
    if not accelerator:
        label_selector = None
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        config.load_kube_config()
    else:
        config.load_incluster_config()
    v1 = client.CoreV1Api()
    nodes = {}
    # how many GPUs allocatable per node
    ret = v1.list_node(label_selector=label_selector)
    if ret.items:
        for i in ret.items:
            if i.status and i.status.allocatable:
                for k, v in i.status.allocatable.items():
                    if k == 'nvidia.com/gpu':
                        nodes[i.metadata.name] = int(v)
                        break
    # how many GPUs requested for each node
    ret = v1.list_pod_for_all_namespaces()
    if ret.items:
        for i in ret.items:
            if i.spec.node_name is not None:
                if i.spec and i.spec.containers:
                    for c in i.spec.containers:
                        if c.resources and c.resources.requests:
                            for k, v in c.resources.requests.items():
                                if k == 'nvidia.com/gpu':
                                    current = nodes.get(i.spec.node_name, 0)
                                    nodes[i.spec.node_name] = max(0, current - int(v))
    # do I have enough GPUs on one of the nodes
    max_available_gpus = max(nodes.values()) if nodes else 0
    is_available = max_available_gpus >= num_gpu
    return is_available, max_available_gpus


def get_cluster_ip(namespace='default'):
    """Get cluster IP of service"""
    try:
        # Load kubeconfig file (optional if running in-cluster)
        if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
            config.load_kube_config()
        else:
            config.load_incluster_config()
        api_instance = client.CoreV1Api()
        service = api_instance.read_namespaced_service(f"{release_name}-service", namespace)
        cluster_ip = service.spec.cluster_ip
        cluster_port = 8000
        for port in service.spec.ports:
            if port.name == "api":
                cluster_port = port.port
        return cluster_ip, cluster_port
    except Exception as e:
        logger.error(f"Error fetching ClusterIP: {e}")
        return None, None


def override_k8_status(job_name, k8_status):
    """Override kubernetes job status with toolkit status"""
    toolkit_status = get_toolkit_status(job_name)
    override_status = ""
    if k8_status == "Pending":  # We don't want to reverse done/error status to running
        if toolkit_status in ("STARTED", "RUNNING"):
            override_status = "Running"
        if not toolkit_status:
            override_status = "Pending"
    if toolkit_status == "SUCCESS":
        override_status = "Done"
    if toolkit_status == "FAILURE":
        override_status = "Error"
    return override_status


def check_service_ready(service_name, namespace):
    """Check if the specified service is ready."""
    try:
        _ = client.CoreV1Api().read_namespaced_service(name=service_name, namespace=namespace)
        return True
    except ApiException as e:
        if e.status == 404:
            return False
        raise e


def check_endpoints_ready(service_name, namespace):
    """Check if the specified service has ready endpoints."""
    try:
        endpoints = client.CoreV1Api().read_namespaced_endpoints(name=service_name, namespace=namespace)
        if not endpoints.subsets:
            return False
        for subset in endpoints.subsets:
            if subset.addresses:
                return True
        return False
    except ApiException as e:
        if e.status == 404:
            return False
        raise e


def get_all_k8s_running_resources():
    """Get all running StatefulSets and Jobs from Kubernetes cluster

    Returns:
        dict: {'statefulsets': [...], 'jobs': [...]}
    """
    if BACKEND == Backend.LOCAL_DOCKER:
        # For Docker Compose, we'll check containers separately
        return {'statefulsets': [], 'jobs': []}

    try:
        # Load kube config
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()

        namespace = os.getenv('NAMESPACE', 'default')

        # List all StatefulSets in the namespace
        apps_api = client.AppsV1Api()
        statefulsets = []
        try:
            # Get all StatefulSets
            ss_list = apps_api.list_namespaced_stateful_set(namespace=namespace)
            for ss in ss_list.items:
                job_id = ''

                # Try to get job-id from selector match labels (multinode uses this)
                if (ss.spec and ss.spec.selector and ss.spec.selector.match_labels):
                    job_id = ss.spec.selector.match_labels.get('job-id', '')

                # If not found, try pod template labels (all TAO StatefulSets have this)
                if not job_id and ss.spec and ss.spec.template and ss.spec.template.metadata:
                    template_labels = ss.spec.template.metadata.labels
                    if template_labels:
                        job_id = template_labels.get('job-id', '')

                # Skip if no job-id found (not a TAO job)
                if not job_id:
                    continue

                # Safely get status fields
                ready_replicas = 0
                if ss.status and ss.status.ready_replicas is not None:
                    ready_replicas = ss.status.ready_replicas

                statefulsets.append({
                    'job_id': job_id,
                    'name': ss.metadata.name,
                    'status': 'Running',
                    'ready_replicas': ready_replicas,
                    'desired_replicas': ss.spec.replicas or 0,
                    'creation_timestamp': ss.metadata.creation_timestamp
                })
        except Exception as e:
            logger.error(f"Error listing StatefulSets: {e}")

        # List all Jobs in the namespace
        batch_api = client.BatchV1Api()
        jobs = []
        try:
            job_list = batch_api.list_namespaced_job(
                namespace=namespace,
                label_selector="purpose=tao-toolkit-job"
            )
            for job in job_list.items:
                job_id = ''
                if job.metadata.labels:
                    job_id = job.metadata.labels.get('job-id', '')
                if not job_id:
                    # Try extracting from name
                    job_id = job.metadata.name

                if job_id:
                    jobs.append({
                        'job_id': job_id,
                        'name': job.metadata.name,
                        'status': 'Running',
                        'creation_timestamp': job.metadata.creation_timestamp
                    })
        except Exception as e:
            logger.error(f"Error listing Jobs: {e}")

        return {'statefulsets': statefulsets, 'jobs': jobs}

    except Exception as e:
        logger.error(f"Error getting K8s resources: {e}")
        return {'statefulsets': [], 'jobs': []}


# Backward compatibility aliases for utility functions


_get_name_space = get_namespace
_get_owner_reference = get_owner_reference
