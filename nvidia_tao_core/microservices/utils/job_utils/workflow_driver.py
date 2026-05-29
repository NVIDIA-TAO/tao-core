# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow manager for normal model actions"""
import os
import logging

from nvidia_tao_core.microservices.constants import NO_SPEC_ACTIONS_MODEL
from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_handler_type, get_job
from nvidia_tao_core.microservices.utils.handler_utils import JobContext, get_num_gpus_from_spec, is_remote_backend
from .workflow import Dependency, Job, Workflow

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


def create_job_context(
    parent_job_id,
    action,
    job_id,
    handler_id,
    user_id,
    org_name,
    kind,
    handler_metadata=None,
    specs=None,
    name=None,
    description=None,
    num_gpu=-1,
    backend_details=None,
    retain_checkpoints_for_resume=False,
    early_stop_epoch=None,
    timeout_minutes=None,
):
    """Calls the create job contexts function"""
    network = get_handler_type(handler_metadata)
    if not network:
        raise ValueError(f"Handler {handler_id} not found for user {user_id}")

    # Validate dataset paths at job creation time (for experiments using direct paths)
    if kind == "experiment" and backend_details:
        backend_type = backend_details.get('backend_type') if isinstance(backend_details, dict) else None
        if backend_type and handler_metadata:
            from nvidia_tao_core.microservices.utils.dataset_uri_validator import validate_all_dataset_uris
            is_valid, error_msg = validate_all_dataset_uris(handler_metadata, backend_type)
            if not is_valid:
                raise ValueError(f"Dataset path validation failed: {error_msg}")

    if not specs and action not in NO_SPEC_ACTIONS_MODEL:
        raise ValueError(f"Specs are required to create a job context for {action} action.")

    # Create a jobcontext
    job_context = JobContext(
        job_id,
        parent_job_id,
        network,
        action,
        handler_id,
        user_id,
        org_name,
        kind,
        specs=specs,
        name=name,
        description=description,
        num_gpu=num_gpu,
        backend_details=backend_details,
        retain_checkpoints_for_resume=retain_checkpoints_for_resume,
        early_stop_epoch=early_stop_epoch,
        timeout_minutes=timeout_minutes,
    )
    return job_context


def on_new_job(job_context):
    """Assigns dependencies for a new job

    Creates job_context dictionary and enqueues the job to workflow
    """
    deps = []
    deps.append(Dependency(type="parent"))
    deps.append(Dependency(type="specs"))
    deps.append(Dependency(type="model"))
    deps.append(Dependency(type="dataset"))

    num_gpu = job_context.num_gpu
    platform_id = job_context.platform_id

    logger.debug(f"Job action: {job_context.action}")
    if job_context.action not in ["convert", "kmeans", "annotation"]:
        if job_context.specs and job_context.specs.get("platform_id"):
            num_gpu = job_context.specs.get("num_gpu")
            platform_id = job_context.specs.get("platform_id")
        if job_context.specs:
            try:
                skip_gpu_check = is_remote_backend(job_context.backend_details)
                num_gpu = get_num_gpus_from_spec(
                    job_context.specs,
                    job_context.action,
                    network=job_context.network,
                    default=num_gpu,
                    skip_gpu_conditions_check=skip_gpu_check
                )
                logger.debug(
                    f"GPU count determined from specs for {job_context.network}/{job_context.action}: {num_gpu}"
                )
            except Exception as e:
                logger.warning(f"Failed to get GPU count from specs: {e}, using default: {num_gpu}")

    logger.debug(f"Num GPU for job {job_context.id} assigned as dependency: {num_gpu}")
    if num_gpu > 0:
        deps.append(Dependency(type="gpu", name=platform_id, num=num_gpu))

    job = {
        'user_id': job_context.user_id,
        'org_name': job_context.org_name,
        'num_gpu': num_gpu,
        'backend_details': job_context.backend_details,
        'kind': job_context.kind,
        'id': job_context.id,
        'parent_id': job_context.parent_id,
        'priority': 1,
        'action': job_context.action,
        'network': job_context.network,
        'handler_id': job_context.handler_id,
        'created_on': job_context.created_on,
        'last_modified': job_context.last_modified,
        'dependencies': deps,
        'specs': job_context.specs,
        'workflow_status': 'enqueued',
        'retain_checkpoints_for_resume': job_context.retain_checkpoints_for_resume,
        'early_stop_epoch': job_context.early_stop_epoch,
        'timeout_minutes': job_context.timeout_minutes,
    }
    j = Job(**job)
    Workflow.enqueue(j)


def on_delete_job(job_id):
    """Dequeue a job"""
    job_metadata = get_job(job_id)
    job_dict = {
        'user_id': job_metadata["user_id"],
        'org_name': job_metadata["org_name"],
        'num_gpu': job_metadata["num_gpu"],
        'backend_details': job_metadata["backend_details"],
        'kind': job_metadata["kind"],
        'id': job_metadata["id"],
        'parent_id': job_metadata["parent_id"],
        'priority': 1,
        'action': job_metadata["action"],
        'network': job_metadata["network"],
        'handler_id': job_metadata["handler_id"],
        'created_on': job_metadata["created_on"],
        'last_modified': job_metadata["last_modified"],
        'specs': job_metadata["specs"],
        'workflow_status': job_metadata["workflow_status"],
    }
    job = Job(**job_dict)
    Workflow.dequeue(job)
