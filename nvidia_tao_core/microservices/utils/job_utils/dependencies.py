# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency check modules

1. Dataset - train (tfrecords, labels, images), evaluate, inference, calibration datasets depending on task
2. Model - base_experiment, resume, .tlt from parent, .engine from parent, class map for some tasks,
   cal cache from parent for convert
3. Platflorm - GPU
4. Specs validation - Use Steve's code hardening
5. Parent job done? - Poll status from metadata

"""
import os
import logging

from nvidia_tao_core.microservices.utils.handler_utils import get_num_gpus_from_spec, is_remote_backend
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    get_handler_log_root,
    update_job_status,
    get_handler_job_metadata,
    get_handler_metadata,
    get_base_experiment_metadata,
    get_job_specs,
    get_automl_controller_info
)
from nvidia_tao_core.microservices.utils.executor_utils import dependency_check
# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


def dependency_check_parent(job_context, dependency):
    """Check if parent job is valid and in Done status"""
    parent_job_id = job_context.parent_id
    # If no parent job, this is always True
    if parent_job_id is None:
        return True, ""
    parent_job_metadata = get_handler_job_metadata(parent_job_id)
    parent_action = parent_job_metadata.get("action", "")
    if parent_action == "annotation":
        return True, ""
    parent_status = parent_job_metadata.get("status", "")
    # parent_root = os.path.join(get_jobs_root(job_context.user_id, org_name), parent_job_id)
    # Parent Job must be done or canceled
    # Parent job output folder must exist
    failure_message = ""

    # Set current job to Error if parent err's out
    if parent_status == "Error":
        handler_kind = "experiments"
        dataset_metadata = get_handler_metadata(job_context.handler_id, "datasets")
        if dataset_metadata:
            handler_kind = "datasets"
        update_job_status(job_context.handler_id, job_context.id, parent_status, kind=handler_kind)
        handler_log_root = get_handler_log_root(job_context.user_id, job_context.org_name, job_context.handler_id)
        os.makedirs(handler_log_root, exist_ok=True)
        logfile = os.path.join(handler_log_root, str(job_context.id) + ".txt")
        with open(logfile, "a", encoding='utf-8') as f:
            f.write(f"Error log: \nParent job {parent_job_id} errored out")
            f.write("\nError EOF\n")

        return False, f"Parent job {parent_job_id} errored out"

    if parent_status not in ("Done", "Canceled"):
        failure_message += f"Parent job {parent_job_id}'s status is not Done/Canceled"
    return bool(parent_status in ("Done", "Canceled")), failure_message
    # if not os.path.isdir(parent_root):
    #     failure_message += f" Parent job {parent_job_id}'s folder {parent_root} doesn't exist"
    # return bool(parent_status in ("Done", "Canceled") and os.path.isdir(parent_root)), failure_message


def dependency_check_specs(job_context, dependency):
    """Check if valid spec exists for the requested action"""
    # If specs is not None, it means specs is already loaded
    if job_context.specs is not None:
        return True, ""

    network = job_context.network
    action = job_context.action

    # For AutoML experiments, specs are stored under the brain job ID (parent_id)
    job_id_for_specs = job_context.parent_id if job_context.parent_id else job_context.id
    specs = get_job_specs(job_id_for_specs)

    failure_message = ""
    if not specs:
        failure_message = f"Specs for network {network} and action {action} can't be found"
        return False, failure_message
    return True, ""


def dependency_check_dataset(job_context, dependency):
    """Returns always true for dataset dependency check"""
    handler_id = job_context.handler_id

    handler_metadata = get_handler_metadata(handler_id, "experiments")
    if not handler_metadata:  # dataset job
        handler_metadata = get_handler_metadata(handler_id, "datasets")

    # Check if using direct paths (new approach) - skip dependency check if so
    # Direct paths don't need pull_complete status validation; user is responsible for path validity
    train_dataset_uris = handler_metadata.get("train_dataset_uris")
    eval_dataset_uri = handler_metadata.get("eval_dataset_uri")
    inference_dataset_uri = handler_metadata.get("inference_dataset_uri")

    if any([train_dataset_uris, eval_dataset_uri, inference_dataset_uri]):
        logger.info(f"Using direct dataset paths - skipping dependency check for job {job_context.id}")
        return True, ""

    valid_datset_structure = True
    train_datasets = handler_metadata.get("train_datasets", None)
    eval_dataset = handler_metadata.get("eval_dataset", None)
    inference_dataset = handler_metadata.get("inference_dataset", None)

    def append_dataset_id_to_message(dataset_metadata, valid_datset_structure):
        if not valid_datset_structure:
            return f'{dataset_metadata.get("id")}, '
        return ''

    invalid_datasets = ""

    # For dataset convert jobs, we have dataset info directly in metadata
    if not handler_metadata.get("network_arch", ""):
        valid_datset_structure = handler_metadata.get("status") == "pull_complete"
        invalid_datasets += append_dataset_id_to_message(handler_metadata, valid_datset_structure)
    elif train_datasets:
        for train_ds in train_datasets:
            dataset_metadata = get_handler_metadata(train_ds, "datasets")
            valid_datset_structure = dataset_metadata.get("status") == "pull_complete"
            invalid_datasets += append_dataset_id_to_message(dataset_metadata, valid_datset_structure)
    if eval_dataset:
        dataset_metadata = get_handler_metadata(eval_dataset, "datasets")
        valid_datset_structure = dataset_metadata.get("status") == "pull_complete"
        invalid_datasets += append_dataset_id_to_message(dataset_metadata, valid_datset_structure)
    if inference_dataset:
        dataset_metadata = get_handler_metadata(inference_dataset, "datasets")
        valid_datset_structure = dataset_metadata.get("status") == "pull_complete"
        invalid_datasets += append_dataset_id_to_message(dataset_metadata, valid_datset_structure)

    failure_message = ""
    if not valid_datset_structure:
        failure_message = (
            f"Dataset(s) {invalid_datasets} still uploading, or uploaded data "
            "doesn't match the directory structure defined for this network"
        )
    return valid_datset_structure, failure_message


def dependency_check_model(job_context, dependency):
    """Checks if valid base_experiment model exists"""
    network = job_context.network
    handler_id = job_context.handler_id

    handler_metadata = get_handler_metadata(handler_id, "experiments")
    # If it is a dataset, no model dependency
    if "train_datasets" not in handler_metadata.keys():
        return True, ""
    from nvidia_tao_core.microservices.constants import NO_PTM_MODELS
    if network in NO_PTM_MODELS:
        return True, ""
    base_experiment_ids = handler_metadata.get("base_experiment_ids", [])
    for base_experiment_id in base_experiment_ids:
        if not base_experiment_id:
            return False, "Base Experiment ID is None"
        base_experiment_metadata = get_base_experiment_metadata(base_experiment_id)
        if not base_experiment_metadata:
            # Search in the base_exp_uuid fails, search in the org_name
            base_experiment_metadata = get_handler_metadata(base_experiment_id, "experiments")

        if base_experiment_metadata.get("base_experiment_metadata", {}).get("spec_file_present"):
            if base_experiment_metadata.get("base_experiment_metadata", {}).get("specs"):
                return True, ""
            return False, "Base experiment specs Flag set to true, but specs not found"
        # For TAO models, if the B.E doesn't have a spec file attached in NGC, then we don't need to check anything
    return True, ""


def dependency_check_gpu(job_context, dependency):
    """Check if GPU dependency is met"""
    logger.debug(f"[GPU_DEP_CHECK] Starting GPU dependency check for job {job_context.id}")

    skip_gpu_check = is_remote_backend(getattr(job_context, 'backend_details', None))
    if skip_gpu_check:
        logger.debug(
            f"[GPU_DEP_CHECK] Job {job_context.id}: SLURM backend detected, "
            f"skipping per-node GPU validation (SLURM handles multi-node scheduling)"
        )

    try:
        num_gpu = get_num_gpus_from_spec(
            job_context.specs, job_context.action, network=job_context.network,
            default=dependency.num, skip_gpu_conditions_check=skip_gpu_check
        )
        logger.debug(f"[GPU_DEP_CHECK] Job {job_context.id}: Determined needs {num_gpu} GPU(s) from spec")
    except ValueError as e:
        error_message = str(e)
        logger.error(f"[GPU_DEP_CHECK] Job {job_context.id}: GPU validation failed: {error_message}")
        return False, error_message

    logger.debug(
        f"[GPU_DEP_CHECK] Job {job_context.id} ({job_context.network}/{job_context.action}): "
        f"requires {num_gpu} GPU(s), calling dependency_check()..."
    )

    gpu_available, available_gpu_count = dependency_check(
        num_gpu=num_gpu,
        accelerator=dependency.name,
        job_id=job_context.id
    )
    message = ""
    if not gpu_available:
        if available_gpu_count >= 0:
            message = (
                f"{num_gpu} GPU's needed to run this job is not available yet "
                f"(currently {available_gpu_count} GPU's available), please wait for other jobs to complete"
            )
            logger.debug(
                f"[GPU_DEP_CHECK] Job {job_context.id}: NOT MET - needs {num_gpu}, "
                f"only {available_gpu_count} available"
            )
        else:
            message = (
                f"{num_gpu} GPU's needed to run this job is not available yet, "
                f"please wait for other jobs to complete"
            )
            logger.debug(f"[GPU_DEP_CHECK] Job {job_context.id}: NOT MET - needs {num_gpu} GPU(s)")
    else:
        logger.debug(
            f"[GPU_DEP_CHECK] Job {job_context.id}: MET - {available_gpu_count} available, {num_gpu} requested"
        )
    return gpu_available, message


def dependency_check_default(job_context, dependency):
    """Returns a default value of False when dependency type is not present in dependency_type_map"""
    return False, "Requested dependency not found"


def dependency_check_automl(job_context, dependency):
    """Makes sure the automl controller has the rec_number requested at the time of creation"""
    rec_number = int(dependency.name)
    # Check if recommendation number is there and can be loaded
    # For AutoML experiments, controller info is stored under the brain job ID (parent_id)
    automl_brain_job_id = job_context.parent_id if job_context.parent_id else job_context.id
    recs_dict = get_automl_controller_info(automl_brain_job_id)
    if not recs_dict:
        return False, f"Automl controller for brain job id {automl_brain_job_id} not found yet"
    try:
        recommendation = recs_dict[rec_number]
        child_metadata = get_handler_job_metadata(str(job_context.id)) or {}
        parent_metadata = get_handler_job_metadata(automl_brain_job_id) or {}
        if (
            child_metadata.get("automl_cancel_requested") is True or
            str(recommendation.get("status", "")).lower() in ("canceled", "canceling") or
            str(parent_metadata.get("status", "")).lower() in (
                "canceled", "canceling", "paused", "pausing"
            )
        ):
            return False, "AutoML recommendation cancellation has been requested"
        return True, ""
    except Exception as e:
        logger.error("Exception thrown in dependency_check_automl: %s", str(e))
        return False, f"Recommendation number {rec_number} requested is not available yet"


dependency_type_map = {
    'parent': dependency_check_parent,
    'specs': dependency_check_specs,
    'dataset': dependency_check_dataset,
    'model': dependency_check_model,
    'gpu':  dependency_check_gpu,
    "automl": dependency_check_automl,
}
