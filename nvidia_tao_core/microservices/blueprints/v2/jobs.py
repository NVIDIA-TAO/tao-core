# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jobs blueprint for API v2 job management endpoints."""

import ast
import logging
import math
import os
from flask import Blueprint, request, jsonify, make_response, send_from_directory
from marshmallow import ValidationError

from nvidia_tao_core.microservices.decorators import disk_space_check
from nvidia_tao_core.microservices.utils.auth_utils import authentication
from nvidia_tao_core.microservices.handlers.job_handler import JobHandler
from nvidia_tao_core.microservices.handlers.experiment_handler import ExperimentHandler
from nvidia_tao_core.microservices.handlers.dataset_handler import DatasetHandler
from nvidia_tao_core.microservices.handlers.spec_handler import SpecHandler
from nvidia_tao_core.microservices.handlers.model_handler import ModelHandler
from nvidia_tao_core.microservices.utils.core_utils import DataMonitorLogTypeEnum, log_api_error
from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_handler_id_and_kind
from nvidia_tao_core.microservices.utils.filter_utils import filtering, pagination
from nvidia_tao_core.microservices.utils.handler_utils import validate_uuid
from nvidia_tao_core.microservices.utils.basic_utils import (
    get_experiment,
    get_dataset,
    get_job
)
from .schemas import (
    ErrorRsp,
    GpuDetails,
    JobReq,
    DatasetJobReq,
    ExperimentJobReq,
    DatasetJobRsp,
    ExperimentJobRsp,
    JobListRsp,
    JobResume,
    LoadAirgappedExperimentsReq,
    LoadAirgappedExperimentsRsp,
    MessageOnly,
    PublishModel,
    JobEventsRsp
)

logger = logging.getLogger(__name__)

# v2 Jobs Blueprint - URL prefix will be set during registration
jobs_bp_v2 = Blueprint('jobs_v2', __name__, template_folder='templates')


def _create_virtual_dataset_for_direct_paths(user_id, org_name, request_dict):
    """Create a minimal virtual dataset record in MongoDB for direct path-based dataset jobs.

    When users provide dataset paths instead of a dataset_id, we create a lightweight
    dataset shell that allows all downstream job machinery to work unchanged.

    Args:
        user_id (str): The authenticated user's ID.
        org_name (str): The organization name.
        request_dict (dict): The deserialized request containing path fields.

    Returns:
        str: The newly created virtual dataset_id (UUID).
    """
    import uuid as uuid_module
    from datetime import datetime, timezone
    from nvidia_tao_core.microservices.utils.stateless_handler_utils import write_handler_metadata
    from nvidia_tao_core.microservices.handlers.mongo_handler import MongoHandler
    from nvidia_tao_core.microservices.utils.basic_utils import get_user_datasets, get_dataset_actions

    dataset_id = str(uuid_module.uuid4())
    dataset_type = request_dict.get("dataset_type", "object_detection")
    dataset_format = request_dict.get("dataset_format")
    if not dataset_format:
        from nvidia_tao_core.microservices.utils.core_utils import read_network_config
        nc = read_network_config(dataset_type)
        formats = nc.get("api_params", {}).get("formats", [])
        dataset_format = formats[0] if formats else "custom"

    # Try to resolve valid actions for this type+format combination
    default_actions = [
        "auto_label", "augment", "analyze", "validate_annotations",
        "validate_images", "dataset_convert", "generate"
    ]
    try:
        actions = get_dataset_actions(dataset_type, dataset_format)
    except Exception:
        actions = default_actions

    # Infer use_for (intent) from which URI fields are populated
    use_for = []
    if request_dict.get("train_dataset_uris"):
        use_for.append("training")
    if request_dict.get("eval_dataset_uri"):
        use_for.append("evaluation")
    if request_dict.get("inference_dataset_uri"):
        use_for.append("testing")

    # Extract cloud_file_path from the primary URI so get_source_root() resolves correctly.
    # For aws://bucket/path/to/data, cloud_file_path = "path/to/data" (everything after bucket/).
    cloud_file_path = ""
    train_uris = request_dict.get("train_dataset_uris") or []
    primary_uri = (train_uris[0] if train_uris else
                   request_dict.get("eval_dataset_uri") or
                   request_dict.get("inference_dataset_uri") or
                   request_dict.get("calibration_dataset_uri"))
    if primary_uri and "://" in primary_uri:
        _, path_after_protocol = primary_uri.split("://", 1)
        parts = path_after_protocol.split("/", 1)
        if len(parts) > 1:
            cloud_file_path = parts[1]

    now = datetime.now(tz=timezone.utc).isoformat()
    metadata = {
        "id": dataset_id,
        "user_id": user_id,
        "org_name": org_name,
        "type": dataset_type,
        "format": dataset_format,
        "status": "pull_complete",
        "created_on": now,
        "last_modified": now,
        "name": "Direct-path dataset",
        "shared": False,
        "actions": actions,
        "use_for": use_for,
        "train_dataset_uris": request_dict.get("train_dataset_uris"),
        "eval_dataset_uri": request_dict.get("eval_dataset_uri"),
        "inference_dataset_uri": request_dict.get("inference_dataset_uri"),
        "calibration_dataset_uri": request_dict.get("calibration_dataset_uri"),
        "workspace": request_dict.get("workspace"),
        "cloud_file_path": cloud_file_path,
        "base_experiment_ids": request_dict.get("base_experiment_ids", []),
    }

    write_handler_metadata(dataset_id, metadata, "dataset")

    # Register the dataset with the user
    mongo_users = MongoHandler("tao", "users")
    datasets = get_user_datasets(user_id, mongo_users)
    datasets.append(dataset_id)
    mongo_users.upsert({'id': user_id}, {'id': user_id, 'datasets': datasets})

    logger.info("Created virtual dataset %s for direct-path job (user=%s, org=%s)", dataset_id, user_id, org_name)
    return dataset_id


@jobs_bp_v2.route('/orgs/<org_name>/jobs', methods=['POST'])
@disk_space_check
def job_create(org_name):
    """Create new Job.

    ---
    post:
      tags:
      - JOB
      summary: Create new Job
      description: Asynchronously starts an Action and returns corresponding Job ID
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      requestBody:
        content:
          application/json:
            schema: JobReq
        description: Metadata for new Job (base_experiment_ids or network_arch required)
        required: true
      responses:
        201:
          description: Returned the new Job
          content:
            application/json:
              schema: JobRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    request_data = request.get_json(force=True)

    try:
        schema = JobReq()
        request_dict = schema.dump(schema.load(request_data))
    except Exception as e:
        metadata = {"error_desc": f"Validation error for job: {str(e)}", "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    kind = request_dict.get('kind')  # Already validated by schema deserialization and serialization
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    dataset_id = None
    if kind == 'dataset':
        dataset_id = request_dict.get('dataset_id', None)
        message = validate_uuid(dataset_id=dataset_id)
        if message:
            metadata = {"error_desc": message, "error_code": 2}
            schema = ErrorRsp()
            schema_dict = schema.dump(schema.load(metadata))
            return make_response(jsonify(schema_dict), 400)

        # Handle direct dataset paths (no dataset_id provided)
        if not dataset_id:
            has_paths = any([
                request_dict.get("train_dataset_uris"),
                request_dict.get("eval_dataset_uri"),
                request_dict.get("inference_dataset_uri"),
                request_dict.get("calibration_dataset_uri"),
            ])
            if has_paths:
                dataset_id = _create_virtual_dataset_for_direct_paths(user_id, org_name, request_dict)
            else:
                metadata = {
                    "error_desc": (
                        "Either 'dataset_id' or at least one dataset path field "
                        "('train_dataset_uris', 'eval_dataset_uri', 'inference_dataset_uri', "
                        "'calibration_dataset_uri') is required for dataset jobs."
                    ),
                    "error_code": 6
                }
                schema = ErrorRsp()
                schema_dict = schema.dump(schema.load(metadata))
                return make_response(jsonify(schema_dict), 400)
    parent_job_id = request_dict.get('parent_job_id', None)
    if parent_job_id:
        parent_job_id = str(parent_job_id)
    action = request_dict.get('action', None)
    if not action:
        metadata = {
            "error_desc": ("Missing required field 'action'. Valid actions include: train, evaluate, "
                           "export, inference, prune, retrain, etc."),
            "error_code": 3
        }
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    specs = request_dict.get('specs', {})
    name = request_dict.get('name', '')
    description = request_dict.get('description', '')
    num_gpu = request_dict.get('num_gpu', -1)
    backend_details = request_dict.get('backend_details', None)
    retain_checkpoints_for_resume = request_dict.get('retain_checkpoints_for_resume', None)
    early_stop_epoch = request_dict.get('early_stop_epoch', None)
    timeout_minutes = request_dict.get('timeout_minutes', 60)
    if isinstance(specs, dict) and "cluster" in specs:
        metadata = {
            "error_desc": ("'cluster' is not allowed in specs. "
                           "Remove the 'cluster' field from your request."),
            "error_code": 4
        }
        schema = ErrorRsp()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)
    experiment_id = None
    if kind == 'experiment':
        # Validate required fields for experiment creation
        network_arch = request_dict.get('network_arch')
        base_experiment_ids = request_dict.get('base_experiment_ids', [])
        if not network_arch and not base_experiment_ids:
            metadata = {
                "error_desc": (
                    "Missing required field: either 'network_arch' or 'base_experiment_ids' must be provided. "
                    "Specify 'network_arch' (e.g., 'vila', 'classification_pyt') for a new experiment, "
                    "or 'base_experiment_ids' to use pretrained models."
                ),
                "error_code": 5
            }
            schema = ErrorRsp()
            schema_dict = schema.dump(schema.load(metadata))
            return make_response(jsonify(schema_dict), 400)

        # Validate dataset paths if using direct paths (new approach)
        train_dataset_uris = request_dict.get("train_dataset_uris")
        eval_dataset_uri = request_dict.get("eval_dataset_uri")
        inference_dataset_uri = request_dict.get("inference_dataset_uri")
        calibration_dataset_uri = request_dict.get("calibration_dataset_uri")

        if any([train_dataset_uris, eval_dataset_uri, inference_dataset_uri, calibration_dataset_uri]):
            # Get backend type for validation
            backend_type = backend_details.get("backend_type") if backend_details else None

            # Validate all dataset paths against backend restrictions
            from nvidia_tao_core.microservices.utils.dataset_uri_validator import validate_all_dataset_uris
            validation_metadata = {
                "train_dataset_uris": train_dataset_uris,
                "eval_dataset_uri": eval_dataset_uri,
                "inference_dataset_uri": inference_dataset_uri,
                "calibration_dataset_uri": calibration_dataset_uri
            }
            is_valid, error_msg = validate_all_dataset_uris(validation_metadata, backend_type)
            if not is_valid:
                metadata = {"error_desc": error_msg, "error_code": 100}
                schema = ErrorRsp()
                schema_dict = schema.dump(schema.load(metadata))
                return make_response(jsonify(schema_dict), 400)

            # Verify workspace access if specified (required for cloud paths)
            workspace_id = request_dict.get("workspace")
            if workspace_id:
                from nvidia_tao_core.microservices.utils.stateless_handler_utils import check_read_access
                if not check_read_access(user_id, org_name, workspace_id, kind="workspaces"):
                    metadata = {
                        "error_desc": f"Workspace {workspace_id} not found or access denied",
                        "error_code": 101
                    }
                    schema = ErrorRsp()
                    schema_dict = schema.dump(schema.load(metadata))
                    return make_response(jsonify(schema_dict), 404)

            # Dataset structure validation (checks for required files like annotations.json)
            skip_validation = request_dict.get("skip_dataset_validation", False)
            logger.info(
                f"Dataset structure validation: skip={skip_validation}, "
                f"has_train={bool(train_dataset_uris)}, has_eval={bool(eval_dataset_uri)}"
            )

            if not skip_validation:
                from nvidia_tao_core.microservices.utils.runtime_dataset_validator import (
                    validate_all_dataset_uris_structure
                )

                network_arch = request_dict.get("network_arch")
                logger.info(f"Running dataset structure validation with network_arch={network_arch}")
                if not network_arch:
                    metadata = {
                        "error_desc": "network_arch is required for dataset validation",
                        "error_code": 102
                    }
                    schema = ErrorRsp()
                    schema_dict = schema.dump(schema.load(metadata))
                    return make_response(jsonify(schema_dict), 400)

                # Prepare metadata for validation
                validation_metadata = {
                    "train_dataset_uris": train_dataset_uris,
                    "eval_dataset_uri": eval_dataset_uri,
                    "inference_dataset_uri": inference_dataset_uri,
                    "calibration_dataset_uri": calibration_dataset_uri,
                    "dataset_format": request_dict.get("dataset_format"),
                    "dataset_type": request_dict.get("dataset_type"),
                    "workspace": request_dict.get("workspace")
                }

                is_valid, error_msg, validation_details = validate_all_dataset_uris_structure(
                    validation_metadata,
                    network_arch,
                    skip_validation=False
                )

                if not is_valid:
                    # Return detailed validation error
                    metadata = {
                        "error_desc": error_msg,
                        "error_code": 103,
                        "validation_details": validation_details
                    }
                    schema = ErrorRsp()
                    schema_dict = schema.dump(schema.load(metadata))
                    return make_response(jsonify(schema_dict), 400)

                logger.info(
                    f"Dataset validation passed for experiment: "
                    f"{validation_details.get('message')}"
                )

        experiment_response = ExperimentHandler.create_experiment(user_id, org_name, request_dict)
        if experiment_response.code != 200:
            schema = ErrorRsp()
            schema_dict = schema.dump(schema.load(experiment_response.data))
            log_api_error(user_id, org_name, schema_dict, DataMonitorLogTypeEnum.tao_experiment, action="creation")
            return make_response(jsonify(schema_dict), experiment_response.code)
        experiment_id = experiment_response.data.get("id")

        # Note: Job deduplication happens later in job_run() based on the experiment's existing jobs
    # Get automl_settings if present (only for experiment jobs)
    automl_settings = request_dict.get('automl_settings') if kind == 'experiment' else None

    # Get force_create parameter
    force_create = request_dict.get('force_create', False)

    # Get job response
    job_response = JobHandler.job_run(
        org_name,
        experiment_id if kind == 'experiment' else dataset_id,
        parent_job_id,
        action,
        kind,
        specs=specs, name=name, description=description, num_gpu=num_gpu,
        backend_details=backend_details,
        job_id=experiment_id if kind == 'experiment' else None,
        retain_checkpoints_for_resume=retain_checkpoints_for_resume,
        early_stop_epoch=early_stop_epoch,
        timeout_minutes=timeout_minutes,
        automl_settings=automl_settings,
        force_create=force_create
    )
    if job_response.code != 200:
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(job_response.data))
        log_api_error(user_id, org_name, schema_dict, DataMonitorLogTypeEnum.tao_experiment, action="creation")
        return make_response(jsonify(schema_dict), job_response.code)
    job_id = job_response.data
    # Validate job ID
    if not isinstance(job_response.data, str):
        metadata = {
            "error_desc": (
                f"Internal error: Job creation returned unexpected type '{type(job_id).__name__}' "
                f"instead of job ID string. Response: {job_id}"
            ),
            "error_code": 5
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        log_api_error(user_id, org_name, schema_dict, DataMonitorLogTypeEnum.tao_experiment, action="creation")
        return make_response(jsonify(schema_dict), 500)

    # validate_uuid returns a message if invalid, None if valid
    validation_error = validate_uuid(job_id=job_id)
    if validation_error:
        metadata = {
            "error_desc": (
                f"Internal error: Job creation returned invalid job ID: '{job_id}'. "
                f"Validation error: {validation_error}"
            ),
            "error_code": 5
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        log_api_error(user_id, org_name, schema_dict, DataMonitorLogTypeEnum.tao_experiment, action="creation")
        return make_response(jsonify(schema_dict), 500)
    job_rsp = JobHandler.job_retrieve(org_name, experiment_id if kind == 'experiment' else dataset_id, job_id, kind)
    job = job_rsp.data
    if kind == 'experiment':
        schema = ExperimentJobRsp()
        exp = experiment_response.data
        experiment = {key: exp[key] for key in exp if key not in ["jobs"]}
        combined_data = experiment | job
    else:
        schema = DatasetJobRsp()
        dataset = get_dataset(dataset_id)
        tags = dataset.get('tags', [])
        combined_data = {"tags": tags} | job
    schema_dict = schema.dump(schema.load(combined_data))

    # Determine appropriate status code:
    # - 201 for newly created job
    # - 200 for existing job returned due to deduplication
    if "already exists" in job_response.message:
        status_code = 200
        # Add informational message to response
        schema_dict['_message'] = (
            "A job with the same parameters already exists. "
            "Returning existing job. "
            "To create a new job anyway, set 'force_create': true in the request body."
        )
        schema_dict['_duplicate'] = True
    else:
        status_code = 201

    return make_response(jsonify(schema_dict), status_code)


@jobs_bp_v2.route('/orgs/<org_name>/jobs/<job_id>', methods=['GET'])
@disk_space_check
def job_retrieve(org_name, job_id):
    """Retrieve Job.

    ---
    get:
      tags:
      - JOB
      summary: Retrieve Job
      description: Returns the Job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: job_id
        in: path
        description: ID of Job to return
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned the Job
          content:
            application/json:
              schema: JobRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    handler_id, kind = get_handler_id_and_kind(job_id)
    if kind not in ['experiment', 'dataset']:
        metadata = {
            "error_desc": (f"Invalid job type '{kind}' for job_id {job_id}. "
                           "Job must be associated with an experiment or dataset."),
            "error_code": 1
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    experiment_id = None
    dataset_id = None
    if kind == 'experiment':
        experiment_id = handler_id
    else:
        dataset_id = handler_id
    if kind == 'experiment':
        # Get experiment response
        experiment_response = ExperimentHandler.retrieve_experiment(org_name, experiment_id)
        if experiment_response.code != 200:
            schema = ErrorRsp()
            schema_dict = schema.dump(schema.load(experiment_response.data))
            return make_response(jsonify(schema_dict), experiment_response.code)
    return_specs = ast.literal_eval(request.args.get('return_specs', "False"))
    job_response = JobHandler.job_retrieve(
        org_name,
        experiment_id if kind == 'experiment' else dataset_id,
        job_id,
        kind,
        return_specs=return_specs
    )
    if job_response.code != 200:
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(job_response.data))
        return make_response(jsonify(schema_dict), job_response.code)
    job = job_response.data

    # Truncate detailed_status message if it exceeds the schema limit (6400 chars)
    max_message_length = 6400

    # Handle top-level detailed_status
    if 'detailed_status' in job and isinstance(job['detailed_status'], dict):
        if 'message' in job['detailed_status'] and isinstance(job['detailed_status']['message'], str):
            msg_len = len(job['detailed_status']['message'])
            if msg_len > max_message_length:
                logger.warning(
                    f"Truncating top-level detailed_status.message for job_id {job_id} "
                    f"from {msg_len} to {max_message_length} characters."
                )
                logger.warning(f"Message preview (first 500 chars): {job['detailed_status']['message'][:500]}")
                logger.warning(
                    f"Message preview (last 200 chars): ...{job['detailed_status']['message'][-200:]}"
                )
                job['detailed_status']['message'] = job['detailed_status']['message'][:max_message_length - 3] + "..."

    # Handle nested job_details (AutoML jobs with sub-jobs)
    if 'job_details' in job and isinstance(job['job_details'], dict):
        logger.info(f"Job retrieve - job_details has {len(job['job_details'])} sub-jobs for job_id {job_id}")
        for sub_job_id, sub_job_data in job['job_details'].items():
            if not isinstance(sub_job_data, dict):
                continue

            if 'value' in sub_job_data:
                target_dict = sub_job_data['value']
                path_prefix = f"job_details[{sub_job_id}]['value']"
            else:
                target_dict = sub_job_data
                path_prefix = f"job_details[{sub_job_id}]"

            if isinstance(target_dict, dict) and 'detailed_status' in target_dict:
                detailed_status = target_dict['detailed_status']
                if isinstance(detailed_status, dict) and 'message' in detailed_status:
                    if isinstance(detailed_status['message'], str):
                        msg_len = len(detailed_status['message'])
                        logger.info(
                            f"Job retrieve - Sub-job {sub_job_id} "
                            f"detailed_status.message length: {msg_len} (path: {path_prefix})"
                        )
                        if msg_len > max_message_length:
                            logger.warning(
                                f"TRUNCATING {path_prefix}.detailed_status.message for parent job_id {job_id} "
                                f"from {msg_len} to {max_message_length} chars"
                            )
                            logger.warning(
                                f"Problematic message preview (first 500 chars): {detailed_status['message'][:500]}"
                            )
                            logger.warning(
                                f"Problematic message preview (last 200 chars): "
                                f"...{detailed_status['message'][-200:]}"
                            )
                            # Truncate in place in the actual location
                            target_dict['detailed_status']['message'] = (
                                detailed_status['message'][:max_message_length - 3] + "..."
                            )
                            truncated_len = len(target_dict['detailed_status']['message'])
                            logger.warning(f"After truncation, message length is now: {truncated_len}")

    if kind == 'experiment':
        schema = ExperimentJobRsp()
        exp = experiment_response.data
        experiment = {key: exp[key] for key in exp if key not in ["jobs"]}
        combined_data = experiment | job
    else:
        schema = DatasetJobRsp()
        dataset = get_dataset(dataset_id)
        tags = dataset.get('tags', [])
        combined_data = {"tags": tags} | job
    try:
        schema_dict = schema.dump(schema.load(combined_data))
        logger.info(f"Job retrieve success for job_id: {job_id}")
        return make_response(jsonify(schema_dict), 200)
    except Exception as e:
        logger.error(f"Error type: {type(e).__name__}")
        metadata = {"error_desc": str(e), "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)


@jobs_bp_v2.route('/orgs/<org_name>/jobs/<job_id>', methods=['DELETE'])
@disk_space_check
def job_delete(org_name, job_id):
    """Delete Job.

    ---
    delete:
      tags:
      - JOB
      summary: Delete Job
      description: Cancels Job if running and returns the deleted Job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: job_id
        in: path
        description: ID of Job to delete
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned the deleted Job
          content:
            application/json:
              schema: JobRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    handler_id, kind = get_handler_id_and_kind(job_id)
    if kind not in ['experiment', 'dataset']:
        metadata = {
            "error_desc": (f"Invalid job type for job_id {job_id}. "
                           "Job must be associated with an experiment or dataset."),
            "error_code": 1
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    experiment_id = None
    dataset_id = None
    if kind == 'experiment':
        experiment_id = handler_id
    else:
        dataset_id = handler_id
    # Get job_response
    job_response = JobHandler.job_delete(
        org_name,
        experiment_id if kind == 'experiment' else dataset_id,
        job_id,
        kind)
    if job_response.code != 200:
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(job_response.data))
        return make_response(jsonify(schema_dict), job_response.code)
    job = get_job(job_id)
    if kind == 'experiment':
        exp = get_experiment(experiment_id)
        # Delete experiment if no more jobs
        if not exp.get('jobs', {}):
            # Get experiment_response
            experiment_response = ExperimentHandler.delete_experiment(org_name, experiment_id)
            if experiment_response.code != 200:
                schema = ErrorRsp()
                schema_dict = schema.dump(schema.load(experiment_response.data))
                return make_response(jsonify(schema_dict), experiment_response.code)
        schema = ExperimentJobRsp()
        experiment = {key: exp[key] for key in exp if key not in ["jobs"]}
        combined_data = experiment | job
    else:
        schema = DatasetJobRsp()
        dataset = get_dataset(dataset_id)
        tags = dataset.get('tags', [])
        combined_data = {"tags": tags} | job
    schema_dict = schema.dump(schema.load(combined_data))
    return make_response(jsonify(schema_dict), 200)


@jobs_bp_v2.route('/orgs/<org_name>/jobs', methods=['GET'])
@disk_space_check
def job_list(org_name):
    """List Jobs.

    ---
    get:
      tags:
      - JOB
      summary: List Jobs
      description: Returns the list of Jobs
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: skip
        in: query
        description: Optional skip for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: size
        in: query
        description: Optional size for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: sort
        in: query
        description: Optional sort
        required: false
        schema:
          type: string
          enum: ["date-descending", "date-ascending", "name-descending", "name-ascending" ]
      - name: name
        in: query
        description: Optional name filter
        required: false
        schema:
          type: string
          maxLength: 5000
          pattern: '.*'
      - name: network_arch
        in: query
        description: Optional network architecture filter
        required: false
        schema:
          type: string
          enum: [
              "action_recognition",
              "classification_pyt",
              "mal",
              "ml_recog",
              "ocdnet",
              "ocrnet",
              "optical_inspection",
              "pointpillars",
              "pose_classification",
              "re_identification",
              "deformable_detr",
              "dino",
              "segformer",
              "visual_changenet_classify",
              "visual_changenet_segment",
              "centerpose"
          ]
      - name: read_only
        in: query
        description: Optional read_only filter
        required: false
        allowEmptyValue: true
        schema:
          type: boolean
      - name: user_only
        in: query
        description: Optional filter to select user owned experiments only
        required: false
        schema:
          type: boolean
      - name: tag
        in: query
        description: Optional tag filter
        required: false
        schema:
          type: string
          maxLength: 5000
          pattern: '.*'
      responses:
        200:
          description: Returned the list of Jobs
          content:
            application/json:
              schema: JobListRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    user_only = str(request.args.get('user_only', None)) in {'True', 'yes', 'y', 'true', 't', '1', 'on'}
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    jobs = []

    # Process experiment jobs
    exp_schema = ExperimentJobRsp()
    experiments = ExperimentHandler.list_experiments(user_id, org_name, user_only)
    for exp in experiments:
        experiment_id = exp.get('id')
        response = JobHandler.job_list(user_id, org_name, experiment_id, "experiment")
        if response.code == 200 and isinstance(response.data, list):
            for job in response.data:
                experiment = {key: exp[key] for key in exp if key not in ["jobs"]}
                combined_data = experiment | job
                # Serialize with schema validation
                jobs.append(dict(exp_schema.dump(exp_schema.load(combined_data))))

    # Process dataset jobs
    ds_schema = DatasetJobRsp()
    datasets = DatasetHandler.list_datasets(user_id, org_name)
    for dataset in datasets:
        dataset_id = dataset.get('id')
        tags = dataset.get('tags', [])
        response = JobHandler.job_list(user_id, org_name, dataset_id, "dataset")
        if response.code == 200 and isinstance(response.data, list):
            for job in response.data:
                combined_data = {"tags": tags} | job
                # Serialize with schema validation
                jobs.append(dict(ds_schema.dump(ds_schema.load(combined_data))))
    filtered_jobs = filtering.apply(request.args, jobs)
    paginated_jobs = pagination.apply(request.args, filtered_jobs)
    metadata = {"jobs": paginated_jobs}
    # Pagination
    skip = request.args.get("skip", None)
    size = request.args.get("size", None)
    if skip is not None and size is not None:
        skip = int(skip)
        size = int(size)
        metadata["pagination_info"] = {
            "total_records": len(filtered_jobs),
            "total_pages": math.ceil(len(filtered_jobs) / size),
            "page_size": size,
            "page_index": skip // size,
        }
    # Validate overall structure with JobListRsp schema
    schema = JobListRsp()
    try:
        schema.load(metadata)  # Validate structure
    except Exception as e:
        # Extract detailed validation errors
        if isinstance(e, ValidationError):
            # Marshmallow ValidationError contains field-specific errors
            error_details = e.messages
            error_message = f"Job list response validation failed. Errors: {error_details}"
        else:
            error_message = f"Job list response validation failed: {str(e)}"

        metadata = {"error_desc": error_message, "error_code": 1}
        error_schema = ErrorRsp()
        schema_dict = error_schema.dump(error_schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    # Return the pre-serialized metadata (OneOfSchema dump() doesn't work with pre-serialized dicts)
    response = make_response(jsonify(metadata))
    return response


@jobs_bp_v2.route('/orgs/<org_name>/jobs:load_airgapped', methods=['POST'])
@disk_space_check
def base_experiment_load_airgapped(org_name):
    """Load Airgapped base experiments.

    ---
    post:
      tags:
      - JOB
      summary: Load pretrained models from airgapped cloud storage
      description: Loads pretrained models metadata from airgapped cloud storage using workspace credentials
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      requestBody:
        required: true
        content:
          application/json:
            schema: LoadAirgappedExperimentsReq
      responses:
        200:
          description: Successfully loaded airgapped pretrained models
          content:
            application/json:
              schema: LoadAirgappedExperimentsRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request
          content:
            application/json:
              schema: ErrorRsp
        404:
          description: Workspace not found
          content:
            application/json:
              schema: ErrorRsp
        500:
          description: Internal server error
          content:
            application/json:
              schema: ErrorRsp
    """
    schema = LoadAirgappedExperimentsReq()
    request_data = request.get_json(force=True)
    request_dict = schema.dump(schema.load(request_data))
    workspace_id = request_dict['workspace_id']
    message = validate_uuid(workspace_id=workspace_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = ExperimentHandler.load_airgapped_experiments(
        user_id,
        org_name,
        workspace_id
    )
    schema = LoadAirgappedExperimentsRsp() if response.code == 200 else ErrorRsp()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@jobs_bp_v2.route('/orgs/<org_name>/jobs:list_base_experiments', methods=['GET'])
@disk_space_check
def base_experiments_list(org_name):
    """List Base Experiments.

    ---
    get:
      tags:
      - JOB
      summary: List Pretrained Models that can be used for transfer learning
      description: Returns the list of models published in NGC public catalog and private org's model registry
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: skip
        in: query
        description: Optional skip for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: size
        in: query
        description: Optional size for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: sort
        in: query
        description: Optional sort
        required: false
        schema:
          type: string
          enum: ["date-descending", "date-ascending", "name-descending", "name-ascending" ]
      - name: name
        in: query
        description: Optional name filter
        required: false
        schema:
          type: string
          maxLength: 5000
          pattern: '.*'
      - name: network_arch
        in: query
        description: Optional network architecture filter
        required: false
        schema:
          type: string
          enum: [
              "action_recognition",
              "classification_pyt",
              "mal",
              "ml_recog",
              "ocdnet",
              "ocrnet",
              "optical_inspection",
              "pointpillars",
              "pose_classification",
              "re_identification",
              "deformable_detr",
              "dino",
              "segformer",
              "visual_changenet_classify",
              "visual_changenet_segment",
              "centerpose"
          ]
      - name: read_only
        in: query
        description: Optional read_only filter
        required: false
        allowEmptyValue: true
        schema:
          type: boolean
      responses:
        200:
          description: Returned the list of Pretrained Models
          content:
            application/json:
              schema: JobListRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    experiments = ExperimentHandler.list_base_experiments(user_id, org_name)
    jobs = []
    # Fields that are not part of ExperimentJobRsp schema and should be removed
    fields_to_remove = ['_id', 'type']
    exp_schema = ExperimentJobRsp()
    for job in experiments:
        # Remove fields that aren't in the schema
        for field in fields_to_remove:
            job.pop(field, None)
        # Serialize each job individually with schema validation
        serialized_job = dict(exp_schema.dump(exp_schema.load(job)))
        # Ensure kind field is present for polymorphic JobRsp routing
        serialized_job['kind'] = 'experiment'
        jobs.append(serialized_job)
    filtered_jobs = filtering.apply(request.args, jobs)
    paginated_jobs = pagination.apply(request.args, filtered_jobs)
    metadata = {"experiments": paginated_jobs}
    # Pagination
    skip = request.args.get("skip", None)
    size = request.args.get("size", None)
    if skip is not None and size is not None:
        skip = int(skip)
        size = int(size)
        metadata["pagination_info"] = {
            "total_records": len(filtered_jobs),
            "total_pages": math.ceil(len(filtered_jobs) / size),
            "page_size": size,
            "page_index": skip // size,
        }
    # Validate overall structure with JobListRsp schema
    schema = JobListRsp()
    try:
        schema.load(metadata)  # Validate structure
    except Exception as e:
        # Extract detailed validation errors
        if isinstance(e, ValidationError):
            # Marshmallow ValidationError contains field-specific errors
            error_details = e.messages
            error_message = f"Base experiments list response validation failed. Errors: {error_details}"
        else:
            error_message = f"Base experiments list response validation failed: {str(e)}"

        logger.error(f"Base experiments list response validation failed: {error_message}")
        metadata = {"error_desc": error_message, "error_code": 1}
        error_schema = ErrorRsp()
        schema_dict = error_schema.dump(error_schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    # Return the pre-serialized metadata (OneOfSchema dump() doesn't work with pre-serialized dicts)
    response = make_response(jsonify(metadata))
    return response


@jobs_bp_v2.route('/orgs/<org_name>/jobs/<job_id>', methods=['PATCH'])
@disk_space_check
def job_partial_update(org_name, job_id):
    """Partial update Job.

    ---
    patch:
      tags:
      - JOB
      summary: Partial update Job
      description: Partially updates an existing job's tags
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: job_id
        in: path
        description: ID of Job to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: JobReq
        description: Updated tags for Job
        required: true
      responses:
        200:
          description: Returned the updated Job
          content:
            application/json:
              schema: JobRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid job ID, missing required fields)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    handler_id, kind = get_handler_id_and_kind(job_id)
    if kind not in ['experiment', 'dataset']:
        metadata = {
            "error_desc": (f"Invalid job type for job_id {job_id}. "
                           "Job must be associated with an experiment or dataset."),
            "error_code": 1
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    experiment_id = None
    dataset_id = None
    if kind == 'experiment':
        experiment_id = handler_id
        schema = ExperimentJobReq()
        request_dict = schema.dump(schema.load(request.get_json(force=True)))
        # Get response
        response = ExperimentHandler.update_experiment(org_name, experiment_id, request_dict)
        if response.code != 200:
            schema = ErrorRsp()
            schema_dict = schema.dump(schema.load(response.data))
            return make_response(jsonify(schema_dict), response.code)
        schema = ExperimentJobRsp()
        job = get_job(job_id)
        if job.get("num_gpu") and job.get("num_gpu") == -1:
            job["num_gpu"] = 1
        exp = response.data
        experiment = {key: exp[key] for key in exp if key not in ["jobs"]}
        combined_data = experiment | job
    else:
        dataset_id = handler_id
        message = validate_uuid(dataset_id=dataset_id)
        if message:
            metadata = {"error_desc": message, "error_code": 1}
            schema = ErrorRsp()
            schema_dict = schema.dump(schema.load(metadata))
            return make_response(jsonify(schema_dict), 400)
        schema = DatasetJobReq()
        request_dict = schema.dump(schema.load(request.get_json(force=True)))
        # Only tags can be updated
        request_dict = {key: request_dict[key] for key in request_dict if key in ["tags"]}
        # Get response
        response = DatasetHandler.update_dataset(org_name, dataset_id, request_dict)
        if response.code != 200:
            schema = ErrorRsp()
            schema_dict = schema.dump(schema.load(response.data))
            return make_response(jsonify(schema_dict), response.code)
        schema = DatasetJobRsp()
        job = get_job(job_id)
        dataset = response.data
        tags = dataset.get('tags', [])
        combined_data = {"tags": tags} | job
    schema_dict = schema.dump(schema.load(combined_data))
    return make_response(jsonify(schema_dict), 200)


@jobs_bp_v2.route('/orgs/<org_name>/jobs:schema', methods=['GET'])
@disk_space_check
def specs_schema(org_name):
    """Retrieve Specs schema.

    ---
    get:
      tags:
      - JOB
      summary: Retrieve Specs schema
      description: Returns the matching Specs schema
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: network_arch
        in: query
        description: Optional network architecture
        required: false
        schema:
          type: string
          enum: [
              "action_recognition",
              "classification_pyt",
              "mal",
              "ml_recog",
              "ocdnet",
              "ocrnet",
              "optical_inspection",
              "pointpillars",
              "pose_classification",
              "re_identification",
              "deformable_detr",
              "dino",
              "segformer",
              "visual_changenet_classify",
              "visual_changenet_segment",
              "centerpose"
          ]
      - name: action
        in: query
        description: Optional Action name
        required: false
        schema:
          type: string
          enum: [
            "dataset_convert", "convert", "kmeans", "augment", "train",
            "evaluate", "prune", "retrain", "export", "gen_trt_engine", "trtexec", "inference",
            "annotation", "analyze", "validate", "auto_label", "calibration_tensorfile"
          ]
      - name: format
        in: query
        description: Optional dataset format
        required: false
        schema:
          type: string
          enum: [
              "kitti", "pascal_voc", "raw", "coco_raw", "unet", "coco", "lprnet", "train", "test",
              "default", "custom", "classification_pyt", "visual_changenet_segment",
              "visual_changenet_classify"
          ]
      - name: datasets
        in: query
        description: Optional datasets
        required: false
        schema:
            type: array
            items:
                type: string
                format: uuid
                maxLength: 36
      - name: job_id
        in: query
        description: Optional job id
        required: false
        schema:
            type: string
            format: uuid
            maxLength: 36
      - name: base_experiment_id
        in: query
        description: Optional pretrained-model id
        required: false
        schema:
            type: string
            format: uuid
            maxLength: 36
      responses:
        200:
          description: Returned the matching Spec schemas
          content:
            application/json:
              schema:
                type: object
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. Invalid base_experiment_id, job_id, or missing required field)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get response
    network = request.args.get('network_arch')
    action = request.args.get('action')
    dataset_format = request.args.get('format')
    datasets = request.args.getlist('datasets')
    job_id = request.args.get('job_id')
    base_experiment_id = request.args.get('base_experiment_id')
    response = None
    if base_experiment_id:
        message = validate_uuid(experiment_id=base_experiment_id)
        if message:
            metadata = {"error_desc": message, "error_code": 1}
            schema = ErrorRsp()
            schema_dict = schema.dump(schema.load(metadata))
            return make_response(jsonify(schema_dict), 400)
        response = SpecHandler.get_base_experiment_spec_schema(base_experiment_id, action)
    elif job_id:
        message = validate_uuid(job_id=job_id)
        if message:
            metadata = {"error_desc": message, "error_code": 1}
            schema = ErrorRsp()
            schema_dict = schema.dump(schema.load(metadata))
            return make_response(jsonify(schema_dict), 400)
        user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
        handler_id, kind = get_handler_id_and_kind(job_id)
        if kind not in ['experiment', 'dataset']:
            metadata = {
                "error_desc": (f"Invalid job: {job_id} is not an experiment job. "
                               "This operation is only supported for experiment jobs."),
                "error_code": 1
            }
            schema = ErrorRsp()
            schema_dict = schema.dump(schema.load(metadata))
            return make_response(jsonify(schema_dict), 400)
        experiment_id = None
        dataset_id = None
        if kind == 'experiment':
            experiment_id = handler_id
        else:
            dataset_id = handler_id
        response = SpecHandler.get_spec_schema_for_job(
            user_id,
            org_name,
            experiment_id if kind == 'experiment' else dataset_id,
            job_id,
            kind
        )
    elif network:
        response = SpecHandler.get_spec_schema_without_handler_id(
            org_name, network, dataset_format, action, datasets)
    else:
        metadata = {"error_desc": "missing network, base_experiment_id or job_id", "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    if response.code != 200:
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(response.data))
        return make_response(jsonify(schema_dict), response.code)
    return make_response(jsonify(response.data), response.code)


@jobs_bp_v2.route('/orgs/<org_name>/jobs/<job_id>:retry', methods=['POST'])
@disk_space_check
def job_retry(org_name, job_id):
    """Retry Job.

    ---
    post:
      tags:
      - JOB
      summary: Retry Job
      description: Asynchronously retries an Action and returns its new Job ID
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        201:
          description: Returned the new Job ID corresponding to requested Action
          content:
            application/json:
              schema:
                type: string
                format: uuid
                maxLength: 36
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User of Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    handler_id, kind = get_handler_id_and_kind(job_id)
    if kind not in ['experiment', 'dataset']:
        metadata = {
            "error_desc": (f"Invalid job type for job_id {job_id}. "
                           "Job must be associated with an experiment or dataset."),
            "error_code": 1
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    experiment_id = None
    dataset_id = None
    if kind == 'experiment':
        experiment_id = handler_id
    else:
        dataset_id = handler_id
    response = JobHandler.job_retry(
        org_name,
        experiment_id if kind == 'experiment' else dataset_id,
        kind,
        job_id
    )
    if response.code != 200:
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(response.data))
        return make_response(jsonify(schema_dict), response.code)
    job_id = response.data
    if not isinstance(response.data, str):
        metadata = {
            "error_desc": f"Internal error: Job retry returned unexpected type '{type(job_id).__name__}'",
            "error_code": 5
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 500)

    validation_error = validate_uuid(job_id=job_id)
    if validation_error:
        metadata = {
            "error_desc": f"Internal error: Job retry returned invalid job ID: '{job_id}'. {validation_error}",
            "error_code": 5
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 500)
    job = get_job(job_id)
    if kind == 'experiment':
        schema = ExperimentJobRsp()
        exp = get_experiment(experiment_id)
        experiment = {key: exp[key] for key in exp if key not in ["jobs"]}
        combined_data = experiment | job
    else:
        schema = DatasetJobRsp()
        dataset = get_dataset(dataset_id)
        tags = dataset.get('tags', [])
        combined_data = {"tags": tags} | job
    schema_dict = schema.dump(schema.load(combined_data))
    return make_response(jsonify(schema_dict), 201)


@jobs_bp_v2.route('/orgs/<org_name>/jobs/<job_id>:publish_model', methods=['POST'])
@disk_space_check
def model_publish(org_name, job_id):
    """Publish models to NGC.

    ---
    post:
      tags:
      - JOB
      summary: Publish models to NGC
      description: Publishes models to NGC private registry
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: PublishModel
      responses:
        200:
          description: String message for successful upload
          content:
            application/json:
              schema:
                type: string
                format: uuid
                maxLength: 36
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid job ID or missing required fields)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    handler_id, kind = get_handler_id_and_kind(job_id)
    if kind not in ['experiment']:
        metadata = {
            "error_desc": (f"Invalid job: {job_id} is not an experiment job. "
                           "This operation is only supported for experiment jobs."),
            "error_code": 1
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    experiment_id = handler_id
    if os.getenv("AIRGAPPED_MODE", "false").lower() == "true":
        schema = MessageOnly()
        message = "Cannot publish model in air-gapped mode."
        schema_dict = schema.dump(schema.load({"message": message}))
        return make_response(jsonify(schema_dict), 400)
    request_data = request.get_json(force=True).copy()
    schema = PublishModel()
    request_schema_data = schema.dump(schema.load(request_data))
    display_name = request_schema_data.get('display_name', '')
    description = request_schema_data.get('description', '')
    team_name = request_schema_data.get('team_name', '')
    response = ModelHandler.publish_model(
        org_name,
        team_name,
        experiment_id,
        job_id,
        display_name=display_name,
        description=description
    )
    schema = MessageOnly() if response.code == 200 else ErrorRsp()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@jobs_bp_v2.route('/orgs/<org_name>/jobs/<job_id>:remove_published_model', methods=['DELETE'])
@disk_space_check
def remove_published_model(org_name, job_id):
    """Remove published models from NGC.

    ---
    delete:
      tags:
      - JOB
      summary: Remove publish models from NGC
      description: Removes models from NGC private registry
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: PublishModel
      responses:
        200:
          description: String message for successfull deletion
          content:
            application/json:
              schema:
                type: string
                format: uuid
                maxLength: 36
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid job ID, missing required fields)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    handler_id, kind = get_handler_id_and_kind(job_id)
    if kind not in ['experiment']:
        metadata = {
            "error_desc": (f"Invalid job: {job_id} is not an experiment job. "
                           "This operation is only supported for experiment jobs."),
            "error_code": 1
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    experiment_id = handler_id
    if os.getenv("AIRGAPPED_MODE", "false").lower() == "true":
        schema = MessageOnly()
        message = "Cannot remove published model in air-gapped mode."
        schema_dict = schema.dump({"message": message})
        return make_response(jsonify(schema_dict), 400)
    request_data = request.args.to_dict()
    schema = PublishModel()
    request_schema_data = schema.dump(schema.load(request_data))
    team_name = request_schema_data.get('team_name', '')
    response = ModelHandler.remove_published_model(org_name, team_name, experiment_id, job_id)
    schema = MessageOnly() if response.code == 200 else ErrorRsp()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@jobs_bp_v2.route('/orgs/<org_name>/jobs/<job_id>:pause', methods=['POST'])
@disk_space_check
def job_pause(org_name, job_id):  # noqa: D214
    """Pause Job (only for training).

    ---
    post:
      tags:
      - JOB
      summary: Pause Job - only for training
      description: Pauses a specific training job.
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully requested training pause of specified Job ID (asynchronous)
          content:
            application/json:
              schema: MessageOnly
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    handler_id, kind = get_handler_id_and_kind(job_id)
    if kind not in ['experiment']:
        metadata = {
            "error_desc": (f"Invalid job: {job_id} is not an experiment job. "
                           "This operation is only supported for experiment jobs."),
            "error_code": 1
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    experiment_id = handler_id
    request_data = request.get_json(silent=True) or {}
    graceful = request_data.get("graceful", False)
    response = JobHandler.job_pause(org_name, experiment_id, job_id, "experiment", graceful=graceful)
    schema = MessageOnly() if response.code == 200 else ErrorRsp()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@jobs_bp_v2.route('/orgs/<org_name>/jobs/<job_id>:cancel', methods=['POST'])
@disk_space_check
def job_cancel(org_name, job_id):
    """Cancel Job (or pause training).

    ---
    post:
      tags:
      - JOB
      summary: Cancel Job or pause training
      description: Cancels a specific job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully requested cancelation or training pause of specified Job ID
          content:
            application/json:
              schema: MessageOnly
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    handler_id, kind = get_handler_id_and_kind(job_id)
    if kind not in ['experiment', 'dataset']:
        metadata = {
            "error_desc": (f"Invalid job type for job_id {job_id}. "
                           "Job must be associated with an experiment or dataset."),
            "error_code": 1
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    experiment_id = None
    dataset_id = None
    if kind == 'experiment':
        experiment_id = handler_id
    else:
        dataset_id = handler_id
    response = JobHandler.job_cancel(
        org_name,
        experiment_id if kind == 'experiment' else dataset_id,
        job_id,
        kind)
    schema = MessageOnly() if response.code == 200 else ErrorRsp()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@jobs_bp_v2.route('/orgs/<org_name>/jobs/<job_id>:resume', methods=['POST'])
@disk_space_check
def job_resume(org_name, job_id):
    """Resume Job - train/retrain only.

    ---
    post:
      tags:
      - JOB
      summary: Resume training Job
      description: Resumes a specific training job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: JobResume
        description: Adjustable metadata for the resumed job.
        required: false
      responses:
        200:
          description: Successfully requested resume of specified Job ID (asynchronous)
          content:
            application/json:
              schema: MessageOnly
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid job ID, missing required fields)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    handler_id, kind = get_handler_id_and_kind(job_id)
    if kind not in ['experiment']:
        metadata = {
            "error_desc": (f"Invalid job: {job_id} is not an experiment job. "
                           "This operation is only supported for experiment jobs."),
            "error_code": 1
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    experiment_id = handler_id
    request_data = request.get_json(force=True).copy()
    schema = JobResume()
    request_schema_data = schema.dump(schema.load(request_data))
    parent_job_id = request_schema_data.get('parent_job_id', None)
    name = request_schema_data.get('name', '')
    description = request_schema_data.get('description', '')
    num_gpu = request_schema_data.get('num_gpu', -1)
    backend_details = request_schema_data.get('backend_details', None)
    timeout_minutes = request_schema_data.get('timeout_minutes', 60)
    if parent_job_id:
        parent_job_id = str(parent_job_id)
    specs = request_schema_data.get('specs', {})
    response = ExperimentHandler.resume_experiment_job(
        org_name,
        experiment_id,
        job_id,
        "experiment",
        parent_job_id,
        specs=specs,
        name=name,
        description=description,
        num_gpu=num_gpu,
        timeout_minutes=timeout_minutes,
        backend_details=backend_details
    )
    schema = MessageOnly() if response.code == 200 else ErrorRsp()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@jobs_bp_v2.route('/orgs/<org_name>/jobs/<job_id>:list_files', methods=['GET'])
@disk_space_check
def job_files_list(org_name, job_id):
    """List Job Files.

    ---
    get:
      tags:
      - JOB
      summary: List Job File
      description: Lists the files produced by a given job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job Files
          content:
            application/json:
              schema:
                type: array
                items:
                  type: string
                  maxLength: 1000
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    handler_id, kind = get_handler_id_and_kind(job_id)
    if kind not in ['experiment', 'dataset']:
        metadata = {
            "error_desc": (f"Invalid job type for job_id {job_id}. "
                           "Job must be associated with an experiment or dataset."),
            "error_code": 1
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    experiment_id = None
    dataset_id = None
    if kind == 'experiment':
        experiment_id = handler_id
    else:
        dataset_id = handler_id
    response = JobHandler.job_list_files(
        org_name,
        experiment_id if kind == 'experiment' else dataset_id,
        job_id,
        kind
    )
    if response.code == 200:
        if isinstance(response.data, list) and (all(isinstance(f, str) for f in response.data) or response.data == []):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: file list invalid", "error_code": 2}
        schema = ErrorRsp()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 500)
    schema = ErrorRsp()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@jobs_bp_v2.route('/orgs/<org_name>/jobs/<job_id>:download', methods=['GET'])
@disk_space_check
def job_download(org_name, job_id):
    """Download Job Artifacts.

    ---
    get:
      tags:
      - JOB
      summary: Download Job Artifacts
      description: Downloads the artifacts produced by a given job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job Artifacts
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
                maxLength: 1000
                maxLength: 1000
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    handler_id, kind = get_handler_id_and_kind(job_id)
    if kind not in ['experiment', 'dataset']:
        metadata = {
            "error_desc": (f"Invalid job type for job_id {job_id}. "
                           "Job must be associated with an experiment or dataset."),
            "error_code": 1
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    experiment_id = None
    dataset_id = None
    if kind == 'experiment':
        experiment_id = handler_id
    else:
        dataset_id = handler_id
    response = JobHandler.job_download(
        org_name,
        experiment_id if kind == 'experiment' else dataset_id,
        job_id,
        kind
    )
    if response.code == 200:
        file_path = response.data  # Response is assumed to have the file path
        file_dir = "/".join(file_path.split("/")[:-1])
        file_name = file_path.split("/")[-1]  # infer the name
        return send_from_directory(file_dir, file_name, as_attachment=True)
    schema = ErrorRsp()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@jobs_bp_v2.route('/orgs/<org_name>/jobs/<job_id>:download_selective_files', methods=['GET'])
@disk_space_check
def job_download_selective_files(org_name, job_id):
    """Download selective Job Artifacts.

    ---
    get:
      tags:
      - JOB
      summary: Download selective Job Artifacts
      description: Downloads selective artifacts produced by a given job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job Artifacts
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
                maxLength: 1000
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid job ID or file list)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    handler_id, kind = get_handler_id_and_kind(job_id)
    if kind not in ['experiment', 'dataset']:
        metadata = {
            "error_desc": (f"Invalid job type for job_id {job_id}. "
                           "Job must be associated with an experiment or dataset."),
            "error_code": 1
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    experiment_id = None
    dataset_id = None
    if kind == 'experiment':
        experiment_id = handler_id
    else:
        dataset_id = handler_id
    file_lists = request.args.getlist('file_lists')
    best_model = ast.literal_eval(request.args.get('best_model', "False"))
    latest_model = ast.literal_eval(request.args.get('latest_model', "False"))
    tar_files = ast.literal_eval(request.args.get('tar_files', "True"))
    if not (file_lists or best_model or latest_model):
        return make_response(
            jsonify("No files passed in list format to download or, best_model or latest_model is not enabled"),
            400
        )
    response = JobHandler.job_download(
        org_name,
        experiment_id if kind == 'experiment' else dataset_id,
        job_id,
        kind,
        file_lists=file_lists,
        best_model=best_model,
        latest_model=latest_model,
        tar_files=tar_files
    )
    if response.code == 200:
        file_path = response.data  # Response is assumed to have the file path
        file_dir = "/".join(file_path.split("/")[:-1])
        file_name = file_path.split("/")[-1]  # infer the name
        return send_from_directory(file_dir, file_name, as_attachment=True)
    schema = ErrorRsp()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@jobs_bp_v2.route('/orgs/<org_name>/jobs/<job_id>:logs', methods=['GET'])
def job_logs(org_name, job_id):
    """Get realtime job logs. AutoML train job will return current recommendation's log.

    ---
    get:
      tags:
      - JOB
      summary: Get Job logs
      description: Returns the job logs for a given experiment and job ID
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: automl_experiment_index
        in: query
        description: Optional filter to retrieve logs from specific autoML experiment
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      responses:
        200:
          description: Returned Job Logs
          content:
            text/plain:
              example: "Execution status: PASS"
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Job not existing or logs not found.
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    handler_id, kind = get_handler_id_and_kind(job_id)
    if kind not in ['experiment', 'dataset']:
        metadata = {
            "error_desc": (f"Invalid job type for job_id {job_id}. "
                           "Job must be associated with an experiment or dataset."),
            "error_code": 1
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    experiment_id = None
    dataset_id = None
    if kind == 'experiment':
        experiment_id = handler_id
    else:
        dataset_id = handler_id
    response = JobHandler.get_job_logs(
        org_name,
        experiment_id if kind == 'experiment' else dataset_id,
        job_id,
        kind,
        request.args.get('automl_experiment_index', None)
    )
    if response.code == 200:
        response = make_response(response.data, 200)
        response.mimetype = 'text/plain'
        return response
    schema = ErrorRsp()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), 400)


@jobs_bp_v2.route('/orgs/<org_name>/jobs/<job_id>:events', methods=['GET'])
def job_events(org_name, job_id):
    """Get all job status events.

    ---
    get:
      tags:
      - JOB
      summary: Get all job status events
      description: Returns all status event lines for a given job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: automl_experiment_number
        in: query
        description: Optional filter to retrieve events from specific autoML experiment
        required: false
        schema:
          type: string
      responses:
        200:
          description: Returned Job Events
          content:
            application/json:
              schema: JobEventsRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Job not exist or events not found.
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    handler_id, kind = get_handler_id_and_kind(job_id)
    if kind not in ['experiment', 'dataset']:
        metadata = {
            "error_desc": (f"Invalid job type for job_id {job_id}. "
                           "Job must be associated with an experiment or dataset."),
            "error_code": 1
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    experiment_id = None
    dataset_id = None
    if kind == 'experiment':
        experiment_id = handler_id
    else:
        dataset_id = handler_id
    response = JobHandler.get_job_events(
        org_name,
        experiment_id if kind == 'experiment' else dataset_id,
        job_id,
        kind,
        request.args.get('automl_experiment_number', "0")
    )
    if response.code == 200:
        schema = JobEventsRsp()
        response_data = {
            "job_id": job_id,
            "events": response.data
        }
        return make_response(jsonify(schema.dump(schema.load(response_data))), 200)
    schema = ErrorRsp()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@jobs_bp_v2.route('/orgs/<org_name>/jobs/<job_id>:log_update', methods=['POST'])
@disk_space_check
def job_log_update(org_name, job_id):
    """Update Job log for Experiment."""
    message = validate_uuid(job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    handler_id, kind = get_handler_id_and_kind(job_id)
    if kind not in ['experiment', 'dataset']:
        metadata = {
            "error_desc": (f"Invalid job type for job_id {job_id}. "
                           "Job must be associated with an experiment or dataset."),
            "error_code": 1
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    experiment_id = None
    dataset_id = None
    if kind == 'experiment':
        experiment_id = handler_id
    else:
        dataset_id = handler_id
    callback_data = request.json
    response = JobHandler.job_log_update(
        org_name,
        experiment_id if kind == 'experiment' else dataset_id,
        job_id,
        kind,
        callback_data=callback_data
    )
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRsp()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@jobs_bp_v2.route('/orgs/<org_name>/jobs/<job_id>:status_update', methods=['POST'])
@disk_space_check
def job_status_update(org_name, job_id):
    """Update Job status for Experiment."""
    message = validate_uuid(job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    handler_id, kind = get_handler_id_and_kind(job_id)
    if kind not in ['experiment', 'dataset']:
        metadata = {
            "error_desc": (f"Invalid job type for job_id {job_id}. "
                           "Job must be associated with an experiment or dataset."),
            "error_code": 1
        }
        schema = ErrorRsp()
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)
    experiment_id = None
    dataset_id = None
    if kind == 'experiment':
        experiment_id = handler_id
    else:
        dataset_id = handler_id
    callback_data = request.json
    response = JobHandler.job_status_update(
        org_name,
        experiment_id if kind == 'experiment' else dataset_id,
        job_id,
        kind,
        callback_data=callback_data
    )
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRsp()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@jobs_bp_v2.route('/orgs/<org_name>/jobs:gpu_types', methods=['GET'])
@disk_space_check
def job_gpu_types(org_name):
    """Retrieve available GPU type.

    ---
    get:
      tags:
      - JOB
      summary: Retrieve available GPU types based on the configured compute backend
      description: Retrieve available GPU types based on the backend set during deployment
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      responses:
        200:
          description: Returned the gpu_types available
          content:
            application/json:
              schema:
                type: object
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: GPU types can't be retrieved for deployed Backend
          content:
            application/json:
              schema:
                type: object
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = SpecHandler.get_gpu_types(user_id, org_name)
    # Get schema
    schema = GpuDetails()
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)
