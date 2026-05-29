# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Datasets blueprint for API v1 dataset management endpoints."""

import ast
import math
import logging
from flask import Blueprint, request, jsonify, make_response, send_from_directory

from nvidia_tao_core.microservices.decorators import disk_space_check
from nvidia_tao_core.microservices.utils.auth_utils import authentication
from nvidia_tao_core.microservices.utils.filter_utils import filtering, pagination
from nvidia_tao_core.microservices.handlers.dataset_handler import DatasetHandler
from nvidia_tao_core.microservices.handlers.job_handler import JobHandler
from nvidia_tao_core.microservices.handlers.spec_handler import SpecHandler
from .schemas import (
    DatasetListRsp,
    DatasetRsp,
    ErrorRsp,
    LstStr,
    DatasetReq,
    DatasetJob,
    DatasetActions,
    DatasetJobList,
    BulkOpsRsp,
    MessageOnly
)
from nvidia_tao_core.microservices.utils.handler_utils import validate_uuid
from nvidia_tao_core.microservices.utils.core_utils import DataMonitorLogTypeEnum, log_api_error

logger = logging.getLogger(__name__)

# v1 Datasets Blueprint - URL prefix will be set during registration
datasets_bp_v1 = Blueprint('datasets_v1', __name__, template_folder='templates')


@datasets_bp_v1.route('/orgs/<org_name>/datasets:get_formats', methods=['GET'])
def get_dataset_formats(org_name):
    """Get dataset formats supported.

    ---
    post:
        tags:
        - DATASET
        summary: Given dataset type return dataset formats or return all formats
        description: Given dataset type return dataset formats or return all formats
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
            description: Returns a list of dataset formats supported
            content:
              application/json:
                schema: LstStr
            headers:
              Access-Control-Allow-Origin:
                $ref: '#/components/headers/Access-Control-Allow-Origin'
              X-RateLimit-Limit:
                $ref: '#/components/headers/X-RateLimit-Limit'
          404:
            description: Bad request, see reply body for details
            content:
              application/json:
                schema: ErrorRsp
            headers:
              Access-Control-Allow-Origin:
                $ref: '#/components/headers/Access-Control-Allow-Origin'
              X-RateLimit-Limit:
                $ref: '#/components/headers/X-RateLimit-Limit'
    """
    dataset_type = str(request.args.get('dataset_type', ''))
    # Get response
    response = DatasetHandler.get_dataset_formats(dataset_type)
    # Get schema
    schema = None
    if response.code == 200:
        schema = LstStr()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets', methods=['GET'])
@disk_space_check
def dataset_list(org_name):
    """List Datasets.

    ---
    get:
      tags:
      - DATASET
      summary: List all accessible datasets
      description: |
        Returns a list of datasets that the authenticated user can access.
        Results can be filtered and paginated using query parameters.
        This includes:
        - Datasets owned by the user
        - Datasets shared with the user
        - Public datasets
      parameters:
      - name: org_name
        in: path
        description: Organization name to list datasets from
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: skip
        in: query
        description: Number of records to skip for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: size
        in: query
        description: Maximum number of records to return per page
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: sort
        in: query
        description: Sort order for the results
        required: false
        schema:
          type: string
          enum: ["date-descending", "date-ascending", "name-descending", "name-ascending" ]
      - name: name
        in: query
        description: Filter datasets by name (case-sensitive partial match)
        required: false
        schema:
          type: string
          maxLength: 5000
          pattern: '.*'
      - name: format
        in: query
        description: Filter datasets by their format type
        required: false
        schema:
          type: string
          enum: [
              "kitti", "pascal_voc", "raw", "coco_raw", "unet", "coco", "lprnet", "train", "test",
              "default", "custom", "classification_pyt", "visual_changenet_segment",
              "visual_changenet_classify"
          ]
      - name: type
        in: query
        description: Filter datasets by their primary type
        required: false
        schema:
          type: string
          enum: [
              "object_detection", "segmentation", "image_classification", "character_recognition",
              "action_recognition", "pointpillars", "pose_classification", "ml_recog", "ocdnet", "ocrnet",
              "optical_inspection", "re_identification", "centerpose"
          ]
      responses:
        200:
          description: Successfully retrieved list of accessible datasets
          content:
            application/json:
              schema: DatasetListRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    datasets = DatasetHandler.list_datasets(user_id, org_name)
    filtered_datasets = filtering.apply(request.args, datasets)
    paginated_datasets = pagination.apply(request.args, filtered_datasets)
    metadata = {"datasets": paginated_datasets}
    # Pagination
    skip = request.args.get("skip", None)
    size = request.args.get("size", None)
    if skip is not None and size is not None:
        skip = int(skip)
        size = int(size)
        metadata["pagination_info"] = {
            "total_records": len(filtered_datasets),
            "total_pages": math.ceil(len(filtered_datasets) / size),
            "page_size": size,
            "page_index": skip // size,
        }
    schema = DatasetListRsp()
    response = make_response(jsonify(schema.dump(schema.load(metadata))))
    return response


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>', methods=['GET'])
@disk_space_check
def dataset_retrieve(org_name, dataset_id):
    """Retrieve Dataset.

    ---
    get:
      tags:
      - DATASET
      summary: Retrieve details of a specific dataset
      description: |
        Returns detailed information about a specific dataset including:
        - Basic metadata (name, description, creation date)
        - Dataset format and type
        - Access permissions
        - Associated jobs and their status
        - Available actions
      parameters:
      - name: org_name
        in: path
        description: Organization name owning the dataset
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: Unique identifier of the dataset to retrieve
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully retrieved dataset details
          content:
            application/json:
              schema: DatasetRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Dataset not found or user lacks access permissions
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = DatasetHandler.retrieve_dataset(org_name, dataset_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRsp()
    else:
        # Check if this is a dataset validation error that should include structured details
        if (response.code == 404 and
                isinstance(response.data, dict) and
                response.data.get("validation_details")):
            # Use DatasetRsp for validation errors to include structured details
            schema = DatasetRsp()
        else:
            # Use ErrorRsp for other error types
            schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>', methods=['DELETE'])
@disk_space_check
def dataset_delete(org_name, dataset_id):
    """Delete Dataset.

    ---
    delete:
      tags:
      - DATASET
      summary: Delete a specific dataset
      description: |
        Deletes a dataset and its associated resources. The operation will:
        - Remove dataset files and metadata
        - Update user permissions

        Deletion is only allowed if:
        - User has write permissions
        - Dataset is not public
        - Dataset is not read-only
        - Dataset is not in use by any experiments
        - No running jobs are using the dataset
      parameters:
      - name: org_name
        in: path
        description: Organization name owning the dataset
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: Unique identifier of the dataset to delete
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Dataset successfully deleted
          content:
            application/json:
              schema: DatasetRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Dataset cannot be deleted due to active usage or permissions
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Dataset not found or user lacks delete permissions
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = DatasetHandler.delete_dataset(org_name, dataset_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRsp()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets', methods=['POST'])
@disk_space_check
def dataset_create(org_name):
    """Create new Dataset.

    ---
    post:
      tags:
      - DATASET
      summary: Create new Dataset
      description: Returns the new Dataset
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
            schema: DatasetReq
        description: Initial metadata for new Dataset (type and format required)
        required: true
      responses:
        200:
          description: Returned the new Dataset
          content:
            application/json:
              schema: DatasetRsp
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
    """
    schema = DatasetReq()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))

    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    # Get response
    response = DatasetHandler.create_dataset(user_id, org_name, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRsp()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    if response.code != 200:
        log_api_error(user_id, org_name, schema_dict, DataMonitorLogTypeEnum.tao_dataset, action="creation")

    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>', methods=['PUT'])
@disk_space_check
def dataset_update(org_name, dataset_id):
    """Update Dataset.

    ---
    put:
      tags:
      - DATASET
      summary: Update Dataset
      description: Returns the updated Dataset
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID of Dataset to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: DatasetReq
        description: Updated metadata for Dataset
        required: true
      responses:
        200:
          description: Returned the updated Dataset
          content:
            application/json:
              schema: DatasetRsp
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
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = DatasetReq()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = DatasetHandler.update_dataset(org_name, dataset_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRsp()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>', methods=['PATCH'])
@disk_space_check
def dataset_partial_update(org_name, dataset_id):
    """Partial update Dataset.

    ---
    patch:
      tags:
      - DATASET
      summary: Partial update Dataset
      description: Returns the updated Dataset
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID of Dataset to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: DatasetReq
        description: Updated metadata for Dataset
        required: true
      responses:
        200:
          description: Returned the updated Dataset
          content:
            application/json:
              schema: DatasetRsp
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
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = DatasetReq()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = DatasetHandler.update_dataset(org_name, dataset_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRsp()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>/jobs', methods=['POST'])
@disk_space_check
def dataset_job_run(org_name, dataset_id):
    """Run Dataset Jobs.

    ---
    post:
      tags:
      - DATASET
      summary: Run Dataset Jobs
      description: |
        Asynchronously starts a dataset action and returns corresponding Job ID. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the requested action is supported
        - Validates the provided specs match the action schema
        - Creates a new job with the provided parameters
        - Queues the job for execution
        - Returns the Job ID for tracking and retrieval
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: DatasetActions
      responses:
        200:
          description: Returned the Job ID corresponding to requested Dataset Action
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
          description: Invalid request (e.g. invalid dataset ID, missing required fields)
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
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.get_json(force=True).copy()
    schema = DatasetActions()
    request_schema_data = schema.dump(schema.load(request_data))
    requested_job = request_schema_data.get('parent_job_id', None)
    if requested_job:
        requested_job = str(requested_job)
    requested_action = request_schema_data.get('action', "")
    specs = request_schema_data.get('specs', {})
    name = request_schema_data.get('name', '')
    description = request_schema_data.get('description', '')
    num_gpu = request_schema_data.get('num_gpu', -1)
    backend_details = request_schema_data.get('backend_details', None)
    timeout_minutes = request_schema_data.get('timeout_minutes', 60)
    # Get response
    response = JobHandler.job_run(
        org_name, dataset_id, requested_job, requested_action, "dataset",
        specs=specs, name=name, description=description, num_gpu=num_gpu,
        timeout_minutes=timeout_minutes,
        backend_details=backend_details
    )
    # Get schema
    if response.code == 200:
        if isinstance(response.data, str) and not validate_uuid(response.data):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: invalid job IDs", "error_code": 2}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>/jobs', methods=['GET'])
@disk_space_check
def dataset_job_list(org_name, dataset_id):
    """List Jobs for Dataset.

    ---
    get:
      tags:
      - DATASET
      summary: List Jobs for Dataset
      description: |
        Returns the list of Jobs for a given dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Retrieves the list of jobs from storage
        - Applies pagination and filtering based on query parameters
        - Returns the filtered and paginated list of jobs
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          pattern: '.*'
          maxLength: 36
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
          enum: ["date-descending", "date-ascending" ]
      responses:
        200:
          description: Returned list of Jobs
          content:
            application/json:
              schema: DatasetJobList
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID)
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
    message = validate_uuid(dataset_id=None if dataset_id in ("*", "all") else dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response

    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    # Get response
    response = JobHandler.job_list(user_id, org_name, dataset_id, "dataset")
    # Get schema
    if response.code == 200:
        filtered_jobs = filtering.apply(request.args, response.data)
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
        schema = DatasetJobList()
        response = make_response(jsonify(schema.dump(schema.load(metadata))))
        return response
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>', methods=['GET'])
@disk_space_check
def dataset_job_retrieve(org_name, dataset_id, job_id):
    """Retrieve Job for Dataset.

    ---
    get:
      tags:
      - DATASET
      summary: Retrieve Job for Dataset
      description: |
        Returns the Job for a given dataset and job ID. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists
        - Retrieves the job from storage
        - Returns the job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
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
          description: Returned Job
          content:
            application/json:
              schema: DatasetJob
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID, job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    return_specs = ast.literal_eval(request.args.get('return_specs', "False"))
    response = JobHandler.job_retrieve(org_name, dataset_id, job_id, "dataset", return_specs=return_specs)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetJob()
    else:
        schema = ErrorRsp()
        # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>', methods=['DELETE'])
@disk_space_check
def dataset_job_delete(org_name, dataset_id, job_id):
    """Delete Dataset Job.

    ---
    delete:
      tags:
      - DATASET
      summary: Delete Dataset Job
      description: |
        Deletes a specific job within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists and is deletable
        - Deletes the job files and metadata
        - Updates job status to 'deleted'
        - Returns the deletion status
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
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
          description: Successfully requested deletion of specified Job ID
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID, job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = JobHandler.job_delete(org_name, dataset_id, job_id, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:retry', methods=['POST'])
@disk_space_check
def dataset_job_retry(org_name, dataset_id, job_id):
    """Retry Dataset Jobs.

    ---
    post:
      tags:
      - DATASET
      summary: Retry Dataset Jobs
      description: |
        Asynchronously retries a dataset action and returns corresponding Job ID. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists and is retryable
        - Creates a new job with the same parameters as the original job
        - Queues the job for execution
        - Returns the new Job ID for tracking and retrieval
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
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
          description: Returned the Job ID corresponding to requested Dataset Action
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
          description: Invalid request (e.g. invalid dataset ID, job ID)
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
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = JobHandler.job_retry(org_name, dataset_id, "dataset", job_id)
    # Get schema
    if response.code == 200:
        if isinstance(response.data, str) and not validate_uuid(response.data):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: invalid job IDs", "error_code": 2}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:cancel', methods=['POST'])
@disk_space_check
def dataset_job_cancel(org_name, dataset_id, job_id):
    """Cancel Dataset Job.

    ---
    post:
      tags:
      - DATASET
      summary: Cancel Dataset Job
      description: |
        Cancels a specific job within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists and is cancellable
        - Updates the job status to 'cancelled'
        - Persists status changes to storage
        - Triggers any necessary cancellation workflows
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
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
          description: Successfully requested cancelation of specified Job ID (asynchronous)
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID, job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = JobHandler.job_cancel(org_name, dataset_id, job_id, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:download', methods=['GET'])
@disk_space_check
def dataset_job_download(org_name, dataset_id, job_id):
    """Download Job Artifacts.

    ---
    get:
      tags:
      - DATASET
      summary: Download Job Artifacts
      description: |
        Downloads all artifacts produced by a given job within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists
        - Downloads all job files
        - Returns the downloaded files as a tarball
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
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
          description: Invalid request (e.g. invalid dataset ID, job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = JobHandler.job_download(org_name, dataset_id, job_id, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        file_path = response.data  # Response is assumed to have the file path
        file_dir = "/".join(file_path.split("/")[:-1])
        file_name = file_path.split("/")[-1]  # infer the name
        return send_from_directory(file_dir, file_name, as_attachment=True)
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>/logs', methods=['GET'])
def dataset_job_logs(org_name, dataset_id, job_id):
    """Get realtime dataset job logs.

    ---
    get:
      tags:
      - DATASET
      summary: Get Job logs for Dataset
      description: |
        Returns the job logs for a given dataset and job ID. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists
        - Retrieves the job logs from storage
        - Returns the job logs
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
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
          description: Invalid request (e.g. invalid dataset ID, job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Job not exist or logs not found.
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
               $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = JobHandler.get_job_logs(org_name, dataset_id, job_id, "dataset")
    if response.code == 200:
        response = make_response(response.data, 200)
        response.mimetype = 'text/plain'
        return response
    # Handle errors
    schema = ErrorRsp()
    response = make_response(jsonify(schema.dump(schema.load(response.data))), 400)
    return response


@datasets_bp_v1.route('/orgs/<org_name>/datasets', methods=['DELETE'])
@disk_space_check
def bulk_dataset_delete(org_name):
    """Bulk Delete Datasets.

    ---
    delete:
      tags:
      - DATASET
      summary: Delete multiple Datasets
      description: |
        Deletes multiple datasets and their associated resources. This endpoint:
        - Validates the datasets exist and user has access
        - Validates the datasets are not public
        - Validates the datasets are not read-only
        - Validates the datasets are not in use by any experiments
        - Validates no running jobs are using the datasets
        - Deletes the dataset files and metadata
        - Updates user permissions
        - Returns the status for each dataset
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
            schema:
              type: object
              properties:
                dataset_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                    maxLength: 36
                  maxItems: 2147483647
      responses:
        200:
          description: Deleted Datasets status
          content:
            application/json:
              schema: DatasetRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset IDs)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: One or more Datasets not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get dataset IDs from request body
    data = request.get_json()
    dataset_ids = data.get('dataset_ids')

    if not dataset_ids or not isinstance(dataset_ids, list):
        metadata = {"error_desc": "Invalid dataset IDs", "error_code": 1}
        schema = ErrorRsp()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

    results = []
    for dataset_id in dataset_ids:
        message = validate_uuid(dataset_id=dataset_id)
        if message:
            metadata = {"id": dataset_id, "error_desc": message, "error_code": 1}
            results.append(metadata)
            continue

        # Attempt to delete the dataset
        response = DatasetHandler.delete_dataset(org_name, dataset_id)
        if response.code == 200:
            results.append({"id": dataset_id, "status": "success"})
        else:
            results.append({"id": dataset_id, "status": "failed", "error_desc": response.data.get("error_desc", "")})

    # Return status for all datasets
    schema = BulkOpsRsp()
    schema_dict = schema.dump(schema.load({"results": results}))
    return make_response(jsonify(schema_dict), 200)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>/jobs', methods=['DELETE'])
@disk_space_check
def bulk_dataset_job_delete(org_name, dataset_id):
    """Bulk Delete Dataset Jobs.

    ---
    delete:
      tags:
      - DATASET
      summary: Delete multiple Dataset Jobs
      description: |
        Deletes multiple jobs within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the jobs exist and are deletable
        - Deletes the job files and metadata
        - Updates job status to 'deleted'
        - Returns the status for each job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                job_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                    maxLength: 36
                  maxItems: 2147483647
      responses:
        200:
          description: Successfully requested deletion of specified Job IDs
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID, job IDs)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Jobs not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get job IDs from request body
    data = request.get_json()
    job_ids = data.get('job_ids')

    if not job_ids or not isinstance(job_ids, list):
        metadata = {"error_desc": "Invalid job IDs", "error_code": 1}
        schema = ErrorRsp()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

    results = []
    for job_id in job_ids:
        message = validate_uuid(job_id=job_id)
        if message:
            metadata = {"id": job_id, "error_desc": message, "error_code": 1}
            results.append(metadata)
            continue

        # Attempt to delete the job
        response = JobHandler.job_delete(org_name, dataset_id, job_id, "dataset")
        if response.code == 200:
            results.append({"id": job_id, "status": "success"})
        else:
            results.append({"id": job_id, "status": "failed", "error_desc": response.data.get("error_desc", "")})

    # Return status for all jobs
    schema = BulkOpsRsp()
    schema_dict = schema.dump(schema.load({"results": results}))
    return make_response(jsonify(schema_dict), 200)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>/schema', methods=['GET'])
@disk_space_check
def dataset_job_schema(org_name, dataset_id, job_id):
    """Retrieve Schema for a job.

    ---
    get:
      tags:
      - DATASET
      summary: Retrieve Schema for a job
      description: |
        Returns the Specs schema for a given job. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists
        - Retrieves the schema for the job's action
        - Returns the schema
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for JOB
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned the Specs schema for given action
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
          description: Invalid request (e.g. invalid dataset ID, job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Dataset or Action not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = SpecHandler.get_spec_schema_for_job(user_id, org_name, dataset_id, job_id, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>/specs/<action>/schema', methods=['GET'])
@disk_space_check
def dataset_specs_schema(org_name, dataset_id, action):
    """Retrieve Specs schema.

    ---
    get:
      tags:
      - DATASET
      summary: Retrieve Specs schema
      description: Returns the Specs schema for a given action
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
          enum: [
              "dataset_convert", "convert", "kmeans", "augment", "train",
              "evaluate", "prune", "retrain", "export", "gen_trt_engine", "trtexec", "inference",
              "annotation", "analyze", "validate", "auto_label", "calibration_tensorfile"
          ]
      responses:
        200:
          description: Returned the Specs schema for given action
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
          description: User, Dataset or Action not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = SpecHandler.get_spec_schema(user_id, org_name, dataset_id, action, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:download_selective_files', methods=['GET'])
@disk_space_check
def dataset_job_download_selective_files(org_name, dataset_id, job_id):
    """Download selective Job Artifacts.

    ---
    get:
      tags:
      - DATASET
      summary: Download selective Job Artifacts
      description: |
        Downloads selective artifacts produced by a given job within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists
        - Validates the requested files exist
        - Downloads the requested files
        - Returns the downloaded files as a tarball
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
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
                maxLength: 5000
                maxLength: 5000
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID, job ID, file list)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    file_lists = request.args.getlist('file_lists')
    tar_files = ast.literal_eval(request.args.get('tar_files', "True"))
    if not file_lists:
        return make_response(jsonify("No files passed in list format to download or"), 400)
    # Get response
    response = JobHandler.job_download(
        org_name,
        dataset_id,
        job_id,
        "dataset",
        file_lists=file_lists,
        tar_files=tar_files
    )
    # Get schema
    schema = None
    if response.code == 200:
        file_path = response.data  # Response is assumed to have the file path
        file_dir = "/".join(file_path.split("/")[:-1])
        file_name = file_path.split("/")[-1]  # infer the name
        return send_from_directory(file_dir, file_name, as_attachment=True)
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:list_files', methods=['GET'])
@disk_space_check
def dataset_job_files_list(org_name, dataset_id, job_id):
    """List Job Files.

    ---
    get:
      tags:
      - DATASET
      summary: List Job Files
      description: |
        Lists the files produced by a given job within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists
        - Retrieves the list of files from storage
        - Returns the list of files
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
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
                  maxLength: 1000
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID, job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = JobHandler.job_list_files(org_name, dataset_id, job_id, "dataset")
    # Get schema
    if response.code == 200:
        if isinstance(response.data, list) and (all(isinstance(f, str) for f in response.data) or response.data == []):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: file list invalid", "error_code": 2}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:log_update', methods=['POST'])
@disk_space_check
def dataset_job_log_update(org_name, dataset_id, job_id):
    """Update Job log for Dataset.

    ---
    post:
      tags:
      - DATASET
      summary: Update log of a dataset job
      description: |
        Updates the log of a specific job within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists
        - Appends the provided log data to the job's log
        - Persists log changes to storage
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              description: Log update data including new log entries
      responses:
        200:
          description: Job log successfully updated
          content:
            application/json:
              schema: DatasetJob
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid log update request (e.g. missing required fields)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    callback_data = request.json
    # Get response
    response = JobHandler.job_log_update(org_name, dataset_id, job_id, "dataset", callback_data=callback_data)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRsp()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:status_update', methods=['POST'])
@disk_space_check
def dataset_job_status_update(org_name, dataset_id, job_id):
    """Update Job status for Dataset.

    ---
    post:
      tags:
      - DATASET
      summary: Update status of a dataset job
      description: |
        Updates the status of a specific job within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists
        - Updates the job status based on provided data
        - Persists status changes to storage
        - Triggers any necessary status-based workflows
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              description: Status update data including new status and any additional metadata
      responses:
        200:
          description: Job status successfully updated
          content:
            application/json:
              schema: DatasetJob
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid status update request (e.g. invalid status value, missing required fields)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    callback_data = request.json
    # Get response
    response = JobHandler.job_status_update(org_name, dataset_id, job_id, "dataset", callback_data=callback_data)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRsp()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets/<dataset_id>:cancel_all_jobs', methods=['POST'])
@disk_space_check
def dataset_jobs_cancel(org_name, dataset_id):
    """Cancel all jobs within dataset (or pause training).

    ---
    post:
      tags:
      - DATASET
      summary: Cancel all Jobs under dataset
      description: |
        Cancels all jobs within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the jobs exist and are cancellable
        - Updates the job status to 'cancelled'
        - Persists status changes to storage
        - Triggers any necessary cancellation workflows
        - Returns the cancellation status
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully canceled all jobs under datasets
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = JobHandler.all_job_cancel(user_id, org_name, dataset_id, "dataset")
    # Get schema
    if response.code == 200:
        schema = MessageOnly()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@datasets_bp_v1.route('/orgs/<org_name>/datasets:cancel_all_jobs', methods=['POST'])
@disk_space_check
def bulk_dataset_jobs_cancel(org_name):
    """Cancel all jobs within multiple datasets.

    ---
    post:
      tags:
      - DATASET
      summary: Cancel all Jobs under multiple datasets
      description: |
        Cancels all jobs within multiple datasets. This endpoint:
        - Validates the datasets exist and user has access
        - Validates the jobs exist and are cancellable
        - Updates the job status to 'cancelled'
        - Persists status changes to storage
        - Triggers any necessary cancellation workflows
        - Returns the cancellation status for each dataset
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
            schema:
              type: object
              properties:
                dataset_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                    maxLength: 36
                  maxItems: 2147483647
      responses:
        200:
          description: Successfully canceled all jobs under the specified datasets
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset IDs)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Datasets or Jobs not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get dataset IDs from the request body
    data = request.get_json()
    dataset_ids = data.get('dataset_ids')

    if not dataset_ids or not isinstance(dataset_ids, list):
        metadata = {"error_desc": "Invalid dataset IDs", "error_code": 1}
        schema = ErrorRsp()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

    results = []
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)

    for dataset_id in dataset_ids:
        message = validate_uuid(dataset_id=dataset_id)
        if message:
            results.append({"id": dataset_id, "error_desc": message, "error_code": 1})
            continue

        # Cancel all jobs for each dataset
        response = JobHandler.all_job_cancel(user_id, org_name, dataset_id, "dataset")
        if response.code == 200:
            results.append({"id": dataset_id, "status": "success"})
        else:
            results.append({"id": dataset_id, "status": "failed", "error_desc": response.data.get("error_desc", "")})

    # Return status for each dataset's job cancellation
    schema = BulkOpsRsp()
    schema_dict = schema.dump(schema.load({"results": results}))
    return make_response(jsonify(schema_dict), 200)
