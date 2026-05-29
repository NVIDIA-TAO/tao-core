# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Datasets blueprint for API v2 dataset management endpoints."""

import math
import logging
from flask import Blueprint, request, jsonify, make_response

from nvidia_tao_core.microservices.decorators import disk_space_check
from nvidia_tao_core.microservices.utils.auth_utils import authentication
from nvidia_tao_core.microservices.utils.filter_utils import filtering, pagination
from .schemas import (
    DatasetListRsp,
    DatasetRsp,
    ErrorRsp,
    LstStr,
    DatasetReq,
    BulkOpsRsp
)
from nvidia_tao_core.microservices.utils.handler_utils import validate_uuid
from nvidia_tao_core.microservices.utils.core_utils import DataMonitorLogTypeEnum, log_api_error
from nvidia_tao_core.microservices.handlers.dataset_handler import DatasetHandler

logger = logging.getLogger(__name__)

# v2 Datasets Blueprint - URL prefix will be set during registration
datasets_bp_v2 = Blueprint('datasets_v2', __name__, template_folder='templates')

# Deprecation constants for dataset APIs
DATASET_API_SUNSET_DATE = 'Tue, 30 Jun 2026 23:59:59 GMT'  # 4 months from Feb 2026
DATASET_API_DEPRECATION_MESSAGE = (
    'Dataset APIs are deprecated. Use direct dataset paths in job/experiment creation instead. '
    'See migration guide: https://docs.nvidia.com/tao/dataset-migration'
)


def add_deprecation_headers(response):
    """Add deprecation headers to response per RFC 8594"""
    response.headers['X-API-Warn'] = DATASET_API_DEPRECATION_MESSAGE
    response.headers['Sunset'] = DATASET_API_SUNSET_DATE
    response.headers['Deprecation'] = 'true'
    return response


@datasets_bp_v2.route('/orgs/<org_name>/datasets:get_formats', methods=['GET'])
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
    logger.warning(
        f"DEPRECATED API USAGE: Get dataset formats via deprecated endpoint. "
        f"Sunset date: {DATASET_API_SUNSET_DATE}"
    )
    response_obj = make_response(jsonify(schema_dict), response.code)
    return add_deprecation_headers(response_obj)


@datasets_bp_v2.route('/orgs/<org_name>/datasets', methods=['GET'])
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
    return add_deprecation_headers(response)


@datasets_bp_v2.route('/orgs/<org_name>/datasets/<dataset_id>', methods=['GET'])
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
        return add_deprecation_headers(response)
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
    logger.warning(
        f"DEPRECATED API USAGE: Dataset retrieve via deprecated endpoint. "
        f"Sunset date: {DATASET_API_SUNSET_DATE}"
    )
    response_obj = make_response(jsonify(schema_dict), response.code)
    return add_deprecation_headers(response_obj)


@datasets_bp_v2.route('/orgs/<org_name>/datasets/<dataset_id>', methods=['DELETE'])
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
        return add_deprecation_headers(response)
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
    logger.warning(
        f"DEPRECATED API USAGE: Dataset delete via deprecated endpoint. "
        f"Sunset date: {DATASET_API_SUNSET_DATE}"
    )
    response_obj = make_response(jsonify(schema_dict), response.code)
    return add_deprecation_headers(response_obj)


@datasets_bp_v2.route('/orgs/<org_name>/datasets', methods=['POST'])
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
        201:
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
        ds_format = request_dict.get("format", "")
        log_type = (DataMonitorLogTypeEnum.medical_dataset
                    if ds_format == "monai"
                    else DataMonitorLogTypeEnum.tao_dataset)
        log_api_error(user_id, org_name, schema_dict, log_type, action="creation")
    else:
        # Log deprecation warning for successful dataset creation
        logger.warning(
            f"DEPRECATED API USAGE: User {user_id} created dataset via deprecated endpoint. "
            f"Sunset date: {DATASET_API_SUNSET_DATE}"
        )

    # Determine appropriate status code:
    # - 201 for newly created dataset
    # - 200 for existing dataset returned due to deduplication
    if response.code == 200:
        if "already exists" in response.message:
            # Return 200 for existing dataset
            http_status = 200
            # Add informational message to response
            schema_dict['_message'] = (
                "A dataset with the same configuration already exists. "
                "Returning existing dataset. "
                "To create a new dataset anyway, set 'force_create': true in the request body."
            )
            schema_dict['_duplicate'] = True
        else:
            # Return 201 for newly created dataset
            http_status = 201
    else:
        http_status = response.code

    resp = make_response(jsonify(schema_dict), http_status)
    return add_deprecation_headers(resp)


@datasets_bp_v2.route('/orgs/<org_name>/datasets/<dataset_id>', methods=['PUT'])
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
        return add_deprecation_headers(response)
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
    logger.warning(
        f"DEPRECATED API USAGE: Dataset update via deprecated endpoint. "
        f"Sunset date: {DATASET_API_SUNSET_DATE}"
    )
    response_obj = make_response(jsonify(schema_dict), response.code)
    return add_deprecation_headers(response_obj)


@datasets_bp_v2.route('/orgs/<org_name>/datasets/<dataset_id>', methods=['PATCH'])
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
        return add_deprecation_headers(response)
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
    logger.warning(
        f"DEPRECATED API USAGE: Dataset partial update via deprecated endpoint. "
        f"Sunset date: {DATASET_API_SUNSET_DATE}"
    )
    response_obj = make_response(jsonify(schema_dict), response.code)
    return add_deprecation_headers(response_obj)


@datasets_bp_v2.route('/orgs/<org_name>/datasets', methods=['DELETE'])
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
        response_obj = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return add_deprecation_headers(response_obj)

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
    logger.warning(
        f"DEPRECATED API USAGE: Bulk dataset delete via deprecated endpoint. "
        f"Sunset date: {DATASET_API_SUNSET_DATE}"
    )
    response_obj = make_response(jsonify(schema_dict), 200)
    return add_deprecation_headers(response_obj)
