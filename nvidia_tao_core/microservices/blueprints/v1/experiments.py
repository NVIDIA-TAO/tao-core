# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experiments blueprint for API v1 experiment management endpoints."""

import ast
import os
import uuid
import math
import logging
import traceback
from flask import Blueprint, request, jsonify, make_response, send_from_directory

from nvidia_tao_core.microservices.decorators import disk_space_check
from nvidia_tao_core.microservices.utils.auth_utils import authentication
from nvidia_tao_core.microservices.utils.filter_utils import filtering, pagination
from nvidia_tao_core.microservices.handlers.experiment_handler import ExperimentHandler
from nvidia_tao_core.microservices.handlers.job_handler import JobHandler
from nvidia_tao_core.microservices.handlers.spec_handler import SpecHandler
from nvidia_tao_core.microservices.handlers.model_handler import ModelHandler
from nvidia_tao_core.microservices.utils.handler_utils import validate_uuid, resolve_metadata
from nvidia_tao_core.microservices.utils.core_utils import DataMonitorLogTypeEnum, log_api_error, log_monitor
from nvidia_tao_core.microservices.handlers.inference_microservice_handler import InferenceMicroserviceHandler
from requests_toolbelt import MultipartEncoder
from marshmallow import exceptions
import shutil
from .schemas import (
    ExperimentListRsp,
    ExperimentRsp,
    ExperimentReq,
    ExperimentJob,
    ExperimentJobList,
    ExperimentTagList,
    ExperimentActions,
    LoadAirgappedExperimentsReq,
    LoadAirgappedExperimentsRsp,
    ErrorRsp,
    JobResume,
    ExperimentDownload,
    ExperimentExportTypeEnum,
    LstInt,
    PublishModel,
    BulkOpsRsp,
    MessageOnly,
    JobEventsRsp
)

logger = logging.getLogger(__name__)

# v1 Experiments Blueprint - URL prefix will be set during registration
experiments_bp_v1 = Blueprint('experiments_v1', __name__, template_folder='templates')


@experiments_bp_v1.route('/orgs/<org_name>/experiments', methods=['GET'])
@disk_space_check
def experiment_list(org_name):
    """List Experiments.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: List Experiments
      description: Returns the list of Experiments
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
          description: Returned the list of Experiments
          content:
            application/json:
              schema: ExperimentListRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    user_only = str(request.args.get('user_only', None)) in {'True', 'yes', 'y', 'true', 't', '1', 'on'}
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    experiments = ExperimentHandler.list_experiments(user_id, org_name, user_only)
    filtered_experiments = filtering.apply(request.args, experiments)
    paginated_experiments = pagination.apply(request.args, filtered_experiments)
    metadata = {"experiments": paginated_experiments}
    # Pagination
    skip = request.args.get("skip", None)
    size = request.args.get("size", None)
    if skip is not None and size is not None:
        skip = int(skip)
        size = int(size)
        metadata["pagination_info"] = {
            "total_records": len(filtered_experiments),
            "total_pages": math.ceil(len(filtered_experiments) / size),
            "page_size": size,
            "page_index": skip // size,
        }
    schema = ExperimentListRsp()
    response = make_response(jsonify(schema.dump(schema.load(metadata))))
    return response


@experiments_bp_v1.route('/orgs/<org_name>/experiments:get_tags', methods=['GET'])
@disk_space_check
def experiment_tags_list(org_name):
    """Retrieve All Unique Experiment Tags.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve all unique experiment tags
      description: Returns all unique experiment tags
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
          description: Returned the unique experiment tags list
          content:
            application/json:
              schema: ExperimentRsp
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    experiments = ExperimentHandler.list_experiments(user_id, org_name, user_only=True)
    tags = [tag for exp in experiments for tag in exp.get('tags', [])]
    unique_tags = list({t.lower(): t for t in tags}.values())
    metadata = {"tags": unique_tags}
    schema = ExperimentTagList()
    response = make_response(jsonify(schema.dump(schema.load(metadata))))
    return response


@experiments_bp_v1.route('/orgs/<org_name>/experiments:base', methods=['GET'])
@disk_space_check
def base_experiment_list(org_name):
    """List Base Experiments.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: List Experiments that can be used for transfer learning
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
          description: Returned the list of Experiments
          content:
            application/json:
              schema: ExperimentListRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    experiments = ExperimentHandler.list_base_experiments(user_id, org_name)
    filtered_experiments = filtering.apply(request.args, experiments)
    paginated_experiments = pagination.apply(request.args, filtered_experiments)
    metadata = {"experiments": paginated_experiments}
    # Pagination
    skip = request.args.get("skip", None)
    size = request.args.get("size", None)
    if skip is not None and size is not None:
        skip = int(skip)
        size = int(size)
        metadata["pagination_info"] = {
            "total_records": len(filtered_experiments),
            "total_pages": math.ceil(len(filtered_experiments) / size),
            "page_size": size,
            "page_index": skip // size,
        }
    schema = ExperimentListRsp()
    response = make_response(jsonify(schema.dump(schema.load(metadata))))
    return response


@experiments_bp_v1.route('/orgs/<org_name>/experiments:load_airgapped', methods=['POST'])
@disk_space_check
def load_airgapped_experiments(org_name):
    """Load Airgapped Experiments.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Load base experiments from airgapped cloud storage
      description: Loads base experiment metadata from airgapped cloud storage using workspace credentials
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
          description: Successfully loaded airgapped experiments
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
    message = validate_uuid(workspace_id=request.get_json().get('workspace_id'))
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response

    schema = LoadAirgappedExperimentsReq()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))

    # Authenticate user
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)

    # Get response from handler
    response = ExperimentHandler.load_airgapped_experiments(
        user_id,
        org_name,
        request_dict['workspace_id']
    )

    # Get appropriate schema based on response code
    schema = None
    if response.code == 200:
        schema = LoadAirgappedExperimentsRsp()
    else:
        schema = ErrorRsp()

    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>', methods=['GET'])
@disk_space_check
def experiment_retrieve(org_name, experiment_id):
    """Retrieve Experiment.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve Experiment
      description: Returns the Experiment
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID of Experiment to return
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned the Experiment
          content:
            application/json:
              schema: ExperimentRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = ExperimentHandler.retrieve_experiment(org_name, experiment_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRsp()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments', methods=['DELETE'])
@disk_space_check
def bulk_experiment_delete(org_name):
    """Bulk Delete Experiments.

    ---
    delete:
      tags:
      - EXPERIMENT
      summary: Delete multiple Experiments
      description: Cancels all related running jobs and returns the status of deleted Experiments
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
                experiment_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                    maxLength: 36
                  maxItems: 2147483647
      responses:
        200:
          description: Deleted Experiments status
          content:
            application/json:
              schema: ExperimentRsp
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
          description: One or more Experiments not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get experiment IDs from request body
    data = request.get_json()
    experiment_ids = data.get('experiment_ids')

    if not experiment_ids or not isinstance(experiment_ids, list):
        metadata = {"error_desc": "Invalid experiment IDs", "error_code": 1}
        schema = ErrorRsp()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

    results = []
    for experiment_id in experiment_ids:
        message = validate_uuid(experiment_id=experiment_id)
        if message:
            metadata = {"id": experiment_id, "error_desc": message, "error_code": 1}
            results.append(metadata)
            continue

        # Attempt to delete the experiment
        response = ExperimentHandler.delete_experiment(org_name, experiment_id)
        if response.code == 200:
            results.append({"id": experiment_id, "status": "success"})
        else:
            results.append({"id": experiment_id, "status": "failed", "error_desc": response.data.get("error_desc", "")})

    # Return status for all experiments
    schema = BulkOpsRsp()
    schema_dict = schema.dump(schema.load({"results": results}))
    return make_response(jsonify(schema_dict), 200)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>', methods=['DELETE'])
@disk_space_check
def experiment_delete(org_name, experiment_id):
    """Delete Experiment.

    ---
    delete:
      tags:
      - EXPERIMENT
      summary: Delete Experiment
      description: Cancels all related running jobs and returns the deleted Experiment
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID of Experiment to delete
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned the deleted Experiment
          content:
            application/json:
              schema: ExperimentRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = ExperimentHandler.delete_experiment(org_name, experiment_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRsp()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments', methods=['POST'])
@disk_space_check
def experiment_create(org_name):
    """Create new Experiment.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Create new Experiment
      description: Returns the new Experiment
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
            schema: ExperimentReq
        description: Initial metadata for new Experiment (base_experiment or network_arch required)
        required: true
      responses:
        200:
          description: Returned the new Experiment
          content:
            application/json:
              schema: ExperimentRsp
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
    schema = ExperimentReq()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    # Get response
    response = ExperimentHandler.create_experiment(user_id, org_name, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRsp()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    if response.code != 200:
        log_api_error(user_id, org_name, schema_dict, DataMonitorLogTypeEnum.tao_experiment, action="creation")

    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>', methods=['PUT'])
@disk_space_check
def experiment_update(org_name, experiment_id):
    """Update Experiment.

    ---
    put:
      tags:
      - EXPERIMENT
      summary: Update Experiment
      description: |
        Updates an existing experiment with new metadata. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the provided metadata matches the schema
        - Updates the experiment metadata in storage
        - Returns the updated experiment metadata
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID of Experiment to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: ExperimentReq
        description: Updated metadata for Experiment
        required: true
      responses:
        200:
          description: Returned the updated Experiment
          content:
            application/json:
              schema: ExperimentRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, missing required fields)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = ExperimentReq()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = ExperimentHandler.update_experiment(org_name, experiment_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRsp()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>', methods=['PATCH'])
@disk_space_check
def experiment_partial_update(org_name, experiment_id):
    """Partial update Experiment.

    ---
    patch:
      tags:
      - EXPERIMENT
      summary: Partial update Experiment
      description: |
        Partially updates an existing experiment with new metadata. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the provided metadata matches the schema
        - Updates the experiment metadata in storage
        - Returns the updated experiment metadata
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID of Experiment to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: ExperimentReq
        description: Updated metadata for Experiment
        required: true
      responses:
        200:
          description: Returned the updated Experiment
          content:
            application/json:
              schema: ExperimentRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, missing required fields)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = ExperimentReq()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = ExperimentHandler.update_experiment(org_name, experiment_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRsp()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/specs/<action>/schema', methods=['GET'])
@disk_space_check
def specs_schema_without_handler_id(org_name, action):
    """Retrieve Specs schema.

    ---
    get:
      summary: Retrieve Specs schema without experiment or dataset id
      description: |
        Returns the Specs schema for a given action. This endpoint:
        - Validates the action is supported
        - Retrieves the schema for the action
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
          description: Returned the Specs schema for given action and network
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
    # Get response
    network = request.args.get('network')
    dataset_format = request.args.get('format')
    train_datasets = request.args.getlist('train_datasets')

    response = SpecHandler.get_spec_schema_without_handler_id(org_name, network, dataset_format, action, train_datasets)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/specs/<action>/schema', methods=['GET'])
@disk_space_check
def experiment_specs_schema(org_name, experiment_id, action):
    """Retrieve Specs schema.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve Specs schema
      description: |
        Returns the Specs schema for a given action and experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the action is supported
        - Retrieves the schema for the action
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
      - name: experiment_id
        in: path
        description: ID for Experiment
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
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = SpecHandler.get_spec_schema(user_id, org_name, experiment_id, action, "experiment")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/specs/<action>/schema:base', methods=['GET'])
@disk_space_check
def base_experiment_specs_schema(org_name, experiment_id, action):
    """Retrieve Base Experiment Specs schema.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve Base Experiment Specs schema
      description: |
        Returns the Specs schema for a given action of the base experiment. This endpoint:
        - Validates the base experiment exists and user has access
        - Validates the action is supported
        - Retrieves the schema for the action
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
      - name: experiment_id
        in: path
        description: ID for Base Experiment
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
            "train", "evaluate", "prune", "retrain", "export", "gen_trt_engine", "trtexec",
            "inference", "auto_label"
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
          description: Action not found or Base spec file not present
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = ExperimentHandler.get_base_experiment_spec_schema(experiment_id, action)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs', methods=['POST'])
@disk_space_check
def experiment_job_run(org_name, experiment_id):
    """Run Experiment Jobs.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Run Experiment Jobs
      description: |
        Asynchronously starts a Experiment Action and returns corresponding Job ID. This endpoint:
        - Validates the experiment exists and user has access
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
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: ExperimentActions
      responses:
        200:
          description: Returned the Job ID corresponding to requested Experiment Action
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
          description: Invalid request (e.g. invalid experiment ID, missing required fields)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.get_json(force=True).copy()
    schema = ExperimentActions()
    request_schema_data = schema.dump(schema.load(request_data))
    requested_job = request_schema_data.get('parent_job_id', None)
    if requested_job:
        requested_job = str(requested_job)
    requested_action = request_schema_data.get('action', None)
    if not requested_action:
        metadata = {"error_desc": "Action is required to run job", "error_code": 400}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    specs = request_schema_data.get('specs', {})
    name = request_schema_data.get('name', '')
    description = request_schema_data.get('description', '')
    num_gpu = request_schema_data.get('num_gpu', -1)
    backend_details = request_schema_data.get('backend_details', None)
    retain_checkpoints_for_resume = request_schema_data.get('retain_checkpoints_for_resume', None)
    early_stop_epoch = request_schema_data.get('early_stop_epoch', None)
    timeout_minutes = request_schema_data.get('timeout_minutes', 60)

    if isinstance(specs, dict) and "cluster" in specs:
        metadata = {"error_desc": "cluster is an invalid spec", "error_code": 3}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = JobHandler.job_run(
        org_name, experiment_id, requested_job, requested_action, "experiment",
        specs=specs, name=name, description=description, num_gpu=num_gpu,
        retain_checkpoints_for_resume=retain_checkpoints_for_resume,
        early_stop_epoch=early_stop_epoch, timeout_minutes=timeout_minutes,
        backend_details=backend_details
    )
    # Get schema
    schema = None
    if response.code == 200:
        if hasattr(response, "attachment_key") and response.attachment_key:
            try:
                output_path = response.data[response.attachment_key]
                all_files = [
                    os.path.join(dirpath, f)
                    for dirpath, dirnames, filenames in os.walk(output_path)
                    for f in filenames
                ]
                files_dict = {}
                for f in all_files:
                    with open(f, "rb") as file:
                        files_dict[os.path.relpath(f, output_path)] = file.read()
                multipart_data = MultipartEncoder(fields=files_dict)
                send_file_response = make_response(multipart_data.to_string())
                send_file_response.headers["Content-Type"] = multipart_data.content_type
                # send_file sets correct response code as 200, should convert back to 200
                if send_file_response.status_code == 200:
                    send_file_response.status_code = response.code
                    # remove sent file as it's useless now
                    shutil.rmtree(response.data[response.attachment_key], ignore_errors=True)
                return send_file_response
            except Exception as e:
                # get user_id for more information
                handler_metadata = resolve_metadata("experiment", experiment_id)
                user_id = handler_metadata.get("user_id")
                logger.error(
                    f"respond attached data for org: {org_name} experiment: {experiment_id} "
                    f"user: {user_id} failed, got error: {e}"
                )
                metadata = {"error_desc": "respond attached data failed", "error_code": 2}
                schema = ErrorRsp()
                response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
                return response
        if isinstance(response.data, str) and not validate_uuid(response.data):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: invalid job IDs", "error_code": 2}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    if response.code != 200:
        try:
            handler_metadata = resolve_metadata("experiment", experiment_id)
            user_id = handler_metadata.get("user_id", None)
            if user_id:
                log_api_error(user_id, org_name, schema_dict, DataMonitorLogTypeEnum.tao_job, action="creation")
        except Exception as e:
            logger.error(f"Exception thrown in experiment_job_run is {str(e)}")
            log_monitor(DataMonitorLogTypeEnum.api, "Cannot parse experiment info for job.")

    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs', methods=['GET'])
@disk_space_check
def experiment_job_list(org_name, experiment_id):
    """List Jobs for Experiment.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: List Jobs for Experiment
      description: |
        Returns the list of Jobs for a given experiment. This endpoint:
        - Validates the experiment exists and user has access
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
      - name: experiment_id
        in: path
        description: ID for Experiment
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
              schema: ExperimentJobList
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=None if experiment_id in ("*", "all") else experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)

    # Get response
    response = JobHandler.job_list(user_id, org_name, experiment_id, "experiment")
    # Get schema
    schema = None
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
        schema = ExperimentJobList()
        response = make_response(jsonify(schema.dump(schema.load(metadata))))
        return response
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>', methods=['GET'])
@disk_space_check
def experiment_job_retrieve(org_name, experiment_id, job_id):
    """Retrieve Job for Experiment.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve Job for Experiment
      description: |
        Returns the Job for a given experiment and job ID. This endpoint:
        - Validates the experiment exists and user has access
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
      - name: experiment_id
        in: path
        description: ID of Experiment
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
              schema: ExperimentJob
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID)
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
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    return_specs = ast.literal_eval(request.args.get('return_specs', "False"))
    response = JobHandler.job_retrieve(org_name, experiment_id, job_id, "experiment", return_specs=return_specs)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentJob()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>', methods=['DELETE'])
@disk_space_check
def experiment_job_delete(org_name, experiment_id, job_id):
    """Delete Experiment Job.

    ---
    delete:
      tags:
      - EXPERIMENT
      summary: Delete Experiment Job
      description: |
        Deletes a specific job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
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
      - name: experiment_id
        in: path
        description: ID for Experiment
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
          description: Invalid request (e.g. invalid experiment ID, job ID)
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
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = JobHandler.job_delete(org_name, experiment_id, job_id, "experiment")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:retry', methods=['POST'])
@disk_space_check
def experiment_job_retry(org_name, experiment_id, job_id):
    """Retry Experiment Job.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Retry Experiment Jobs
      description: |
        Asynchronously retries a Experiment Action and returns corresponding Job ID. This endpoint:
        - Validates the experiment exists and user has access
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
      - name: experiment_id
        in: path
        description: ID for Experiment
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
          description: Returned the Job ID corresponding to requested Experiment Action
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
          description: Invalid request (e.g. invalid experiment ID, job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response

    # Get response
    response = JobHandler.job_retry(org_name, experiment_id, "experiment", job_id)
    # Get schema
    schema = None
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
    if response.code != 200:
        try:
            handler_metadata = resolve_metadata("experiment", experiment_id)
            user_id = handler_metadata.get("user_id", None)
            if user_id:
                log_api_error(user_id, org_name, schema_dict, DataMonitorLogTypeEnum.tao_job, action="creation")
        except Exception as e:
            logger.error(f"Exception thrown in experiment_job_retry is {str(e)}")
            log_monitor(DataMonitorLogTypeEnum.api, "Cannot parse experiment info for job.")

    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:publish_model', methods=['POST'])
@disk_space_check
def experiment_model_publish(org_name, experiment_id, job_id):
    """Publish models to NGC.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Publish models to NGC
      description: |
        Publishes models to NGC private registry. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists and is publishable
        - Validates the provided metadata matches the schema
        - Publishes the model to NGC with the provided metadata
        - Returns a success message
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
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
          description: Invalid request (e.g. invalid experiment ID, job ID, missing required fields)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    if os.getenv("AIRGAPPED_MODE", "false").lower() == "true":
        schema = MessageOnly()
        message = "Cannot publish model in air-gapped mode."
        response = make_response(jsonify(schema.dump({"message": message})), 400)
        return response

    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.get_json(force=True).copy()
    schema = PublishModel()
    request_schema_data = schema.dump(schema.load(request_data))
    display_name = request_schema_data.get('display_name', '')
    description = request_schema_data.get('description', '')
    team_name = request_schema_data.get('team_name', '')
    # Get response
    response = ModelHandler.publish_model(
        org_name,
        team_name,
        experiment_id,
        job_id,
        display_name=display_name,
        description=description
    )
    # Get schema
    schema_dict = None

    if response.code == 200:
        schema = MessageOnly()
        logger.info("Returning success response: %s", response.data)
        schema_dict = schema.dump({"message": "Published model into requested org"})
    else:
        schema = ErrorRsp()
        # Load metadata in schema and return
        schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:pause', methods=['POST'])
@disk_space_check
def experiment_job_pause(org_name, experiment_id, job_id):
    """Pause Experiment Job (only for training).

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Pause Experiment Job - only for training
      description: |
        Pauses a specific job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists and is pausable
        - Updates the job status to 'paused'
        - Persists status changes to storage
        - Triggers any necessary pause workflows
        - Returns the pause status
        - Supports graceful pause which allows checkpoints to be uploaded before shutdown
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
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
      requestBody:
        description: Optional parameters for job pause
        required: false
        content:
          application/json:
            schema:
              type: object
              properties:
                graceful:
                  type: boolean
                  description: |
                    If true, performs graceful pause by signaling the job to terminate
                    and upload checkpoints before shutting down. Default is false (abrupt pause).
                  default: false
      responses:
        200:
          description: Successfully requested training pause of specified Job ID (asynchronous)
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    logger.debug(
        f"[BLUEPRINT-PAUSE] Received pause request: org_name={org_name}, "
        f"experiment_id={experiment_id}, job_id={job_id}"
    )

    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        logger.debug(
            f"[BLUEPRINT-PAUSE] UUID validation failed: experiment_id={experiment_id}, "
            f"job_id={job_id}, message={message}"
        )
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response

    # Parse request body for graceful parameter
    request_data = request.get_json()
    graceful = request_data.get("graceful", False) if request_data else False
    logger.debug(f"[BLUEPRINT-PAUSE] Parsed request parameters: job_id={job_id}, graceful={graceful}")

    # Get response
    logger.debug(f"[BLUEPRINT-PAUSE] Calling JobHandler.job_pause: job_id={job_id}, graceful={graceful}")
    response = JobHandler.job_pause(org_name, experiment_id, job_id, "experiment", graceful=graceful)
    logger.debug(f"[BLUEPRINT-PAUSE] JobHandler.job_pause completed: job_id={job_id}, response_code={response.code}")
    # Get schema
    if response.code == 200:
        schema = MessageOnly()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:cancel', methods=['POST'])
@disk_space_check
def experiment_job_cancel(org_name, experiment_id, job_id):
    """Cancel Experiment Job (or pause training).

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Cancel Experiment Job or pause training
      description: |
        Cancels a specific job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists and is cancellable
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
      - name: experiment_id
        in: path
        description: ID for Experiment
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
          description: Successfully requested cancelation or training pause of specified Job ID
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID)
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
    logger.debug(
        f"[BLUEPRINT-CANCEL] Received cancel request: org_name={org_name}, "
        f"experiment_id={experiment_id}, job_id={job_id}"
    )

    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        logger.debug(
            f"[BLUEPRINT-CANCEL] UUID validation failed: experiment_id={experiment_id}, "
            f"job_id={job_id}, message={message}"
        )
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    logger.debug(f"[BLUEPRINT-CANCEL] Calling JobHandler.job_cancel: job_id={job_id}")
    response = JobHandler.job_cancel(org_name, experiment_id, job_id, "experiment")
    logger.debug(f"[BLUEPRINT-CANCEL] JobHandler.job_cancel completed: job_id={job_id}, response_code={response.code}")
    # Get schema
    if response.code == 200:
        schema = MessageOnly()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:resume', methods=['POST'])
@disk_space_check
def experiment_job_resume(org_name, experiment_id, job_id):
    """Resume Experiment Job - train/retrain only.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Resume Experiment Job
      description: |
        Resumes a specific job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists and is resumable
        - Validates the provided metadata matches the schema
        - Updates the job metadata
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
      - name: experiment_id
        in: path
        description: ID for Experiment
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
      requestBody:
        content:
          application/json:
            schema: JobResume
        description: Adjustable metadata for the resumed job.
        required: false
      responses:
        200:
          description: Successfully requested resume of specified Job ID (asynchronous)
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID, missing required fields)
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
    logger.debug(
        f"[BLUEPRINT-RESUME] Received resume request: org_name={org_name}, "
        f"experiment_id={experiment_id}, job_id={job_id}"
    )

    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        logger.debug(
            f"[BLUEPRINT-RESUME] UUID validation failed: experiment_id={experiment_id}, "
            f"job_id={job_id}, message={message}"
        )
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
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
    logger.debug(
        f"[BLUEPRINT-RESUME] Parsed request parameters: job_id={job_id}, name={name}, "
        f"num_gpu={num_gpu}, backend_details={backend_details}, "
        f"timeout_minutes={timeout_minutes}, has_specs={bool(specs)}"
    )
    # Get response
    logger.debug(f"[BLUEPRINT-RESUME] Calling ExperimentHandler.resume_experiment_job: job_id={job_id}")
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
    logger.debug(
        f"[BLUEPRINT-RESUME] ExperimentHandler.resume_experiment_job completed: "
        f"job_id={job_id}, response_code={response.code}"
    )
    logger.debug(
        f"[BLUEPRINT-RESUME] ExperimentHandler.resume_experiment_job completed: "
        f"job_id={job_id}, response_code={response.code}"
    )
    # Get schema
    if response.code == 200:
        schema = MessageOnly()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:download', methods=['GET'])
@disk_space_check
def experiment_job_download(org_name, experiment_id, job_id):
    """Download Job Artifacts.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Download Job Artifacts
      description: |
        Downloads the artifacts produced by a given job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists
        - Validates the requested export type is supported
        - Downloads the job artifacts
        - Returns the downloaded artifacts as a tarball
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID of Experiment
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
                maxLength: 1000
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID)
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
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.get_json(force=True, silent=True)
    request_data = {} if request_data is None else request_data
    try:
        request_schema_data = ExperimentDownload().load(request_data)
    except exceptions.ValidationError as err:
        metadata = {"error_desc": str(err)}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 404)
        return response
    export_type = request_schema_data.get("export_type", ExperimentExportTypeEnum.tao)
    # Get response
    response = JobHandler.job_download(org_name, experiment_id, job_id, "experiment", export_type=export_type.name)
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


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>/logs', methods=['GET'])
def experiment_job_logs(org_name, experiment_id, job_id):
    """Get realtime job logs. AutoML train job will return current recommendation's experiment log.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Get Job logs for Experiment
      description: |
        Returns the job logs for a given experiment and job ID. This endpoint:
        - Validates the experiment exists and user has access
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
      - name: experiment_id
        in: path
        description: ID of Experiment
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
          description: Invalid request (e.g. invalid experiment ID, job ID)
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
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = JobHandler.get_job_logs(
        org_name,
        experiment_id,
        job_id,
        "experiment",
        request.args.get('automl_experiment_index', None)
    )
    if response.code == 200:
        response = make_response(response.data, 200)
        response.mimetype = 'text/plain'
        return response
    # Handle errors
    schema = ErrorRsp()
    response = make_response(jsonify(schema.dump(schema.load(response.data))), 400)
    return response


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/inference_microservice/start', methods=['POST'])
@disk_space_check
def inference_microservice_start(org_name, experiment_id):
    """Start a new Inference Microservice and return job_id.

    ---
    post:
      tags:
      - INFERENCE_MICROSERVICE
      summary: Start Inference Microservice
      description: |
        Creates a new Inference Microservice job and starts the StatefulSet. Returns job_id for subsequent operations.
        - Creates a new unique job_id
        - Starts Inference Microservice StatefulSet microservice
        - Returns job_id for file uploads and inference requests
      parameters:
        - name: org_name
          in: path
          required: true
          description: Name of the organization
          schema:
            type: string
        - name: experiment_id
          in: path
          required: true
          description: Experiment ID
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                model_path:
                  type: string
                  description: Path to the model
                  example: "/workspace/model"
                docker_image:
                  type: string
                  description: Docker image for inference
                  example: "nvcr.io/nvidia/vila-inference:latest"
                gpu_type:
                  type: string
                  description: GPU type required
                  example: "H100"
                num_gpus:
                  type: integer
                  description: Number of GPUs required
                  example: 1
              required:
                - model_path
      responses:
        200:
          description: Inference Microservice started successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id:
                    type: string
                    description: Unique job ID for this Inference Microservice
                  status:
                    type: string
                    description: Service status
                  message:
                    type: string
                    description: Success message
        400:
          description: Bad request
        500:
          description: Internal server error
    """
    try:
        request_data = request.get_json(force=True)

        # Generate unique job_id
        job_id = str(uuid.uuid4())

        # Create job configuration
        response = InferenceMicroserviceHandler.start_inference_microservice(
            org_name, experiment_id, job_id, request_data
        )

        if response.code == 200:
            return make_response(jsonify({
                'job_id': job_id,
                'status': 'starting',
                'message': f'Inference Microservice started with job_id: {job_id}'
            }), 200)
        schema = ErrorRsp()
        metadata = {"error_desc": response.data['error_desc'], "error_code": response.data['error_code']}
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 500)

    except Exception as err:
        logger.error("Error in inference_microservice_start: %s", str(traceback.format_exc()))
        return make_response(jsonify({
            'error': str(err),
            'error_code': 1
        }), 500)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>/inference_microservice/inference',
                         methods=['POST'])
@disk_space_check
def inference_microservice_inference(org_name, experiment_id, job_id):
    """Make an inference request to a running Inference Microservice.

    ---
    post:
      tags:
      - INFERENCE_MICROSERVICE
      summary: Make Inference Microservice inference request
      description: |
        Sends inference request to a running Inference Microservice. This endpoint:
        - Validates the Inference Microservice is running
        - Processes prompts and images for Inference Microservice inference
        - Returns generated content from the Inference Microservice model
        - Supports both single and batch inference requests
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: Experiment ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Inference Microservice Job ID
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
                input:
                  type: array
                  items:
                    type: string
                  description: Base64-encoded images/videos with data URI format (data:image/jpeg;base64,...)
                model:
                  type: string
                  description: Model identifier (e.g. nvidia/nvdino-v2)
                prompt:
                  type: string
                  description: Text prompt for Inference Microservice inference
                  default: ""
              required: [input, model]
      responses:
        200:
          description: Inference completed successfully
        400:
          description: Invalid request parameters
        404:
          description: Inference Microservice not found
        500:
          description: Inference request failed
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response

    try:
        request_data = request.get_json()

        if not request_data:
            metadata = {"error_desc": "Input data is required", "error_code": 1}
            schema = ErrorRsp()
            return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

        # Process Inference Microservice inference via direct StatefulSet call
        result = InferenceMicroserviceHandler.process_inference_microservice_request_direct(job_id, request_data)

        return make_response(jsonify(result), 200 if result.get("status") != "error" else 500)

    except Exception as e:
        logger.error("Error processing Inference Microservice inference request: %s", str(e))
        metadata = {"error_desc": str(e), "error_code": 1}
        schema = ErrorRsp()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 500)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>/inference_microservice/status',
                         methods=['GET'])
@disk_space_check
def inference_microservice_status(org_name, experiment_id, job_id):  # noqa: D214
    """Get status of a Inference Microservice.

    ---
    get:
      tags:
      - INFERENCE_MICROSERVICE
      summary: Get Inference Microservice status
      description: |
        Returns the status of a Inference Microservice.
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: Experiment ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Inference Microservice Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Service status retrieved successfully
        404:
          description: Inference Microservice not found
        500:
          description: Failed to get service status
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response

    try:
        # Get Inference Microservice service status directly
        result = InferenceMicroserviceHandler.get_inference_microservice_status_direct(job_id)
        return make_response(jsonify(result), 200 if result.get("status") != "error" else 500)

    except Exception as e:
        logger.error("Error getting Inference Microservice status: %s", str(e))
        metadata = {"error_desc": str(e), "error_code": 1}
        schema = ErrorRsp()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 500)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>/inference_microservice/stop',
                         methods=['POST'])
@disk_space_check
def stop_inference_microservice(org_name, experiment_id, job_id):  # noqa: D214
    """Stop a Inference Microservice.

    ---
    post:
      tags:
      - INFERENCE_MICROSERVICE
      summary: Stop Inference Microservice
      description: |
        Stops a running Inference Microservice.
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: Experiment ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Inference Microservice Job ID to stop
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Inference Microservice stopped successfully
        400:
          description: Invalid request parameters
        404:
          description: Inference Microservice not found
        500:
          description: Failed to stop service
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response

    try:
        # Stop the Inference Microservice
        result = InferenceMicroserviceHandler.stop_inference_microservice(job_id)

        # Update job status if stopped successfully
        if result.code == 200:
            from nvidia_tao_core.microservices.utils.stateless_handler_utils import update_job_status
            update_job_status(
                experiment_id,
                job_id,
                status="Done",
                kind="experiments"
            )

        return make_response(jsonify(result.data), result.code)

    except Exception as e:
        logger.error("Error stopping Inference Microservice: %s", str(e))
        metadata = {"error_desc": str(e), "error_code": 1}
        schema = ErrorRsp()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 500)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs', methods=['DELETE'])
@disk_space_check
def bulk_experiment_job_delete(org_name, experiment_id):
    """Bulk Delete Experiment Jobs.

    ---
    delete:
      tags:
      - EXPERIMENT
      summary: Delete multiple Experiment Jobs
      description: |
        Deletes multiple jobs within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the jobs exist and are deletable
        - Deletes the job files and metadata
        - Updates job status to 'deleted'
        - Returns the deletion status for each job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
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
          description: Invalid request (e.g. invalid experiment ID, job IDs)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment, or Jobs not found
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
        response = JobHandler.job_delete(org_name, experiment_id, job_id, "experiment")
        if response.code == 200:
            results.append({"id": job_id, "status": "success"})
        else:
            results.append({"id": job_id, "status": "failed", "error_desc": response.data.get("error_desc", "")})

    # Return status for all jobs
    schema = BulkOpsRsp()
    schema_dict = schema.dump(schema.load({"results": results}))
    return make_response(jsonify(schema_dict), 200)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>/schema', methods=['GET'])
@disk_space_check
def experiment_job_schema(org_name, experiment_id, job_id):
    """Retrieve Schema for a job.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve Schema for a job
      description: |
        Returns the Specs schema for a given job. This endpoint:
        - Validates the experiment exists and user has access
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
      - name: experiment_id
        in: path
        description: ID for Experiment
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
          description: Invalid request (e.g. invalid experiment ID, job ID)
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
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = JobHandler.get_spec_schema_for_job(user_id, org_name, experiment_id, job_id, "experiment")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:get_epoch_numbers',
                         methods=['GET'])
@disk_space_check
def experiment_job_get_epoch_numbers(org_name, experiment_id, job_id):
    """Get the epoch numbers for the checkpoints present for this job.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Get epoch numbers present for this job
      description: |
        Retrieves the epoch numbers for the checkpoints present for this job. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists
        - Retrieves the list of epoch numbers from storage
        - Returns the list of epoch numbers
      parameters:
      - name: org_name
        in: path
        description: Organization name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
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
          description: List of epoch numbers
        content:
          application/json:
              schema: LstInt
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Experiment or Job not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = JobHandler.job_get_epoch_numbers(user_id, org_name, experiment_id, job_id, "experiment")
    # Get schema
    schema_dict = None
    if response.code == 200:
        schema = LstInt()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:remove_published_model',
                         methods=['DELETE'])
@disk_space_check
def experiment_remove_published_model(org_name, experiment_id, job_id):
    """Remove published models from NGC.

    ---
    delete:
      tags:
      - EXPERIMENT
      summary: Remove publish models from NGC
      description: |
        Removes models from NGC private registry. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists and is publishable
        - Validates the provided metadata matches the schema
        - Removes the model from NGC
        - Returns a success message
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
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
          description: Invalid request (e.g. invalid experiment ID, job ID, missing required fields)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    if os.getenv("AIRGAPPED_MODE", "false").lower() == "true":
        schema = MessageOnly()
        message = "Cannot remove published model in air-gapped mode."
        response = make_response(jsonify(schema.dump({"message": message})), 400)
        return response

    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.args.to_dict()
    schema = PublishModel()
    request_schema_data = schema.dump(schema.load(request_data))
    team_name = request_schema_data.get('team_name', '')
    # Get response
    response = ModelHandler.remove_published_model(org_name, team_name, experiment_id, job_id)
    # Get schema
    schema_dict = None

    if response.code == 200:
        schema = MessageOnly()
        logger.info("Returning success response")
        schema_dict = schema.dump({"message": "Removed model"})
    else:
        schema = ErrorRsp()
        # Load metadata in schema and return
        schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:list_files', methods=['GET'])
@disk_space_check
def experiment_job_files_list(org_name, experiment_id, job_id):
    """List Job Files.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: List Job Files
      description: |
        Lists the files produced by a given job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
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
      - name: experiment_id
        in: path
        description: ID of Experiment
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
          description: Invalid request (e.g. invalid experiment ID, job ID)
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
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = JobHandler.job_list_files(org_name, experiment_id, job_id, "experiment")
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


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:download_selective_files',
                         methods=['GET'])
@disk_space_check
def experiment_job_download_selective_files(org_name, experiment_id, job_id):
    """Download selective Job Artifacts.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Download selective Job Artifacts
      description: |
        Downloads selective artifacts produced by a given job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
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
      - name: experiment_id
        in: path
        description: ID of Experiment
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
          description: Invalid request (e.g. invalid experiment ID, job ID, file list)
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
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    file_lists = request.args.getlist('file_lists')
    best_model = ast.literal_eval(request.args.get('best_model', "False"))
    latest_model = ast.literal_eval(request.args.get('latest_model', "False"))
    tar_files = ast.literal_eval(request.args.get('tar_files', "True"))
    if not (file_lists or best_model or latest_model):
        return make_response(
            jsonify("No files passed in list format to download or, best_model or latest_model is not enabled"),
            400
        )
    # Get response
    response = JobHandler.job_download(
        org_name,
        experiment_id,
        job_id,
        "experiment",
        file_lists=file_lists,
        best_model=best_model,
        latest_model=latest_model,
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


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:log_update', methods=['POST'])
@disk_space_check
def experiment_job_log_update(org_name, experiment_id, job_id):
    """Update Job log for Experiment.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Update log of an experiment job
      description: |
        Updates the log of a specific job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists
        - Updates the job log based on provided data
      parameters:
      - name: org_name
        in: path
        description: Organization name owning the experiment
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: Unique identifier of the experiment containing the job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Unique identifier of the job to update
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
              description: Log update data including
      responses:
        200:
          description: Job logs successfully updated
          content:
            application/json:
              schema: ExperimentJob
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid log update request (e.g. invalid log value, missing required fields)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Experiment or job not found, or user lacks permission to update
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    callback_data = request.json
    # Get response
    response = JobHandler.job_log_update(org_name, experiment_id, job_id, "experiment", callback_data=callback_data)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRsp()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:events', methods=['GET'])
def experiment_job_events(org_name, experiment_id, job_id):
    """Get all job status events.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Get all job status events
      description: |
        Returns all status event lines for a given job. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists
        - Retrieves all status events from storage
        - Returns the complete event history
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID of Experiment
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
          description: Invalid request (e.g. invalid experiment ID, job ID)
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
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = JobHandler.get_job_events(
        org_name,
        experiment_id,
        job_id,
        "experiment",
        request.args.get('automl_experiment_number', "0")
    )
    if response.code == 200:
        schema = JobEventsRsp()
        response_data = {
            "job_id": job_id,
            "events": response.data
        }
        return make_response(jsonify(schema.dump(schema.load(response_data))), 200)
    # Handle errors
    schema = ErrorRsp()
    response = make_response(jsonify(schema.dump(schema.load(response.data))), response.code)
    return response


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:status_update', methods=['POST'])
@disk_space_check
def experiment_job_status_update(org_name, experiment_id, job_id):
    """Update Job status for Experiment.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Update status of an experiment job
      description: |
        Updates the status of a specific job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists
        - Updates the job status based on provided data
        - Persists status changes to storage
        - Triggers any necessary status-based workflows
      parameters:
      - name: org_name
        in: path
        description: Organization name owning the experiment
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: Unique identifier of the experiment containing the job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Unique identifier of the job to update
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
              schema: ExperimentJob
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
          description: Experiment or job not found, or user lacks permission to update
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    callback_data = request.json
    # Get response
    response = JobHandler.job_status_update(org_name, experiment_id, job_id, "experiment", callback_data=callback_data)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRsp()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>:cancel_all_jobs', methods=['POST'])
@disk_space_check
def experiment_jobs_cancel(org_name, experiment_id):
    """Cancel all jobs within experiment (or pause training).

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Cancel all Jobs under experiment
      description: |
        Cancels all jobs within an experiment. This endpoint:
        - Validates the experiment exists and user has access
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
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully canceled all jobs under experiments
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID)
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
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = JobHandler.all_job_cancel(user_id, org_name, experiment_id, "experiment")
    # Get schema
    if response.code == 200:
        schema = MessageOnly()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@experiments_bp_v1.route('/orgs/<org_name>/experiments:cancel_all_jobs', methods=['POST'])
@disk_space_check
def bulk_experiment_jobs_cancel(org_name):
    """Cancel all jobs within multiple experiments.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Cancel all Jobs under multiple experiments
      description: |
        Cancels all jobs within multiple experiments. This endpoint:
        - Validates the experiments exist and user has access
        - Validates the jobs exist and are cancellable
        - Updates the job status to 'cancelled'
        - Persists status changes to storage
        - Triggers any necessary cancellation workflows
        - Returns the cancellation status for each experiment
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
                experiment_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                    maxLength: 36
                  maxItems: 2147483647
      responses:
        200:
          description: Successfully canceled all jobs under the specified experiments
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment IDs)
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiments or Jobs not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get experiment IDs from the request body
    data = request.get_json()
    experiment_ids = data.get('experiment_ids')

    if not experiment_ids or not isinstance(experiment_ids, list):
        metadata = {"error_desc": "Invalid experiment IDs", "error_code": 1}
        schema = ErrorRsp()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

    results = []
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)

    for experiment_id in experiment_ids:
        message = validate_uuid(experiment_id=experiment_id)
        if message:
            results.append({"id": experiment_id, "error_desc": message, "error_code": 1})
            continue

        # Cancel all jobs for each experiment
        response = JobHandler.all_job_cancel(user_id, org_name, experiment_id, "experiment")
        if response.code == 200:
            results.append({"id": experiment_id, "status": "success"})
        else:
            results.append({"id": experiment_id, "status": "failed", "error_desc": response.data.get("error_desc", "")})

    # Return status for each experiment's job cancellation
    schema = BulkOpsRsp()
    schema_dict = schema.dump(schema.load({"results": results}))
    return make_response(jsonify(schema_dict), 200)
