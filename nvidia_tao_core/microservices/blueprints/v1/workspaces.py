# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workspaces blueprint for API v1 workspace management endpoints."""

import math
import logging
from flask import Blueprint, request, jsonify, make_response

from nvidia_tao_core.microservices.decorators import disk_space_check
from nvidia_tao_core.microservices.utils.auth_utils import authentication
from nvidia_tao_core.microservices.utils.filter_utils import filtering, pagination
from nvidia_tao_core.microservices.handlers.workspace_handler import WorkspaceHandler
from nvidia_tao_core.microservices.handlers import MongoBackupHandler
from .schemas import (
    WorkspaceListRsp,
    WorkspaceRsp,
    ErrorRsp,
    DatasetUriLst,
    MessageOnly,
    BulkOpsRsp,
    WorkspaceReq,
    WorkspaceBackupReq
)
from nvidia_tao_core.microservices.utils.handler_utils import validate_uuid

logger = logging.getLogger(__name__)

# v1 Workspaces Blueprint - URL prefix will be set during registration
workspaces_bp_v1 = Blueprint('workspaces_v1', __name__, template_folder='templates')


@workspaces_bp_v1.route('/orgs/<org_name>/workspaces', methods=['GET'])
@disk_space_check
def workspace_list(org_name):
    """List workspaces.

    ---
    get:
      tags:
      - WORKSPACE
      summary: List workspaces
      description: Returns the list of workspaces
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
      - name: format
        in: query
        description: Optional format filter
        required: false
        schema:
          type: string
          enum: ["unet", "custom"]
      - name: type
        in: query
        description: Optional type filter
        required: false
        schema:
          type: string
          enum: [ "object_detection", "segmentation", "image_classification" ]
      responses:
        200:
          description: Returned list of workspaces
          content:
            application/json:
              schema:
                type: array
                items: WorkspaceRsp
                maxItems: 2147483647
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    workspaces = WorkspaceHandler.list_workspaces(user_id, org_name)
    filtered_workspaces = filtering.apply(request.args, workspaces)
    paginated_workspaces = pagination.apply(request.args, filtered_workspaces)
    metadata = {"workspaces": paginated_workspaces}
    # Pagination
    skip = request.args.get("skip", None)
    size = request.args.get("size", None)
    if skip is not None and size is not None:
        skip = int(skip)
        size = int(size)
        metadata["pagination_info"] = {
            "total_records": len(filtered_workspaces),
            "total_pages": math.ceil(len(filtered_workspaces) / size),
            "page_size": size,
            "page_index": skip // size,
        }
    schema = WorkspaceListRsp()
    response = make_response(jsonify(schema.dump(schema.load(metadata))))
    return response


@workspaces_bp_v1.route('/orgs/<org_name>/workspaces/<workspace_id>', methods=['GET'])
@disk_space_check
def workspace_retrieve(org_name, workspace_id):
    """Retrieve Workspace.

    ---
    get:
      tags:
      - WORKSPACE
      summary: Retrieve Workspace
      description: Returns the Workspace
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: workspace_id
        in: path
        description: ID of Workspace to return
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Workspace
          content:
            application/json:
              schema: WorkspaceRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Workspace not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(workspace_id=workspace_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = WorkspaceHandler.retrieve_workspace(user_id, org_name, workspace_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = WorkspaceRsp()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@workspaces_bp_v1.route('/orgs/<org_name>/workspaces/<workspace_id>:get_datasets', methods=['GET'])
@disk_space_check
def workspace_retrieve_datasets(org_name, workspace_id):
    """Retrieve Datasets from Workspace.

    ---
    get:
      tags:
      - WORKSPACE
      summary: Retrieve datasets from Workspace
      description: Returns the datasets matched with the request body args inside the Workspace
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: workspace_id
        in: path
        description: ID of Workspace to return
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned list of dataset paths within Workspace
          content:
            application/json:
              schema: DatasetUriLst
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Workspace not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(workspace_id=workspace_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    dataset_type = request.args.get("dataset_type", None)
    dataset_format = request.args.get("dataset_format", None)
    dataset_intention = request.args.getlist("dataset_intention")
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = WorkspaceHandler.retrieve_cloud_datasets(
        user_id,
        org_name,
        workspace_id,
        dataset_type,
        dataset_format,
        dataset_intention
    )
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetUriLst()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@workspaces_bp_v1.route('/orgs/<org_name>/workspaces/<workspace_id>', methods=['DELETE'])
@disk_space_check
def workspace_delete(org_name, workspace_id):
    """Delete Workspace.

    ---
    delete:
      tags:
      - WORKSPACE
      summary: Delete Workspace
      description: Cancels all related running jobs and returns the deleted Workspace
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: workspace_id
        in: path
        description: ID of Workspace to delete
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Deleted Workspace
          content:
            application/json:
              schema: WorkspaceRsp
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
          description: User or Workspace not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(workspace_id=workspace_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = WorkspaceHandler.delete_workspace(org_name, workspace_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = MessageOnly()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@workspaces_bp_v1.route('/orgs/<org_name>/workspaces', methods=['DELETE'])
@disk_space_check
def bulk_workspace_delete(org_name):
    """Bulk Delete Workspaces.

    ---
    delete:
      tags:
      - WORKSPACE
      summary: Delete multiple Workspaces
      description: Cancels all related running jobs and returns the status of deleted Workspaces
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
                workspace_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                    maxLength: 36
                  maxItems: 2147483647
      responses:
        200:
          description: Deleted Workspaces status
          content:
            application/json:
              schema: WorkspaceRsp
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
          description: One or more Workspaces not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get workspace IDs from request body
    data = request.get_json()
    workspace_ids = data.get('workspace_ids')

    if not workspace_ids or not isinstance(workspace_ids, list):
        metadata = {"error_desc": "Invalid workspace IDs", "error_code": 1}
        schema = ErrorRsp()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

    results = []
    for workspace_id in workspace_ids:
        message = validate_uuid(workspace_id=workspace_id)
        if message:
            metadata = {"id": workspace_id, "error_desc": message, "error_code": 1}
            results.append(metadata)
            continue

        # Attempt to delete the workspace
        response = WorkspaceHandler.delete_workspace(org_name, workspace_id)
        if response.code == 200:
            results.append({"id": workspace_id, "status": "success"})
        else:
            results.append({"id": workspace_id, "status": "failed", "error_desc": response.data.get("error_desc", "")})

    # Return status for all workspaces
    schema = BulkOpsRsp()
    schema_dict = schema.dump(schema.load({"results": results}))
    return make_response(jsonify(schema_dict), 200)


@workspaces_bp_v1.route('/orgs/<org_name>/workspaces', methods=['POST'])
@disk_space_check
def workspace_create(org_name):
    """Create new Workspace.

    ---
    post:
      tags:
      - WORKSPACE
      summary: Create new Workspace
      description: Returns the new Workspace
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
            schema: WorkspaceReq
            examples:
              aws_example:
                summary: Example with AWS cloud details
                value:
                  cloud_details:
                    cloud_type: aws
                    cloud_specific_details:
                      access_key: my_access_key
                      secret_key: my_secret_key
                      cloud_region: us-west-1
                      cloud_bucket_name: my_bucket_name
              azure_example:
                summary: Example with Azure cloud details
                value:
                  cloud_details:
                    cloud_type: azure
                    cloud_specific_details:
                      access_key: my_access_key
                      account_name: my_account_name
                      cloud_bucket_name: my_container_name
              huggingface_example:
                summary: Example with Hugging Face cloud details
                value:
                  cloud_details:
                    cloud_type: huggingface
                    cloud_specific_details:
                      token: my_token
        description: Initial metadata for new Workspace (type and format required)
        required: true
      responses:
        200:
          description: Retuned the new Workspace
          content:
            application/json:
              schema: WorkspaceRsp
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
    schema = WorkspaceReq()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))

    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    # Get response
    response = WorkspaceHandler.create_workspace(user_id, org_name, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = WorkspaceRsp()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@workspaces_bp_v1.route('/orgs/<org_name>/workspaces/<workspace_id>', methods=['PUT'])
@disk_space_check
def workspace_update(org_name, workspace_id):
    """Update Workspace.

    ---
    put:
      tags:
      - WORKSPACE
      summary: Update Workspace
      description: Returns the updated Workspace
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: workspace_id
        in: path
        description: ID of Workspace to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: WorkspaceReq
        description: Updated metadata for Workspace
        required: true
      responses:
        200:
          description: Returned the updated Workspace
          content:
            application/json:
              schema: WorkspaceRsp
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
          description: User or Workspace not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(workspace_id=workspace_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = WorkspaceReq()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = WorkspaceHandler.update_workspace(user_id, org_name, workspace_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = WorkspaceRsp()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@workspaces_bp_v1.route('/orgs/<org_name>/workspaces/<workspace_id>', methods=['PATCH'])
@disk_space_check
def workspace_partial_update(org_name, workspace_id):
    """Partial update Workspace.

    ---
    patch:
      tags:
      - WORKSPACE
      summary: Partial update Workspace
      description: Returns the updated Workspace
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: workspace_id
        in: path
        description: ID of Workspace to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: WorkspaceReq
        description: Updated metadata for Workspace
        required: true
      responses:
        200:
          description: Returned the updated Workspace
          content:
            application/json:
              schema: WorkspaceRsp
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
          description: User or Workspace not found
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(workspace_id=workspace_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = WorkspaceRsp()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = WorkspaceHandler.update_workspace(user_id, org_name, workspace_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = WorkspaceRsp()
    else:
        schema = ErrorRsp()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@workspaces_bp_v1.route('/orgs/<org_name>/workspaces:backup', methods=['POST'])
@disk_space_check
def workspace_backup(org_name):
    """Backup MongoDB data using workspace metadata.

    ---
    post:
      tags:
      - WORKSPACES
      summary: Backup MongoDB data using workspace metadata
      description: Backs up all MongoDB databases using provided workspace cloud credentials
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
                workspace_metadata:
                  type: object
                  description: Workspace metadata containing cloud credentials
                  required: true
                backup_file_name:
                  type: string
                  description: Optional backup file name
              required:
                - workspace_metadata
      responses:
        200:
          description: Backup successful
          content:
            application/json:
              schema: MessageOnly
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    try:
        request_data = request.get_json(force=True)
        schema = WorkspaceBackupReq()
        validated_request = schema.dump(schema.load(request_data))

        workspace_metadata = validated_request.get("workspace_metadata")
        backup_file_name = validated_request.get("backup_file_name", "mongodb_backup.tar.gz")

        if not workspace_metadata:
            metadata = {"error_desc": "workspace_metadata is required", "error_code": 1}
            schema = ErrorRsp()
            response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
            return response

        # Perform backup
        response = MongoBackupHandler.backup_for_workspace(workspace_metadata, backup_file_name)

        # Return response
        if response.code == 200:
            schema = MessageOnly()
        else:
            schema = ErrorRsp()

        schema_dict = schema.dump(schema.load(response.data))
        return make_response(jsonify(schema_dict), response.code)

    except Exception as e:
        metadata = {"error_desc": f"Error in MongoDB backup: {str(e)}", "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response


@workspaces_bp_v1.route('/orgs/<org_name>/workspaces:restore', methods=['POST'])
@disk_space_check
def workspace_restore(org_name):
    """Restore MongoDB data using workspace metadata.

    ---
    post:
      tags:
      - WORKSPACES
      summary: Restore MongoDB data using workspace metadata
      description: Restores all MongoDB databases using provided workspace cloud credentials
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
                workspace_metadata:
                  type: object
                  description: Workspace metadata containing cloud credentials
                  required: true
                backup_file_name:
                  type: string
                  description: Optional backup file name to restore from
              required:
                - workspace_metadata
      responses:
        200:
          description: Restore successful
          content:
            application/json:
              schema: MessageOnly
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    try:
        request_data = request.get_json(force=True)
        schema = WorkspaceBackupReq()
        validated_request = schema.dump(schema.load(request_data))

        workspace_metadata = validated_request.get("workspace_metadata")
        backup_file_name = validated_request.get("backup_file_name", "mongodb_backup.tar.gz")

        if not workspace_metadata:
            metadata = {"error_desc": "workspace_metadata is required", "error_code": 1}
            schema = ErrorRsp()
            response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
            return response

        # Perform restore
        response = MongoBackupHandler.restore_for_workspace(workspace_metadata, backup_file_name)

        # Return response
        if response.code == 200:
            schema = MessageOnly()
        else:
            schema = ErrorRsp()

        schema_dict = schema.dump(schema.load(response.data))
        return make_response(jsonify(schema_dict), response.code)

    except Exception as e:
        metadata = {"error_desc": f"Error in MongoDB restore: {str(e)}", "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
