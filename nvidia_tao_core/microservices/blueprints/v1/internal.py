# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal blueprint for API v1 container jobs and internal endpoints."""

import uuid
import logging
import traceback
from flask import Blueprint, request, jsonify, make_response

from nvidia_tao_core.microservices.decorators import disk_space_check
from nvidia_tao_core.microservices.utils.auth_utils import authentication
from .schemas import ContainerJob, ContainerJobStatus, ErrorRsp, GpuDetails
from nvidia_tao_core.microservices.utils.handler_utils import send_statefulset_request
from nvidia_tao_core.microservices.handlers.spec_handler import SpecHandler

from nvidia_tao_core.microservices.handlers.container_handler import ContainerJobHandler as container_handler

logger = logging.getLogger(__name__)

# v1 Internal Blueprint - URL prefix will be set during registration
internal_bp_v1 = Blueprint('internal_v1', __name__, template_folder='templates')


@internal_bp_v1.route('/internal/container_job', methods=['POST'])
@disk_space_check
def container_job_run():
    """Run Job within container.

    ---
    post:
      tags:
        - INTERNAL
      summary: Run Container Job
      description:
        Starts a job within a container asynchronously and returns immediately with job ID.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ContainerJob'
      responses:
        200:
          description: The container job was successfully launched.
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id:
                    type: string
                    description: The ID of the launched job
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request payload or job execution failed.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorRsp'
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or dataset not found.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorRsp'
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        500:
          description: Internal server error encountered while processing the job.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorRsp'
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    input_schema = ContainerJob()
    job_dict = input_schema.dump(input_schema.load(request.get_json(force=True)))
    try:
        if "job_id" not in job_dict:
            job_dict["job_id"] = str(uuid.uuid4())

        statefulset_replicas = job_dict.get("statefulset_replicas")
        if statefulset_replicas:  # proxy requests to all replicas from master
            for replica_index in range(1, statefulset_replicas):
                send_statefulset_request(
                    api_endpoint="post_action",
                    network=job_dict["neural_network_name"],
                    action=job_dict["action_name"],
                    cloud_metadata=job_dict["cloud_metadata"],
                    specs=job_dict["specs"],
                    docker_env_vars=job_dict["docker_env_vars"],
                    job_id=job_dict["job_id"],
                    statefulset_replica_index=replica_index,
                    statefulset_replicas=statefulset_replicas
                )

        job_id = container_handler.entrypoint_wrapper(job_dict)
        if job_id:
            return make_response(jsonify({'job_id': job_id}), 200)
        metadata = {"error": "Failed to launch job", "error_code": 1}
        schema = ErrorRsp()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)
    except Exception as err:
        logger.error("Error in container_job_run: %s", str(traceback.format_exc()))
        metadata = {"error": str(err), "error_code": 1}
        schema = ErrorRsp()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)


@internal_bp_v1.route('/internal/container_job:status', methods=['GET'])
@disk_space_check
def container_job_status():
    """Get status of job running inside container.

    ---
    get:
      tags:
        - INTERNAL
      summary: Get Status of Container Job
      description:
        Retrieves the current status of a job running inside a container. The response
        includes whether the job is pending, in progress, completed, or failed.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                specs:
                  type: object
                  description: Specification details required to check job status.
      responses:
        200:
          description: Job status retrieved successfully.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/MessageOnly'
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request payload or unable to determine job status.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorRsp'
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or dataset not found.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorRsp'
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        500:
          description: Internal server error encountered while retrieving job status.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorRsp'
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    try:
        response_code = 400
        # For GET requests, data should be in query parameters
        results_dir = request.args.get("results_dir")
        status = container_handler.get_current_job_status(results_dir)
        if status:
            response_code = 200
        schema = ContainerJobStatus()
        schema_dict = schema.dump(schema.load({"status": status}))
        return make_response(jsonify(schema_dict), response_code)
    except Exception as err:
        logger.error("Error in container_job_status: %s", str(traceback.format_exc()))
        metadata = {"error": str(err), "error_code": 1}
        schema = ErrorRsp()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)


@internal_bp_v1.route('/orgs/<org_name>:gpu_types', methods=['GET'])
@disk_space_check
def org_gpu_types(org_name):
    """Retrieve available GPU type.

    ---
    get:
      tags:
      - ORGS
      summary: Retrieve available GPU types based on the backend during deployment
      description: Retrieve available GPU types based on the backend during deployment
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


@internal_bp_v1.route('internal/container_job:pause', methods=['POST'])
@disk_space_check
def container_job_pause():
    """Pause Job within container (graceful termination).

    ---
    post:
      tags:
        - INTERNAL
      summary: Pause Container Job Gracefully
      description:
        Signals a running job within a container to gracefully terminate by writing
        a termination signal file. The job will detect this signal and perform cleanup
        operations including uploading checkpoints before shutting down.
        The results directory is inferred as /results/{job_id}.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - job_id
              properties:
                job_id:
                  type: string
                  description: The ID of the job to pause (results_dir is inferred as /results/{job_id})
      responses:
        200:
          description: The graceful termination signal was successfully written.
          content:
            application/json:
              schema:
                type: object
                properties:
                  message:
                    type: string
                    description: Success message
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request payload or failed to write signal.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorRsp'
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        500:
          description: Internal server error encountered while processing the request.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorRsp'
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    try:
        request_data = request.get_json(force=True)
        job_id = request_data.get("job_id")

        if not job_id:
            metadata = {"error": "job_id is required", "error_code": 1}
            schema = ErrorRsp()
            return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

        # Write graceful termination signal (results_dir is inferred from job_id)
        signal_written = container_handler.write_graceful_termination_signal(job_id)

        if signal_written:
            return make_response(jsonify({'message': f'Graceful termination signal written for job {job_id}'}), 200)

        metadata = {"error": "Failed to write graceful termination signal", "error_code": 1}
        schema = ErrorRsp()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

    except Exception as err:
        logger.error("Error in container_job_pause: %s", str(traceback.format_exc()))
        metadata = {"error": str(err), "error_code": 1}
        schema = ErrorRsp()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 500)
