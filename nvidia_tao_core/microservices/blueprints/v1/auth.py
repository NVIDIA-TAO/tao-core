# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authentication blueprint for API v1 login and auth endpoints."""

import os
import uuid
import logging
from flask import Blueprint, request, jsonify, make_response

from nvidia_tao_core.microservices.decorators import disk_space_check
from .schemas import LoginReq, LoginRsp, ErrorRsp
from nvidia_tao_core.microservices.utils.auth_utils import credentials, authentication, access_control

try:
    from nvidia_tao_core.microservices.utils.auth_utils import metrics
except ImportError:
    metrics = None
from nvidia_tao_core.microservices.utils.mongo_utils import MongoHandler
from nvidia_tao_core.microservices.constants import AIRGAP_DEFAULT_USER
from nvidia_tao_core.microservices.utils.core_utils import log_monitor, DataMonitorLogTypeEnum

logger = logging.getLogger(__name__)

# v1 Authentication Blueprint - URL prefix will be set during registration
auth_bp_v1 = Blueprint('auth_v1', __name__, template_folder='templates')


@auth_bp_v1.route('/login', methods=['POST'])
@disk_space_check
def login():
    """User Login or Exchange username for user_id for air-gapped mode.

    ---
    post:
      tags:
      - AUTHENTICATION
      summary: Authenticate user with NGC credentials and set telemetry preferences
      description: |
        Authenticates a user using their NGC API key and organization name.
        Returns JWT token and user credentials upon successful authentication.
        The token can be used for subsequent API requests.
      security:
        - api-key: []
      requestBody:
        content:
          application/json:
            schema: LoginReq
        description: |
          Login credentials including:
          - ngc_key: NGC API key for authentication
          - ngc_org_name: Organization name in NGC
          - enable_telemetry: Optional telemetry preference (default: False)
        required: true
      responses:
        200:
          description: Successfully authenticated. Returns user credentials and JWT token.
          content:
            application/json:
              schema: LoginRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        401:
          description: Authentication failed due to invalid credentials or permissions
          content:
            application/json:
              schema: ErrorRsp
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # air-gapped mode skips NGC authentication and return user_id directly
    if os.getenv("AIRGAPPED_MODE", "false").lower() == "true":
        try:
            # Get username from request or use default
            request_data = request.get_json(force=True)
            username = request_data.get("username", AIRGAP_DEFAULT_USER)

            # Create UUID mapping of username
            user_id = str(uuid.uuid5(uuid.UUID(int=0), username))

            # Store user mapping in MongoDB for consistency
            mongo = MongoHandler("tao", "users")
            mongo.upsert({"id": user_id}, {"user_name": username, "id": user_id})

            logger.info("Airgapped mode login: Mapped username '%s' to user_id '%s'", username, user_id)
            return make_response(jsonify({"user_id": user_id}), 200)
        except Exception as e:
            logger.error("Airgapped mode login failed: %s", str(e))
            metadata = {"error_desc": "Login failed: " + str(e), "error_code": 1}
            schema = ErrorRsp()
            return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

    # Regular NGC authentication flow
    schema = LoginReq()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    key = request_dict.get('ngc_key', 'invalid_key')
    org_name = request_dict.get('ngc_org_name', '')
    enable_telemetry = request_dict.get('enable_telemetry', None)

    creds, err = credentials.get_from_ngc(key, org_name, enable_telemetry)
    if err:
        logger.warning("Unauthorized: %s", err)
        metadata = {"error_desc": "Unauthorized: " + err, "error_code": 1}
        schema = ErrorRsp()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 401)
    schema = LoginRsp()
    schema_dict = schema.dump(schema.load(creds))
    return make_response(jsonify(schema_dict), 200)

# Internal endpoint for ingress controller to check authentication


@auth_bp_v1.route('/auth', methods=['GET'])
@disk_space_check
def auth():
    """authentication endpoint"""
    ingress_enabled = os.getenv("INGRESSENABLED", "false") == "true"
    # Skip ngc authentication for air-gapped environments
    if os.getenv("AIRGAPPED_MODE", "false").lower() == "true":
        try:
            # Get user_id from Authorization header
            user_id = None
            auth_header = request.headers.get('Authorization', '')
            if auth_header:
                user_id = auth_header.removeprefix("Bearer ").strip()

            # fall back to anonymous user
            if not user_id:
                user_id = str(uuid.uuid5(uuid.UUID(int=0), AIRGAP_DEFAULT_USER))

            logger.info("Airgapped mode auth with user_id '%s'", user_id)
            return make_response(jsonify({'user_id': user_id}), 200)
        except Exception as e:
            logger.error("Airgapped mode auth failed: %s", str(e))
            return make_response(jsonify({'error': str(e)}), 400)

    # retrieve jwt from headers
    token = ''
    url = request.headers.get('X-Original-Url', '') if ingress_enabled else request.path
    logger.info('URL: %s', url)
    method = request.headers.get('X-Original-Method', '') if ingress_enabled else request.method
    logger.info('Method: %s', method)
    # bypass authentication for http OPTIONS requests
    if method == 'OPTIONS':
        return make_response(jsonify({}), 200)
    # retrieve authorization token
    authorization = request.headers.get('Authorization', '')
    authorization_parts = authorization.split()
    if len(authorization_parts) == 2 and authorization_parts[0].lower() == 'bearer':
        token = authorization_parts[1]

    # if token is not found, try to get it from basic auth for special endpoints
    if not token:
        logger.warning("token cannot be obtained")
        if len(authorization_parts) == 2 and authorization_parts[0].lower() == 'basic':
            basic_auth = request.authorization
            if basic_auth:
                # status callback: service to service authentication
                if basic_auth.username == '$oauthtoken':
                    try:
                        org_name, key = basic_auth.password.split(",")
                    except Exception as e:
                        logger.error("Exception thrown in auth: %s", str(e))
                        metadata = {
                            "error_desc": "Basic auth password not in the format of org_name,ngc_personal_key",
                            "error_code": 1
                        }
                        schema = ErrorRsp()
                        response = make_response(jsonify(schema.dump(schema.load(metadata))), 401)
                        return response
                    creds, err = credentials.get_from_ngc(key, org_name, True)
                    if 'token' in creds:
                        token = creds['token']
                # special metrics case
                elif basic_auth.username == '$metricstoken' and url.split('/', 3)[-1] == 'api/v1/metrics':
                    key = basic_auth.password
                    if metrics is not None and metrics.validate(key):
                        return make_response(jsonify({}), 200)
                    metadata = {"error_desc": "wrong metrics key", "error_code": 1}
                    schema = ErrorRsp()
                    response = make_response(jsonify(schema.dump(schema.load(metadata))), 401)
                    return response

    # if token is still not found, return 401
    if not token:
        schema = ErrorRsp()
        metadata = {"error_desc": "Unauthorized: missing token", "error_code": 1}
        rsp = make_response(jsonify(schema.dump(schema.load(metadata))), 401)
        return rsp

    logger.info('Token: ...%s', token[-10:])
    # authentication
    user_id, org_name, err = authentication.validate(url, token)
    log_content = f"user_id:{user_id}, org_name:{org_name}, method:{method}, url:{url}"
    log_monitor(log_type=DataMonitorLogTypeEnum.api, log_content=log_content)
    if err:
        logger.warning("Unauthorized: %s", err)
        metadata = {"error_desc": str(err), "error_code": 1}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 401)
        return response
    # access control
    err = access_control.validate(user_id, org_name, url, token)
    if err:
        logger.warning("Forbidden: %s", err)
        metadata = {"error_desc": str(err), "error_code": 2}
        schema = ErrorRsp()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 403)
        return response
    return make_response(jsonify({'user_id': user_id}), 200)


@auth_bp_v1.before_request
def authenticate_without_ingress():
    """Authentication endpoint if ingress-nginx is not enabled"""
    ingress_enabled = os.getenv("INGRESSENABLED", "false") == "true"
    skip_api_endpoints = ['/health', '/liveness', '/swagger', '/login', '/auth',
                          '/redoc', '/version', '/rapipdf', '/container_job',
                          '/openapi', '/version', '/tao_api_notebooks']
    if ingress_enabled or any(endpoint in request.path for endpoint in skip_api_endpoints):
        return None
    if "super_endpoint" in request.path:
        request_body = request.get_json(force=True)
        if "container_job" in request_body.get("api_endpoint") or "status_update" in request_body.get("api_endpoint"):
            logger.info("skipping authentication")
            return None
    logger.info("authenticate without ingress, auth being called now for %s", request.path)
    auth_response = auth()
    if auth_response.status_code == 200:
        return None
    logger.warning("authenticate failed")
    return auth_response
