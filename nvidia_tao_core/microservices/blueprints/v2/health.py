# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Health check blueprint for API v2 liveness and readiness endpoints."""

import os
import logging
import traceback
from flask import Blueprint, jsonify, make_response

from nvidia_tao_core.microservices.decorators import disk_space_check
from nvidia_tao_core.microservices.utils.health_utils import health_check
from nvidia_tao_core.microservices.utils.job_utils.workflow import Workflow

logger = logging.getLogger(__name__)

# v2 Health Blueprint - URL prefix will be set during registration
health_bp_v2 = Blueprint('health_v2', __name__, template_folder='templates')


@health_bp_v2.route('/health', methods=['GET'])
def api_health():
    """api health endpoint"""
    return make_response(jsonify(['liveness', 'readiness']))


@health_bp_v2.route('/health/liveness', methods=['GET'])
@disk_space_check
def liveness():
    """api liveness endpoint"""
    try:
        live_state = health_check.check_logging()
        if live_state:
            return make_response(jsonify("OK"), 200)
    except Exception as e:
        logger.error("Exception thrown in liveness: %s", str(e))
        logger.error("liveness error: %s", traceback.format_exc())
    return make_response(jsonify("Error"), 400)


@health_bp_v2.route('/health/readiness', methods=['GET'])
@disk_space_check
def readiness():
    """api readiness endpoint"""
    try:
        if health_check.check_logging():
            ready_state = True
            if os.getenv("BACKEND"):
                if not (health_check.check_k8s() and Workflow.healthy()):
                    ready_state = False
            if ready_state:
                return make_response(jsonify("OK"), 200)
    except Exception as e:
        logger.error("Exception thrown in readiness: %s", str(e))
        logger.error("readiness error: %s", traceback.format_exc())
    return make_response(jsonify("Error"), 400)
