# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admin blueprint for v2 API - documentation, metrics, and utility endpoints."""

import os
import bson
import shutil
import logging
from flask import Blueprint, jsonify, make_response, render_template, send_file
from flask import current_app, request

from nvidia_tao_core.microservices.decorators import disk_space_check
from nvidia_tao_core.telemetry.processor import MetricProcessor
from .schemas import ErrorRsp, TelemetryReq
from nvidia_tao_core.microservices.config import get_tao_version
from nvidia_tao_core.microservices.utils.stateless_handler_utils import set_metrics, get_metrics, get_root
from nvidia_tao_core.microservices.utils.core_utils import safe_load_file

TIMEOUT = 240
logger = logging.getLogger(__name__)

# v2 Admin Blueprint - URL prefix will be set during registration (/api/v2)
admin_bp_v2 = Blueprint('admin_v2', __name__, template_folder='templates')

tao_version = get_tao_version()


@admin_bp_v2.route('/', methods=['GET'])
def version_v2():
    """api v2 root endpoint"""
    return make_response(jsonify([
        'login',
        'auth',
        'health',
        'openapi.yaml',
        'openapi.json',
        'rapipdf',
        'redoc',
        'swagger',
        'version',
        'tao_api_notebooks.zip',
        'orgs',
        'metrics'
    ]))


@admin_bp_v2.route('/orgs', methods=['GET'])
def user_list():
    """user list endpoint"""
    error = {"error_desc": "Listing orgs is not authorized: Missing Org Name", "error_code": 1}
    schema = ErrorRsp()
    return make_response(jsonify(schema.dump(schema.load(error))), 403)


@admin_bp_v2.route('/orgs/<org_name>', methods=['GET'])
@disk_space_check
def user(org_name):
    """user endpoint"""
    return make_response(jsonify(['dataset', 'experiment']))


@admin_bp_v2.route('/openapi.yaml', methods=['GET'])
def openapi_yaml():
    """openapi_yaml endpoint"""
    spec = current_app.spec_v2
    r = make_response(spec.to_yaml())
    r.mimetype = 'text/x-yaml'
    return r


@admin_bp_v2.route('/openapi.json', methods=['GET'])
def openapi_json():
    """openapi_json endpoint"""
    spec = current_app.spec_v2
    r = make_response(jsonify(spec.to_dict()))
    r.mimetype = 'application/json'
    return r


@admin_bp_v2.route('/rapipdf', methods=['GET'])
def rapipdf():
    """rapipdf endpoint"""
    return render_template('v2/rapipdf.html')


@admin_bp_v2.route('/redoc', methods=['GET'])
def redoc():
    """redoc endpoint"""
    return render_template('v2/redoc.html')


@admin_bp_v2.route('/swagger', methods=['GET'])
def swagger():
    """swagger endpoint"""
    logger.info(
        f"Received swagger request for v2 API, root_path={admin_bp_v2.root_path}, "
        f"template_folder={admin_bp_v2.template_folder}")
    return render_template('v2/swagger.html')


@admin_bp_v2.route('/version', methods=['GET'])
def version():
    """version endpoint"""
    git_branch = os.environ.get('GIT_BRANCH', 'unknown')
    git_commit_sha = os.environ.get('GIT_COMMIT_SHA', 'unknown')
    git_commit_time = os.environ.get('GIT_COMMIT_TIME', 'unknown')
    version = {'version': tao_version, 'branch': git_branch, 'sha': git_commit_sha, 'time': git_commit_time}
    r = make_response(jsonify(version))
    r.mimetype = 'application/json'
    return r


@admin_bp_v2.route('/tao_api_notebooks.zip', methods=['GET'])
@disk_space_check
def download_folder():
    """Download notebooks endpoint"""
    # Create a temporary zip file containing the folder
    shutil.make_archive("/tmp/tao_api_notebooks", 'zip', "/shared/notebooks/")

    # Return the zip file as a downloadable attachment
    return send_file("/tmp/tao_api_notebooks.zip", as_attachment=True,
                     download_name="tao_api_notebooks.zip", mimetype='application/zip')


@admin_bp_v2.route('/metrics', methods=['POST'])
def metrics_upsert():
    """Report execution of new action."""
    # Validate and load telemetry data
    try:
        raw_data = TelemetryReq().load(request.get_json(force=True))
    except Exception as e:
        logger.error("Exception thrown in metrics_upsert: %s", str(e))
        return make_response(jsonify({}), 400)

    # Load existing metrics
    metrics = get_metrics()
    if not metrics:
        metrics = safe_load_file(os.path.join(get_root(), 'metrics.json'))
        if not metrics:
            # Warning: No historical metrics data found, starting with new record.
            logger.warning("No existing metrics history found; starting new metrics record.")
            metrics = {}  # Start a new, empty metrics dict

    # Process metrics using the extensible MetricProcessor
    # This orchestrator handles all metric building using configured builders
    processor = MetricProcessor()
    metrics = processor.process(metrics, raw_data)

    # Persist metrics
    set_metrics(metrics)

    return make_response(bson.json_util.dumps(metrics), 201)
