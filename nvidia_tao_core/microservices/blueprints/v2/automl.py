# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoML Flask Routes - Manage parameter ranges and details"""
import logging
import os
import traceback
from flask import Blueprint, request, jsonify

from .schemas import AutoMLParameterDetailsRsp
from nvidia_tao_core.microservices.automl.params import flatten_properties
from nvidia_tao_core.scripts.generate_schema import generate_schema
from nvidia_tao_core.microservices.utils import get_microservices_network_and_action
from nvidia_tao_core.microservices.enum_constants import ExperimentNetworkArch

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)

# Create blueprint for automl params routes
automl_params_bp_v2 = Blueprint('automl_params_v2', __name__, template_folder='templates')


@automl_params_bp_v2.route('/orgs/<org_name>/automl:get_param_details', methods=['GET'])
def get_automl_param_details(org_name):
    """Get detailed information about AutoML parameters including valid ranges, options, and math conditions.

    Args:
        org_name: Organization name
        Query params:
            network_arch: Network architecture (required)
            parameters: Comma-separated list of parameter names (e.g., ?parameters=train.num_epochs,train.learning_rate)

    Returns:
        Parameter details including default ranges from schema
    """
    try:
        # Get network_arch from query string
        network_arch = request.args.get('network_arch', '')
        if not network_arch:
            return jsonify({"error_desc": "Missing 'network_arch' query parameter", "error_code": 1}), 400

        # Get parameters from query string
        parameters_query = request.args.get('parameters', '')
        if not parameters_query:
            return jsonify({"error_desc": "Missing 'parameters' query parameter", "error_code": 1}), 400

        # Split comma-separated parameters
        parameter_names = [p.strip() for p in parameters_query.split(',') if p.strip()]

        if not parameter_names:
            return jsonify({"error_desc": "No parameters provided", "error_code": 1}), 400

        # Get the actual network name from the enum mapping
        try:
            # Try to get the network name from ExperimentNetworkArch enum
            network_arch_obj = ExperimentNetworkArch(network_arch)
            network_name, _ = get_microservices_network_and_action(network_arch_obj.value, "train")
        except (ValueError, AttributeError):
            network_name = network_arch

        logger.info(f"Getting parameter details for network: {network_name}, parameters: {parameter_names}")

        # Generate schema for the network
        try:
            json_schema = generate_schema(network_name, "train")
        except Exception as e:
            logger.error(f"Error generating schema for network: {network_name}")
            return jsonify({
                "error_desc": f"Unable to generate schema for network: {network_name}. Error: {str(e)}",
                "error_code": 1
            }), 400

        # Flatten the schema properties to get parameter details
        format_json_schema = flatten_properties(json_schema["properties"])

        # Build response for requested parameters
        parameter_details = []
        for param_name in parameter_names:
            if param_name in format_json_schema:
                param_info = format_json_schema[param_name]

                # Build the parameter detail object
                # Convert empty strings to None for numeric fields
                valid_min = param_info.get("valid_min", None)
                valid_max = param_info.get("valid_max", None)
                if valid_min == "":
                    valid_min = None
                if valid_max == "":
                    valid_max = None

                # Build default section from schema
                default_section = {
                    "default_value": param_info.get("default_value", None),
                    "valid_min": valid_min,
                    "valid_max": valid_max,
                    "valid_options": param_info.get("valid_options", []),
                    "option_weights": param_info.get("option_weights", None),
                    "math_cond": param_info.get("math_cond", None),
                    "depends_on": param_info.get("depends_on", None),
                    "parent_param": param_info.get("parent_param", None)
                }

                param_detail = {
                    "parameter": param_name,
                    "value_type": param_info.get("value_type", ""),
                    "default": default_section
                }

                parameter_details.append(param_detail)
            else:
                logger.warning(f"Parameter '{param_name}' not found in schema for network '{network_name}'")

        if not parameter_details:
            return jsonify({
                "error_desc": f"None of the requested parameters were found in the schema for network '{network_name}'",
                "error_code": 1
            }), 404

        # Validate and serialize response
        rsp_schema = AutoMLParameterDetailsRsp()
        response_data = {"parameter_details": parameter_details}
        serialized = rsp_schema.dump(response_data)

        return jsonify(serialized), 200

    except Exception as e:
        logger.error(f"Error in get_automl_param_details: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error_desc": f"Internal server error: {str(e)}",
            "error_code": 1
        }), 500
