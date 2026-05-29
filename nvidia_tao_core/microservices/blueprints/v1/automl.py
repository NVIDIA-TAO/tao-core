# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoML Flask Routes - Manage parameter ranges and details"""
import logging
import os
import traceback
from flask import Blueprint, request, jsonify

from .schemas import AutoMLParameterDetailsRsp, AutoMLUpdateParameterRangesReq
from nvidia_tao_core.microservices.automl.params import flatten_properties
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    get_automl_custom_param_ranges,
    save_automl_custom_param_ranges,
    get_handler_metadata
)
from nvidia_tao_core.microservices.utils.handler_utils import validate_uuid
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
automl_params_bp_v1 = Blueprint('automl_params_v1', __name__, template_folder='templates')


@automl_params_bp_v1.route('/orgs/<org_name>/experiments/<experiment_id>:get_automl_param_details', methods=['GET'])
def get_automl_param_details(org_name, experiment_id):
    """Get detailed information about AutoML parameters including valid ranges, options, and math conditions.

    Args:
        org_name: Organization name
        experiment_id: Experiment ID
        Query params:
            parameters: Comma-separated list of parameter names (e.g., ?parameters=train.num_epochs,train.learning_rate)

    Returns:
        Parameter details including default ranges and any custom ranges set by the user
    """
    try:
        # Validate experiment_id is a valid UUID
        validation_error = validate_uuid(experiment_id=experiment_id)
        if validation_error:
            return jsonify({"error_desc": validation_error, "error_code": 1}), 400

        # Get parameters from query string
        parameters_query = request.args.get('parameters', '')
        if not parameters_query:
            return jsonify({"error_desc": "Missing 'parameters' query parameter", "error_code": 1}), 400

        # Split comma-separated parameters
        parameter_names = [p.strip() for p in parameters_query.split(',') if p.strip()]

        if not parameter_names:
            return jsonify({"error_desc": "No parameters provided", "error_code": 1}), 400

        # Get experiment metadata to determine network architecture
        handler_metadata = get_handler_metadata(experiment_id, "experiments")
        if not handler_metadata:
            return jsonify({
                "error_desc": f"Experiment {experiment_id} not found",
                "error_code": 1
            }), 404

        network_arch_enum = handler_metadata.get("network_arch")
        if not network_arch_enum:
            return jsonify({
                "error_desc": "Network architecture not found in experiment metadata",
                "error_code": 1
            }), 400

        # Convert enum to string if needed
        if hasattr(network_arch_enum, 'value'):
            network_arch = network_arch_enum.value
        else:
            network_arch = str(network_arch_enum)

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

        # Get custom parameter ranges if they exist
        custom_ranges = get_automl_custom_param_ranges(experiment_id)

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

                # Build custom section from user overrides
                custom_section = None
                if param_name in custom_ranges:
                    custom_section = {
                        "valid_min": custom_ranges[param_name].get("valid_min"),
                        "valid_max": custom_ranges[param_name].get("valid_max"),
                        "valid_options": custom_ranges[param_name].get("valid_options"),
                        "option_weights": custom_ranges[param_name].get("option_weights"),
                        "depends_on": custom_ranges[param_name].get("depends_on"),
                        "math_cond": custom_ranges[param_name].get("math_cond"),
                        "parent_param": custom_ranges[param_name].get("parent_param"),
                        "disable_list": custom_ranges[param_name].get("disable_list")
                    }

                param_detail = {
                    "parameter": param_name,
                    "value_type": param_info.get("value_type", ""),
                    "default": default_section,
                    "custom": custom_section
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


@automl_params_bp_v1.route(
    '/orgs/<org_name>/experiments/<experiment_id>:update_automl_param_ranges',
    methods=['POST', 'PATCH']
)
def update_automl_param_ranges(org_name, experiment_id):
    """Update custom valid ranges for AutoML parameters.

    Args:
        org_name: Organization name
        experiment_id: Experiment ID

    Returns:
        Success message
    """
    try:
        # Validate experiment_id is a valid UUID
        validation_error = validate_uuid(experiment_id=experiment_id)
        if validation_error:
            return jsonify({"error_desc": validation_error, "error_code": 1}), 400

        # Check if request has JSON content-type
        if not request.is_json:
            return jsonify({
                "error_desc": "Request Content-Type must be 'application/json'",
                "error_code": 1
            }), 400

        # Validate request body
        req_schema = AutoMLUpdateParameterRangesReq()
        errors = req_schema.validate(request.json)
        if errors:
            return jsonify({
                "error_desc": f"Invalid request: {errors}",
                "error_code": 1
            }), 400

        request_data = request.json
        parameter_ranges = request_data.get("parameter_ranges", [])

        if not parameter_ranges:
            return jsonify({"error_desc": "No parameter ranges provided", "error_code": 1}), 400

        # Get experiment metadata to verify it exists
        handler_metadata = get_handler_metadata(experiment_id, "experiments")
        if not handler_metadata:
            return jsonify({"error_desc": f"Experiment {experiment_id} not found", "error_code": 1}), 404

        network_arch_enum = handler_metadata.get("network_arch")
        if not network_arch_enum:
            return jsonify({
                "error_desc": "Network architecture not found in experiment metadata",
                "error_code": 1
            }), 400

        # Convert enum to string if needed
        if hasattr(network_arch_enum, 'value'):
            network_arch = network_arch_enum.value
        else:
            network_arch = str(network_arch_enum)

        # Get the actual network name from the enum mapping
        try:
            network_arch_obj = ExperimentNetworkArch(network_arch)
            network_name, _ = get_microservices_network_and_action(network_arch_obj.value, "train")
        except (ValueError, AttributeError):
            network_name = network_arch

        logger.info(f"Updating parameter ranges for network: {network_name}, experiment: {experiment_id}")

        # Generate schema to validate parameters exist
        try:
            json_schema = generate_schema(network_name, "train")
        except Exception as e:
            logger.error(f"Error generating schema for network: {network_name}")
            return jsonify({
                "error_desc": f"Unable to generate schema for network: {network_name}. Error: {str(e)}",
                "error_code": 1
            }), 400

        format_json_schema = flatten_properties(json_schema["properties"])

        # Validate that all parameters exist in the schema and ranges are valid
        validated_ranges = {}
        errors = []
        for param_range in parameter_ranges:
            param_name = param_range.get("parameter")
            custom_min = param_range.get("valid_min")
            custom_max = param_range.get("valid_max")
            custom_options = param_range.get("valid_options")
            custom_weights = param_range.get("option_weights")
            custom_depends_on = param_range.get("depends_on")
            custom_math_cond = param_range.get("math_cond")
            custom_parent_param = param_range.get("parent_param")
            disable_list = param_range.get("disable_list", False)

            # Check if parameter exists in schema
            if param_name not in format_json_schema:
                errors.append(f"Parameter '{param_name}' not found in schema for network '{network_name}'")
                continue

            param_info = format_json_schema[param_name]
            schema_min = param_info.get("valid_min")
            schema_max = param_info.get("valid_max")

            # Validate custom ranges (min/max)
            if custom_min is not None and custom_max is not None:
                # Handle both scalar and list types
                if isinstance(custom_min, list) and isinstance(custom_max, list):
                    # Validate list ranges element-wise
                    if len(custom_min) != len(custom_max):
                        errors.append(
                            f"Parameter '{param_name}': valid_min and valid_max "
                            "must have same length"
                        )
                        continue

                    for i, (min_val, max_val) in enumerate(zip(custom_min, custom_max)):
                        if min_val >= max_val:
                            errors.append(
                                f"Parameter '{param_name}': valid_min[{i}] must be less "
                                f"than valid_max[{i}]"
                            )
                            continue

                        # Validate against schema bounds if they exist and are lists
                        if isinstance(schema_min, list) and len(schema_min) > i:
                            if (schema_min[i] is not None and schema_min[i] != "" and
                                    min_val < schema_min[i]):
                                errors.append(
                                    f"Parameter '{param_name}': valid_min[{i}] ({min_val}) cannot be less than "
                                    f"schema minimum ({schema_min[i]})"
                                )
                                continue

                        if isinstance(schema_max, list) and len(schema_max) > i:
                            if (schema_max[i] is not None and schema_max[i] != "" and
                                    max_val > schema_max[i]):
                                errors.append(
                                    f"Parameter '{param_name}': valid_max[{i}] ({max_val}) "
                                    f"cannot be greater than schema maximum ({schema_max[i]})"
                                )
                                continue
                else:
                    # Scalar validation
                    if custom_min >= custom_max:
                        errors.append(
                            f"Parameter '{param_name}': valid_min must be less than "
                            "valid_max"
                        )
                        continue

                    # Validate against schema bounds if they exist
                    if (schema_min is not None and schema_min != "" and
                            custom_min < schema_min):
                        errors.append(
                            f"Parameter '{param_name}': valid_min ({custom_min}) cannot be less than "
                            f"schema minimum ({schema_min})"
                        )
                        continue

                    if (schema_max is not None and schema_max != "" and
                            custom_max > schema_max):
                        errors.append(
                            f"Parameter '{param_name}': valid_max ({custom_max}) cannot be greater than "
                            f"schema maximum ({schema_max})"
                        )
                        continue

            # Validate valid_options if provided
            if custom_options is not None:
                schema_options = param_info.get("valid_options", [])
                if schema_options:
                    # Ensure custom options are a subset of schema options
                    invalid_options = [opt for opt in custom_options if opt not in schema_options]
                    if invalid_options:
                        errors.append(
                            f"Parameter '{param_name}': valid_options {invalid_options} are not in "
                            f"schema options {schema_options}"
                        )
                        continue

            # Validate option_weights if provided
            if custom_weights is not None:
                # Get the options to validate against (custom or schema)
                options_to_check = custom_options if custom_options is not None else param_info.get("valid_options", [])

                if not options_to_check:
                    errors.append(
                        f"Parameter '{param_name}': option_weights provided but no valid_options exist"
                    )
                    continue

                if len(custom_weights) != len(options_to_check):
                    errors.append(
                        f"Parameter '{param_name}': option_weights length ({len(custom_weights)}) "
                        f"must match valid_options length ({len(options_to_check)})"
                    )
                    continue

                # Validate all weights are positive
                if any(w <= 0 for w in custom_weights):
                    errors.append(
                        f"Parameter '{param_name}': all option_weights must be positive numbers"
                    )
                    continue

                # Validate weights sum to a reasonable value (allow flexibility, just check they're not all zeros)
                if sum(custom_weights) <= 0:
                    errors.append(
                        f"Parameter '{param_name}': option_weights must sum to a positive value"
                    )
                    continue

            validated_ranges[param_name] = {
                "valid_min": custom_min,
                "valid_max": custom_max,
                "valid_options": custom_options,
                "option_weights": custom_weights,
                "depends_on": custom_depends_on,
                "math_cond": custom_math_cond,
                "parent_param": custom_parent_param,
                "disable_list": disable_list
            }

        if errors:
            return jsonify({
                "error_desc": f"Validation errors: {'; '.join(errors)}",
                "error_code": 1
            }), 400

        # Save the custom ranges
        save_automl_custom_param_ranges(experiment_id, validated_ranges)

        logger.info(f"Successfully updated {len(validated_ranges)} parameter range(s) for experiment {experiment_id}")

        return jsonify({
            "message": f"Successfully updated parameter ranges for {len(validated_ranges)} parameter(s)",
            "updated_parameters": list(validated_ranges.keys())
        }), 200

    except Exception as e:
        logger.error(f"Error in update_automl_param_ranges: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error_desc": f"Internal server error: {str(e)}",
            "error_code": 1
        }), 500
