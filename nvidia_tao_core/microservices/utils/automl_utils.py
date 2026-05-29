# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utils for AutoML"""
import os
import math
import glob
import datetime
import time
from kubernetes import client, config
import logging
import traceback

from .stateless_handler_utils import (
    BACKEND,
    get_automl_controller_info,
    get_automl_current_rec,
    get_automl_best_rec_info,
    update_job_metadata
)
from nvidia_tao_core.microservices.enum_constants import Backend

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


def fix_input_dimension(dimension_value, factor=32):
    """Return dimension as a multiple of factor"""
    if int(dimension_value) % factor == 0:
        return dimension_value
    return (int(dimension_value / factor) + 1) * factor


def fix_power_of_factor(value, factor=2):
    """Return the nearest power of factor that is >= value"""
    if value <= 0:
        return factor  # Return the base factor for non-positive values
    # Calculate the power needed: factor^power >= value
    power = math.ceil(math.log(value) / math.log(factor))
    return int(factor ** power)


def clamp_value(value, v_min, v_max):
    """Clamps value within the given range"""
    if value >= v_max:
        epsilon = v_max / 10
        if epsilon == 0.0:
            epsilon = 0.0000001
        value = v_max - epsilon
    if value <= v_min:
        epsilon = v_min / 10
        if epsilon == 0.0:
            epsilon = 0.0000001
        value = v_min + epsilon
    return value


def get_valid_range(parameter_config, parent_params, custom_ranges=None):
    """Compute the clamp range for the given parameter

    Args:
        parameter_config: Configuration dict for the parameter
        parent_params: Dict of parent parameter values
        custom_ranges: Optional dict of custom parameter ranges from user

    Returns:
        Tuple of (v_min, v_max)
    """
    parameter_name = parameter_config.get("parameter", "")

    # Handle empty strings and None values for numeric parameters
    valid_min = parameter_config.get("valid_min")
    valid_max = parameter_config.get("valid_max")
    default_val = parameter_config.get("default_value")

    # Convert to float, handling empty strings and None
    v_min = float(valid_min) if valid_min not in (None, '', "") else 0.0
    v_max = float(valid_max) if valid_max not in (None, '', "") else float('inf')
    default_value = float(default_val) if default_val not in (None, '', "") else 0.0
    if math.isinf(v_min):
        v_min = default_value
    if math.isinf(v_max):
        v_max = default_value

    # Apply custom ranges if provided
    if custom_ranges and parameter_name in custom_ranges:
        custom_min = custom_ranges[parameter_name].get("valid_min")
        custom_max = custom_ranges[parameter_name].get("valid_max")
        if custom_min is not None:
            v_min = float(custom_min) if not isinstance(custom_min, list) else custom_min
        if custom_max is not None:
            v_max = float(custom_max) if not isinstance(custom_max, list) else custom_max

    # Check for custom depends_on, otherwise use schema depends_on
    dependent_on_param = parameter_config.get("depends_on", None)
    if custom_ranges and parameter_name in custom_ranges:
        custom_depends_on = custom_ranges[parameter_name].get("depends_on")
        if custom_depends_on is not None:
            dependent_on_param = custom_depends_on
    if type(dependent_on_param) is str and dependent_on_param:
        parts = dependent_on_param.split(" ")
        if len(parts) >= 2:
            dependent_on_param_op = parts[0]
            dependent_on_param_name = parts[1]
        else:
            dependent_on_param_name = parts[0]
            math_cond = parameter_config.get("math_cond", "")
            if type(math_cond) is str and "depends_on" in math_cond:
                dependent_on_param_op = math_cond.strip().split()[0]
            else:
                dependent_on_param_op = None

        if dependent_on_param_op is not None:
            if dependent_on_param_name in parent_params.keys():
                limit_value = parent_params[dependent_on_param_name]
            else:
                limit_value = default_value

            epsilon = 0.000001
            if limit_value == epsilon:
                epsilon /= 10

            if dependent_on_param_op == ">":
                v_min = limit_value + epsilon
            elif dependent_on_param_op == ">=":
                v_min = limit_value
            elif dependent_on_param_op == "<":
                v_max = limit_value - epsilon
            elif dependent_on_param_op == "<=":
                v_max = limit_value

    return v_min, v_max


def get_valid_options(parameter_config, custom_ranges=None):
    """Get the valid options for a parameter, considering custom overrides

    Args:
        parameter_config: Configuration dict for the parameter
        custom_ranges: Optional dict of custom parameter ranges from user

    Returns:
        List of valid options (or schema default if no custom options)
    """
    parameter_name = parameter_config.get("parameter", "")
    valid_options = parameter_config.get("valid_options", [])

    # Apply custom valid_options if provided
    if custom_ranges and parameter_name in custom_ranges:
        custom_options = custom_ranges[parameter_name].get("valid_options")
        if custom_options is not None:
            valid_options = custom_options

    return valid_options


def get_option_weights(parameter_config, custom_ranges=None):
    """Get the weights for valid options, considering custom overrides

    Args:
        parameter_config: Configuration dict for the parameter
        custom_ranges: Optional dict of custom parameter ranges from user

    Returns:
        List of weights corresponding to valid_options, or None for uniform sampling
    """
    parameter_name = parameter_config.get("parameter", "")
    option_weights = parameter_config.get("option_weights", None)

    # Apply custom option_weights if provided
    if custom_ranges and parameter_name in custom_ranges:
        custom_weights = custom_ranges[parameter_name].get("option_weights")
        if custom_weights is not None:
            option_weights = custom_weights

    return option_weights


def report_healthy(path, message, clear=False):
    """Write health message to the provided file"""
    mode = "w" if clear else "a"
    with open(path, mode, encoding='utf-8') as f:
        f.write(f"Healthy at {datetime.datetime.now().isoformat()}\n")
        if message:
            f.write(str(message) + "\n")


def wait_for_job_completion(job_id):
    """Check if the provided job_id is actively running and wait until completion"""
    if BACKEND == Backend.LOCAL_DOCKER:
        from nvidia_tao_core.microservices.handlers.execution_handlers.docker_handler import DockerHandler
        while True:
            handler = DockerHandler.get_handler_for_container(job_id)
            if not handler:
                return
            time.sleep(5)

    config.load_incluster_config()
    while True:
        ret = client.BatchV1Api().list_job_for_all_namespaces()
        active_jobs = [job.metadata.name for job in ret.items]
        if job_id not in active_jobs:
            break
        time.sleep(5)


def delete_lingering_checkpoints(epoch_number, path):
    """Delete checkpoints which are present even after job deletion"""
    trained_files = (
        glob.glob(path + "/**/*.tlt", recursive=True) +
        glob.glob(path + "/**/*.hdf5", recursive=True) +
        glob.glob(path + "/**/*.pth", recursive=True) +
        glob.glob(path + "/**/*.ckzip", recursive=True)
    )
    for file_name in trained_files:
        if os.path.isfile(file_name):
            if not (f"{epoch_number}.tlt" in file_name or
                    f"{epoch_number}.hdf5" in file_name or
                    f"{epoch_number}.pth" in file_name or
                    f"{epoch_number}.ckzip" in file_name):
                os.remove(file_name)


class Recommendation:
    """Recommendation class for AutoML recommendations"""

    def __init__(self, identifier, specs, metric):
        """Initialize the Recommendation class

        Args:
            identity: the id of the recommendation
            specs: the specs/config of the recommendation
        """
        assert type(identifier) is int, f"Recommendation identifier must be an integer, got {type(identifier)}"
        self.id = identifier

        assert type(specs) is dict, f"Recommendation specs must be a dictionary, got {type(specs)}"
        self.specs = specs

        self.job_id = None
        self.status = JobStates.pending
        self.result = 0.0
        self.best_epoch_number = ""
        self.metric = metric
        self.resume_from_job_id = None  # For PBT: job ID to resume checkpoint from
        self.early_stop_epoch = None  # For PBT/Hyperband: epoch limit when this rec was launched

        # Add timestamps for timeout tracking
        current_time = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        self.created_on = current_time
        self.last_modified = current_time

        # Add timestamps for timeout tracking
        current_time = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        self.created_on = current_time
        self.last_modified = current_time

    def items(self):
        """Returns specs.items"""
        return self.specs.items()

    def get(self, key):
        """Returns value of requested key in the spec"""
        return self.specs.get(key, None)

    def assign_job_id(self, job_id):
        """Associates provided job id to the class objects job id"""
        assert type(job_id) is str, f"Job ID must be a string, got {type(job_id)}"
        self.job_id = job_id

        # Update last_modified timestamp when job is assigned
        self.last_modified = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    def update_result(self, result):
        """Update the result value"""
        result = float(result)
        assert type(result) is float, f"Result must be a float value, got {type(result)}"
        self.result = result

    def update_status(self, status):
        """Update the status value"""
        assert type(status) is str, f"Status must be a string, got {type(status)}"
        self.status = status

        # Update last_modified timestamp when status changes
        self.last_modified = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    def __repr__(self):
        """Constructs a dictionary with the class members and returns them"""
        return f"id: {self.id}\njob_id: {self.job_id}\nresult: {self.result}\nstatus: {self.status}"


class ResumeRecommendation:
    """Recommendation class for Hyperband resume experiments"""

    def __init__(self, identity, specs, job_id, resume_from_job_id=None):
        """Initialize the ResumeRecommendation class

        Args:
            identity: the id of the recommendation
            specs: the specs/config of the recommendation
            job_id: the job id of the recommendation
            resume_from_job_id: (PBT) the job id to resume checkpoint from if member was replaced
        """
        self.id = identity
        self.specs = specs
        self.job_id = job_id
        self.resume_from_job_id = resume_from_job_id


class JobStates():
    """Various states of an automl job"""

    pending = "pending"
    started = "started"
    running = "running"
    success = "success"
    failure = "failure"
    error = "error"  # alias for failure
    done = "done"  # alias for success
    canceled = "canceled"
    canceling = "canceling"


def update_automl_details_metadata(brain_job_id, handler_id, handler_kind="experiments"):
    """Update AutoML details in job metadata for continuous updates

    This function extracts AutoML controller information and updates it in the job metadata.
    It's designed to be called after save_automl_controller_info to keep the metadata in sync.

    Args:
        brain_job_id: The brain job ID for the AutoML run
        handler_id: The handler (experiment) ID
        handler_kind: The kind of handler (default: "experiments")
    """
    try:
        automl_controller_data = get_automl_controller_info(brain_job_id)
        automl_interpretable_result = {}

        # Get current experiment id
        current_rec = get_automl_current_rec(brain_job_id)
        if not current_rec:
            current_rec = 0
        automl_interpretable_result["current_experiment_id"] = current_rec

        # Get per experiment result and status
        automl_interpretable_result["experiments"] = {}
        for experiment_details in automl_controller_data:
            automl_interpretable_result["metric"] = experiment_details.get("metric")
            exp_id = experiment_details.get("id")
            # Convert exp_id to string for MongoDB compatibility (MongoDB requires string keys)
            exp_id_str = str(exp_id)
            automl_interpretable_result["experiments"][exp_id_str] = {}
            automl_interpretable_result["experiments"][exp_id_str]["result"] = experiment_details.get("result")
            automl_interpretable_result["experiments"][exp_id_str]["status"] = experiment_details.get("status")
            automl_interpretable_result["experiments"][exp_id_str]["specs"] = experiment_details.get("specs", {})

        # Get the best experiment id from the automl_jobs table
        best_rec_number, _, _ = get_automl_best_rec_info(brain_job_id)
        if best_rec_number and best_rec_number != "-1":
            automl_interpretable_result["best_experiment_id"] = int(best_rec_number)

        # Update job metadata with automl details
        update_job_metadata(
            handler_id,
            brain_job_id,
            metadata_key="automl_details",
            data=automl_interpretable_result,
            kind=handler_kind
        )
        logger.info(
            f"Updated AutoML details for job {brain_job_id}: "
            f"current_exp={current_rec}, total_experiments={len(automl_interpretable_result['experiments'])}"
        )
    except Exception as e:
        logger.error("Exception while updating AutoML details: %s", str(e))
        logger.error(traceback.format_exc())


def apply_automl_custom_param_ranges(job_id, network_arch, automl_range_override):
    """Utility function to validate and save custom AutoML parameter ranges.

    This function should be called from job_handler before AutoMLHandler starts.

    Args:
        job_id: Job ID to associate custom ranges with
        network_arch: Network architecture for validation
        automl_range_override: List of parameter range overrides from automl_settings

    Returns:
        Tuple of (success: bool, error_message: str or None)
    """
    from nvidia_tao_core.microservices.automl.params import flatten_properties
    from nvidia_tao_core.scripts.generate_schema import generate_schema
    from nvidia_tao_core.microservices.utils import get_microservices_network_and_action
    from nvidia_tao_core.microservices.enum_constants import ExperimentNetworkArch
    from nvidia_tao_core.microservices.utils.stateless_handler_utils import save_automl_custom_param_ranges

    try:
        # Check if custom parameter ranges are provided
        if not automl_range_override:
            # No custom ranges provided, this is OK
            logger.info(f"No custom parameter ranges provided for job {job_id}")
            return True, None

        # Get the actual network name from the enum mapping
        try:
            network_arch_obj = ExperimentNetworkArch(network_arch)
            network_name, _ = get_microservices_network_and_action(network_arch_obj.value, "train")
        except (ValueError, AttributeError):
            network_name = network_arch

        logger.info(f"Applying custom parameter ranges for network: {network_name}, job: {job_id}")

        # Generate schema to validate parameters exist
        try:
            json_schema = generate_schema(network_name, "train")
        except Exception as e:
            logger.error(f"Error generating schema for network: {network_name}")
            return False, f"Unable to generate schema for network: {network_name}. Error: {str(e)}"

        format_json_schema = flatten_properties(json_schema["properties"])

        # Validate that all parameters exist in the schema and ranges are valid
        validated_ranges = {}
        errors = []
        for param_range in automl_range_override:
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
            # Debug: Log when disable_list is True
            if disable_list:
                logger.info(
                    f"[AUTOML-CUSTOM-RANGE] Parameter '{param_name}': disable_list=True, "
                    f"valid_min={custom_min}, valid_max={custom_max}"
                )

        if errors:
            error_msg = f"Validation errors: {'; '.join(errors)}"
            logger.error(error_msg)
            return False, error_msg

        # Save the custom ranges
        save_automl_custom_param_ranges(job_id, validated_ranges)

        logger.info(f"Successfully applied {len(validated_ranges)} custom parameter range(s) for job {job_id}")
        return True, None

    except Exception as e:
        logger.error(f"Error in apply_automl_custom_param_ranges: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, f"Internal error: {str(e)}"
