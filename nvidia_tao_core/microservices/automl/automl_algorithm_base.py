# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoML algorithm's Base Class"""
import math
import numpy as np
import random
import logging


from nvidia_tao_core.microservices.utils.automl_utils import (
    fix_input_dimension,
    fix_power_of_factor,
    get_valid_options,
    get_option_weights,
    get_valid_range
)
from nvidia_tao_core.microservices.automl import network_utils
from nvidia_tao_core.microservices.utils.network_utils import network_constants
from nvidia_tao_core.microservices.utils.network_utils import automl_helper
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    get_job_specs,
    get_automl_custom_param_ranges
)


logger = logging.getLogger(__name__)


def is_nan_value(val):
    """Check if value is NaN, handling different types safely.

    For lists/tuples, checks if ANY element is NaN.

    Args:
        val: Value to check (can be float, int, str, list, tuple, etc.)

    Returns:
        bool: True if value is NaN (or contains NaN for lists), False otherwise
    """
    if isinstance(val, (list, tuple)):
        # Check each element in the list/tuple
        return any(is_nan_value(v) for v in val)
    if isinstance(val, str):
        return False
    try:
        return math.isnan(val)
    except (TypeError, ValueError):
        return False


class AutoMLAlgorithmBase:
    """AutoML algorithms base class"""

    def __init__(self, job_context, root, network, parameters):
        """AutoML algorithm Base class"""
        self.job_context = job_context
        self.root = root
        self.handler_root = "/".join(self.root.split("/")[0:-2])
        self.network = network
        self.parameters = parameters
        self.parent_params = {}
        self.default_train_spec = get_job_specs(self.job_context.id)
        self.default_train_spec_flattened = {}

        # Load custom parameter ranges from experiment
        experiment_id = self.job_context.handler_id
        self.custom_ranges = get_automl_custom_param_ranges(experiment_id)
        logger.info(f"Loaded {len(self.custom_ranges)} custom parameter range(s) for experiment {experiment_id}")

        # Initialize random seeds to ensure different values across experiments
        # Using job_context.id hash to get different seeds for different jobs
        seed = hash(str(job_context.id)) % 2**31
        np.random.seed(seed)
        random.seed(seed)

        logger.info(f"Initialized random seed: {seed} for job {job_context.id}")

    def _apply_power_constraint_with_equal_priority(self, v_min, v_max, factor, fallback_value=None):
        """Apply power constraint by sampling directly from valid powers to give equal priority.

        Args:
            v_min: Minimum valid value
            v_max: Maximum valid value
            factor: Power factor (e.g., 2 for powers of 2)
            fallback_value: Value to use if no valid powers found (optional)

        Returns:
            A value that is a power of factor within the range, or fallback if none exist
        """
        # Generate all valid powers within the range
        valid_powers = []
        power = 1
        while True:
            power_value = factor ** power
            if power_value > v_max:
                break
            if power_value >= v_min:
                valid_powers.append(power_value)
            power += 1

        if valid_powers:
            result = np.random.choice(valid_powers)
            logger.info(f"Sampled from valid powers {valid_powers}: {result}")
            return result
        # Fallback: use provided fallback value or apply fix_power_of_factor
        if fallback_value is not None:
            result = fix_power_of_factor(fallback_value, factor)
            logger.info(f"Applied power constraint fallback: {result}")
            return result
        logger.warning(f"No valid powers of {factor} found in range [{v_min}, {v_max}]")
        return v_min

    def _apply_relational_constraint(self, value, math_cond, depends_on, parameter_name, v_min, v_max):
        """Apply relational constraints based on depends_on parameter value.

        Supports math_cond operators:
        - "> depends_on": value must be greater than depends_on parameter
        - ">= depends_on": value must be greater than or equal to depends_on parameter
        - "< depends_on": value must be less than depends_on parameter
        - "<= depends_on": value must be less than or equal to depends_on parameter

        Args:
            value: The sampled value to constrain
            math_cond: The math condition string (e.g., "> depends_on")
            depends_on: The parameter name this depends on
            parameter_name: Current parameter name (for logging)
            v_min: Minimum valid value for this parameter
            v_max: Maximum valid value for this parameter

        Returns:
            Constrained value that satisfies the relational constraint
        """
        if "depends_on" not in math_cond:
            return value

        # Check if depends_on parameter has been sampled
        if depends_on not in self.parent_params:
            logger.info(f"{parameter_name}: depends_on '{depends_on}' not yet sampled, skipping constraint")
            return value

        parent_value = self.parent_params[depends_on]
        parts = math_cond.strip().split()

        if len(parts) >= 2:
            operator = parts[0]
            constrained_value = value

            # Apply the relational constraint
            if operator == ">":
                # Value must be strictly greater than parent
                min_allowed = parent_value + 1
                if value <= parent_value:
                    constrained_value = max(min_allowed, v_min)
                    logger.warning(
                        f"CONSTRAINT: {parameter_name}={value} must be > {depends_on}={parent_value}. "
                        f"Adjusted to {constrained_value}"
                    )
                else:
                    logger.info(f"{parameter_name}={value} satisfies > {depends_on}={parent_value}")

            elif operator == ">=":
                # Value must be greater than or equal to parent
                if value < parent_value:
                    constrained_value = max(parent_value, v_min)
                    logger.warning(
                        f"CONSTRAINT: {parameter_name}={value} must be >= {depends_on}={parent_value}. "
                        f"Adjusted to {constrained_value}"
                    )
                else:
                    logger.info(f"{parameter_name}={value} satisfies >= {depends_on}={parent_value}")

            elif operator == "<":
                # Value must be strictly less than parent
                max_allowed = parent_value - 1
                if value >= parent_value:
                    constrained_value = min(max_allowed, v_max)
                    logger.warning(
                        f"CONSTRAINT: {parameter_name}={value} must be < {depends_on}={parent_value}. "
                        f"Adjusted to {constrained_value}"
                    )
                else:
                    logger.info(f"{parameter_name}={value} satisfies < {depends_on}={parent_value}")

            elif operator == "<=":
                # Value must be less than or equal to parent
                if value > parent_value:
                    constrained_value = min(parent_value, v_max)
                    logger.warning(
                        f"CONSTRAINT: {parameter_name}={value} must be <= {depends_on}={parent_value}. "
                        f"Adjusted to {constrained_value}"
                    )
                else:
                    logger.info(f"{parameter_name}={value} satisfies <= {depends_on}={parent_value}")

            return constrained_value

        return value

    def generate_automl_param_rec_value(self, parameter_config):
        """Generate a random value for the parameter passed"""
        parameter_name = parameter_config.get("parameter")
        data_type = parameter_config.get("value_type")
        default_value = parameter_config.get("default_value", None)

        # Apply custom overrides if provided
        if self.custom_ranges and parameter_name in self.custom_ranges:
            custom_range = self.custom_ranges[parameter_name]
            logger.debug(
                f"[AUTOML-BASE] Applying custom range for {parameter_name}: {custom_range}"
            )
            for override_key, override_value in custom_range.items():
                if override_value is not None:
                    parameter_config[override_key] = override_value
                    if override_key == "disable_list":
                        logger.info(
                            f"[AUTOML-BASE] Applied disable_list={override_value} to {parameter_name}"
                        )

        # Get potentially overridden values
        math_cond = parameter_config.get("math_cond", None)
        parent_param = parameter_config.get("parent_param", None)

        if data_type == "float":
            v_min = parameter_config.get("valid_min", "")
            v_max = parameter_config.get("valid_max", "")

            # If no valid range, generate diverse values around default
            if v_min == "" or v_max == "":
                if default_value is not None and default_value != "":
                    default_val = float(default_value)
                    # Generate values in diverse range around default
                    if default_val > 0:
                        v_min = default_val / 10.0
                        v_max = default_val * 10.0
                    elif default_val < 0:
                        v_min = default_val * 10.0
                        v_max = default_val / 10.0
                    else:  # default is 0
                        v_min = -1.0
                        v_max = 1.0
                    random_float = np.random.uniform(v_min, v_max)
                    logger.info(
                        f"Generated random float for {parameter_name} (no range): "
                        f"{random_float} around default {default_val}"
                    )
                    return random_float
                return np.random.uniform(0.0, 1.0)

            if is_nan_value(v_min) or is_nan_value(v_max):
                # NaN ranges, use default-based range
                if default_value is not None:
                    default_val = float(default_value)
                    if default_val > 0:
                        return np.random.uniform(default_val / 10.0, default_val * 10.0)
                    return np.random.uniform(0.0, 1.0)
                return np.random.uniform(0.0, 1.0)

            # Handle list-based ranges (e.g., per-model-part learning rates)
            # Generate a base value and let network-specific handler convert to list
            if isinstance(v_min, list) or isinstance(v_max, list):
                # Use first element of list for base range
                if isinstance(v_min, list) and isinstance(v_max, list):
                    base_min = float(v_min[0]) if v_min else 0.0
                    base_max = float(v_max[0]) if v_max else 1.0
                elif isinstance(v_min, list):
                    base_min = float(v_min[0]) if v_min else 0.0
                    base_max = float(v_max) if v_max not in (None, '', "") else base_min * 10
                else:
                    base_min = float(v_min) if v_min not in (None, '', "") else 0.0
                    base_max = float(v_max[0]) if v_max else 1.0

                # Generate base value using log-uniform sampling (better for LR)
                if base_min > 0 and base_max > 0:
                    log_min = np.log10(base_min)
                    log_max = np.log10(base_max)
                    base_value = float(10 ** np.random.uniform(log_min, log_max))
                else:
                    base_value = float(np.random.uniform(base_min, base_max))

                # Check for disable_list option - if True, skip network-specific logic
                # and return pure float value for optimization
                disable_list = parameter_config.get("disable_list", False)
                if disable_list:
                    logger.info(
                        f"disable_list=True for {parameter_name}: "
                        f"returning pure float {base_value} (skipping network-specific logic)"
                    )
                    return base_value

                # Let network-specific handler convert to list format
                return network_utils.apply_network_specific_param_logic(
                    network=self.network,
                    data_type=data_type,
                    parameter_name=parameter_name,
                    value=base_value,
                    v_max=v_max,
                    default_train_spec=self.default_train_spec,
                    parent_params=self.parent_params
                )

            # Existing handling for defined ranges
            v_min, v_max = get_valid_range(parameter_config, self.parent_params, self.custom_ranges)
            random_float = np.random.uniform(v_min, v_max)

            if not (type(parent_param) is float and math.isnan(parent_param)):
                if ((isinstance(parent_param, str) and parent_param != "nan" and parent_param == "TRUE") or
                        (isinstance(parent_param, bool) and parent_param)):
                    self.parent_params[parameter_name] = random_float

            return random_float

        if data_type in ("int", "integer"):
            if parameter_name == "augmentation_config.preprocessing.output_image_height":
                if "model_config.input_image_config.size_height_width.height" in self.parent_params.keys():
                    return self.parent_params["model_config.input_image_config.size_height_width.height"]
            if parameter_name == "augmentation_config.preprocessing.output_image_width":
                if "model_config.input_image_config.size_height_width.width" in self.parent_params.keys():
                    return self.parent_params["model_config.input_image_config.size_height_width.width"]

            # Check if this parameter has a dependency and math_cond for calculation
            depends_on = parameter_config.get("depends_on", None)
            if depends_on and math_cond and type(math_cond) is str:
                # Skip if this is a relational constraint (handled later)
                if "depends_on" in math_cond:
                    pass  # Will be handled by _apply_relational_constraint
                # Calculate value based on dependency for numeric operations
                elif depends_on in self.parent_params:
                    parent_value = self.parent_params[depends_on]
                    parts = math_cond.split(" ")
                    if len(parts) >= 2:
                        operator = parts[0]
                        factor = int(parts[1])
                        if operator == "/":
                            # Divide parent value by factor
                            calculated_int = int(parent_value // factor)
                            if not (type(parent_param) is float and math.isnan(parent_param)):
                                if ((isinstance(parent_param, str) and parent_param != "nan" and
                                     parent_param == "TRUE") or
                                        (isinstance(parent_param, bool) and parent_param)):
                                    self.parent_params[parameter_name] = calculated_int
                            return calculated_int

            v_min = parameter_config.get("valid_min", "")
            v_max = parameter_config.get("valid_max", "")
            if v_min == "" or v_max == "":
                # Generate diverse values around default
                if default_value is not None and default_value != "":
                    default_val = int(default_value)
                    # Generate values in range [max(1, default/2), default*2] for diversity
                    if default_val > 0:
                        v_min = max(1, default_val // 2)
                        v_max = default_val * 2
                    else:
                        v_min = 1
                        v_max = 100
                    random_int = np.random.randint(v_min, v_max + 1)
                    logger.info(
                        f"Generated random int for {parameter_name} (no range): "
                        f"{random_int} around default {default_val}"
                    )
                    return random_int
                return np.random.randint(1, 100)
            if is_nan_value(v_min) or is_nan_value(v_max):
                # NaN ranges, use default-based range
                if default_value is not None:
                    default_val = int(default_value)
                    return np.random.randint(max(1, default_val // 2), default_val * 2 + 1)
                return np.random.randint(1, 100)

            v_min = int(v_min)
            if (type(v_max) is not str and math.isinf(v_max)) or v_max == "inf":
                v_max = int(default_value)
            else:
                v_max = int(v_max)
            if math_cond and type(math_cond) is str and "depends_on" not in math_cond:
                # Only process numeric math_cond here (like "/ 2", "^ 2")
                # Relational constraints (like "> depends_on") are handled later
                parts = math_cond.split(" ")
                if len(parts) >= 2:
                    operator = parts[0]
                    factor = int(parts[1])
                    if operator == "^":
                        # Use helper function for power constraints with equal priority
                        fallback = np.random.randint(v_min, v_max + 1)
                        random_int = int(self._apply_power_constraint_with_equal_priority(
                            v_min, v_max, factor, fallback))
                    else:
                        # Regular sampling for non-power constraints
                        random_int = np.random.randint(v_min, v_max + 1)
                        if operator == "/":
                            # Multiple/factor constraint (existing behavior)
                            random_int = fix_input_dimension(random_int, factor)
            else:
                # No math condition, or relational constraint (handled later), regular sampling
                random_int = np.random.randint(v_min, v_max + 1)

            # Apply relational constraints based on depends_on parameter
            depends_on = parameter_config.get("depends_on", None)
            if depends_on and math_cond and type(math_cond) is str:
                random_int = self._apply_relational_constraint(
                    random_int, math_cond, depends_on, parameter_name, v_min, v_max
                )

            if not (type(parent_param) is float and math.isnan(parent_param)):
                if (isinstance(parent_param, str) and parent_param != "nan" and parent_param == "TRUE") or (
                    isinstance(parent_param, bool) and parent_param
                ):
                    self.parent_params[parameter_name] = random_int

            # Apply network-specific parameter logic
            random_int = network_utils.apply_network_specific_param_logic(
                network=self.network,
                data_type=data_type,
                parameter_name=parameter_name,
                value=random_int,
                v_max=v_max,
                default_train_spec=self.default_train_spec,
                parent_params=self.parent_params
            )

            return random_int

        if data_type == "bool":
            return np.random.randint(0, 2) == 1

        if data_type == "ordered_int":
            valid_values = get_valid_options(parameter_config, self.custom_ranges)
            if not valid_values or valid_values == "":
                return default_value
            # Get weights for weighted sampling
            weights = get_option_weights(parameter_config, self.custom_ranges)
            if weights and len(weights) == len(valid_values):
                # Normalize weights
                total_weight = sum(weights)
                probabilities = [w / total_weight for w in weights]
                sample = int(np.random.choice(valid_values, p=probabilities))
            else:
                sample = int(np.random.choice(valid_values))
            return sample

        if data_type in ("categorical", "ordered"):
            valid_values = get_valid_options(parameter_config, self.custom_ranges)
            if not valid_values or valid_values == "":
                return default_value
            # Get weights for weighted sampling
            weights = get_option_weights(parameter_config, self.custom_ranges)
            if weights and len(weights) == len(valid_values):
                # Normalize weights
                total_weight = sum(weights)
                probabilities = [w / total_weight for w in weights]
                sample = np.random.choice(valid_values, p=probabilities)
            else:
                sample = np.random.choice(valid_values)
            return sample

        if data_type == "subset_list":
            # Generate a random subset from valid_options
            valid_options = get_valid_options(parameter_config, self.custom_ranges)
            if valid_options == "" or valid_options is None:
                return []  # Return empty list if no valid options

            if isinstance(valid_options, str):
                valid_options = [valid_options]  # Convert single string to list
            elif isinstance(valid_options, list):
                pass  # Already a list
            else:
                return []

            # Randomly decide whether to include items (30% chance for empty list)
            selected_items = []
            if np.random.random() < 0.3:
                # Handle LoRA target_modules with constraints even for empty list
                if self.network == "cosmos-rl" and "target_modules" in parameter_name:
                    # Constraint function will return "all-linear" if list is empty
                    return network_utils.apply_lora_constraints(
                        self.parent_params, selected_items
                    )
                return selected_items
            # Randomly select 1 or more items
            num_items = np.random.randint(1, len(valid_options) + 1)
            selected_items = np.random.choice(valid_options, size=num_items, replace=False).tolist()
            # Handle LoRA target_modules with constraints (modules_to_save is already processed)
            if self.network == "cosmos-rl" and "target_modules" in parameter_name:
                # Apply LoRA-specific constraints - modules_to_save is already decided
                return network_utils.apply_lora_constraints(
                    self.parent_params, selected_items
                )

        if data_type == "optional_list":
            # Generate either None or a list with items from valid_options
            valid_options = get_valid_options(parameter_config, self.custom_ranges)
            if valid_options == "" or valid_options is None:
                result = None
            else:
                if isinstance(valid_options, str):
                    valid_options = [valid_options]  # Convert single string to list
                # 50% chance for None, 50% chance for list with all valid options
                if np.random.random() < 0.5:
                    result = None
                else:
                    result = valid_options.copy()  # Return all valid options

            # Store in parent_params for dependency tracking
            param_key = parameter_name.split('.')[-1]  # Get the last part (e.g., "modules_to_save")
            self.parent_params[param_key] = result

            return result

        if "list_1_" in data_type:
            if data_type == "list_1_backbone":
                # List needed in the form of consective numbers [1,2,3,4,5],
                # where the continuous numbers are decided by dependent parameters
                # Get backbone constant name from network_utils
                backbone_parameter = network_constants.backbone_mapper.get(self.network, "")
                backbone = self.parent_params.get(
                    backbone_parameter,
                    self.default_train_spec_flattened.get(backbone_parameter, None)
                )
                # Get the bounds from automl_helper
                bound_start, bound_end = (
                    automl_helper.automl_list_helper.get(self.network, {})
                    .get(data_type, {})
                    .get(parameter_name, {})
                    .get(backbone, {})
                )
            elif data_type == "list_1_normal":
                bound_start, bound_end = (
                    automl_helper.automl_list_helper.get(self.network, {})
                    .get(data_type, {})
                    .get(parameter_name, {})
                )
            else:
                return []
            # Generate two random numbers within the bounds
            random_number1 = random.randint(bound_start, bound_end)
            random_number2 = random.randint(bound_start, bound_end)
            # Make sure the numbers are in ascending order
            bound_start = min(random_number1, random_number2)
            bound_end = max(random_number1, random_number2)
            # Create a list of consecutive numbers between start_number and end_number
            automl_suggested_value = list(range(bound_start, bound_end + 1))
            return automl_suggested_value

        if data_type in ("list_2", "list_3"):
            automl_suggested_value = []
            helper_result = (
                automl_helper.automl_list_helper.get(self.network, {})
                .get(data_type, {})
                .get(parameter_name, {})
            )

            # Handle case where helper_result might be empty or not a tuple
            if not helper_result:
                logger.warning(f"No helper configuration found for {parameter_name} with type {data_type}")
                return []

            if isinstance(helper_result, dict) and len(helper_result) >= 2:
                bound_type, dependent_parameter = list(helper_result.items())[0]
            elif isinstance(helper_result, (list, tuple)) and len(helper_result) >= 2:
                bound_type, dependent_parameter = helper_result[0], helper_result[1]
            else:
                logger.warning(f"Invalid helper configuration for {parameter_name}: {helper_result}")
                return []

            if dependent_parameter is not None:
                bound_value = self.parent_params.get(
                    dependent_parameter,
                    self.default_train_spec_flattened.get(dependent_parameter, None)
                )
            else:
                bound_value = None

            if not bound_value:
                if bound_type == "img_size":
                    bound_value = 1080  # Default value considering a HD image
                elif bound_type == "lr_steps":
                    bound_value = 50  # Default value of 50 epochs
                elif bound_type == "optimizer_betas":
                    bound_value = None  # No bound needed for optimizer betas
                else:
                    return []

            # List needed in the form of multiple numbers operated with bounds
            if data_type == "list_2":
                if bound_type == "optimizer_betas":
                    # Generate two beta values: beta1 (momentum) and beta2 (RMSprop)
                    # Get ranges from schema (custom overrides already applied at function start)
                    schema_min = parameter_config.get("valid_min", [0.8, 0.9])
                    schema_max = parameter_config.get("valid_max", [0.95, 0.999])

                    # Ensure we have valid lists
                    if not isinstance(schema_min, list) or len(schema_min) < 2:
                        schema_min = [0.8, 0.9]
                    if not isinstance(schema_max, list) or len(schema_max) < 2:
                        schema_max = [0.95, 0.999]

                    beta1 = np.random.uniform(schema_min[0], schema_max[0])
                    beta2 = np.random.uniform(schema_min[1], schema_max[1])
                    automl_suggested_value = [beta1, beta2]
                    return automl_suggested_value

                # Generate a random number between 3 and 6 (inclusive) for other list_2 types
                num_random_numbers = random.randint(3, 6)
                # Generate a list of random numbers
                if bound_type == "lr_steps":
                    automl_suggested_value = [random.randint(1, bound_value) for _ in range(num_random_numbers)]
                    return sorted(automl_suggested_value)
                if bound_type == "img_size":
                    # Calculate the range of valid multiples of 16
                    min_multiple = max(bound_value // 2, 16)
                    min_multiple -= min_multiple % 16  # Ensure min_multiple is a multiple of 16
                    max_multiple = bound_value - (bound_value % 16)
                    # Calculate the number of valid multiples of 16 within the range
                    num_multiples = ((max_multiple - min_multiple) // 16) + 1
                    # Generate random multiples of 16
                    automl_suggested_value = [
                        min_multiple + 16 * random.randint(0, num_multiples - 1)
                        for _ in range(num_random_numbers)
                    ]
                    return sorted(automl_suggested_value)
                return []

            # List needed in the form of pair of same numbers lke [15,15]
            if data_type == "list_3":
                if bound_type == "img_size":
                    min_value = bound_value // 100  # 1/100th of the bound value
                    max_value = bound_value // 10   # 1/10th of the bound value
                    # Generate a random integer within the specified range
                    random_integer = random.randint(min_value, max_value)
                    if self.network == "ml_recog":
                        # For ml_recog, the random integer needs to be a odd number
                        if random_integer % 2 == 0:
                            random_integer += 1
                    automl_suggested_value = [random_integer, random_integer]
                    return automl_suggested_value
                return []

        if data_type in ("dict", "collection"):
            # Handle dictionary-type parameters via network-specific handlers
            return network_utils.apply_network_specific_param_logic(
                network=self.network,
                data_type=data_type,
                parameter_name=parameter_name,
                value=None,
                parent_params=self.parent_params
            )

        if data_type == "float":
            # Handle float parameters
            v_min = parameter_config.get("valid_min", "")
            v_max = parameter_config.get("valid_max", "")

            # If no valid range, generate values around default
            if v_min == "" or v_max == "":
                if default_value is not None and default_value != "":
                    default_val = float(default_value)
                    # Generate values in range [default/10, default*10] for diversity
                    if default_val > 0:
                        v_min = default_val / 10.0
                        v_max = default_val * 10.0
                    elif default_val < 0:
                        v_min = default_val * 10.0
                        v_max = default_val / 10.0
                    else:  # default is 0
                        v_min = -1.0
                        v_max = 1.0
                    random_float = np.random.uniform(v_min, v_max)
                    logger.info(
                        f"Generated random float for {parameter_name} (no range): "
                        f"{random_float} around default {default_val}"
                    )
                    return random_float
                # No default either, use reasonable range
                return np.random.uniform(0.0, 1.0)

            # Convert to float
            if v_min == "-inf":
                v_min = float('-inf')
            elif v_min != "":
                v_min = float(v_min)
            else:
                v_min = 0.0

            if v_max == "inf":
                v_max = float('inf')
            elif v_max != "":
                v_max = float(v_max)
            else:
                v_max = 1.0

            # Handle infinite bounds by using default or reasonable values
            if v_min == float('-inf'):
                v_min = 0.0
            if v_max == float('inf'):
                # Use default as max, or a reasonable value
                if default_value is not None and default_value != "":
                    v_max = float(default_value) * 10  # 10x default as upper bound
                else:
                    v_max = 1.0

            # Generate random float in range
            random_float = np.random.uniform(v_min, v_max)
            logger.info(f"Generated random float for {parameter_name}: {random_float} in range [{v_min}, {v_max}]")
            return random_float

        if data_type == "string":
            # Handle string parameters with valid_options
            valid_options = parameter_config.get("valid_options", "")
            if valid_options and valid_options != "":
                if isinstance(valid_options, str):
                    options = [opt.strip() for opt in valid_options.split(',')]
                else:
                    options = valid_options
                if options:
                    return np.random.choice(options)
            # No valid options, return default
            return default_value if default_value is not None else ""

        return default_value
