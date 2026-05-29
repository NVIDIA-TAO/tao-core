# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos-RL specific AutoML utilities"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Vision model LoRA modules that should be excluded when visual modules are fully tuned
VISION_MODEL_LORA_MODULES = [
    "attn.qkv", "attn.proj"
]

# Common regex patterns for LoRA module matching
COMMON_LORA_PATTERNS = [
    "visual\\..*",      # Visual/vision tower modules
    "attn\\..*",        # Attention modules
    "mlp\\..*",         # MLP modules
    ".*proj",           # Projection layers
    ".*_proj",          # Projection layers with underscore
    "embed.*",          # Embedding layers
]


def apply_lora_constraints(parent_params: dict, selected_items: list):
    """Apply cosmos-rl LoRA constraint to target_modules valid_options.

    Since modules_to_save is processed first (parameter sorting), we know its value.
    Simple rule: If modules_to_save has "visual", exclude vision modules from valid_options.

    Args:
        parent_params: Dictionary of parent parameters already processed
        selected_items: List of selected LoRA target modules

    Returns:
        List of modules after applying constraints, or "all-linear" if list becomes empty
    """
    modules_to_save = parent_params.get("modules_to_save")
    if modules_to_save and "visual" in modules_to_save:
        # Remove vision modules from valid options
        selected_items = [opt for opt in selected_items if opt not in VISION_MODEL_LORA_MODULES]

    # If list becomes empty after filtering, return "all-linear" as a safe fallback
    if selected_items == []:
        logger.info("All target_modules filtered out by constraints. Returning 'all-linear' as fallback.")
        return "all-linear"

    return selected_items


def is_full_sft_training(default_train_spec):
    """Check if this is full SFT training (no LoRA) by checking original train spec.

    Args:
        default_train_spec: The original train specification dictionary

    Returns:
        bool: True if policy.lora is not present in config (full SFT), False otherwise
    """
    # Check if policy.lora exists in the original train spec
    if "policy" not in default_train_spec:
        return False

    policy_config = default_train_spec.get("policy", {})
    # If lora key is not present or is None/empty, it's full SFT
    lora_config = policy_config.get("lora", None)

    is_full_sft = lora_config is None or (isinstance(lora_config, dict) and len(lora_config) == 0)
    logger.info(f"Cosmos-RL training mode: {'Full SFT' if is_full_sft else 'LoRA'}")
    return is_full_sft


def generate_lora_pattern(base_value, param_type="alpha", valid_multipliers=None):
    """Generate random pattern dictionary for LoRA configuration.

    Args:
        base_value: The base value to use as reference
        param_type: Type of parameter - "alpha" (float) or "rank" (int)
        valid_multipliers: List of valid multipliers. If None, uses defaults based on param_type

    Returns:
        dict or None: Dictionary mapping regex patterns to values, or None
    """
    # 50% chance to return None (no pattern override)
    lora_pattern_chance = np.random.random()
    logger.info(f"Lora pattern chance: {lora_pattern_chance}")
    if lora_pattern_chance < 0.5:
        return None

    # Set default multipliers based on parameter type
    if valid_multipliers is None:
        if param_type == "alpha":
            valid_multipliers = [0.5, 1.0, 2.0, 4.0]
        else:  # rank
            valid_multipliers = [0.5, 1.0, 2.0]

    # Randomly select 1-3 patterns
    num_patterns = np.random.randint(1, 4)
    logger.info(f"Number of patterns: {num_patterns}")
    selected_patterns = np.random.choice(COMMON_LORA_PATTERNS, size=num_patterns, replace=False)
    logger.info(f"Selected patterns: {selected_patterns}")
    # Generate pattern values
    pattern_dict = {}
    for pattern in selected_patterns:
        # Convert numpy string to regular Python string
        pattern = str(pattern)
        multiplier = np.random.choice(valid_multipliers)
        value = base_value * multiplier

        if param_type == "alpha":
            # Float values for alpha
            value = float(value)
            value = max(1.0, value)  # Ensure minimum of 1.0
        else:  # rank
            # Integer values for rank, rounded to power of 2
            value = int(value)
            value = max(1, value)  # Ensure minimum of 1
            # Round to nearest power of 2
            if value > 1:
                value = 2 ** round(np.log2(value))

        pattern_dict[pattern] = value

    logger.info(f"Generated {param_type}_pattern: {pattern_dict}")
    return pattern_dict


# Default learning rate multiplier configuration for VLM model parts
# Model parts order: [language_model, vision_model, multi_modal_projector, lm_head]
# Each entry: (multiplier_min, multiplier_max) relative to base LLM learning rate
DEFAULT_LR_MULTIPLIERS = {
    "llm": {"multiplier_min": 1.0, "multiplier_max": 1.0},
    "vision": {"multiplier_min": 5.0, "multiplier_max": 20.0},
    "projector": {"multiplier_min": 2.0, "multiplier_max": 10.0},
    "lm_head": {"multiplier_min": 1.0, "multiplier_max": 2.0},
}

# Part names for logging
PART_NAMES = ["LLM", "Vision", "Projector", "LM_Head"]


def get_optm_lr_range_override(default_train_spec):
    """Extract optm_lr range override from automl_settings.

    Looks for train.optm_lr in automl_range_override and returns the
    valid_min and valid_max if they are lists (per-model-part bounds).

    Args:
        default_train_spec: The original train specification dictionary

    Returns:
        tuple: (valid_min_list, valid_max_list, num_parts) or (None, None, None)
    """
    automl_settings = default_train_spec.get("automl_settings", {})
    if not automl_settings:
        return None, None, None

    range_overrides = automl_settings.get("automl_range_override", [])
    if not range_overrides:
        return None, None, None

    for override in range_overrides:
        param_name = override.get("parameter", "")
        if param_name == "train.optm_lr":
            valid_min = override.get("valid_min")
            valid_max = override.get("valid_max")

            # Check if bounds are lists (per-model-part bounds)
            if isinstance(valid_min, list) and isinstance(valid_max, list):
                if len(valid_min) != len(valid_max):
                    logger.warning(
                        f"valid_min and valid_max lists have different lengths: "
                        f"{len(valid_min)} vs {len(valid_max)}. Using min length."
                    )
                num_parts = min(len(valid_min), len(valid_max))
                return valid_min[:num_parts], valid_max[:num_parts], num_parts
            if isinstance(valid_min, list):
                # Only valid_min is a list, use it for num_parts
                num_parts = len(valid_min)
                # Use valid_max as scalar for all parts or default to 1.0
                v_max = valid_max if valid_max is not None else 1.0
                return valid_min, [v_max] * num_parts, num_parts
            if isinstance(valid_max, list):
                # Only valid_max is a list, use it for num_parts
                num_parts = len(valid_max)
                # Use valid_min as scalar for all parts or default to 0
                v_min = valid_min if valid_min is not None else 0
                return [v_min] * num_parts, valid_max, num_parts

    return None, None, None


def get_num_model_parts(default_train_spec):
    """Determine the number of model parts from the train spec.

    Priority order:
    1. automl_range_override for train.optm_lr (if valid_min/valid_max are lists)
    2. Existing optm_lr list length in train config
    3. Default: 3 for VLM (llm, vision, projector)

    For VLM models, model parts can be:
    - 2 parts: [language_model, vision_model]
    - 3 parts: [language_model, vision_model, multi_modal_projector]
    - 4 parts: [language_model, vision_model, multi_modal_projector, lm_head]

    Args:
        default_train_spec: The original train specification dictionary

    Returns:
        int: Number of model parts (1 for non-VLM, 2-4 for VLM)
    """
    # Check automl_range_override first
    _, _, num_parts_from_override = get_optm_lr_range_override(default_train_spec)
    if num_parts_from_override is not None:
        return num_parts_from_override

    # Check for existing optm_lr list length to infer model parts
    train_config = default_train_spec.get("train", {})
    existing_lr = train_config.get("optm_lr")
    if isinstance(existing_lr, list):
        return len(existing_lr)

    # Default: assume 3 model parts for VLM (llm, vision, projector)
    # This is the common case for models like Cosmos-Reason2
    return 3


def generate_lr_list_from_bounds(valid_min_list, valid_max_list):
    """Generate learning rates by sampling from per-part bounds.

    Args:
        valid_min_list: List of minimum learning rates per model part
        valid_max_list: List of maximum learning rates per model part

    Returns:
        list: List of sampled learning rates for each model part
    """
    lr_list = []
    for i, (v_min, v_max) in enumerate(zip(valid_min_list, valid_max_list)):
        v_min = float(v_min)
        v_max = float(v_max)

        if v_min >= v_max:
            part_lr = v_min
        else:
            # Log-uniform sampling for learning rates (better for LR search)
            log_min = np.log10(v_min) if v_min > 0 else -10
            log_max = np.log10(v_max) if v_max > 0 else -10
            part_lr = float(10 ** np.random.uniform(log_min, log_max))

        lr_list.append(part_lr)
        logger.debug(f"Part {i} ({PART_NAMES[i] if i < len(PART_NAMES) else 'Extra'}): "
                     f"sampled LR={part_lr:.2e} from [{v_min:.2e}, {v_max:.2e}]")

    return lr_list


def generate_lr_list_from_multipliers(base_lr, num_parts):
    """Generate learning rates using multipliers relative to base LR.

    Args:
        base_lr: Base learning rate (typically for LLM)
        num_parts: Number of model parts

    Returns:
        list: List of learning rates for each model part
    """
    base_lr = float(base_lr)
    part_keys = ["llm", "vision", "projector", "lm_head"]
    lr_list = []

    for i in range(num_parts):
        part_key = part_keys[i] if i < len(part_keys) else part_keys[-1]
        config = DEFAULT_LR_MULTIPLIERS.get(part_key, DEFAULT_LR_MULTIPLIERS["llm"])

        mult_min = config["multiplier_min"]
        mult_max = config["multiplier_max"]

        if mult_min == mult_max:
            multiplier = mult_min
        else:
            multiplier = float(np.random.uniform(mult_min, mult_max))

        part_lr = float(base_lr * multiplier)
        part_lr = min(part_lr, 1.0)  # Cap at 1.0 for safety
        lr_list.append(part_lr)

    return lr_list


def apply_optm_lr_logic(parameter_name, lr_value, v_max, default_train_spec, parent_params=None):
    """Apply cosmos-rl specific logic for optm_lr parameter.

    For full SFT training (no LoRA), generates learning rates for each model part.
    VLM models can have 2-4 parts: [llm, vision, projector, lm_head]

    Learning rate bounds can be specified via automl_range_override:
    - If valid_min/valid_max are lists, they define per-part bounds
    - Otherwise, uses default multipliers relative to base LR

    Args:
        parameter_name: The parameter name to check
        lr_value: The generated learning rate value
        v_max: Maximum valid value for learning rate
        default_train_spec: The original train specification to check for LoRA presence
                           and automl_range_override for per-part bounds
        parent_params: Dictionary of already-sampled parameters (optional)

    Returns:
        Either the original lr_value (float) or a list of learning rates per model part
    """
    # Only apply for cosmos-rl optm_lr parameter
    if parameter_name != "train.optm_lr":
        return lr_value

    # Check if this is full SFT training (no LoRA in original config)
    if not is_full_sft_training(default_train_spec):
        return lr_value

    # 80% chance to use multi-part learning rates for vision-language models
    multi_lr_chance = np.random.random()
    logger.info(f"Multi-part learning rates chance: {multi_lr_chance}")
    if multi_lr_chance < 0.5:
        # Check for per-part bounds from automl_range_override
        valid_min_list, valid_max_list, num_parts = get_optm_lr_range_override(default_train_spec)

        if valid_min_list is not None and valid_max_list is not None:
            # Use explicit per-part bounds from automl_range_override
            logger.info(f"Using per-part LR bounds from automl_range_override: "
                        f"num_parts={num_parts}")
            lr_list = generate_lr_list_from_bounds(valid_min_list, valid_max_list)
        else:
            # Use default multipliers relative to base LR
            num_parts = get_num_model_parts(default_train_spec)
            logger.info(f"Using default LR multipliers: num_parts={num_parts}")
            lr_list = generate_lr_list_from_multipliers(lr_value, num_parts)

        lr_log = ", ".join([
            f"{PART_NAMES[i] if i < len(PART_NAMES) else f'Part{i}'}={lr:.2e}"
            for i, lr in enumerate(lr_list)
        ])
        logger.info(f"Generated multi-part learning rates for Full SFT: {lr_log}")
        return lr_list

    return lr_value


def apply_dict_logic(parameter_name, value, parent_params):
    """Apply cosmos-rl specific logic for dict parameters (LoRA patterns).

    Generates alpha_pattern or r_pattern dictionaries based on the base values.

    Args:
        parameter_name: The parameter name to check
        value: The generated parameter value (not used, we generate new dict)
        parent_params: Dictionary of parent parameters to get base values

    Returns:
        Generated pattern dictionary or None
    """
    logger.info(f"Applying dict logic for parameter: {parameter_name}")
    logger.info(f"Value: {value}")
    logger.info(f"Parent params: {parent_params}")
    if "alpha_pattern" in parameter_name:
        # Get the base lora_alpha value
        base_alpha = parent_params.get("policy.lora.lora_alpha", 8)
        return generate_lora_pattern(base_alpha, param_type="alpha")

    if "r_pattern" in parameter_name:
        # Get the base r value
        base_r = parent_params.get("policy.lora.r", 8)
        return generate_lora_pattern(base_r, param_type="rank")

    # For other dict types, 80% chance to return None
    lora_pattern_chance = np.random.random()
    logger.info(f"Lora pattern chance: {lora_pattern_chance}")
    if lora_pattern_chance < 0.8:
        return None
    return {}


# Register handlers for cosmos-rl
HANDLERS = {
    "float": apply_optm_lr_logic,
    "dict": apply_dict_logic,
    "collection": apply_dict_logic,
}
