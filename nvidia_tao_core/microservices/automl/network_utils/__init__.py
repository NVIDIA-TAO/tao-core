# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Network-specific AutoML utilities

This package contains network-specific logic for AutoML parameter generation.
Each network has its own module with custom handlers that can be registered
in the dispatcher.

Structure:
- common.py: Shared utilities across all networks
- cosmos_rl.py: Cosmos-RL specific handlers
- __init__.py: Dispatcher that routes to network-specific handlers
"""

import logging

# Import network modules
from nvidia_tao_core.microservices.automl.network_utils import cosmos_rl
from nvidia_tao_core.microservices.automl.network_utils import dino

logger = logging.getLogger(__name__)

# Network-specific parameter handlers registry
# Maps network name -> data type -> handler function
NETWORK_PARAM_HANDLERS = {
    "cosmos-rl": cosmos_rl.HANDLERS,
    "dino": dino.HANDLERS,
    "deformable_detr": dino.HANDLERS,  # Same constraint as DINO
    "grounding_dino": dino.HANDLERS,  # Same constraint as DINO
    "rtdetr": dino.HANDLERS,  # Same constraint as DINO
}

# Export cosmos-rl functions for backward compatibility
apply_lora_constraints = cosmos_rl.apply_lora_constraints
generate_lora_pattern = cosmos_rl.generate_lora_pattern


def apply_network_specific_param_logic(
    network,
    data_type,
    parameter_name,
    value,
    v_max=None,
    default_train_spec=None,
    parent_params=None
):
    """Apply network-specific parameter generation logic.

    This is a dispatcher that routes to network-specific handlers based on the network name
    and parameter data type. Allows each network to customize parameter generation behavior.

    Args:
        network: Network name (e.g., "cosmos-rl", "detectnet_v2", etc.)
        data_type: Parameter data type (e.g., "float", "int", "dict", etc.)
        parameter_name: Full parameter name (e.g., "train.optm_lr")
        value: The generated parameter value
        v_max: Maximum valid value (optional, used for some handlers)
        default_train_spec: The original train specification (optional, for network-specific checks)
        parent_params: Dictionary of parent parameters (optional, for dependent params)

    Returns:
        Modified value or original value if no network-specific logic applies
    """
    # Check if network has any registered handlers
    if network not in NETWORK_PARAM_HANDLERS:
        return value

    network_handlers = NETWORK_PARAM_HANDLERS[network]

    # Check if there's a handler for this data type
    if data_type not in network_handlers:
        return value

    # Call the appropriate handler with appropriate arguments based on handler signature
    handler = network_handlers[data_type]

    # Different handlers need different arguments
    if data_type in ("dict", "collection"):
        result = handler(parameter_name, value, parent_params)
    else:
        result = handler(parameter_name, value, v_max, default_train_spec, parent_params)

    # Log if handler modified the value (handle both scalar and list types)
    try:
        # For lists/arrays, always log as modified. For scalars, check equality
        if isinstance(result, (list, tuple)):
            logger.info(f"Applied {network} network-specific logic for {parameter_name}")
        elif result != value:
            logger.info(f"Applied {network} network-specific logic for {parameter_name}")
    except (ValueError, TypeError):
        # If comparison fails, assume they're different
        logger.info(f"Applied {network} network-specific logic for {parameter_name}")

    return result
