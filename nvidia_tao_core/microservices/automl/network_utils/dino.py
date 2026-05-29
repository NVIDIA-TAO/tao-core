# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DETR-based models AutoML parameter constraints

This module implements AutoML parameter constraints for DETR-based detection models:
- DINO
- Deformable DETR
- Grounding DINO
- RT-DETR

Constraint: num_select < num_queries * num_classes

During post-processing, these models perform:
    topk(logits.view(batch, -1), num_select)
where logits shape is [batch, num_queries, num_classes].
Therefore, num_select must be < num_queries * num_classes (strictly less than).
We enforce (num_queries * num_classes - 1) as the maximum to avoid edge cases.
"""

import logging

logger = logging.getLogger(__name__)


def apply_int_logic(parameter_name, value, v_max, default_train_spec, parent_params=None):
    """Apply DETR-based models logic for integer parameters.

    Enforces the constraint: num_select < num_queries * num_classes (strictly less than)
    Applies to: DINO, Deformable DETR, Grounding DINO, RT-DETR

    Args:
        parameter_name: The parameter name to check
        value: The generated parameter value
        v_max: Maximum valid value from schema
        default_train_spec: The original train specification
        parent_params: Dictionary of already-sampled parameters in this recommendation (optional)

    Returns:
        Constrained value or original value
    """
    # Handle num_select constraint
    if "num_select" in parameter_name:
        return _constrain_num_select(value, default_train_spec, parent_params)

    return value


def _constrain_num_select(value, default_train_spec, parent_params):
    """Constrain num_select to be <= num_queries (and < num_queries * num_classes for non-open-vocab models)."""
    # Get num_queries and num_classes, checking parent_params first (for AutoML-modified values)
    try:
        # Check parent_params first for AutoML-modified num_queries
        # Since num_select depends_on num_queries, it will be sampled first and available here
        if parent_params and "model.num_queries" in parent_params:
            num_queries = parent_params["model.num_queries"]
            logger.info(f"Using AutoML-modified num_queries={num_queries} from parent_params")
        else:
            # Fall back to default spec
            num_queries = default_train_spec.get("model", {}).get("num_queries", 300)
            logger.info(f"Using default num_queries={num_queries} from spec")

        # CRITICAL: For open-vocabulary models (Grounding DINO), num_select must be <= num_queries
        # because the number of labels can be as low as 1 at inference time (single caption)
        # For other models, we also apply num_classes constraint
        max_num_select = num_queries  # Primary constraint
        logger.info(f"Primary constraint: num_select <= num_queries ({num_queries})")

        # For non-open-vocabulary models, also check num_classes constraint
        # num_classes typically comes from dataset config (not modified by AutoML)
        if parent_params and "dataset.num_classes" in parent_params:
            num_classes = parent_params["dataset.num_classes"]
            logger.info(f"Using num_classes={num_classes} from parent_params")
        else:
            num_classes = default_train_spec.get("dataset", {}).get("num_classes", 91)
            logger.info(f"Using num_classes={num_classes} from experiment spec")

        # Secondary constraint: num_select < num_queries * num_classes (with safety margin)
        max_num_select_with_classes = max(1, (num_queries * num_classes) - 1)
        max_num_select = min(max_num_select, max_num_select_with_classes)
        logger.info(
            f"Final max_num_select={max_num_select} "
            f"(min of num_queries={num_queries} and (num_queries * num_classes - 1)={(num_queries * num_classes) - 1})"
        )

        # Always constrain num_select to be < (num_queries * num_classes)
        # This is a hard constraint to prevent runtime errors from torch.topk
        if value > max_num_select:
            logger.warning(
                f"CONSTRAINT VIOLATION: num_select={value} exceeds maximum {max_num_select}. "
                f"Capping to max_num_select={max_num_select} "
                f"(num_queries={num_queries} * num_classes={num_classes})"
            )
            value = max_num_select
        else:
            logger.info(f"num_select={value} is within valid range [1, {max_num_select}]")

    except (KeyError, AttributeError, TypeError) as e:
        logger.warning(f"Could not apply DINO num_select constraint: {e}")

    return value


# Handler registry for DETR-based networks
HANDLERS = {
    "int": apply_int_logic,
    "integer": apply_int_logic,
}
