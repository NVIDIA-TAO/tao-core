# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoML read parameters modules"""
import pandas as pd

from nvidia_tao_core.microservices.constants import AUTOML_DISABLED_NETWORKS
from nvidia_tao_core.microservices.utils.handler_utils import get_flatten_specs
from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_job_specs
from nvidia_tao_core.scripts.generate_schema import generate_schema
from nvidia_tao_core.microservices.utils.core_utils import get_microservices_network_and_action

import logging
logger = logging.getLogger(__name__)

_VALID_TYPES = ["int", "integer",
                "float",
                "ordered_int", "bool",
                "ordered", "categorical",
                "list_1_backbone", "list_1_normal", "list_2", "list_3", "subset_list", "optional_list",
                "collection", "dict"]


def flatten_properties(data, parent_key='', sep='.'):
    """Convert schema to a readable dict"""
    flattened = {}

    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict) and v.get('type') in ('object', 'collection', 'dict') and 'properties' in v:
            # Recurse if the value has nested properties
            flattened.update(flatten_properties(v['properties'], new_key, sep))
        elif isinstance(v, dict):
            # Otherwise, gather metadata
            dtype = v.get('type', '')

            # Handle union types (anyOf schemas)
            if 'anyOf' in v and not dtype:
                # For union types, determine the primary type from anyOf
                any_of_types = v.get('anyOf', [])
                if any_of_types:
                    # Use the first type as the primary type for AutoML
                    first_type = any_of_types[0].get('type', '')
                    if first_type == 'integer':
                        dtype = 'int'
                    elif first_type == 'number':
                        dtype = 'float'
                    elif first_type == 'boolean':
                        dtype = 'bool'
                    else:
                        dtype = first_type

            if v.get('type', '') == "number":
                dtype = "float"
            if v.get('type', '') == "boolean":
                dtype = "bool"
            flattened[new_key] = {
                'parameter': new_key,
                'value_type': dtype,
                'default_value': v.get('default', ''),
                'valid_min': v.get('minimum', ''),
                'valid_max': v.get('maximum', ''),
                'valid_options': v.get('enum', []),
                'option_weights': v.get('option_weights', None),
                'automl_enabled': v.get('automl_enabled', ''),
                'math_cond': v.get('math_cond', ''),
                'parent_param': v.get('parent_param', ''),
                'depends_on': v.get('depends_on', ''),
            }

    return flattened


def generate_hyperparams_to_search(
    job_context,
    automl_hyperparameters,
    handler_root,
    override_automl_disabled_params=False
):
    """This function determines which hyperparameters should be included in the AutoML search space for a training job.

    It works by extracting the hyperparameters from the network's JSON schema (train spec), checking which are
    currently present in the updated configuration, accounting for removed/deleted or conditionally-excluded parameters,
    and then marking which parameters (from automl_hyperparameters) should be enabled for AutoML.

    Note: The function handles network-specific logic (e.g., for 'cosmos-rl' excluding absent LoRA blocks)
    and returns a tuple: (list of parameter dicts, list of parameter names).

    Returns: (list of parameter dicts, list of parameter names) where parameter dicts are returned in a list format.

    Note: Parameter names use dot notation to represent nested paths in the spec structure
    (e.g., "train.optim.lr", "dataset.batch_size", "model.num_queries").

    Example:
        (
            [
                {
                    'parameter': 'train.optim.lr',
                    'value_type': 'float',
                    'default_value': 0.001,
                    'valid_min': 0.0001,
                    'valid_max': 0.01,
                    'valid_options': [0.0001, 0.001, 0.01],
                    'option_weights': [1, 1, 1],
                    'automl_enabled': True,
                    'math_cond': '',
                    'parent_param': '',
                    'depends_on': ''
                },
                {
                    'parameter': 'dataset.batch_size',
                    'value_type': 'int',
                    'default_value': 4,
                    'valid_min': 1,
                    'valid_max': 32,
                    'automl_enabled': True,
                    ...
                },
                # ... more param dicts ...
            ],
            ['train.optim.lr', 'dataset.batch_size', ...]
        )

    The returned parameter names are used as flat dictionary keys in recommendation specs,
    not as nested paths. For example, rec.specs["train.optim.lr"] accesses the value directly,
    not rec.specs["train"]["optim"]["lr"].
    """
    network_arch, _ = get_microservices_network_and_action(job_context.network, job_context.action)
    logger.info(f"Network arch: {network_arch}")
    if network_arch not in AUTOML_DISABLED_NETWORKS:
        try:
            json_schema = generate_schema(network_arch, "train")
        except Exception as e:
            logger.info(f"Error generating schema for network: {network_arch}")
            logger.info(f"Job Context Network: {job_context.network}")
            logger.info(f"Job Context Action: {job_context.action}")
            raise Exception(e) from e

        original_train_spec = json_schema.get("default", {})
        original_spec_with_keys_flattened = {}
        get_flatten_specs(original_train_spec, original_spec_with_keys_flattened)

        updated_train_spec = get_job_specs(job_context.id)
        updated_spec_with_keys_flattened = {}
        get_flatten_specs(updated_train_spec, updated_spec_with_keys_flattened)

        deleted_params = original_spec_with_keys_flattened.keys() - updated_spec_with_keys_flattened

        format_json_schema = flatten_properties(json_schema["properties"])

        # Check if specific parent objects exist in the updated spec (e.g., policy.lora for cosmos-rl)
        # If not, exclude all parameters under that parent
        params_to_exclude = set()

        # For cosmos-rl: if policy.lora.* parameters are not in the spec, exclude all policy.lora.* parameters
        if network_arch == "cosmos-rl":
            # Check if ANY policy.lora.* key exists in the flattened spec
            has_lora_params = any(key.startswith("policy.lora.") for key in updated_spec_with_keys_flattened)
            if not has_lora_params:
                logger.info("policy.lora not found in updated spec - excluding LoRA parameters from AutoML")
                # Filter schema parameters that start with policy.lora.
                params_to_exclude.update([p for p in format_json_schema.keys() if p.startswith("policy.lora.")])
                logger.info(f"Excluding {len(params_to_exclude)} LoRA parameters: {params_to_exclude}")

        data_frame = pd.DataFrame.from_dict(format_json_schema, orient='index').reset_index()
        data_frame = data_frame[data_frame['value_type'].isin(_VALID_TYPES)]

        # Optionally, filter based on `automl_enabled` flag if provided in your data
        # (default=True in `flatten_properties`)
        if not override_automl_disabled_params:
            data_frame = data_frame.loc[data_frame['automl_enabled'] != False]  # pylint: disable=C0121  # noqa: E712

        data_frame['automl_enabled'] = False

        # Set `automl_enabled` for specific parameters in automl_hyperparameters
        data_frame.loc[data_frame.parameter.isin(automl_hyperparameters), 'automl_enabled'] = True

        # Filter for automl-enabled and non-deleted parameters
        automl_params = data_frame.loc[data_frame['automl_enabled'] == True]  # pylint: disable=C0121  # noqa: E712
        automl_params = automl_params.loc[~automl_params['parameter'].isin(deleted_params)]
        automl_params = automl_params.loc[~automl_params['parameter'].isin(params_to_exclude)]

        # Sort automl parameters: push params that are dependent on other params to the bottom
        # Use na_position='first' to put NaN (no depends_on) first, non-NaN (has depends_on) last
        automl_params = automl_params.sort_values(by=['depends_on'], na_position='first')

        # Select the required columns
        automl_params = automl_params[[
            "parameter",
            "value_type",
            "default_value",
            "valid_min",
            "valid_max",
            "valid_options",
            "option_weights",
            "math_cond",
            "parent_param",
            "depends_on"
        ]]
        logger.info(f"Automl params enabled: {automl_params['parameter'].values}")
        return automl_params.to_dict('records'), list(automl_params["parameter"].values)
    return [{}], []
