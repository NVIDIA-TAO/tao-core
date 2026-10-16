# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generating JSON schemas"""

import logging
from nvidia_tao_core.api_utils import dataclass2json_converter
from nvidia_tao_core.microservices import enum_constants

# Configure logger for this module
logger = logging.getLogger(__name__)


def generate_schema(neural_network_name, action=""):
    """Generates JSON schema for network"""
    if neural_network_name == "cosmos-rl":
        imported_module = dataclass2json_converter.import_module_from_path(
            f"nvidia_tao_core.config.{neural_network_name}.{action}"
        )
        expConfig = imported_module.ExperimentConfig()
    else:
        imported_module = dataclass2json_converter.import_module_from_path(
            f"nvidia_tao_core.config.{neural_network_name}.default_config"
        )
        if neural_network_name == "bevfusion" and action == "dataset_convert":
            expConfig = imported_module.BEVFusionDataConvertExpConfig()
        elif neural_network_name == "stylegan_xl" and action == "dataset_convert":
            imported_module = dataclass2json_converter.import_module_from_path(
                f"nvidia_tao_core.config.{neural_network_name}.dataset"
            )
            expConfig = imported_module.DataConvertExpConfig()
        elif neural_network_name == "clip":
            expConfig = imported_module.CLIPExperimentConfig()
        elif neural_network_name == "video_clip":
            expConfig = imported_module.VideoCLIPExperimentConfig()
        else:
            expConfig = imported_module.ExperimentConfig()
    json_with_meta_config = dataclass2json_converter.dataclass_to_json(expConfig)
    schema = dataclass2json_converter.create_json_schema(json_with_meta_config)
    # Only keep relevant top-level keys
    valid_actions = enum_constants._get_valid_config_json_param_for_network(neural_network_name, "actions")
    schema = filter_schema(schema, valid_actions, action)
    return schema


def filter_schema(schema, valid_actions, current_action):
    """Filter the schema to only include the allowed keys"""
    # Always keep 'train' and the current action, plus all non-action keys
    allowed_keys = set(['train', 'distill', 'quantize', current_action])
    # Add all non-action keys (not in valid_actions)
    allowed_keys.update([k for k in schema['properties'] if k not in valid_actions])

    # Filter top-level properties and default
    schema['properties'] = {k: v for k, v in schema['properties'].items() if k in allowed_keys}
    schema['default'] = {k: v for k, v in schema['default'].items() if k in allowed_keys}
    return schema


def validate_and_clean_merged_spec(original_schema, merged_spec):
    """Validate merged spec against original schema and remove invalid keys.

    Args:
        original_schema (dict): The original JSON schema with properties definition
        merged_spec (dict): The merged specification that may contain invalid keys

    Returns:
        dict: Cleaned specification with only valid keys according to the schema
    """
    if not isinstance(original_schema, dict) or not isinstance(merged_spec, dict):
        return merged_spec

    schema_properties = original_schema.get('properties', {})
    if not schema_properties:
        return merged_spec

    corrupted_keys = []  # Track all corrupted keys found

    def clean_nested_dict(spec_dict, schema_props, path=""):
        """Recursively clean nested dictionaries based on schema properties."""
        if not isinstance(spec_dict, dict) or not isinstance(schema_props, dict):
            return spec_dict

        cleaned_dict = {}
        for key, value in spec_dict.items():
            current_path = f"{path}.{key}" if path else key

            if key in schema_props:
                # Key is valid according to schema
                key_schema = schema_props[key]

                # If the value is a dict and the schema defines nested properties, recurse
                if (isinstance(value, dict) and
                        isinstance(key_schema, dict) and
                        'properties' in key_schema):
                    cleaned_dict[key] = clean_nested_dict(value, key_schema['properties'], current_path)
                else:
                    # Keep the value as-is if it's not a nested dict or no nested schema
                    cleaned_dict[key] = value
            else:
                # Key is not in schema, it gets dropped (this handles corrupt base_experiment_spec)
                corrupted_keys.append(current_path)

        return cleaned_dict

    cleaned_spec = clean_nested_dict(merged_spec, schema_properties)

    # Log corrupted keys if any were found
    if corrupted_keys:
        logger.warning(
            "Found and removed %d corrupted keys from base_experiment_spec: %s",
            len(corrupted_keys),
            ", ".join(corrupted_keys)
        )

    return cleaned_spec
