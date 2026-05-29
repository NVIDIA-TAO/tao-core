# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Original source taken from https://github.com/NVIDIA/NeMo

"""Utility module to validate json-schema from the api."""

import sys


def validate_union_type(_value, union_schemas, hierarchy_str):
    """Validate value against Union type schemas (anyOf)"""
    for schema in union_schemas:
        if "type" in schema:
            schema_type = schema["type"]
            # Try to validate against this type
            if schema_type == "integer" and isinstance(_value, int):
                return None
            if schema_type == "number" and isinstance(_value, (int, float)):
                return None
            if schema_type == "boolean" and isinstance(_value, bool):
                return None
            if schema_type == "string" and isinstance(_value, str):
                # If enum is specified (for literal values), check if value is in the enum
                if "enum" in schema:
                    if _value in schema["enum"]:
                        return None
                else:
                    return None

    # If none of the union types matched, return error
    valid_types = []
    for schema in union_schemas:
        if "enum" in schema:
            valid_types.extend(schema["enum"])
        else:
            valid_types.append(schema.get("type", "unknown"))

    return (
        f"Type Error : {hierarchy_str} should be one of {valid_types}, "
        f"but got {type(_value).__name__} with value '{_value}'"
    )


def validate_schema(_value, _properties, hierarchy):
    """Validate schema based on each parameters data type"""
    if isinstance(_value, dict):
        for _value_key, _value_obj in _value.items():
            if _value_key == 'automl_default_parameters' or "properties" not in _properties:
                continue

            if _value_key not in _properties["properties"]:
                hierarchy_str = ".".join(hierarchy)
                return (
                    f"Invalid schema : key = {_value_key} not present in the "
                    f"{hierarchy_str} config specs. "
                )

            hierarchy.append(_value_key)
            status = validate_schema(_value_obj, _properties["properties"][_value_key], hierarchy)
            hierarchy.pop()
            if status:
                return status

    hierarchy_str = ".".join(hierarchy)

    if _properties:
        # Handle Union types (anyOf)
        if "anyOf" in _properties:
            union_error = validate_union_type(_value, _properties["anyOf"], hierarchy_str)
            if union_error:
                return union_error

        # type check
        if "type" in _properties:
            if _properties["type"] in ("integer", "ordered_int") and not isinstance(_value, int):
                return (
                    f"Type Error : {hierarchy_str} should be of type integer. "
                )

            if _properties["type"] == "number" and not (isinstance(_value, (float, int))):
                return (
                    f"Type Error : {hierarchy_str} should be of type float. "
                )

            if _properties["type"] == "boolean" and not isinstance(_value, int):
                return (
                    f"Type Error : {hierarchy_str} should be of type boolean. "
                )

            if _properties["type"] == ("string", "categorical") and not isinstance(_value, str):
                return (
                    f"Type Error : {hierarchy_str} should be of type string. "
                )

            # valid_min range check
            if _properties["type"] == "number" and "minimum" in _properties:
                if float(_value) < float(_properties["minimum"]):
                    return (
                        f"Invalid schema : {hierarchy_str} should be >= "
                        f"{str(_properties['minimum'])}, current value is {_value}"
                    )

            # valid_max range check
            if _properties["type"] == "number" and "maximum" in _properties:
                if float(_value) > float(_properties["maximum"]):
                    return (
                        f"Invalid schema : {hierarchy_str} should be <= "
                        f"{str(_properties['maximum'])}, current value is '{_value}'"
                    )

        # valid_options check
        if "enum" in _properties and _value not in _properties["enum"]:
            if 'None' in _properties["enum"] and _value in (None, ''):
                return None
            return (
                f"Invalid schema : Allowed values for the {hierarchy_str} "
                f"are {_properties['enum']}, current value is '{_value}'"
            )

    return None


# json schema validation script
def validate_jsonschema(json_schema, json_metadata):
    """JSON schema validation function"""
    hierarchy = []

    try:
        for key, value_obj in json_schema.items():
            if key == 'automl_default_parameters':
                continue

            if key not in json_metadata.keys():
                return f"Invalid schema : key = {key} not present in the config specs."

            hierarchy.append(key)
            status = validate_schema(value_obj, json_metadata[key], hierarchy)
            hierarchy.pop()
            if status:
                return status
        return None
    except Exception as err:
        tb = sys.exception().__traceback__
        return f"Invalid schema : {err.with_traceback(tb)}"
