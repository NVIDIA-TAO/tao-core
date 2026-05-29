# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataclass to JSON converter"""

import requests
import importlib
import json
import os
import logging
from dataclasses import asdict, fields, is_dataclass

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)

__type_mapping = {
    "collection": "object",
    "list": "array",
    "list_1_backbone": "array",
    "list_1_normal": "array",
    "list_2": "array",
    "list_3": "array",
    "subset_list": "array",
    "optional_list": "array",
    "float": "number",
    "bool": "boolean",
    "integer": "integer",
    "string": "string",
    "str": "string",
    "int": "integer",
    "dict": "object",
    "const": "const",
    "ordered": "ordered",
    "categorical": "categorical",
    "ordered_int": "ordered_int",
    "enum": "string",
    "union": "union",
}


def __union_type_fix(union_types, value, literal_values=None):
    """Converts specification values for Union types by trying each type in order.

    Args:
        union_types (list): List of type names that this Union field can accept.
        value (any): The value to be converted.
        literal_values (list, optional): List of specific literal string values allowed.

    Returns:
        Converted value (various types): The value converted to the appropriate datatype, or the original value.
    """
    if value in (None, ""):
        return None

    # If literal_values are specified and value is a string, check if it's in the allowed literals
    if literal_values and isinstance(value, str) and value in literal_values:
        return value

    # Try each type in the union until one works
    for type_name in union_types:
        try:
            if type_name in ("bool", "boolean"):
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    if value.lower() in ("true", "false"):
                        return str(value).lower() == "true"
            elif type_name in ("int", "integer"):
                if isinstance(value, int):
                    return value
                if isinstance(value, str) and value.isdigit():
                    return int(value)
            elif type_name in ("float", "number"):
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    return float(value)
            elif type_name in ("list", "array"):
                if isinstance(value, list):
                    return value
                # Try to parse string representation of list
                if isinstance(value, str):
                    try:
                        import ast
                        parsed = ast.literal_eval(value)
                        if isinstance(parsed, list):
                            return parsed
                    except (ValueError, SyntaxError):
                        pass
            elif type_name in ("str", "string"):
                if isinstance(value, str):
                    # If literal_values are specified, only allow those specific values
                    if literal_values is None or value in literal_values:
                        return value
                return str(value)
        except (ValueError, TypeError):
            continue

    # If no type conversion worked, return the original value
    return value


def __basic_type_fix(value_type, value, union_types=None, literal_values=None):
    """Converts specification values to appropriate types based on their datatype.

    Args:
        value_type (str): The expected type of the value (e.g., 'integer', 'number').
        value (str or int or float or bool): The value to be converted.
        union_types (list, optional): List of union types if value_type is 'union'.
        literal_values (list, optional): List of specific literal string values allowed.

    Returns:
        Converted value (various types): The value converted to the appropriate datatype, or None for invalid inputs.
    """
    if value_type == "union" and union_types:
        return __union_type_fix(union_types, value, literal_values)
    if value_type == "string" and not value:
        return "" if value == "" else None
    if value in (None, ""):
        return None
    if value in ("inf", "-inf"):
        return float(value)
    if value_type in ("integer", "ordered_int"):
        return int(value)
    if value_type == "number":
        return float(value)
    if value_type == "boolean":
        return str(value).lower() == "true"
    if value_type == "array":
        return value
    if value_type == "object":
        return value
    return value


def __array_type_fix(value_type, value):
    """Converts specification values within an array to their appropriate types based on their datatype.

    Args:
        value_type (str): The expected type of the individual values in the array.
        value (str): A string representing the array, with values separated by commas.

    Returns:
        list: List with each item converted to the appropriate datatype, or None for invalid inputs.
    """
    if value in (None, ""):
        return None

    # Handle case where value is already a list (for new field types)
    if isinstance(value, list):
        values = value
    else:
        values = value.replace(" ", "").split(",")
    if value_type in ("integer", "ordered_int"):
        return [int(i) for i in values]
    if value_type == "number":
        return [float(i) for i in values]
    if value_type == "boolean":
        return [str(i).lower() == "true" for i in values]
    if value_type == "array":
        return [json.loads(i) for i in values]
    if value_type == "object":
        return [json.loads(i) for i in values]
    return values


def __merge(d1, d2):
    """Merges two dictionaries recursively. Modifies the first dictionary to include values from the second.

    Args:
        d1 (dict): The first dictionary to merge.
        d2 (dict): The second dictionary whose values are merged into the first.

    Returns:
        dict: The modified first dictionary with merged values.
    """
    for key in d2.keys():
        if key not in d1:
            d1[key] = d2[key]
        elif d1[key] is None:
            d1[key] = d2[key]
        elif type(d1[key]) is list and type(d2[key]) is list:
            if d1[key] != [] and type(d1[key][0]) is dict:
                for i in range(0, min(len(d1[key]), len(d2[key]))):
                    __merge(d1[key][i], d2[key][i])
            else:
                d1[key] = d1[key] + [i for i in d2[key] if i not in d1[key]]
        elif type(d2[key]) is not dict:
            d1[key] = d2[key]
        else:
            __merge(d1[key], d2[key])
    return d1


def dataclass_to_json_without_metadata(dataclass_instance, filename):
    """Serializes a dataclass instance to a JSON file without including any metadata.

    Args:
        dataclass_instance (dataclass): The dataclass instance to serialize.
        filename (str): The name of the output JSON file.

    Returns:
        None: Writes output to a file. Prints error message on failure.
    """
    try:
        # Convert the dataclass to a dictionary
        dataclass_dict = asdict(dataclass_instance)

        # Write the dictionary to a JSON file
        with open(f"./json_schema/{filename}", "w", encoding='utf-8') as output_file:
            json.dump(dataclass_dict, output_file, indent=4)

    except TypeError as e:
        logger.error("Error during serialization: %s", e)

    except IOError as e:
        logger.error("Error writing file: %s", e)


def serialize_with_metadata(dataclass_instance):
    """Serializes a dataclass instance to a dictionary, including its metadata.

    Args:
        dataclass_instance (dataclass): The dataclass instance to serialize.

    Returns:
        result (dict): Dictionary representation of the dataclass, including its metadata.
    """
    result = {}
    for field in fields(dataclass_instance):
        value = getattr(dataclass_instance, field.name)

        # Check if the value is itself a dataclass and serialize it recursively
        if is_dataclass(value):
            value = serialize_with_metadata(value)

        # Convert metadata MappingProxyType to a dict
        metadata = dict(field.metadata)

        # Include the metadata if it's not empty
        if metadata:
            result[field.name] = {"value": value, "metadata": metadata}
        else:
            result[field.name] = value

    return result


def dataclass_to_json(dataclass_instance):
    """Serializes a dataclass instance to a JSON-like dictionary, including metadata.

    Args:
        dataclass_instance (dataclass): The dataclass instance to serialize.

    Returns:
        result (dict): Dictionary representation of the dataclass, including metadata. Prints error message on failure.
    """
    try:
        # Serialize the dataclass to a dictionary including metadata
        dataclass_dict = serialize_with_metadata(dataclass_instance)

        # # Write the dictionary to a JSON file
        # with open(f"./json_schema/{filename}", "w") as output_file:
        #     json.dump(dataclass_dict, output_file, indent=4)
        return dataclass_dict

    except TypeError as e:
        logger.error("Error during serialization: %s", e)

    except IOError as e:
        logger.error("Error writing file: %s", e)

    return None


def auto_ml_parameters_fix(json_schema):
    """Fixes the auto-ml parameters in the given JSON schema.

    Args:
        json_schema (dict): The JSON schema to be fixed.

    Returns:
        Fixed schema (dict): The fixed JSON schema with correct auto-ml parameters.
    """
    def update_specs(key, obj, parentObj):
        if type(obj) is not dict:
            return

        automl_flag = False
        for key_name in ["automl_default_parameters", "automl_disabled_parameters"]:
            if key_name in obj:
                automl_flag = True
                if key == "default":
                    parentObj[key_name] = obj[key_name]
                    del obj[key_name]
                else:
                    del obj[key_name]
        if automl_flag:
            return

        if obj.get("properties") == {}:
            del obj["properties"]

        if obj.get("default") == {}:
            del obj["default"]

        for k in list(obj):
            update_specs(k, obj[k], obj)

    for k in list(json_schema):
        update_specs(k, json_schema[k], json_schema["properties"])

    return json_schema


def additional_parameters_fix(json_schema, parameter_name, additional_param_list):
    """Updates a JSON schema to fix additional parameters like `popular` and `required` fileds in json-schema.

    Args:
        json_schema (dict): The JSON schema to be modified.
        parameter_name (str): Name of the parameter.
        additional_param_list (list of str): List of dot-separated strings indicating paths in the JSON schema.

    Returns:
        dict: The updated JSON schema.
    """
    def rectify_schema(schema, param_list, idx):
        """Recursively updated the json-schema additional parameters to mimic api specs json format

        Args:
            schema (dict): The JSON schema to be modified.
            param_list (list): list of parameters to be modified
            idx (int): parameter index to point to.

        Returns: None
        """
        if idx == len(param_list) - 1:
            return
        rectify_schema(schema[param_list[idx]]["properties"], param_list, idx + 1)
        if parameter_name not in schema[param_list[idx]]:
            schema[param_list[idx]][parameter_name] = []
        schema[param_list[idx]][parameter_name].append(param_list[idx + 1])
        schema[param_list[idx]][parameter_name] = list(
            set(schema[param_list[idx]][parameter_name])
        )

    def extract_default_values(schema_props, default_dict, param_list, idx):
        """Recursively extract default values for parameters marked with parameter_name.

        Args:
            schema_props (dict): The schema properties to traverse.
            default_dict (dict): The default values dictionary to build.
            param_list (list): list of parameters in the path
            idx (int): parameter index to point to.

        Returns: dict or value
        """
        if idx >= len(param_list):
            return None

        param = param_list[idx]
        if param not in schema_props:
            return None

        if idx == len(param_list) - 1:
            # This is the leaf parameter - return its default value
            return schema_props[param].get("default")

        # Recursively process nested structure
        nested_props = schema_props[param].get("properties", {})

        result = extract_default_values(nested_props, {}, param_list, idx + 1)
        return result

    base_param = set()
    for additional_param in additional_param_list:
        additional_param_split = additional_param.split(".")
        rectify_schema(json_schema["properties"], additional_param_split, 0)
        base_param.add(additional_param_split[0])

    # For 'popular' parameter, build a dictionary of default values
    if parameter_name == "popular":
        popular_dict = {}
        for additional_param in additional_param_list:
            additional_param_split = additional_param.split(".")
            # Build nested dictionary structure
            current_dict = popular_dict
            schema_props = json_schema["properties"]

            for i, param in enumerate(additional_param_split):
                if i == len(additional_param_split) - 1:
                    # Leaf parameter - get its default value
                    if param in schema_props and "default" in schema_props[param]:
                        current_dict[param] = schema_props[param]["default"]
                else:
                    # Intermediate level - ensure nested structure exists
                    if param not in current_dict:
                        current_dict[param] = {}
                    current_dict = current_dict[param]
                    # Update schema_props to the nested properties
                    if param in schema_props and "properties" in schema_props[param]:
                        schema_props = schema_props[param]["properties"]

        json_schema[parameter_name] = popular_dict
    else:
        # For 'required' and other parameters, keep as list
        json_schema[parameter_name] = list(base_param)

    return json_schema


def create_json_schema(json_data):
    """Creates a JSON Schema based on the given JSON data.

    Args:
        json_data (dict): JSON data for which the schema is to be created.

    Returns:
        schema (dict): Dictionary representing the JSON schema.
    """
    schema = {"type": "object", "properties": {}, "default": {}}
    auto_ml_parameters = []
    auto_ml_disabled_parameters = []
    popular_parameter = []
    required_parameter = []

    # parse the json and build json schema
    def build_schema(param, param_obj, parent_prop, parent_default, hierarchy):
        """Parse json and build JSON-Schema

        Args:
            param (string): parameter name.
            param_obj (dict) : parameter object.
            parent_prop (dict) : parent properties.
            parent_default (dict) : parent defaults.
            hierarchy (list) : json heirarchy path from root to current node.

        Returns: None
        """
        param_name = param
        hierarchy.append(param_name)

        if type(param_obj) is not dict:
            hierarchy.pop()
            return

        if "value" not in param_obj:
            hierarchy.pop()
            return

        param_value = param_obj["value"]
        param_meta = param_obj["metadata"]

        # get metadata
        display_name = param_meta.get("display_name")
        value_type = param_meta.get("value_type")
        union_types = param_meta.get("union_types")
        literal_values = param_meta.get("literal_values")
        description = param_meta.get("description")
        default_value = param_meta.get("default_value")
        examples = param_meta.get("examples")
        valid_min = param_meta.get("valid_min")
        valid_max = param_meta.get("valid_max")
        valid_options = param_meta.get("valid_options")
        option_weights = param_meta.get("option_weights")
        required = param_meta.get("required")
        math_cond = param_meta.get("math_cond")
        parent_param = param_meta.get("parent_param")
        depends_on = param_meta.get("depends_on")
        popular = param_meta.get("popular")
        automl_enabled = param_meta.get("automl_enabled")
        regex = param_meta.get("regex")
        link = param_meta.get("link")

        # convert value type
        mapped_value_type = __type_mapping.get(value_type)
        if mapped_value_type is None:
            hierarchy.pop()
            return

        # fix data types
        value = __basic_type_fix(mapped_value_type, param_value, union_types, literal_values)
        default_value = __basic_type_fix(mapped_value_type, default_value, union_types, literal_values)
        valid_min = __basic_type_fix(mapped_value_type, valid_min, union_types, literal_values)
        valid_max = __basic_type_fix(mapped_value_type, valid_max, union_types, literal_values)
        valid_options = __array_type_fix(value_type, valid_options)
        examples = __array_type_fix(value_type, examples)

        # create new schema
        props = parent_prop
        if mapped_value_type == "const":
            props[param_name] = {"const": value}
            parent_default[param_name] = default_value
            hierarchy.pop()
            return

        # Handle Union types
        if mapped_value_type == "union" and union_types:
            # Create a schema that accepts multiple types
            union_schema = []
            for union_type in union_types:
                mapped_union_type = __type_mapping.get(union_type, union_type)
                if mapped_union_type in ("string", "boolean", "integer", "number", "array"):
                    type_schema = {"type": mapped_union_type}
                    # If this is a string type and we have literal values, add enum constraint
                    if mapped_union_type == "string" and literal_values:
                        type_schema["enum"] = literal_values
                    # If this is an array type and we have literal values, add items constraint
                    elif mapped_union_type == "array" and literal_values:
                        type_schema["items"] = {"type": "string", "enum": literal_values}
                    union_schema.append(type_schema)

            if union_schema:
                props[param_name] = {"anyOf": union_schema, "properties": {}, "default": {}}
            else:
                props[param_name] = {"type": param_meta.get("value_type"), "properties": {}, "default": {}}
        else:
            props[param_name] = {"type": param_meta.get("value_type"), "properties": {}, "default": {}}

        # props[param_name]["default"] = default_value
        # if parent_default:
        #     parent_default[param_name] = default_value

        if display_name:
            props[param_name]["title"] = display_name
        if description:
            props[param_name]["description"] = description
        if examples:
            props[param_name]["examples"] = examples
        if default_value == "" or default_value is not None:
            # Apply type conversion to default_value for union types
            processed_default = __basic_type_fix(value_type, default_value, union_types, literal_values)
            props[param_name]["default"] = processed_default
            parent_default[param_name] = processed_default
        # if default_value not in (None, ""):
        #     props[param_name]["default"] = default_value
        #     parent_default[param_name] = default_value
        if valid_min is not None and valid_min != "":
            props[param_name]["minimum"] = valid_min
        if valid_max is not None and valid_max != "":
            props[param_name]["maximum"] = valid_max
        if math_cond:
            props[param_name]["math_cond"] = math_cond
        if parent_param:
            props[param_name]["parent_param"] = parent_param
        if depends_on:
            props[param_name]["depends_on"] = depends_on
        if valid_options:
            props[param_name]["enum"] = valid_options
        if option_weights is not None:
            props[param_name]["option_weights"] = option_weights
        if regex and mapped_value_type == "string":
            props[param_name]["pattern"] = regex
        if link and link.startswith("http"):
            props[param_name]["link"] = link
        if required and required.lower() == "yes":
            required_parameter.append(".".join(hierarchy))
        if popular and popular.lower() == "yes":
            props[param_name]["popular"] = True
            popular_parameter.append(".".join(hierarchy))
        if automl_enabled and automl_enabled.lower() == "true":
            props[param_name]["automl_enabled"] = True
            if parent_default.get("automl_default_parameters") is None:
                parent_default["automl_default_parameters"] = []
            parent_default["automl_default_parameters"].append(".".join(hierarchy))
            auto_ml_parameters.append(".".join(hierarchy))
        if automl_enabled and automl_enabled.lower() == "false":
            props[param_name]["automl_enabled"] = False
            if parent_default.get("automl_disabled_parameters") is None:
                parent_default["automl_disabled_parameters"] = []
            parent_default["automl_disabled_parameters"].append(".".join(hierarchy))
            auto_ml_disabled_parameters.append(".".join(hierarchy))

        # add object hierarchy
        if mapped_value_type == "object":
            props[param_name]["properties"] = {}
            if param_value:
                for p, pObj in param_value.items():
                    build_schema(
                        p,
                        pObj,
                        props[param_name]["properties"],
                        props[param_name]["default"],
                        hierarchy,
                    )
                    # update parent default
                    tempDict = {param_name: props[param_name]["default"]}
                    __merge(parent_default, tempDict)
        hierarchy.pop()

    for parameter, parameterObj in json_data.items():
        build_schema(
            parameter, parameterObj, schema["properties"], schema["default"], []
        )

    # auto-ml parameter addition in json-schema
    schema = auto_ml_parameters_fix(schema)
    schema["automl_default_parameters"] = list(set(auto_ml_parameters))
    schema["automl_disabled_parameters"] = list(set(auto_ml_disabled_parameters))

    # `popular` field correction in json-schema
    if popular_parameter:
        unique_popular_parameters = list(set(popular_parameter))
        schema = additional_parameters_fix(schema, "popular", unique_popular_parameters)

    # `required` field correction in json-schema
    if required_parameter:
        unique_required_parameters = list(set(required_parameter))
        schema = additional_parameters_fix(
            schema, "required", unique_required_parameters
        )

    return schema


def write_json_to_file(json_data, file_path):
    """Writes a given JSON object to a specified file in JSON format.

    Args:
        json_data (dict): Dictionary representing the JSON object.
        file_path (str): The complete file path for the JSON data to be saved.

    Returns:
        message (str): Success or error message.
    """
    try:
        with open(file_path, "w", encoding='utf-8') as file:
            json.dump(json_data, file, indent=4)
        return f"JSON data has been successfully written to {file_path}."
    except Exception as e:
        return f"An error occurred: {e}"


def download_file_from_github(url, dir_path, file_name):
    """Downloads a file from GitHub and saves it to a specified local directory.

    Args:
        url (str): URL of the file on GitHub.
        dir_path (str): Directory path on the local system for the file.
        file_name (str): Name for the saved file.

    Returns:
        file_path (str or None): Path to the downloaded file if successful, or None. Status of operation is logged.
    """
    response = requests.get(url)   # noqa pylint: disable=W3101
    if response.status_code == 200:
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, "wb") as file:
            file.write(response.content)
        logger.info("File downloaded successfully and saved to %s", dir_path)
        return file_path

    logger.error("Failed to download the file. HTTP status code: %s", response.status_code)
    return None


def import_module_from_path(module_name):
    """Dynamically imports a Python module from a specified path.

    Args:
        module_name (str): Name of the module to be imported.
        path_to_module (str): Filesystem path of the module.

    Returns:
        imported_module (module or None): The imported module if successful, or None. Status of operation is logged.
    """
    try:
        imported_module = importlib.import_module(module_name)
        logger.info("Module '%s' imported successfully.", module_name)
        return imported_module
    except ImportError as e:
        logger.error("Error importing module: %s", e)
        return None


def remove_none_empty_fields(json_schema):
    """Recursively remove all None and empty string values and their corresponding keys from a dictionary.

    Parameters:
    json_schema (dict): The input dictionary from which None and empty string values should be removed.

    Returns:
    dict: A new dictionary with all None and empty string values removed.
    """
    if not isinstance(json_schema, dict):
        return json_schema

    new_dict = {}
    for key, value in json_schema.items():
        if isinstance(value, dict):
            nested_dict = remove_none_empty_fields(value)
            if nested_dict:  # only add if nested_dict is not empty
                new_dict[key] = nested_dict
        elif isinstance(value, list):
            new_list = [
                remove_none_empty_fields(item)
                for item in value
                if item is not None and item != ""
            ]
            if new_list:  # only add if new_list is not empty
                new_dict[key] = new_list
        elif value is not None and value != "":
            new_dict[key] = value

    return new_dict
