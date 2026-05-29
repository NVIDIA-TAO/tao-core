# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility function to convert spec csv to json schema"""
import csv
import json
import logging
import os

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
    'collection': 'object',
    'list': 'array',
    'list_1_backbone': 'array',
    'list_1_normal': 'array',
    'list_2': 'array',
    'list_3': 'array',
    'float': 'number',
    'bool': 'boolean',
    'integer': 'integer',
    'string': 'string',
    'str': 'string',
    'int': 'integer',
    'dict': 'object',
    'const': 'const',
    'ordered': 'ordered',
    'categorical': 'categorical',
    'ordered_int': 'ordered_int',
    'enum': 'string'
}


def __basic_type_fix(value_type, value):
    """Converts spec values based on their datatype"""
    if value in (None, ''):
        return None
    if value in ('inf', '-inf'):
        return float(value)
    if value_type in ('integer', 'ordered_int'):
        return int(value)
    if value_type == 'number':
        return float(value)
    if value_type == 'boolean':
        return str(value).lower() == "true"
    if value_type == 'array':
        return json.loads(value)
    if value_type == 'object':
        return json.loads(value)
    return value


def __array_type_fix(value_type, value):
    """Converts spec values in an array based on their datatype"""
    if value in (None, ''):
        return None
    # We dont need this for list / dict (examples, valid_options only for simple types)
    # if value_type in ["array","object"]:
    #     return None
    values = value.replace(' ', '').split(',')
    if value_type == 'integer':
        return [int(i) for i in values]
    if value_type == 'number':
        return [float(i) for i in values]
    if value_type == 'boolean':
        return [str(i).lower() == "true" for i in values]
    if value_type == 'array':
        return [json.loads(i) for i in values]
    if value_type == 'object':
        return [json.loads(i) for i in values]
    return values


def __merge(d1, d2):
    """Merges two dictionaries"""
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


def harden_parameter_name(parameter):
    """Fix parameter names by removing flanking "." and remove all spaces"""
    if not parameter:
        return None
    return parameter.rstrip(".").lstrip(".").replace(" ", "")


def harden_value_type(value_type):
    """If value type is an unknown, make it string"""
    if value_type not in __type_mapping.keys():
        return "string"
    return value_type


def harden_numerical_value(value):
    """If the value cannot become a float, then return None"""
    if value:
        try:
            float(value)
            return value
        except Exception as e:
            logger.error("Exception thrown in harden_numerical_value: %s", str(e))
            return None
    else:
        return None


def convert(path):
    """Convert csv spec to json schema"""
    array_parameters = []
    schema = {}

    with open(path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # get row data
            parameter = row.get('parameter')
            # if parameter:
            #     parameter = harden_parameter_name(parameter)
            display_name = row.get('display_name')
            value_type = row.get('value_type')
            if value_type and value_type.lower() == 'hidden':
                continue
            # if value_type:
            #     value_type = harden_value_type(value_type)
            description = row.get('description')
            default_value = row.get('default_value')
            examples = row.get('examples')
            valid_min = row.get('valid_min')
            # if valid_min:
            #     valid_min = harden_numerical_value(valid_min)
            valid_max = row.get('valid_max')
            # if valid_max:
            #     valid_max = harden_numerical_value(valid_max)
            valid_options = row.get('valid_options')
            required = row.get('required')
            popular = row.get('popular')
            automl_enabled = row.get('automl_enabled')
            regex = row.get('regex')
            link = row.get('link')
            # convert value type
            value_type = __type_mapping.get(value_type)
            if value_type is None:
                continue
            if value_type == 'array':
                array_parameters.append(parameter)
            # fix data types
            default_value = __basic_type_fix(value_type, default_value)
            valid_min = __basic_type_fix(value_type, valid_min)
            valid_max = __basic_type_fix(value_type, valid_max)
            valid_options = __array_type_fix(value_type, valid_options)
            examples = __array_type_fix(value_type, examples)
            # compose object
            params = parameter.split('.')
            last_param = params.pop()
            if value_type == 'const':
                obj = {'type': 'object',
                       'properties': {last_param: {'const': default_value}}, 'default': {last_param: default_value}}
            else:
                obj = {'type': 'object',
                       'properties': {last_param: {'type': value_type}}}
                # add known object details
                props = obj['properties'][last_param]
                if display_name not in (None, ''):
                    props['title'] = display_name
                if description not in (None, ''):
                    props['description'] = description
                if examples not in (None, []):
                    props['examples'] = examples
                if default_value not in (None, ''):
                    props['default'] = default_value
                    obj['default'] = {last_param: default_value}
                if valid_min is not None:
                    props['minimum'] = valid_min
                if valid_max is not None:
                    props['maximum'] = valid_max
                if valid_options not in (None, []):
                    props['enum'] = valid_options
                if regex not in (None, '') and value_type == 'string':
                    props['pattern'] = regex
                if link is not None and link.startswith('http'):
                    props['link'] = link
                if required is not None and required.lower() == 'yes':
                    if obj.get('required') is None:
                        obj['required'] = []
                    obj['required'].append(last_param)
                if popular not in (None, '') and popular.lower() == 'yes':
                    obj['popular'] = {last_param: default_value}
                if automl_enabled is not None and automl_enabled.lower() == 'true':
                    if obj.get('automl_default_parameters') is None:
                        obj['automl_default_parameters'] = []
                    obj['automl_default_parameters'].append(parameter)
                # special override of default with array of class strings
                isArray = parameter in array_parameters
            # add object hierarchy
            while len(params) > 0:
                joined_params = '.'.join(params)
                isArray = joined_params in array_parameters
                isRequired = obj.get('required') is not None
                isPopular = obj.get('popular') is not None
                hasDefault = obj.get('default') is not None
                isAutomlenabled = obj.get('automl_default_parameters') is not None
                param = params.pop()
                if isArray:
                    default = []
                    popular = []
                    if hasDefault:
                        default = [obj['default']]
                    if isPopular:
                        popular = [obj['popular']]
                    obj = {
                        'type': 'object',
                        'properties': {
                            param: {
                                'type': 'array',
                                'items': obj,
                                'default': default
                            }
                        }
                    }
                    if hasDefault or default:
                        obj['default'] = {param: default}
                    if isPopular or popular:
                        obj['popular'] = {param: default}
                else:
                    default = obj.get('default')
                    popular = obj.get('popular')
                    obj = {
                        'type': 'object',
                        'properties': {param: obj},
                    }
                    if hasDefault:
                        obj['default'] = {param: default}
                    if isPopular:
                        obj['popular'] = {param: popular}
                if isRequired:
                    obj['required'] = [param]
                if isAutomlenabled:
                    if not schema.get('automl_default_parameters'):
                        schema['automl_default_parameters'] = [parameter]
                    else:
                        schema['automl_default_parameters'].append(parameter)
            # update schema with obj
            __merge(schema, obj)

    # return json schema
    return schema
