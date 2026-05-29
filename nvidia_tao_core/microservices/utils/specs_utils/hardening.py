# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Json spec schema hardening modules"""
import copy
import logging
import os

from ..specs_utils import csv_to_json_schema

from jsonschema import validate as validationDriver, exceptions

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


def __merge(d1, d2):
    """Merge 2 dictionaries"""
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


def harden(data, schema):
    """Harden the schema provided"""
    try:
        return __merge(copy.deepcopy(schema['default']), data)
    except Exception as e:
        err = f"Exception thrown in harden is {str(e)}"
        logger.error(err)
        return False


def validate(data, schema):
    """Validate the schema provided"""
    try:
        validationDriver(instance=data, schema=schema)
    except exceptions.ValidationError as e:
        return e.message
    return None


# test code


if __name__ == '__main__':
    schema = csv_to_json_schema.convert("specs/detectnet_v2/detectnet_v2 - train.csv")

    # positive test
    hardened_data = harden(data={'random_seed': 99}, schema=schema)
    err = validate(data=hardened_data, schema=schema)
    if err:
        logger.error(err)

    # negative test
    err = validate(data={'random_seed': 99}, schema=schema)
    if err:
        logger.error(err)
