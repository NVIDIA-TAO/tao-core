# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Json to yaml file conversion"""
from nvidia_tao_core.microservices.utils.core_utils import safe_load_file


def yml(data):
    """Writes the dictionary data into yaml file"""
    if type(data) is dict:
        data.pop("version", None)
    return data


def convert(path):
    """Reads from json and dumps into yaml file"""
    data = safe_load_file(path)
    return yml(data)
