# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Json to toml file conversion"""
import toml
from nvidia_tao_core.microservices.utils.core_utils import safe_load_file


def toml_format(data):
    """Converts the dictionary data into toml format string"""
    if type(data) is dict:
        data = data.copy()  # Don't modify original
        data.pop("version", None)
    return toml.dumps(data)


def convert(path):
    """Reads from json and dumps into toml format"""
    data = safe_load_file(path)
    return toml_format(data)
