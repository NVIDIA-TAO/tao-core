# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Json to kitti file conversion"""
from nvidia_tao_core.microservices.utils.core_utils import safe_load_file


def kitti(data, level=0):
    """Writes the dictionary data into kitti file"""
    if type(data) is dict and level == 0:
        data.pop("version", None)
    specs = []
    level_space = ''
    for _ in range(level):
        level_space += '  '
    for key in data:
        if data[key] is None:
            continue
        if type(data[key]) is dict:
            specs.append(level_space + key + ' {')
            specs.append(kitti(data[key], level + 1))
            specs.append(level_space + '}')
        elif type(data[key]) is list:
            for d in data[key]:
                t = type(d)
                s = str(d)
                isEnum = bool(s.startswith('__') and s.endswith('__'))
                if type(d) is dict:
                    specs.append(level_space + key + ' {')
                    specs.append(kitti(d, level + 1))
                    specs.append(level_space + '}')
                # WARNING: LIST OF LIST NOT SUPPORTED
                else:
                    if isEnum:
                        specs.append(level_space + key + ': ' + s[2:-2])
                    elif t in [bool, int, float]:
                        specs.append(level_space + key + ': ' + s)
                    else:
                        specs.append(level_space + key + ': "' + s + '"')
        else:
            t = type(data[key])
            s = str(data[key])
            isEnum = bool(s.startswith('__') and s.endswith('__'))
            if isEnum:
                specs.append(level_space + key + ': ' + s[2:-2])
            elif t in [bool, int, float]:
                specs.append(level_space + key + ': ' + s)
            else:
                specs.append(level_space + key + ': "' + s + '"')
    return '\n'.join(specs)


def convert(path):  # NOTE: Not calling this function. Just using kitti() in current workflow.
    """Reads from json and dumps into kitti file"""
    data = safe_load_file(path)
    # remove version from schema for now since containers do not yet support it
    return kitti(data)
