# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flask decorators for common functionality."""

from functools import wraps
from flask import make_response, jsonify

from nvidia_tao_core.microservices.utils.core_utils import is_pvc_space_free


def disk_space_check(f):
    """Decorator to check disk space for API endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        threshold_bytes = 100 * 1024 * 1024

        pvc_free_space, pvc_free_bytes = is_pvc_space_free(threshold_bytes)
        msg = (f"PVC free space remaining is {pvc_free_bytes} bytes "
               f"which is less than {threshold_bytes} bytes")
        if not pvc_free_space:
            return make_response(
                jsonify({
                    'error': f'Disk space is nearly full. {msg}. Delete appropriate experiments/datasets'
                }),
                500
            )

        return f(*args, **kwargs)

    return decorated_function
