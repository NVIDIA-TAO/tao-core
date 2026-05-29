# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API response pagination modules"""
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


def apply(args, data):
    """Apply pagination to reduce the number of results that are returned"""
    pagination_skip = args.get('skip')
    pagination_size = args.get('size')

    if pagination_skip is not None:
        try:
            data = data[int(pagination_skip):]
        except Exception as e:
            logger.error("Exception thrown in pagination_skip: %s", str(e))
            pass
    if pagination_size is not None:
        try:
            data = data[:int(pagination_size)]
        except Exception as e:
            logger.error("Exception thrown in pagination_size: %s", str(e))
            pass

    return data
