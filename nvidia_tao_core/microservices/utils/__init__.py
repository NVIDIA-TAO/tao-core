# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Microservices utilities with lazy optional-module loading."""

from importlib import import_module

# Import key utilities for easy access
from .core_utils import *  # noqa: F403

__all__ = [  # noqa: F405
    # Core utilities (files)
    'core_utils',
    'handler_utils',
    'stateless_handler_utils',
    'dataset_utils',
    'cloud_utils',
    'encrypt_utils',
    'mongo_utils',
    'ngc_utils',
    'basic_utils',
    'automl_utils',
    'automl_job_utils',
    'network_utils',
    'executor_utils',

    # Utility packages (directories)
    'airgapped_utils',
    'auth_utils',
    'filter_utils',
    'health_utils',
    'job_utils',
    'specs_utils'
]

_LAZY_MODULES = set(__all__)
_LEGACY_SYMBOL_MODULES = ("handler_utils", "stateless_handler_utils")


def __getattr__(name):
    if name in _LAZY_MODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    # Preserve legacy direct symbol access without importing these large
    # modules until the symbol is actually requested.
    for module_name in _LEGACY_SYMBOL_MODULES:
        module = import_module(f"{__name__}.{module_name}")
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(name)
