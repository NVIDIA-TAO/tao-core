# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API handlers with backwards-compatible lazy class imports.

Importing one lightweight handler must not initialize every cloud, database,
and orchestration handler.  This is important in model action containers,
which use the inference server but not the TAO API service.
"""

from importlib import import_module

# Export all handlers
__all__ = [
    'DatasetHandler',
    'WorkspaceHandler',
    'ExperimentHandler',
    'JobHandler',
    'SpecHandler',
    'MongoBackupHandler',
    'ModelHandler',
    # Inference microservice servers
    'BaseInferenceMicroserviceServer',
    'HuggingFaceInferenceMicroserviceServer',
]

_HANDLERS = {
    "DatasetHandler": ("dataset_handler", "DatasetHandler"),
    "WorkspaceHandler": ("workspace_handler", "WorkspaceHandler"),
    "ExperimentHandler": ("experiment_handler", "ExperimentHandler"),
    "JobHandler": ("job_handler", "JobHandler"),
    "SpecHandler": ("spec_handler", "SpecHandler"),
    "MongoBackupHandler": ("mongo_handler", "MongoBackupHandler"),
    "ModelHandler": ("model_handler", "ModelHandler"),
    "BaseInferenceMicroserviceServer": (
        "base_inference_microservice_server",
        "BaseInferenceMicroserviceServer",
    ),
    "HuggingFaceInferenceMicroserviceServer": (
        "huggingface_inference_microservice_server",
        "HuggingFaceInferenceMicroserviceServer",
    ),
}


def __getattr__(name):
    if name not in _HANDLERS:
        raise AttributeError(name)
    module_name, attribute = _HANDLERS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
