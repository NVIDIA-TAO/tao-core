# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API handlers module - provides direct access to specialized handler classes"""

# Import all handlers for direct access
from .dataset_handler import DatasetHandler
from .workspace_handler import WorkspaceHandler
from .experiment_handler import ExperimentHandler
from .job_handler import JobHandler
from .spec_handler import SpecHandler
from .mongo_handler import MongoBackupHandler
from .model_handler import ModelHandler

# Import inference microservice servers
from .base_inference_microservice_server import BaseInferenceMicroserviceServer
from .huggingface_inference_microservice_server import HuggingFaceInferenceMicroserviceServer

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
