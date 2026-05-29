# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executor module with clean class-based architecture.

This module provides specialized executor classes for different Kubernetes operations.
Each class handles a specific type of operation for better code organization and maintainability.

Usage:
    from nvidia_tao_core.microservices.utils.job_utils.executor import JobExecutor

    job_executor = JobExecutor()
    job_executor.create_job(...)
"""

# Import all executor classes
from .microservice_executor import MicroserviceExecutor
from .deployment_executor import DeploymentExecutor

# Import essential utilities
from nvidia_tao_core.microservices.utils.executor_utils import (
    release_name,
    logger
)

# Expose only the classes and essential utilities
__all__ = [
    # Executor Classes - The main interface for this module
    'MicroserviceExecutor',
    'DeploymentExecutor',

    # Essential utilities
    'release_name',
    'logger'
]
