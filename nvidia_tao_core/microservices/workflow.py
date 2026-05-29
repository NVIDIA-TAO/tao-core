#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start API workflow"""
import threading

from nvidia_tao_core.microservices.utils.job_utils.workflow import Workflow

Workflow.start()

for thread in threading.enumerate():
    if thread.name == "WorkflowThreadTAO":
        thread.join()
