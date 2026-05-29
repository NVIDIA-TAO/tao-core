#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

umask 0
python3 $(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/nvidia_tao_core/microservices/workflow.py