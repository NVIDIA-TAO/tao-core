#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generate _pb2.py files

apt-get install -y protobuf-compiler

protoc nvidia_tao_core/proto/*.proto --python_out=.