#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

umask 0
PYTHON_LIB_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')

# Check if NGC API key is provided as 6th argument
if [ $# -ge 6 ] && [ -n "$6" ]; then
    python3 $PYTHON_LIB_PATH/nvidia_tao_core/microservices/pretrained_models.py --shared-folder-path ptms --org-teams "$1" --ngc-key "$6" --use-both
else
    python3 $PYTHON_LIB_PATH/nvidia_tao_core/microservices/pretrained_models.py --shared-folder-path ptms --org-teams "$1" --use-both
fi


## MongoDB backup (only if AWS credentials are provided)
if [ -n "$2" ] && [ -n "$3" ] && [ -n "$4" ] && [ -n "$5" ]; then
    ## Clear users session cache of expired tokens
    python3 $PYTHON_LIB_PATH/nvidia_tao_core/microservices/mongo_users_cleanup.py
    echo "AWS credentials found, performing MongoDB backup..."
    python3 $PYTHON_LIB_PATH/nvidia_tao_core/microservices/app_handlers/mongo_handler.py --access-key "$2" --secret-key "$3" --s3-bucket-name "$4" --region "$5"
else
    echo "AWS credentials not provided, skipping MongoDB backup"
fi