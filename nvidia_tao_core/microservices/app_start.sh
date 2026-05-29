#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

umask 0
rm -rf /shared/orgs/00000000-0000-0000-0000-000000000000/*
# cp -r shared/* /shared/ ; chmod 777 /shared/orgs ; chmod -R 777 /shared/orgs/00000000-0000-0000-0000-000000000000 2>/dev/null ; true
# cp -r notebooks /shared/

if [ -n "$DOCKER_HOST" ]; then
    # Dynamically get Docker socket GID and add www-data to it
    DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)

    # If a group with this GID doesn't exist, create one (e.g., 'docker')
    if ! getent group "$DOCKER_GID" >/dev/null; then
        groupadd -g "$DOCKER_GID" docker
    fi

    # Add www-data to that group
    usermod -aG "$DOCKER_GID" www-data
fi
service nginx start
export _PYTHON_LIB_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
# Set HOME to /tmp to avoid permission issues with leptonai cache access
export HOME=/tmp
uwsgi --ini $_PYTHON_LIB_PATH/nvidia_tao_core/microservices/uwsgi.ini