# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional test conftest."""
import os
import pytest


@pytest.fixture
def ngc_key():
    """Configure NGC key. Skip test if not set."""
    key = os.environ.get("NGC_KEY")
    if not key:
        pytest.skip("NGC_KEY environment variable not set")
    return key


@pytest.fixture
def ngc_path():
    """Configure NGC path. Skip test if not set."""
    path = os.environ.get("NGC_PATH")
    if not path:
        pytest.skip("NGC_PATH environment variable not set")
    return path
