# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from khai.core.cv import container
from khai.core.cv.schema import packs


def test_contract_versions():
    assert container.CONTRACT_VERSION >= 1
    assert packs.CONTRACT_VERSION >= 1
