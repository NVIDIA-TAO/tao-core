# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nvidia_tao_core.microservices.constants import TAO_NETWORKS
from nvidia_tao_core.microservices.utils.core_utils import read_network_config
from nvidia_tao_core.scripts.generate_schema import generate_schema
from nvidia_tao_core.microservices.utils.core_utils import get_microservices_network_and_action


@pytest.mark.parametrize("network", TAO_NETWORKS)
def test_data_sources_actions_exist(network):
    """Test that data_sources.{action} fields exist in schema for each network."""
    try:
        config = read_network_config(network)
    except Exception:
        return

    if not config or "data_sources" not in config:
        return

    # test that each data_sources parameter exist in schema
    for action in config["data_sources"]:
        for override in config["data_sources"][action]:
            network, action = get_microservices_network_and_action(network, action)
            schema = generate_schema(network, action).get('default', {})
            for key in override.split("."):
                if key not in schema:
                    print(key, schema, key in schema)
                assert key in schema, f"Override path {override} not found in schema for {network} {action}"
                schema = schema[key]
