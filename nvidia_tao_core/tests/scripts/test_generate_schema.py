# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Api tests to generate schema for networks"""

import json
import os
import pathlib
import pytest

from nvidia_tao_core.microservices.constants import TAO_NETWORKS
from nvidia_tao_core.microservices.enum_constants import _get_network_architectures
from nvidia_tao_core.microservices.utils.core_utils import get_microservices_network_and_action
from nvidia_tao_core.scripts.generate_schema import generate_schema

EXCLUDED_KEYWORDS = [
    'vlm', 'segmentation',
    'image_classification', 'character_recognition', 'object_detection'
]

# Get networks that have config modules (directories in nvidia_tao_core/config/)
CONFIG_MODULE_DIR = pathlib.Path(__file__).parent.parent.parent / "config"
networks_with_config_modules = set()
if CONFIG_MODULE_DIR.exists():
    for item in CONFIG_MODULE_DIR.iterdir():
        if item.is_dir() and not item.name.startswith(('_', '.')):
            networks_with_config_modules.add(item.name)

config_networks = [
    network for network in _get_network_architectures()
    if not any(keyword in network for keyword in EXCLUDED_KEYWORDS) and
    network in networks_with_config_modules
]
constant_networks = [
    network for network in TAO_NETWORKS
    if not any(keyword in network for keyword in EXCLUDED_KEYWORDS) and
    network in networks_with_config_modules
]


def get_network_actions(network_name):
    """Get supported actions for a specific network from its config file"""
    config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "microservices", "handlers", "network_configs")
    config_file = os.path.join(config_dir, f"{network_name}.config.json")

    if not os.path.exists(config_file):
        # Fallback to default actions if config file doesn't exist
        return ["train", "evaluate", "export", "inference"]

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config.get("api_params", {}).get("actions", ["train", "evaluate", "export", "inference"])
    except (json.JSONDecodeError, KeyError):
        # Fallback to default actions if config is malformed
        return ["train", "evaluate", "export", "inference"]


def get_network_action_pairs():
    """Generate (network, action) pairs for all networks and their supported actions"""
    pairs = []

    # Add pairs from config_networks
    for network in config_networks:
        actions = get_network_actions(network)
        for action in actions:
            pairs.append((network, action))

    # Add pairs from constant_networks
    for network in constant_networks:
        actions = get_network_actions(network)
        for action in actions:
            pairs.append((network, action))

    return pairs


# Generate all network-action pairs
network_action_pairs = get_network_action_pairs()


@pytest.mark.parametrize("network,action", network_action_pairs)
def test_networks_with_valid_actions(network, action):
    """Test schema generation for networks with their supported actions"""
    network_arch, mapped_action = get_microservices_network_and_action(network, action)
    schema = generate_schema(network_arch, mapped_action)
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "default" in schema
