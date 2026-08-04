# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Api tests to generate schema for networks"""

import json
import os
import pathlib
import pytest

from omegaconf import OmegaConf

from nvidia_tao_core.microservices.constants import TAO_NETWORKS
from nvidia_tao_core.microservices.enum_constants import _get_network_architectures
from nvidia_tao_core.microservices.utils.core_utils import get_microservices_network_and_action
from nvidia_tao_core.config.clip.default_config import CLIPValDataConfig
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


def test_clip_train_metadata_masking_schema():
    """CLIP train schema exposes metadata-masked SigLIP configuration."""
    schema = generate_schema("clip", "train")

    train_dataset = schema["properties"]["dataset"]["properties"]["train"]
    train_dataset_default = schema["default"]["dataset"]["train"]
    train = schema["properties"]["train"]
    train_default = schema["default"]["train"]

    include_metadata = train_dataset["properties"]["include_attribute_metadata"]
    assert include_metadata["type"] == "bool"
    assert include_metadata["default"] is False
    assert train_dataset_default["include_attribute_metadata"] is False

    dist_impl = train["properties"]["siglip_loss_dist_impl"]
    assert dist_impl["default"] == "gather"
    assert dist_impl["enum"] == ["bidir", "shift", "reduce", "gather", "local"]
    assert train_default["siglip_loss_dist_impl"] == "gather"
    assert "Metadata masking supports local and gather." in dist_impl["description"]

    mask_mode = train["properties"]["siglip_loss_mask_mode"]
    assert mask_mode["default"] == "none"
    assert mask_mode["enum"] == [
        "none",
        "attribute_match_ignore",
        "attribute_plus_accessory_match_ignore",
    ]
    assert train_default["siglip_loss_mask_mode"] == "none"
    assert "Metadata masking supports local and gather." in mask_mode["description"]
    assert "train.siglip_loss_dist_impl to be 'local' or 'gather'" in mask_mode["description"]


def test_clip_metadata_evaluation_schema():
    """CLIP schemas expose metadata-aware validation and evaluation options."""
    schema = generate_schema("clip", "evaluate")
    inference_schema = generate_schema("clip", "inference")

    val = schema["properties"]["dataset"]["properties"]["val"]
    val_default = schema["default"]["dataset"]["val"]
    evaluate = schema["properties"]["evaluate"]
    evaluate_default = schema["default"]["evaluate"]
    inference = inference_schema["properties"]["inference"]
    inference_default = inference_schema["default"]["inference"]

    metadata_match_eval = val["properties"]["metadata_match_eval"]
    assert metadata_match_eval["type"] == "bool"
    assert metadata_match_eval["default"] is False
    assert val_default["metadata_match_eval"] is False

    metadata_match_mode = val["properties"]["metadata_match_mode"]
    assert metadata_match_mode["default"] == "scalar_attributes"
    assert metadata_match_mode["enum"] == [
        "scalar_attributes",
        "scalar_plus_accessories",
    ]
    assert val_default["metadata_match_mode"] == "scalar_attributes"

    pas_ground_truth_mode = evaluate["properties"]["pas_ground_truth_mode"]
    assert pas_ground_truth_mode["default"] == "paired_caption"
    assert pas_ground_truth_mode["enum"] == [
        "paired_caption",
        "scalar_attributes",
        "scalar_plus_accessories",
    ]
    assert evaluate_default["pas_ground_truth_mode"] == "paired_caption"
    assert "pas_ground_truth_mode" not in inference["properties"]
    assert "pas_ground_truth_mode" not in inference_default


def test_clip_attribute_pairs_file_structured_config():
    """CLIP dataset items preserve attribute_pairs_file through OmegaConf."""
    config = OmegaConf.structured(CLIPValDataConfig())
    config = OmegaConf.merge(
        config,
        {
            "datasets": [
                {
                    "image_dir": "/data/images",
                    "attribute_pairs_file": "/data/test_pairs.json",
                }
            ]
        },
    )

    serialized = OmegaConf.to_container(config, resolve=True)

    assert serialized["datasets"][0]["attribute_pairs_file"] == "/data/test_pairs.json"
