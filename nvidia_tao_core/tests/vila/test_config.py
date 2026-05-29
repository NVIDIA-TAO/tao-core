# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion."""

import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json
from nvidia_tao_core.api_utils.json_schema_validation import validate_jsonschema
from nvidia_tao_core.config.vila.default_config import TrainConfig, SystemConfig, ExperimentConfig


sample_trainer_config = """
num_epochs: 1
batch_size: 32
learning_rate: 0.001
weight_decay: 0.0001
warmup_ratio: 0.03
gradient_accumulation_steps: 2
lora_r: 16
max_tiles: 12
video_max_tiles: 6
system:
    num_gpus: 8
    num_nodes: 2
    master_addr: 127.0.0.1
    node_rank: 0
    port: 24501
"""

simple_experiment_config = """
model_path: /models/vila
results_dir: /path/to/result/lora
train:
    num_epochs: 10
    batch_size: 8
    learning_rate: 0.001
    weight_decay: 0.0001
    warmup_ratio: 0.03
    gradient_accumulation_steps: 2
    dataset:
        dataset_name: scienceqa
    system:
        num_gpus: 1
evaluate:
    task: youcook2_val
inference:
    conv_mode: auto
    text: "What is this video about?"
"""

sample_experiment_config = """
model_path: /models/vila
results_dir: /path/to/result/lora
train:
    num_epochs: 1
    batch_size: 32
    learning_rate: 0.001
    weight_decay: 0.0001
    warmup_ratio: 0.03
    gradient_accumulation_steps: 2
    dataset:
        dataset_name: scienceqa
    llm_mode: lora
    vision_mode: ft
    system:
        num_gpus: 8
evaluate:
    task: youcook2_val
inference:
    conv_mode: auto
    text: "What is this video about?"
"""


def generate_json_schema(dataclass_instance):
    """Simple function to generate json schema from an instance of the dataclass."""
    json_with_meta_config = dataclass_to_json(dataclass_instance)
    return create_json_schema(json_with_meta_config)


@pytest.fixture
def _test_trainer_spec():
    trainer_config = TrainConfig()
    yield trainer_config


@pytest.fixture
def _test_system_spec():
    system_config = SystemConfig()
    yield system_config


@pytest.fixture
def _test_experiment_spec():
    experiment_config = ExperimentConfig()
    yield experiment_config


@pytest.mark.vila
@pytest.mark.vlm_unit
@pytest.mark.config
def test_trainer_jsonschema_config(_test_trainer_spec):
    """Test jsonschema conversion for train spec."""
    json_with_meta_config = dataclass_to_json(_test_trainer_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Failed to dump train schema to JSON"


@pytest.mark.vila
@pytest.mark.vlm_unit
@pytest.mark.config
def test_system_jsonschema_config(_test_system_spec):
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_system_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Failed to dump evaluate schema to JSON"


@pytest.mark.vila
@pytest.mark.vlm_unit
@pytest.mark.config
def test_experiment_jsonschema_conversion(_test_experiment_spec):
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_experiment_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Failed to dump inference schema to JSON"


TEST_CONFIG_BLOCKS = [
    (sample_trainer_config, TrainConfig),
    (sample_experiment_config, ExperimentConfig),
    (simple_experiment_config, ExperimentConfig)]


@pytest.mark.vila
@pytest.mark.vlm_unit
@pytest.mark.config
@pytest.mark.schema_validation
@pytest.mark.parametrize(
    "yaml_string, dataclass_class_name",
    TEST_CONFIG_BLOCKS
)
def test_load_experiment_spec(
    yaml_string,
    dataclass_class_name,
):
    """Simple function to load and validate the structure config from a yaml file."""
    schema = OmegaConf.structured(dataclass_class_name)
    config = OmegaConf.create(yaml_string)
    assert OmegaConf.merge(schema, config), "Failed to merge schema with config"
    json_schema = generate_json_schema(dataclass_class_name())
    validation_status = validate_jsonschema(config, json_schema["properties"])
    assert not (validation_status), "Validation should have failed for invalid config"
