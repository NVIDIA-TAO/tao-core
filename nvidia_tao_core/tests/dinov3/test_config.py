# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test cases for the DINOv3 dataclass config and its jsonschema conversion.

The DINOv3 config in tao-core is the source of truth for the generated skill
schemas in tao-skills-external and must stay aligned with
``nvidia_tao_pytorch.config.dinov3.default_config`` (bug 6465432).
"""

from dataclasses import asdict, fields

import pytest

from nvidia_tao_core.api_utils.dataclass2json_converter import (
    create_json_schema,
    dataclass_to_json,
)
from nvidia_tao_core.config.dinov3.default_config import (
    DINOv3CuDNNConfig,
    DINOv3ExportExpConfig,
    DINOv3TrainExpConfig,
    DINOv3TransformConfig,
    ExperimentConfig,
    LoRAConfig,
    PreservationConfig,
    SUPPORTED_BACKBONES,
    SUPPORTED_IMAGE_SIZES,
    map_params,
)

# Must match nvidia_tao_pytorch.config.dinov3.default_config.SUPPORTED_BACKBONES.
EXPECTED_BACKBONES = ["vit_s", "vit_s_plus", "vit_b", "vit_l", "vit_h_plus", "vit_7b"]
EXPECTED_PRETRAINED_DESCRIPTION = (
    "Path to DINOv3 pretrained weights matching the configured backbone. "
    "Accepts a timm-format directory or file, or a stripped TAO DINOv3 "
    "backbone checkpoint. DINOv2/NVDINOv2 checkpoints are not supported."
)


def test_supported_backbones_match_tao_pytorch():
    assert SUPPORTED_BACKBONES == EXPECTED_BACKBONES


def test_cudnn_defaults_match_dinov3_templates():
    config = ExperimentConfig()
    assert config.train.cudnn.benchmark is True
    assert config.train.cudnn.deterministic is False

    fields = DINOv3CuDNNConfig.__dataclass_fields__
    assert fields["benchmark"].metadata["description"]
    assert fields["benchmark"].metadata["display_name"] == "CuDNN benchmark"
    assert fields["benchmark"].metadata["popular"] == "no"
    assert fields["deterministic"].metadata["description"]
    assert fields["deterministic"].metadata["display_name"] == "CuDNN deterministic"
    assert fields["deterministic"].metadata["popular"] == "no"

    schema = create_json_schema(dataclass_to_json(config))
    cudnn = schema["properties"]["train"]["properties"]["cudnn"]
    assert cudnn["default"] == {"benchmark": True, "deterministic": False}
    assert cudnn["properties"]["benchmark"]["default"] is True
    assert cudnn["properties"]["benchmark"]["description"]
    assert cudnn["properties"]["benchmark"]["title"] == "CuDNN benchmark"
    assert cudnn["properties"]["deterministic"]["default"] is False
    assert cudnn["properties"]["deterministic"]["description"]
    assert cudnn["properties"]["deterministic"]["title"] == "CuDNN deterministic"


def test_logging_interval_defaults_to_every_step():
    """Short DINOv3 runs publish component losses without extra overrides."""
    config = ExperimentConfig()
    assert config.train.log_every_n_steps == 1

    field = DINOv3TrainExpConfig.__dataclass_fields__["log_every_n_steps"]
    assert field.metadata["valid_min"] == 1
    schema = create_json_schema(dataclass_to_json(config))
    logging_interval = schema["properties"]["train"]["properties"]["log_every_n_steps"]
    assert logging_interval["default"] == 1


def test_lora_and_preservation_schema_match_runtime_config():
    """Generated schemas expose the complete runtime adaptation controls."""
    config = ExperimentConfig()
    assert config.model.lora == LoRAConfig()
    assert config.model.preservation == PreservationConfig()

    schema = create_json_schema(dataclass_to_json(config))
    model = schema["properties"]["model"]["properties"]
    assert set(model["lora"]["properties"]) == {
        "enable", "rank", "alpha", "dropout", "target_modules", "num_last_blocks"
    }
    assert set(model["preservation"]["properties"]) == {
        "enable", "cls_mse_weight", "cls_cosine_weight"
    }


@pytest.mark.parametrize(
    ("config_name", "core_config"),
    (("LoRAConfig", LoRAConfig), ("PreservationConfig", PreservationConfig)),
)
def test_adaptation_defaults_match_tao_pytorch_when_available(
    config_name, core_config
):
    """Keep tao-core's schema source aligned with the optional runtime package."""
    runtime = pytest.importorskip(
        "nvidia_tao_pytorch.config.dinov3.default_config",
        reason="tao-pytorch is not installed in tao-core-only environments",
    )
    runtime_config = getattr(runtime, config_name)

    assert [field.name for field in fields(core_config)] == [
        field.name for field in fields(runtime_config)
    ]
    assert asdict(core_config()) == asdict(runtime_config())


def test_pretrained_model_path_description_is_dinov3_specific():
    """DINOv3 metadata must not inherit the NVDINOv2 checkpoint contract."""
    field = DINOv3TrainExpConfig.__dataclass_fields__["pretrained_model_path"]
    assert field.metadata["description"] == EXPECTED_PRETRAINED_DESCRIPTION
    assert field.metadata["default_value"] is None

    schema = create_json_schema(dataclass_to_json(ExperimentConfig()))
    pretrained = schema["properties"]["train"]["properties"]["pretrained_model_path"]
    assert pretrained["description"] == EXPECTED_PRETRAINED_DESCRIPTION
    assert "default" not in pretrained


def test_global_crop_description_is_not_single_resolution():
    """DINOv3 transform metadata must not imply that only 256 is supported."""
    field = DINOv3TransformConfig.__dataclass_fields__["global_crops_size"]
    assert field.metadata["description"] == "Size of global crops for DINOv3 training."

    schema = create_json_schema(dataclass_to_json(ExperimentConfig()))
    global_crops_size = (
        schema["properties"]["dataset"]["properties"]["transform"]["properties"]["global_crops_size"]
    )
    assert global_crops_size["description"] == field.metadata["description"]


def test_map_params_cover_all_backbones():
    for param, per_arch in map_params.items():
        assert sorted(per_arch.keys()) == sorted(EXPECTED_BACKBONES), (
            f"map_params[{param!r}] does not cover all supported backbones"
        )


def test_backbone_enum_in_json_schema():
    schema = create_json_schema(dataclass_to_json(ExperimentConfig()))
    backbone = schema["properties"]["model"]["properties"]["backbone"]["properties"]
    assert backbone["teacher_type"]["enum"] == EXPECTED_BACKBONES
    assert backbone["student_type"]["enum"] == EXPECTED_BACKBONES
    assert backbone["teacher_type"]["default"] == "vit_b"
    assert backbone["student_type"]["default"] == "vit_b"
    assert SUPPORTED_IMAGE_SIZES == (256, 512, 768)
    assert backbone["img_size"]["enum"] == list(SUPPORTED_IMAGE_SIZES)
    assert backbone["img_size"]["description"] == (
        "Backbone image size. Supported values are 256, 512, and 768."
    )


def test_experiment_config_actions():
    schema = create_json_schema(dataclass_to_json(ExperimentConfig()))
    for action in ("train", "inference", "export", "gen_trt_engine", "convert"):
        assert action in schema["properties"], f"missing {action} section"


def test_export_checkpoint_contract_selects_teacher():
    """The generated schema documents teacher selection for full checkpoints."""
    field = DINOv3ExportExpConfig.__dataclass_fields__["checkpoint"]
    assert "full Lightning training checkpoint" in field.metadata["description"]
    assert "always selects the EMA teacher backbone" in field.metadata["description"]

    schema = create_json_schema(dataclass_to_json(ExperimentConfig()))
    checkpoint = schema["properties"]["export"]["properties"]["checkpoint"]
    assert checkpoint["description"] == field.metadata["description"]
    assert checkpoint["title"] == "Path to checkpoint file"


def test_export_trace_shape_matches_backbone():
    """Export ONNX trace defaults match the patch-16 backbone (256, not nvdinov2's 518)."""
    schema = create_json_schema(dataclass_to_json(ExperimentConfig()))
    export = schema["properties"]["export"]["properties"]
    patch_size = schema["properties"]["model"]["properties"]["backbone"]["properties"]["patch_size"]["default"]
    assert export["input_width"]["default"] == 256
    assert export["input_height"]["default"] == 256
    assert export["input_width"]["default"] % patch_size == 0
    assert export["input_height"]["default"] % patch_size == 0
