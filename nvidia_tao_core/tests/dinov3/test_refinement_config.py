# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the API schema and runtime parity for SSL refinement."""

from dataclasses import asdict, fields
import importlib
import os

import pytest

from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json
from nvidia_tao_core.config.dinov3.default_config import (
    DINOv3DatasetConfig,
    DINOv3TrainExpConfig,
    ExperimentConfig,
)
from nvidia_tao_core.config.dinov3.grit_score import GRITScoreConfig


def test_refinement_fields_are_exposed_in_api_schema():
    """Generated schemas expose manifest training and model-owned GRIT scoring."""
    schema = create_json_schema(dataclass_to_json(ExperimentConfig()))["properties"]
    manifest = schema["dataset"]["properties"]["train_manifest"]
    assert manifest["type"] == "string"
    assert "storage_type" in manifest["description"]
    scoring = schema["grit_score"]["properties"]
    assert scoring["neighbor_backend"]["enum"] == ["auto", "torch_exact", "faiss_exact"]
    assert scoring["batch_size"]["default"] == 12
    assert scoring["precomputed_consensus"]["default"] is False
    assert scoring["input_size"]["enum"] == [512]
    assert schema["dataset"]["properties"]["archive_cache_size"]["default"] == 8
    assert schema["train"]["properties"]["checkpoint_keep_last_n"]["default"] == 0
    assert schema["train"]["properties"]["auto_resume"]["default"] is True
    assert set(scoring) == {field.name for field in fields(GRITScoreConfig)}


def test_refinement_schema_matches_runtime_when_available():
    """Keep core defaults and metadata aligned with the optional PyTorch runtime."""
    try:
        runtime = importlib.import_module("nvidia_tao_pytorch.config.dinov3.default_config")
        if not hasattr(runtime, "GRITScoreConfig"):
            raise ImportError("Runtime release lacks the DEFT scoring schema")
    except ImportError:
        if os.environ.get("TAO_DEFT_REQUIRE_SCHEMA_PARITY") == "1":
            raise
        pytest.skip("Standalone Core CI does not install the stacked PyTorch feature; source handoff requires parity")
    assert asdict(GRITScoreConfig()) == asdict(runtime.GRITScoreConfig())
    core_fields = {field.name: dict(field.metadata) for field in fields(GRITScoreConfig)}
    runtime_fields = {field.name: dict(field.metadata) for field in fields(runtime.GRITScoreConfig)}
    assert core_fields == runtime_fields
    core_manifest = DINOv3DatasetConfig.__dataclass_fields__["train_manifest"]
    runtime_manifest = runtime.DINOv3DatasetConfig.__dataclass_fields__["train_manifest"]
    assert core_manifest.default == runtime_manifest.default
    assert dict(core_manifest.metadata) == dict(runtime_manifest.metadata)
    for config_type, runtime_type in (
        (DINOv3DatasetConfig, runtime.DINOv3DatasetConfig),
        (DINOv3TrainExpConfig, runtime.DINOv3TrainExpConfig),
    ):
        for name in config_type.__dataclass_fields__:
            core_field = config_type.__dataclass_fields__[name]
            runtime_field = runtime_type.__dataclass_fields__[name]
            assert core_field.default == runtime_field.default
            assert dict(core_field.metadata) == dict(runtime_field.metadata)


def test_nullable_defaults_labels_and_required_scoring_input():
    """API metadata must not turn nullable defaults into empty strings."""
    scoring = create_json_schema(dataclass_to_json(GRITScoreConfig()))
    for name in ("results_dir", "neighbor_device", "work_dir"):
        assert scoring["properties"][name].get("default") is None
        assert getattr(GRITScoreConfig(), name) is None
    assert "input_parquet" in scoring["required"]
    assert "checkpoint" not in scoring["required"]  # Valid in precomputed mode.
    assert all(field.metadata["display_name"] for field in fields(GRITScoreConfig))
