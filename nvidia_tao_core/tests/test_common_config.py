# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for configuration shared by PyTorch TAO models."""

from nvidia_tao_core.config.common.common_config import TrainConfig


def test_checkpointer_defaults_are_safe_and_bounded_opt_in():
    config = TrainConfig().checkpointer
    fields = type(config).__dataclass_fields__

    assert config.enable_topk is False
    assert config.replace_periodic is False
    assert config.monitor is None
    assert config.mode is None
    assert config.save_top_k == 1
    assert config.filename == "model_best_{epoch:03d}"
    assert config.dirpath is None
    assert config.auto_insert_metric_name is False
    assert fields["monitor"].metadata["default_value"] is None
    assert fields["mode"].metadata["default_value"] is None
    assert fields["dirpath"].metadata["default_value"] is None
