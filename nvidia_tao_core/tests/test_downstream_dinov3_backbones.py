# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift guard for DINOv3 backbone options in downstream task configs (bug 6465432).

DINOv3 backbones are registered as selectable backbones for the segformer and
visual_changenet dense tasks (see nvidia_tao_pytorch/config/{segformer,
visual_changenet}/default_config.py). tao-core mirrors those configs and is the
source the tao-skills-external schemas are generated from, so it must carry the
same backbone options or the generated skill schemas drift.
"""
from dataclasses import fields

import pytest

from nvidia_tao_core.config.segformer.default_config import (
    BackboneConfig as SegformerBackboneConfig,
)
from nvidia_tao_core.config.visual_changenet.default_config import (
    BackboneConfig as ChangeNetBackboneConfig,
)

# DINOv3 backbones shared by SegFormer and Visual ChangeNet in tao-pytorch.
DINOV3_SHARED_BACKBONES = [
    "vit_small_dinov3",
    "vit_small_plus_dinov3",
    "vit_base_dinov3",
    "vit_large_dinov3",
    "vit_huge_plus_dinov3",
]


def _backbone_type_options(backbone_config_cls):
    """Return the backbone ``type`` field's valid_options as a list of strings."""
    (type_field,) = [f for f in fields(backbone_config_cls) if f.name == "type"]
    return type_field.metadata["valid_options"].split(",")


@pytest.mark.parametrize(
    "backbone_config_cls",
    [SegformerBackboneConfig, ChangeNetBackboneConfig],
    ids=["segformer", "visual_changenet"],
)
def test_dinov3_backbones_present(backbone_config_cls):
    options = _backbone_type_options(backbone_config_cls)
    missing = [b for b in DINOV3_SHARED_BACKBONES if b not in options]
    assert not missing, f"config backbone enum missing DINOv3 options: {missing}"


def test_visual_changenet_has_7b_dinov3():
    """Visual ChangeNet additionally supports the DINOv3 7B variant."""
    options = _backbone_type_options(ChangeNetBackboneConfig)
    assert "vit_7b_dinov3" in options
