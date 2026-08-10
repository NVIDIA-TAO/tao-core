# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Qwen3-VL patch-embedding runtime workaround."""

from types import SimpleNamespace

import pytest
import torch
from nvidia_tao_core.microservices.handlers import (
    huggingface_inference_microservice_server as server,
)


def _qwen3vl_patch_embed_class():
    """Load the optional Transformers implementation used by this regression."""
    module = pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
    return module.Qwen3VLVisionPatchEmbed


def test_qwen3vl_patch_embed_workaround_supports_backward():
    qwen3vl_patch_embed = _qwen3vl_patch_embed_class()
    config = SimpleNamespace(
        patch_size=4,
        temporal_patch_size=2,
        in_channels=3,
        hidden_size=8,
    )
    module = qwen3vl_patch_embed(config)
    hidden_states = torch.randn(5 * 3 * 2 * 4 * 4, requires_grad=True)

    output = module(hidden_states)
    output.sum().backward()

    assert getattr(module.forward, "_tao_linear_patch_embed", False)
    assert output.shape == (5, 8)
    assert hidden_states.grad is not None
    assert module.proj.weight.grad is not None


def test_qwen3vl_patch_embed_workaround_is_idempotent():
    qwen3vl_patch_embed = _qwen3vl_patch_embed_class()
    patched_forward = qwen3vl_patch_embed.forward

    server._apply_qwen3vl_cudnn_workaround()

    assert qwen3vl_patch_embed.forward is patched_forward
