# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Qwen3-VL patch-embedding runtime workaround."""

from types import SimpleNamespace

import torch
from nvidia_tao_core.microservices.handlers import (
    huggingface_inference_microservice_server as server,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed


def test_qwen3vl_patch_embed_workaround_supports_backward():
    config = SimpleNamespace(
        patch_size=4,
        temporal_patch_size=2,
        in_channels=3,
        hidden_size=8,
    )
    module = Qwen3VLVisionPatchEmbed(config)
    hidden_states = torch.randn(5 * 3 * 2 * 4 * 4, requires_grad=True)

    output = module(hidden_states)
    output.sum().backward()

    assert getattr(module.forward, "_tao_linear_patch_embed", False)
    assert output.shape == (5, 8)
    assert hidden_states.grad is not None
    assert module.proj.weight.grad is not None


def test_qwen3vl_patch_embed_workaround_is_idempotent():
    patched_forward = Qwen3VLVisionPatchEmbed.forward

    server._apply_qwen3vl_cudnn_workaround()

    assert Qwen3VLVisionPatchEmbed.forward is patched_forward
