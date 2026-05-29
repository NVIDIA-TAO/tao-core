# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file"""

from typing import Optional
from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    DATACLASS_FIELD,
    BOOL_FIELD,
)


@dataclass
class InferenceConfig:
    """Inference config."""

    media: str = STR_FIELD(
        default_value="",
        value="",
        display_name="Media path",
        description="Path to a single video/image file or a folder containing multiple files for batch inference"
    )
    type: str = STR_FIELD(
        default_value="video",
        value="video",
        display_name="Media type",
        description="Type of media to process",
        valid_options="video,image",
    )
    prompt: str = STR_FIELD(
        default_value="Describe this video.",
        value="Describe this video.",
        display_name="Prompt",
        description="Prompt for inference"
    )
    fps: Optional[int] = INT_FIELD(
        default_value=4,
        value=4,
        display_name="FPS",
        description="FPS for inference"
    )
    total_pixels: Optional[int] = INT_FIELD(
        default_value=6422528,
        value=6422528,
        display_name="Total pixels",
        description="Total pixels for inference"
    )
    max_new_tokens: Optional[int] = INT_FIELD(
        default_value=4096,
        value=4096,
        display_name="Max new tokens",
        description="Max new tokens for inference"
    )
    num_gpus: Optional[int] = INT_FIELD(
        default_value=1,
        value=1,
        valid_min=1,
        valid_max=8,
        display_name="Number of GPUs",
        description="Number of GPUs to use for inference (enables multi-GPU tensor parallelism)"
    )
    enable_lora: bool = BOOL_FIELD(
        default_value=False,
        value=False,
        display_name="Enable LoRA merging",
        description="Enable LoRA model merging (merge LoRA weights with base model before inference)"
    )
    base_model_path: Optional[str] = STR_FIELD(
        default_value="",
        value="",
        display_name="Base model path",
        description="Path to base model for LoRA merging (used when enable_lora is True)"
    )


@dataclass
class ExperimentConfig:
    """Experiment config."""

    results_dir: str = STR_FIELD(
        default_value="",
        value="",
        display_name="Results directory",
        description="Directory to save inference results"
    )
    inference: InferenceConfig = DATACLASS_FIELD(InferenceConfig(), description="Inference config.")
