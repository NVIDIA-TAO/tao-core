# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default quantization config file for Cosmos-RL"""

from typing import Optional
from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    BOOL_FIELD,
    FLOAT_FIELD,
    STR_FIELD,
    INT_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class CalibrationDatasetConfig:
    """Calibration dataset configuration for quantization."""

    # HuggingFace dataset options
    dataset_id: Optional[str] = STR_FIELD(
        default_value="lmms-lab/flickr30k",
        value="lmms-lab/flickr30k",
        display_name="Dataset ID",
        description=(
            "HuggingFace dataset ID for calibration (e.g., lmms-lab/flickr30k). "
            "Leave empty to use custom dataset."
        )
    )
    dataset_split: Optional[str] = STR_FIELD(
        default_value="test[:512]",
        value="test[:512]",
        display_name="Dataset split",
        description=(
            "Dataset split to use for calibration (e.g., test[:512]). "
            "Only used with HuggingFace datasets."
        )
    )

    # Custom dataset options (alternative to HuggingFace)
    annotation_path: Optional[str] = STR_FIELD(
        default_value="",
        value="",
        display_name="Annotation path",
        description=(
            "Path to custom annotation JSON file for calibration. "
            "Use this instead of dataset_id for local datasets."
        )
    )
    media_dir: Optional[str] = STR_FIELD(
        default_value="",
        value="",
        display_name="Media directory",
        description=(
            "Directory containing media files (images/videos) for custom dataset. "
            "Paths in annotations are relative to this directory."
        )
    )

    # Common calibration parameters
    num_calibration_samples: int = INT_FIELD(
        default_value=512,
        value=512,
        valid_min=1,
        valid_max=10000,
        display_name="Number of calibration samples",
        description="Number of samples to use for calibration"
    )
    max_sequence_length: int = INT_FIELD(
        default_value=2048,
        value=2048,
        valid_min=128,
        valid_max=8192,
        display_name="Maximum sequence length",
        description="Maximum sequence length for tokenization during calibration"
    )


@dataclass
class QuantizationMethodConfig:
    """Quantization method configuration."""

    quantization_scheme: str = STR_FIELD(
        default_value="FP8_DYNAMIC",
        value="FP8_DYNAMIC",
        display_name="Quantization scheme",
        valid_options="FP8_DYNAMIC,NVFP4,FP8,W8A8,W8A16,W4A16",
        description="Quantization scheme to use (FP8_DYNAMIC, NVFP4, FP8, W8A8, W8A16, W4A16)"
    )
    kv_precision: str = STR_FIELD(
        default_value="bf16",
        value="bf16",
        display_name="KV cache precision",
        valid_options="bf16,fp8",
        description="Precision for KV cache quantization (bf16 = no KV quantization, fp8 = FP8 KV cache)"
    )
    smoothing_strength: float = FLOAT_FIELD(
        default_value=0.8,
        value=0.8,
        valid_min=0.0,
        valid_max=1.0,
        display_name="SmoothQuant smoothing strength",
        description="Smoothing strength for SmoothQuant (0.0 to 1.0, higher = more smoothing)"
    )
    skip_test_generation: bool = BOOL_FIELD(
        default_value=False,
        value=False,
        display_name="Skip test generation",
        description="Skip running a test generation after quantization to verify the model"
    )


@dataclass
class ModelConfig:
    """Model configuration for quantization."""

    model_path: str = STR_FIELD(
        default_value="nvidia/Cosmos-Reason1-7B",
        value="nvidia/Cosmos-Reason1-7B",
        display_name="Model path",
        description="Path or HuggingFace model ID to quantize (e.g., nvidia/Cosmos-Reason1-7B)"
    )

    # LoRA configuration
    enable_lora: bool = BOOL_FIELD(
        default_value=False,
        value=False,
        display_name="Enable LoRA",
        description="Enable LoRA model merging (required if model_path is a LoRA checkpoint)"
    )
    base_model_path: Optional[str] = STR_FIELD(
        default_value="",
        value="",
        display_name="Base model path",
        description="Base model path for LoRA merging (required if enable_lora is True)"
    )


@dataclass
class QuantizeConfig:
    """Main quantization configuration."""

    model: ModelConfig = DATACLASS_FIELD(
        ModelConfig(),
        description="Model configuration"
    )
    calibration_dataset: CalibrationDatasetConfig = DATACLASS_FIELD(
        CalibrationDatasetConfig(),
        description="Calibration dataset configuration"
    )
    quantization_method: QuantizationMethodConfig = DATACLASS_FIELD(
        QuantizationMethodConfig(),
        description="Quantization method configuration"
    )


@dataclass
class ExperimentConfig:
    """Experiment configuration for Cosmos-RL quantization."""

    results_dir: str = STR_FIELD(
        default_value="",
        value="",
        display_name="Results directory",
        description="Directory to save quantization results and logs"
    )
    quantize: QuantizeConfig = DATACLASS_FIELD(
        QuantizeConfig(),
        description="Quantization configuration"
    )
