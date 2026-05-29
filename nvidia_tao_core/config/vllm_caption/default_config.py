# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file for vLLM-based video captioning."""

from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class LLMParams:
    """Parameters for initializing vLLM's LLM engine."""

    model: str = STR_FIELD(
        value="Qwen/Qwen3-VL-235B-A22B-Instruct",
        description="HuggingFace model name or path."
    )
    tensor_parallel_size: int = INT_FIELD(
        value=8,
        default_value=8,
        description="Number of GPUs to use for tensor parallelism."
    )
    dtype: str = STR_FIELD(
        value="bfloat16",
        description="Data type for model weights.",
        valid_options="bfloat16,float16,float32"
    )
    gpu_memory_utilization: float = FLOAT_FIELD(
        value=0.95,
        default_value=0.95,
        valid_min=0.0,
        valid_max=1.0,
        description="Fraction of GPU memory to use."
    )
    max_model_len: int = INT_FIELD(
        value=131072,
        default_value=131072,
        description="Maximum sequence length (tokens) the model can handle."
    )
    enable_expert_parallel: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable expert parallelism for MoE models."
    )
    allowed_local_media_path: str = STR_FIELD(
        value="",
        default_value="",
        description="Base directory for local media files. If empty, derived from video paths."
    )


@dataclass
class SamplingParams:
    """Parameters for vLLM's SamplingParams."""

    max_tokens: int = INT_FIELD(
        value=8192,
        default_value=8192,
        description="Maximum number of tokens to generate."
    )
    temperature: float = FLOAT_FIELD(
        value=0.7,
        default_value=0.7,
        valid_min=0.0,
        description="Sampling temperature. Lower values produce more deterministic output."
    )
    top_p: float = FLOAT_FIELD(
        value=0.8,
        default_value=0.8,
        valid_min=0.0,
        valid_max=1.0,
        description="Top-p (nucleus) sampling probability."
    )


@dataclass
class VLLMCaptionModelConfig:
    """Model configuration for vLLM-based video captioning."""

    llm: LLMParams = DATACLASS_FIELD(
        LLMParams(),
        description="Parameters for the vLLM LLM engine."
    )
    sampling: SamplingParams = DATACLASS_FIELD(
        SamplingParams(),
        description="Parameters for vLLM sampling."
    )


@dataclass
class VLLMCaptionInferenceExpConfig:
    """Inference configuration for vLLM-based video captioning."""

    data_jsonl: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to input JSONL file. Each line must have a 'video_id' field."
    )
    caption_prompt_file: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to a text file containing the caption prompt/instructions."
    )
