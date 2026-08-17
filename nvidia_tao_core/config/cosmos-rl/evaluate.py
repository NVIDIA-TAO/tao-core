# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default evaluation config file for Cosmos-RL"""

from typing import Optional, List
from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    BOOL_FIELD,
    FLOAT_FIELD,
    STR_FIELD,
    INT_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class TaskConfig:
    """Task configuration for evaluation."""

    type: str = STR_FIELD(
        default_value="its_directionality",
        value="its_directionality",
        display_name="Task type",
        valid_options="its_directionality,general",
        description="Type of evaluation task (general, its_directionality)"
    )


@dataclass
class MetricsConfig:
    """Metrics configuration for general evaluation."""

    names: List[str] = LIST_FIELD(
        ["bleu", "rouge", "bertscore"],
        display_name="Metric names",
        valid_options="bleu,rouge,bertscore",
        description="List of metrics to compute (bleu, rouge, bertscore)"
    )
    bertscore_model: Optional[str] = STR_FIELD(
        default_value="microsoft/deberta-xlarge-mnli",
        value="microsoft/deberta-xlarge-mnli",
        display_name="BERTScore model",
        description="Model to use for BERTScore computation (e.g., microsoft/deberta-xlarge-mnli)"
    )
    bertscore_lang: str = STR_FIELD(
        default_value="en",
        value="en",
        display_name="BERTScore language",
        description="Language for BERTScore computation"
    )


@dataclass
class SoftAccuracyConfig:
    """Soft accuracy configuration for general evaluation."""

    enabled: bool = BOOL_FIELD(
        default_value=True,
        value=True,
        display_name="Enable soft accuracy",
        description="Enable soft accuracy computation based on token overlap F1"
    )
    f1_threshold: float = FLOAT_FIELD(
        default_value=0.8,
        value=0.8,
        valid_min=0.0,
        valid_max=1.0,
        display_name="F1 threshold",
        description="F1 threshold for soft accuracy (predictions with F1 >= threshold are considered correct)"
    )


@dataclass
class DatasetConfig:
    """Dataset configuration for evaluation."""

    annotation_path: str = STR_FIELD(
        default_value="",
        value="",
        display_name="Annotation path",
        description="Path to the annotation JSON file containing evaluation samples"
    )
    media_dir: Optional[str] = STR_FIELD(
        default_value="",
        value="",
        display_name="Media directory",
        description="Optional path to media files directory (if different from annotation paths)"
    )
    system_prompt: str = STR_FIELD(
        default_value="You are a helpful assistant that can answer questions about a street-view CCTV footage. "
                      "The vehicles that need attention are marked with bounding boxes and IDs.",
        value="You are a helpful assistant that can answer questions about a street-view CCTV footage. "
              "The vehicles that need attention are marked with bounding boxes and IDs.",
        display_name="System prompt",
        description="System prompt for the evaluation tasks"
    )


@dataclass
class ModelConfig:
    """Model configuration for evaluation."""

    model_name: str = STR_FIELD(
        default_value="nvidia/Cosmos-Reason2-8B",
        value="nvidia/Cosmos-Reason2-8B",
        display_name="Model name",
        description="Model name or path to safetensors directory"
    )
    save_folder: str = STR_FIELD(
        default_value="cr1_1_zero_shot",
        value="cr1_1_zero_shot",
        display_name="Save folder",
        description="Folder name to save the output results"
    )
    tokenizer_model_name: str = STR_FIELD(
        default_value="qwen2.5-vl-7b",
        value="qwen2.5-vl-7b",
        display_name="Tokenizer model name",
        description="Tokenizer model name (qwen2.5-vl-7b, qwen2-vl-2b, qwen2.5-vl-32b, qwen2.5-vl-72b)"
    )
    dtype: str = STR_FIELD(
        default_value="bfloat16",
        value="bfloat16",
        display_name="Data type",
        description="Data type for model weights (bfloat16, float16)"
    )
    max_length: int = INT_FIELD(
        default_value=128000,
        value=128000,
        valid_min=1024,
        valid_max=1000000,
        display_name="Maximum sequence length",
        description="Maximum sequence length for the model"
    )
    tp_size: int = INT_FIELD(
        default_value=1,
        value=1,
        valid_min=1,
        valid_max=8,
        display_name="Tensor parallel size",
        description="Tensor parallel size for vLLM model loading (num_gpus = total_shard x tp_size)"
    )
    enable_lora: bool = BOOL_FIELD(
        default_value=False,
        value=False,
        display_name="Enable LoRA merging",
        description="Enable LoRA model merging (merge LoRA weights with base model before evaluation)"
    )
    base_model_path: Optional[str] = STR_FIELD(
        default_value="",
        value="",
        display_name="Base model path",
        description="Path to base model for LoRA merging (used when enable_lora is True)"
    )


@dataclass
class EvaluationConfig:
    """Evaluation parameters configuration."""

    answer_type: str = STR_FIELD(
        default_value="freeform",
        value="freeform",
        display_name="Answer type",
        description="Expected answer format (letter, reasoning, freeform)"
    )
    num_processes: int = INT_FIELD(
        default_value=40,
        value=40,
        valid_min=1,
        valid_max=128,
        display_name="Number of processes",
        description="Number of parallel workers for evaluation"
    )
    skip_saved: bool = BOOL_FIELD(
        default_value=False,
        value=False,
        display_name="Skip saved results",
        description="Skip tasks for which results are already saved"
    )
    seed: int = INT_FIELD(
        default_value=1,
        value=1,
        valid_min=0,
        valid_max=999999,
        display_name="Random seed",
        description="Random seed for reproducibility"
    )
    limit: int = INT_FIELD(
        default_value=-1,
        value=-1,
        valid_min=-1,
        valid_max=999999,
        display_name="Task limit",
        description="Limit the number of tasks to evaluate (-1 for no limit, useful for debugging)"
    )
    shard_id: int = INT_FIELD(
        default_value=0,
        value=0,
        valid_min=0,
        valid_max=63,
        display_name="Shard ID",
        description="Current shard ID (0-based)"
    )
    batch_size: int = INT_FIELD(
        default_value=50,
        value=50,
        valid_min=1,
        valid_max=500,
        display_name="Batch size",
        description="Number of requests to process in each batch during inference"
    )
    soft_accuracy: SoftAccuracyConfig = DATACLASS_FIELD(
        SoftAccuracyConfig(),
        description="Soft accuracy configuration for general evaluation"
    )


@dataclass
class VisionConfig:
    """Vision options forwarded to the Cosmos-RL media loader."""

    fps: Optional[float] = FLOAT_FIELD(
        value=None,
        default_value=None,
        valid_min=0.01,
        valid_max="inf",
        display_name="FPS",
        description=(
            "Frames per second sampled from each video. Mutually exclusive "
            "with nframes. min_frames and max_frames apply only in FPS mode."
        ),
        automl_enabled="TRUE"
    )

    nframes: Optional[int] = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        display_name="Number of frames",
        description=(
            "Number of frames sampled uniformly from each video. Mutually "
            "exclusive with fps."
        )
    )

    min_frames: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        valid_min=1,
        valid_max="inf",
        display_name="Minimum frames",
        description="Minimum sampled frames per video when fps is used."
    )

    max_frames: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        valid_min=1,
        valid_max="inf",
        display_name="Maximum frames",
        description="Maximum sampled frames per video when fps is used."
    )

    video_start: Optional[float] = FLOAT_FIELD(
        value=None,
        default_value=None,
        valid_min=0.0,
        valid_max="inf",
        display_name="Video start",
        description="Optional start time in seconds for video sampling."
    )

    video_end: Optional[float] = FLOAT_FIELD(
        value=None,
        default_value=None,
        valid_min=0.0,
        valid_max="inf",
        display_name="Video end",
        description="Optional end time in seconds for video sampling."
    )

    resized_height: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        valid_min=1,
        valid_max="inf",
        display_name="Resized height",
        description=(
            "Explicit resized video-frame height. Set together with "
            "resized_width."
        )
    )

    resized_width: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        valid_min=1,
        valid_max="inf",
        display_name="Resized width",
        description=(
            "Explicit resized video-frame width. Set together with "
            "resized_height."
        )
    )

    min_pixels: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        valid_min=1,
        valid_max="inf",
        display_name="Minimum pixels",
        description="Minimum per-frame pixel budget for vision processing."
    )

    max_pixels: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        valid_min=1,
        valid_max="inf",
        display_name="Maximum pixels",
        description="Maximum per-frame pixel budget for vision processing."
    )

    total_pixels: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        valid_min=1,
        valid_max="inf",
        display_name="Total pixels",
        description="Total number of pixels for vision processing."
    )


@dataclass
class GenerationConfig:
    """Generation parameters configuration."""

    max_retries: int = INT_FIELD(
        default_value=10,
        value=10,
        valid_min=0,
        valid_max=50,
        display_name="Maximum retries",
        description="Maximum number of retries for failed generations"
    )
    max_tokens: int = INT_FIELD(
        default_value=1024,
        value=1024,
        valid_min=1,
        valid_max=8192,
        display_name="Maximum tokens",
        description="Maximum number of tokens in the generated response"
    )
    temperature: float = FLOAT_FIELD(
        default_value=0.0,
        value=0.0,
        valid_min=0.0,
        valid_max=2.0,
        display_name="Temperature",
        description="Temperature for sampling (0.0 for greedy decoding)"
    )
    repetition_penalty: float = FLOAT_FIELD(
        default_value=1.0,
        value=1.0,
        valid_min=0.1,
        valid_max=2.0,
        display_name="Repetition penalty",
        description="Repetition penalty for generation"
    )
    presence_penalty: float = FLOAT_FIELD(
        default_value=0.0,
        value=0.0,
        valid_min=-2.0,
        valid_max=2.0,
        display_name="Presence penalty",
        description="Presence penalty for generation"
    )
    frequency_penalty: float = FLOAT_FIELD(
        default_value=0.0,
        value=0.0,
        valid_min=-2.0,
        valid_max=2.0,
        display_name="Frequency penalty",
        description="Frequency penalty for generation"
    )


@dataclass
class ResultsConfig:
    """Results and output configuration."""

    save_individual_results: bool = BOOL_FIELD(
        default_value=True,
        value=True,
        display_name="Save individual results",
        description="Save individual result JSON files for each sample"
    )
    save_confusion_matrix: bool = BOOL_FIELD(
        default_value=True,
        value=True,
        display_name="Save confusion matrix",
        description="Generate and save confusion matrix visualization"
    )
    save_metrics_summary: bool = BOOL_FIELD(
        default_value=True,
        value=True,
        display_name="Save metrics summary",
        description="Save overall metrics summary JSON file"
    )


@dataclass
class EvaluateConfig:
    """Main evaluation configuration."""

    task: TaskConfig = DATACLASS_FIELD(
        TaskConfig(),
        description="Task configuration for evaluation"
    )
    dataset: DatasetConfig = DATACLASS_FIELD(
        DatasetConfig(),
        description="Dataset configuration for evaluation"
    )
    model: ModelConfig = DATACLASS_FIELD(
        ModelConfig(),
        description="Model configuration"
    )
    evaluation: EvaluationConfig = DATACLASS_FIELD(
        EvaluationConfig(),
        description="Evaluation parameters"
    )
    vision: VisionConfig = DATACLASS_FIELD(
        VisionConfig(),
        description="Vision processing configuration"
    )
    generation: GenerationConfig = DATACLASS_FIELD(
        GenerationConfig(),
        description="Generation parameters"
    )
    metrics: MetricsConfig = DATACLASS_FIELD(
        MetricsConfig(),
        description="Metrics configuration for general evaluation"
    )
    results: ResultsConfig = DATACLASS_FIELD(
        ResultsConfig(),
        description="Results and output configuration"
    )
    num_gpus: int = INT_FIELD(
        default_value=1,
        value=1,
        valid_min=1,
        valid_max=8,
        display_name="Number of GPUs",
        description=(
            "Total number of GPUs to use. "
            "Automatically calculates total_shard = num_gpus / tp_size. "
            "Default: data parallelism (tp_size=1)."
        )
    )


@dataclass
class ExperimentConfig:
    """Experiment configuration for Cosmos-RL evaluation."""

    results_dir: str = STR_FIELD(
        default_value="",
        value="",
        display_name="Results directory",
        description="Directory to save evaluation results"
    )
    evaluate: EvaluateConfig = DATACLASS_FIELD(
        EvaluateConfig(),
        description="Evaluation configuration"
    )
