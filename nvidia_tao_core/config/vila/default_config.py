# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file"""

from typing import Optional
from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    DATACLASS_FIELD
)


@dataclass
class SystemConfig:
    """GPU and Multinode System config."""

    num_gpus: Optional[int] = INT_FIELD(
        value=1,
        default_value=1,
        description="Number of gpus. Should range from 1 - 8.",
        display_name="Number of GPUs",
        valid_min=1,
        popular="yes"
    )
    num_nodes: Optional[int] = INT_FIELD(
        value=1,
        default_value=1,
        description="Number of nodes. Only set if using multinode.",
        display_name="Number of Nodes",
        valid_min=1,
        popular="yes"
    )
    master_addr: Optional[str] = STR_FIELD(
        value="127.0.0.1",
        default_value="127.0.0.1",
        description="Master address. Only set if using multinode.",
        display_name="Master address"
    )
    node_rank: Optional[int] = INT_FIELD(
        value=0,
        default_value=0,
        description="Node rank. Only set if using multinode.",
        display_name="Node rank",
        valid_min=0
    )
    port: Optional[int] = INT_FIELD(
        value=24501,
        default_value=24501,
        description="Port number. Default is 24501",
        display_name="Port number"
    )
    save_on_each_node: Optional[str] = STR_FIELD(
        value="False",
        default_value="False",
        description='''
        Whether or not to save checkpoint on each node.
        Only set if using multinode without shared storage.
        Default is False''',
        display_name="Save on each Node"
    )


@dataclass
class DatasetConfig:
    """Dataset config."""

    dataset_name: Optional[str] = STR_FIELD(
        value="scienceqa",
        default_value="scienceqa",
        display_name="Dataset name",
        description=(
            "Dataset name. Default is scienceqa. Dataset name must be registered"
        )
    )
    dataset_path: Optional[str] = STR_FIELD(
        value=None,
        display_name="Dataset path",
        description="Path to the dataset file"
    )
    dataset_yaml_path: Optional[str] = STR_FIELD(
        value=None,
        display_name="Dataset yaml path",
        description="Path to the dataset yaml file"
    )
    mixture_path: Optional[str] = STR_FIELD(
        value=None,
        display_name="Mixture path",
        description="Path to the mixture file"
    )
    media_dir: Optional[str] = STR_FIELD(
        value=None,
        display_name="Media directory path",
        description="Path to the media directory"
    )
    data_path: Optional[str] = STR_FIELD(
        value=None,
        display_name="Annotation path",
        description="Path to the annotation file"
    )


@dataclass
class TrainConfig:
    """Train Config."""

    num_epochs: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        display_name="Number of Epochs",
        description="The number of epochs.",
        popular="yes"
    )
    batch_size: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        display_name="Batch size",
        description="The batch size during training.",
        popular="yes"
    )
    learning_rate: float = FLOAT_FIELD(
        value=1e-4,
        default_value=1e-4,
        valid_min=0,
        valid_max="inf",
        description="Learning rate. Default is 1e-4.",
        display_name="Learning rate",
        popular="yes"
    )
    vision_learning_rate: float = FLOAT_FIELD(
        value=1e-5,
        default_value=1e-5,
        valid_min=0,
        valid_max="inf",
        description="Vision learning rate. Default is 1e-5.",
    )
    weight_decay: float = FLOAT_FIELD(
        value=0.,
        default_value=0.,
        display_name="Weight decay",
        description="The weight decay coefficient.",
        popular="yes"
    )
    warmup_ratio: float = FLOAT_FIELD(
        value=0.03,
        default_value=0.03,
        valid_min=0,
        valid_max="inf",
        description="Warmup ratio.",
        display_name="Warmup ratio",
        popular="yes"
    )
    gradient_accumulation_steps: int = INT_FIELD(
        value=2,
        default_value=2,
        valid_min=1,
        valid_max="inf",
        display_name="Gradient accumulation steps.",
        description=(
            "Gradient accumulation steps. Your effective batch size is batch_size * gradient_accumulation_steps."
        ),
        popular="yes"
    )
    lora_r: int = INT_FIELD(
        value=16,
        default_value=16,
        valid_min=1,
        valid_max="inf",
        display_name="LORA r",
        description="LORA r. Default is 16.",
        popular="yes"
    )
    max_tiles: int = INT_FIELD(
        value=12,
        default_value=12,
        valid_min=1,
        valid_max="inf",
        display_name="Max tiles",
        description="Max tiles. Default is 12.",
        popular="yes"
    )
    video_max_tiles: int = INT_FIELD(
        value=6,
        default_value=6,
        valid_min=1,
        valid_max="inf",
        display_name="Video max tiles",
        description="Video max tiles. Default is 6.",
        popular="yes"
    )
    llm_mode: Optional[str] = STR_FIELD(
        value="lora",
        default_value="lora",
        valid_options="freeze,ft,lora",
        display_name="LLM mode",
        description="LLM mode: freeze, ft, or lora. Default is lora"
    )
    vision_mode: Optional[str] = STR_FIELD(
        value="ft",
        default_value="ft",
        valid_options="freeze,ft,lora",
        display_name="Vision tower mode",
        description="Vision tower mode: freeze, ft, or lora. Default is ft"
    )
    model_max_length: int = INT_FIELD(
        value=32768,
        default_value=32768,
        valid_min=1,
        valid_max="inf",
        display_name="Model max length",
        description="Model max length. Default is 32,768.",
        popular="yes"
    )
    checkpoint_interval: int = INT_FIELD(
        value=100,
        default_value=100,
        valid_min=1,
        valid_max="inf",
        display_name="Checkpoint interval",
        description="Checkpoint interval. Default is 100.",
        popular="yes"
    )
    total_checkpoint_limit: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        display_name="Total checkpoint limit",
        description="Total checkpoint limit. Default is 0.",
        popular="yes"
    )
    logging_interval: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        display_name="Logging interval",
        description="Logging interval. Default is 1.",
        popular="yes"
    )
    disable_wandb: Optional[str] = STR_FIELD(
        value="true",
        default_value="true",
        valid_options="true,false",
        display_name="Vision tower mode",
        description="Enable or disable wandb logging"
    )
    num_video_frames: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        display_name="Number of video frames",
        description="Number of video frames. Default is 8.",
        popular="yes"
    )
    num_time_tokens: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        display_name="Number of time tokens",
    )
    system: SystemConfig = DATACLASS_FIELD(SystemConfig(), description="GPU and Multinode System config")
    dataset: DatasetConfig = DATACLASS_FIELD(DatasetConfig(), description="Dataset config")


@dataclass
class EvaluateConfig:
    """Evaluation Config."""

    task: str = STR_FIELD(
        value="youcook2_val",
        default_value="youcook2_val",
        display_name="Task name",
        description="Task name for evaluation.",
        required="yes",
        valid_options=(
            "youcook2_val,scienceqa_image,scienceqa_image_text,scienceqa_text,"
            "scienceqa_video,scienceqa_video_text"
        )
    )
    dataset_yaml_path: Optional[str] = STR_FIELD(
        value=None,
        display_name="Dataset yaml path",
        description="Path to the dataset yaml file"
    )
    model_base: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="Base model name",
        description=(
            "Base model name. This is the original checkpoint if we're evaluating PEFT model."
        )
    )
    num_gpus: Optional[int] = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        display_name="Number of GPUs",
        description=(
            "Number of GPUS. If not provided, script attempts to detect using torch.cuda.device_count()."
        )
    )


@dataclass
class InferenceConfig:
    """Inference Config."""

    model_base: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="Base model name",
        description="Base model name for inference."
    )
    conv_mode: str = STR_FIELD(
        value="auto",
        default_value="auto",
        display_name="Conversation mode",
        description="Conversation mode for inference."
    )
    text: Optional[str] = STR_FIELD(
        value="What is this video about?",
        default_value="What is this video about?",
        display_name="Text input",
        description="Text prompt for inference."
    )
    media: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="Media path(s)",
        description="Path(s) to media files for inference (images, videos)."
    )
    num_video_frames: int = INT_FIELD(
        value=-1,
        default_value=-1,
        valid_min=-1,
        valid_max="inf",
        display_name="Number of video frames",
        description="Number of video frames. Default is -1.",
        popular="yes"
    )
    video_max_tiles: int = INT_FIELD(
        value=-1,
        default_value=-1,
        valid_min=-1,
        valid_max="inf",
        display_name="Video max tiles",
        description="Video max tiles. Default is -1.",
        popular="yes"
    )


@dataclass
class ExperimentConfig:
    """Experiment config."""

    model_path: str = STR_FIELD(
        value="",
        default_value="",
        display_name="Pretrained model path",
        description="Pretrained model path"
    )
    results_dir: str = STR_FIELD(
        value="",
        display_name="Output directory",
        description=(
            "Output directory. Must contain `lora` in the output name."
        )
    )
    train: TrainConfig = DATACLASS_FIELD(TrainConfig(), description="Train config")
    evaluate: EvaluateConfig = DATACLASS_FIELD(EvaluateConfig(), description="Evaluation config")
    inference: InferenceConfig = DATACLASS_FIELD(InferenceConfig(), description="Inference config")
