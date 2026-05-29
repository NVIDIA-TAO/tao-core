# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

from typing import List, Optional
from dataclasses import dataclass
from omegaconf import MISSING
from nvidia_tao_core.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class HueConfig:
    """Hue configuration template."""

    angle: int = INT_FIELD(value=0)


@dataclass
class BrightnessConfig:
    """Contrast configuration template."""

    offset: int = INT_FIELD(value=0)


@dataclass
class ColorAugmentationConfig:
    """Color augmentation configuration template."""

    hue: HueConfig = DATACLASS_FIELD(HueConfig())
    lum: BrightnessConfig = DATACLASS_FIELD(BrightnessConfig())


@dataclass
class DataConfig:
    """Dataset configuration template."""

    input_dir: str = STR_FIELD(value=MISSING, default_value="<specify image directory>")
    image_size: int = INT_FIELD(value=448)


@dataclass
class ExperimentConfig:
    """Experiment configuration template."""

    gpu_id: List[int] = LIST_FIELD(arrList=[0])
    dataset: DataConfig = DATACLASS_FIELD(DataConfig())
    color_aug: ColorAugmentationConfig = DATACLASS_FIELD(ColorAugmentationConfig())
    results_dir: Optional[str] = STR_FIELD(
        "", default_value=""
    )
