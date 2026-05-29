# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file for DEFT."""

from dataclasses import dataclass
from typing import Optional
from omegaconf import MISSING
from nvidia_tao_core.config.utils.types import STR_FIELD, DATACLASS_FIELD


@dataclass
class DataConfig:
    """Dataset configuration for KPI gap analysis."""

    predictions_json: str = STR_FIELD(
        value=MISSING,
        default_value="<path to predictions JSON>",
        description="Path to predictions JSON file with response and gt fields."
    )
    videos_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Directory containing videos. If empty, video_id in predictions are treated as absolute paths."
    )


@dataclass
class ExperimentConfig:
    """Experiment configuration for KPI gap analysis."""

    data: DataConfig = DATACLASS_FIELD(DataConfig())
    results_dir: Optional[str] = STR_FIELD(
        value=MISSING,
        default_value="<path to output directory>",
        description="Output directory for kpi_gaps.jsonl and kpi_gaps_report.txt."
    )
