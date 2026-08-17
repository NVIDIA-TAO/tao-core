# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Cosmos-RL vision configuration exposure."""

from dataclasses import fields
import importlib

from nvidia_tao_core.api_utils import dataclass2json_converter


TRAIN_AND_EVALUATE_VISION_FIELDS = {
    "fps",
    "nframes",
    "min_frames",
    "max_frames",
    "video_start",
    "video_end",
    "resized_height",
    "resized_width",
    "min_pixels",
    "max_pixels",
    "total_pixels",
}


def _module(action):
    return importlib.import_module(f"nvidia_tao_core.config.cosmos-rl.{action}")


def _schema_properties(action, *path):
    module = _module(action)
    schema = dataclass2json_converter.create_json_schema(
        dataclass2json_converter.dataclass_to_json(module.ExperimentConfig())
    )
    properties = schema["properties"]
    for part in path:
        properties = properties[part]["properties"]
    return properties


def test_train_exposes_daft_vision_options():
    names = {field.name for field in fields(_module("train").VisionConfig)}
    assert TRAIN_AND_EVALUATE_VISION_FIELDS <= names


def test_evaluate_exposes_daft_vision_options():
    names = {field.name for field in fields(_module("evaluate").VisionConfig)}
    assert TRAIN_AND_EVALUATE_VISION_FIELDS <= names


def test_inference_exposes_runtime_num_frames_option():
    names = {field.name for field in fields(_module("inference").InferenceConfig)}
    assert {"fps", "num_frames", "total_pixels"} <= names


def test_generated_train_and_evaluate_schemas_expose_daft_vision_options():
    train = _schema_properties("train", "custom", "vision")
    evaluate = _schema_properties("evaluate", "evaluate", "vision")

    assert TRAIN_AND_EVALUATE_VISION_FIELDS <= train.keys()
    assert TRAIN_AND_EVALUATE_VISION_FIELDS <= evaluate.keys()
    assert train["fps"]["type"] == evaluate["fps"]["type"] == "float"
    assert train["max_frames"]["type"] == evaluate["max_frames"]["type"] == "int"


def test_generated_inference_schema_exposes_num_frames():
    inference = _schema_properties("inference", "inference")
    assert inference["num_frames"]["type"] == "int"
