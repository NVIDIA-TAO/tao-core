# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contracts for Sparse4D service schemas and metric validation."""

from dataclasses import fields, is_dataclass
import importlib
from numbers import Real

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError, ValidationError

from nvidia_tao_core.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_core.microservices.enum_constants import Metrics
from nvidia_tao_core.microservices.utils.core_utils import (
    get_microservices_network_and_action, read_network_config,
)
from nvidia_tao_core.scripts.generate_schema import generate_schema


def _leaf_fields(config, prefix=""):
    """Walk instantiated dataclasses without duplicating their tuned defaults."""
    for field in fields(config):
        value = getattr(config, field.name)
        path = f"{prefix}.{field.name}" if prefix else field.name
        if is_dataclass(value):
            yield from _leaf_fields(value, path)
        else:
            yield path, field, value, type(config).__module__


def _schema_property(schema, path):
    """Resolve a nested generated-schema property."""
    for key in path.split("."):
        schema = schema["properties"][key]
    return schema


@pytest.mark.parametrize("action", ["train", "evaluate", "inference", "export"])
def test_sparse4d_generated_schema_preserves_dataclass_defaults(action):
    """Shared model/dataset defaults survive action filtering without drift."""
    schema = generate_schema("sparse4d", action)
    config = ExperimentConfig()
    for section in ("dataset", "model"):
        for path, _, value, _ in _leaf_fields(getattr(config, section), section):
            default = schema["default"]
            for key in path.split("."):
                default = default.get(key)
                if default is None:
                    break
            # The schema serializer omits optional None defaults.
            assert default == value, path


def test_sparse4d_metadata_constraints_are_consistent():
    """Ranges/options and UI labels remain valid after legitimate default tuning."""
    config = ExperimentConfig()
    schema = generate_schema("sparse4d", "train")
    for section in ("dataset", "model", "train"):
        for path, field, value, owner_module in _leaf_fields(getattr(config, section), section):
            metadata = field.metadata
            if ".sparse4d." in owner_module:
                assert str(metadata.get("description", "")).strip(), path
                assert str(metadata.get("display_name", "")).strip(), path
            if isinstance(value, Real) and not isinstance(value, bool):
                for key, compare in (("valid_min", lambda a, b: a >= b),
                                     ("valid_max", lambda a, b: a <= b)):
                    bound = metadata.get(key)
                    if bound not in (None, ""):
                        assert compare(value, float(bound)), (path, value, key, bound)
            options = metadata.get("valid_options")
            if options and isinstance(value, (str, int, float)) and value != "???":
                choices = str(options).split(",")
                assert str(value) in choices, (path, value, choices)
            if path.startswith(("model.head.loose_to_tight.", "model.sv_aux_head.")):
                assert _schema_property(schema, path)["default"] == value


def test_sparse4d_changed_eval_defaults_and_opt_in_switches():
    """Pin intentional compatibility decisions, not numerical tuning choices."""
    config = ExperimentConfig()
    assert config.dataset.eval_dist_fcn == "center_distance"
    assert config.dataset.eval_hota is False
    for enabled in (
        config.dataset.sync_route, config.dataset.resize_to_canonical_2d,
        config.model.cotrain_param_touch, config.model.sv_aux_head.enable,
        config.model.head.loose_to_tight.enable,
        config.model.head.loose_to_tight.pseudo_enable,
        config.model.head.instance_bank.reset_on_time_gap,
        config.train.scrub_nan_gradients,
    ):
        assert enabled is False
    assert config.dataset.ltt_2dgt_dedup_regex == config.dataset.rtdetr_2d_dedup_regex


def test_sparse4d_structured_config_accepts_cotrain_overrides():
    """Nested route overrides survive composition instead of being dropped."""
    override = {
        "dataset": {"ltt_2dgt_sidecar_dir": "/results/ltt", "sync_route": True,
                    "rtdetr_2d_per_class_score_thr": {"person": 0.75}},
        "model": {"head": {"loose_to_tight": {"enable": True, "mlp_ckpt": "/models/ltt.pth"}},
                  "sv_aux_head": {"enable": True}},
        "train": {"scrub_nan_gradients": True},
    }
    config = OmegaConf.merge(OmegaConf.structured(ExperimentConfig()), override)
    for section, fields_override in override.items():
        for name, value in fields_override.items():
            if not isinstance(value, dict):
                assert config[section][name] == value
    assert config.model.head.loose_to_tight.mlp_ckpt == "/models/ltt.pth"
    assert config.dataset.rtdetr_2d_per_class_score_thr.person == 0.75
    assert config.model.sv_aux_head.enable


@pytest.mark.parametrize("override", [
    {"dataset": {"rtdetr_2d_per_class_score_thr": {"person": "high"}}},
    {"dataset": {"real_scene_keywords": "SV2D"}},
    {"model": {"head": {"loose_to_tight": {"min_cams": 1.5}}}},
    {"model": {"sv_aux_head": {"enable": "maybe"}}},
])
def test_sparse4d_structured_config_rejects_invalid_cotrain_types(override):
    """Invalid route option types fail before the runtime consumes them."""
    with pytest.raises(ValidationError):
        OmegaConf.merge(OmegaConf.structured(ExperimentConfig()), override)


def test_sparse4d_structured_config_rejects_unknown_cotrain_keys():
    """Typos cannot silently disappear from a service-generated spec."""
    with pytest.raises(ConfigKeyError):
        OmegaConf.merge(OmegaConf.structured(ExperimentConfig()),
                        {"dataset": {"unknown_cotrain_key": True}})


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_sparse4d_metrics_use_production_validator(version, monkeypatch):
    """The real API re.match validator recognizes every emitted metric family."""
    module = importlib.import_module(f"nvidia_tao_core.microservices.blueprints.{version}.schemas")
    patterns = read_network_config("sparse4d")["metrics"]["dynamic_metric_patterns"]
    # Isolate this network's patterns; another network's broad regex must not
    # conceal a missing Sparse4D allowlist entry in this test.
    monkeypatch.setattr(module, "_get_dynamic_metric_patterns", lambda: patterns)
    validator = module.EnumFieldPrefix(Metrics)
    for name in ("loss_box_2d_0", "loss_box_2d_pseudo_5", "loss_cls_pseudo_3",
                 "loss_sv_head_1", "loss_sv_aux_cls", "loss_id_2", "loss_visibility_2",
                 "loss_cls_dn_0", "loss_box_dn_0", "loss_box_vel_dn_0", "loss_yns_0",
                 "loss_box_vel_0", "img_bbox_NuScenes/mAP", "img_bbox_IoU3D/mAP",
                 "img_bbox_HOTA/HOTA", "img_bbox_HOTA_IoU3D/person_HOTA"):
        assert validator._validate_dynamic_metric(name), name
    for name in ("loss_param_touch", "loss_id_x", "loss_cls_dn_0_suffix", "img_bbox_HOTA/"):
        assert not validator._validate_dynamic_metric(name), name


def test_sparse4d_service_data_source_paths_exist_in_schema():
    """Existing dataset bindings still resolve in the generated action schemas."""
    config = read_network_config("sparse4d")
    for action, overrides in config["data_sources"].items():
        network, mapped_action = get_microservices_network_and_action("sparse4d", action)
        for override in overrides:
            default = generate_schema(network, mapped_action)["default"]
            for key in override.split("."):
                assert key in default, override
                default = default[key]
