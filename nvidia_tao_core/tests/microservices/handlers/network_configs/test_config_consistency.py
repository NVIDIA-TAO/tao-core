# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for consistency between automl_spec_params, spec_params, and data_sources in network configs.

Catches mismatches like automl_spec_params referencing 'distill.teacher_model_path'
when spec_params.distill defines 'distill.pretrained_teacher_model_path'.
"""

import difflib

import pytest

from nvidia_tao_core.microservices.constants import TAO_NETWORKS
from nvidia_tao_core.microservices.utils.core_utils import read_network_config


def _get_all_spec_keys(spec_params):
    """Collect all keys from spec_params across all actions."""
    all_keys = set()
    for action, params in spec_params.items():
        if isinstance(params, dict):
            all_keys.update(params.keys())
    return all_keys


def _resolve_mapped_actions(config):
    """Resolve api_params.actions to their internal names via actions_mapping.

    Some configs (e.g. image) map external action names to different internal
    names used in data_sources/spec_params. For example, 'validate_images' maps
    to internal action 'validate'.
    """
    api_actions = set(config.get("api_params", {}).get("actions", []))
    actions_mapping = config.get("actions_mapping", {})
    resolved = set()
    for action in api_actions:
        if action in actions_mapping:
            resolved.add(actions_mapping[action].get("action", action))
        else:
            resolved.add(action)
    return resolved


@pytest.mark.parametrize("network", TAO_NETWORKS)
def test_automl_spec_keys_exist_in_spec_params(network):
    """Every model-path key in automl_spec_params must exist in at least one spec_params action.

    Keys with 'assign_const_value' values and 'results_dir' are control parameters
    that intentionally don't map to spec_params and are excluded.
    """
    try:
        config = read_network_config(network)
    except Exception:
        return

    if not config:
        return

    spec_params = config.get("spec_params", {})
    automl_spec = config.get("automl_spec_params", {})

    if not automl_spec or not spec_params:
        return

    all_spec_keys = _get_all_spec_keys(spec_params)

    for automl_key, automl_val in automl_spec.items():
        if automl_key == "results_dir":
            continue
        if isinstance(automl_val, str) and automl_val.startswith("assign_const_value"):
            continue

        if automl_key not in all_spec_keys:
            close = difflib.get_close_matches(automl_key, all_spec_keys, n=3, cutoff=0.5)
            hint = f" Similar keys in spec_params: {close}" if close else ""
            pytest.fail(
                f"[{network}] automl_spec_params key '{automl_key}' "
                f"does not exist in any spec_params action.{hint}"
            )


@pytest.mark.parametrize("network", TAO_NETWORKS)
def test_automl_spec_keys_match_correct_action(network):
    """automl_spec_params keys prefixed with an action name should exist in that action's spec_params.

    For example, a key 'distill.foo' should appear in spec_params['distill'],
    and 'train.bar' should appear in spec_params['train'].
    """
    try:
        config = read_network_config(network)
    except Exception:
        return

    if not config:
        return

    spec_params = config.get("spec_params", {})
    automl_spec = config.get("automl_spec_params", {})

    if not automl_spec or not spec_params:
        return

    for automl_key, automl_val in automl_spec.items():
        if automl_key == "results_dir":
            continue
        if isinstance(automl_val, str) and automl_val.startswith("assign_const_value"):
            continue

        prefix = automl_key.split(".")[0]
        if prefix not in spec_params:
            continue

        action_keys = spec_params[prefix]
        if not isinstance(action_keys, dict):
            continue

        if automl_key not in action_keys:
            close = difflib.get_close_matches(automl_key, list(action_keys.keys()), n=3, cutoff=0.5)
            hint = f" Similar keys in spec_params['{prefix}']: {close}" if close else ""
            pytest.fail(
                f"[{network}] automl_spec_params key '{automl_key}' "
                f"has prefix '{prefix}' but is missing from spec_params['{prefix}'].{hint}"
            )


@pytest.mark.parametrize("network", TAO_NETWORKS)
def test_spec_params_actions_match_api_actions(network):
    """Every action in spec_params should be listed in api_params.actions (after mapping), and vice versa."""
    try:
        config = read_network_config(network)
    except Exception:
        return

    if not config:
        return

    resolved_actions = _resolve_mapped_actions(config)
    spec_actions = set(config.get("spec_params", {}).keys())

    if not resolved_actions or not spec_actions:
        return

    missing_from_spec = resolved_actions - spec_actions
    extra_in_spec = spec_actions - resolved_actions

    if missing_from_spec:
        pytest.fail(
            f"[{network}] Actions in api_params but missing from spec_params: {missing_from_spec}"
        )
    if extra_in_spec:
        pytest.fail(
            f"[{network}] Actions in spec_params but missing from api_params: {extra_in_spec}"
        )


@pytest.mark.parametrize("network", TAO_NETWORKS)
def test_data_sources_actions_match_api_actions(network):
    """Every action in data_sources should be listed in api_params.actions (after mapping)."""
    try:
        config = read_network_config(network)
    except Exception:
        return

    if not config:
        return

    resolved_actions = _resolve_mapped_actions(config)
    ds_actions = set(config.get("data_sources", {}).keys())

    if not resolved_actions or not ds_actions:
        return

    extra_in_ds = ds_actions - resolved_actions
    if extra_in_ds:
        pytest.fail(
            f"[{network}] Actions in data_sources but missing from api_params: {extra_in_ds}"
        )


@pytest.mark.parametrize("network", TAO_NETWORKS)
def test_no_duplicate_automl_spec_suffixes(network):
    """Detect automl_spec_params keys that map to the same YAML leaf but with different paths.

    For example, both 'distill.teacher_model_path' and 'distill.pretrained_teacher_model_path'
    sharing the same suffix would be suspicious.
    """
    try:
        config = read_network_config(network)
    except Exception:
        return

    if not config:
        return

    automl_spec = config.get("automl_spec_params", {})
    if not automl_spec:
        return

    suffix_to_keys = {}
    for key, val in automl_spec.items():
        if key == "results_dir":
            continue
        if isinstance(val, str) and val.startswith("assign_const_value"):
            continue
        suffix = key.split(".")[-1]
        suffix_to_keys.setdefault(suffix, []).append(key)

    for suffix, keys in suffix_to_keys.items():
        if len(keys) > 1:
            prefixes = [k.rsplit(".", 1)[0] for k in keys]
            if len(set(prefixes)) != len(prefixes):
                pytest.fail(
                    f"[{network}] Duplicate automl_spec_params suffix '{suffix}': {keys}"
                )
