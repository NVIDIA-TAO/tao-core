# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for relational depends_on constraints (new split format).

Verifies that all AutoML algorithms and get_valid_range handle the new-style
depends_on format where the operator lives in math_cond (e.g. "> depends_on")
and depends_on holds only the parameter name. This is the format used by DINO,
Grounding DINO, and Deformable DETR for train_random_crop_max.

Regression test for: IndexError / ValueError when math_cond="> depends_on"
and depends_on="dataset.augmentation.train_random_crop_min".
"""

import numpy as np
from unittest.mock import Mock, patch

from nvidia_tao_core.microservices.utils.automl_utils import get_valid_range


CROP_MIN_CONFIG = {
    "parameter": "dataset.augmentation.train_random_crop_min",
    "value_type": "int",
    "valid_min": 32,
    "valid_max": 1024,
    "default_value": 384,
    "valid_options": [],
    "option_weights": "",
    "math_cond": "",
    "parent_param": "TRUE",
    "depends_on": "",
}

CROP_MAX_CONFIG = {
    "parameter": "dataset.augmentation.train_random_crop_max",
    "value_type": "int",
    "valid_min": 32,
    "valid_max": 1333,
    "default_value": 600,
    "valid_options": [],
    "option_weights": "",
    "math_cond": "> depends_on",
    "parent_param": "",
    "depends_on": "dataset.augmentation.train_random_crop_min",
}


def _make_algo(algo_cls, extra_init_kwargs=None, mock_override_epochs=False):
    """Instantiate an algorithm with all DB/file dependencies mocked out."""
    patchers = [
        patch(
            "nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges",
            return_value={},
        ),
        patch(
            "nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs",
            return_value={},
        ),
    ]
    if mock_override_epochs:
        patchers.append(
            patch.object(algo_cls, "override_num_epochs", lambda self, *a, **kw: None)
        )

    for p in patchers:
        p.start()

    try:
        job_context = Mock()
        job_context.id = "test_job"
        job_context.handler_id = "test_handler"

        init_kwargs = dict(
            job_context=job_context,
            root="/tmp/test_root/subdir",
            network="dino",
            parameters=[CROP_MIN_CONFIG, CROP_MAX_CONFIG],
        )
        if extra_init_kwargs:
            init_kwargs.update(extra_init_kwargs)

        algo = algo_cls(**init_kwargs)
        algo.parent_params["dataset.augmentation.train_random_crop_min"] = 390
    finally:
        for p in patchers:
            p.stop()

    return algo


# --------------------------------------------------------- get_valid_range
class TestGetValidRangeNewDependsOnFormat:
    """get_valid_range must handle the split depends_on + math_cond format."""

    def test_new_format_greater_than(self):
        parent_params = {"dataset.augmentation.train_random_crop_min": 390}
        v_min, v_max = get_valid_range(CROP_MAX_CONFIG, parent_params, None)
        assert v_min > 390, f"v_min ({v_min}) should be > 390"
        assert v_max == 1333

    def test_new_format_less_than(self):
        config = {**CROP_MAX_CONFIG, "math_cond": "< depends_on"}
        parent_params = {"dataset.augmentation.train_random_crop_min": 390}
        v_min, v_max = get_valid_range(config, parent_params, None)
        assert v_min == 32
        assert v_max < 390, f"v_max ({v_max}) should be < 390"

    def test_new_format_greater_equal(self):
        config = {**CROP_MAX_CONFIG, "math_cond": ">= depends_on"}
        parent_params = {"dataset.augmentation.train_random_crop_min": 390}
        v_min, v_max = get_valid_range(config, parent_params, None)
        assert v_min == 390
        assert v_max == 1333

    def test_new_format_less_equal(self):
        config = {**CROP_MAX_CONFIG, "math_cond": "<= depends_on"}
        parent_params = {"dataset.augmentation.train_random_crop_min": 390}
        v_min, v_max = get_valid_range(config, parent_params, None)
        assert v_min == 32
        assert v_max == 390

    def test_old_format_still_works(self):
        config = {
            **CROP_MAX_CONFIG,
            "math_cond": "",
            "depends_on": "> dataset.augmentation.train_random_crop_min",
        }
        parent_params = {"dataset.augmentation.train_random_crop_min": 390}
        v_min, v_max = get_valid_range(config, parent_params, None)
        assert v_min > 390
        assert v_max == 1333

    def test_no_math_cond_no_crash(self):
        config = {
            **CROP_MAX_CONFIG,
            "math_cond": "",
            "depends_on": "dataset.augmentation.train_random_crop_min",
        }
        parent_params = {"dataset.augmentation.train_random_crop_min": 390}
        v_min, v_max = get_valid_range(config, parent_params, None)
        assert v_min == 32
        assert v_max == 1333


# --------------------------------------------------------- Bayesian
class TestBayesianRelationalConstraint:

    @patch("nvidia_tao_core.microservices.automl.bayesian.get_total_epochs", return_value=10)
    def test_int_param_with_depends_on_does_not_crash(self, _):
        from nvidia_tao_core.microservices.automl.bayesian import Bayesian
        algo = _make_algo(Bayesian)
        value = algo.generate_automl_param_rec_value(CROP_MAX_CONFIG, 0.5)
        assert isinstance(value, (int, np.integer))
        assert 32 <= value <= 1333


# --------------------------------------------------------- BFBO
class TestBFBORelationalConstraint:

    @patch("nvidia_tao_core.microservices.automl.bfbo.get_total_epochs", return_value=10)
    def test_int_param_with_depends_on_does_not_crash(self, _):
        from nvidia_tao_core.microservices.automl.bfbo import BFBO
        algo = _make_algo(BFBO)
        value = algo.generate_automl_param_rec_value(CROP_MAX_CONFIG, 0.5)
        assert isinstance(value, (int, np.integer))
        assert 32 <= value <= 1333


# --------------------------------------------------------- HyperBand
class TestHyperBandRelationalConstraint:

    def test_int_param_with_depends_on_does_not_crash(self):
        from nvidia_tao_core.microservices.automl.hyperband import HyperBand
        algo = _make_algo(
            HyperBand,
            extra_init_kwargs=dict(max_epochs=10, reduction_factor=3, epoch_multiplier=1),
            mock_override_epochs=True,
        )
        value = algo.generate_automl_param_rec_value(CROP_MAX_CONFIG)
        assert isinstance(value, (int, np.integer))
        assert 32 <= value <= 1333


# --------------------------------------------------------- ASHA
class TestASHARelationalConstraint:

    def test_int_param_with_depends_on_does_not_crash(self):
        from nvidia_tao_core.microservices.automl.asha import ASHA
        algo = _make_algo(
            ASHA,
            extra_init_kwargs=dict(max_epochs=10, reduction_factor=3, epoch_multiplier=1),
            mock_override_epochs=True,
        )
        value = algo.generate_automl_param_rec_value(CROP_MAX_CONFIG)
        assert isinstance(value, (int, np.integer))
        assert 32 <= value <= 1333


# --------------------------------------------------------- BOHB
class TestBOHBRelationalConstraint:

    def test_int_param_with_depends_on_does_not_crash(self):
        from nvidia_tao_core.microservices.automl.bohb import BOHB
        algo = _make_algo(
            BOHB,
            extra_init_kwargs=dict(max_epochs=10, reduction_factor=3, epoch_multiplier=1),
            mock_override_epochs=True,
        )
        value = algo.generate_automl_param_rec_value(CROP_MAX_CONFIG, 0.5)
        assert isinstance(value, (int, np.integer))
        assert 32 <= value <= 1333


# --------------------------------------------------------- DEHB
class TestDEHBRelationalConstraint:

    def test_vector_to_config_with_depends_on_does_not_crash(self):
        from nvidia_tao_core.microservices.automl.dehb import DEHB
        algo = _make_algo(
            DEHB,
            extra_init_kwargs=dict(max_epochs=10, reduction_factor=3, epoch_multiplier=1),
            mock_override_epochs=True,
        )
        vector = np.array([0.5, 0.7])
        config = algo._vector_to_config(vector)
        crop_max = config.get("dataset.augmentation.train_random_crop_max")
        assert crop_max is not None
        assert isinstance(crop_max, (int, np.integer))
        assert 32 <= crop_max <= 1333
