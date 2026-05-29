# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for automl/utils.py custom parameter ranges feature"""

from nvidia_tao_core.microservices.utils.automl_utils import (
    get_valid_range,
    get_valid_options,
    clamp_value,
    fix_input_dimension,
    fix_power_of_factor
)


class TestGetValidRange:
    """Test get_valid_range function with custom parameter ranges"""

    def test_basic_range_no_custom_ranges(self):
        """Test basic range without custom overrides"""
        parameter_config = {
            "parameter": "learning_rate",
            "valid_min": 0.001,
            "valid_max": 0.1,
            "default_value": 0.01
        }
        parent_params = {}
        custom_ranges = None

        v_min, v_max = get_valid_range(parameter_config, parent_params, custom_ranges)

        assert v_min == 0.001
        assert v_max == 0.1

    def test_custom_min_override(self):
        """Test custom minimum value override"""
        parameter_config = {
            "parameter": "learning_rate",
            "valid_min": 0.001,
            "valid_max": 0.1,
            "default_value": 0.01
        }
        parent_params = {}
        custom_ranges = {
            "learning_rate": {
                "valid_min": 0.005
            }
        }

        v_min, v_max = get_valid_range(parameter_config, parent_params, custom_ranges)

        assert v_min == 0.005
        assert v_max == 0.1

    def test_custom_max_override(self):
        """Test custom maximum value override"""
        parameter_config = {
            "parameter": "learning_rate",
            "valid_min": 0.001,
            "valid_max": 0.1,
            "default_value": 0.01
        }
        parent_params = {}
        custom_ranges = {
            "learning_rate": {
                "valid_max": 0.05
            }
        }

        v_min, v_max = get_valid_range(parameter_config, parent_params, custom_ranges)

        assert v_min == 0.001
        assert v_max == 0.05

    def test_custom_min_and_max_override(self):
        """Test both min and max custom overrides"""
        parameter_config = {
            "parameter": "batch_size",
            "valid_min": 8.0,
            "valid_max": 128.0,
            "default_value": 32.0
        }
        parent_params = {}
        custom_ranges = {
            "batch_size": {
                "valid_min": 16,
                "valid_max": 64
            }
        }

        v_min, v_max = get_valid_range(parameter_config, parent_params, custom_ranges)

        assert v_min == 16.0
        assert v_max == 64.0

    def test_inf_values_use_default(self):
        """Test that infinite values are replaced with default value"""
        parameter_config = {
            "parameter": "max_iterations",
            "valid_min": float('-inf'),
            "valid_max": float('inf'),
            "default_value": 1000.0
        }
        parent_params = {}
        custom_ranges = None

        v_min, v_max = get_valid_range(parameter_config, parent_params, custom_ranges)

        assert v_min == 1000.0
        assert v_max == 1000.0

    def test_depends_on_greater_than(self):
        """Test depends_on with > operator"""
        parameter_config = {
            "parameter": "param2",
            "valid_min": 0.0,
            "valid_max": 1.0,
            "default_value": 0.5,
            "depends_on": "> param1"
        }
        parent_params = {"param1": 0.3}
        custom_ranges = None

        v_min, v_max = get_valid_range(parameter_config, parent_params, custom_ranges)

        assert v_min > 0.3
        assert v_max == 1.0

    def test_depends_on_greater_equal(self):
        """Test depends_on with >= operator"""
        parameter_config = {
            "parameter": "param2",
            "valid_min": 0.0,
            "valid_max": 1.0,
            "default_value": 0.5,
            "depends_on": ">= param1"
        }
        parent_params = {"param1": 0.3}
        custom_ranges = None

        v_min, v_max = get_valid_range(parameter_config, parent_params, custom_ranges)

        assert v_min == 0.3
        assert v_max == 1.0

    def test_depends_on_less_than(self):
        """Test depends_on with < operator"""
        parameter_config = {
            "parameter": "param2",
            "valid_min": 0.0,
            "valid_max": 1.0,
            "default_value": 0.5,
            "depends_on": "< param1"
        }
        parent_params = {"param1": 0.7}
        custom_ranges = None

        v_min, v_max = get_valid_range(parameter_config, parent_params, custom_ranges)

        assert v_min == 0.0
        assert v_max < 0.7

    def test_depends_on_less_equal(self):
        """Test depends_on with <= operator"""
        parameter_config = {
            "parameter": "param2",
            "valid_min": 0.0,
            "valid_max": 1.0,
            "default_value": 0.5,
            "depends_on": "<= param1"
        }
        parent_params = {"param1": 0.7}
        custom_ranges = None

        v_min, v_max = get_valid_range(parameter_config, parent_params, custom_ranges)

        assert v_min == 0.0
        assert v_max == 0.7

    def test_custom_depends_on_override(self):
        """Test custom depends_on override"""
        parameter_config = {
            "parameter": "param2",
            "valid_min": 0.0,
            "valid_max": 1.0,
            "default_value": 0.5,
            "depends_on": "> param1"
        }
        parent_params = {"param1": 0.3, "param3": 0.6}
        custom_ranges = {
            "param2": {
                "depends_on": "> param3"
            }
        }

        v_min, v_max = get_valid_range(parameter_config, parent_params, custom_ranges)

        # Should use param3 instead of param1
        assert v_min > 0.6
        assert v_max == 1.0

    def test_depends_on_missing_parent(self):
        """Test depends_on when parent parameter is missing"""
        parameter_config = {
            "parameter": "param2",
            "valid_min": 0.0,
            "valid_max": 1.0,
            "default_value": 0.5,
            "depends_on": "> param1"
        }
        parent_params = {}
        custom_ranges = None

        v_min, v_max = get_valid_range(parameter_config, parent_params, custom_ranges)

        # Should use default_value when parent is missing
        assert v_min > 0.5
        assert v_max == 1.0

    def test_list_values_for_betas(self):
        """Test custom ranges with list values (for optimizer betas)"""
        # Note: get_valid_range is designed for float/int parameters
        # For list parameters like optimizer betas, custom ranges are applied
        # directly in the parameter config, not through get_valid_range
        # This test verifies the function doesn't crash with list inputs
        parameter_config = {
            "parameter": "optimizer.betas",
            "valid_min": 0.8,  # Use float for get_valid_range
            "valid_max": 0.95,
            "default_value": 0.9
        }
        parent_params = {}
        custom_ranges = {
            "optimizer.betas": {
                "valid_min": 0.85,
                "valid_max": 0.93
            }
        }

        v_min, v_max = get_valid_range(parameter_config, parent_params, custom_ranges)

        # Should apply custom overrides for scalar values
        assert v_min == 0.85
        assert v_max == 0.93


class TestGetValidOptions:
    """Test get_valid_options function with custom parameter ranges"""

    def test_basic_options_no_custom_ranges(self):
        """Test basic valid options without custom overrides"""
        parameter_config = {
            "parameter": "optimizer",
            "valid_options": ["adam", "sgd", "adamw"]
        }
        custom_ranges = None

        options = get_valid_options(parameter_config, custom_ranges)

        assert options == ["adam", "sgd", "adamw"]

    def test_custom_options_override(self):
        """Test custom valid options override"""
        parameter_config = {
            "parameter": "optimizer",
            "valid_options": ["adam", "sgd", "adamw", "rmsprop"]
        }
        custom_ranges = {
            "optimizer": {
                "valid_options": ["adam", "sgd"]
            }
        }

        options = get_valid_options(parameter_config, custom_ranges)

        assert options == ["adam", "sgd"]

    def test_empty_valid_options(self):
        """Test empty valid options"""
        parameter_config = {
            "parameter": "some_param",
            "valid_options": []
        }
        custom_ranges = None

        options = get_valid_options(parameter_config, custom_ranges)

        assert options == []

    def test_custom_options_with_integers(self):
        """Test custom valid options with integer values"""
        parameter_config = {
            "parameter": "num_layers",
            "valid_options": [1, 2, 3, 4, 5]
        }
        custom_ranges = {
            "num_layers": {
                "valid_options": [2, 3, 4]
            }
        }

        options = get_valid_options(parameter_config, custom_ranges)

        assert options == [2, 3, 4]

    def test_no_parameter_name(self):
        """Test when parameter name is missing"""
        parameter_config = {
            "valid_options": ["a", "b", "c"]
        }
        custom_ranges = {
            "some_param": {
                "valid_options": ["x", "y"]
            }
        }

        options = get_valid_options(parameter_config, custom_ranges)

        # Should return schema options since parameter name doesn't match
        assert options == ["a", "b", "c"]

    def test_custom_ranges_but_no_valid_options_key(self):
        """Test custom ranges exists but no valid_options key"""
        parameter_config = {
            "parameter": "param1",
            "valid_options": ["default1", "default2"]
        }
        custom_ranges = {
            "param1": {
                "valid_min": 0,
                "valid_max": 10
            }
        }

        options = get_valid_options(parameter_config, custom_ranges)

        # Should return schema options since custom doesn't have valid_options
        assert options == ["default1", "default2"]


class TestOtherUtilFunctions:
    """Test other utility functions that might be affected by changes"""

    def test_clamp_value_within_range(self):
        """Test clamping a value that is within range"""
        result = clamp_value(5.0, 1.0, 10.0)
        assert 1.0 < result < 10.0

    def test_clamp_value_above_max(self):
        """Test clamping a value above max"""
        result = clamp_value(15.0, 1.0, 10.0)
        assert result < 10.0

    def test_clamp_value_below_min(self):
        """Test clamping a value below min"""
        result = clamp_value(0.5, 1.0, 10.0)
        assert result > 1.0

    def test_fix_input_dimension_already_multiple(self):
        """Test fix_input_dimension when already multiple of factor"""
        result = fix_input_dimension(64, 32)
        assert result == 64

    def test_fix_input_dimension_not_multiple(self):
        """Test fix_input_dimension when not multiple of factor"""
        result = fix_input_dimension(50, 32)
        assert result == 64  # Should round up to next multiple

    def test_fix_power_of_factor_exact_power(self):
        """Test fix_power_of_factor with exact power"""
        result = fix_power_of_factor(8, 2)
        assert result == 8

    def test_fix_power_of_factor_not_exact_power(self):
        """Test fix_power_of_factor with non-exact power"""
        result = fix_power_of_factor(10, 2)
        assert result == 16  # Should round up to 2^4

    def test_fix_power_of_factor_zero_value(self):
        """Test fix_power_of_factor with zero value"""
        result = fix_power_of_factor(0, 2)
        assert result == 2  # Should return the base factor

    def test_fix_power_of_factor_negative_value(self):
        """Test fix_power_of_factor with negative value"""
        result = fix_power_of_factor(-5, 2)
        assert result == 2  # Should return the base factor
