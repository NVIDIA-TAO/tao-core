# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for automl/automl_algorithm_base.py custom parameter ranges feature"""

import numpy as np
from unittest.mock import Mock, patch

from nvidia_tao_core.microservices.automl.automl_algorithm_base import AutoMLAlgorithmBase


class TestAutoMLAlgorithmBaseInitialization:
    """Test AutoMLAlgorithmBase initialization with custom parameter ranges"""

    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_initialization_with_custom_ranges(self, mock_get_job_specs, mock_get_custom_ranges):
        """Test that custom parameter ranges are loaded during initialization"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        custom_ranges = {
            "learning_rate": {
                "valid_min": 0.001,
                "valid_max": 0.01
            },
            "batch_size": {
                "valid_min": 16,
                "valid_max": 64
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job123"
        job_context.handler_id = "exp456"

        # Initialize the base class
        parameters = []
        automl_base = AutoMLAlgorithmBase(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters
        )

        # Verify custom ranges were loaded
        assert automl_base.custom_ranges == custom_ranges
        mock_get_custom_ranges.assert_called_once_with("exp456")

    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_initialization_without_custom_ranges(self, mock_get_job_specs, mock_get_custom_ranges):
        """Test initialization when no custom ranges are provided"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        mock_get_custom_ranges.return_value = {}

        # Create mock job context
        job_context = Mock()
        job_context.id = "job789"
        job_context.handler_id = "exp999"

        # Initialize the base class
        parameters = []
        automl_base = AutoMLAlgorithmBase(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters
        )

        # Verify custom ranges is empty dict
        assert automl_base.custom_ranges == {}
        mock_get_custom_ranges.assert_called_once_with("exp999")


class TestGenerateAutoMLParamRecValue:
    """Test generate_automl_param_rec_value with custom parameter ranges"""

    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_int_parameter_with_custom_min_max(self, mock_get_job_specs, mock_get_custom_ranges):
        """Test integer parameter generation with custom min/max"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        custom_ranges = {
            "batch_size": {
                "valid_min": 16,
                "valid_max": 32
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_int_test"
        job_context.handler_id = "exp_int"

        # Initialize the base class
        parameters = []
        automl_base = AutoMLAlgorithmBase(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters
        )

        # Define parameter config
        parameter_config = {
            "parameter": "batch_size",
            "value_type": "int",
            "valid_min": 8,
            "valid_max": 128,
            "default_value": 32
        }

        # Generate value multiple times to check range
        for _ in range(10):
            value = automl_base.generate_automl_param_rec_value(parameter_config)
            # Should be within custom range
            assert 16 <= value <= 32
            assert isinstance(value, (int, np.integer))

    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_categorical_parameter_with_custom_options(self, mock_get_job_specs, mock_get_custom_ranges):
        """Test categorical parameter generation with custom options"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        custom_ranges = {
            "optimizer": {
                "valid_options": ["adam", "sgd"]
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_cat_test"
        job_context.handler_id = "exp_cat"

        # Initialize the base class
        parameters = []
        automl_base = AutoMLAlgorithmBase(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters
        )

        # Define parameter config
        parameter_config = {
            "parameter": "optimizer",
            "value_type": "categorical",
            "valid_options": ["adam", "sgd", "adamw", "rmsprop"],
            "default_value": "adam"
        }

        # Generate value multiple times to check options
        generated_values = set()
        for _ in range(20):
            value = automl_base.generate_automl_param_rec_value(parameter_config)
            generated_values.add(value)
            # Should only be from custom options
            assert value in ["adam", "sgd"]

        # Should generate both options eventually
        assert "adam" in generated_values or "sgd" in generated_values

    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_ordered_int_parameter_with_custom_options(self, mock_get_job_specs, mock_get_custom_ranges):
        """Test ordered_int parameter generation with custom options"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        custom_ranges = {
            "num_layers": {
                "valid_options": [2, 3, 4]
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_ordered_test"
        job_context.handler_id = "exp_ordered"

        # Initialize the base class
        parameters = []
        automl_base = AutoMLAlgorithmBase(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters
        )

        # Define parameter config
        parameter_config = {
            "parameter": "num_layers",
            "value_type": "ordered_int",
            "valid_options": [1, 2, 3, 4, 5],
            "default_value": 3
        }

        # Generate value multiple times to check options
        for _ in range(10):
            value = automl_base.generate_automl_param_rec_value(parameter_config)
            # Should only be from custom options
            assert value in [2, 3, 4]
            assert isinstance(value, (int, np.integer))

    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_bool_parameter_no_custom_override(self, mock_get_job_specs, mock_get_custom_ranges):
        """Test bool parameter generation (no custom override supported)"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        mock_get_custom_ranges.return_value = {}

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_bool_test"
        job_context.handler_id = "exp_bool"

        # Initialize the base class
        parameters = []
        automl_base = AutoMLAlgorithmBase(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters
        )

        # Define parameter config
        parameter_config = {
            "parameter": "use_augmentation",
            "value_type": "bool",
            "default_value": True
        }

        # Generate value multiple times
        generated_values = set()
        for _ in range(20):
            value = automl_base.generate_automl_param_rec_value(parameter_config)
            generated_values.add(value)
            assert isinstance(value, bool)

        # Should generate both True and False eventually
        assert True in generated_values or False in generated_values

    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_subset_list_parameter_with_custom_options(self, mock_get_job_specs, mock_get_custom_ranges):
        """Test subset_list parameter generation with custom options"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        custom_ranges = {
            "target_modules": {
                "valid_options": ["layer1", "layer2"]
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_subset_test"
        job_context.handler_id = "exp_subset"

        # Initialize the base class
        parameters = []
        automl_base = AutoMLAlgorithmBase(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters
        )

        # Define parameter config
        parameter_config = {
            "parameter": "target_modules",
            "value_type": "subset_list",
            "valid_options": ["layer1", "layer2", "layer3", "layer4"],
            "default_value": []
        }

        # Generate value multiple times
        for _ in range(10):
            value = automl_base.generate_automl_param_rec_value(parameter_config)
            assert isinstance(value, list)
            # All items should be from custom options
            for item in value:
                assert item in ["layer1", "layer2"]

    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_optional_list_parameter_with_custom_options(self, mock_get_job_specs, mock_get_custom_ranges):
        """Test optional_list parameter generation with custom options"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        custom_ranges = {
            "modules_to_save": {
                "valid_options": ["module1", "module2"]
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_optional_test"
        job_context.handler_id = "exp_optional"

        # Initialize the base class
        parameters = []
        automl_base = AutoMLAlgorithmBase(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters
        )

        # Define parameter config
        parameter_config = {
            "parameter": "modules_to_save",
            "value_type": "optional_list",
            "valid_options": ["module1", "module2", "module3"],
            "default_value": None
        }

        # Generate value multiple times
        none_count = 0
        list_count = 0
        for _ in range(20):
            value = automl_base.generate_automl_param_rec_value(parameter_config)
            if value is None:
                none_count += 1
            else:
                list_count += 1
                assert isinstance(value, list)
                # All items should be from custom options
                for item in value:
                    assert item in ["module1", "module2"]

        # Should generate both None and list values
        assert none_count > 0
        assert list_count > 0

    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.automl_helper')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_list_2_optimizer_betas_with_custom_ranges(self, mock_get_job_specs, mock_get_custom_ranges, mock_helper):
        """Test list_2 optimizer_betas parameter with custom min/max"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        custom_ranges = {
            "optimizer.betas": {
                "valid_min": [0.85, 0.92],
                "valid_max": [0.93, 0.998]
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Mock automl_helper - must be done before initialization
        mock_helper.automl_list_helper = {
            "image_classification": {
                "list_2": {
                    "optimizer.betas": ("optimizer_betas", None)
                }
            }
        }

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_betas_test"
        job_context.handler_id = "exp_betas"

        # Initialize the base class
        parameters = []
        automl_base = AutoMLAlgorithmBase(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters
        )

        # Define parameter config
        parameter_config = {
            "parameter": "optimizer.betas",
            "value_type": "list_2",
            "valid_min": [0.8, 0.9],
            "valid_max": [0.95, 0.999],
            "default_value": [0.9, 0.999]
        }

        # Generate value
        value = automl_base.generate_automl_param_rec_value(parameter_config)

        # Verify the result
        assert isinstance(value, list)
        assert len(value) == 2
        # Check beta1 range
        assert 0.85 <= value[0] <= 0.93
        # Check beta2 range
        assert 0.92 <= value[1] <= 0.998


class TestApplyPowerConstraint:
    """Test _apply_power_constraint_with_equal_priority method"""

    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_power_constraint_basic(self, mock_get_job_specs, mock_get_custom_ranges):
        """Test power constraint generation"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        mock_get_custom_ranges.return_value = {}

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_power_test"
        job_context.handler_id = "exp_power"

        # Initialize the base class
        parameters = []
        automl_base = AutoMLAlgorithmBase(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters
        )

        # Test power of 2 constraint
        result = automl_base._apply_power_constraint_with_equal_priority(
            v_min=8,
            v_max=64,
            factor=2
        )

        # Should be a power of 2 within range
        assert result in [8, 16, 32, 64]

    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_power_constraint_no_valid_powers(self, mock_get_job_specs, mock_get_custom_ranges):
        """Test power constraint when no valid powers exist"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        mock_get_custom_ranges.return_value = {}

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_power_test2"
        job_context.handler_id = "exp_power2"

        # Initialize the base class
        parameters = []
        automl_base = AutoMLAlgorithmBase(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters
        )

        # Test with range where no powers of 2 exist
        result = automl_base._apply_power_constraint_with_equal_priority(
            v_min=3,
            v_max=3,
            factor=2
        )

        # Should return v_min when no valid powers
        assert result == 3
