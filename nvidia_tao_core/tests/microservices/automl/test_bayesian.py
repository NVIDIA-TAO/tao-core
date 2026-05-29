# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for automl/bayesian.py custom parameter ranges feature"""

import numpy as np
from unittest.mock import Mock, patch

from nvidia_tao_core.microservices.automl.bayesian import Bayesian


class TestBayesianCustomRanges:
    """Test Bayesian algorithm with custom parameter ranges"""

    @patch('nvidia_tao_core.microservices.automl.bayesian.get_total_epochs')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.'
           'get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_initialization_with_custom_ranges(
        self, mock_get_job_specs, mock_get_custom_ranges, mock_get_total_epochs
    ):
        """Test Bayesian initialization with custom parameter ranges"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        mock_get_total_epochs.return_value = 10
        custom_ranges = {
            "learning_rate": {
                "valid_min": 0.001,
                "valid_max": 0.01
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_bayesian_test"
        job_context.handler_id = "exp_bayesian"

        # Initialize Bayesian
        parameters = [{"parameter": "learning_rate"}]
        bayesian = Bayesian(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters
        )

        # Verify custom ranges were loaded
        assert bayesian.custom_ranges == custom_ranges

    @patch('nvidia_tao_core.microservices.automl.bayesian.get_total_epochs')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.'
           'get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_generate_param_rec_value_float_with_custom_range(
        self, mock_get_job_specs, mock_get_custom_ranges, mock_get_total_epochs
    ):
        """Test float parameter generation with custom range in Bayesian"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        mock_get_total_epochs.return_value = 10
        custom_ranges = {
            "learning_rate": {
                "valid_min": 0.002,
                "valid_max": 0.008
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_bayesian_float"
        job_context.handler_id = "exp_bayesian_float"

        # Initialize Bayesian
        parameters = [{"parameter": "learning_rate"}]
        bayesian = Bayesian(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters
        )

        # Define parameter config
        parameter_config = {
            "parameter": "learning_rate",
            "value_type": "float",
            "valid_min": 0.0001,
            "valid_max": 0.1,
            "default_value": 0.01
        }

        # Generate value with suggestion
        suggestion = 0.5  # Middle of 0-1 range
        value = bayesian.generate_automl_param_rec_value(parameter_config, suggestion)

        # Verify the value is within custom range
        assert isinstance(value, (float, np.floating))
        # The value should be influenced by custom ranges (0.002 to 0.008)
        assert 0.002 <= value <= 0.008

    @patch('nvidia_tao_core.microservices.automl.bayesian.get_total_epochs')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.'
           'get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_generate_param_rec_value_int_with_custom_range(
        self, mock_get_job_specs, mock_get_custom_ranges, mock_get_total_epochs
    ):
        """Test integer parameter generation with custom range in Bayesian"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        mock_get_total_epochs.return_value = 10
        custom_ranges = {
            "batch_size": {
                "valid_min": 16,
                "valid_max": 32
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_bayesian_int"
        job_context.handler_id = "exp_bayesian_int"

        # Initialize Bayesian
        parameters = [{"parameter": "batch_size"}]
        bayesian = Bayesian(
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

        # Generate value with suggestion
        suggestion = 0.5  # Middle of 0-1 range
        value = bayesian.generate_automl_param_rec_value(parameter_config, suggestion)

        # Verify the value is within custom range
        assert isinstance(value, (int, np.integer))
        assert 16 <= value <= 32

    @patch('nvidia_tao_core.microservices.automl.bayesian.get_total_epochs')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.'
           'get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_generate_param_rec_value_categorical_with_custom_options(
        self, mock_get_job_specs, mock_get_custom_ranges, mock_get_total_epochs
    ):
        """Test categorical parameter generation with custom options in Bayesian"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        mock_get_total_epochs.return_value = 10
        custom_ranges = {
            "optimizer": {
                "valid_options": ["adam", "sgd"]
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_bayesian_cat"
        job_context.handler_id = "exp_bayesian_cat"

        # Initialize Bayesian
        parameters = [{"parameter": "optimizer"}]
        bayesian = Bayesian(
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

        # Generate value with suggestion (should select from custom options)
        suggestion = 0.3
        value = bayesian.generate_automl_param_rec_value(parameter_config, suggestion)

        # Verify the value is from custom options
        assert value in ["adam", "sgd"]

    @patch('nvidia_tao_core.microservices.automl.bayesian.get_total_epochs')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.'
           'get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_multiple_parameters_with_different_custom_ranges(
        self, mock_get_job_specs, mock_get_custom_ranges, mock_get_total_epochs
    ):
        """Test multiple parameters with different custom ranges"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        mock_get_total_epochs.return_value = 10
        custom_ranges = {
            "learning_rate": {
                "valid_min": 0.001,
                "valid_max": 0.01
            },
            "batch_size": {
                "valid_min": 16,
                "valid_max": 32
            },
            "optimizer": {
                "valid_options": ["adam"]
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_bayesian_multi"
        job_context.handler_id = "exp_bayesian_multi"

        # Initialize Bayesian
        parameters = [
            {"parameter": "learning_rate"},
            {"parameter": "batch_size"},
            {"parameter": "optimizer"}
        ]
        bayesian = Bayesian(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters
        )

        # Test learning rate
        lr_config = {
            "parameter": "learning_rate",
            "value_type": "float",
            "valid_min": 0.0001,
            "valid_max": 0.1,
            "default_value": 0.01
        }
        lr_value = bayesian.generate_automl_param_rec_value(lr_config, 0.5)
        assert 0.001 <= lr_value <= 0.01

        # Test batch size
        batch_config = {
            "parameter": "batch_size",
            "value_type": "int",
            "valid_min": 8,
            "valid_max": 128,
            "default_value": 32
        }
        batch_value = bayesian.generate_automl_param_rec_value(batch_config, 0.5)
        assert 16 <= batch_value <= 32

        # Test optimizer
        opt_config = {
            "parameter": "optimizer",
            "value_type": "categorical",
            "valid_options": ["adam", "sgd", "adamw"],
            "default_value": "adam"
        }
        opt_value = bayesian.generate_automl_param_rec_value(opt_config, 0.5)
        assert opt_value == "adam"

    @patch('nvidia_tao_core.microservices.automl.bayesian.get_total_epochs')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.'
           'get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_custom_override_applied_before_processing(
        self, mock_get_job_specs, mock_get_custom_ranges, mock_get_total_epochs
    ):
        """Test that custom overrides are applied to parameter_config before processing"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        mock_get_total_epochs.return_value = 10
        custom_ranges = {
            "dropout": {
                "valid_min": 0.1,
                "valid_max": 0.3
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_bayesian_override"
        job_context.handler_id = "exp_bayesian_override"

        # Initialize Bayesian
        parameters = [{"parameter": "dropout"}]
        bayesian = Bayesian(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters
        )

        # Define parameter config with original values
        parameter_config = {
            "parameter": "dropout",
            "value_type": "float",
            "valid_min": 0.0,
            "valid_max": 0.5,
            "default_value": 0.2
        }

        # Generate value
        value = bayesian.generate_automl_param_rec_value(parameter_config, 0.5)

        # Verify custom ranges were applied
        # The parameter_config should have been modified with custom ranges
        assert 0.1 <= value <= 0.3

    @patch('nvidia_tao_core.microservices.automl.bayesian.get_total_epochs')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.'
           'get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_no_custom_ranges_uses_schema_defaults(
        self, mock_get_job_specs, mock_get_custom_ranges, mock_get_total_epochs
    ):
        """Test that schema defaults are used when no custom ranges provided"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        mock_get_total_epochs.return_value = 10
        mock_get_custom_ranges.return_value = {}

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_bayesian_no_custom"
        job_context.handler_id = "exp_bayesian_no_custom"

        # Initialize Bayesian
        parameters = [{"parameter": "learning_rate"}]
        bayesian = Bayesian(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters
        )

        # Define parameter config
        parameter_config = {
            "parameter": "learning_rate",
            "value_type": "float",
            "valid_min": 0.0001,
            "valid_max": 0.1,
            "default_value": 0.01
        }

        # Generate value
        value = bayesian.generate_automl_param_rec_value(parameter_config, 0.5)

        # Verify schema ranges are used
        assert 0.0001 <= value <= 0.1


class TestBayesianGetValidRangeIntegration:
    """Test that Bayesian correctly uses get_valid_range with custom_ranges"""

    @patch('nvidia_tao_core.microservices.automl.bayesian.get_total_epochs')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.'
           'get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_get_valid_range_called_with_custom_ranges(
        self, mock_get_job_specs, mock_get_custom_ranges, mock_get_total_epochs
    ):
        """Test that get_valid_range is called with custom_ranges parameter"""
        # Mock the dependencies
        mock_get_job_specs.return_value = {}
        mock_get_total_epochs.return_value = 10
        custom_ranges = {
            "learning_rate": {
                "valid_min": 0.005,
                "valid_max": 0.05
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_bayesian_gvr"
        job_context.handler_id = "exp_bayesian_gvr"

        # Initialize Bayesian
        parameters = [{"parameter": "learning_rate"}]
        bayesian = Bayesian(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters
        )

        # Define parameter config
        parameter_config = {
            "parameter": "learning_rate",
            "value_type": "float",
            "valid_min": 0.0001,
            "valid_max": 0.1,
            "default_value": 0.01
        }

        # Patch get_valid_range to verify it's called with custom_ranges
        with patch('nvidia_tao_core.microservices.automl.bayesian.get_valid_range') as mock_gvr:
            mock_gvr.return_value = (0.005, 0.05)

            # Generate value
            bayesian.generate_automl_param_rec_value(parameter_config, 0.5)

            # Verify get_valid_range was called with custom_ranges
            mock_gvr.assert_called_once()
            call_args = mock_gvr.call_args
            # Third argument should be custom_ranges
            assert call_args[0][2] == custom_ranges
