# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for automl/hyperband.py custom parameter ranges feature"""

import numpy as np
from unittest.mock import Mock, patch

from nvidia_tao_core.microservices.automl.hyperband import HyperBand


class TestHyperBandCustomRanges:
    """Test HyperBand algorithm with custom parameter ranges"""

    @patch('nvidia_tao_core.microservices.automl.hyperband.save_job_specs')
    @patch('nvidia_tao_core.microservices.automl.hyperband.get_job_specs')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_initialization_with_custom_ranges(
        self, mock_get_job_specs_base, mock_get_custom_ranges,
        mock_get_job_specs_hb, mock_save_job_specs
    ):
        """Test HyperBand initialization with custom parameter ranges"""
        # Mock the dependencies
        mock_get_job_specs_base.return_value = {}
        mock_get_job_specs_hb.return_value = {"training_config": {"num_epochs": 10}}
        mock_save_job_specs.return_value = None
        custom_ranges = {
            "learning_rate": {
                "valid_min": 0.001,
                "valid_max": 0.01
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_hyperband_test"
        job_context.handler_id = "exp_hyperband"

        # Initialize HyperBand
        parameters = [{"parameter": "learning_rate"}]
        hyperband = HyperBand(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters,
            max_epochs=81,
            reduction_factor=3,
            epoch_multiplier=1
        )

        # Verify custom ranges were loaded
        assert hyperband.custom_ranges == custom_ranges

    @patch('nvidia_tao_core.microservices.automl.hyperband.save_job_specs')
    @patch('nvidia_tao_core.microservices.automl.hyperband.get_job_specs')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_generate_param_rec_value_float_with_custom_range(
        self, mock_get_job_specs_base, mock_get_custom_ranges,
        mock_get_job_specs_hb, mock_save_job_specs
    ):
        """Test float parameter generation with custom range in HyperBand"""
        # Mock the dependencies
        mock_get_job_specs_base.return_value = {}
        mock_get_job_specs_hb.return_value = {"training_config": {"num_epochs": 10}}
        mock_save_job_specs.return_value = None
        custom_ranges = {
            "learning_rate": {
                "valid_min": 0.002,
                "valid_max": 0.008
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_hyperband_float"
        job_context.handler_id = "exp_hyperband_float"

        # Initialize HyperBand
        parameters = [{"parameter": "learning_rate"}]
        hyperband = HyperBand(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters,
            max_epochs=81,
            reduction_factor=3,
            epoch_multiplier=1
        )

        # Define parameter config
        parameter_config = {
            "parameter": "learning_rate",
            "value_type": "float",
            "valid_min": 0.0001,
            "valid_max": 0.1,
            "default_value": 0.01
        }

        # Generate value multiple times to check range
        for _ in range(10):
            value = hyperband.generate_automl_param_rec_value(parameter_config)
            # Verify the value is within custom range
            assert isinstance(value, (float, np.floating))
            assert 0.002 <= value <= 0.008

    @patch('nvidia_tao_core.microservices.automl.hyperband.save_job_specs')
    @patch('nvidia_tao_core.microservices.automl.hyperband.get_job_specs')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_generate_param_rec_value_int_with_custom_range(
        self, mock_get_job_specs_base, mock_get_custom_ranges,
        mock_get_job_specs_hb, mock_save_job_specs
    ):
        """Test integer parameter generation with custom range in HyperBand"""
        # Mock the dependencies
        mock_get_job_specs_base.return_value = {}
        mock_get_job_specs_hb.return_value = {"training_config": {"num_epochs": 10}}
        mock_save_job_specs.return_value = None
        custom_ranges = {
            "batch_size": {
                "valid_min": 16,
                "valid_max": 32
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_hyperband_int"
        job_context.handler_id = "exp_hyperband_int"

        # Initialize HyperBand
        parameters = [{"parameter": "batch_size"}]
        hyperband = HyperBand(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters,
            max_epochs=81,
            reduction_factor=3,
            epoch_multiplier=1
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
            value = hyperband.generate_automl_param_rec_value(parameter_config)
            # Verify the value is within custom range
            assert isinstance(value, (int, np.integer))
            assert 16 <= value <= 32

    @patch('nvidia_tao_core.microservices.automl.hyperband.save_job_specs')
    @patch('nvidia_tao_core.microservices.automl.hyperband.get_job_specs')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_generate_param_rec_value_categorical_with_custom_options(
        self, mock_get_job_specs_base, mock_get_custom_ranges,
        mock_get_job_specs_hb, mock_save_job_specs
    ):
        """Test categorical parameter generation with custom options in HyperBand"""
        # Mock the dependencies
        mock_get_job_specs_base.return_value = {}
        mock_get_job_specs_hb.return_value = {"training_config": {"num_epochs": 10}}
        mock_save_job_specs.return_value = None
        custom_ranges = {
            "optimizer": {
                "valid_options": ["adam", "sgd"]
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_hyperband_cat"
        job_context.handler_id = "exp_hyperband_cat"

        # Initialize HyperBand
        parameters = [{"parameter": "optimizer"}]
        hyperband = HyperBand(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters,
            max_epochs=81,
            reduction_factor=3,
            epoch_multiplier=1
        )

        # Define parameter config
        parameter_config = {
            "parameter": "optimizer",
            "value_type": "categorical",
            "valid_options": ["adam", "sgd", "adamw", "rmsprop"],
            "default_value": "adam"
        }

        # Generate value multiple times
        generated_values = set()
        for _ in range(20):
            value = hyperband.generate_automl_param_rec_value(parameter_config)
            generated_values.add(value)
            # Verify the value is from custom options
            assert value in ["adam", "sgd"]

    @patch('nvidia_tao_core.microservices.automl.hyperband.save_job_specs')
    @patch('nvidia_tao_core.microservices.automl.hyperband.get_job_specs')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_multiple_parameters_with_different_custom_ranges(
        self, mock_get_job_specs_base, mock_get_custom_ranges,
        mock_get_job_specs_hb, mock_save_job_specs
    ):
        """Test multiple parameters with different custom ranges in HyperBand"""
        # Mock the dependencies
        mock_get_job_specs_base.return_value = {}
        mock_get_job_specs_hb.return_value = {"training_config": {"num_epochs": 10}}
        mock_save_job_specs.return_value = None
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
        job_context.id = "job_hyperband_multi"
        job_context.handler_id = "exp_hyperband_multi"

        # Initialize HyperBand
        parameters = [
            {"parameter": "learning_rate"},
            {"parameter": "batch_size"},
            {"parameter": "optimizer"}
        ]
        hyperband = HyperBand(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters,
            max_epochs=81,
            reduction_factor=3,
            epoch_multiplier=1
        )

        # Test learning rate
        lr_config = {
            "parameter": "learning_rate",
            "value_type": "float",
            "valid_min": 0.0001,
            "valid_max": 0.1,
            "default_value": 0.01
        }
        lr_value = hyperband.generate_automl_param_rec_value(lr_config)
        assert 0.001 <= lr_value <= 0.01

        # Test batch size
        batch_config = {
            "parameter": "batch_size",
            "value_type": "int",
            "valid_min": 8,
            "valid_max": 128,
            "default_value": 32
        }
        batch_value = hyperband.generate_automl_param_rec_value(batch_config)
        assert 16 <= batch_value <= 32

        # Test optimizer
        opt_config = {
            "parameter": "optimizer",
            "value_type": "categorical",
            "valid_options": ["adam", "sgd", "adamw"],
            "default_value": "adam"
        }
        opt_value = hyperband.generate_automl_param_rec_value(opt_config)
        assert opt_value == "adam"

    @patch('nvidia_tao_core.microservices.automl.hyperband.save_job_specs')
    @patch('nvidia_tao_core.microservices.automl.hyperband.get_job_specs')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_custom_override_applied_before_processing(
        self, mock_get_job_specs_base, mock_get_custom_ranges,
        mock_get_job_specs_hb, mock_save_job_specs
    ):
        """Test that custom overrides are applied to parameter_config before processing"""
        # Mock the dependencies
        mock_get_job_specs_base.return_value = {}
        mock_get_job_specs_hb.return_value = {"training_config": {"num_epochs": 10}}
        mock_save_job_specs.return_value = None
        custom_ranges = {
            "dropout": {
                "valid_min": 0.1,
                "valid_max": 0.3
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_hyperband_override"
        job_context.handler_id = "exp_hyperband_override"

        # Initialize HyperBand
        parameters = [{"parameter": "dropout"}]
        hyperband = HyperBand(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters,
            max_epochs=81,
            reduction_factor=3,
            epoch_multiplier=1
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
        value = hyperband.generate_automl_param_rec_value(parameter_config)

        # Verify custom ranges were applied
        assert 0.1 <= value <= 0.3

    @patch('nvidia_tao_core.microservices.automl.hyperband.save_job_specs')
    @patch('nvidia_tao_core.microservices.automl.hyperband.get_job_specs')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_no_custom_ranges_uses_schema_defaults(
        self, mock_get_job_specs_base, mock_get_custom_ranges,
        mock_get_job_specs_hb, mock_save_job_specs
    ):
        """Test that schema defaults are used when no custom ranges provided"""
        # Mock the dependencies
        mock_get_job_specs_base.return_value = {}
        mock_get_job_specs_hb.return_value = {"training_config": {"num_epochs": 10}}
        mock_save_job_specs.return_value = None
        mock_get_custom_ranges.return_value = {}

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_hyperband_no_custom"
        job_context.handler_id = "exp_hyperband_no_custom"

        # Initialize HyperBand
        parameters = [{"parameter": "learning_rate"}]
        hyperband = HyperBand(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters,
            max_epochs=81,
            reduction_factor=3,
            epoch_multiplier=1
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
        value = hyperband.generate_automl_param_rec_value(parameter_config)

        # Verify schema ranges are used
        assert 0.0001 <= value <= 0.1


class TestHyperBandGetValidRangeIntegration:
    """Test that HyperBand correctly uses get_valid_range with custom_ranges"""

    @patch('nvidia_tao_core.microservices.automl.hyperband.save_job_specs')
    @patch('nvidia_tao_core.microservices.automl.hyperband.get_job_specs')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_get_valid_range_called_with_custom_ranges(
        self, mock_get_job_specs_base, mock_get_custom_ranges,
        mock_get_job_specs_hb, mock_save_job_specs
    ):
        """Test that get_valid_range is called with custom_ranges parameter"""
        # Mock the dependencies
        mock_get_job_specs_base.return_value = {}
        mock_get_job_specs_hb.return_value = {"training_config": {"num_epochs": 10}}
        mock_save_job_specs.return_value = None
        custom_ranges = {
            "learning_rate": {
                "valid_min": 0.005,
                "valid_max": 0.05
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_hyperband_gvr"
        job_context.handler_id = "exp_hyperband_gvr"

        # Initialize HyperBand
        parameters = [{"parameter": "learning_rate"}]
        hyperband = HyperBand(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters,
            max_epochs=81,
            reduction_factor=3,
            epoch_multiplier=1
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
        with patch('nvidia_tao_core.microservices.automl.hyperband.get_valid_range') as mock_gvr:
            mock_gvr.return_value = (0.005, 0.05)

            # Generate value
            hyperband.generate_automl_param_rec_value(parameter_config)

            # Verify get_valid_range was called with custom_ranges
            mock_gvr.assert_called_once()
            call_args = mock_gvr.call_args
            assert call_args[0][2] == custom_ranges  # Third argument should be custom_ranges

    @patch('nvidia_tao_core.microservices.automl.hyperband.save_job_specs')
    @patch('nvidia_tao_core.microservices.automl.hyperband.get_job_specs')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges')
    @patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs')
    def test_ordered_int_parameter_with_custom_options(
        self, mock_get_job_specs_base, mock_get_custom_ranges,
        mock_get_job_specs_hb, mock_save_job_specs
    ):
        """Test ordered_int parameter with custom options in HyperBand"""
        # Mock the dependencies
        mock_get_job_specs_base.return_value = {}
        mock_get_job_specs_hb.return_value = {"training_config": {"num_epochs": 10}}
        mock_save_job_specs.return_value = None
        custom_ranges = {
            "num_layers": {
                "valid_options": [2, 3]
            }
        }
        mock_get_custom_ranges.return_value = custom_ranges

        # Create mock job context
        job_context = Mock()
        job_context.id = "job_hyperband_ordered"
        job_context.handler_id = "exp_hyperband_ordered"

        # Initialize HyperBand
        parameters = [{"parameter": "num_layers"}]
        hyperband = HyperBand(
            job_context=job_context,
            root="/path/to/root/subdir",
            network="image_classification",
            parameters=parameters,
            max_epochs=81,
            reduction_factor=3,
            epoch_multiplier=1
        )

        # Define parameter config
        parameter_config = {
            "parameter": "num_layers",
            "value_type": "ordered_int",
            "valid_options": [1, 2, 3, 4, 5],
            "default_value": 3
        }

        # Generate value multiple times
        for _ in range(10):
            value = hyperband.generate_automl_param_rec_value(parameter_config)
            # Should only be from custom options
            assert value in [2, 3]
