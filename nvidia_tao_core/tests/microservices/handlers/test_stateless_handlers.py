# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for handlers/stateless_handlers.py custom parameter ranges feature"""

from unittest.mock import Mock, patch

from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    get_automl_custom_param_ranges,
    save_automl_custom_param_ranges
)


class TestGetAutoMLCustomParamRanges:
    """Test get_automl_custom_param_ranges function"""

    @patch('nvidia_tao_core.microservices.utils.stateless_handler_utils.MongoHandler')
    def test_get_custom_param_ranges_exists(self, mock_mongo_handler):
        """Test getting custom parameter ranges when they exist"""
        # Mock the MongoHandler
        mock_handler_instance = Mock()
        mock_mongo_handler.return_value = mock_handler_instance

        # Mock experiment data with custom_param_ranges
        experiment_data = {
            "id": "exp123",
            "custom_param_ranges": {
                "learning_rate": {
                    "valid_min": 0.001,
                    "valid_max": 0.01
                },
                "batch_size": {
                    "valid_min": 16,
                    "valid_max": 64
                }
            }
        }
        mock_handler_instance.find_one.return_value = experiment_data

        # Call the function
        result = get_automl_custom_param_ranges("exp123")

        # Verify the result
        assert result == {
            "learning_rate": {
                "valid_min": 0.001,
                "valid_max": 0.01
            },
            "batch_size": {
                "valid_min": 16,
                "valid_max": 64
            }
        }

        # Verify MongoHandler was called correctly
        mock_mongo_handler.assert_called_once_with("tao", "experiments")
        mock_handler_instance.find_one.assert_called_once_with({'id': 'exp123'})

    @patch('nvidia_tao_core.microservices.utils.stateless_handler_utils.MongoHandler')
    def test_get_custom_param_ranges_empty(self, mock_mongo_handler):
        """Test getting custom parameter ranges when they don't exist"""
        # Mock the MongoHandler
        mock_handler_instance = Mock()
        mock_mongo_handler.return_value = mock_handler_instance

        # Mock experiment data without custom_param_ranges
        experiment_data = {
            "id": "exp123",
            "name": "My Experiment"
        }
        mock_handler_instance.find_one.return_value = experiment_data

        # Call the function
        result = get_automl_custom_param_ranges("exp123")

        # Verify the result is empty dict
        assert result == {}

        # Verify MongoHandler was called correctly
        mock_mongo_handler.assert_called_once_with("tao", "experiments")
        mock_handler_instance.find_one.assert_called_once_with({'id': 'exp123'})

    @patch('nvidia_tao_core.microservices.utils.stateless_handler_utils.MongoHandler')
    def test_get_custom_param_ranges_experiment_not_found(self, mock_mongo_handler):
        """Test getting custom parameter ranges when experiment doesn't exist"""
        # Mock the MongoHandler
        mock_handler_instance = Mock()
        mock_mongo_handler.return_value = mock_handler_instance

        # Mock no experiment found
        mock_handler_instance.find_one.return_value = None

        # Call the function
        result = get_automl_custom_param_ranges("nonexistent_exp")

        # Verify the result is empty dict
        assert result == {}

        # Verify MongoHandler was called correctly
        mock_mongo_handler.assert_called_once_with("tao", "experiments")
        mock_handler_instance.find_one.assert_called_once_with({'id': 'nonexistent_exp'})

    @patch('nvidia_tao_core.microservices.utils.stateless_handler_utils.MongoHandler')
    def test_get_custom_param_ranges_with_various_types(self, mock_mongo_handler):
        """Test getting custom parameter ranges with various data types"""
        # Mock the MongoHandler
        mock_handler_instance = Mock()
        mock_mongo_handler.return_value = mock_handler_instance

        # Mock experiment data with different types of custom ranges
        experiment_data = {
            "id": "exp456",
            "custom_param_ranges": {
                "learning_rate": {
                    "valid_min": 0.0001,
                    "valid_max": 0.1
                },
                "optimizer": {
                    "valid_options": ["adam", "sgd"]
                },
                "dropout": {
                    "valid_min": 0.0,
                    "valid_max": 0.5,
                    "depends_on": "> min_dropout"
                },
                "optimizer.betas": {
                    "valid_min": [0.85, 0.92],
                    "valid_max": [0.93, 0.998]
                }
            }
        }
        mock_handler_instance.find_one.return_value = experiment_data

        # Call the function
        result = get_automl_custom_param_ranges("exp456")

        # Verify the result
        assert result == experiment_data["custom_param_ranges"]
        assert "learning_rate" in result
        assert "optimizer" in result
        assert "dropout" in result
        assert "optimizer.betas" in result
        assert result["optimizer"]["valid_options"] == ["adam", "sgd"]
        assert result["dropout"]["depends_on"] == "> min_dropout"


class TestSaveAutoMLCustomParamRanges:
    """Test save_automl_custom_param_ranges function"""

    @patch('nvidia_tao_core.microservices.utils.stateless_handler_utils.MongoHandler')
    def test_save_custom_param_ranges(self, mock_mongo_handler):
        """Test saving custom parameter ranges"""
        # Mock the MongoHandler
        mock_handler_instance = Mock()
        mock_mongo_handler.return_value = mock_handler_instance

        # Custom ranges to save
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

        # Call the function
        save_automl_custom_param_ranges("exp789", custom_ranges)

        # Verify MongoHandler was called correctly
        mock_mongo_handler.assert_called_once_with("tao", "experiments")
        mock_handler_instance.upsert.assert_called_once_with(
            {'id': 'exp789'},
            {"custom_param_ranges": custom_ranges}
        )

    @patch('nvidia_tao_core.microservices.utils.stateless_handler_utils.MongoHandler')
    def test_save_empty_custom_param_ranges(self, mock_mongo_handler):
        """Test saving empty custom parameter ranges"""
        # Mock the MongoHandler
        mock_handler_instance = Mock()
        mock_mongo_handler.return_value = mock_handler_instance

        # Call the function with empty dict
        save_automl_custom_param_ranges("exp999", {})

        # Verify MongoHandler was called correctly
        mock_mongo_handler.assert_called_once_with("tao", "experiments")
        mock_handler_instance.upsert.assert_called_once_with(
            {'id': 'exp999'},
            {"custom_param_ranges": {}}
        )

    @patch('nvidia_tao_core.microservices.utils.stateless_handler_utils.MongoHandler')
    def test_save_complex_custom_param_ranges(self, mock_mongo_handler):
        """Test saving complex custom parameter ranges with various types"""
        # Mock the MongoHandler
        mock_handler_instance = Mock()
        mock_mongo_handler.return_value = mock_handler_instance

        # Complex custom ranges
        custom_ranges = {
            "learning_rate": {
                "valid_min": 0.0001,
                "valid_max": 0.1
            },
            "optimizer": {
                "valid_options": ["adam", "sgd", "adamw"]
            },
            "dropout": {
                "valid_min": 0.0,
                "valid_max": 0.5,
                "depends_on": "> min_dropout"
            },
            "num_layers": {
                "valid_options": [2, 3, 4, 5]
            },
            "optimizer.betas": {
                "valid_min": [0.85, 0.92],
                "valid_max": [0.93, 0.998]
            }
        }

        # Call the function
        save_automl_custom_param_ranges("exp_complex", custom_ranges)

        # Verify MongoHandler was called correctly
        mock_mongo_handler.assert_called_once_with("tao", "experiments")
        mock_handler_instance.upsert.assert_called_once_with(
            {'id': 'exp_complex'},
            {"custom_param_ranges": custom_ranges}
        )

    @patch('nvidia_tao_core.microservices.utils.stateless_handler_utils.MongoHandler')
    def test_save_and_get_roundtrip(self, mock_mongo_handler):
        """Test saving and then getting custom parameter ranges"""
        # Mock the MongoHandler
        mock_handler_instance = Mock()
        mock_mongo_handler.return_value = mock_handler_instance

        custom_ranges = {
            "learning_rate": {
                "valid_min": 0.005,
                "valid_max": 0.05
            }
        }

        # Save the ranges
        save_automl_custom_param_ranges("exp_roundtrip", custom_ranges)

        # Verify save was called
        mock_handler_instance.upsert.assert_called_once_with(
            {'id': 'exp_roundtrip'},
            {"custom_param_ranges": custom_ranges}
        )

        # Mock the get operation
        mock_handler_instance.find_one.return_value = {
            "id": "exp_roundtrip",
            "custom_param_ranges": custom_ranges
        }

        # Get the ranges
        result = get_automl_custom_param_ranges("exp_roundtrip")

        # Verify the result matches what we saved
        assert result == custom_ranges
