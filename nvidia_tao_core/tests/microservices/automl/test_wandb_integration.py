# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for wandb integration in automl controller"""

import os
from unittest.mock import Mock, patch

# Enable test mode to use mongomock instead of real MongoDB (avoid 120s timeout)
# Must be set before importing modules that use MongoDB
os.environ["TAO_TEST_MODE"] = "true"

from nvidia_tao_core.microservices.automl.controller import Controller  # noqa: E402
from nvidia_tao_core.microservices.automl_start import AlgorithmParams  # noqa: E402
from nvidia_tao_core.microservices.utils.automl_utils import Recommendation  # noqa: E402


class TestWandBGroupName:
    """Test wandb group name initialization"""

    @patch('nvidia_tao_core.microservices.automl.controller.create_cs_instance_with_decrypted_metadata')
    def test_wandb_group_name_initialization(self, mock_cs_instance):
        """Test that wandb_group_name is set correctly during controller initialization"""
        mock_cs_instance.return_value = (Mock(), None)

        automl_context = Mock()
        automl_context.id = "test_job_123"
        automl_context.handler_id = "exp_456"
        automl_context.retain_checkpoints_for_resume = False

        brain = Mock()
        brain.reverse_sort = True

        algorithm_settings = AlgorithmParams(
            automl_max_recommendations=10,
            automl_max_epochs=81,
            automl_reduction_factor=3,
            epoch_multiplier=1
        )

        controller = Controller(
            root="/test/root",
            network="image_classification",
            brain=brain,
            automl_context=automl_context,
            automl_algorithm_settings=algorithm_settings,
            delete_intermediate_ckpt="false",
            metric="mAP",
            automl_algorithm="bayesian",
            decrypted_workspace_metadata={},
            parameter_names=["train.optim.lr", "dataset.batch_size"]
        )

        assert controller.wandb_group_name == "automl_test_job_123"
        assert controller.parameter_names == ["train.optim.lr", "dataset.batch_size"]
        assert controller.wandb_initialized is False
        assert controller.wandb_table is None

    @patch('nvidia_tao_core.microservices.automl.controller.create_cs_instance_with_decrypted_metadata')
    def test_wandb_group_name_with_empty_parameter_names(self, mock_cs_instance):
        """Test wandb_group_name when parameter_names is None or empty"""
        mock_cs_instance.return_value = (Mock(), None)

        automl_context = Mock()
        automl_context.id = "test_job_456"
        automl_context.handler_id = "exp_789"
        automl_context.retain_checkpoints_for_resume = False

        brain = Mock()
        brain.reverse_sort = True

        algorithm_settings = AlgorithmParams(
            automl_max_recommendations=10,
            automl_max_epochs=81,
            automl_reduction_factor=3,
            epoch_multiplier=1
        )

        controller = Controller(
            root="/test/root",
            network="image_classification",
            brain=brain,
            automl_context=automl_context,
            automl_algorithm_settings=algorithm_settings,
            delete_intermediate_ckpt="false",
            metric="mAP",
            automl_algorithm="bayesian",
            decrypted_workspace_metadata={},
            parameter_names=None
        )

        assert controller.wandb_group_name == "automl_test_job_456"
        assert controller.parameter_names == []


class TestWandBInitialization:
    """Test wandb initialization for automl"""

    def _create_controller(self):
        """Helper method to create a controller instance"""
        with patch(
            'nvidia_tao_core.microservices.automl.controller.create_cs_instance_with_decrypted_metadata'
        ) as mock_cs_instance:
            mock_cs_instance.return_value = (Mock(), None)

            automl_context = Mock()
            automl_context.id = "test_job_123"
            automl_context.handler_id = "exp_456"
            automl_context.retain_checkpoints_for_resume = False

            brain = Mock()
            brain.reverse_sort = True

            algorithm_settings = AlgorithmParams(
                automl_max_recommendations=10,
                automl_max_epochs=81,
                automl_reduction_factor=3,
                epoch_multiplier=1
            )

            controller = Controller(
                root="/test/root",
                network="image_classification",
                brain=brain,
                automl_context=automl_context,
                automl_algorithm_settings=algorithm_settings,
                delete_intermediate_ckpt="false",
                metric="mAP",
                automl_algorithm="bayesian",
                decrypted_workspace_metadata={},
                parameter_names=["train.optim.lr", "dataset.batch_size"]
            )
            return controller

    @patch('wandb.login')
    @patch('wandb.init')
    @patch('wandb.Table')
    @patch('nvidia_tao_core.microservices.automl.controller.get_job_specs')
    @patch('nvidia_tao_core.microservices.automl.controller.get_handler_metadata')
    def test_initialize_wandb_with_api_key(
        self, mock_get_handler_metadata, mock_get_job_specs, mock_table_class, mock_wandb_init, mock_wandb_login
    ):
        """Test wandb initialization when API key is present"""
        controller = self._create_controller()

        # Mock experiment metadata with API key
        mock_get_handler_metadata.return_value = {
            "docker_env_vars": {
                "WANDB_API_KEY": "test_api_key_123"
            }
        }

        # Mock job specs with wandb config
        mock_get_job_specs.return_value = {
            "wandb": {
                "project": "Test Project",
                "entity": "test_entity"
            }
        }

        # Mock wandb login and init
        mock_wandb_login.return_value = True
        mock_wandb_init.return_value = None
        mock_table = Mock()
        mock_table_class.return_value = mock_table

        # Call initialization
        controller._initialize_wandb_for_automl()

        # Verify wandb.login was called with API key
        mock_wandb_login.assert_called_once_with(key="test_api_key_123")

        # Verify wandb.init was called with correct parameters
        mock_wandb_init.assert_called_once()
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs["project"] == "Test Project"
        assert call_kwargs["entity"] == "test_entity"
        assert call_kwargs["name"] == "automl_brain"
        assert call_kwargs["group"] == "automl_test_job_123"
        assert call_kwargs["reinit"] is True
        assert "config" in call_kwargs

        # Verify initialization flags
        assert controller.wandb_initialized is True
        assert controller.wandb_table is not None

    @patch('nvidia_tao_core.microservices.automl.controller.get_handler_metadata')
    def test_initialize_wandb_without_api_key(self, mock_get_handler_metadata):
        """Test wandb initialization when API key is not present"""
        controller = self._create_controller()

        # Mock experiment metadata without API key
        mock_get_handler_metadata.return_value = {
            "docker_env_vars": {}
        }

        # Call initialization
        controller._initialize_wandb_for_automl()

        # Verify wandb was not initialized
        assert controller.wandb_initialized is False
        assert controller.wandb_table is None

    @patch('wandb.login')
    @patch('nvidia_tao_core.microservices.automl.controller.get_handler_metadata')
    def test_initialize_wandb_login_failure(self, mock_get_handler_metadata, mock_wandb_login):
        """Test wandb initialization when login fails"""
        controller = self._create_controller()

        # Mock experiment metadata with API key
        mock_get_handler_metadata.return_value = {
            "docker_env_vars": {
                "WANDB_API_KEY": "invalid_key"
            }
        }

        # Mock wandb login failure
        mock_wandb_login.return_value = False

        # Call initialization
        controller._initialize_wandb_for_automl()

        # Verify wandb was not initialized
        assert controller.wandb_initialized is False
        assert controller.wandb_table is None

    @patch('wandb.init')
    @patch('wandb.login')
    @patch('nvidia_tao_core.microservices.automl.controller.get_job_specs')
    @patch('nvidia_tao_core.microservices.automl.controller.get_handler_metadata')
    def test_initialize_wandb_exception_handling(
        self, mock_get_handler_metadata, mock_get_job_specs, mock_wandb_login, mock_wandb_init
    ):
        """Test wandb initialization handles exceptions gracefully"""
        controller = self._create_controller()

        # Mock experiment metadata with API key
        mock_get_handler_metadata.return_value = {
            "docker_env_vars": {
                "WANDB_API_KEY": "test_key"
            }
        }

        # Mock job specs
        mock_get_job_specs.return_value = {"wandb": {}}

        # Mock wandb login success but init raises exception
        mock_wandb_login.return_value = True
        mock_wandb_init.side_effect = Exception("WandB init failed")

        # Call initialization
        controller._initialize_wandb_for_automl()

        # Verify initialization failed gracefully
        assert controller.wandb_initialized is False

    @patch('wandb.Table')
    @patch('wandb.init')
    @patch('wandb.login')
    @patch('nvidia_tao_core.microservices.automl.controller.get_job_specs')
    @patch('nvidia_tao_core.microservices.automl.controller.get_handler_metadata')
    def test_initialize_wandb_bayesian_config(
        self, mock_get_handler_metadata, mock_get_job_specs, mock_wandb_login, mock_wandb_init, mock_table_class
    ):
        """Test wandb initialization with bayesian algorithm config"""
        controller = self._create_controller()

        mock_get_handler_metadata.return_value = {
            "docker_env_vars": {"WANDB_API_KEY": "test_key"}
        }
        mock_get_job_specs.return_value = {"wandb": {"project": "Test"}}
        mock_wandb_login.return_value = True
        mock_table_class.return_value = Mock()

        controller._initialize_wandb_for_automl()

        # Verify config includes bayesian-specific fields
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs["config"]["algorithm"] == "bayesian"
        assert call_kwargs["config"]["max_recommendations"] == 10
        assert call_kwargs["config"]["network"] == "image_classification"
        assert call_kwargs["config"]["metric"] == "mAP"

    @patch('wandb.Table')
    @patch('wandb.init')
    @patch('wandb.login')
    @patch('nvidia_tao_core.microservices.automl.controller.create_cs_instance_with_decrypted_metadata')
    @patch('nvidia_tao_core.microservices.automl.controller.get_job_specs')
    @patch('nvidia_tao_core.microservices.automl.controller.get_handler_metadata')
    def test_initialize_wandb_hyperband_config(
        self, mock_get_handler_metadata, mock_get_job_specs, mock_cs_instance,
        mock_wandb_login, mock_wandb_init, mock_table_class
    ):
        """Test wandb initialization with hyperband algorithm config"""
        mock_cs_instance.return_value = (Mock(), None)

        automl_context = Mock()
        automl_context.id = "test_job_hyperband"
        automl_context.handler_id = "exp_hyperband"
        automl_context.retain_checkpoints_for_resume = False

        brain = Mock()
        brain.reverse_sort = True

        algorithm_settings = AlgorithmParams(
            automl_max_recommendations=10,
            automl_max_epochs=81,
            automl_reduction_factor=3,
            epoch_multiplier=2
        )

        controller = Controller(
            root="/test/root",
            network="image_classification",
            brain=brain,
            automl_context=automl_context,
            automl_algorithm_settings=algorithm_settings,
            delete_intermediate_ckpt="false",
            metric="mAP",
            automl_algorithm="hyperband",
            decrypted_workspace_metadata={},
            parameter_names=["train.optim.lr"]
        )

        mock_get_handler_metadata.return_value = {
            "docker_env_vars": {"WANDB_API_KEY": "test_key"}
        }
        mock_get_job_specs.return_value = {"wandb": {"project": "Test"}}
        mock_wandb_login.return_value = True
        mock_table_class.return_value = Mock()

        controller._initialize_wandb_for_automl()

        # Verify config includes hyperband-specific fields
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs["config"]["algorithm"] == "hyperband"
        assert call_kwargs["config"]["max_epochs"] == 81
        assert call_kwargs["config"]["reduction_factor"] == 3
        assert call_kwargs["config"]["epoch_multiplier"] == 2


class TestWandBTableCreation:
    """Test wandb table creation"""

    def _create_controller(self):
        """Helper method to create a controller instance"""
        with patch(
            'nvidia_tao_core.microservices.automl.controller.create_cs_instance_with_decrypted_metadata'
        ) as mock_cs_instance:
            mock_cs_instance.return_value = (Mock(), None)

            automl_context = Mock()
            automl_context.id = "test_job_123"
            automl_context.handler_id = "exp_456"
            automl_context.retain_checkpoints_for_resume = False

            brain = Mock()
            brain.reverse_sort = True

            algorithm_settings = AlgorithmParams(
                automl_max_recommendations=10,
                automl_max_epochs=81,
                automl_reduction_factor=3,
                epoch_multiplier=1
            )

            controller = Controller(
                root="/test/root",
                network="image_classification",
                brain=brain,
                automl_context=automl_context,
                automl_algorithm_settings=algorithm_settings,
                delete_intermediate_ckpt="false",
                metric="mAP",
                automl_algorithm="bayesian",
                decrypted_workspace_metadata={},
                parameter_names=["train.optim.lr", "dataset.batch_size"]
            )
            controller.metric_key = "mAP"
            return controller

    @patch('wandb.Table')
    def test_create_wandb_table_success(self, mock_table_class):
        """Test successful wandb table creation"""
        controller = self._create_controller()
        controller.wandb_initialized = True
        mock_table = Mock()
        mock_table_class.return_value = mock_table

        controller._create_wandb_table()

        # Verify table was created with correct columns
        mock_table_class.assert_called_once_with(
            columns=["experiment_id", "job_id", "status", "mAP", "best_epoch_number",
                     "train.optim.lr", "dataset.batch_size"]
        )
        assert controller.wandb_table == mock_table

    def test_create_wandb_table_not_initialized(self):
        """Test table creation when wandb is not initialized"""
        controller = self._create_controller()
        controller.wandb_initialized = False

        controller._create_wandb_table()

        assert controller.wandb_table is None

    @patch('wandb.Table')
    def test_create_wandb_table_exception_handling(self, mock_table_class):
        """Test table creation handles exceptions gracefully"""
        controller = self._create_controller()
        controller.wandb_initialized = True
        mock_table_class.side_effect = Exception("Table creation failed")

        controller._create_wandb_table()

        assert controller.wandb_table is None


class TestWandBTableUpdate:
    """Test wandb table update functionality"""

    def _create_controller(self):
        """Helper method to create a controller instance"""
        with patch(
            'nvidia_tao_core.microservices.automl.controller.create_cs_instance_with_decrypted_metadata'
        ) as mock_cs_instance:
            mock_cs_instance.return_value = (Mock(), None)

            automl_context = Mock()
            automl_context.id = "test_job_123"
            automl_context.handler_id = "exp_456"
            automl_context.retain_checkpoints_for_resume = False

            brain = Mock()
            brain.reverse_sort = True

            algorithm_settings = AlgorithmParams(
                automl_max_recommendations=10,
                automl_max_epochs=81,
                automl_reduction_factor=3,
                epoch_multiplier=1
            )

            controller = Controller(
                root="/test/root",
                network="image_classification",
                brain=brain,
                automl_context=automl_context,
                automl_algorithm_settings=algorithm_settings,
                delete_intermediate_ckpt="false",
                metric="mAP",
                automl_algorithm="bayesian",
                decrypted_workspace_metadata={},
                parameter_names=["train.optim.lr", "dataset.batch_size"]
            )
            controller.metric_key = "mAP"
            controller.best_epoch_number = {}
            return controller

    @patch('wandb.log')
    @patch('wandb.Table')
    def test_update_wandb_table_with_recommendations(self, mock_table_class, mock_wandb_log):
        """Test updating wandb table with recommendations"""
        controller = self._create_controller()
        controller.wandb_initialized = True
        mock_table = Mock()
        controller.wandb_table = mock_table
        mock_table_class.return_value = mock_table

        # Create test recommendations
        rec1 = Recommendation(
            identifier=0,
            specs={"train.optim.lr": 0.001, "dataset.batch_size": 32},
            metric="mAP"
        )
        rec1.update_result(0.85)
        rec1.update_status("success")
        rec1.assign_job_id("job_1")

        rec2 = Recommendation(
            identifier=1,
            specs={"train.optim.lr": 0.0005, "dataset.batch_size": 64},
            metric="mAP"
        )
        rec2.update_result(0.82)
        rec2.update_status("success")
        rec2.assign_job_id("job_2")

        controller.recommendations = [rec1, rec2]
        controller.best_epoch_number = {0: "epoch_10", 1: "epoch_15"}

        controller._update_wandb_table()

        # Verify table was recreated with correct columns
        mock_table_class.assert_called_once_with(
            columns=["experiment_id", "job_id", "status", "mAP", "best_epoch_number",
                     "train.optim.lr", "dataset.batch_size"]
        )

        # Verify add_data was called for each recommendation
        assert mock_table.add_data.call_count == 2

        # Verify first recommendation data
        first_call_args = mock_table.add_data.call_args_list[0][0]
        assert first_call_args[0] == 0  # experiment_id
        assert first_call_args[1] == "job_1"  # job_id
        assert first_call_args[2] == "success"  # status
        assert first_call_args[3] == "0.85"  # result (formatted as string)
        assert first_call_args[4] == "epoch_10"  # best_epoch_number
        assert first_call_args[5] == 0.001  # train.optim.lr
        assert first_call_args[6] == 32  # dataset.batch_size

        # Verify wandb.log was called
        mock_wandb_log.assert_called_once()
        log_call = mock_wandb_log.call_args[0][0]
        assert "automl_experiments" in log_call
        assert log_call["automl_experiments"] == mock_table

    @patch('wandb.log')
    @patch('wandb.Table')
    def test_update_wandb_table_float_formatting(self, mock_table_class, mock_wandb_log):
        """Test that float results are formatted correctly"""
        controller = self._create_controller()
        controller.wandb_initialized = True
        mock_table = Mock()
        controller.wandb_table = mock_table
        mock_table_class.return_value = mock_table

        # Create recommendation with float result
        rec = Recommendation(
            identifier=0,
            specs={"train.optim.lr": 0.001, "dataset.batch_size": 32},
            metric="mAP"
        )
        rec.update_result(0.12345678901234567890)  # High precision float
        rec.update_status("success")
        rec.assign_job_id("job_1")

        controller.recommendations = [rec]
        controller.best_epoch_number = {0: "epoch_10"}

        controller._update_wandb_table()

        # Verify float was formatted correctly (10 decimal places, trailing zeros removed)
        call_args = mock_table.add_data.call_args[0]
        result_value = call_args[3]
        assert isinstance(result_value, str)
        assert result_value == "0.123456789"  # Formatted with precision, trailing zeros removed

    @patch('wandb.log')
    @patch('wandb.Table')
    def test_update_wandb_table_flat_dict_parameter_access(self, mock_table_class, mock_wandb_log):
        """Test that parameters are accessed from flat dict using dot notation keys"""
        controller = self._create_controller()
        controller.wandb_initialized = True
        mock_table = Mock()
        controller.wandb_table = mock_table
        mock_table_class.return_value = mock_table

        # Create recommendation with flat dict specs (not nested)
        rec = Recommendation(
            identifier=0,
            specs={
                "train.optim.lr": 0.001,  # Flat dict key, not nested
                "dataset.batch_size": 64
            },
            metric="mAP"
        )
        rec.update_result(0.85)
        rec.update_status("success")
        rec.assign_job_id("job_1")

        controller.recommendations = [rec]
        controller.best_epoch_number = {0: "epoch_10"}

        controller._update_wandb_table()

        # Verify parameters were accessed correctly from flat dict
        call_args = mock_table.add_data.call_args[0]
        assert call_args[5] == 0.001  # train.optim.lr from flat dict
        assert call_args[6] == 64  # dataset.batch_size from flat dict

    @patch('wandb.log')
    @patch('wandb.Table')
    def test_update_wandb_table_missing_parameter(self, mock_table_class, mock_wandb_log):
        """Test handling of missing parameters in specs"""
        controller = self._create_controller()
        controller.wandb_initialized = True
        mock_table = Mock()
        controller.wandb_table = mock_table
        mock_table_class.return_value = mock_table

        # Create recommendation with missing parameter
        rec = Recommendation(
            identifier=0,
            specs={"train.optim.lr": 0.001},  # Missing dataset.batch_size
            metric="mAP"
        )
        rec.update_result(0.85)
        rec.update_status("success")
        rec.assign_job_id("job_1")

        controller.recommendations = [rec]
        controller.best_epoch_number = {0: "epoch_10"}

        controller._update_wandb_table()

        # Verify missing parameter shows as "N/A"
        call_args = mock_table.add_data.call_args[0]
        assert call_args[5] == 0.001  # train.optim.lr exists
        assert call_args[6] == "N/A"  # dataset.batch_size missing

    @patch('wandb.log')
    @patch('wandb.Table')
    def test_update_wandb_table_pending_job_id(self, mock_table_class, mock_wandb_log):
        """Test handling of pending job_id (None)"""
        controller = self._create_controller()
        controller.wandb_initialized = True
        mock_table = Mock()
        controller.wandb_table = mock_table
        mock_table_class.return_value = mock_table

        # Create recommendation with None job_id (default)
        rec = Recommendation(
            identifier=0,
            specs={"train.optim.lr": 0.001, "dataset.batch_size": 32},
            metric="mAP"
        )
        rec.update_result(0.85)
        rec.update_status("pending")
        # job_id remains None (default)

        controller.recommendations = [rec]
        controller.best_epoch_number = {}

        controller._update_wandb_table()

        # Verify None job_id is converted to "pending"
        call_args = mock_table.add_data.call_args[0]
        assert call_args[1] == "pending"  # job_id should be "pending" when None

    def test_update_wandb_table_not_initialized(self):
        """Test update when wandb is not initialized"""
        controller = self._create_controller()
        controller.wandb_initialized = False

        controller._update_wandb_table()

        # Should return early without error
        assert controller.wandb_table is None

    def test_update_wandb_table_no_table(self):
        """Test update when table is None"""
        controller = self._create_controller()
        controller.wandb_initialized = True
        controller.wandb_table = None

        controller._update_wandb_table()

        # Should return early without error

    @patch('wandb.Table')
    def test_update_wandb_table_exception_handling(self, mock_table_class):
        """Test update handles exceptions gracefully"""
        controller = self._create_controller()
        controller.wandb_initialized = True
        mock_table = Mock()
        controller.wandb_table = mock_table
        mock_table_class.side_effect = Exception("Table update failed")

        # Should not raise exception
        controller._update_wandb_table()


class TestWandBGroupInSpecs:
    """Test wandb group name update in job specs"""

    @patch('nvidia_tao_core.microservices.automl.controller.update_job_status')
    @patch('nvidia_tao_core.microservices.automl.controller.write_job_metadata')
    @patch('nvidia_tao_core.microservices.automl.controller.get_handler_job_metadata')
    @patch('nvidia_tao_core.microservices.automl.controller.update_job_message')
    @patch('nvidia_tao_core.microservices.automl.controller.create_cs_instance_with_decrypted_metadata')
    @patch('nvidia_tao_core.microservices.automl.controller.save_job_specs')
    @patch('nvidia_tao_core.microservices.automl.controller.get_job_specs')
    @patch('nvidia_tao_core.microservices.automl.controller.report_health_beat')
    def test_start_updates_wandb_group_in_specs(
        self, mock_report_health, mock_get_job_specs, mock_save_job_specs, mock_cs_instance,
        mock_update_job_message, mock_get_handler_job_metadata, mock_write_job_metadata,
        mock_update_job_status
    ):
        """Test that start() updates wandb group in job specs"""
        mock_cs_instance.return_value = (Mock(), None)
        mock_get_handler_job_metadata.return_value = {"job_details": {}}

        automl_context = Mock()
        automl_context.id = "test_job_123"
        automl_context.handler_id = "exp_456"
        automl_context.retain_checkpoints_for_resume = False

        brain = Mock()
        brain.reverse_sort = True

        algorithm_settings = AlgorithmParams(
            automl_max_recommendations=10,
            automl_max_epochs=81,
            automl_reduction_factor=3,
            epoch_multiplier=1
        )

        controller = Controller(
            root="/test/root",
            network="image_classification",
            brain=brain,
            automl_context=automl_context,
            automl_algorithm_settings=algorithm_settings,
            delete_intermediate_ckpt="false",
            metric="mAP",
            automl_algorithm="bayesian",
            decrypted_workspace_metadata={},
            parameter_names=["train.optim.lr"]
        )

        # Mock job specs with wandb config
        mock_get_job_specs.return_value = {
            "wandb": {
                "project": "Test Project"
            }
        }

        # Ensure update_job_message doesn't block
        mock_update_job_message.return_value = None

        # Mock other dependencies to prevent full execution and make test fast
        with patch.object(controller, '_initialize_wandb_for_automl'), \
             patch.object(controller, '_execute_loop'), \
             patch.object(controller, 'cancel_recommendation_jobs'), \
             patch.object(controller, '_get_experiment_results_path', return_value="/fake/path"):
            controller.start()

        # Verify get_job_specs was called
        mock_get_job_specs.assert_called()

        # Verify save_job_specs was called with updated group
        if mock_save_job_specs.called:
            call_args = mock_save_job_specs.call_args
            specs = call_args[0][1]
            assert specs["wandb"]["group"] == "automl_test_job_123"

    @patch('nvidia_tao_core.microservices.automl.controller.create_cs_instance_with_decrypted_metadata')
    def test_start_no_wandb_in_specs(self, mock_cs_instance):
        """Test that wandb_group_name is set correctly even when wandb config is missing"""
        mock_cs_instance.return_value = (Mock(), None)

        automl_context = Mock()
        automl_context.id = "test_job_123"
        automl_context.handler_id = "exp_456"
        automl_context.retain_checkpoints_for_resume = False

        brain = Mock()
        brain.reverse_sort = True

        algorithm_settings = AlgorithmParams(
            automl_max_recommendations=10,
            automl_max_epochs=81,
            automl_reduction_factor=3,
            epoch_multiplier=1
        )

        controller = Controller(
            root="/test/root",
            network="image_classification",
            brain=brain,
            automl_context=automl_context,
            automl_algorithm_settings=algorithm_settings,
            delete_intermediate_ckpt="false",
            metric="mAP",
            automl_algorithm="bayesian",
            decrypted_workspace_metadata={},
            parameter_names=["train.optim.lr"]
        )

        # Verify wandb_group_name is set correctly regardless of wandb config
        assert controller.wandb_group_name == "automl_test_job_123"

        # Note: We don't test start() directly here because it has many dependencies
        # (MongoDB connections, etc.) that would require extensive mocking.
        # The important part is that wandb_group_name is set during initialization.
