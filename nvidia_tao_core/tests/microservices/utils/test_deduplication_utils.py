# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deduplication_utils.py"""

import importlib
import os
import sys
from unittest.mock import Mock, patch

# Import with BACKEND set so MongoHandler gets imported
# Need to reload the module to ensure it's imported with the correct environment
with patch.dict(os.environ, {"BACKEND": "local-k8s"}):
    with patch('nvidia_tao_core.microservices.utils.mongo_utils.MongoHandler'):
        # If module was already imported, reload it
        if 'nvidia_tao_core.microservices.utils.deduplication_utils' in sys.modules:
            import nvidia_tao_core.microservices.utils.deduplication_utils
            importlib.reload(nvidia_tao_core.microservices.utils.deduplication_utils)
        from nvidia_tao_core.microservices.utils.deduplication_utils import (
            normalize_cloud_details,
            normalize_specs,
            normalize_automl_settings,
            find_duplicate_workspace,
            find_duplicate_dataset,
            find_duplicate_job,
            find_duplicate_experiment,
            create_indexes_for_deduplication
        )


class TestNormalizeCloudDetails:
    """Test normalize_cloud_details function"""

    def test_empty_cloud_details(self):
        """Test with empty cloud details"""
        assert normalize_cloud_details(None) == {}
        assert normalize_cloud_details({}) == {}

    def test_basic_normalization(self):
        """Test basic cloud details normalization"""
        cloud_details = {
            "bucket": "my-bucket",
            "region": "us-west-2",
            "access_key": "key123"
        }
        normalized = normalize_cloud_details(cloud_details)

        # Should be sorted by keys
        assert list(normalized.keys()) == ["access_key", "bucket", "region"]
        assert normalized["bucket"] == "my-bucket"

    def test_removes_none_values(self):
        """Test that None values are filtered out"""
        cloud_details = {
            "bucket": "my-bucket",
            "region": None,
            "access_key": "key123",
            "secret_key": None
        }
        normalized = normalize_cloud_details(cloud_details)

        assert "region" not in normalized
        assert "secret_key" not in normalized
        assert len(normalized) == 2

    def test_maintains_order_independence(self):
        """Test that different ordering produces same result"""
        details1 = {"z": "last", "a": "first", "m": "middle"}
        details2 = {"a": "first", "m": "middle", "z": "last"}

        assert normalize_cloud_details(details1) == normalize_cloud_details(details2)


class TestNormalizeSpecs:
    """Test normalize_specs function"""

    def test_empty_specs(self):
        """Test with empty specs"""
        assert normalize_specs(None) == {}
        assert normalize_specs({}) == {}

    def test_basic_normalization(self):
        """Test basic specs normalization"""
        specs = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10
        }
        normalized = normalize_specs(specs)

        # Should be sorted by keys
        assert list(normalized.keys()) == ["batch_size", "epochs", "learning_rate"]

    def test_excludes_output_paths(self):
        """Test that output path fields are excluded"""
        specs = {
            "batch_size": 32,
            "results_dir": "/tmp/results",
            "checkpoint_dir": "/tmp/checkpoints",
            "log_file": "/tmp/log.txt",
            "learning_rate": 0.001
        }
        normalized = normalize_specs(specs)

        assert "results_dir" not in normalized
        assert "checkpoint_dir" not in normalized
        assert "log_file" not in normalized
        assert "batch_size" in normalized
        assert "learning_rate" in normalized

    def test_removes_none_values(self):
        """Test that None values are filtered out"""
        specs = {
            "batch_size": 32,
            "learning_rate": None,
            "epochs": 10
        }
        normalized = normalize_specs(specs)

        assert "learning_rate" not in normalized
        assert len(normalized) == 2


class TestNormalizeAutomlSettings:
    """Test normalize_automl_settings function"""

    def test_empty_automl_settings(self):
        """Test with empty AutoML settings"""
        assert normalize_automl_settings(None) == {}
        assert normalize_automl_settings({}) == {}

    def test_basic_normalization(self):
        """Test basic AutoML settings normalization"""
        settings = {
            "algorithm": "bayesian",
            "max_recommendations": 10,
            "epochs": 20
        }
        normalized = normalize_automl_settings(settings)

        # Should be sorted by keys
        assert list(normalized.keys()) == ["algorithm", "epochs", "max_recommendations"]

    def test_excludes_transient_fields(self):
        """Test that transient fields are excluded"""
        settings = {
            "algorithm": "bayesian",
            "run_id": "abc123",
            "timestamp": "2025-01-01T00:00:00",
            "max_recommendations": 10
        }
        normalized = normalize_automl_settings(settings)

        assert "run_id" not in normalized
        assert "timestamp" not in normalized
        assert "algorithm" in normalized
        assert "max_recommendations" in normalized

    def test_removes_none_values(self):
        """Test that None values are filtered out"""
        settings = {
            "algorithm": "bayesian",
            "max_recommendations": None,
            "epochs": 20
        }
        normalized = normalize_automl_settings(settings)

        assert "max_recommendations" not in normalized
        assert len(normalized) == 2


class TestFindDuplicateWorkspace:
    """Test find_duplicate_workspace function"""

    def test_no_backend_env_returns_none(self):
        """Test that function returns None when BACKEND env is not set"""
        with patch.dict(os.environ, {}, clear=True):
            result = find_duplicate_workspace("user1", "org1", "aws", {})
            assert result is None

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_no_workspaces_found(self, mock_mongo):
        """Test when no workspaces match the query"""
        mock_handler = Mock()
        mock_handler.find.return_value = []
        mock_mongo.return_value = mock_handler

        result = find_duplicate_workspace(
            "user1", "org1", "aws",
            {"bucket": "my-bucket", "region": "us-west-2"}
        )

        assert result is None
        mock_handler.find.assert_called_once()

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_finds_exact_duplicate(self, mock_mongo):
        """Test finding exact duplicate workspace"""
        mock_handler = Mock()
        mock_handler.find.return_value = [
            {
                "id": "workspace123",
                "cloud_specific_details": {
                    "bucket": "my-bucket",
                    "region": "us-west-2"
                }
            }
        ]
        mock_mongo.return_value = mock_handler

        result = find_duplicate_workspace(
            "user1", "org1", "aws",
            {"bucket": "my-bucket", "region": "us-west-2"}
        )

        assert result == "workspace123"

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_finds_duplicate_different_order(self, mock_mongo):
        """Test finding duplicate with different key ordering"""
        mock_handler = Mock()
        mock_handler.find.return_value = [
            {
                "id": "workspace123",
                "cloud_specific_details": {
                    "region": "us-west-2",
                    "bucket": "my-bucket"
                }
            }
        ]
        mock_mongo.return_value = mock_handler

        result = find_duplicate_workspace(
            "user1", "org1", "aws",
            {"bucket": "my-bucket", "region": "us-west-2"}
        )

        assert result == "workspace123"

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_no_match_different_details(self, mock_mongo):
        """Test no match when cloud details differ"""
        mock_handler = Mock()
        mock_handler.find.return_value = [
            {
                "id": "workspace123",
                "cloud_specific_details": {
                    "bucket": "other-bucket",
                    "region": "us-west-2"
                }
            }
        ]
        mock_mongo.return_value = mock_handler

        result = find_duplicate_workspace(
            "user1", "org1", "aws",
            {"bucket": "my-bucket", "region": "us-west-2"}
        )

        assert result is None

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_exception_handling(self, mock_mongo):
        """Test exception handling"""
        mock_handler = Mock()
        mock_handler.find.side_effect = Exception("Database error")
        mock_mongo.return_value = mock_handler

        result = find_duplicate_workspace("user1", "org1", "aws", {})

        assert result is None


class TestFindDuplicateDataset:
    """Test find_duplicate_dataset function"""

    def test_no_backend_env_returns_none(self):
        """Test that function returns None when BACKEND env is not set"""
        with patch.dict(os.environ, {}, clear=True):
            result = find_duplicate_dataset("user1", "org1", {})
            assert result is None

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_finds_duplicate_dataset(self, mock_mongo):
        """Test finding duplicate dataset"""
        mock_handler = Mock()
        mock_handler.find.return_value = [
            {"id": "dataset123"}
        ]
        mock_mongo.return_value = mock_handler

        params = {
            "type": "object_detection",
            "format": "kitti",
            "workspace": "workspace1",
            "cloud_file_path": "/path/to/data"
        }

        result = find_duplicate_dataset("user1", "org1", params)

        assert result == "dataset123"
        mock_handler.find.assert_called_once()

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_no_duplicate_found(self, mock_mongo):
        """Test when no duplicate dataset exists"""
        mock_handler = Mock()
        mock_handler.find.return_value = []
        mock_mongo.return_value = mock_handler

        params = {
            "type": "object_detection",
            "format": "kitti"
        }

        result = find_duplicate_dataset("user1", "org1", params)

        assert result is None

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_optional_fields_in_query(self, mock_mongo):
        """Test that optional fields are only added to query if present"""
        mock_handler = Mock()
        mock_handler.find.return_value = []
        mock_mongo.return_value = mock_handler

        params = {
            "type": "object_detection",
            "format": "kitti",
            "workspace": "workspace1",
            "url": "https://example.com/data"
        }

        find_duplicate_dataset("user1", "org1", params)

        # Verify query includes optional fields
        call_args = mock_handler.find.call_args[0][0]
        assert "workspace" in call_args
        assert "url" in call_args
        assert call_args["workspace"] == "workspace1"


class TestFindDuplicateJob:
    """Test find_duplicate_job function"""

    def test_no_backend_env_returns_none(self):
        """Test that function returns None when BACKEND env is not set"""
        with patch.dict(os.environ, {}, clear=True):
            result = find_duplicate_job("user1", "org1", {})
            assert result is None

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_finds_duplicate_experiment_job(self, mock_mongo):
        """Test finding duplicate experiment job"""
        # Create mock handlers
        mock_experiments_handler = Mock()
        mock_jobs_handler = Mock()

        # Setup experiment with jobs
        mock_experiments_handler.find.return_value = [
            {
                "id": "exp1",
                "network_arch": "resnet18",
                "jobs": {
                    "job123": {
                        "action": "train",
                        "status": "Done"
                    }
                }
            }
        ]

        # Setup full job details
        mock_jobs_handler.find_one.return_value = {
            "id": "job123",
            "specs": {
                "batch_size": 32,
                "epochs": 10
            }
        }

        # Configure MongoHandler to return different handlers
        def mongo_side_effect(db, collection):
            handler = Mock()
            if collection == "experiments":
                return mock_experiments_handler
            elif collection == "jobs":
                return mock_jobs_handler
            return handler

        mock_mongo.side_effect = mongo_side_effect

        params = {
            "kind": "experiment",
            "handler_id": "exp1",
            "action": "train",
            "network_arch": "resnet18",
            "specs": {
                "batch_size": 32,
                "epochs": 10
            }
        }

        result = find_duplicate_job("user1", "org1", params)

        assert result == "job123"

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_no_duplicate_different_specs(self, mock_mongo):
        """Test no duplicate when specs differ"""
        mock_experiments_handler = Mock()
        mock_jobs_handler = Mock()

        mock_experiments_handler.find.return_value = [
            {
                "id": "exp1",
                "network_arch": "resnet18",
                "jobs": {
                    "job123": {
                        "action": "train",
                        "status": "Done"
                    }
                }
            }
        ]

        mock_jobs_handler.find_one.return_value = {
            "id": "job123",
            "specs": {
                "batch_size": 64,  # Different
                "epochs": 10
            }
        }

        def mongo_side_effect(db, collection):
            if collection == "experiments":
                return mock_experiments_handler
            elif collection == "jobs":
                return mock_jobs_handler
            return Mock()

        mock_mongo.side_effect = mongo_side_effect

        params = {
            "kind": "experiment",
            "handler_id": "exp1",
            "action": "train",
            "network_arch": "resnet18",
            "specs": {
                "batch_size": 32,  # Different
                "epochs": 10
            }
        }

        result = find_duplicate_job("user1", "org1", params)

        assert result is None

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_no_duplicate_different_action(self, mock_mongo):
        """Test no duplicate when action differs"""
        mock_experiments_handler = Mock()
        mock_jobs_handler = Mock()

        mock_experiments_handler.find.return_value = [
            {
                "id": "exp1",
                "network_arch": "resnet18",
                "jobs": {
                    "job123": {
                        "action": "evaluate",  # Different
                        "status": "Done"
                    }
                }
            }
        ]

        def mongo_side_effect(db, collection):
            if collection == "experiments":
                return mock_experiments_handler
            elif collection == "jobs":
                return mock_jobs_handler
            return Mock()

        mock_mongo.side_effect = mongo_side_effect

        params = {
            "kind": "experiment",
            "handler_id": "exp1",
            "action": "train",
            "network_arch": "resnet18",
            "specs": {}
        }

        result = find_duplicate_job("user1", "org1", params)

        assert result is None

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_automl_settings_matching(self, mock_mongo):
        """Test duplicate detection with AutoML settings"""
        mock_experiments_handler = Mock()
        mock_jobs_handler = Mock()

        mock_experiments_handler.find.return_value = [
            {
                "id": "exp1",
                "network_arch": "resnet18",
                "automl_settings": {
                    "algorithm": "bayesian",
                    "max_recommendations": 10
                },
                "jobs": {
                    "job123": {
                        "action": "train",
                        "status": "Done"
                    }
                }
            }
        ]

        mock_jobs_handler.find_one.return_value = {
            "id": "job123",
            "specs": {"batch_size": 32}
        }

        def mongo_side_effect(db, collection):
            if collection == "experiments":
                return mock_experiments_handler
            elif collection == "jobs":
                return mock_jobs_handler
            return Mock()

        mock_mongo.side_effect = mongo_side_effect

        params = {
            "kind": "experiment",
            "handler_id": "exp1",
            "action": "train",
            "network_arch": "resnet18",
            "specs": {"batch_size": 32},
            "automl_settings": {
                "algorithm": "bayesian",
                "max_recommendations": 10
            }
        }

        result = find_duplicate_job("user1", "org1", params)

        assert result == "job123"

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_no_duplicate_different_automl(self, mock_mongo):
        """Test no duplicate when AutoML settings differ"""
        mock_experiments_handler = Mock()
        mock_jobs_handler = Mock()

        mock_experiments_handler.find.return_value = [
            {
                "id": "exp1",
                "network_arch": "resnet18",
                "automl_settings": {
                    "algorithm": "hyperband",  # Different
                    "max_recommendations": 10
                },
                "jobs": {
                    "job123": {
                        "action": "train",
                        "status": "Done"
                    }
                }
            }
        ]

        mock_jobs_handler.find_one.return_value = {
            "id": "job123",
            "specs": {"batch_size": 32}
        }

        def mongo_side_effect(db, collection):
            if collection == "experiments":
                return mock_experiments_handler
            elif collection == "jobs":
                return mock_jobs_handler
            return Mock()

        mock_mongo.side_effect = mongo_side_effect

        params = {
            "kind": "experiment",
            "handler_id": "exp1",
            "action": "train",
            "network_arch": "resnet18",
            "specs": {"batch_size": 32},
            "automl_settings": {
                "algorithm": "bayesian",  # Different
                "max_recommendations": 10
            }
        }

        result = find_duplicate_job("user1", "org1", params)

        assert result is None

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_output_paths_ignored_in_specs(self, mock_mongo):
        """Test that output paths in specs don't affect duplicate detection"""
        mock_experiments_handler = Mock()
        mock_jobs_handler = Mock()

        mock_experiments_handler.find.return_value = [
            {
                "id": "exp1",
                "network_arch": "resnet18",
                "jobs": {
                    "job123": {
                        "action": "train",
                        "status": "Done"
                    }
                }
            }
        ]

        mock_jobs_handler.find_one.return_value = {
            "id": "job123",
            "specs": {
                "batch_size": 32,
                "results_dir": "/old/path",
                "checkpoint_dir": "/old/checkpoint"
            }
        }

        def mongo_side_effect(db, collection):
            if collection == "experiments":
                return mock_experiments_handler
            elif collection == "jobs":
                return mock_jobs_handler
            return Mock()

        mock_mongo.side_effect = mongo_side_effect

        params = {
            "kind": "experiment",
            "handler_id": "exp1",
            "action": "train",
            "network_arch": "resnet18",
            "specs": {
                "batch_size": 32,
                "results_dir": "/new/path",  # Different but should be ignored
                "checkpoint_dir": "/new/checkpoint"  # Different but should be ignored
            }
        }

        result = find_duplicate_job("user1", "org1", params)

        assert result == "job123"


class TestFindDuplicateExperiment:
    """Test find_duplicate_experiment function"""

    def test_no_backend_env_returns_none(self):
        """Test that function returns None when BACKEND env is not set"""
        with patch.dict(os.environ, {}, clear=True):
            result = find_duplicate_experiment(
                "user1", "org1", "exp1", "resnet18",
                "workspace1", [], [], None
            )
            assert result is None

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_finds_duplicate_experiment(self, mock_mongo):
        """Test finding duplicate experiment"""
        mock_handler = Mock()
        mock_handler.find.return_value = [
            {
                "id": "exp123",
                "name": "my_experiment",
                "workspace": "workspace1",
                "base_experiment_ids": [],
                "train_datasets": ["dataset1", "dataset2"],
                "eval_dataset": "eval_dataset1"
            }
        ]
        mock_mongo.return_value = mock_handler

        result = find_duplicate_experiment(
            "user1", "org1", "my_experiment", "resnet18",
            "workspace1", [], ["dataset1", "dataset2"], "eval_dataset1"
        )

        assert result == "exp123"

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_no_duplicate_different_workspace(self, mock_mongo):
        """Test no duplicate when workspace differs"""
        mock_handler = Mock()
        mock_handler.find.return_value = [
            {
                "id": "exp123",
                "workspace": "workspace2",  # Different
                "base_experiment_ids": [],
                "train_datasets": ["dataset1"],
                "eval_dataset": "eval_dataset1"
            }
        ]
        mock_mongo.return_value = mock_handler

        result = find_duplicate_experiment(
            "user1", "org1", "my_experiment", "resnet18",
            "workspace1", [], ["dataset1"], "eval_dataset1"
        )

        assert result is None

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_train_datasets_order_independent(self, mock_mongo):
        """Test that train datasets order doesn't matter"""
        mock_handler = Mock()
        mock_handler.find.return_value = [
            {
                "id": "exp123",
                "workspace": "workspace1",
                "base_experiment_ids": [],
                "train_datasets": ["dataset2", "dataset1"],  # Different order
                "eval_dataset": "eval_dataset1"
            }
        ]
        mock_mongo.return_value = mock_handler

        result = find_duplicate_experiment(
            "user1", "org1", "my_experiment", "resnet18",
            "workspace1", [], ["dataset1", "dataset2"], "eval_dataset1"
        )

        assert result == "exp123"

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_base_experiment_ids_order_independent(self, mock_mongo):
        """Test that base experiment IDs order doesn't matter"""
        mock_handler = Mock()
        mock_handler.find.return_value = [
            {
                "id": "exp123",
                "workspace": "workspace1",
                "base_experiment_ids": ["base2", "base1"],  # Different order
                "train_datasets": ["dataset1"],
                "eval_dataset": "eval_dataset1"
            }
        ]
        mock_mongo.return_value = mock_handler

        result = find_duplicate_experiment(
            "user1", "org1", "my_experiment", "resnet18",
            "workspace1", ["base1", "base2"], ["dataset1"], "eval_dataset1"
        )

        assert result == "exp123"

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_no_duplicate_different_eval_dataset(self, mock_mongo):
        """Test no duplicate when eval dataset differs"""
        mock_handler = Mock()
        mock_handler.find.return_value = [
            {
                "id": "exp123",
                "workspace": "workspace1",
                "base_experiment_ids": [],
                "train_datasets": ["dataset1"],
                "eval_dataset": "eval_dataset2"  # Different
            }
        ]
        mock_mongo.return_value = mock_handler

        result = find_duplicate_experiment(
            "user1", "org1", "my_experiment", "resnet18",
            "workspace1", [], ["dataset1"], "eval_dataset1"
        )

        assert result is None

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_no_experiments_found(self, mock_mongo):
        """Test when no experiments exist"""
        mock_handler = Mock()
        mock_handler.find.return_value = []
        mock_mongo.return_value = mock_handler

        result = find_duplicate_experiment(
            "user1", "org1", "my_experiment", "resnet18",
            "workspace1", [], ["dataset1"], "eval_dataset1"
        )

        assert result is None


class TestCreateIndexesForDeduplication:
    """Test create_indexes_for_deduplication function"""

    def test_no_backend_env_returns_early(self):
        """Test that function returns early when BACKEND env is not set"""
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise any exception
            create_indexes_for_deduplication()

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_creates_workspace_index(self, mock_mongo):
        """Test workspace index creation"""
        mock_workspaces_handler = Mock()
        mock_datasets_handler = Mock()

        def mongo_side_effect(db, collection):
            if collection == "workspaces":
                return mock_workspaces_handler
            elif collection == "datasets":
                return mock_datasets_handler
            return Mock()

        mock_mongo.side_effect = mongo_side_effect

        create_indexes_for_deduplication()

        # Verify workspace index was created
        mock_workspaces_handler.collection.create_index.assert_called_once()
        call_args = mock_workspaces_handler.collection.create_index.call_args
        assert call_args[0][0] == [("user_id", 1), ("org_name", 1), ("cloud_type", 1)]

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_creates_dataset_index(self, mock_mongo):
        """Test dataset index creation"""
        mock_workspaces_handler = Mock()
        mock_datasets_handler = Mock()

        def mongo_side_effect(db, collection):
            if collection == "workspaces":
                return mock_workspaces_handler
            elif collection == "datasets":
                return mock_datasets_handler
            return Mock()

        mock_mongo.side_effect = mongo_side_effect

        create_indexes_for_deduplication()

        # Verify dataset index was created
        mock_datasets_handler.collection.create_index.assert_called_once()
        call_args = mock_datasets_handler.collection.create_index.call_args
        expected_fields = [
            ("user_id", 1), ("org_name", 1), ("type", 1),
            ("format", 1), ("workspace", 1)
        ]
        assert call_args[0][0] == expected_fields

    @patch.dict(os.environ, {"BACKEND": "local-k8s"})
    @patch('nvidia_tao_core.microservices.utils.deduplication_utils.MongoHandler')
    def test_handles_index_creation_errors(self, mock_mongo):
        """Test graceful handling of index creation errors"""
        mock_handler = Mock()
        mock_handler.collection.create_index.side_effect = Exception("Index error")
        mock_mongo.return_value = mock_handler

        # Should not raise exception
        create_indexes_for_deduplication()
