# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dataset handler fixes.

Covers calibration intent support, virtual dataset use_for inference,
runtime validator format fallback, model handler error format,
publish model artifact isolation, and monitor_and_upload exit handling.
"""
import inspect
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


class TestCalibrationDatasetIntent(unittest.TestCase):
    """Bug: DatasetIntentEnum was missing 'calibration', so datasets with
    use_for=['calibration'] were rejected by the schema (max length was 3)."""

    def test_v2_schema_has_calibration_intent(self):
        from nvidia_tao_core.microservices.blueprints.v2.schemas import DatasetIntentEnum
        members = [e.value for e in DatasetIntentEnum]
        self.assertIn('calibration', members,
                      "DatasetIntentEnum must include 'calibration'")

    def test_v1_schema_has_calibration_intent(self):
        from nvidia_tao_core.microservices.blueprints.v1.schemas import DatasetIntentEnum
        members = [e.value for e in DatasetIntentEnum]
        self.assertIn('calibration', members,
                      "DatasetIntentEnum must include 'calibration'")

    def test_v2_dataset_intent_allows_4_items(self):
        """use_for list should accept up to 4 intents (training, evaluation, testing, calibration)."""
        from nvidia_tao_core.microservices.blueprints.v2.schemas import DatasetReq
        schema = DatasetReq()
        use_for_field = schema.fields.get('use_for')
        self.assertIsNotNone(use_for_field, "DatasetReq must have 'use_for' field")
        for validator in use_for_field.validators:
            if hasattr(validator, 'max'):
                self.assertGreaterEqual(validator.max, 4,
                                        "use_for max length must be >= 4 to allow calibration")


class TestVirtualDatasetUseFor(unittest.TestCase):
    """Verify _create_virtual_dataset_for_direct_paths populates both use_for
    intent tags and cloud_file_path from the provided URIs."""

    @patch('nvidia_tao_core.microservices.utils.stateless_handler_utils.write_handler_metadata')
    def test_use_for_populated_from_uri_fields(self, mock_write):
        mock_mongo = MagicMock()
        mock_mongo.find_one.return_value = None
        mock_mongo_cls = MagicMock(return_value=mock_mongo)

        with patch.dict('sys.modules', {}):
            pass

        with patch('nvidia_tao_core.microservices.handlers.mongo_handler.MongoHandler', mock_mongo_cls), \
             patch('nvidia_tao_core.microservices.utils.basic_utils.get_user_datasets', return_value=[]), \
             patch('nvidia_tao_core.microservices.utils.basic_utils.get_dataset_actions',
                   return_value=["dataset_convert"]), \
             patch('nvidia_tao_core.microservices.utils.core_utils.read_network_config',
                   return_value={"api_params": {"formats": ["coco"]}}):
            from nvidia_tao_core.microservices.blueprints.v2.jobs import (
                _create_virtual_dataset_for_direct_paths
            )
            request_dict = {
                "train_dataset_uris": ["s3://bucket/train"],
                "eval_dataset_uri": "s3://bucket/eval",
                "inference_dataset_uri": None,
                "calibration_dataset_uri": None,
                "dataset_type": "object_detection",
                "dataset_format": "coco",
                "workspace": "ws-123",
            }
            _create_virtual_dataset_for_direct_paths("user1", "org1", request_dict)

        self.assertTrue(mock_write.called, "write_handler_metadata should be called")
        written_metadata = mock_write.call_args[0][1]
        self.assertIn('use_for', written_metadata,
                      "Virtual dataset metadata must contain 'use_for'")
        self.assertIn('training', written_metadata['use_for'])
        self.assertIn('evaluation', written_metadata['use_for'])
        self.assertIn('cloud_file_path', written_metadata,
                      "Virtual dataset metadata must contain 'cloud_file_path'")
        self.assertEqual(written_metadata['cloud_file_path'], 'train',
                         "cloud_file_path should be extracted from the primary URI")


class TestInferenceDatasetIntentMapping(unittest.TestCase):
    """Bug: validate_all_dataset_uris_structure passed intent=["inference"] for
    inference_dataset_uri, but the schema/configs expected intent="testing"."""

    @patch('nvidia_tao_core.microservices.utils.runtime_dataset_validator.validate_dataset_uri_structure')
    @patch('nvidia_tao_core.microservices.utils.runtime_dataset_validator.read_network_config')
    @patch('nvidia_tao_core.microservices.utils.runtime_dataset_validator.is_uuid', return_value=False)
    def test_inference_uri_uses_testing_intent(self, mock_is_uuid, mock_read_config, mock_validate):
        mock_read_config.return_value = {
            "api_params": {
                "formats": ["coco"],
                "dataset_type": "object_detection"
            }
        }
        mock_validate.return_value = (True, {})

        from nvidia_tao_core.microservices.utils.runtime_dataset_validator import (
            validate_all_dataset_uris_structure
        )
        experiment_metadata = {
            "inference_dataset_uri": "s3://bucket/inference_data",
        }
        validate_all_dataset_uris_structure(
            experiment_metadata=experiment_metadata,
            network_arch="dino",
        )
        self.assertTrue(mock_validate.called, "validate_dataset_uri_structure should be called")
        call_kwargs = mock_validate.call_args[1]
        intent_arg = call_kwargs.get('dataset_intent')
        self.assertEqual(intent_arg, ["testing"],
                         f"inference_dataset_uri must use intent=['testing'], not {intent_arg}")


class TestRuntimeValidatorFormatFallback(unittest.TestCase):
    """Bug: When primary format validation failed, the validator didn't try
    alternative formats from api_params['formats']. After the fix, it retries."""

    @patch('nvidia_tao_core.microservices.utils.runtime_dataset_validator.validate_dataset_uri_structure')
    @patch('nvidia_tao_core.microservices.utils.runtime_dataset_validator.read_network_config')
    @patch('nvidia_tao_core.microservices.utils.runtime_dataset_validator.is_uuid', return_value=False)
    def test_fallback_to_alt_format_on_validation_failure(self, mock_is_uuid,
                                                          mock_read_config, mock_validate):
        mock_read_config.return_value = {
            "api_params": {
                "formats": ["odvg", "coco", "kitti"],
                "dataset_type": "object_detection"
            }
        }

        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            fmt = kwargs.get('dataset_format', '')
            if fmt == 'odvg':
                return (False, {"error_details": "Wrong format"})
            elif fmt == 'coco':
                return (True, {})
            return (False, {"error_details": "Unknown format"})

        mock_validate.side_effect = side_effect

        from nvidia_tao_core.microservices.utils.runtime_dataset_validator import (
            validate_all_dataset_uris_structure
        )
        experiment_metadata = {
            "eval_dataset_uri": "s3://bucket/eval_data",
            "dataset_format": "odvg",
        }
        is_valid, error_msg, details = validate_all_dataset_uris_structure(
            experiment_metadata=experiment_metadata,
            network_arch="dino",
        )
        self.assertTrue(is_valid,
                        f"Validation should pass with alt format 'coco', got error: {error_msg}")
        self.assertGreater(call_count[0], 1,
                           "Validator should retry with alternative formats")


class TestModelHandlerErrorResponseFormat(unittest.TestCase):
    """Bug: ModelHandler error responses used {"message": ...} key, not the
    standard {"error_desc": ..., "error_code": ...} format expected by the UI."""

    @patch('nvidia_tao_core.microservices.handlers.model_handler.resolve_metadata', return_value=None)
    def test_publish_model_error_response_has_error_desc(self, mock_resolve):
        from nvidia_tao_core.microservices.handlers.model_handler import ModelHandler
        result = ModelHandler.publish_model("org", "team", "exp-123", "job-456", "name", "desc")
        self.assertIn('error_desc', result.data,
                      f"Error response data must contain 'error_desc'. Got: {result.data}")
        self.assertIn('error_code', result.data,
                      f"Error response data must contain 'error_code'. Got: {result.data}")

    @patch('nvidia_tao_core.microservices.handlers.model_handler.resolve_metadata', return_value=None)
    def test_remove_published_model_error_response_has_error_desc(self, mock_resolve):
        from nvidia_tao_core.microservices.handlers.model_handler import ModelHandler
        result = ModelHandler.remove_published_model("org", "team", "exp-123", "job-456")
        self.assertIn('error_desc', result.data,
                      f"Error response data must contain 'error_desc'. Got: {result.data}")
        self.assertIn('error_code', result.data,
                      f"Error response data must contain 'error_code'. Got: {result.data}")


class TestPublishModelArtifactsDirectory(unittest.TestCase):
    """Bug: publish_model_artifacts used a shared directory without job_id suffix,
    so concurrent publishes could overwrite each other's artifacts."""

    def test_artifact_dir_includes_job_id(self):
        from nvidia_tao_core.microservices.utils import ngc_utils
        source = inspect.getsource(ngc_utils.upload_model)
        self.assertIn('publish_model_artifacts_{job_id}', source,
                      "upload_model must use job-specific directory "
                      "(publish_model_artifacts_{job_id}), not shared one")


class TestMonitorAndUploadExitEvent(unittest.TestCase):
    """Bug: monitor_and_upload used time.sleep(30) which couldn't be interrupted,
    causing 30s delays on graceful shutdown. Fix uses exit_event.wait(30)."""

    def test_exit_event_wait_is_used(self):
        from nvidia_tao_core.microservices.handlers.cloud_handlers import utils
        source = inspect.getsource(utils.monitor_and_upload)
        self.assertIn('exit_event.wait', source,
                      "monitor_and_upload should use exit_event.wait() instead of time.sleep()")
        self.assertNotIn('time.sleep(30)', source,
                         "monitor_and_upload should NOT use time.sleep(30)")


if __name__ == '__main__':
    unittest.main()
