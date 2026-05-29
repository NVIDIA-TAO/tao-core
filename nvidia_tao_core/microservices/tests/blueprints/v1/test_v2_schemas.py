# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for v1 schemas module."""

from nvidia_tao_core.microservices.blueprints.v1.schemas import JobStatusEnum


class TestExperimentSchemas:
    """Tests for experiment schemas."""

    def test_experiment_req_schema_loads_valid_data(self):
        """Test ExperimentReq schema loads valid experiment data."""
        from nvidia_tao_core.microservices.blueprints.v1.schemas import ExperimentReq

        schema = ExperimentReq()
        valid_data = {
            "name": "Test Experiment",
            "description": "A test experiment",
            "network_arch": "classification_pyt",
            "encryption_key": "tlt_encode",
        }

        result = schema.load(valid_data)
        assert result["name"] == "Test Experiment"
        assert result["network_arch"] == "classification_pyt"

    def test_experiment_req_schema_without_type_field(self):
        """Test ExperimentReq schema does not require type field."""
        from nvidia_tao_core.microservices.blueprints.v1.schemas import ExperimentReq

        schema = ExperimentReq()
        data = {"name": "Test Experiment"}

        result = schema.load(data)
        # type field should not exist in the schema
        assert "type" not in result or result.get("type") is None

    def test_experiment_rsp_schema_dumps_valid_data(self):
        """Test ExperimentRsp schema dumps experiment response data."""
        from nvidia_tao_core.microservices.blueprints.v1.schemas import ExperimentRsp

        schema = ExperimentRsp()
        response_data = {
            "id": "test-exp-123",
            "name": "Test Experiment",
            "description": "A test experiment",
            "network_arch": "classification_pyt",
            "status": JobStatusEnum.Pending.value,
            "all_jobs_cancel_status": JobStatusEnum.Pending.value,
        }

        result = schema.dump(schema.load(response_data))
        assert result["id"] == "test-exp-123"
        assert result["name"] == "Test Experiment"

    def test_experiment_export_type_enum_has_tao(self):
        """Test ExperimentExportTypeEnum contains tao value."""
        from nvidia_tao_core.microservices.blueprints.v1.schemas import ExperimentExportTypeEnum

        assert hasattr(ExperimentExportTypeEnum, 'tao')
        assert ExperimentExportTypeEnum.tao.value == 'tao'

    def test_experiment_export_type_enum_no_monai_bundle(self):
        """Test ExperimentExportTypeEnum does not contain monai_bundle value."""
        from nvidia_tao_core.microservices.blueprints.v1.schemas import ExperimentExportTypeEnum

        assert not hasattr(ExperimentExportTypeEnum, 'monai_bundle')


class TestDatasetSchemas:
    """Tests for dataset schemas."""

    def test_dataset_req_schema_loads_valid_data(self):
        """Test DatasetReq schema loads valid dataset data."""
        from nvidia_tao_core.microservices.blueprints.v1.schemas import DatasetReq

        schema = DatasetReq()
        valid_data = {
            "name": "Test Dataset",
            "description": "A test dataset",
            "type": "object_detection",
            "format": "kitti",
        }

        result = schema.load(valid_data)
        assert result["name"] == "Test Dataset"
        assert result["type"] == "object_detection"
        assert result["format"] == "kitti"

    def test_dataset_rsp_schema_dumps_valid_data(self):
        """Test DatasetRsp schema dumps dataset response data."""
        from nvidia_tao_core.microservices.blueprints.v1.schemas import DatasetRsp

        schema = DatasetRsp()
        response_data = {
            "id": "test-ds-123",
            "name": "Test Dataset",
            "type": "object_detection",
            "format": "kitti",
            "status": "pull_complete",
        }

        result = schema.dump(schema.load(response_data))
        assert result["id"] == "test-ds-123"
        assert result["name"] == "Test Dataset"


class TestJobSchemas:
    """Tests for job schemas."""

    def test_job_status_enum_values(self):
        """Test JobStatusEnum contains expected values."""
        from nvidia_tao_core.microservices.blueprints.v1.schemas import JobStatusEnum

        expected_statuses = ['Pending', 'Running', 'Done', 'Error', 'Canceled']
        for status in expected_statuses:
            assert hasattr(JobStatusEnum, status), f"Missing status: {status}"


class TestErrorSchemas:
    """Tests for error response schemas."""

    def test_error_rsp_schema_loads_error_data(self):
        """Test ErrorRsp schema loads error response data."""
        from nvidia_tao_core.microservices.blueprints.v1.schemas import ErrorRsp

        schema = ErrorRsp()
        error_data = {
            "error_desc": "Something went wrong",
            "error_code": 500,
        }

        result = schema.load(error_data)
        assert result["error_desc"] == "Something went wrong"
        assert result["error_code"] == 500


class TestWorkspaceSchemas:
    """Tests for workspace schemas."""

    def test_workspace_req_schema_loads_valid_data(self):
        """Test WorkspaceReq schema loads valid workspace data."""
        from nvidia_tao_core.microservices.blueprints.v1.schemas import WorkspaceReq

        schema = WorkspaceReq()
        valid_data = {
            "name": "Test Workspace",
            "description": "A test workspace",
        }

        result = schema.load(valid_data)
        assert result["name"] == "Test Workspace"

    def test_workspace_rsp_schema_dumps_valid_data(self):
        """Test WorkspaceRsp schema dumps workspace response data."""
        from nvidia_tao_core.microservices.blueprints.v1.schemas import WorkspaceRsp

        schema = WorkspaceRsp()
        response_data = {
            "id": "test-ws-123",
            "name": "Test Workspace",
        }

        result = schema.dump(schema.load(response_data))
        assert result["id"] == "test-ws-123"
        assert result["name"] == "Test Workspace"
