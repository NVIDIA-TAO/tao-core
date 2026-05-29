# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for v2 schemas module."""


class TestDatasetSchemas:
    """Tests for v2 dataset schemas."""

    def test_dataset_req_schema_loads_valid_data(self):
        """Test DatasetReq schema loads valid dataset data."""
        from nvidia_tao_core.microservices.blueprints.v2.schemas import DatasetReq

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
        from nvidia_tao_core.microservices.blueprints.v2.schemas import DatasetRsp

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


class TestWorkspaceSchemas:
    """Tests for v2 workspace schemas."""

    def test_workspace_req_schema_loads_valid_data(self):
        """Test WorkspaceReq schema loads valid workspace data."""
        from nvidia_tao_core.microservices.blueprints.v2.schemas import WorkspaceReq

        schema = WorkspaceReq()
        valid_data = {
            "name": "Test Workspace",
            "description": "A test workspace",
        }

        result = schema.load(valid_data)
        assert result["name"] == "Test Workspace"

    def test_workspace_rsp_schema_dumps_valid_data(self):
        """Test WorkspaceRsp schema dumps workspace response data."""
        from nvidia_tao_core.microservices.blueprints.v2.schemas import WorkspaceRsp

        schema = WorkspaceRsp()
        response_data = {
            "id": "test-ws-123",
            "name": "Test Workspace",
        }

        result = schema.dump(schema.load(response_data))
        assert result["id"] == "test-ws-123"
        assert result["name"] == "Test Workspace"


class TestJobSchemas:
    """Tests for v2 job schemas."""

    def test_job_status_enum_values(self):
        """Test JobStatusEnum contains expected values."""
        from nvidia_tao_core.microservices.blueprints.v2.schemas import JobStatusEnum

        expected_statuses = ['Pending', 'Running', 'Done', 'Error', 'Canceled']
        for status in expected_statuses:
            assert hasattr(JobStatusEnum, status), f"Missing status: {status}"


class TestErrorSchemas:
    """Tests for v2 error response schemas."""

    def test_error_rsp_schema_loads_error_data(self):
        """Test ErrorRsp schema loads error response data."""
        from nvidia_tao_core.microservices.blueprints.v2.schemas import ErrorRsp

        schema = ErrorRsp()
        error_data = {
            "error_desc": "Something went wrong",
            "error_code": 500,
        }

        result = schema.load(error_data)
        assert result["error_desc"] == "Something went wrong"
        assert result["error_code"] == 500
