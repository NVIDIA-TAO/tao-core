# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for filtering utilities."""

from datetime import datetime, timezone


class TestFilteringApply:
    """Tests for filtering.apply function."""

    def test_filter_by_name(self):
        """Test filtering by name works correctly."""
        from nvidia_tao_core.microservices.utils.filter_utils import filtering

        data = [
            {"name": "experiment1", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "experiment2", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "experiment3", "created_on": datetime.now(tz=timezone.utc)},
        ]

        result = filtering.apply({"name": "experiment2"}, data)
        assert len(result) == 1
        assert result[0]["name"] == "experiment2"

    def test_filter_by_name_negation(self):
        """Test filtering by negated name works correctly."""
        from nvidia_tao_core.microservices.utils.filter_utils import filtering

        data = [
            {"name": "experiment1", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "experiment2", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "experiment3", "created_on": datetime.now(tz=timezone.utc)},
        ]

        result = filtering.apply({"name": "!experiment2"}, data)
        assert len(result) == 2
        assert all(r["name"] != "experiment2" for r in result)

    def test_filter_by_type(self):
        """Test filtering by type works correctly."""
        from nvidia_tao_core.microservices.utils.filter_utils import filtering

        data = [
            {"name": "ds1", "type": "object_detection", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "ds2", "type": "segmentation", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "ds3", "type": "object_detection", "created_on": datetime.now(tz=timezone.utc)},
        ]

        result = filtering.apply({"type": "object_detection"}, data)
        assert len(result) == 2
        assert all(r["type"] == "object_detection" for r in result)

    def test_filter_by_network_arch(self):
        """Test filtering by network_arch works correctly."""
        from nvidia_tao_core.microservices.utils.filter_utils import filtering

        data = [
            {"name": "exp1", "network_arch": "classification_pyt", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "exp2", "network_arch": "segformer", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "exp3", "network_arch": "classification_pyt", "created_on": datetime.now(tz=timezone.utc)},
        ]

        result = filtering.apply({"network_arch": "classification_pyt"}, data)
        assert len(result) == 2
        assert all(r["network_arch"] == "classification_pyt" for r in result)

    def test_filter_by_format(self):
        """Test filtering by format works correctly."""
        from nvidia_tao_core.microservices.utils.filter_utils import filtering

        data = [
            {"name": "ds1", "format": "kitti", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "ds2", "format": "coco", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "ds3", "format": "kitti", "created_on": datetime.now(tz=timezone.utc)},
        ]

        result = filtering.apply({"format": "kitti"}, data)
        assert len(result) == 2
        assert all(r["format"] == "kitti" for r in result)

    def test_filter_by_read_only(self):
        """Test filtering by read_only works correctly."""
        from nvidia_tao_core.microservices.utils.filter_utils import filtering

        data = [
            {"name": "exp1", "read_only": True, "created_on": datetime.now(tz=timezone.utc)},
            {"name": "exp2", "read_only": False, "created_on": datetime.now(tz=timezone.utc)},
            {"name": "exp3", "read_only": True, "created_on": datetime.now(tz=timezone.utc)},
        ]

        result = filtering.apply({"read_only": "true"}, data)
        assert len(result) == 2
        assert all(r["read_only"] is True for r in result)

    def test_filter_by_status(self):
        """Test filtering by status works correctly."""
        from nvidia_tao_core.microservices.utils.filter_utils import filtering

        data = [
            {"name": "job1", "status": "running", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "job2", "status": "done", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "job3", "status": "running", "created_on": datetime.now(tz=timezone.utc)},
        ]

        result = filtering.apply({"status": "running"}, data)
        assert len(result) == 2
        assert all(r["status"] == "running" for r in result)

    def test_filter_by_search(self):
        """Test filtering by search term works correctly."""
        from nvidia_tao_core.microservices.utils.filter_utils import filtering

        data = [
            {"name": "my_experiment", "description": "training model", "id": "123",
             "created_on": datetime.now(tz=timezone.utc)},
            {"name": "another_exp", "description": "testing", "id": "456",
             "created_on": datetime.now(tz=timezone.utc)},
            {"name": "third", "description": "experiment for training", "id": "789",
             "created_on": datetime.now(tz=timezone.utc)},
        ]

        result = filtering.apply({"search": "training"}, data)
        assert len(result) == 2

    def test_sort_by_name_ascending(self):
        """Test sorting by name ascending works correctly."""
        from nvidia_tao_core.microservices.utils.filter_utils import filtering

        data = [
            {"name": "charlie", "version": "1.0", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "alice", "version": "1.0", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "bob", "version": "1.0", "created_on": datetime.now(tz=timezone.utc)},
        ]

        result = filtering.apply({"sort": "name-ascending"}, data)
        assert result[0]["name"] == "alice"
        assert result[1]["name"] == "bob"
        assert result[2]["name"] == "charlie"

    def test_sort_by_name_descending(self):
        """Test sorting by name descending works correctly."""
        from nvidia_tao_core.microservices.utils.filter_utils import filtering

        data = [
            {"name": "alice", "version": "1.0", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "charlie", "version": "1.0", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "bob", "version": "1.0", "created_on": datetime.now(tz=timezone.utc)},
        ]

        result = filtering.apply({"sort": "name-descending"}, data)
        assert result[0]["name"] == "charlie"
        assert result[1]["name"] == "bob"
        assert result[2]["name"] == "alice"

    def test_sort_by_date_ascending(self):
        """Test sorting by date ascending works correctly."""
        from nvidia_tao_core.microservices.utils.filter_utils import filtering

        data = [
            {"name": "exp1", "created_on": datetime(2024, 3, 1, tzinfo=timezone.utc)},
            {"name": "exp2", "created_on": datetime(2024, 1, 1, tzinfo=timezone.utc)},
            {"name": "exp3", "created_on": datetime(2024, 2, 1, tzinfo=timezone.utc)},
        ]

        result = filtering.apply({"sort": "date-ascending"}, data)
        assert result[0]["name"] == "exp2"
        assert result[1]["name"] == "exp3"
        assert result[2]["name"] == "exp1"

    def test_sort_by_date_descending(self):
        """Test sorting by date descending works correctly (default sort)."""
        from nvidia_tao_core.microservices.utils.filter_utils import filtering

        data = [
            {"name": "exp1", "created_on": datetime(2024, 1, 1, tzinfo=timezone.utc)},
            {"name": "exp2", "created_on": datetime(2024, 3, 1, tzinfo=timezone.utc)},
            {"name": "exp3", "created_on": datetime(2024, 2, 1, tzinfo=timezone.utc)},
        ]

        result = filtering.apply({"sort": "date-descending"}, data)
        assert result[0]["name"] == "exp2"
        assert result[1]["name"] == "exp3"
        assert result[2]["name"] == "exp1"

    def test_filter_by_tags(self):
        """Test filtering by tags works correctly."""
        from nvidia_tao_core.microservices.utils.filter_utils import filtering

        data = [
            {"name": "exp1", "tags": ["production", "v1"], "created_on": datetime.now(tz=timezone.utc)},
            {"name": "exp2", "tags": ["development"], "created_on": datetime.now(tz=timezone.utc)},
            {"name": "exp3", "tags": ["production", "v2"], "created_on": datetime.now(tz=timezone.utc)},
        ]

        result = filtering.apply({"tags": "production"}, data)
        assert len(result) == 2
        assert all("production" in r["tags"] for r in result)

    def test_empty_filter_returns_all(self):
        """Test that empty filter returns all data (sorted by date descending)."""
        from nvidia_tao_core.microservices.utils.filter_utils import filtering

        data = [
            {"name": "exp1", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "exp2", "created_on": datetime.now(tz=timezone.utc)},
            {"name": "exp3", "created_on": datetime.now(tz=timezone.utc)},
        ]

        result = filtering.apply({}, data)
        assert len(result) == 3

    def test_combined_filters(self):
        """Test combining multiple filters works correctly."""
        from nvidia_tao_core.microservices.utils.filter_utils import filtering

        data = [
            {"name": "exp1", "type": "object_detection", "status": "done",
             "created_on": datetime.now(tz=timezone.utc)},
            {"name": "exp2", "type": "segmentation", "status": "done",
             "created_on": datetime.now(tz=timezone.utc)},
            {"name": "exp3", "type": "object_detection", "status": "running",
             "created_on": datetime.now(tz=timezone.utc)},
            {"name": "exp4", "type": "object_detection", "status": "done",
             "created_on": datetime.now(tz=timezone.utc)},
        ]

        result = filtering.apply({"type": "object_detection", "status": "done"}, data)
        assert len(result) == 2
        assert all(r["type"] == "object_detection" and r["status"] == "done" for r in result)


class TestPaginationApply:
    """Tests for pagination.apply function."""

    def test_pagination_skip_and_size(self):
        """Test pagination with skip and size works correctly."""
        from nvidia_tao_core.microservices.utils.filter_utils import pagination

        data = [{"id": i} for i in range(10)]

        result = pagination.apply({"skip": "2", "size": "3"}, data)
        assert len(result) == 3
        assert result[0]["id"] == 2
        assert result[1]["id"] == 3
        assert result[2]["id"] == 4

    def test_pagination_first_page(self):
        """Test pagination for first page works correctly."""
        from nvidia_tao_core.microservices.utils.filter_utils import pagination

        data = [{"id": i} for i in range(10)]

        result = pagination.apply({"skip": "0", "size": "5"}, data)
        assert len(result) == 5
        assert result[0]["id"] == 0
        assert result[4]["id"] == 4

    def test_pagination_no_params_returns_all(self):
        """Test pagination without params returns all data."""
        from nvidia_tao_core.microservices.utils.filter_utils import pagination

        data = [{"id": i} for i in range(10)]

        result = pagination.apply({}, data)
        assert len(result) == 10

    def test_pagination_beyond_data_length(self):
        """Test pagination beyond data length returns empty."""
        from nvidia_tao_core.microservices.utils.filter_utils import pagination

        data = [{"id": i} for i in range(5)]

        result = pagination.apply({"skip": "10", "size": "5"}, data)
        assert len(result) == 0
