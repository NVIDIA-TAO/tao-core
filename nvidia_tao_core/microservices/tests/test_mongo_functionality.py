# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example tests demonstrating MongoDB functionality testing.

This module shows how to write tests that actually test MongoDB operations
without requiring Docker or Kubernetes. The tests use mongomock which provides
a full MongoDB API implementation in pure Python.

Run these tests with:
    pytest test_mongo_functionality.py -v
"""

import pytest
from datetime import datetime


def test_mongo_handler_initialization(mongo_handler):
    """Test that MongoHandler can be initialized in test mode."""
    print("\n✓ MongoDB test running with mongomock (no Docker/K8s required)")
    print(f"  Client type: {type(mongo_handler.mongo_client).__name__}")
    assert mongo_handler is not None
    assert mongo_handler.collection is not None
    assert mongo_handler.db is not None
    print("  MongoHandler initialized successfully")


def test_upsert_operation(mongo_handler):
    """Test basic upsert operation."""
    print("\n✓ Testing MongoDB upsert operation")
    # Insert a document
    test_data = {
        "id": "test_job_001",
        "name": "Training Job",
        "status": "running"
    }

    mongo_handler.upsert({"id": "test_job_001"}, test_data)
    print(f"  Inserted data: {test_data}")

    # Verify it was inserted
    result = mongo_handler.find_one({"id": "test_job_001"})
    print(f"  Retrieved data: {{'id': '{result['id']}', 'name': '{result['name']}', 'status': '{result['status']}'}}")
    assert result is not None
    assert result["name"] == "Training Job"
    assert result["status"] == "running"
    print("  ✓ Data verified: Storage and retrieval working correctly")


def test_upsert_update(mongo_handler):
    """Test that upsert updates existing documents."""
    # Insert initial document
    mongo_handler.upsert(
        {"id": "job_001"},
        {"id": "job_001", "status": "pending", "progress": 0}
    )

    # Update the document
    mongo_handler.upsert(
        {"id": "job_001"},
        {"id": "job_001", "status": "running", "progress": 50}
    )

    # Verify the update
    result = mongo_handler.find_one({"id": "job_001"})
    assert result["status"] == "running"
    assert result["progress"] == 50


def test_find_multiple_documents(mongo_handler):
    """Test finding multiple documents."""
    # Insert multiple documents
    for i in range(5):
        mongo_handler.upsert(
            {"id": f"job_{i}"},
            {"id": f"job_{i}", "type": "training", "epoch": i}
        )

    # Find all training jobs
    results = mongo_handler.find({"type": "training"})
    assert len(results) == 5


def test_find_with_filter(mongo_handler):
    """Test finding documents with complex filters."""
    print("\n✓ Testing MongoDB filtering/query operations")
    # Insert documents with different statuses
    statuses = ["pending", "running", "completed", "failed", "running"]
    print(f"  Inserting {len(statuses)} jobs with statuses: {statuses}")

    for i, status in enumerate(statuses):
        mongo_handler.upsert(
            {"id": f"job_{i}"},
            {"id": f"job_{i}", "status": status}
        )

    # Find only running jobs
    running_jobs = mongo_handler.find({"status": "running"})
    print("\n  Query: Find jobs with status='running'")
    print(f"  Results found: {len(running_jobs)} jobs")
    for job in running_jobs:
        print(f"    - {job['id']}: status={job['status']}")

    assert len(running_jobs) == 2
    print("  ✓ Filter query working correctly")
    assert all(job["status"] == "running" for job in running_jobs)


def test_delete_operations(mongo_handler):
    """Test delete operations."""
    # Insert test documents
    for i in range(3):
        mongo_handler.upsert(
            {"id": f"job_{i}"},
            {"id": f"job_{i}", "temp": True}
        )

    # Delete one document
    mongo_handler.delete_one({"id": "job_1"})
    result = mongo_handler.find_one({"id": "job_1"})
    assert result == {}

    # Verify others still exist
    results = mongo_handler.find({"temp": True})
    assert len(results) == 2


def test_delete_many(mongo_handler):
    """Test deleting multiple documents."""
    # Insert documents
    for i in range(5):
        mongo_handler.upsert(
            {"id": f"job_{i}"},
            {"id": f"job_{i}", "to_delete": i < 3}
        )

    # Delete multiple documents
    mongo_handler.delete_many({"to_delete": True})

    # Verify deletion
    remaining = mongo_handler.find({})
    assert len(remaining) == 2
    assert all(not doc.get("to_delete", False) for doc in remaining)


def test_upsert_append(mongo_handler):
    """Test upsert_append operation for status arrays."""
    job_id = "job_with_status_history"

    # Insert with first status
    mongo_handler.upsert_append(
        {"id": job_id},
        {"timestamp": datetime.utcnow().isoformat(), "status": "created"}
    )

    # Append more statuses
    mongo_handler.upsert_append(
        {"id": job_id},
        {"timestamp": datetime.utcnow().isoformat(), "status": "running"}
    )

    mongo_handler.upsert_append(
        {"id": job_id},
        {"timestamp": datetime.utcnow().isoformat(), "status": "completed"}
    )

    # Verify status array
    result = mongo_handler.find_one({"id": job_id})
    assert "status" in result
    assert len(result["status"]) == 3
    assert result["status"][0]["status"] == "created"
    assert result["status"][2]["status"] == "completed"


def test_save_job_metadata(mongo_handler, sample_job_metadata):
    """Test saving job metadata - a real-world use case."""
    print("\n✓ Testing real-world use case: saving job metadata")
    print(f"  Storing job: {sample_job_metadata['name']} ({sample_job_metadata['network']})")
    print("  Input data: {")
    print(f"    'id': '{sample_job_metadata['id']}',")
    print(f"    'action': '{sample_job_metadata['action']}',")
    print(f"    'status': '{sample_job_metadata['status']}',")
    print(f"    'specs': {sample_job_metadata['specs']}")
    print("  }")

    # Save job metadata
    mongo_handler.upsert(
        {"id": sample_job_metadata["id"]},
        sample_job_metadata
    )

    # Retrieve and verify
    saved_job = mongo_handler.find_one({"id": sample_job_metadata["id"]})
    print("\n  Retrieved job data:")
    print(f"    Name: {saved_job['name']}")
    print(f"    Network: {saved_job['network']}")
    print(f"    Status: {saved_job['status']}")
    print(f"    Epochs: {saved_job['specs']['num_epochs']}")
    print(f"    Batch size: {saved_job['specs']['batch_size']}")

    assert saved_job["name"] == sample_job_metadata["name"]
    assert saved_job["network"] == sample_job_metadata["network"]
    assert saved_job["specs"]["num_epochs"] == 10
    print("  ✓ All fields match - job metadata stored and retrieved successfully")


def test_update_job_status(mongo_handler, sample_job_metadata):
    """Test updating job status - a common workflow operation."""
    # Save initial job
    mongo_handler.upsert(
        {"id": sample_job_metadata["id"]},
        sample_job_metadata
    )

    # Update status to running
    mongo_handler.upsert(
        {"id": sample_job_metadata["id"]},
        {"status": "running", "progress": 0.5}
    )

    # Verify update
    job = mongo_handler.find_one({"id": sample_job_metadata["id"]})
    assert job["status"] == "running"
    assert job["progress"] == 0.5

    # Original data should still be there
    assert job["name"] == sample_job_metadata["name"]


def test_multiple_collections(mongo_client):
    """Test working with multiple collections."""
    from nvidia_tao_core.microservices.utils.mongo_utils import MongoHandler

    # Create handlers for different collections
    jobs_handler = MongoHandler("tao", "jobs")
    experiments_handler = MongoHandler("tao", "experiments")

    # Insert data into both
    jobs_handler.upsert({"id": "job1"}, {"id": "job1", "type": "training"})
    experiments_handler.upsert({"id": "exp1"}, {"id": "exp1", "type": "detection"})

    # Verify isolation
    jobs = jobs_handler.find({})
    experiments = experiments_handler.find({})

    assert len(jobs) == 1
    assert len(experiments) == 1
    assert jobs[0]["id"] == "job1"
    assert experiments[0]["id"] == "exp1"


def test_update_many_operation(mongo_handler):
    """Test updating multiple documents at once."""
    # Insert multiple jobs
    for i in range(5):
        mongo_handler.upsert(
            {"id": f"job_{i}"},
            {"id": f"job_{i}", "status": "pending", "priority": "normal"}
        )

    # Update all jobs to high priority
    mongo_handler.update_many(
        {"status": "pending"},
        {"priority": "high"}
    )

    # Verify all were updated
    jobs = mongo_handler.find({"status": "pending"})
    assert len(jobs) == 5
    assert all(job["priority"] == "high" for job in jobs)


def test_find_latest(mongo_handler):
    """Test finding the latest document."""
    import time

    # Insert documents with small delays to ensure different timestamps
    for i in range(3):
        mongo_handler.upsert(
            {"id": f"job_{i}"},
            {"id": f"job_{i}", "created": i}
        )
        time.sleep(0.01)  # Small delay to ensure different _id timestamps

    # Get latest document
    latest = mongo_handler.find_latest()
    assert latest is not None
    # The latest should be the last one inserted
    assert latest["id"] == "job_2"


def test_complex_job_workflow(mongo_handler):
    """Test a complete job workflow with status updates."""
    print("\n✓ Testing complex job workflow (create → run → update → complete)")
    job_id = "complex_job_001"

    # 1. Create job
    initial_data = {
        "id": job_id,
        "name": "ResNet Training",
        "status": "created",
        "progress": 0,
        "epoch": 0,
        "max_epochs": 10
    }
    mongo_handler.upsert({"id": job_id}, initial_data)
    print(f"  Step 1: Job created - {initial_data}")

    # 2. Start job
    mongo_handler.upsert(
        {"id": job_id},
        {"status": "running", "progress": 0.1, "epoch": 1}
    )
    job_state = mongo_handler.find_one({"id": job_id})
    status = job_state['status']
    epoch = job_state['epoch']
    progress = job_state['progress']
    print(f"  Step 2: Job started - status: {status}, epoch: {epoch}, progress: {progress:.1%}")

    # 3. Update progress
    for epoch in range(2, 6):
        mongo_handler.upsert(
            {"id": job_id},
            {"epoch": epoch, "progress": epoch / 10}
        )
    job_state = mongo_handler.find_one({"id": job_id})
    print(f"  Step 3: Progress updates - epoch: {job_state['epoch']}, progress: {job_state['progress']:.1%}")

    # 4. Complete job
    mongo_handler.upsert(
        {"id": job_id},
        {"status": "completed", "progress": 1.0, "epoch": 10}
    )

    # Verify final state
    final_job = mongo_handler.find_one({"id": job_id})
    print("  Step 4: Job completed")
    print("\n  Final job state retrieved from DB:")
    print(f"    Name: {final_job['name']}")
    print(f"    Status: {final_job['status']}")
    print(f"    Progress: {final_job['progress']:.0%}")
    print(f"    Epoch: {final_job['epoch']}/{final_job['max_epochs']}")

    assert final_job["status"] == "completed"
    assert final_job["progress"] == 1.0
    assert final_job["epoch"] == 10
    assert final_job["name"] == "ResNet Training"
    print("  ✓ Workflow completed successfully - all state transitions persisted")


def test_database_isolation_between_tests(mongo_handler):
    """Test that database is clean between tests.

    This test verifies that the reset_test_db fixture works correctly
    by ensuring no data from previous tests exists.
    """
    # This should be empty since database is reset before each test
    all_docs = mongo_handler.find({})
    assert len(all_docs) == 0


if __name__ == "__main__":
    # Allow running tests directly with python
    pytest.main([__file__, "-v"])
