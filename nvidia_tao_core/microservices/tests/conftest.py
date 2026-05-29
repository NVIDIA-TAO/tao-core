# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and fixtures for microservices MongoDB testing.

This conftest.py provides fixtures for testing MongoDB functionality without Docker/K8s.
It uses mongomock to simulate MongoDB in-memory.
"""

import pytest
import os

# Check if mongomock is available
try:
    import mongomock  # noqa: F401
    MONGOMOCK_AVAILABLE = True
except ImportError:
    MONGOMOCK_AVAILABLE = False


@pytest.fixture(scope="session")
def test_environment():
    """Set up test environment for the entire test session.

    This fixture sets up the test environment when explicitly requested.
    It ensures:
    - TAO_TEST_MODE is enabled
    - mongomock is used instead of real MongoDB
    - Test database is cleaned up after tests

    Note: This is NOT autouse, so tests must explicitly request it or use
    the mongo_handler/mongo_client fixtures which depend on it.
    """
    if not MONGOMOCK_AVAILABLE:
        pytest.skip("mongomock not installed - skipping MongoDB tests")

    # Setup: Enable test mode and remove BACKEND
    os.environ["TAO_TEST_MODE"] = "true"
    os.environ.pop("BACKEND", None)

    yield

    # Teardown
    os.environ.pop("TAO_TEST_MODE", None)


@pytest.fixture(scope="function")
def reset_test_db(test_environment):
    """Reset test database before each test function.

    Use this fixture when you need a clean database state for your test:

    def test_something(reset_test_db):
        # Database is clean here
        pass

    Note: Depends on test_environment fixture to ensure test mode is enabled.
    """
    from nvidia_tao_core.microservices.utils.mongo_utils import reset_test_db as reset_db
    reset_db()
    yield
    # Optionally reset again after test
    # reset_db()


@pytest.fixture(scope="function")
def mongo_handler(test_environment):
    """Provide a MongoHandler instance for testing.

    Returns a MongoHandler connected to a test database and collection.
    The database is automatically reset before each test.

    Example:
        def test_mongo_operations(mongo_handler):
            mongo_handler.upsert({"id": "test1"}, {"name": "Test"})
            result = mongo_handler.find_one({"id": "test1"})
            assert result["name"] == "Test"

    Note: Depends on test_environment fixture to ensure test mode is enabled.
    """
    from nvidia_tao_core.microservices.utils.mongo_utils import reset_test_db, MongoHandler

    # Reset database before test
    reset_test_db()

    # Create and return handler
    handler = MongoHandler("test_db", "test_collection")
    yield handler

    # Cleanup after test
    reset_test_db()


@pytest.fixture(scope="function")
def mongo_client(test_environment):
    """Provide direct access to the MongoDB client for advanced testing.

    Use this when you need to test multiple databases or collections:

    Example:
        def test_multi_collection(mongo_client):
            db = mongo_client["test_db"]
            collection1 = db["collection1"]
            collection2 = db["collection2"]
            # ... test operations

    Note: Depends on test_environment fixture to ensure test mode is enabled.
    """
    from nvidia_tao_core.microservices.utils.mongo_utils import get_test_client, reset_test_db

    # Reset database before test
    reset_test_db()

    # Get and return client
    client = get_test_client()
    yield client

    # Cleanup after test
    reset_test_db()


@pytest.fixture(scope="function")
def sample_job_metadata():
    """Provide sample job metadata for testing.

    Returns a dictionary with typical job metadata structure.
    """
    return {
        "id": "test-job-001",
        "name": "Test Training Job",
        "network": "resnet18",
        "action": "train",
        "status": "pending",
        "user_id": "test-user",
        "workspace_id": "test-workspace",
        "created_at": "2024-01-01T00:00:00Z",
        "specs": {
            "num_epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001
        }
    }


@pytest.fixture(scope="function")
def sample_experiment_metadata():
    """Provide sample experiment metadata for testing.

    Returns a dictionary with typical experiment metadata structure.
    """
    return {
        "id": "test-exp-001",
        "name": "Test Experiment",
        "network": "efficientdet",
        "type": "object_detection",
        "user_id": "test-user",
        "workspace_id": "test-workspace",
        "created_at": "2024-01-01T00:00:00Z",
        "description": "Test experiment for unit testing"
    }


# Add any additional fixtures needed for your specific tests here
