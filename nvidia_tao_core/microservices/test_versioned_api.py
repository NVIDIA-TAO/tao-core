#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test suite for the versioned API structure."""

import os
import unittest
import json
import signal
from unittest.mock import MagicMock

# Set BACKEND to None to prevent MongoHandler import in basic_utils
os.environ.pop('BACKEND', None)


# Create a mock MongoHandler class
class MockMongoHandler:
    """Mock MongoHandler for testing."""
    def __init__(self, *args, **kwargs):
        self.mock = MagicMock()

    def __getattr__(self, name):
        return getattr(self.mock, name)

    def find(self, *args, **kwargs):
        return []

    def find_one(self, *args, **kwargs):
        return None

    def upsert(self, *args, **kwargs):
        return MagicMock()

    def delete_many(self, *args, **kwargs):
        return MagicMock()


# Inject mock into modules BEFORE importing app
# Note: These imports must be after MockMongoHandler to avoid MongoDB connection attempts
from nvidia_tao_core.microservices.utils import basic_utils  # noqa: E402

basic_utils.MongoHandler = MockMongoHandler

# Now import the app
from nvidia_tao_core.microservices.app import app  # noqa: E402
from nvidia_tao_core.microservices.api_versions import api_version_manager  # noqa: E402


class TimeoutException(Exception):
    """Exception raised when a request times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Request timed out")


class TestVersionedAPI(unittest.TestCase):
    """Test cases for the versioned Flask API."""

    def setUp(self):
        """Set up test fixtures."""
        app.config['TESTING'] = True
        self.client = app.test_client()

    def make_request_with_timeout(self, method, url, timeout=5, **kwargs):
        """Make a request with a timeout.

        Args:
            method: HTTP method ('get', 'post', etc.)
            url: URL to request
            timeout: Timeout in seconds
            **kwargs: Additional arguments to pass to the request

        Returns:
            Response object
        """
        # Set up the timeout alarm
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            # Make the request
            request_func = getattr(self.client, method)
            response = request_func(url, **kwargs)
            signal.alarm(0)  # Cancel the alarm
            return response
        except TimeoutException:
            signal.alarm(0)  # Cancel the alarm
            # Return a mock timeout response

            class TimeoutResponse:
                status_code = 408
                data = json.dumps({"error": "Request timed out"}).encode()

            return TimeoutResponse()

    def test_app_creation(self):
        """Test that the versioned app can be created successfully."""
        self.assertIsNotNone(app)
        self.assertTrue(app.config['TESTING'])

    def test_version_discovery_endpoint(self):
        """Test the version discovery endpoint."""
        response = self.client.get('/api/')
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertIn('v1', data)
        self.assertIn('v2', data)

    def test_version_manager(self):
        """Test the API version manager functionality."""
        versions = api_version_manager.get_available_versions()
        self.assertIn('v1', versions)

        v1_info = api_version_manager.get_version_info('v1')
        self.assertIn('blueprints', v1_info)
        self.assertIn('description', v1_info)
        self.assertIn('enabled', v1_info)

    def test_v1_endpoints_accessible(self):
        """Test that v1 endpoints are accessible."""
        # Test v1 health endpoint
        response = self.make_request_with_timeout('get', '/api/v1/health', timeout=5)
        self.assertNotEqual(response.status_code, 404)

        # Test v1 auth endpoint (will fail auth but should reach endpoint)
        response = self.make_request_with_timeout('post', '/api/v1/login', timeout=5)
        self.assertNotEqual(response.status_code, 404)

    def test_v1_workspace_endpoints(self):
        """Test that v1 workspace endpoints are accessible."""
        # Test workspace list endpoint
        response = self.make_request_with_timeout('get', '/api/v1/orgs/test-org/workspaces', timeout=5)
        self.assertNotEqual(response.status_code, 404)
        # Should be valid response (either placeholder with 'message' or actual data with 'workspaces')
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertTrue(
                'message' in data or 'workspaces' in data,
                f"Expected 'message' or 'workspaces' in response, got: {data}"
            )

    def test_v1_dataset_endpoints(self):
        """Test that v1 dataset endpoints are accessible."""
        # Test dataset list endpoint (placeholder)
        response = self.make_request_with_timeout('get', '/api/v1/orgs/test-org/datasets', timeout=5)
        self.assertNotEqual(response.status_code, 404)

    def test_v1_experiment_endpoints(self):
        """Test that v1 experiment endpoints are accessible."""
        # Test experiment list endpoint (placeholder)
        response = self.make_request_with_timeout('get', '/api/v1/orgs/test-org/experiments', timeout=5)
        self.assertNotEqual(response.status_code, 404)

    def test_admin_endpoints_non_versioned(self):
        """Test that admin endpoints work without versioning."""
        # Test root endpoint
        response = self.make_request_with_timeout('get', '/', timeout=5)
        self.assertEqual(response.status_code, 200)

        # Test api versions endpoint
        response = self.make_request_with_timeout('get', '/api/', timeout=5)
        self.assertEqual(response.status_code, 200)

        versions = json.loads(response.data)

        # Test OpenAPI endpoints
        for version in versions:
            response = self.make_request_with_timeout('get', f"/api/{version}/openapi.json", timeout=5)
            self.assertEqual(response.status_code, 200)

    def test_v2_example_endpoints(self):
        """Test that v2 example endpoints are accessible (if available)."""
        # Note: v2 is not fully implemented yet, but test if endpoints exist
        response = self.make_request_with_timeout('post', '/api/v2/login', timeout=5)
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn('version', data)
            self.assertEqual(data['version'], '2.0')

    def test_error_handling(self):
        """Test that error handlers work correctly."""
        # Test 404 error handling
        response = self.make_request_with_timeout('get', '/api/v1/nonexistent-endpoint', timeout=5)
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn('code', data)
        self.assertIn('name', data)

    def test_blueprint_registration(self):
        """Test that blueprints are registered correctly."""
        # Get all registered blueprint names
        blueprint_names = [bp.name for bp in app.blueprints.values()]

        # Check that versioned blueprints are registered
        versioned_blueprints = [name for name in blueprint_names if '_v1' in name]
        self.assertGreater(len(versioned_blueprints), 0, "No v1 blueprints found")

        # Check for expected v1 blueprints
        expected_v1_blueprints = ['auth_v1', 'health_v1', 'workspaces_v1', 'datasets_v1',
                                  'experiments_v1', 'internal_v1']
        for expected_bp in expected_v1_blueprints:
            self.assertIn(expected_bp, blueprint_names, f"Blueprint {expected_bp} not registered")

    def test_url_prefix_structure(self):
        """Test that URL prefixes are structured correctly."""
        # Gather all URL rules and check prefixes
        v1_rules = [rule for rule in app.url_map.iter_rules() if rule.rule.startswith('/api/v1/')]
        self.assertGreater(len(v1_rules), 0, "No v1 URL rules found")

        # Check that we have the expected endpoint patterns
        v1_rule_strings = [rule.rule for rule in v1_rules]

        # Should have authentication endpoints
        auth_rules = [rule for rule in v1_rule_strings if '/login' in rule or '/auth' in rule]
        self.assertGreater(len(auth_rules), 0, "No auth endpoints found")

        # Should have health endpoints
        health_rules = [rule for rule in v1_rule_strings if '/health' in rule]
        self.assertGreater(len(health_rules), 0, "No health endpoints found")

    def test_version_comparison_structure(self):
        """Test the structure supports version comparison."""
        # This test verifies that the structure supports adding v2 endpoints
        # alongside v1 without conflicts

        response_v1 = self.make_request_with_timeout('get', '/api/v1/health', timeout=5)
        self.assertNotEqual(response_v1.status_code, 404)

        # If v2 health endpoint exists, it should be separate
        response_v2 = self.make_request_with_timeout('get', '/api/v2/health', timeout=5)
        # Either 404 (not implemented) or 200 (implemented differently)
        self.assertIn(response_v2.status_code, [404, 200, 408])  # Include 408 for timeout


class TestAPIVersionManager(unittest.TestCase):
    """Test the API Version Manager separately."""

    def test_version_registration(self):
        """Test version registration functionality."""
        from nvidia_tao_core.microservices.api_versions import APIVersionManager
        from flask import Blueprint

        # Create a test version manager
        test_manager = APIVersionManager()

        # Create a test blueprint
        test_bp = Blueprint('test', __name__)

        # Register a test version
        test_manager.register_version('v3', [(test_bp, '')], 'Test version')

        # Verify registration
        self.assertIn('v3', test_manager.get_available_versions())
        v3_info = test_manager.get_version_info('v3')
        self.assertEqual(v3_info['description'], 'Test version')
        self.assertTrue(v3_info['enabled'])

    def test_version_enable_disable(self):
        """Test enabling and disabling versions."""
        from nvidia_tao_core.microservices.api_versions import APIVersionManager
        from flask import Blueprint

        test_manager = APIVersionManager()
        test_bp = Blueprint('test', __name__)
        test_manager.register_version('v4', [(test_bp, '')], 'Test version')

        # Test disable
        test_manager.disable_version('v4')
        v4_info = test_manager.get_version_info('v4')
        self.assertFalse(v4_info['enabled'])

        # Test enable
        test_manager.enable_version('v4')
        v4_info = test_manager.get_version_info('v4')
        self.assertTrue(v4_info['enabled'])


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
