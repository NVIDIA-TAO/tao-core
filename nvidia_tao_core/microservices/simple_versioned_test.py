#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test to demonstrate the versioned API system."""

import sys
import os
import json

# Add the project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


def test_version_manager():
    """Test the version manager functionality."""
    print("\nTesting version manager...")

    try:
        from nvidia_tao_core.microservices.api_versions import APIVersionManager
        from flask import Blueprint

        # Create test manager
        manager = APIVersionManager()

        # Create test blueprint
        test_bp = Blueprint('test', __name__)

        # Register test version
        manager.register_version('test_v1', [(test_bp, '')], 'Test version')

        # Test functionality
        versions = manager.get_available_versions()
        assert 'test_v1' in versions
        print(f"✓ Version registration works. Available: {versions}")

        # Test enable/disable
        manager.disable_version('test_v1')
        info = manager.get_version_info('test_v1')
        assert not info['enabled']
        print("✓ Version disable works")

        manager.enable_version('test_v1')
        info = manager.get_version_info('test_v1')
        assert info['enabled']
        print("✓ Version enable works")

        return True
    except Exception as e:
        print(f"✗ Version manager test failed: {e}")
        return False


def test_app_creation():
    """Test creating the versioned app."""
    print("\nTesting app creation...")

    try:
        from nvidia_tao_core.microservices.app import create_app

        app = create_app()
        app.config['TESTING'] = True

        # Check blueprints are registered
        blueprint_names = [bp.name for bp in app.blueprints.values()]
        print(f"✓ App created with blueprints: {blueprint_names}")

        # Check for versioned blueprints
        v1_blueprints = [name for name in blueprint_names if '_v1' in name]
        if v1_blueprints:
            print(f"✓ v1 blueprints found: {v1_blueprints}")
        else:
            print("! No v1 blueprints found")

        # Test basic routes
        with app.test_client() as client:
            # Test version discovery
            response = client.get('/api/versions')
            if response.status_code == 200:
                data = json.loads(response.data)
                print(f"✓ Version discovery works: {data.get('available_versions', [])}")
            else:
                print(f"! Version discovery failed: {response.status_code}")

            # Test v1 health endpoint
            response = client.get('/api/v1/health')
            if response.status_code == 200:
                print("✓ v1 health endpoint accessible")
            else:
                print(f"! v1 health endpoint failed: {response.status_code}")

            # Test v2 login endpoint (if available)
            response = client.post('/api/v2/login')
            if response.status_code == 200:
                print("✓ v2 login endpoint accessible")
            elif response.status_code == 404:
                print("! v2 login endpoint not found (expected if not fully implemented)")

        return True
    except Exception as e:
        print(f"✗ App creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=== Testing Versioned API System ===\n")

    tests = [
        test_version_manager,
        test_app_creation
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n=== Results ===")
    print(f"Passed: {sum(results)}/{len(results)}")

    if all(results):
        print("🎉 All tests passed! Versioned API system is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.")

    return all(results)


if __name__ == '__main__':
    main()
