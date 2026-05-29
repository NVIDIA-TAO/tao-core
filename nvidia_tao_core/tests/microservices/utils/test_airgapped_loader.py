# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the airgapped experiment loader flow.

Tests the full round-trip: login -> create workspace -> load airgapped PTMs
-> list base experiments and verify they appear.

Requires a running TAO API server. Set environment variables before running:
    export NODE_ADDRESS=localhost
    export NODE_PORT=8090
    pytest test_airgapped_loader.py -v

The tests will be skipped if NODE_ADDRESS or NODE_PORT are not set.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'tao-client'))

NODE_ADDRESS = os.environ.get("NODE_ADDRESS")
NODE_PORT = os.environ.get("NODE_PORT")
SKIP_REASON = "NODE_ADDRESS and NODE_PORT environment variables must be set to run integration tests"

pytestmark = pytest.mark.skipif(
    not NODE_ADDRESS or not NODE_PORT,
    reason=SKIP_REASON
)


@pytest.fixture(scope="module")
def base_url():
    return f"http://{NODE_ADDRESS}:{NODE_PORT}/api/v2"


@pytest.fixture(scope="module")
def tao_client(base_url):
    """Initialize and login TaoClient against the live server."""
    from tao_sdk.client import TaoClient

    client = TaoClient(
        base_url=base_url,
        org_name="nvstaging",
    )

    secrets_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', '..', '..',
        'ngc-collaterals', 'cv', 'setup', 'tao-docker-compose', 'secrets.json'
    )
    if os.path.isfile(secrets_path):
        with open(secrets_path) as f:
            secrets = json.load(f)
        ngc_key = secrets.get("ngc_api_key", "")
    else:
        ngc_key = os.environ.get("NGC_KEY", "")

    login_resp = client.login(
        ngc_key=ngc_key,
        ngc_org_name="nvstaging",
        enable_telemetry=False,
    )
    assert login_resp is not None, "Login failed"
    return client


@pytest.fixture(scope="module")
def workspace_id(tao_client):
    """Create a SeaweedFS workspace for the airgapped loader to use."""
    resp = tao_client.create_workspace(
        name="test_airgapped_workspace",
        cloud_type="seaweedfs",
        cloud_specific_details={
            "cloud_type": "seaweedfs",
            "cloud_region": "us-east-1",
            "cloud_bucket_name": "tao-storage",
            "access_key": "seaweedfs",
            "secret_key": "seaweedfs123",
            "endpoint_url": "http://seaweedfs-s3:8333",
        },
    )
    ws_id = resp["id"]
    assert ws_id, "Workspace creation returned no id"
    return ws_id


class TestAirgappedLoadAndList:
    """End-to-end: load airgapped PTMs then list them via the API."""

    def test_load_airgapped_model(self, tao_client, workspace_id):
        """POST jobs:load_airgapped should succeed and report experiments loaded."""
        result = tao_client.load_airgapped_model(
            model_data={"workspace_id": workspace_id}
        )

        assert result.get("success") is True, (
            f"load_airgapped_model failed: {result}"
        )
        assert result.get("experiments_loaded", 0) >= 1, (
            f"Expected at least 1 experiment loaded, got: {result}"
        )

    def test_list_base_experiments_returns_loaded_ptm(self, tao_client, workspace_id):
        """After loading, GET jobs:list_base_experiments must return the PTM.

        This test fails without both fixes:
        - Loader must write to 'jobs' collection (not 'experiments')
        - Loader must set public=True on each document
        """
        # Ensure experiments are loaded first
        tao_client.load_airgapped_model(
            model_data={"workspace_id": workspace_id}
        )

        base_experiments = tao_client.list_base_experiments()
        assert len(base_experiments) >= 1, (
            "No base experiments returned. "
            "Loader likely wrote to wrong collection or missing public=True flag."
        )

    def test_list_base_experiments_filter_by_network_arch(self, tao_client, workspace_id):
        """Filtering by network_arch=dino should return the loaded DINO PTM.

        This is the exact flow that was broken before the fix.
        """
        tao_client.load_airgapped_model(
            model_data={"workspace_id": workspace_id}
        )

        dino_experiments = tao_client.list_base_experiments(
            filter_params={"network_arch": "dino"}
        )
        assert len(dino_experiments) >= 1, (
            "No DINO experiments found after airgapped load. "
            "Loader must write to 'jobs' collection with public=True."
        )

        exp = dino_experiments[0]
        assert exp.get("network_arch") == "dino"
        assert "ngc_path" in exp
        assert "name" in exp
        print(f"Found DINO PTM: {exp.get('name')} (ngc_path: {exp.get('ngc_path')})")

    def test_loaded_experiment_has_expected_fields(self, tao_client, workspace_id):
        """The loaded PTM should have all fields needed for downstream job creation."""
        tao_client.load_airgapped_model(
            model_data={"workspace_id": workspace_id}
        )

        experiments = tao_client.list_base_experiments(
            filter_params={"network_arch": "dino"}
        )
        assert len(experiments) >= 1

        exp = experiments[0]
        required_fields = ["id", "name", "network_arch", "ngc_path", "actions"]
        missing = [f for f in required_fields if f not in exp]
        assert not missing, (
            f"Loaded PTM missing required fields: {missing}. "
            f"Available fields: {list(exp.keys())}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
