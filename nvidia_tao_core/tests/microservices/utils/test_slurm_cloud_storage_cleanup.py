# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure propagation tests for checkpoint deletion over Slurm SSH."""

import pytest

from nvidia_tao_core.microservices.utils.slurm_cloud_storage import (
    SlurmCloudStorageAdapter,
)


@pytest.mark.parametrize(
    ("method_name", "remote_path"),
    (("delete_folder", "trial/checkpoints"), ("delete_file", "trial/model.pth")),
)
def test_slurm_checkpoint_delete_propagates_remote_rm_failure(
    method_name, remote_path
):
    """Controller cleanup must see a failed remote rm as an exception."""
    adapter = SlurmCloudStorageAdapter.__new__(SlurmCloudStorageAdapter)
    adapter._get_full_path = lambda path: f"/results/{path}"
    adapter._run_ssh_command = lambda command: (
        False,
        "",
        "Permission denied",
    )

    undecorated_method = getattr(
        SlurmCloudStorageAdapter, method_name
    ).__wrapped__
    with pytest.raises(RuntimeError, match="Permission denied"):
        undecorated_method(adapter, remote_path)


@pytest.mark.parametrize(
    ("method_name", "remote_path"),
    (("delete_folder", "trial/checkpoints"), ("delete_file", "trial/model.pth")),
)
def test_slurm_checkpoint_delete_acknowledges_success(method_name, remote_path):
    """A zero-status remote rm provides an explicit cleanup acknowledgement."""
    adapter = SlurmCloudStorageAdapter.__new__(SlurmCloudStorageAdapter)
    adapter._get_full_path = lambda path: f"/results/{path}"
    adapter._run_ssh_command = lambda command: (True, "", "")

    undecorated_method = getattr(
        SlurmCloudStorageAdapter, method_name
    ).__wrapped__
    assert undecorated_method(adapter, remote_path) is True
