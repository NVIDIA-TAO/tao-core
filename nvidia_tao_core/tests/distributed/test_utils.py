# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from unittest.mock import patch, MagicMock

from nvidia_tao_core.distributed.utils import is_master_node


def test_is_master_node_default():
    """Test is_master_node returns True by default when no distributed setup exists."""
    assert is_master_node() is True


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_is_master_node_rank_env(rank):
    """Test is_master_node with RANK environment variable."""
    with patch.dict(os.environ, {"RANK": str(rank)}):
        assert is_master_node() is False

    with patch.dict(os.environ, {"RANK": "0"}):
        assert is_master_node() is True


@pytest.mark.parametrize("node_rank", [1, 2, 3])
def test_is_master_node_node_rank_env(node_rank):
    """Test is_master_node with NODE_RANK environment variable."""
    with patch.dict(os.environ, {"NODE_RANK": str(node_rank)}):
        assert is_master_node() is False

    with patch.dict(os.environ, {"NODE_RANK": "0"}):
        assert is_master_node() is True


def test_is_master_node_pytorch():
    """Test is_master_node with PyTorch distributed setup."""
    pytest.importorskip("torch", reason="PyTorch not installed")

    with patch("torch.distributed") as mock_dist:
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True

        # Test non-master node
        mock_dist.get_rank.return_value = 1
        assert is_master_node() is False

        # Test master node
        mock_dist.get_rank.return_value = 0
        assert is_master_node() is True


def test_is_master_node_mpi():
    """Test is_master_node with MPI setup."""
    mock_comm = MagicMock()

    with patch.dict('sys.modules', {'mpi4py': MagicMock()}):
        from mpi4py import MPI
        MPI.COMM_WORLD = mock_comm

        # Test non-master node
        mock_comm.Get_rank.return_value = 1
        assert is_master_node() is False

        # Test master node
        mock_comm.Get_rank.return_value = 0
        assert is_master_node() is True
