# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed utilities."""

import os


def is_master_node():
    """Check if the current node is the master node.

    This function handles both PyTorch Lightning,
    PyTorch native, and MPI distributed training cases.

    Returns:
        True if the current node is the master node, False otherwise.
    """
    # Check environment variables
    if "RANK" in os.environ:
        return int(os.environ.get("RANK", -1)) == 0

    if "NODE_RANK" in os.environ:
        return int(os.environ.get("NODE_RANK", -1)) == 0

    # Check PyTorch Lightning case
    try:
        from pytorch_lightning.utilities.rank_zero import rank_zero_only
        if getattr(rank_zero_only, "rank", None) not in [None, 0]:
            return False
    except ImportError:
        pass

    # Check PyTorch distributed case
    try:
        import torch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return False
    except ImportError:
        pass

    # Check TensorFlow distributed case
    try:
        import tensorflow as tf
        if tf.distribute.has_strategy():
            strategy = tf.distribute.get_strategy()
            if hasattr(strategy, 'cluster_resolver'):
                task_id = strategy.cluster_resolver.task_id
                if task_id != 0:
                    return False
    except ImportError:
        pass

    # Check MPI case
    try:
        from mpi4py import MPI
        if MPI.Is_initialized():
            comm = MPI.COMM_WORLD
            if comm.Get_rank() != 0:
                return False
    except ImportError:
        pass

    return True
