# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed decorators."""

from functools import wraps

from nvidia_tao_core.distributed.utils import is_master_node


def master_node_only(fn):
    """Decorator to ensure function only runs on master node.

    This decorator handles both PyTorch Lightning (NODE_RANK),
    PyTorch DDP/torchrun (RANK), and MPI (comm.Get_rank()) distributed training cases.

    Returns:
        The decorated function that only executes on rank zero.
    """
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if not is_master_node():
            return None

        return fn(*args, **kwargs)
    return wrapped_fn
