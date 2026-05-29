# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SLURM and cosmos-rl handler fixes.

Covers remote backend detection, GPU condition check skipping for SLURM,
cosmos-rl evaluate/inference GPU calculation, _to_compact_json serializers,
and None guards in SLURM and execution handlers.
"""
import inspect
import json
import os
import sys
import unittest
from datetime import datetime
from pathlib import PurePosixPath
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


class TestIsRemoteBackend(unittest.TestCase):
    """Bug: No helper existed to determine if a backend is remote (SLURM/Lepton),
    so GPU condition checks couldn't be skipped for remote schedulers."""

    def test_function_exists(self):
        from nvidia_tao_core.microservices.utils.handler_utils import is_remote_backend
        self.assertTrue(callable(is_remote_backend))

    def test_slurm_is_remote(self):
        from nvidia_tao_core.microservices.utils.handler_utils import is_remote_backend
        self.assertTrue(is_remote_backend({'backend_type': 'slurm'}))

    def test_lepton_is_remote(self):
        from nvidia_tao_core.microservices.utils.handler_utils import is_remote_backend
        self.assertTrue(is_remote_backend({'backend_type': 'lepton'}))

    def test_local_k8s_is_not_remote(self):
        from nvidia_tao_core.microservices.utils.handler_utils import is_remote_backend
        self.assertFalse(is_remote_backend({'backend_type': 'local-k8s'}))

    def test_none_is_not_remote(self):
        from nvidia_tao_core.microservices.utils.handler_utils import is_remote_backend
        self.assertFalse(is_remote_backend(None))

    def test_empty_dict_is_not_remote(self):
        from nvidia_tao_core.microservices.utils.handler_utils import is_remote_backend
        self.assertFalse(is_remote_backend({}))


class TestGetNumGpusSkipConditionsCheck(unittest.TestCase):
    """Bug: get_num_gpus_from_spec always ran _check_gpu_conditions which
    validates against NUM_GPU_PER_NODE. For SLURM backends where total GPUs
    may exceed per-node count, this caused spurious failures."""

    def test_skip_gpu_conditions_check_param_exists(self):
        from nvidia_tao_core.microservices.utils.handler_utils import get_num_gpus_from_spec
        sig = inspect.signature(get_num_gpus_from_spec)
        self.assertIn('skip_gpu_conditions_check', sig.parameters,
                      "get_num_gpus_from_spec must accept skip_gpu_conditions_check parameter")

    @patch.dict(os.environ, {"NUM_GPU_PER_NODE": "2"})
    def test_skip_allows_more_gpus_than_per_node(self):
        from nvidia_tao_core.microservices.utils.handler_utils import get_num_gpus_from_spec
        spec = {"num_gpus": 16}
        result = get_num_gpus_from_spec(spec, "train", skip_gpu_conditions_check=True)
        self.assertEqual(result, 16,
                         "With skip_gpu_conditions_check=True, 16 GPUs should be allowed "
                         "even with NUM_GPU_PER_NODE=2")

    @patch.dict(os.environ, {"NUM_GPU_PER_NODE": "2"})
    def test_without_skip_raises_for_too_many_gpus(self):
        from nvidia_tao_core.microservices.utils.handler_utils import get_num_gpus_from_spec
        spec = {"num_gpus": 16}
        with self.assertRaises(ValueError):
            get_num_gpus_from_spec(spec, "train", skip_gpu_conditions_check=False)


class TestCosmosRlEvalInferenceGpuCalc(unittest.TestCase):
    """Bug: cosmos-rl used the training-specific GPU calculation (policy + rollout)
    for ALL actions, including evaluate/inference. But evaluate/inference should
    use the top-level num_gpus from the spec instead."""

    @patch.dict(os.environ, {"NUM_GPU_PER_NODE": "8"})
    def test_evaluate_uses_top_level_num_gpus(self):
        from nvidia_tao_core.microservices.utils.handler_utils import get_num_gpus_from_spec
        spec = {
            "num_gpus": 2,
            "policy": {"parallelism": {"tp_size": 4, "dp_shard_size": 2, "n_init_replicas": 1}},
            "rollout": {"parallelism": {"tp_size": 4, "n_init_replicas": 1}},
            "train": {"train_policy": {"type": "grpo"}},
        }
        result = get_num_gpus_from_spec(spec, "evaluate", network="cosmos-rl")
        self.assertEqual(result, 2,
                         "cosmos-rl evaluate should use top-level num_gpus=2, "
                         "not training GPU calculation")

    @patch.dict(os.environ, {"NUM_GPU_PER_NODE": "8"})
    def test_inference_does_not_use_training_gpu_calc(self):
        """cosmos-rl inference must NOT use GRPO training calc (policy+rollout=12 GPUs).
        It should fall through to standard param detection (gpu_mapper / top-level num_gpus)."""
        from nvidia_tao_core.microservices.utils.handler_utils import get_num_gpus_from_spec
        spec = {
            "num_gpus": 2,
            "policy": {"parallelism": {"tp_size": 4, "dp_shard_size": 2, "n_init_replicas": 1}},
            "rollout": {"parallelism": {"tp_size": 4, "n_init_replicas": 1}},
            "train": {"train_policy": {"type": "grpo"}},
        }
        training_total = (4 * 2 * 1) + (4 * 1)  # policy + rollout = 12
        result = get_num_gpus_from_spec(spec, "inference", network="cosmos-rl")
        self.assertNotEqual(result, training_total,
                            "cosmos-rl inference must NOT use GRPO training GPU calculation")

    @patch.dict(os.environ, {"NUM_GPU_PER_NODE": "32"})
    def test_train_still_uses_cosmos_rl_calculation(self):
        from nvidia_tao_core.microservices.utils.handler_utils import get_num_gpus_from_spec
        spec = {
            "num_gpus": 2,
            "policy": {"parallelism": {"tp_size": 4, "dp_shard_size": 2, "n_init_replicas": 1}},
            "rollout": {"parallelism": {"tp_size": 4, "n_init_replicas": 1}},
            "train": {"train_policy": {"type": "grpo"}},
        }
        result = get_num_gpus_from_spec(spec, "train", network="cosmos-rl")
        expected = (4 * 2 * 1) + (4 * 1)  # policy_gpus + rollout_gpus = 12
        self.assertEqual(result, expected,
                         f"cosmos-rl train should use GRPO GPU calculation ({expected}), not top-level num_gpus")


class TestSlurmToCompactJsonSerializers(unittest.TestCase):
    """Bug: _to_compact_json couldn't serialize numpy scalars, datetime objects,
    bytes, or pathlib.Path -- causing TypeError and 'Error when creating
    microservice pod' for evaluate jobs."""

    def _get_to_compact_json(self):
        from nvidia_tao_core.microservices.handlers.execution_handlers.slurm_handler import (
            SlurmHandler
        )
        return SlurmHandler._to_compact_json

    def test_datetime_serializable(self):
        to_compact = self._get_to_compact_json()
        dt = datetime(2026, 3, 9, 12, 0, 0)
        result = to_compact({"timestamp": dt})
        parsed = json.loads(result)
        self.assertEqual(parsed["timestamp"], "2026-03-09T12:00:00")

    def test_pathlib_serializable(self):
        to_compact = self._get_to_compact_json()
        p = PurePosixPath("/data/models/checkpoint.pt")
        result = to_compact({"path": p})
        parsed = json.loads(result)
        self.assertEqual(parsed["path"], "/data/models/checkpoint.pt")

    def test_bytes_serializable(self):
        to_compact = self._get_to_compact_json()
        result = to_compact({"data": b"hello"})
        parsed = json.loads(result)
        self.assertEqual(parsed["data"], "hello")

    def test_numpy_like_scalar_serializable(self):
        """Test with an object that has .item() method (mimics numpy scalar)."""
        to_compact = self._get_to_compact_json()

        class NumpyLikeScalar:
            def __init__(self, val):
                self._val = val

            def item(self):
                return self._val

        result = to_compact({"value": NumpyLikeScalar(42)})
        parsed = json.loads(result)
        self.assertEqual(parsed["value"], 42)

    def test_plain_dict_still_works(self):
        to_compact = self._get_to_compact_json()
        result = to_compact({"key": "value", "num": 123})
        self.assertEqual(result, '{"key":"value","num":123}')


class TestSlurmSubmitJobNoneGuards(unittest.TestCase):
    """Bug: slurm_handler.submit_job crashed with AttributeError when specs,
    docker_env_vars, or cloud_metadata was None."""

    def test_create_job_has_none_guards(self):
        from nvidia_tao_core.microservices.handlers.execution_handlers.slurm_handler import (
            SlurmHandler
        )
        source = inspect.getsource(SlurmHandler.create_job)
        self.assertIn('if specs is None', source,
                      "create_job must guard against specs=None")
        self.assertIn('if docker_env_vars is None', source,
                      "create_job must guard against docker_env_vars=None")
        self.assertIn('if cloud_metadata is None', source,
                      "create_job must guard against cloud_metadata=None")


class TestExecutionHandlerNoneGuards(unittest.TestCase):
    """Bug: create_microservice_and_send_request crashed when docker_env_vars
    or backend_details was None for SLURM/Lepton backends."""

    def test_docker_env_vars_none_guard(self):
        from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import (
            ExecutionHandler
        )
        source = inspect.getsource(ExecutionHandler.create_microservice_and_send_request)
        self.assertIn('if docker_env_vars is None', source,
                      "create_microservice_and_send_request must guard against docker_env_vars=None")

    def test_backend_details_none_guard(self):
        from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import (
            ExecutionHandler
        )
        source = inspect.getsource(ExecutionHandler.create_microservice_and_send_request)
        self.assertIn('if backend_details is None', source,
                      "create_microservice_and_send_request must guard against backend_details=None")


if __name__ == '__main__':
    unittest.main()
