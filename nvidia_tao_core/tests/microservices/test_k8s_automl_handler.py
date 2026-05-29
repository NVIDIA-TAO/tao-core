# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for K8s and AutoML handler fixes.

Covers Backend enum .value usage, K8s job creation error propagation,
AutoML start/resume error handling with status updates,
and job/experiment handler error status management.
"""
import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


class TestBackendEnumValueUsage(unittest.TestCase):
    """Bug: KubernetesHandler.create_job and DeploymentExecutor.create_deployment
    used the Backend enum object directly (BACKEND / self.backend) instead of
    its .value string when setting the BACKEND env var. The container received
    'Backend.LOCAL_K8S' instead of 'local-k8s'."""

    def test_kubernetes_handler_create_job_uses_backend_value(self):
        from nvidia_tao_core.microservices.handlers.execution_handlers.kubernetes_handler import (
            KubernetesHandler
        )
        source = inspect.getsource(KubernetesHandler.create_job)
        self.assertIn('BACKEND.value', source,
                      "KubernetesHandler.create_job must use BACKEND.value, "
                      "not BACKEND directly, for the BACKEND env var")

    def test_deployment_executor_uses_backend_dot_value(self):
        """DeploymentExecutor must use self.backend.value for the BACKEND env var."""
        src_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..',
            'nvidia_tao_core', 'microservices', 'utils', 'job_utils',
            'executor', 'deployment_executor.py'
        )
        with open(src_path, 'r') as f:
            source = f.read()
        self.assertIn('self.backend.value', source,
                      "DeploymentExecutor.create_deployment must use self.backend.value, "
                      "not self.backend directly, for the BACKEND env var")


class TestKubernetesHandlerRaisesOnFailure(unittest.TestCase):
    """Bug: KubernetesHandler.create_job silently returned on failure
    (bare 'return' in except block), so callers never knew K8s job
    creation failed. Jobs would hang forever in 'Pending' state."""

    def test_create_job_raises_on_exception(self):
        from nvidia_tao_core.microservices.handlers.execution_handlers.kubernetes_handler import (
            KubernetesHandler
        )
        source = inspect.getsource(KubernetesHandler.create_job)
        except_block = source.split('except Exception')[-1]
        self.assertIn('raise', except_block,
                      "create_job except block must re-raise the exception "
                      "instead of silently returning")


class TestAutoMLHandlerErrorHandling(unittest.TestCase):
    """Bug: AutoMLHandler.start and .resume didn't wrap create_job_with_handler
    in try/except, so if K8s job creation failed, the AutoML job was left
    in 'Running' status forever."""

    def test_start_wraps_create_job_in_try_except(self):
        from nvidia_tao_core.microservices.handlers.automl_handler import AutoMLHandler
        source = inspect.getsource(AutoMLHandler.start)
        create_idx = source.find('create_job_with_handler')
        self.assertGreater(create_idx, -1, "start must call create_job_with_handler")
        pre_context = source[:create_idx]
        self.assertIn('try:', pre_context,
                      "start must have a try block wrapping create_job_with_handler")

    def test_start_updates_status_on_failure(self):
        from nvidia_tao_core.microservices.handlers.automl_handler import AutoMLHandler
        source = inspect.getsource(AutoMLHandler.start)
        self.assertIn('update_job_status', source,
                      "start must call update_job_status on failure "
                      "to mark the job as Error")
        self.assertIn('"Error"', source,
                      "start must set status to 'Error' on failure")

    def test_resume_wraps_create_job_in_try_except(self):
        from nvidia_tao_core.microservices.handlers.automl_handler import AutoMLHandler
        source = inspect.getsource(AutoMLHandler.resume)
        create_idx = source.find('create_job_with_handler')
        self.assertGreater(create_idx, -1, "resume must call create_job_with_handler")
        pre_context = source[:create_idx]
        self.assertIn('try:', pre_context,
                      "resume must have a try block wrapping create_job_with_handler")

    def test_resume_updates_status_on_failure(self):
        from nvidia_tao_core.microservices.handlers.automl_handler import AutoMLHandler
        source = inspect.getsource(AutoMLHandler.resume)
        self.assertIn('update_job_status', source,
                      "resume must call update_job_status on failure "
                      "to mark the job as Error")


class TestJobHandlerErrorHandling(unittest.TestCase):
    """Bug: JobHandler.job_run didn't update job status to 'Error' when an
    exception occurred, leaving failed jobs in an indeterminate state."""

    def test_job_run_updates_status_on_failure(self):
        from nvidia_tao_core.microservices.handlers.job_handler import JobHandler
        source = inspect.getsource(JobHandler.job_run)
        self.assertIn('update_job_status', source,
                      "job_run must call update_job_status on exception "
                      "to mark the job as Error")
        self.assertIn('"Error"', source,
                      "job_run must set status to 'Error' on failure")

    def test_job_run_includes_error_details(self):
        from nvidia_tao_core.microservices.handlers.job_handler import JobHandler
        source = inspect.getsource(JobHandler.job_run)
        self.assertIn('write_job_metadata', source,
                      "job_run must write detailed error info to job metadata")


class TestExperimentHandlerResumeErrorHandling(unittest.TestCase):
    """Bug: ExperimentHandler.resume_experiment_job didn't update job status
    to 'Error' on failure, leaving resumed jobs stuck in their previous state."""

    def test_resume_updates_status_on_failure(self):
        from nvidia_tao_core.microservices.handlers.experiment_handler import ExperimentHandler
        source = inspect.getsource(ExperimentHandler.resume_experiment_job)
        self.assertIn('update_job_status', source,
                      "resume_experiment_job must call update_job_status on exception")

    def test_resume_includes_error_in_response(self):
        from nvidia_tao_core.microservices.handlers.experiment_handler import ExperimentHandler
        source = inspect.getsource(ExperimentHandler.resume_experiment_job)
        except_block = source.split('except Exception as e:')[-1]
        self.assertIn('str(e)', except_block,
                      "resume_experiment_job must include the exception message in the response")


class TestBackendEnumValueType(unittest.TestCase):
    """Verify Backend enum .value returns a plain string, not the enum itself."""

    def test_backend_value_is_string(self):
        from nvidia_tao_core.microservices.enum_constants import Backend
        for member in Backend:
            self.assertIsInstance(
                member.value, str,
                f"Backend.{member.name}.value must be a string, "
                f"got {type(member.value)}"
            )

    def test_backend_local_k8s_value(self):
        from nvidia_tao_core.microservices.enum_constants import Backend
        self.assertEqual(Backend.LOCAL_K8S.value, "local-k8s")
        self.assertNotEqual(str(Backend.LOCAL_K8S), "local-k8s",
                            "str(Backend.LOCAL_K8S) returns enum repr, not the value -- "
                            "which is exactly why .value is needed")


if __name__ == '__main__':
    unittest.main()
