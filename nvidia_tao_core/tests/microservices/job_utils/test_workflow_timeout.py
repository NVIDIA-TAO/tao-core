# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for job timeout monitoring feature in workflow.py"""

import os
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

from nvidia_tao_core.microservices.utils.job_utils.timeout_monitor import (
    get_last_status_timestamp,
    check_job_timeout,
    terminate_timed_out_job
)
from nvidia_tao_core.microservices.utils.job_utils.workflow import (
    check_for_timed_out_jobs
)
from nvidia_tao_core.microservices.utils.timeout_utils import (
    get_all_running_jobs,
    get_all_running_automl_experiments,
)


class TestGetLastStatusTimestamp:
    """Test get_last_status_timestamp function"""

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_dnn_status')
    def test_get_last_status_timestamp_with_valid_status(self, mock_get_dnn_status):
        """Test getting timestamp from valid status data"""
        job_id = "test-job-123"
        now = datetime.now(tz=timezone.utc)
        timestamp_str = now.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        mock_get_dnn_status.return_value = [
            {'timestamp': timestamp_str, 'message': 'Training started'},
            {'timestamp': (now - timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'message': 'Epoch 1'}
        ]

        result = get_last_status_timestamp(job_id, automl=False, experiment_number="0")

        assert result is not None
        assert isinstance(result, datetime)
        # Should return the most recent timestamp
        assert abs((result - now).total_seconds()) < 1

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_dnn_status')
    def test_get_last_status_timestamp_no_status(self, mock_get_dnn_status):
        """Test when no status data is available"""
        mock_get_dnn_status.return_value = None

        result = get_last_status_timestamp("test-job-123")

        assert result is None

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_dnn_status')
    def test_get_last_status_timestamp_empty_status(self, mock_get_dnn_status):
        """Test when status data is empty"""
        mock_get_dnn_status.return_value = []

        result = get_last_status_timestamp("test-job-123")

        assert result is None

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_dnn_status')
    def test_get_last_status_timestamp_multiple_formats(self, mock_get_dnn_status):
        """Test parsing multiple timestamp formats"""
        now = datetime.now(tz=timezone.utc)

        mock_get_dnn_status.return_value = [
            {'timestamp': now.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'message': 'Format 1'},
            {'timestamp': (now - timedelta(minutes=1)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'message': 'Format 2'},
            {'timestamp': (now - timedelta(minutes=2)).strftime('%Y-%m-%dT%H:%M:%S.%f'), 'message': 'Format 3'}
        ]

        result = get_last_status_timestamp("test-job-123")

        assert result is not None
        assert abs((result - now).total_seconds()) < 1

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_dnn_status')
    def test_get_last_status_timestamp_automl_experiment(self, mock_get_dnn_status):
        """Test getting timestamp for AutoML experiment"""
        job_id = "automl-job-123"
        experiment_number = "5"
        now = datetime.now(tz=timezone.utc)

        mock_get_dnn_status.return_value = [
            {'timestamp': now.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'message': 'Training'}
        ]

        result = get_last_status_timestamp(job_id, automl=True, experiment_number=experiment_number)

        assert result is not None
        mock_get_dnn_status.assert_called_once_with(job_id, automl=True, experiment_number=experiment_number)


class TestCheckJobTimeout:
    """Test check_job_timeout function"""

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_last_status_timestamp')
    def test_check_job_timeout_not_timed_out(self, mock_get_timestamp):
        """Test job that has not timed out"""
        job_id = "test-job-123"
        job_info = {
            'job_id': job_id,
            'is_automl': False,
            'handler_id': 'handler-123',
            'kind': 'experiment',
            'timeout_minutes': 1
        }

        # Job last updated 30 seconds ago (within 1 minute timeout)
        last_update = datetime.now(tz=timezone.utc) - timedelta(seconds=30)
        mock_get_timestamp.return_value = last_update

        result = check_job_timeout(job_info)

        assert result is False

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.internal_job_status_update')
    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_last_status_timestamp')
    def test_check_job_timeout_timed_out(self, mock_get_timestamp, mock_status_update):
        """Test job that has timed out"""
        job_id = "test-job-123"
        job_info = {
            'job_id': job_id,
            'is_automl': False,
            'handler_id': 'handler-123',
            'kind': 'experiment',
            'timeout_minutes': 1
        }

        # Job last updated 5 minutes ago (exceeds 1 minute timeout)
        last_update = datetime.now(tz=timezone.utc) - timedelta(minutes=5)
        mock_get_timestamp.return_value = last_update

        result = check_job_timeout(job_info)

        assert result is True
        mock_status_update.assert_called_once()

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_last_status_timestamp')
    def test_check_job_timeout_with_custom_timeout(self, mock_get_timestamp):
        """Test job with custom per-job timeout"""
        job_id = "test-job-123"
        job_info = {
            'job_id': job_id,
            'is_automl': False,
            'handler_id': 'handler-123',
            'kind': 'experiment',
            'timeout_minutes': 120  # 2 hour custom timeout
        }

        # Job last updated 90 minutes ago (within 2 hour timeout)
        last_update = datetime.now(tz=timezone.utc) - timedelta(minutes=90)
        mock_get_timestamp.return_value = last_update

        result = check_job_timeout(job_info)

        assert result is False

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.internal_job_status_update')
    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_last_status_timestamp')
    def test_check_job_timeout_uses_default_when_none(self, mock_get_timestamp, mock_status_update):
        """Test that default timeout (60 min) is used when timeout_minutes is None"""
        job_id = "test-job-123"
        job_info = {
            'job_id': job_id,
            'is_automl': False,
            'handler_id': 'handler-123',
            'kind': 'experiment',
            'timeout_minutes': None  # Should use default 60 minutes
        }

        # Job last updated 90 minutes ago (exceeds default 60 minute timeout)
        last_update = datetime.now(tz=timezone.utc) - timedelta(minutes=90)
        mock_get_timestamp.return_value = last_update

        result = check_job_timeout(job_info)

        assert result is True
        mock_status_update.assert_called_once()

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_last_status_timestamp')
    def test_check_job_timeout_done_status(self, mock_get_timestamp):
        """Test that jobs with recent updates don't time out even if old"""
        job_info = {
            'job_id': 'test-job-123',
            'is_automl': False,
            'timeout_minutes': 1
        }

        # Recent timestamp (30 seconds ago)
        last_update = datetime.now(tz=timezone.utc) - timedelta(seconds=30)
        mock_get_timestamp.return_value = last_update

        result = check_job_timeout(job_info)

        assert result is False

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_handler_job_metadata')
    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.check_pod_liveness')
    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_last_status_timestamp')
    def test_check_job_timeout_no_status_pod_alive(self, mock_get_timestamp, mock_pod_liveness, mock_get_metadata):
        """Test that jobs with no status but alive pods don't time out if recently started"""
        job_info = {
            'job_id': 'test-job-123',
            'is_automl': False,
            'timeout_minutes': 1
        }

        # No status updates
        mock_get_timestamp.return_value = None
        # But pod is alive
        mock_pod_liveness.return_value = True
        # Job started recently (30 seconds ago, within 1 minute timeout)
        mock_get_metadata.return_value = {
            'last_modified': (datetime.now(tz=timezone.utc) - timedelta(seconds=30)).isoformat()
        }

        result = check_job_timeout(job_info)

        assert result is False

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.internal_job_status_update')
    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_last_status_timestamp')
    def test_check_job_timeout_automl_experiment(self, mock_get_timestamp, mock_status_update):
        """Test timeout check for AutoML experiment"""
        job_id = "automl-job-123"
        experiment_number = "5"
        job_info = {
            'job_id': job_id,
            'is_automl': True,
            'experiment_number': experiment_number,
            'handler_id': 'handler-123',
            'kind': 'experiment',
            'timeout_minutes': 2
        }

        # Experiment timed out (5 minutes exceeds 2 minute timeout)
        last_update = datetime.now(tz=timezone.utc) - timedelta(minutes=5)
        mock_get_timestamp.return_value = last_update

        result = check_job_timeout(job_info)

        assert result is True
        mock_status_update.assert_called_once()

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_last_status_timestamp')
    def test_check_job_timeout_automl_experiment_not_timed_out(self, mock_get_timestamp):
        """Test that AutoML experiments with recent updates don't time out"""
        job_info = {
            'job_id': 'automl-job-123',
            'is_automl': True,
            'experiment_number': '5',
            'timeout_minutes': 1
        }

        # Recent update (30 seconds ago, within 1 minute timeout)
        last_update = datetime.now(tz=timezone.utc) - timedelta(seconds=30)
        mock_get_timestamp.return_value = last_update

        result = check_job_timeout(job_info)

        assert result is False

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_handler_job_metadata')
    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.check_pod_liveness')
    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_last_status_timestamp')
    def test_check_job_timeout_no_timestamp_uses_last_modified(
        self, mock_get_timestamp, mock_pod_liveness, mock_get_metadata
    ):
        """Test fallback to last_modified when no status timestamp exists"""
        job_id = "test-job-123"
        job_info = {
            'job_id': job_id,
            'is_automl': False,
            'timeout_minutes': 1
        }

        # No status timestamp
        mock_get_timestamp.return_value = None
        # Pod is alive
        mock_pod_liveness.return_value = True

        # Has last_modified that's recent (within 1 minute timeout)
        last_modified = datetime.now(tz=timezone.utc) - timedelta(seconds=30)
        mock_get_metadata.return_value = {
            'status': 'Running',
            'last_modified': last_modified.isoformat()
        }

        result = check_job_timeout(job_info)

        assert result is False

    def test_check_job_timeout_missing_job_id(self):
        """Test handling of missing job_id"""
        job_info = {
            'is_automl': False
        }

        result = check_job_timeout(job_info)

        assert result is False


class TestTerminateTimedOutJob:
    """Test terminate_timed_out_job function"""

    @patch('nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler.ExecutionHandler')
    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.update_job_status')
    def test_terminate_timed_out_regular_job(self, mock_update_status, mock_executor_class):
        """Test terminating a timed out regular job"""
        job_id = "test-job-123"
        handler_id = "handler-123"
        job_info = {
            'job_id': job_id,
            'handler_id': handler_id,
            'kind': 'experiment',
            'is_automl': False
        }

        # Mock the class method directly
        mock_executor_class.delete_with_handler.return_value = True

        result = terminate_timed_out_job(job_info)

        assert result is True
        mock_update_status.assert_called_once_with(handler_id, job_id, status="Error", kind='experiment')
        mock_executor_class.delete_with_handler.assert_called_once_with(job_id)

    @patch('nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler.ExecutionHandler')
    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.save_automl_controller_info')
    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_automl_controller_info')
    def test_terminate_timed_out_automl_experiment(
        self, mock_get_controller, mock_save_controller, mock_executor_class
    ):
        """Test terminating a timed out AutoML experiment"""
        job_id = "automl-job-123"
        brain_job_id = "brain-job-123"
        handler_id = "handler-123"
        experiment_number = "5"
        job_info = {
            'job_id': job_id,
            'brain_job_id': brain_job_id,
            'handler_id': handler_id,
            'kind': 'experiment',
            'is_automl': True,
            'experiment_number': experiment_number
        }

        # Mock controller info - need to return it twice (one for getting, one for verifying after save)
        controller_data = [
            {'id': '5', 'status': 'running', 'message': 'Training'},
            {'id': '6', 'status': 'pending', 'message': 'Waiting'}
        ]
        mock_get_controller.return_value = controller_data

        # Mock the class method directly
        mock_executor_class.delete_with_handler.return_value = True

        result = terminate_timed_out_job(job_info)

        assert result is True

        # Verify controller info was updated
        call_args = mock_save_controller.call_args[0]
        assert call_args[0] == brain_job_id  # Should use brain_job_id, not job_id
        updated_controller = call_args[1]

        # Find the experiment that should be marked as failure
        experiment_5 = next(exp for exp in updated_controller if exp['id'] == '5')
        assert experiment_5['status'] == 'failure'
        assert 'timeout' in experiment_5['message'].lower()

        # Verify ExecutionHandler was deleted with the correct job_id
        mock_executor_class.delete_with_handler.assert_called_once_with(job_id)

    @patch('nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler.ExecutionHandler')
    def test_terminate_timed_out_job_missing_info(self, mock_executor_class):
        """Test handling of missing job information - treated as orphaned job"""
        job_info = {
            'job_id': 'test-job-123',
            # Missing handler_id - will be treated as orphaned job
            'is_automl': False
        }

        # Mock the class method directly
        mock_executor_class.delete_with_handler.return_value = True

        result = terminate_timed_out_job(job_info)

        # Orphaned jobs (without handler_id) can still be terminated
        assert result is True
        # Verify ExecutionHandler deletion was attempted
        mock_executor_class.delete_with_handler.assert_called_once_with('test-job-123')

    @patch('nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler.ExecutionHandler')
    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.update_job_status')
    def test_terminate_timed_out_job_execution_handler_deletion_fails(self, mock_update_status, mock_executor_class):
        """Test when ExecutionHandler deletion fails"""
        job_id = "test-job-123"
        handler_id = "handler-123"
        job_info = {
            'job_id': job_id,
            'handler_id': handler_id,
            'kind': 'experiment',
            'is_automl': False
        }

        # Mock the class method directly
        mock_executor_class.delete_with_handler.return_value = False

        result = terminate_timed_out_job(job_info)

        assert result is False
        # Status should still be updated even if deletion fails
        mock_update_status.assert_called_once()


class TestCheckForTimedOutJobs:
    """Test check_for_timed_out_jobs function"""

    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.terminate_timed_out_job')
    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.check_job_timeout')
    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.get_all_running_automl_experiments')
    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.get_all_running_jobs')
    @patch.dict(os.environ, {'JOB_TIMEOUT_MONITORING_ENABLED': 'true'})
    def test_check_for_timed_out_jobs_with_timeouts(
        self, mock_get_jobs, mock_get_automl, mock_check_timeout, mock_terminate
    ):
        """Test checking for timed out jobs when some have timed out"""
        # Mock running jobs
        mock_get_jobs.return_value = [
            {'job_id': 'job-1', 'is_automl': False, 'handler_id': 'handler-1', 'kind': 'experiment'},
            {'job_id': 'job-2', 'is_automl': False, 'handler_id': 'handler-2', 'kind': 'experiment'}
        ]

        # Mock running AutoML experiments
        mock_get_automl.return_value = [
            {
                'job_id': 'automl-1',
                'is_automl': True,
                'experiment_number': '3',
                'handler_id': 'handler-3',
                'kind': 'experiment'
            }
        ]

        # First job timed out, others did not
        mock_check_timeout.side_effect = [True, False, False]
        mock_terminate.return_value = True

        result = check_for_timed_out_jobs()

        assert len(result) == 1
        assert 'job-1' in result[0]
        mock_terminate.assert_called_once()

    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.get_all_running_automl_experiments')
    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.get_all_running_jobs')
    @patch.dict(os.environ, {'JOB_TIMEOUT_MONITORING_ENABLED': 'false'})
    def test_check_for_timed_out_jobs_monitoring_disabled(self, mock_get_jobs, mock_get_automl):
        """Test that timeout monitoring can be disabled"""
        result = check_for_timed_out_jobs()

        assert result == []
        # Should not even query for running jobs
        mock_get_jobs.assert_not_called()
        mock_get_automl.assert_not_called()

    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.terminate_timed_out_job')
    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.check_job_timeout')
    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.get_all_running_automl_experiments')
    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.get_all_running_jobs')
    @patch.dict(os.environ, {'JOB_TIMEOUT_MONITORING_ENABLED': 'true'})
    def test_check_for_timed_out_jobs_no_timeouts(
        self, mock_get_jobs, mock_get_automl, mock_check_timeout, mock_terminate
    ):
        """Test when no jobs have timed out"""
        mock_get_jobs.return_value = [
            {'job_id': 'job-1', 'is_automl': False, 'handler_id': 'handler-1', 'kind': 'experiment'}
        ]
        mock_get_automl.return_value = []

        # No jobs timed out
        mock_check_timeout.return_value = False

        result = check_for_timed_out_jobs()

        assert len(result) == 0
        mock_terminate.assert_not_called()

    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.check_job_timeout')
    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.get_all_running_automl_experiments')
    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.get_all_running_jobs')
    @patch.dict(os.environ, {'JOB_TIMEOUT_MONITORING_ENABLED': 'true'})
    def test_check_for_timed_out_jobs_handles_exceptions(
        self, mock_get_jobs, mock_get_automl, mock_check_timeout
    ):
        """Test that exceptions during timeout check don't crash the function"""
        mock_get_jobs.return_value = [
            {'job_id': 'job-1', 'is_automl': False, 'handler_id': 'handler-1', 'kind': 'experiment'}
        ]
        mock_get_automl.return_value = []

        # Simulate an exception during timeout check
        mock_check_timeout.side_effect = Exception("Test exception")

        # Should not raise exception
        result = check_for_timed_out_jobs()

        assert len(result) == 0

    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.terminate_timed_out_job')
    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.check_job_timeout')
    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.get_all_running_automl_experiments')
    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.get_all_running_jobs')
    @patch.dict(os.environ, {'JOB_TIMEOUT_MONITORING_ENABLED': '1'})
    def test_check_for_timed_out_jobs_with_both_regular_and_automl(
        self, mock_get_jobs, mock_get_automl, mock_check_timeout, mock_terminate
    ):
        """Test checking both regular jobs and AutoML experiments"""
        mock_get_jobs.return_value = [
            {'job_id': 'job-1', 'is_automl': False, 'handler_id': 'handler-1', 'kind': 'experiment'}
        ]
        mock_get_automl.return_value = [
            {
                'job_id': 'automl-1',
                'is_automl': True,
                'experiment_number': '3',
                'handler_id': 'handler-3',
                'kind': 'experiment'
            }
        ]

        # Both timed out
        mock_check_timeout.return_value = True
        mock_terminate.return_value = True

        result = check_for_timed_out_jobs()

        assert len(result) == 2
        assert mock_terminate.call_count == 2


class TestTimeoutResetOnRestart:
    """Test that timeout timer resets when jobs are restarted or resumed"""

    def test_resume_job_includes_delete_status_call(self):
        """Test that experiment_handler.py includes delete_dnn_status import and call"""
        # Verify that the resume function has the timeout reset code
        import inspect
        from nvidia_tao_core.microservices.handlers import experiment_handler

        # Check that delete_dnn_status is imported
        assert hasattr(experiment_handler, 'delete_dnn_status')

        # Check that resume_experiment_job exists
        assert hasattr(experiment_handler.ExperimentHandler, 'resume_experiment_job')

        # Verify the function source contains the delete_dnn_status call
        source = inspect.getsource(experiment_handler.ExperimentHandler.resume_experiment_job)
        assert 'delete_dnn_status' in source
        assert 'automl=False' in source

    def test_restart_threads_includes_delete_status_call(self):
        """Test that workflow.py restart_threads includes delete_dnn_status call"""
        # Verify that the restart_threads function has the timeout reset code
        import inspect
        from nvidia_tao_core.microservices.utils.job_utils import workflow

        # Check that delete_dnn_status is imported
        assert hasattr(workflow, 'delete_dnn_status')

        # Check that restart_threads exists
        assert hasattr(workflow.Workflow, 'restart_threads')

        # Verify the function source contains the delete_dnn_status call
        source = inspect.getsource(workflow.Workflow.restart_threads)
        assert 'delete_dnn_status' in source
        assert 'automl=False' in source

    def test_automl_controller_includes_delete_status_call(self):
        """Test that AutoML controller includes delete_dnn_status for experiment restarts"""
        # This test verifies the existing AutoML behavior
        import inspect
        from nvidia_tao_core.microservices.automl import controller

        # Check that delete_dnn_status is imported
        assert hasattr(controller, 'delete_dnn_status')

        # Check that Controller class exists
        assert hasattr(controller, 'Controller')

        # Verify Controller source code contains delete_dnn_status usage
        source = inspect.getsource(controller.Controller)
        assert 'delete_dnn_status' in source


class TestTimeoutConfiguration:
    """Test timeout configuration and per-job timeout handling"""

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_last_status_timestamp')
    def test_per_job_timeout_overrides_default(self, mock_get_timestamp):
        """Test that per-job timeout is used when specified"""
        job_info = {
            'job_id': 'test-job-123',
            'is_automl': False,
            'timeout_minutes': 5  # Custom 5 minute timeout
        }

        # Job updated 3 minutes ago (within custom 5 minute timeout)
        last_update = datetime.now(tz=timezone.utc) - timedelta(minutes=3)
        mock_get_timestamp.return_value = last_update

        result = check_job_timeout(job_info)

        # Should not timeout
        assert result is False

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.internal_job_status_update')
    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_last_status_timestamp')
    def test_default_timeout_used_when_not_specified(self, mock_get_timestamp, mock_status_update):
        """Test that default 60 minute timeout is used when not specified"""
        job_info = {
            'job_id': 'test-job-123',
            'is_automl': False,
            'handler_id': 'handler-123',
            'kind': 'experiment',
            'timeout_minutes': None  # No custom timeout
        }

        # Job updated 90 minutes ago (exceeds default 60 minute timeout)
        last_update = datetime.now(tz=timezone.utc) - timedelta(minutes=90)
        mock_get_timestamp.return_value = last_update

        result = check_job_timeout(job_info)

        # Should timeout with default
        assert result is True
        mock_status_update.assert_called_once()

    @patch.dict(os.environ, {'JOB_TIMEOUT_MONITORING_ENABLED': 'false'})
    @patch('nvidia_tao_core.microservices.utils.job_utils.workflow.get_all_running_jobs')
    def test_timeout_monitoring_can_be_disabled(self, mock_get_jobs):
        """Test that timeout monitoring can be completely disabled"""
        result = check_for_timed_out_jobs()

        assert result == []
        mock_get_jobs.assert_not_called()


class TestTimeoutWithStatusUpdates:
    """Test timeout behavior with various status update patterns"""

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_last_status_timestamp')
    def test_timeout_with_continuous_updates(self, mock_get_timestamp):
        """Test that jobs with continuous updates don't time out"""
        job_id = "test-job-123"
        job_info = {
            'job_id': job_id,
            'is_automl': False,
            'timeout_minutes': 1
        }

        now = datetime.now(tz=timezone.utc)

        # Job has regular status updates (most recent is 10 seconds ago, within 1 minute window)
        mock_get_timestamp.return_value = now - timedelta(seconds=10)

        result = check_job_timeout(job_info)

        assert result is False

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.internal_job_status_update')
    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_last_status_timestamp')
    def test_timeout_with_stale_updates(self, mock_get_timestamp, mock_status_update):
        """Test that jobs with stale updates time out"""
        job_id = "test-job-123"
        job_info = {
            'job_id': job_id,
            'is_automl': False,
            'handler_id': 'handler-123',
            'kind': 'experiment',
            'timeout_minutes': 1
        }

        now = datetime.now(tz=timezone.utc)

        # Job has old status updates, most recent was 5 minutes ago (exceeds 1 minute timeout)
        mock_get_timestamp.return_value = now - timedelta(minutes=5)

        result = check_job_timeout(job_info)

        assert result is True
        mock_status_update.assert_called_once()

    @patch('nvidia_tao_core.microservices.utils.job_utils.timeout_monitor.get_last_status_timestamp')
    def test_timeout_boundary_condition(self, mock_get_timestamp):
        """Test timeout just below boundary (1 minute)"""
        job_id = "test-job-123"
        job_info = {
            'job_id': job_id,
            'is_automl': False,
            'timeout_minutes': 1
        }

        now = datetime.now(tz=timezone.utc)

        # Job last updated 59 seconds ago (just under 1 minute timeout)
        mock_get_timestamp.return_value = now - timedelta(seconds=59)

        result = check_job_timeout(job_info)

        # Just under 1 minute, should not be timed out
        assert result is False


class TestAutoMLBrainExperimentIdFallback:
    """Test that AutoML brain jobs with experiment_id (no handler_id) are handled.

    AutoML brain jobs created by automl_handler.py write experiment_id and
    omit handler_id/kind in tao.jobs.  Before the fix, get_all_running_jobs()
    returned handler_id=None for these jobs, and
    get_all_running_automl_experiments() skipped them entirely, so their
    child experiment jobs were never checked for timeout.
    """

    # -- get_all_running_jobs: handler_id resolved from experiment_id --

    @patch('nvidia_tao_core.microservices.utils.timeout_utils.BACKEND', 'local-docker')
    @patch('nvidia_tao_core.microservices.utils.timeout_utils.is_request_automl', return_value=False)
    @patch('nvidia_tao_core.microservices.utils.timeout_utils.get_handler_metadata', return_value=None)
    @patch('nvidia_tao_core.microservices.utils.timeout_utils.MongoHandler')
    def test_running_jobs_resolves_experiment_id_when_handler_id_missing(
        self, MockMongo, mock_get_handler, mock_is_automl
    ):
        """Brain job with handler_id=None but experiment_id set should have
        handler_id resolved in the returned job_info."""
        brain_job_doc = {
            'id': 'brain-001',
            'handler_id': None,
            'experiment_id': 'experiment-handler-001',
            'kind': None,
            'status': 'Running',
            'action': 'train',
            'job_details': {'brain-001': {'automl_brain_info': []}},
        }

        mock_mongo_instance = MagicMock()
        # First call: find({'status': ...}) for running jobs
        # Second call: find({'status': {'$exists': True}}) for automl_experiment_job_ids
        mock_mongo_instance.find.side_effect = [[brain_job_doc], []]
        MockMongo.return_value = mock_mongo_instance

        jobs = get_all_running_jobs()

        assert len(jobs) == 1
        job = jobs[0]
        assert job['job_id'] == 'brain-001'
        assert job['handler_id'] == 'experiment-handler-001', \
            "handler_id should be resolved from experiment_id"
        assert job['kind'] == 'experiment', \
            "kind should default to 'experiment' when missing"
        assert job['is_automl_brain'] is True

    @patch('nvidia_tao_core.microservices.utils.timeout_utils.BACKEND', 'local-docker')
    @patch('nvidia_tao_core.microservices.utils.timeout_utils.is_request_automl', return_value=False)
    @patch('nvidia_tao_core.microservices.utils.timeout_utils.get_handler_metadata', return_value=None)
    @patch('nvidia_tao_core.microservices.utils.timeout_utils.MongoHandler')
    def test_running_jobs_keeps_handler_id_when_present(
        self, MockMongo, mock_get_handler, mock_is_automl
    ):
        """Normal job with handler_id set should keep it (no fallback needed)."""
        normal_job_doc = {
            'id': 'job-001',
            'handler_id': 'handler-001',
            'kind': 'experiment',
            'status': 'Running',
            'action': 'train',
            'job_details': {},
        }

        mock_mongo_instance = MagicMock()
        mock_mongo_instance.find.side_effect = [[normal_job_doc], []]
        MockMongo.return_value = mock_mongo_instance

        jobs = get_all_running_jobs()

        assert len(jobs) == 1
        assert jobs[0]['handler_id'] == 'handler-001'
        assert jobs[0]['kind'] == 'experiment'

    # -- get_all_running_automl_experiments: discovers experiments for brain with experiment_id --

    @patch('nvidia_tao_core.microservices.utils.timeout_utils.BACKEND', 'local-docker')
    @patch('nvidia_tao_core.microservices.utils.timeout_utils.get_automl_controller_info')
    @patch('nvidia_tao_core.microservices.utils.timeout_utils.get_handler_metadata')
    @patch('nvidia_tao_core.microservices.utils.timeout_utils.is_request_automl', return_value=False)
    @patch('nvidia_tao_core.microservices.utils.timeout_utils.MongoHandler')
    def test_automl_experiments_found_when_brain_has_experiment_id(
        self, MockMongo, mock_is_automl, mock_get_handler, mock_get_controller
    ):
        """Brain job with handler_id=None but experiment_id set should still
        have its child experiments discovered for timeout monitoring."""
        brain_job_doc = {
            'id': 'brain-001',
            'handler_id': None,
            'experiment_id': 'experiment-handler-001',
            'kind': None,
            'status': 'Running',
            'action': 'train',
            'job_details': {'brain-001': {'automl_brain_info': []}},
        }

        mock_mongo_instance = MagicMock()
        # get_all_running_jobs: find running, then find all for automl_experiment_job_ids
        # get_all_running_automl_experiments calls get_all_running_jobs internally
        mock_mongo_instance.find.side_effect = [
            [brain_job_doc], [],  # get_all_running_jobs (called from get_all_running_automl_experiments)
            [brain_job_doc], [],  # get_all_running_jobs (2nd call in get_all_running_automl_experiments flow)
        ]
        # find_one for the raw job doc fallback in get_all_running_automl_experiments
        mock_mongo_instance.find_one.return_value = brain_job_doc
        MockMongo.return_value = mock_mongo_instance

        # Handler metadata lookup returns automl_enabled=True for the resolved handler_id
        mock_get_handler.return_value = {
            'automl_settings': {'automl_enabled': True}
        }

        # Controller has one running experiment
        mock_get_controller.return_value = [
            {
                'id': 1,
                'job_id': 'train-exp-001',
                'status': 'started',
            }
        ]

        experiments = get_all_running_automl_experiments()

        assert len(experiments) == 1, \
            "Should discover the running experiment from the brain's controller"
        exp = experiments[0]
        assert exp['job_id'] == 'train-exp-001'
        assert exp['is_automl'] is True
        assert exp['brain_job_id'] == 'brain-001'
        assert exp['experiment_number'] == '1'
        assert exp['handler_id'] == 'experiment-handler-001'

    @patch('nvidia_tao_core.microservices.utils.timeout_utils.BACKEND', 'local-docker')
    @patch('nvidia_tao_core.microservices.utils.timeout_utils.get_automl_controller_info')
    @patch('nvidia_tao_core.microservices.utils.timeout_utils.get_handler_metadata')
    @patch('nvidia_tao_core.microservices.utils.timeout_utils.is_request_automl', return_value=False)
    @patch('nvidia_tao_core.microservices.utils.timeout_utils.MongoHandler')
    def test_automl_experiments_not_found_when_brain_has_no_ids(
        self, MockMongo, mock_is_automl, mock_get_handler, mock_get_controller
    ):
        """Brain job with neither handler_id nor experiment_id should be skipped
        gracefully (no crash, no experiments returned)."""
        brain_job_doc = {
            'id': 'brain-orphan',
            'handler_id': None,
            'kind': None,
            'status': 'Running',
            'action': 'train',
            'job_details': {},
        }

        mock_mongo_instance = MagicMock()
        mock_mongo_instance.find.side_effect = [
            [brain_job_doc], [],
            [brain_job_doc], [],
        ]
        # find_one returns the same doc (no experiment_id either)
        mock_mongo_instance.find_one.return_value = brain_job_doc
        MockMongo.return_value = mock_mongo_instance

        experiments = get_all_running_automl_experiments()

        assert len(experiments) == 0
        mock_get_controller.assert_not_called()
