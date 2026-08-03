# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AutoML checkpoint cleanup when the controller is canceled."""

import os
import shutil
from types import SimpleNamespace
from unittest.mock import DEFAULT, MagicMock, Mock, call, patch


# Avoid connecting to a real MongoDB instance while importing controller dependencies.
os.environ["TAO_TEST_MODE"] = "true"

from nvidia_tao_core.microservices.automl.controller import Controller  # noqa: E402
from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import (  # noqa: E402
    ExecutionHandler,
)
from nvidia_tao_core.microservices.utils.automl_utils import JobStates  # noqa: E402


CONTROLLER_MODULE = "nvidia_tao_core.microservices.automl.controller"


def _recommendation(rec_id, status, result=0.0, job_id=None, resume_from_job_id=None):
    """Build the recommendation attributes used by cancellation cleanup."""
    return SimpleNamespace(
        id=rec_id,
        status=status,
        result=result,
        job_id=job_id or f"job-{rec_id}",
        resume_from_job_id=resume_from_job_id
    )


def _controller(recommendations, algorithm="bayesian", delete_checkpoints=True, retain_for_resume=False):
    """Build a controller without invoking external storage initialization."""
    controller = Controller.__new__(Controller)
    controller.automl_context = SimpleNamespace(id="automl-job", handler_id="experiment")
    controller.delete_intermediate_ckpt = delete_checkpoints
    controller.retain_checkpoints_for_resume = retain_for_resume
    controller.automl_algorithm = algorithm
    controller.recommendations = recommendations
    controller.min_max = max
    controller.refresh_recommendations = Mock()
    controller._get_experiment_results_path = Mock(
        side_effect=lambda job_id: f"/results/{job_id}"
    )
    controller.delete_checkpoint_files = Mock()
    controller.delete_not_best_model_checkpoints = Mock()
    return controller


def _start_controller(loop_error=None):
    """Build the state needed to exercise Controller.start control flow."""
    controller = Controller.__new__(Controller)
    controller.automl_context = SimpleNamespace(id="automl-job", handler_id="experiment")
    controller.network = "cosmos-rl"
    controller.wandb_initialized = False
    controller.best_model_copied = False
    controller.best_rec_id = -1
    controller._checkpoint_cleanup_pending = False
    controller._initialize_wandb_for_automl = Mock()
    controller._execute_loop = Mock(side_effect=loop_error)
    controller.cancel_recommendation_jobs = Mock()

    def persist_cleanup_intent():
        controller._checkpoint_cleanup_pending = True
        return True

    controller._schedule_checkpoint_cleanup = Mock(
        side_effect=persist_cleanup_intent
    )
    return controller


def _start_dependency_mocks(metadata):
    """Return a patch.multiple context for Controller.start dependencies."""
    patcher = patch.multiple(
        CONTROLLER_MODULE,
        report_health_beat=DEFAULT,
        get_job_specs=DEFAULT,
        update_job_message=DEFAULT,
        get_handler_job_metadata=DEFAULT,
        write_job_metadata=DEFAULT,
        update_job_status=DEFAULT,
        delete_health_beat=DEFAULT,
    )
    mocks = patcher.start()
    mocks["get_job_specs"].return_value = {}
    mocks["get_handler_job_metadata"].return_value = metadata
    return patcher, mocks


def test_cancellation_cleanup_honors_delete_intermediate_flag():
    """Disabling intermediate deletion must leave every recommendation untouched."""
    controller = _controller(
        [_recommendation(0, JobStates.canceled)],
        delete_checkpoints=False
    )

    assert controller._cleanup_checkpoints_on_cancellation() is True

    controller.refresh_recommendations.assert_not_called()
    controller.delete_checkpoint_files.assert_not_called()
    controller.delete_not_best_model_checkpoints.assert_not_called()


def test_cancellation_cleanup_preserves_best_and_deletes_non_best_artifacts():
    """Independent trials retain the best checkpoint and delete other trial artifacts."""
    best = _recommendation(0, JobStates.success, result=0.9)
    non_best = _recommendation(1, JobStates.success, result=0.5)
    canceled = _recommendation(2, JobStates.canceled)
    failed = _recommendation(3, JobStates.failure)
    controller = _controller([best, non_best, canceled, failed])

    controller._cleanup_checkpoints_on_cancellation()

    controller.delete_checkpoint_files.assert_called_once_with(
        "/results/job-0", best, retain_resume_checkpoint=False
    )
    assert controller.delete_not_best_model_checkpoints.call_args_list == [
        call("/results/job-1", non_best, True),
        call("/results/job-2", canceled, True),
        call("/results/job-3", failed, True),
    ]


def test_canceled_hyperband_prunes_nonbest_resume_and_promotion_sources():
    """Terminal cancel prunes nonbest trials; pause is the resumable path."""
    best = _recommendation(0, JobStates.success, result=0.9)
    promotion_source = _recommendation(1, JobStates.success, result=0.7)
    current = _recommendation(2, JobStates.canceled)
    unused_canceled = _recommendation(3, JobStates.canceled)
    failed = _recommendation(4, JobStates.failure, resume_from_job_id="job-1")
    controller = _controller(
        [best, promotion_source, current, unused_canceled, failed],
        algorithm="hyperband",
        retain_for_resume=True,
    )

    assert controller._cleanup_checkpoints_on_cancellation() is True

    controller.delete_checkpoint_files.assert_called_once_with(
        "/results/job-0", best, retain_resume_checkpoint=False
    )
    assert controller.delete_not_best_model_checkpoints.call_args_list == [
        call("/results/job-1", promotion_source, True),
        call("/results/job-2", current, True),
        call("/results/job-3", unused_canceled, True),
        call("/results/job-4", failed, True),
    ]


def test_cancellation_cleanup_continues_after_storage_error():
    """A failure deleting one trial must not prevent cleanup of later trials."""
    first = _recommendation(0, JobStates.canceled)
    second = _recommendation(1, JobStates.failure)
    controller = _controller([first, second])
    controller.delete_not_best_model_checkpoints.side_effect = [
        RuntimeError("storage unavailable"),
        None
    ]

    assert controller._cleanup_checkpoints_on_cancellation() is False

    assert controller.delete_not_best_model_checkpoints.call_args_list == [
        call("/results/job-0", first, True),
        call("/results/job-1", second, True),
    ]


def test_cancellation_cleanup_retries_when_best_pruning_is_unverified():
    """A silent/partial storage delete must keep the durable cleanup pending."""
    best = _recommendation(0, JobStates.success, result=0.9)
    controller = _controller([best])
    controller.delete_checkpoint_files.return_value = False

    assert controller._cleanup_checkpoints_on_cancellation() is False


def test_nonbest_cleanup_detects_acknowledged_but_still_visible_artifact():
    """Deletion acknowledgement alone is not proof that a remote prefix is gone."""
    rec = _recommendation(0, JobStates.failure)
    controller = Controller.__new__(Controller)
    controller.recommendations = [rec]
    controller.min_max = max
    controller.decrypted_workspace_metadata = {}
    controller._uses_folder_lookup = Mock(return_value=False)
    controller.cs_instance = SimpleNamespace(
        is_file=Mock(return_value=True),
        delete_file=Mock(return_value=True),
    )
    checkpoint = "/results/job-0/model.pth"

    with patch(
        f"{CONTROLLER_MODULE}.get_file_list_from_cloud_storage",
        side_effect=[[checkpoint], [checkpoint]],
    ):
        controller.delete_not_best_model_checkpoints(
            "/results/job-0", rec, True
        )

    assert controller._last_checkpoint_delete_verified is False


def test_multifidelity_trial_retains_best_and_rung_boundary_checkpoint():
    """Promotion needs the rung checkpoint even when an earlier epoch scored best."""
    best_path = "/results/job-0/model_epoch_1.pth"
    intermediate_path = "/results/job-0/model_epoch_2.pth"
    resume_path = "/results/job-0/model_epoch_3.pth"
    remaining = {best_path, intermediate_path, resume_path}

    class Storage:
        @staticmethod
        def is_file(path):
            return path in remaining

        @staticmethod
        def delete_file(path):
            remaining.discard(path)
            return True

    rec = _recommendation(0, JobStates.success, result=0.9)
    rec.early_stop_epoch = 3
    controller = Controller.__new__(Controller)
    controller.automl_context = SimpleNamespace(id="automl-job")
    controller.automl_algorithm = "hyperband"
    controller.retain_checkpoints_for_resume = True
    controller.decrypted_workspace_metadata = {}
    controller.cs_instance = Storage()
    controller.ckpt_path = {}
    controller._uses_folder_lookup = Mock(return_value=False)
    controller.get_best_checkpoint_path = Mock(
        side_effect=lambda path, item, filter_by_format=False: controller.ckpt_path.update(
            {path: {"pth": best_path}}
        )
    )
    controller.get_checkpoint_paths_matching_epoch_number = Mock(
        return_value=([], [], [resume_path], [], [])
    )

    with (
        patch.dict(os.environ, {"CI_PROJECT_DIR": "test"}),
        patch(
            f"{CONTROLLER_MODULE}.get_file_list_from_cloud_storage",
            side_effect=lambda *_args, **_kwargs: sorted(remaining),
        ),
        patch(f"{CONTROLLER_MODULE}.report_health_beat"),
    ):
        assert controller.delete_checkpoint_files(
            "/results/job-0", rec, retain_resume_checkpoint=True
        ) is True

    assert remaining == {best_path, resume_path}
    controller.get_checkpoint_paths_matching_epoch_number.assert_called_once_with(
        "/results/job-0", 0, epoch_number=3
    )


@patch("nvidia_tao_core.microservices.automl.controller.get_automl_controller_info", return_value=[])
@patch(
    "nvidia_tao_core.microservices.automl.controller."
    "set_automl_checkpoint_cleanup_state",
    new=Mock(return_value=True),
)
@patch(
    "nvidia_tao_core.microservices.automl.controller.get_handler_job_metadata",
    return_value={"status": "canceling"}
)
@patch("nvidia_tao_core.microservices.automl.controller.report_health_beat")
@patch("nvidia_tao_core.microservices.automl.controller.update_job_status")
def test_execute_loop_schedules_checkpoint_cleanup_on_cancellation_exit(
    mock_update_status, mock_health_beat, mock_job_metadata, mock_controller_info
):
    """The cancellation status must not bypass eligible checkpoint cleanup."""
    controller = Controller.__new__(Controller)
    controller.automl_context = SimpleNamespace(id="automl-job", handler_id="experiment")
    controller.completed_recommendations = 0
    controller.automl_algorithm_settings = SimpleNamespace(automl_max_recommendations=4)
    controller._checkpoint_cleanup_pending = False

    assert controller._execute_loop() is None

    assert controller._checkpoint_cleanup_pending is True
    mock_job_metadata.assert_called_once_with("automl-job")
    mock_controller_info.assert_called_once_with("automl-job")
    assert mock_health_beat.call_count == 2
    mock_update_status.assert_called_once()


@patch(
    "nvidia_tao_core.microservices.automl.controller."
    "set_automl_checkpoint_cleanup_state",
    return_value=True,
)
@patch(
    "nvidia_tao_core.microservices.automl.controller.get_automl_child_job_ids",
    return_value=[],
)
@patch("nvidia_tao_core.microservices.automl.controller.on_delete_automl_job")
@patch("nvidia_tao_core.microservices.automl.controller.cancel_automl_child_job")
def test_cancel_jobs_cleans_checkpoints_after_stopping_children(
    mock_cancel_job, mock_delete_job, mock_child_ids, mock_cleanup_state
):
    """Deferred cleanup runs after child cancellation and before deleting the parent."""
    controller = Controller.__new__(Controller)
    controller.automl_context = SimpleNamespace(id="automl-job")
    controller.recommendations = [_recommendation(0, JobStates.canceled)]
    controller._checkpoint_cleanup_pending = True
    controller._cleanup_checkpoints_on_cancellation = Mock()

    events = []
    mock_cancel_job.side_effect = lambda job_id: events.append(("cancel", job_id)) or True
    controller._cleanup_checkpoints_on_cancellation.side_effect = (
        lambda: events.append(("cleanup", None)) or True
    )
    mock_delete_job.side_effect = lambda job_id: events.append(("delete", job_id))

    with patch.dict(os.environ, {"TAO_EXECUTION_BACKEND": "remote"}, clear=True):
        controller.cancel_recommendation_jobs()

    assert events == [
        ("cancel", "job-0"),
        ("cleanup", None),
        ("delete", "automl-job"),
    ]
    assert controller._checkpoint_cleanup_pending is False


@patch(
    "nvidia_tao_core.microservices.automl.controller."
    "set_automl_checkpoint_cleanup_state",
    return_value=True,
)
@patch(
    "nvidia_tao_core.microservices.automl.controller.get_automl_child_job_ids",
    return_value=[],
)
@patch("nvidia_tao_core.microservices.automl.controller.on_delete_automl_job")
@patch(
    "nvidia_tao_core.microservices.automl.controller.cancel_automl_child_job",
    return_value=False,
)
def test_cancel_jobs_retains_checkpoints_when_writer_stop_is_unconfirmed(
    mock_cancel_job, mock_delete_job, mock_child_ids, mock_cleanup_state
):
    """A failed cancellation barrier must never be followed by storage deletion."""
    controller = Controller.__new__(Controller)
    controller.automl_context = SimpleNamespace(id="automl-job")
    controller.recommendations = [_recommendation(0, JobStates.canceled)]
    controller._checkpoint_cleanup_pending = True
    controller._cleanup_checkpoints_on_cancellation = Mock()

    with patch.dict(os.environ, {"TAO_EXECUTION_BACKEND": "remote"}, clear=True):
        assert controller.cancel_recommendation_jobs() is False

    controller._cleanup_checkpoints_on_cancellation.assert_not_called()
    assert controller._checkpoint_cleanup_pending is True
    mock_cancel_job.assert_called_once_with("job-0")
    mock_delete_job.assert_not_called()


def test_pending_cleanup_retries_after_a_false_barrier():
    """The guardian controller retries instead of treating False as completion."""
    controller = Controller.__new__(Controller)
    controller.automl_context = SimpleNamespace(id="automl-job")
    controller.cancel_recommendation_jobs = Mock(side_effect=[False, True])

    with (
        patch(f"{CONTROLLER_MODULE}.report_health_beat") as report_health_beat,
        patch(f"{CONTROLLER_MODULE}.time.sleep") as sleep,
    ):
        assert controller._finish_pending_checkpoint_cleanup() is True

    assert controller.cancel_recommendation_jobs.call_count == 2
    report_health_beat.assert_called_once()
    sleep.assert_called_once_with(5)


@patch(
    "nvidia_tao_core.microservices.automl.controller."
    "set_automl_checkpoint_cleanup_state",
    return_value=True,
)
@patch(
    "nvidia_tao_core.microservices.automl.controller.get_automl_child_job_ids",
    return_value=[],
)
@patch("nvidia_tao_core.microservices.automl.controller.on_delete_automl_job")
@patch(
    "nvidia_tao_core.microservices.automl.controller.cancel_automl_child_job",
    return_value=True,
)
def test_artifact_failure_remains_durably_pending(
    mock_cancel_job, mock_delete_job, mock_child_ids, mock_cleanup_state
):
    """A partial storage failure blocks parent deletion and remains retryable."""
    controller = Controller.__new__(Controller)
    controller.automl_context = SimpleNamespace(id="automl-job")
    controller.recommendations = [_recommendation(0, JobStates.canceled)]
    controller._checkpoint_cleanup_pending = True
    controller._cleanup_checkpoints_on_cancellation = Mock(return_value=False)

    with patch.dict(os.environ, {"TAO_EXECUTION_BACKEND": "remote"}, clear=True):
        assert controller.cancel_recommendation_jobs() is False

    assert controller._checkpoint_cleanup_pending is True
    mock_delete_job.assert_not_called()
    assert call(
        "automl-job",
        "pending",
        error="one or more checkpoint artifacts could not be removed",
    ) in mock_cleanup_state.call_args_list


def test_delete_with_handler_propagates_explicit_cancellation_failure():
    """The controller must be able to distinguish acceptance from failure."""
    handler = Mock()
    handler.delete.return_value = False
    with patch.object(ExecutionHandler, "create_handler", return_value=handler):
        assert not ExecutionHandler.delete_with_handler("job-0", {})


def test_kubernetes_delete_waits_until_child_pods_are_gone():
    """Foreground deletion is complete only after every writer pod exits."""
    from nvidia_tao_core.microservices.handlers.execution_handlers.kubernetes_handler import (  # noqa: E501
        KubernetesHandler,
    )

    handler = KubernetesHandler.__new__(KubernetesHandler)
    handler.logger = Mock()
    handler.delete_statefulset = Mock(return_value=True)
    handler.get_namespace = Mock(return_value="default")
    core_api = MagicMock()
    core_api.list_namespaced_pod.side_effect = [
        SimpleNamespace(items=[object()]),
        SimpleNamespace(items=[]),
    ]
    with (
        patch(
            "nvidia_tao_core.microservices.handlers.execution_handlers."
            "kubernetes_handler.client.CoreV1Api",
            return_value=core_api,
        ),
        patch(
            "nvidia_tao_core.microservices.handlers.execution_handlers."
            "kubernetes_handler.time.sleep"
        ),
    ):
        assert handler.delete("job-0")

    assert core_api.list_namespaced_pod.call_count == 2


def test_slurm_cancel_waits_for_terminal_scheduler_state():
    """Successful scancel submission alone is not a writer-stop barrier."""
    from nvidia_tao_core.microservices.handlers.execution_handlers.slurm_handler import (  # noqa: E501
        SlurmHandler,
    )

    handler = SlurmHandler.__new__(SlurmHandler)
    handler.logger = Mock()
    handler.get_automl_aware_handler_params = Mock(
        return_value={"brain_job_id": "brain-0"}
    )
    handler.get_slurm_job_id = Mock(return_value="42")
    handler.get_slurm_job_status = Mock(
        side_effect=["RUNNING", "RUNNING", "CANCELLED"]
    )
    handler._build_ssh_command = Mock(return_value=["scancel", "42"])
    handler.update_job_status = Mock()
    completed = SimpleNamespace(returncode=0, stdout="", stderr="")
    module = (
        "nvidia_tao_core.microservices.handlers.execution_handlers.slurm_handler"
    )
    with (
        patch(f"{module}.get_handler_job_metadata", return_value={"id": "brain-0"}),
        patch(f"{module}.subprocess.run", return_value=completed),
        patch(f"{module}.time.sleep"),
    ):
        assert handler.cancel_job("job-0")

    assert handler.get_slurm_job_status.call_count == 3


def test_slurm_cancel_without_metadata_is_already_quiescent():
    """A job that was never submitted has no SLURM writer to stop."""
    from nvidia_tao_core.microservices.handlers.execution_handlers.slurm_handler import (  # noqa: E501
        SlurmHandler,
    )

    handler = SlurmHandler.__new__(SlurmHandler)
    handler.logger = Mock()
    handler.get_automl_aware_handler_params = Mock(
        return_value={"brain_job_id": "brain-0"}
    )
    handler.get_slurm_job_id = Mock()
    module = (
        "nvidia_tao_core.microservices.handlers.execution_handlers.slurm_handler"
    )

    with patch(f"{module}.get_handler_job_metadata", return_value=None):
        assert handler.cancel_job("job-0") is True

    handler.get_slurm_job_id.assert_not_called()


def test_slurm_cancel_without_scheduler_id_is_already_quiescent():
    """Metadata without a scheduler ID means there is no submitted job to cancel."""
    from nvidia_tao_core.microservices.handlers.execution_handlers.slurm_handler import (  # noqa: E501
        SlurmHandler,
    )

    handler = SlurmHandler.__new__(SlurmHandler)
    handler.logger = Mock()
    handler.get_automl_aware_handler_params = Mock(
        return_value={"brain_job_id": "brain-0"}
    )
    handler.get_slurm_job_id = Mock(return_value=None)
    handler.get_slurm_job_status = Mock()
    module = (
        "nvidia_tao_core.microservices.handlers.execution_handlers.slurm_handler"
    )

    with patch(
        f"{module}.get_handler_job_metadata", return_value={"id": "brain-0"}
    ):
        assert handler.cancel_job("job-0") is True

    handler.get_slurm_job_status.assert_not_called()


def test_slurm_cancel_fails_closed_when_scheduler_query_fails():
    """An unconfirmed scheduler state must not release the cleanup barrier."""
    from nvidia_tao_core.microservices.handlers.execution_handlers.slurm_handler import (  # noqa: E501
        SlurmHandler,
    )

    handler = SlurmHandler.__new__(SlurmHandler)
    handler.logger = Mock()
    handler.get_automl_aware_handler_params = Mock(
        return_value={"brain_job_id": "brain-0"}
    )
    handler.get_slurm_job_id = Mock(return_value="42")
    handler.get_slurm_job_status = Mock(return_value="ERROR")
    handler._build_ssh_command = Mock()
    module = (
        "nvidia_tao_core.microservices.handlers.execution_handlers.slurm_handler"
    )

    with patch(
        f"{module}.get_handler_job_metadata", return_value={"id": "brain-0"}
    ):
        assert handler.cancel_job("job-0") is False

    handler._build_ssh_command.assert_not_called()


def test_start_preserves_canceled_parent_status_and_runs_deferred_cleanup():
    """A handled cancellation must not fall through to Error/Done finalization."""
    controller = _start_controller()
    controller._execute_loop.side_effect = lambda: setattr(
        controller, "_checkpoint_cleanup_pending", True
    )
    cleanup_pending_at_cancel = []
    controller.cancel_recommendation_jobs.side_effect = lambda: cleanup_pending_at_cancel.append(
        controller._checkpoint_cleanup_pending
    )
    patcher, mocks = _start_dependency_mocks({"status": "Canceled", "job_details": {}})
    try:
        controller.start()
    finally:
        patcher.stop()

    assert cleanup_pending_at_cancel == [True]
    controller.cancel_recommendation_jobs.assert_called_once_with()
    mocks["write_job_metadata"].assert_not_called()
    mocks["update_job_status"].assert_not_called()
    mocks["delete_health_beat"].assert_called_once_with("automl-job")


def test_successful_completion_schedules_durable_final_trial_pruning():
    """ASHA/non-incremental leftovers are pruned through the terminal barrier."""
    controller = _start_controller()
    controller.automl_algorithm = "asha"
    controller.best_model_copied = True
    controller._get_experiment_results_path = Mock(return_value="/results/automl-job")
    events = []
    controller._schedule_checkpoint_cleanup = Mock(
        side_effect=lambda: events.append("cleanup_pending") or True
    )
    controller._finish_pending_checkpoint_cleanup = Mock(
        side_effect=lambda: events.append("cleanup_complete") or True
    )
    metadata = {"status": "Running", "job_details": {}}
    patcher, mocks = _start_dependency_mocks(metadata)
    try:
        controller.start()
    finally:
        patcher.stop()

    assert events == ["cleanup_pending", "cleanup_complete"]
    mocks["update_job_status"].assert_called_once_with(
        "experiment", "automl-job", status="Done", kind="experiments"
    )
    mocks["delete_health_beat"].assert_called_once_with("automl-job")


def test_start_exception_schedules_cleanup_before_stopping_children():
    """Abnormal exits schedule eligible cleanup before child teardown."""
    controller = _start_controller(loop_error=RuntimeError("training interrupted"))
    cleanup_pending_at_cancel = []
    controller.cancel_recommendation_jobs.side_effect = lambda: cleanup_pending_at_cancel.append(
        controller._checkpoint_cleanup_pending
    )
    metadata = {"status": "Running", "job_details": {}}
    patcher, mocks = _start_dependency_mocks(metadata)
    try:
        controller.start()
    finally:
        patcher.stop()

    assert cleanup_pending_at_cancel == [True]
    mocks["write_job_metadata"].assert_called_once_with("automl-job", metadata)
    mocks["update_job_status"].assert_called_once_with(
        "experiment", "automl-job", status="Error", kind="experiments"
    )
    mocks["delete_health_beat"].assert_called_once_with("automl-job")


def test_start_exception_does_not_overwrite_cancellation_status():
    """An exception racing with cancellation must leave parent status externally managed."""
    controller = _start_controller(loop_error=RuntimeError("controller interrupted"))
    patcher, mocks = _start_dependency_mocks({"status": "canceling", "job_details": {}})
    try:
        controller.start()
    finally:
        patcher.stop()

    controller.cancel_recommendation_jobs.assert_called_once_with()
    mocks["write_job_metadata"].assert_not_called()
    mocks["update_job_status"].assert_not_called()


def test_folder_checkpoint_cleanup_removes_cosmos_sidecars(tmp_path):
    """Deleting a Cosmos checkpoint leaf also removes extensionless sidecar files."""
    policy_folder = tmp_path / "train_output" / "checkpoints" / "epoch_1" / "policy"
    policy_folder.mkdir(parents=True)
    for name in (
        "model_rank_0.pth",
        "optimizer_rank_0.pth",
        ".rank_0_complete",
        "cosmos_config",
    ):
        (policy_folder / name).write_text("checkpoint data", encoding="utf-8")
    unrelated_file = tmp_path / "microservices_log.txt"
    unrelated_file.write_text("keep", encoding="utf-8")

    class LocalStorage:
        """Minimal recursive-delete storage adapter for the folder cleanup test."""

        @staticmethod
        def delete_folder(folder):
            shutil.rmtree(folder)

    rec = _recommendation(0, JobStates.canceled)
    controller = Controller.__new__(Controller)
    controller.recommendations = [rec]
    controller.min_max = max
    controller.decrypted_workspace_metadata = {}
    controller.cs_instance = LocalStorage()
    controller._uses_folder_lookup = Mock(return_value=True)
    listed_files = [str(file_path) for file_path in tmp_path.rglob("*") if file_path.is_file()]

    with patch(f"{CONTROLLER_MODULE}.get_file_list_from_cloud_storage", return_value=listed_files):
        controller.delete_not_best_model_checkpoints(str(tmp_path), rec, True)

    assert not policy_folder.exists()
    assert policy_folder.parent.exists()  # Empty structural ancestors are intentionally retained.
    assert unrelated_file.exists()
