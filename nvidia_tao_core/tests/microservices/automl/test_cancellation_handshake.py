# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Race and API-boundary tests for the AutoML cancellation barrier."""

import os
from types import SimpleNamespace
from unittest.mock import Mock, call, patch


os.environ["TAO_TEST_MODE"] = "true"

from nvidia_tao_core.microservices.enum_constants import Backend  # noqa: E402
from nvidia_tao_core.microservices.automl.controller import Controller  # noqa: E402
from nvidia_tao_core.microservices.utils.automl_utils import (  # noqa: E402
    JobStates,
    Recommendation,
    ResumeRecommendation,
)
from nvidia_tao_core.microservices.handlers.actions import AutoMLPipeline  # noqa: E402
from nvidia_tao_core.microservices.handlers.automl_handler import AutoMLHandler  # noqa: E402
from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import (  # noqa: E402
    ExecutionHandler,
)
from nvidia_tao_core.microservices.handlers.job_handler import JobHandler  # noqa: E402
from nvidia_tao_core.microservices.utils.automl_job_utils import (  # noqa: E402
    cancel_automl_child_job,
    on_cancel_automl_job,
)
from nvidia_tao_core.microservices.utils.handler_utils import Code  # noqa: E402
from nvidia_tao_core.microservices.utils.job_utils.dependencies import (  # noqa: E402
    dependency_check_automl,
)


ACTIONS_MODULE = "nvidia_tao_core.microservices.handlers.actions"
CONTROLLER_MODULE = "nvidia_tao_core.microservices.automl.controller"
AUTOML_HANDLER_MODULE = "nvidia_tao_core.microservices.handlers.automl_handler"
AUTOML_JOB_UTILS_MODULE = "nvidia_tao_core.microservices.utils.automl_job_utils"
DEPENDENCIES_MODULE = "nvidia_tao_core.microservices.utils.job_utils.dependencies"
JOB_HANDLER_MODULE = "nvidia_tao_core.microservices.handlers.job_handler"


def test_canceled_automl_dependency_is_never_runnable():
    """A queued recommendation cannot pass dependency checks after fencing."""
    job_context = SimpleNamespace(id="child-0", parent_id="brain-0")
    dependency = SimpleNamespace(name="0")
    with (
        patch(
            f"{DEPENDENCIES_MODULE}.get_automl_controller_info",
            return_value=[{"job_id": "child-0", "status": "pending"}],
        ),
        patch(
            f"{DEPENDENCIES_MODULE}.get_handler_job_metadata",
            side_effect=[
                {"automl_cancel_requested": True},
                {"status": "Running"},
            ],
        ),
    ):
        dependency_met, message = dependency_check_automl(job_context, dependency)

    assert dependency_met is False
    assert "cancellation" in message.lower()


def test_missing_execution_handler_fails_closed():
    """An unknown backend is not evidence that its writer has stopped."""
    with patch.object(ExecutionHandler, "create_handler", return_value=None):
        assert ExecutionHandler.delete_with_handler("child-0", {}) is False
        assert ExecutionHandler.delete_job_with_handler("brain-0") is False


def test_submitting_child_is_not_quiescent_even_if_delete_found_no_resource():
    """Cancellation waits for the launcher handshake after a pre-create delete."""
    lifecycle = {
        "cancel_requested": True,
        "launch_state": "submitting",
        "error": "",
    }
    with (
        patch(
            "nvidia_tao_core.microservices.utils.stateless_handler_utils."
            "request_automl_child_cancellation",
            return_value=True,
        ),
        patch(
            "nvidia_tao_core.microservices.utils.stateless_handler_utils."
            "get_automl_child_lifecycle",
            side_effect=[lifecycle, lifecycle],
        ),
        patch(
            "nvidia_tao_core.microservices.utils.stateless_handler_utils."
            "set_automl_child_launch_state"
        ) as set_launch_state,
        patch(f"{AUTOML_JOB_UTILS_MODULE}.on_delete_automl_job"),
        patch(f"{AUTOML_JOB_UTILS_MODULE}.on_cancel_automl_job", return_value=True),
    ):
        assert cancel_automl_child_job("child-0") is False

    set_launch_state.assert_not_called()


def test_launcher_quiescence_wins_over_concurrent_delete_failure():
    """A launcher's stronger post-create acknowledgement is never downgraded."""
    submitting = {
        "cancel_requested": True,
        "launch_state": "submitting",
        "error": "",
    }
    quiescent = {
        "cancel_requested": True,
        "launch_state": "quiescent",
        "error": "",
    }
    with (
        patch(
            "nvidia_tao_core.microservices.utils.stateless_handler_utils."
            "request_automl_child_cancellation",
            return_value=True,
        ),
        patch(
            "nvidia_tao_core.microservices.utils.stateless_handler_utils."
            "get_automl_child_lifecycle",
            side_effect=[submitting, quiescent],
        ),
        patch(
            "nvidia_tao_core.microservices.utils.stateless_handler_utils."
            "set_automl_child_launch_state"
        ) as set_launch_state,
        patch(f"{AUTOML_JOB_UTILS_MODULE}.on_delete_automl_job"),
        patch(f"{AUTOML_JOB_UTILS_MODULE}.on_cancel_automl_job", return_value=False),
    ):
        assert cancel_automl_child_job("child-0") is True

    set_launch_state.assert_not_called()


def test_automl_pipeline_honors_fence_before_backend_create():
    """A tombstone observed after the launch claim prevents resource creation."""
    pipeline = AutoMLPipeline.__new__(AutoMLPipeline)
    pipeline.job_name = "child-0"
    pipeline._cancellation_requested = Mock(return_value=True)
    pipeline._finish_canceled_launch = Mock(return_value=True)
    pipeline.create_microservice_action_job = Mock()

    with (
        patch(f"{ACTIONS_MODULE}.claim_automl_child_launch", return_value=True),
        patch(
            f"{ACTIONS_MODULE}.get_automl_child_lifecycle",
            return_value={"cancel_requested": True, "launch_state": "quiescent"},
        ),
    ):
        assert pipeline.run() is False

    pipeline._finish_canceled_launch.assert_called_once_with(
        backend_may_exist=False
    )
    pipeline.create_microservice_action_job.assert_not_called()


def test_automl_pipeline_deletes_resource_when_cancel_races_after_create():
    """A fence that lands during create is acknowledged only after exact delete."""
    pipeline = AutoMLPipeline.__new__(AutoMLPipeline)
    pipeline.job_name = "child-0"
    pipeline.rec_number = 0
    pipeline.recs_dict = [{"specs": {}}]
    pipeline.workspace_metadata = {}
    pipeline.automl_brain_job_id = "brain-0"
    pipeline.expt_root = "/results/child-0"
    pipeline.handler_metadata = {"docker_env_vars": {}}
    pipeline.job_context = SimpleNamespace(org_name="org", backend_details={})
    pipeline.add_ptm_dependency = Mock()
    pipeline.generate_config = Mock(return_value={})
    pipeline.handle_ptm_anomalies = Mock()
    pipeline.get_handler_cloud_details = Mock()
    pipeline.save_recommendation_specs = Mock()
    pipeline.generate_run_command = Mock(return_value="train")
    pipeline.detailed_print = Mock()
    pipeline.decrypt_docker_env_vars = Mock()
    pipeline.generate_env_variables = Mock()
    pipeline.generate_nv_job_metadata = Mock()
    pipeline.create_microservice_action_job = Mock()
    pipeline.monitor_job = Mock()
    pipeline._cancellation_requested = Mock(
        side_effect=[False, False, True]
    )
    pipeline._finish_canceled_launch = Mock(return_value=True)

    with (
        patch(f"{ACTIONS_MODULE}.BACKEND", Backend.LOCAL_DOCKER),
        patch(f"{ACTIONS_MODULE}.claim_automl_child_launch", return_value=True),
        patch(
            f"{ACTIONS_MODULE}.get_automl_child_lifecycle",
            return_value={"cancel_requested": True, "launch_state": "quiescent"},
        ),
        patch(f"{ACTIONS_MODULE}.create_cs_instance", return_value=(Mock(), None)),
        patch(f"{ACTIONS_MODULE}.wait_for_job_completion"),
        patch(f"{ACTIONS_MODULE}.delete_lingering_checkpoints"),
    ):
        assert pipeline.run() is False

    pipeline.create_microservice_action_job.assert_called_once()
    pipeline._finish_canceled_launch.assert_called_once_with(
        backend_may_exist=True
    )
    pipeline.monitor_job.assert_not_called()


def test_multifidelity_promotion_rearms_quiesced_child_before_enqueue():
    """A completed child tombstone must not permanently block its next rung."""
    rec = Recommendation(0, {"train.num_epochs": 1}, "loss")
    rec.assign_job_id("child-0")
    rec.update_status(JobStates.success)
    promoted = ResumeRecommendation(
        0,
        {"train.num_epochs": 2},
        "child-0",
    )
    controller = Controller.__new__(Controller)
    controller.automl_context = SimpleNamespace(id="brain-0", early_stop_epoch=1)
    controller.network = "cosmos-rl"
    controller.automl_algorithm = "hyperband"
    controller.automl_algorithm_settings = SimpleNamespace(
        automl_max_recommendations=4
    )
    controller.recommendations = [rec]
    controller.brain = SimpleNamespace(
        max_concurrent=1,
        epoch_number=2,
        generate_recommendations=Mock(return_value=[promoted]),
        save_state=Mock(),
    )
    controller.best_epoch_number = {0: 1}
    controller.metric_key = "loss"
    controller.decrypted_workspace_metadata = {}
    controller.cs_instance = SimpleNamespace(delete_file=Mock())
    controller.root = "/nonexistent/automl"
    controller.hyperband_cancel_condition_seen = False
    controller.save_state = Mock()
    controller._inject_automl_env_vars = Mock()
    controller._get_experiment_results_path = Mock(
        return_value="/results/child-0"
    )
    events = []
    controller.on_new_automl_job = Mock(
        side_effect=lambda item: events.append(("enqueue", item.job_id))
    )

    with (
        patch(f"{CONTROLLER_MODULE}.report_health_beat"),
        patch(f"{CONTROLLER_MODULE}.save_automl_current_rec"),
        patch(f"{CONTROLLER_MODULE}.save_job_specs"),
        patch(
            f"{CONTROLLER_MODULE}.get_file_list_from_cloud_storage",
            return_value=[],
        ),
        patch(f"{CONTROLLER_MODULE}.delete_dnn_status"),
        patch(
            f"{CONTROLLER_MODULE}.reset_automl_child_lifecycle_for_resume",
            side_effect=lambda job_id: events.append(("rearm", job_id)) or True,
        ),
    ):
        controller.run_experiments()

    assert events == [("rearm", "child-0"), ("enqueue", "child-0")]


def test_child_cancel_resolves_workspace_from_experiment_not_brain():
    """Remote cancellation uses the child's experiment workspace credentials."""
    workspace = {"id": "workspace-0", "cloud_type": "slurm", "_id": "mongo-id"}
    with (
        patch(
            f"{AUTOML_JOB_UTILS_MODULE}.get_handler_job_metadata",
            return_value={"handler_id": "experiment-0", "parent_id": "brain-0"},
        ),
        patch(
            f"{AUTOML_JOB_UTILS_MODULE}.get_handler_metadata",
            side_effect=[{"workspace": "workspace-0"}, workspace],
        ),
        patch(
            "nvidia_tao_core.microservices.utils.handler_utils."
            "decrypt_handler_metadata"
        ),
        patch.object(
            ExecutionHandler, "delete_with_handler", return_value=True
        ) as delete_with_handler,
    ):
        assert on_cancel_automl_job("child-0") is True

    delete_with_handler.assert_called_once_with(
        "child-0",
        workspace_metadata={"id": "workspace-0", "cloud_type": "slurm"},
    )


def test_automl_stop_orders_fence_and_cleanup_before_brain_delete():
    """The public stop boundary retains the brain until cleanup is acknowledged."""
    events = []
    recommendations = [{"job_id": "child-0", "status": "running"}]
    termination_handler = Mock()
    termination_handler.wait_for_job_termination.side_effect = (
        lambda *args, **kwargs: events.append("brain_wait") or True
    )

    with (
        patch(
            f"{AUTOML_HANDLER_MODULE}.get_automl_controller_info",
            side_effect=[recommendations, recommendations],
        ),
        patch(f"{AUTOML_HANDLER_MODULE}.get_automl_child_job_ids", return_value=[]),
        patch(
            f"{AUTOML_HANDLER_MODULE}.request_automl_child_cancellation",
            side_effect=lambda job_id: events.append("intent") or True,
        ),
        patch(
            f"{AUTOML_HANDLER_MODULE}.set_automl_checkpoint_cleanup_state",
            side_effect=lambda *args, **kwargs: events.append("cleanup_intent") or True,
        ),
        patch(
            f"{AUTOML_HANDLER_MODULE}.cancel_automl_child_job",
            side_effect=lambda job_id: events.append("child_quiescent") or True,
        ),
        patch(
            f"{AUTOML_HANDLER_MODULE}.get_handler_job_metadata",
            return_value={
                "checkpoint_cleanup_pending": False,
                "checkpoint_cleanup_status": "complete",
            },
        ),
        patch(f"{AUTOML_HANDLER_MODULE}.save_automl_controller_info"),
        patch(f"{AUTOML_HANDLER_MODULE}.update_automl_details_metadata"),
        patch.object(
            ExecutionHandler,
            "delete_job_with_handler",
            side_effect=lambda job_id: events.append("brain_delete") or True,
        ),
        patch.object(
            ExecutionHandler,
            "create_handler",
            return_value=termination_handler,
        ),
        patch(f"{AUTOML_HANDLER_MODULE}.stop_monitoring_job"),
        patch(f"{AUTOML_HANDLER_MODULE}.BACKEND", Backend.LOCAL_DOCKER),
    ):
        response = AutoMLHandler.stop(
            "user-0", "org", "experiment-0", "brain-0"
        )

    assert response.code == 200
    assert events == [
        "intent",
        "cleanup_intent",
        "child_quiescent",
        "brain_delete",
        "brain_wait",
    ]


def test_automl_stop_returns_conflict_and_retains_brain_when_barrier_fails():
    """A failed child barrier is retryable and cannot delete the guardian brain."""
    recommendations = [{"job_id": "child-0", "status": "running"}]
    with (
        patch.dict(os.environ, {"AUTOML_CANCEL_TIMEOUT_SECONDS": "0"}),
        patch(
            f"{AUTOML_HANDLER_MODULE}.get_automl_controller_info",
            return_value=recommendations,
        ),
        patch(f"{AUTOML_HANDLER_MODULE}.get_automl_child_job_ids", return_value=[]),
        patch(
            f"{AUTOML_HANDLER_MODULE}.request_automl_child_cancellation",
            return_value=True,
        ),
        patch(
            f"{AUTOML_HANDLER_MODULE}.set_automl_checkpoint_cleanup_state",
            return_value=True,
        ),
        patch(
            f"{AUTOML_HANDLER_MODULE}.cancel_automl_child_job",
            return_value=False,
        ),
        patch(
            f"{AUTOML_HANDLER_MODULE}.get_handler_job_metadata",
            return_value={"checkpoint_cleanup_pending": True},
        ),
        patch(f"{AUTOML_HANDLER_MODULE}.save_automl_controller_info"),
        patch(f"{AUTOML_HANDLER_MODULE}.update_automl_details_metadata"),
        patch.object(
            ExecutionHandler, "delete_job_with_handler"
        ) as delete_brain,
    ):
        response = AutoMLHandler.stop(
            "user-0", "org", "experiment-0", "brain-0"
        )

    assert response.code == 409
    delete_brain.assert_not_called()


def test_job_handler_does_not_finalize_parent_when_automl_barrier_is_pending():
    """The REST-facing handler leaves status Canceling on a retryable conflict."""
    pending_response = Code(409, [], "cancellation pending")
    handler_metadata = {
        "jobs": {"brain-0": {}},
        "workspace": "workspace-0",
        "user_id": "user-0",
    }
    with (
        patch(f"{JOB_HANDLER_MODULE}.resolve_metadata", return_value=handler_metadata),
        patch(f"{JOB_HANDLER_MODULE}.get_handler_metadata", return_value={}),
        patch(f"{JOB_HANDLER_MODULE}.check_write_access", return_value=True),
        patch(
            f"{JOB_HANDLER_MODULE}.get_handler_job_metadata",
            return_value={"action": "train", "status": "Running"},
        ),
        patch(f"{JOB_HANDLER_MODULE}.is_request_automl", return_value=True),
        patch(f"{JOB_HANDLER_MODULE}.AutoMLHandler.stop", return_value=pending_response),
        patch(f"{JOB_HANDLER_MODULE}.update_job_status") as update_status,
        patch(f"{JOB_HANDLER_MODULE}.on_delete_automl_job") as dequeue,
    ):
        response = JobHandler.job_cancel(
            "org", "experiment-0", "brain-0", "experiment"
        )

    assert response is pending_response
    assert update_status.call_args_list == [
        call(
            "experiment-0",
            "brain-0",
            status="Canceling",
            kind="experiments",
        )
    ]
    dequeue.assert_not_called()
