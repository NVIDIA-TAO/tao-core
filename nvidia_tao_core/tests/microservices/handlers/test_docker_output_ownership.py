# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership safety tests for TAO Core's local Docker backend."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from nvidia_tao_core.microservices.enum_constants import Backend
from nvidia_tao_core.microservices.handlers.execution_handlers.docker_handler import (
    DockerHandler,
    _configured_container_identity,
    _writable_bind_mounts,
)
from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import (
    ExecutionHandler,
)


def _handler():
    handler = object.__new__(DockerHandler)
    ExecutionHandler.__init__(handler, backend_type=Backend.LOCAL_DOCKER)
    handler._docker_image = "nvcr.io/nvidia/tao/example:test"
    handler._docker_client = MagicMock()
    handler._docker_client.info.return_value = {"SecurityOptions": []}
    handler._api_client = MagicMock()
    handler._container = None
    return handler


def test_writable_bind_detection_skips_read_only_and_docker_socket():
    assert _writable_bind_mounts(["/data"]) == []
    assert _writable_bind_mounts([
        "/host/results:/results",
        "/host/data:/data:ro",
        "/var/run/docker.sock:/var/run/docker.sock",
    ]) == [("/host/results", "/results", "rw")]
    assert _writable_bind_mounts({
        "/host/results": {"bind": "/results", "mode": "rw,z"},
        "/host/data": {"bind": "/data", "mode": "ro"},
    }) == [("/host/results", "/results", "rw,z")]
    assert _writable_bind_mounts({
        "/host/data": {"bind": "/data", "ro": True},
        "/host/results": {
            "bind": "/results",
            "ro": False,
            "propagation": "rshared",
        },
    }) == [("/host/results", "/results", "rw,rshared")]
    # Ownership discovery must not add restrictions to mounts that cannot
    # create host output.
    assert _writable_bind_mounts(["/host/root:/:ro"]) == []
    with pytest.raises(ValueError, match='both "ro" and "mode"'):
        _writable_bind_mounts({
            "/host/results": {"bind": "/results", "ro": False, "mode": "rw"}
        })


@pytest.mark.parametrize("value", ["", "root", "0:0", "0:1000", "1000", "a:b"])
def test_writable_bind_identity_requires_explicit_non_root_uid_gid(monkeypatch, value):
    monkeypatch.setenv("TAO_DOCKER_CONTAINER_USER", value)
    with pytest.raises(RuntimeError, match="verified non-root"):
        _configured_container_identity()


def test_explicit_identity_and_supplementary_groups_are_numeric(monkeypatch):
    monkeypatch.setenv("TAO_DOCKER_CONTAINER_USER", "1234:5678")
    monkeypatch.setenv("TAO_DOCKER_GROUP_ADD", "5678, 99,100,99")
    assert _configured_container_identity() == ("1234:5678", ["99", "100"])

    monkeypatch.setenv("TAO_DOCKER_GROUP_ADD", "99,developers")
    with pytest.raises(RuntimeError, match="numeric supplementary"):
        _configured_container_identity()


def test_bind_preflight_uses_exact_identity_and_prepares_runtime_home():
    handler = _handler()

    handler._preflight_writable_bind(
        "/host/results", "rw,z", "1234:5678", ["99"], prepare_home=True
    )

    kwargs = handler._docker_client.containers.run.call_args.kwargs
    assert kwargs["user"] == "1234:5678"
    assert kwargs["group_add"] == ["99"]
    assert kwargs["volumes"] == {
        "/host/results": {"bind": "/ownership-probe", "mode": "rw,z"}
    }
    assert kwargs["remove"] is True
    assert "mkdir \"$probe\"" in kwargs["command"][1]
    assert 'test ! -L "$path"' in kwargs["command"][1]
    assert "chmod 700" in kwargs["command"][1]
    assert '"$cache/huggingface"' in kwargs["command"][1]


def test_bind_preflight_supports_writable_file_bind_without_using_it_as_home():
    handler = _handler()

    handler._preflight_writable_bind(
        "/host/config.yaml", "rw", "1234:5678", [], prepare_home=False
    )

    script = handler._docker_client.containers.run.call_args.kwargs["command"][1]
    assert "elif [ -f /ownership-probe ]" in script
    assert "test -w /ownership-probe" in script


def test_bind_preflight_failure_blocks_workload_launch():
    handler = _handler()
    handler._docker_client.containers.run.side_effect = RuntimeError("permission denied")

    with pytest.raises(PermissionError, match="could not create and delete"):
        handler._preflight_writable_bind(
            "/host/results", "rw", "1234:5678", [], prepare_home=True
        )


def test_rootless_or_userns_daemon_is_rejected(monkeypatch):
    handler = _handler()
    handler._docker_client.info.return_value = {
        "SecurityOptions": ["name=seccomp,profile=builtin", "name=rootless"]
    }
    monkeypatch.setenv("TAO_DOCKER_CONTAINER_USER", "1234:5678")

    with pytest.raises(RuntimeError, match="rootless or userns-remapped"):
        handler._prepare_writable_bind_runtime(["/host/results:/results"])

    handler._docker_client.containers.run.assert_not_called()


def test_bind_runtime_prefers_results_for_home_and_checks_each_source(monkeypatch):
    handler = _handler()
    handler._preflight_writable_bind = MagicMock()
    monkeypatch.setenv("TAO_DOCKER_CONTAINER_USER", "1234:5678")
    monkeypatch.setenv("TAO_DOCKER_GROUP_ADD", "99")

    user, groups, environment = handler._prepare_writable_bind_runtime({
        "/host/scratch": {"bind": "/scratch", "mode": "rw"},
        "/host/results": {"bind": "/results", "mode": "rw"},
    })

    assert user == "1234:5678"
    assert groups == ["99"]
    assert environment["HOME"] == "/results/.tao-runtime/home"
    assert environment["USER"] == "1234"
    assert environment["LOGNAME"] == "1234"
    assert environment["HF_HOME"] == "/results/.tao-runtime/home/.cache/huggingface"
    assert handler._preflight_writable_bind.call_args_list == [
        call(
            "/host/scratch", "rw", "1234:5678", ["99"], prepare_home=False
        ),
        call(
            "/host/results", "rw", "1234:5678", ["99"], prepare_home=True
        ),
    ]


def test_start_container_maps_verified_user_and_writable_home(monkeypatch):
    handler = _handler()
    monkeypatch.setenv("TAO_DOCKER_CONTAINER_USER", "1234:5678")
    monkeypatch.setenv("TAO_DOCKER_GROUP_ADD", "99")
    launched = SimpleNamespace(name="job-1")
    handler._docker_client.containers.get.side_effect = RuntimeError("not found")
    handler._docker_client.containers.run.return_value = launched
    handler._check_image_exists = MagicMock(return_value=True)
    handler.update_image_pull_status = MagicMock()
    handler.get_device_requests = MagicMock(return_value=[])
    handler._prepare_writable_bind_runtime = MagicMock(return_value=(
        "1234:5678",
        ["99"],
        {
            "HOME": "/results/.tao-runtime/home",
            "USER": "1234",
            "LOGNAME": "1234",
            "XDG_CACHE_HOME": "/results/.tao-runtime/home/.cache",
        },
    ))

    with patch(
        "nvidia_tao_core.microservices.handlers.execution_handlers.docker_handler."
        "should_use_nvidia_runtime",
        return_value=False,
    ), patch(
        "nvidia_tao_core.microservices.handlers.execution_handlers.docker_handler."
        "get_handler_job_metadata",
        return_value={},
    ):
        handler.start_container(
            container_name="job-1",
            docker_env_vars={"HOME": "/root", "KEEP": "yes"},
            command=["train"],
            num_gpus=0,
            volumes=["/host/results:/results"],
        )

    kwargs = handler._docker_client.containers.run.call_args.kwargs
    assert kwargs["user"] == "1234:5678"
    assert kwargs["group_add"] == ["99"]
    assert kwargs["environment"]["HOME"] == "/results/.tao-runtime/home"
    assert kwargs["environment"]["USER"] == "1234"
    assert kwargs["environment"]["LOGNAME"] == "1234"
    assert kwargs["environment"]["XDG_CACHE_HOME"].endswith("/.cache")
    assert kwargs["environment"]["KEEP"] == "yes"
    handler._prepare_writable_bind_runtime.assert_called_once_with(
        ["/host/results:/results"],
        writable_mounts=[("/host/results", "/results", "rw")],
        configured_identity=("1234:5678", ["99"]),
        identity_mapping_verified=True,
    )


def test_start_container_without_writable_bind_preserves_image_user():
    handler = _handler()
    assert handler._prepare_writable_bind_runtime([
        "/host/data:/data:ro",
        "/var/run/docker.sock:/var/run/docker.sock",
    ]) is None
    assert handler._prepare_writable_bind_runtime([
        "/host/config.yaml:/config.yaml",
    ]) is None


def test_start_container_rejects_missing_identity_before_image_check(monkeypatch):
    handler = _handler()
    handler._check_image_exists = MagicMock()
    monkeypatch.delenv("TAO_DOCKER_CONTAINER_USER", raising=False)

    with pytest.raises(RuntimeError, match="verified non-root"):
        handler.start_container(
            container_name="job-1",
            num_gpus=0,
            volumes=["/host/results:/results"],
        )

    handler._check_image_exists.assert_not_called()
    handler._docker_client.containers.run.assert_not_called()


def test_start_container_preflight_failure_never_launches_workload(monkeypatch):
    handler = _handler()
    handler._check_image_exists = MagicMock(return_value=True)
    handler.update_image_pull_status = MagicMock()
    handler.get_device_requests = MagicMock()
    handler._docker_client.containers.run.side_effect = RuntimeError("permission denied")
    monkeypatch.setenv("TAO_DOCKER_CONTAINER_USER", "1234:5678")

    with pytest.raises(PermissionError, match="could not create and delete"):
        handler.start_container(
            container_name="job-1",
            num_gpus=0,
            volumes=["/host/results:/results"],
        )

    assert handler._docker_client.containers.run.call_count == 1
    handler.get_device_requests.assert_not_called()
    assert handler._container is None


def test_full_start_with_anonymous_and_file_volumes_preserves_image_user():
    handler = _handler()
    handler._check_image_exists = MagicMock(return_value=True)
    handler.update_image_pull_status = MagicMock()
    handler.get_device_requests = MagicMock(return_value=[])
    handler._docker_client.containers.get.side_effect = RuntimeError("not found")
    handler._docker_client.containers.run.return_value = SimpleNamespace(name="job-1")

    with patch(
        "nvidia_tao_core.microservices.handlers.execution_handlers.docker_handler."
        "should_use_nvidia_runtime",
        return_value=False,
    ), patch(
        "nvidia_tao_core.microservices.handlers.execution_handlers.docker_handler."
        "get_handler_job_metadata",
        return_value={},
    ):
        handler.start_container(
            container_name="job-1",
            num_gpus=0,
            volumes=["/data", "/host/config.yaml:/config.yaml"],
        )

    kwargs = handler._docker_client.containers.run.call_args.kwargs
    assert "user" not in kwargs
    assert "group_add" not in kwargs
    assert kwargs["volumes"] == ["/data", "/host/config.yaml:/config.yaml"]
