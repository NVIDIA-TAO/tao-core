# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import json
import sys


def test_handlers_package_does_not_eagerly_import_api_handlers() -> None:
    import nvidia_tao_core.microservices.handlers  # noqa: F401

    assert "nvidia_tao_core.microservices.handlers.dataset_handler" not in sys.modules
    assert "nvidia_tao_core.microservices.handlers.job_handler" not in sys.modules


def test_file_status_logger_survives_missing_control_plane_dependencies(tmp_path, monkeypatch) -> None:
    from nvidia_tao_core.loggers.logging import Status, StatusLogger

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("nvidia_tao_core.microservices.handlers.cloud_handlers"):
            raise ImportError("control plane intentionally absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    status_path = tmp_path / "status.json"
    status = StatusLogger(filename=status_path, is_master=True)
    status.kpi = {"validation_loss": 0.25}
    status.write(status_level=Status.SUCCESS, message="done")

    record = json.loads(status_path.read_text().strip())
    assert record["status"] == "SUCCESS"
    assert record["kpi"]["validation_loss"] == 0.25
