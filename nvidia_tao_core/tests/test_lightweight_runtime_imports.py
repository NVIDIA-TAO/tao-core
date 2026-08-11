# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import importlib
import json
import sys


def test_handlers_package_does_not_eagerly_import_api_handlers() -> None:
    package_name = "nvidia_tao_core.microservices.handlers"
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == package_name or name.startswith(f"{package_name}.")
    }
    for name in previous_modules:
        sys.modules.pop(name, None)
    try:
        importlib.import_module(package_name)
        assert f"{package_name}.dataset_handler" not in sys.modules
        assert f"{package_name}.job_handler" not in sys.modules
    finally:
        for name in list(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)


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
