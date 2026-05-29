# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import pytest

from nvidia_tao_core.microservices.utils.ngc_utils import split_ngc_path, download_ngc_model

# import logging
# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)


@pytest.mark.ngc_handler
def client():
    from ngcsdk import Client
    clt = Client()
    return clt


@pytest.mark.ngc_handler
def test_split_ngc_path(ngc_path):
    ngc_path = "org/team/model:version"
    org, team, model_name, model_version = split_ngc_path(ngc_path)
    assert org == "org", f"Expected org to be 'org', but got '{org}'"
    assert team == "team", f"Expected team to be 'team', but got '{team}'"
    assert model_name == "model", f"Expected model_name to be 'model', but got '{model_name}'"
    assert model_version == "version", f"Expected model_version to be 'version', but got '{model_version}'"


@pytest.mark.ngc_handler
def test_download_ngc_model_success(ngc_key, ngc_path, tmpdir):
    ptm_root = f'.{tmpdir.strpath}'
    os.makedirs(ptm_root, exist_ok=True)
    assert download_ngc_model(ngc_path, ptm_root, ngc_key), "NGC model download failed with valid key"
    shutil.rmtree('tmp')


@pytest.mark.ngc_handler
def test_download_ngc_model_invalid_key(ngc_path, tmpdir):
    ptm_root = tmpdir.strpath
    key = "test-key"
    assert not download_ngc_model(ngc_path, ptm_root, key), (
        "NGC model download succeeded with invalid key"
    )
