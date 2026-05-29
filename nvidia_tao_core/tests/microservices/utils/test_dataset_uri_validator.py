# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dataset path validation"""
import pytest
from nvidia_tao_core.microservices.utils.dataset_uri_validator import (
    validate_dataset_uri,
    validate_all_dataset_uris,
    VALID_PROTOCOLS,
    REMOTE_BACKENDS
)


class TestValidateDatasetUri:
    """Tests for validate_dataset_uri function"""

    def test_valid_aws_path(self):
        """Test valid AWS S3 path"""
        is_valid, error = validate_dataset_uri("aws://bucket/path/to/data")
        assert is_valid is True
        assert error == ""

    def test_valid_s3_path_normalized(self):
        """Test that s3:// is normalized to aws://"""
        is_valid, error = validate_dataset_uri("s3://bucket/path/to/data")
        assert is_valid is True
        assert error == ""

    def test_valid_azure_path(self):
        """Test valid Azure path"""
        is_valid, error = validate_dataset_uri("azure://container/path/to/data")
        assert is_valid is True
        assert error == ""

    def test_valid_lustre_path(self):
        """Test valid Lustre path"""
        is_valid, error = validate_dataset_uri("lustre:///scratch/data/train")
        assert is_valid is True
        assert error == ""

    def test_valid_file_path(self):
        """Test valid file:// path"""
        is_valid, error = validate_dataset_uri("file:///local/data/train")
        assert is_valid is True
        assert error == ""

    def test_valid_local_path_no_prefix(self):
        """Test valid local path without prefix"""
        is_valid, error = validate_dataset_uri("/local/data/train")
        assert is_valid is True
        assert error == ""

    def test_invalid_protocol(self):
        """Test invalid protocol"""
        is_valid, error = validate_dataset_uri("ftp://server/path")
        assert is_valid is False
        assert "Invalid protocol 'ftp'" in error
        assert "Supported:" in error

    def test_empty_path(self):
        """Test empty path"""
        is_valid, error = validate_dataset_uri("")
        assert is_valid is False
        assert "cannot be empty" in error

    def test_local_path_on_slurm_backend(self):
        """Test that local paths are not allowed on SLURM backend"""
        is_valid, error = validate_dataset_uri("/local/data", backend_type="slurm")
        assert is_valid is False
        assert "not allowed for SLURM backend" in error
        assert "lustre://" in error

    def test_file_path_on_slurm_backend(self):
        """Test that file:// paths are not allowed on SLURM backend"""
        is_valid, error = validate_dataset_uri("file:///local/data", backend_type="slurm")
        assert is_valid is False
        assert "not allowed for SLURM backend" in error

    def test_local_path_on_lepton_backend(self):
        """Test that local paths are not allowed on Lepton backend"""
        is_valid, error = validate_dataset_uri("/local/data", backend_type="lepton")
        assert is_valid is False
        assert "not allowed for LEPTON backend" in error
        assert "aws://" in error or "azure://" in error

    def test_file_path_on_lepton_backend(self):
        """Test that file:// paths are not allowed on Lepton backend"""
        is_valid, error = validate_dataset_uri("file:///local/data", backend_type="lepton")
        assert is_valid is False
        assert "not allowed for LEPTON backend" in error

    def test_local_path_on_nvcf_backend(self):
        """Test that local paths are not allowed on NVCF backend"""
        is_valid, error = validate_dataset_uri("/local/data", backend_type="nvcf")
        assert is_valid is False
        assert "not allowed for NVCF backend" in error

    def test_lustre_path_on_slurm_backend(self):
        """Test that lustre:// paths are allowed on SLURM backend"""
        is_valid, error = validate_dataset_uri("lustre:///scratch/data", backend_type="slurm")
        assert is_valid is True
        assert error == ""

    def test_aws_path_on_lepton_backend(self):
        """Test that aws:// paths are allowed on Lepton backend"""
        is_valid, error = validate_dataset_uri("aws://bucket/data", backend_type="lepton")
        assert is_valid is True
        assert error == ""

    def test_azure_path_on_lepton_backend(self):
        """Test that azure:// paths are allowed on Lepton backend"""
        is_valid, error = validate_dataset_uri("azure://container/data", backend_type="lepton")
        assert is_valid is True
        assert error == ""

    def test_local_path_on_local_backend(self):
        """Test that local paths are allowed on local backend"""
        is_valid, error = validate_dataset_uri("/local/data", backend_type="local")
        assert is_valid is True
        assert error == ""

    def test_all_paths_allowed_without_backend_type(self):
        """Test that all valid protocols are allowed when backend_type is None"""
        paths = [
            "aws://bucket/data",
            "azure://container/data",
            "lustre:///scratch/data",
            "file:///local/data",
            "/local/data"
        ]
        for path in paths:
            is_valid, error = validate_dataset_uri(path, backend_type=None)
            assert is_valid is True, f"Path {path} should be valid without backend_type"
            assert error == ""


class TestValidateAllDatasetUris:
    """Tests for validate_all_dataset_uris function"""

    def test_valid_all_paths(self):
        """Test validation with all valid paths"""
        metadata = {
            "train_dataset_uris": ["lustre:///data/train1", "lustre:///data/train2"],
            "eval_dataset_uri": "lustre:///data/val",
            "inference_dataset_uri": "lustre:///data/test",
            "calibration_dataset_uri": "lustre:///data/calib"
        }
        is_valid, error = validate_all_dataset_uris(metadata, backend_type="slurm")
        assert is_valid is True
        assert error == ""

    def test_invalid_train_path(self):
        """Test that invalid train path is caught"""
        metadata = {
            "train_dataset_uris": ["/local/data"],  # Invalid for SLURM
            "eval_dataset_uri": "lustre:///data/val"
        }
        is_valid, error = validate_all_dataset_uris(metadata, backend_type="slurm")
        assert is_valid is False
        assert "train_dataset_uris" in error
        assert "not allowed for SLURM backend" in error

    def test_invalid_eval_path(self):
        """Test that invalid eval path is caught"""
        metadata = {
            "train_dataset_uris": ["lustre:///data/train"],
            "eval_dataset_uri": "/local/data"  # Invalid for SLURM
        }
        is_valid, error = validate_all_dataset_uris(metadata, backend_type="slurm")
        assert is_valid is False
        assert "eval_dataset_uri" in error

    def test_invalid_inference_path(self):
        """Test that invalid inference path is caught"""
        metadata = {
            "train_dataset_uris": ["lustre:///data/train"],
            "inference_dataset_uri": "/local/data"  # Invalid for SLURM
        }
        is_valid, error = validate_all_dataset_uris(metadata, backend_type="slurm")
        assert is_valid is False
        assert "inference_dataset_uri" in error

    def test_invalid_calibration_path(self):
        """Test that invalid calibration path is caught"""
        metadata = {
            "train_dataset_uris": ["lustre:///data/train"],
            "calibration_dataset_uri": "/local/data"  # Invalid for SLURM
        }
        is_valid, error = validate_all_dataset_uris(metadata, backend_type="slurm")
        assert is_valid is False
        assert "calibration_dataset_uri" in error

    def test_empty_metadata(self):
        """Test validation with no dataset paths"""
        metadata = {}
        is_valid, error = validate_all_dataset_uris(metadata, backend_type="slurm")
        assert is_valid is True
        assert error == ""

    def test_none_values(self):
        """Test validation with None values"""
        metadata = {
            "train_dataset_uris": None,
            "eval_dataset_uri": None,
            "inference_dataset_uri": None,
            "calibration_dataset_uri": None
        }
        is_valid, error = validate_all_dataset_uris(metadata, backend_type="slurm")
        assert is_valid is True
        assert error == ""

    def test_mixed_valid_and_none(self):
        """Test validation with mix of valid paths and None"""
        metadata = {
            "train_dataset_uris": ["lustre:///data/train"],
            "eval_dataset_uri": "lustre:///data/val",
            "inference_dataset_uri": None,
            "calibration_dataset_uri": None
        }
        is_valid, error = validate_all_dataset_uris(metadata, backend_type="slurm")
        assert is_valid is True
        assert error == ""

    def test_multiple_invalid_paths(self):
        """Test that first invalid path is reported"""
        metadata = {
            "train_dataset_uris": ["/local/data1", "/local/data2"],  # Both invalid for SLURM
            "eval_dataset_uri": "/local/val"  # Also invalid
        }
        is_valid, error = validate_all_dataset_uris(metadata, backend_type="slurm")
        assert is_valid is False
        # Should report first invalid path encountered
        assert "not allowed for SLURM backend" in error


class TestBackendRestrictions:
    """Tests specifically for backend restrictions"""

    @pytest.mark.parametrize("backend", REMOTE_BACKENDS)
    def test_local_paths_rejected_for_remote_backends(self, backend):
        """Test that local paths are rejected for all remote backends"""
        is_valid, error = validate_dataset_uri("/local/data", backend_type=backend)
        assert is_valid is False
        assert backend.upper() in error

    @pytest.mark.parametrize("backend", REMOTE_BACKENDS)
    def test_file_paths_rejected_for_remote_backends(self, backend):
        """Test that file:// paths are rejected for all remote backends"""
        is_valid, error = validate_dataset_uri("file:///local/data", backend_type=backend)
        assert is_valid is False
        assert backend.upper() in error

    @pytest.mark.parametrize("protocol", ["aws", "azure", "lustre"])
    @pytest.mark.parametrize("backend", REMOTE_BACKENDS)
    def test_cloud_paths_accepted_for_remote_backends(self, protocol, backend):
        """Test that cloud paths are accepted for all remote backends"""
        path = f"{protocol}://some/path"
        is_valid, error = validate_dataset_uri(path, backend_type=backend)
        assert is_valid is True
        assert error == ""

    def test_all_paths_accepted_for_local_backend(self):
        """Test that all valid paths are accepted for local backend"""
        paths = [
            "aws://bucket/data",
            "azure://container/data",
            "lustre:///scratch/data",
            "file:///local/data",
            "/local/data"
        ]
        for path in paths:
            is_valid, error = validate_dataset_uri(path, backend_type="local")
            assert is_valid is True, f"Path {path} should be valid for local backend"
            assert error == ""


class TestProtocolNormalization:
    """Tests for protocol normalization"""

    def test_s3_normalized_to_aws(self):
        """Test that s3:// protocol is normalized to aws://"""
        # Both should be valid
        is_valid_s3, _ = validate_dataset_uri("s3://bucket/path")
        is_valid_aws, _ = validate_dataset_uri("aws://bucket/path")

        assert is_valid_s3 is True
        assert is_valid_aws is True

    def test_case_insensitive_protocols(self):
        """Test that protocol matching is case-insensitive"""
        paths = [
            "AWS://bucket/path",
            "Azure://container/path",
            "LUSTRE:///path",
            "FILE:///path"
        ]
        for path in paths:
            is_valid, error = validate_dataset_uri(path)
            assert is_valid is True, f"Path {path} should be valid (case-insensitive)"


class TestErrorMessages:
    """Tests for error message quality"""

    def test_slurm_error_suggests_lustre(self):
        """Test that SLURM error suggests using lustre://"""
        _, error = validate_dataset_uri("/local/data", backend_type="slurm")
        assert "lustre://" in error.lower()

    def test_lepton_error_suggests_cloud(self):
        """Test that Lepton error suggests using cloud storage"""
        _, error = validate_dataset_uri("/local/data", backend_type="lepton")
        assert "aws://" in error.lower() or "azure://" in error.lower()

    def test_invalid_protocol_lists_valid_ones(self):
        """Test that invalid protocol error lists valid protocols"""
        _, error = validate_dataset_uri("ftp://server/path")
        for protocol in VALID_PROTOCOLS:
            # Check that valid protocols are mentioned (except 'local' which is implied by no prefix)
            if protocol != "local":
                assert protocol in error.lower()
