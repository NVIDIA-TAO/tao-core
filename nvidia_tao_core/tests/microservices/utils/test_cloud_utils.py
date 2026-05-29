# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for cloud workspace creation with real cloud providers.

These tests validate that CloudStorage instances can be created successfully
with real credentials from environment variables. Tests are skipped if
credentials are not available.

Environment variables required:
    AWS:
        - AWS_ACCESS_KEY_ID
        - AWS_SECRET_ACCESS_KEY
        - AWS_REGION (optional, defaults to us-east-1)
        - AWS_BUCKET_NAME

    Azure:
        - AZURE_ACCOUNT_NAME
        - AZURE_ACCOUNT_KEY
        - AZURE_CONTAINER_NAME

    Lepton:
        - LEPTON_ACCESS_KEY
        - LEPTON_SECRET_KEY
        - LEPTON_REGION (optional, defaults to us-east-1)
        - LEPTON_BUCKET_NAME
        - LEPTON_WORKSPACE_ID
        - LEPTON_AUTH_TOKEN

    SLURM:
        - SLURM_USER
        - SLURM_HOSTNAME (comma-separated list)
        - SLURM_BASE_RESULTS_DIR (optional)
        - SSH_KEY_PATH (optional, auto-detected if not provided)
"""

import os
import pytest
from unittest.mock import patch, Mock

from nvidia_tao_core.microservices.utils.cloud_utils import (
    CloudStorage,
    create_cs_instance,
    CloudStorageCredentialError,
    CloudStorageConnectionError
)


@pytest.mark.cloud_integration
class TestAWSWorkspaceCreation:
    """Test AWS workspace creation with real credentials from environment variables."""

    @pytest.fixture
    def aws_credentials(self):
        """Get AWS credentials from environment variables."""
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        region = os.getenv('AWS_REGION', 'us-east-1')
        bucket_name = os.getenv('AWS_BUCKET_NAME')

        if not all([access_key, secret_key, bucket_name]):
            pytest.skip("AWS credentials not available in environment variables")

        return {
            'access_key': access_key,
            'secret_key': secret_key,
            'region': region,
            'bucket_name': bucket_name
        }

    def test_aws_cloud_storage_creation(self, aws_credentials):
        """Test creating AWS CloudStorage instance with real credentials."""
        cs_instance = CloudStorage(
            cloud_type='aws',
            bucket_name=aws_credentials['bucket_name'],
            region=aws_credentials['region'],
            key=aws_credentials['access_key'],
            secret=aws_credentials['secret_key']
        )

        assert cs_instance is not None
        assert cs_instance.cloud_type == 'aws'
        assert cs_instance.bucket_name == aws_credentials['bucket_name']
        assert cs_instance.fs is not None

    def test_aws_workspace_validation(self, aws_credentials):
        """Test validating AWS workspace connection."""
        cs_instance = CloudStorage(
            cloud_type='aws',
            bucket_name=aws_credentials['bucket_name'],
            region=aws_credentials['region'],
            key=aws_credentials['access_key'],
            secret=aws_credentials['secret_key']
        )

        # Should not raise an exception if credentials are valid
        cs_instance.validate_connection()

    def test_aws_create_cs_instance(self, aws_credentials):
        """Test creating AWS CloudStorage via create_cs_instance."""
        workspace_metadata = {
            'cloud_type': 'aws',
            'cloud_specific_details': {
                'cloud_type': 'aws',
                'access_key': aws_credentials['access_key'],
                'secret_key': aws_credentials['secret_key'],
                'cloud_region': aws_credentials['region'],
                'cloud_bucket_name': aws_credentials['bucket_name']
            }
        }

        # Mock encryption to bypass vault - only mock VAULT_SECRET_PATH
        # Save reference to real os.getenv before patching
        real_getenv = os.getenv

        def mock_getenv_side_effect(key, default=None):
            if key == "VAULT_SECRET_PATH":
                return None
            return real_getenv(key, default)

        with patch('nvidia_tao_core.microservices.utils.cloud_utils.os.getenv', side_effect=mock_getenv_side_effect):
            cs_instance, cloud_details = create_cs_instance(workspace_metadata)

            assert cs_instance is not None
            assert cs_instance.cloud_type == 'aws'
            assert cloud_details['cloud_bucket_name'] == aws_credentials['bucket_name']

    def test_aws_invalid_credentials(self, aws_credentials):
        """Test that invalid AWS credentials raise appropriate errors."""
        with pytest.raises(Exception):  # Will be CloudStorageCredentialError or similar
            cs_instance = CloudStorage(
                cloud_type='aws',
                bucket_name=aws_credentials['bucket_name'],
                region=aws_credentials['region'],
                key='invalid_key',
                secret='invalid_secret'
            )
            cs_instance.validate_connection()


@pytest.mark.cloud_integration
class TestAzureWorkspaceCreation:
    """Test Azure workspace creation with real credentials from environment variables."""

    @pytest.fixture
    def azure_credentials(self):
        """Get Azure credentials from environment variables."""
        account_name = os.getenv('AZURE_ACCOUNT_NAME')
        account_key = os.getenv('AZURE_ACCOUNT_KEY')
        container_name = os.getenv('AZURE_CONTAINER_NAME')

        if not all([account_name, account_key, container_name]):
            pytest.skip("Azure credentials not available in environment variables")

        return {
            'account_name': account_name,
            'account_key': account_key,
            'container_name': container_name
        }

    def test_azure_cloud_storage_creation(self, azure_credentials):
        """Test creating Azure CloudStorage instance with real credentials."""
        cs_instance = CloudStorage(
            cloud_type='azure',
            bucket_name=azure_credentials['container_name'],
            key=azure_credentials['account_name'],
            secret=azure_credentials['account_key']
        )

        assert cs_instance is not None
        assert cs_instance.cloud_type == 'azure'
        assert cs_instance.bucket_name == azure_credentials['container_name']
        assert cs_instance.fs is not None

    def test_azure_workspace_validation(self, azure_credentials):
        """Test validating Azure workspace connection."""
        cs_instance = CloudStorage(
            cloud_type='azure',
            bucket_name=azure_credentials['container_name'],
            key=azure_credentials['account_name'],
            secret=azure_credentials['account_key']
        )

        # Should not raise an exception if credentials are valid
        cs_instance.validate_connection()

    def test_azure_create_cs_instance(self, azure_credentials):
        """Test creating Azure CloudStorage via create_cs_instance."""
        workspace_metadata = {
            'cloud_type': 'azure',
            'cloud_specific_details': {
                'cloud_type': 'azure',
                'account_name': azure_credentials['account_name'],
                'access_key': azure_credentials['account_key'],
                'cloud_bucket_name': azure_credentials['container_name']
            }
        }

        # Mock encryption to bypass vault - only mock VAULT_SECRET_PATH
        # Save reference to real os.getenv before patching
        real_getenv = os.getenv

        def mock_getenv_side_effect(key, default=None):
            if key == "VAULT_SECRET_PATH":
                return None
            return real_getenv(key, default)

        with patch('nvidia_tao_core.microservices.utils.cloud_utils.os.getenv', side_effect=mock_getenv_side_effect):
            cs_instance, cloud_details = create_cs_instance(workspace_metadata)

            assert cs_instance is not None
            assert cs_instance.cloud_type == 'azure'
            assert cloud_details['cloud_bucket_name'] == azure_credentials['container_name']

    def test_azure_invalid_credentials(self, azure_credentials):
        """Test that invalid Azure credentials raise appropriate errors."""
        with pytest.raises(Exception):  # Will be CloudStorageCredentialError or similar
            cs_instance = CloudStorage(
                cloud_type='azure',
                bucket_name=azure_credentials['container_name'],
                key='invalid_account',
                secret='invalid_key'
            )
            cs_instance.validate_connection()


@pytest.mark.cloud_integration
class TestLeptonWorkspaceCreation:
    """Test Lepton workspace creation with real credentials from environment variables."""

    @pytest.fixture
    def lepton_credentials(self):
        """Get Lepton credentials from environment variables."""
        access_key = os.getenv('LEPTON_ACCESS_KEY')
        secret_key = os.getenv('LEPTON_SECRET_KEY')
        region = os.getenv('LEPTON_REGION', 'us-east-1')
        bucket_name = os.getenv('LEPTON_BUCKET_NAME')
        workspace_id = os.getenv('LEPTON_WORKSPACE_ID')
        auth_token = os.getenv('LEPTON_AUTH_TOKEN')

        if not all([access_key, secret_key, bucket_name, workspace_id, auth_token]):
            pytest.skip("Lepton credentials not available in environment variables")

        return {
            'access_key': access_key,
            'secret_key': secret_key,
            'region': region,
            'bucket_name': bucket_name,
            'workspace_id': workspace_id,
            'auth_token': auth_token
        }

    def test_lepton_cloud_storage_creation(self, lepton_credentials):
        """Test creating Lepton CloudStorage instance with real credentials.

        Note: Lepton uses AWS S3 backend, so it creates an 'aws' type CloudStorage.
        Lepton requires lepton_workspace_id and lepton_auth_token in addition to AWS credentials.
        """
        workspace_metadata = {
            'cloud_type': 'lepton',
            'cloud_specific_details': {
                'cloud_type': 'lepton',
                'access_key': lepton_credentials['access_key'],
                'secret_key': lepton_credentials['secret_key'],
                'cloud_region': lepton_credentials['region'],
                'cloud_bucket_name': lepton_credentials['bucket_name'],
                'lepton_workspace_id': lepton_credentials.get('workspace_id', 'test-workspace-id'),
                'lepton_auth_token': lepton_credentials.get('auth_token', 'test-auth-token')
            }
        }

        # Mock encryption to bypass vault - only mock VAULT_SECRET_PATH
        # Save reference to real os.getenv before patching
        real_getenv = os.getenv

        def mock_getenv_side_effect(key, default=None):
            if key == "VAULT_SECRET_PATH":
                return None
            return real_getenv(key, default)

        with patch('nvidia_tao_core.microservices.utils.cloud_utils.os.getenv', side_effect=mock_getenv_side_effect):
            cs_instance, cloud_details = create_cs_instance(workspace_metadata)

            assert cs_instance is not None
            # Lepton uses AWS S3 backend internally
            assert cs_instance.cloud_type in ['aws', 'lepton']
            assert cloud_details['cloud_bucket_name'] == lepton_credentials['bucket_name']

    def test_lepton_workspace_validation(self, lepton_credentials):
        """Test validating Lepton workspace connection."""
        workspace_metadata = {
            'cloud_type': 'lepton',
            'cloud_specific_details': {
                'cloud_type': 'lepton',
                'access_key': lepton_credentials['access_key'],
                'secret_key': lepton_credentials['secret_key'],
                'cloud_region': lepton_credentials['region'],
                'cloud_bucket_name': lepton_credentials['bucket_name'],
                'lepton_workspace_id': lepton_credentials.get('workspace_id', 'test-workspace-id'),
                'lepton_auth_token': lepton_credentials.get('auth_token', 'test-auth-token')
            }
        }

        # Mock encryption to bypass vault - only mock VAULT_SECRET_PATH
        # Save reference to real os.getenv before patching
        real_getenv = os.getenv

        def mock_getenv_side_effect(key, default=None):
            if key == "VAULT_SECRET_PATH":
                return None
            return real_getenv(key, default)

        with patch('nvidia_tao_core.microservices.utils.cloud_utils.os.getenv', side_effect=mock_getenv_side_effect):
            cs_instance, _ = create_cs_instance(workspace_metadata)

            # Should not raise an exception if credentials are valid
            cs_instance.validate_connection()


@pytest.mark.cloud_integration
class TestSlurmWorkspaceCreation:
    """Test SLURM workspace creation with real credentials from environment variables."""

    @pytest.fixture
    def slurm_credentials(self):
        """Get SLURM credentials from environment variables."""
        slurm_user = os.getenv('SLURM_USER')
        slurm_hostname = os.getenv('SLURM_HOSTNAME')  # Comma-separated
        base_results_dir = os.getenv('SLURM_BASE_RESULTS_DIR')

        if not all([slurm_user, slurm_hostname]):
            pytest.skip("SLURM credentials not available in environment variables")

        # Parse hostname list
        hostname_list = [h.strip() for h in slurm_hostname.split(',')]

        return {
            'slurm_user': slurm_user,
            'slurm_hostname': hostname_list,
            'base_results_dir': base_results_dir or f"/lustre/fsw/portfolios/edgeai/users/{slurm_user}"
        }

    def test_slurm_workspace_creation(self, slurm_credentials):
        """Test creating SLURM workspace via create_cs_instance."""
        workspace_metadata = {
            'cloud_type': 'slurm',
            'cloud_specific_details': {
                'cloud_type': 'slurm',
                'slurm_user': slurm_credentials['slurm_user'],
                'slurm_hostname': slurm_credentials['slurm_hostname'],
                'base_results_dir': slurm_credentials['base_results_dir']
            }
        }

        cs_instance, cloud_details = create_cs_instance(workspace_metadata)

        assert cs_instance is not None
        assert cloud_details['slurm_user'] == slurm_credentials['slurm_user']
        assert cloud_details['slurm_hostname'] == slurm_credentials['slurm_hostname']

    def test_slurm_workspace_validation(self, slurm_credentials):
        """Test validating SLURM workspace connection."""
        workspace_metadata = {
            'cloud_type': 'slurm',
            'cloud_specific_details': {
                'cloud_type': 'slurm',
                'slurm_user': slurm_credentials['slurm_user'],
                'slurm_hostname': slurm_credentials['slurm_hostname'],
                'base_results_dir': slurm_credentials['base_results_dir']
            }
        }

        cs_instance, _ = create_cs_instance(workspace_metadata)

        # Should not raise an exception if connection is valid
        cs_instance.validate_connection()

    def test_slurm_missing_hostname_validation(self):
        """Test that SLURM workspace creation fails without hostname."""
        workspace_metadata = {
            'cloud_type': 'slurm',
            'cloud_specific_details': {
                'cloud_type': 'slurm',
                'slurm_user': 'testuser',
                # Missing slurm_hostname
            }
        }

        with pytest.raises(ValueError, match="SLURM workspace requires slurm_user and slurm_hostname"):
            create_cs_instance(workspace_metadata)

    def test_slurm_invalid_hostname_type(self):
        """Test that SLURM workspace creation fails with invalid hostname type."""
        workspace_metadata = {
            'cloud_type': 'slurm',
            'cloud_specific_details': {
                'cloud_type': 'slurm',
                'slurm_user': 'testuser',
                'slurm_hostname': 'not-a-list',  # Should be a list
            }
        }

        with pytest.raises(ValueError, match="SLURM slurm_hostname must be a list of strings"):
            create_cs_instance(workspace_metadata)


class TestCloudStorageUnitTests:
    """Unit tests for CloudStorage class (mocked, no real credentials needed)."""

    def test_unsupported_cloud_type(self):
        """Test that unsupported cloud type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported cloud_type"):
            CloudStorage(
                cloud_type='invalid_type',
                bucket_name='test-bucket',
                region='us-east-1'
            )

    def test_aws_cloud_storage_initialization(self):
        """Test AWS CloudStorage initialization with mocked fsspec."""
        with patch('nvidia_tao_core.microservices.utils.cloud_utils.fsspec.filesystem') as mock_fs:
            mock_fs.return_value = Mock()

            cs_instance = CloudStorage(
                cloud_type='aws',
                bucket_name='test-bucket',
                region='us-east-1',
                key='test-key',
                secret='test-secret'
            )

            assert cs_instance.cloud_type == 'aws'
            assert cs_instance.bucket_name == 'test-bucket'
            assert cs_instance.region == 'us-east-1'
            mock_fs.assert_called_once()

    def test_azure_cloud_storage_initialization(self):
        """Test Azure CloudStorage initialization with mocked fsspec."""
        with patch('nvidia_tao_core.microservices.utils.cloud_utils.fsspec.filesystem') as mock_fs:
            mock_fs.return_value = Mock()

            cs_instance = CloudStorage(
                cloud_type='azure',
                bucket_name='test-container',
                key='test-account',
                secret='test-key'
            )

            assert cs_instance.cloud_type == 'azure'
            assert cs_instance.bucket_name == 'test-container'
            mock_fs.assert_called_once()

            # Verify that Azure-specific parameters were passed correctly
            call_args = mock_fs.call_args
            assert call_args[0][0] == 'az'  # First positional arg should be 'az'
            # Check that account_name and account_key are in kwargs
            kwargs = call_args[1]
            assert 'account_name' in kwargs
            assert 'account_key' in kwargs

    def test_seaweedfs_cloud_storage_initialization(self):
        """Test SeaweedFS CloudStorage initialization with mocked fsspec."""
        with patch('nvidia_tao_core.microservices.utils.cloud_utils.fsspec.filesystem') as mock_fs:
            mock_fs.return_value = Mock()

            cs_instance = CloudStorage(
                cloud_type='seaweedfs',
                bucket_name='test-bucket',
                key='test-key',
                secret='test-secret',
                use_ssl=False,
                client_kwargs={'endpoint_url': 'http://localhost:8333'}
            )

            assert cs_instance.cloud_type == 'seaweedfs'
            assert cs_instance.bucket_name == 'test-bucket'
            mock_fs.assert_called_once()

    @patch('nvidia_tao_core.microservices.utils.cloud_utils.fsspec.filesystem')
    def test_validate_connection_success(self, mock_filesystem):
        """Test successful connection validation."""
        mock_fs_instance = Mock()
        mock_fs_instance.ls.return_value = []
        mock_filesystem.return_value = mock_fs_instance

        cs_instance = CloudStorage(
            cloud_type='aws',
            bucket_name='test-bucket',
            key='test-key',
            secret='test-secret'
        )

        # Should not raise exception
        cs_instance.validate_connection()
        mock_fs_instance.ls.assert_called_once()

    @patch('nvidia_tao_core.microservices.utils.cloud_utils.fsspec.filesystem')
    def test_validate_connection_invalid_credentials(self, mock_filesystem):
        """Test connection validation with invalid credentials."""
        mock_fs_instance = Mock()
        mock_fs_instance.ls.side_effect = Exception("InvalidAccessKeyId")
        mock_filesystem.return_value = mock_fs_instance

        cs_instance = CloudStorage(
            cloud_type='aws',
            bucket_name='test-bucket',
            key='invalid-key',
            secret='invalid-secret'
        )

        with pytest.raises(CloudStorageCredentialError):
            cs_instance.validate_connection()

    @patch('nvidia_tao_core.microservices.utils.cloud_utils.fsspec.filesystem')
    def test_validate_connection_bucket_not_found(self, mock_filesystem):
        """Test connection validation with non-existent bucket."""
        mock_fs_instance = Mock()
        mock_fs_instance.ls.side_effect = Exception("NoSuchBucket")
        mock_filesystem.return_value = mock_fs_instance

        cs_instance = CloudStorage(
            cloud_type='aws',
            bucket_name='non-existent-bucket',
            key='test-key',
            secret='test-secret'
        )

        with pytest.raises(CloudStorageConnectionError, match="does not exist"):
            cs_instance.validate_connection()

    def test_create_cs_instance_unsupported_type(self):
        """Test create_cs_instance with unsupported cloud type."""
        workspace_metadata = {
            'cloud_type': 'unsupported_type',
            'cloud_specific_details': {
                'cloud_type': 'unsupported_type'
            }
        }

        with pytest.raises(ValueError, match="Unsupported cloud_type: unsupported_type"):
            create_cs_instance(workspace_metadata)
