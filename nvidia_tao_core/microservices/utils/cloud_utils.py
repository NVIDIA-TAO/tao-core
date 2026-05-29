# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cloud storage fsspec client"""
import os
import copy
import time
import traceback
import fsspec
import logging
import functools
from datetime import datetime

# Lazy import to avoid circular dependencies
from nvidia_tao_core.distributed.decorators import master_node_only
from nvidia_tao_core.microservices.utils.stateless_handler_utils import report_health_beat
from nvidia_tao_core.microservices.utils.slurm_cloud_storage import SlurmCloudStorageAdapter

NUM_RETRY = 5

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


class CloudStorageCredentialError(Exception):
    """Exception raised for invalid cloud storage credentials."""

    pass


class CloudStorageConnectionError(Exception):
    """Exception raised when unable to connect to cloud storage."""

    pass


def clear_fsspec_caches():
    """Clear all fsspec caches to prevent state corruption."""
    try:
        # Clear fsspec registry caches
        if hasattr(fsspec, 'registry') and hasattr(fsspec.registry, 'clear_cache'):
            fsspec.registry.clear_cache()

        # Clear s3fs specific caches
        try:
            import s3fs
            if hasattr(s3fs, 'S3FileSystem'):
                if hasattr(s3fs.S3FileSystem, 'clear_instance_cache'):
                    s3fs.S3FileSystem.clear_instance_cache()
                if hasattr(s3fs.S3FileSystem, '_cache'):
                    s3fs.S3FileSystem._cache.clear()
            # Clear session pools in s3fs core
            if hasattr(s3fs, 'core') and hasattr(s3fs.core, '_cache'):
                s3fs.core._cache.clear()
        except ImportError:
            pass

        # Clear generic fsspec filesystem caches
        try:
            if hasattr(fsspec, 'filesystem'):
                # Clear any cached filesystem instances
                if hasattr(fsspec.filesystem, '_cache'):
                    fsspec.filesystem._cache.clear()
        except Exception:
            pass

        logger.info("Cleared all fsspec caches")
    except Exception as e:
        logger.warning(f"Could not clear some fsspec caches: {e}")


def reset_storage_state():
    """Reset storage client state to prevent corruption."""
    clear_fsspec_caches()

    # Force garbage collection to clean up any hanging connections
    import gc
    gc.collect()

    logger.info("Reset storage state and caches")


def retry_method(func):
    """Retry Cloud storage methods for NUM_RETRY times"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(NUM_RETRY):
            try:
                if attempt > 0:
                    # Clear caches before retry
                    logger.warning(f"Retrying {func.__name__} attempt {attempt + 1}/{NUM_RETRY}")
                    clear_fsspec_caches()
                    time.sleep(1)
                return func(*args, **kwargs)
            except Exception as e:
                # Log or handle the exception as needed
                logger.error("Exception in %s (attempt %d/%d): %s", func.__name__, attempt + 1, NUM_RETRY, e)
                if attempt == NUM_RETRY - 1:
                    # Last attempt failed
                    raise
        # If all retries fail, raise an exception or handle it accordingly
        raise ValueError(f"Failed to execute {func.__name__} after multiple retries")
    return wrapper


def create_cs_instance_with_decrypted_metadata(decrypted_metadata):
    """Create a cloud storage instance with decrypted metadata"""
    handler_metadata_copy = copy.deepcopy(decrypted_metadata)
    cloud_type = handler_metadata_copy.get("cloud_type", "aws")
    cloud_specific_details = handler_metadata_copy.get("cloud_specific_details", {})
    cloud_bucket_name = cloud_specific_details.get("cloud_bucket_name")
    if 'container_name' in cloud_specific_details:
        cloud_bucket_name = cloud_specific_details.get("container_name")

    # Original cloud providers
    cs_instance = None
    if cloud_specific_details and (cloud_bucket_name or cloud_type == "slurm"):
        if cloud_type == "aws":
            cs_instance = CloudStorage(
                cloud_type="aws",
                bucket_name=cloud_bucket_name,
                region=cloud_specific_details.get("cloud_region"),
                key=cloud_specific_details.get("access_key"),
                secret=cloud_specific_details.get("secret_key"),
                client_kwargs={"endpoint_url": cloud_specific_details.get("endpoint_url")}
            )
        elif cloud_type == "azure":
            cs_instance = CloudStorage(
                cloud_type="azure",
                bucket_name=cloud_bucket_name,
                region=cloud_specific_details.get("cloud_region"),
                key=cloud_specific_details.get("account_name"),
                secret=cloud_specific_details.get("access_key"),
                client_kwargs={"endpoint_url": cloud_specific_details.get("endpoint_url")}
            )
        elif cloud_type == "seaweedfs":
            return _create_seaweedfs_instance(cloud_specific_details)
        elif cloud_type == "lepton":
            # Lepton workspaces use cloud storage
            cloud_type = "aws"
            key = cloud_specific_details.get("access_key")
            secret = cloud_specific_details.get("secret_key")
            if 'account_name' in cloud_specific_details:
                cloud_type = "azure"
                key = cloud_specific_details.get("account_name")
                secret = cloud_specific_details.get("access_key")
            cs_instance = CloudStorage(
                cloud_type=cloud_type,
                bucket_name=cloud_bucket_name,
                region=cloud_specific_details.get("cloud_region"),
                key=key,
                secret=secret,
                client_kwargs={"endpoint_url": cloud_specific_details.get("endpoint_url")}
            )
        elif cloud_type == "slurm":
            # SLURM uses SSH-based remote filesystem access
            slurm_user = cloud_specific_details.get("slurm_user")
            slurm_hostname = cloud_specific_details.get("slurm_hostname")
            base_results_dir = cloud_specific_details.get("base_results_dir")

            if not base_results_dir:
                # Default path if not specified
                base_results_dir = f"/lustre/fsw/portfolios/edgeai/users/{slurm_user}"

            if not slurm_user or not slurm_hostname:
                raise ValueError("SLURM workspace requires slurm_user and slurm_hostname")

            # Validate hostname is a list
            if not isinstance(slurm_hostname, list):
                raise ValueError("SLURM slurm_hostname must be a list of strings")
            if not slurm_hostname:
                raise ValueError("SLURM hostname list cannot be empty")

            logger.info(f"SLURM workspace configured with {len(slurm_hostname)} hostname(s) for failover")

            # Use SSH key from environment variable or auto-detect from common locations
            ssh_key_path = os.getenv('SSH_KEY_PATH')
            if not ssh_key_path or not os.path.exists(ssh_key_path):
                # Try common locations for SSH keys
                for candidate in ["/root/.ssh/id_ed25519", "/root/.ssh/id_rsa",
                                  "/home/www-data/.ssh/id_ed25519", "/home/www-data/.ssh/id_rsa"]:
                    if os.path.exists(candidate):
                        ssh_key_path = candidate
                        logger.info(f"Auto-detected SSH key: {ssh_key_path}")
                        break
                else:
                    ssh_key_path = None
                    logger.warning("No SSH key found for SLURM connection")

            cs_instance = SlurmCloudStorageAdapter(
                slurm_user=slurm_user,
                slurm_hostname=slurm_hostname,
                base_results_dir=base_results_dir,
                ssh_key_path=ssh_key_path
            )
        else:
            raise ValueError(f"Unsupported cloud_type: {cloud_type}")

    return cs_instance, cloud_specific_details


def create_cs_instance(workspace_metadata):
    """Create a cloud storage instance based on handler metadata

    Args:
        handler_metadata (dict): Metadata containing cloud configuration

    Returns:
        tuple: (CloudStorage instance, cloud_specific_details dict)

    Raises:
        CloudStorageCredentialError: If credentials are invalid
        CloudStorageConnectionError: If unable to connect to cloud storage
        ValueError: If configuration is invalid
    """
    # Clear caches before creating new instance
    clear_fsspec_caches()

    handler_metadata_copy = copy.deepcopy(workspace_metadata)
    cloud_type = handler_metadata_copy.get("cloud_type", "aws")
    cloud_specific_details = handler_metadata_copy.get("cloud_specific_details", {})

    # Decrypt cloud details for original cloud providers (not needed for Slurm)
    if cloud_type != "slurm":
        config_path = os.getenv("VAULT_SECRET_PATH", None)
        if config_path:
            from .encrypt_utils import NVVaultEncryption
            encryption = NVVaultEncryption(config_path)
            for key, encrypted_value in cloud_specific_details.items():
                if encryption.check_config()[0]:
                    cloud_specific_details[key] = encryption.decrypt(encrypted_value)

    cloud_bucket_name = cloud_specific_details.get("cloud_bucket_name")
    if 'container_name' in cloud_specific_details:
        cloud_bucket_name = cloud_specific_details.get("container_name")

    cs_instance = None
    if cloud_specific_details:
        if cloud_type == "aws":
            cs_instance = CloudStorage(
                cloud_type="aws",
                bucket_name=cloud_bucket_name,
                region=cloud_specific_details.get("cloud_region"),
                key=cloud_specific_details.get("access_key"),
                secret=cloud_specific_details.get("secret_key"),
                client_kwargs={"endpoint_url": cloud_specific_details.get("endpoint_url")}
            )
        elif cloud_type == "azure":
            cs_instance = CloudStorage(
                cloud_type="azure",
                bucket_name=cloud_bucket_name,
                region=cloud_specific_details.get("cloud_region"),
                key=cloud_specific_details.get("account_name"),
                secret=cloud_specific_details.get("access_key"),
                client_kwargs={"endpoint_url": cloud_specific_details.get("endpoint_url")}
            )
        elif cloud_type == "seaweedfs":
            cs_instance, _ = _create_seaweedfs_instance(cloud_specific_details)
        elif cloud_type == "lepton":
            # Lepton workspaces use AWS S3 for storage with the same credential structure as AWS
            cloud_type = "aws"
            key = cloud_specific_details.get("access_key")
            secret = cloud_specific_details.get("secret_key")
            if 'account_name' in cloud_specific_details:
                cloud_type = "azure"
                key = cloud_specific_details.get("account_name")
                secret = cloud_specific_details.get("access_key")
            cs_instance = CloudStorage(
                cloud_type=cloud_type,
                bucket_name=cloud_bucket_name,
                region=cloud_specific_details.get("cloud_region"),
                key=key,
                secret=secret,
                client_kwargs={"endpoint_url": cloud_specific_details.get("endpoint_url")}
            )
        elif cloud_type == "slurm":
            # SLURM uses SSH-based remote filesystem access
            slurm_user = cloud_specific_details.get("slurm_user")
            slurm_hostname = cloud_specific_details.get("slurm_hostname")
            base_results_dir = cloud_specific_details.get("base_results_dir")

            if not base_results_dir:
                # Default path if not specified
                base_results_dir = f"/lustre/fsw/portfolios/edgeai/users/{slurm_user}"

            if not slurm_user or not slurm_hostname:
                raise ValueError("SLURM workspace requires slurm_user and slurm_hostname")

            # Validate hostname is a list
            if not isinstance(slurm_hostname, list):
                raise ValueError("SLURM slurm_hostname must be a list of strings")
            if not slurm_hostname:
                raise ValueError("SLURM hostname list cannot be empty")

            logger.info(f"SLURM workspace configured with {len(slurm_hostname)} hostname(s) for failover")

            # Use SSH key from environment variable or auto-detect from common locations
            ssh_key_path = os.getenv('SSH_KEY_PATH')
            if not ssh_key_path or not os.path.exists(ssh_key_path):
                # Try common locations for SSH keys
                for candidate in ["/root/.ssh/id_ed25519", "/root/.ssh/id_rsa",
                                  "/home/www-data/.ssh/id_ed25519", "/home/www-data/.ssh/id_rsa"]:
                    if os.path.exists(candidate):
                        ssh_key_path = candidate
                        logger.info(f"Auto-detected SSH key: {ssh_key_path}")
                        break
                else:
                    ssh_key_path = None
                    logger.warning("No SSH key found for SLURM connection")

            cs_instance = SlurmCloudStorageAdapter(
                slurm_user=slurm_user,
                slurm_hostname=slurm_hostname,
                base_results_dir=base_results_dir,
                ssh_key_path=ssh_key_path
            )
        else:
            raise ValueError(f"Unsupported cloud_type: {cloud_type}")

    # Validate connection for all cloud providers including SLURM
    if cs_instance and (cloud_bucket_name or cloud_type == "slurm"):
        logger.info(f"Validating {cloud_type} credentials...")
        try:
            # Directly validate the connection using the instance method
            cs_instance.validate_connection()
            logger.info("Credentials validated successfully")
        except (CloudStorageCredentialError, CloudStorageConnectionError) as e:
            logger.error(f"Credential validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during credential validation: {e}")
            raise CloudStorageConnectionError(f"Failed to validate credentials: {e}") from e

    return cs_instance, cloud_specific_details


def _create_seaweedfs_instance(cloud_specific_details):
    """Create a SeaweedFS storage instance"""
    # SeaweedFS configuration with environment variable fallbacks
    endpoint = cloud_specific_details.get("endpoint_url", "http://seaweedfs-s3:8333")
    bucket_name = cloud_specific_details.get("cloud_bucket_name", "tao-storage")
    access_key = cloud_specific_details.get("access_key", "")
    secret_key = cloud_specific_details.get("secret_key", "")
    region = cloud_specific_details.get("cloud_region", "us-east-1")

    # Create SeaweedFS-compatible storage instance
    seaweedfs_instance = CloudStorage(
        cloud_type="seaweedfs",
        bucket_name=bucket_name,
        region=region,
        key=access_key,
        secret=secret_key,
        use_ssl=False,
        client_kwargs={
            'endpoint_url': endpoint,
            'addressing_style': 'path'
        }
    )

    return seaweedfs_instance, cloud_specific_details


def get_date_time():
    """Get current date and time"""
    now = datetime.now()
    return now.strftime("%m/%d/%Y"), now.strftime("%H:%M:%S")


class CloudStorage:
    """Cloud storage CRUD operations using fsspec for S3, Azure, and SeaweedFS."""

    @retry_method
    def __init__(self, cloud_type, bucket_name, region=None, **kwargs):
        """Initialize the CloudStorage object.

        cloud_type: 'aws', 'azure', or 'seaweedfs'.
        bucket_name: Name of the bucket/container.
        region: Region for the cloud storage provider.
        kwargs: Additional arguments for fsspec.filesystem
        """
        # Clear caches before creating filesystem
        clear_fsspec_caches()

        self.cloud_type = cloud_type
        self.bucket_name = bucket_name
        self.region = region
        self.fs = None

        # Prepare client_kwargs with region if provided
        if 'client_kwargs' not in kwargs:
            kwargs['client_kwargs'] = {}
        if region:
            kwargs['client_kwargs']['region_name'] = region

        # Configure for reliability - disable filesystem caching
        fsspec_config = {
            'use_listings_cache': False,  # Disable listings cache
            'listings_expiry_time': 0,    # Immediately expire any cached listings
            'skip_instance_cache': True   # Skip fsspec instance caching
        }

        if cloud_type in ['aws', 'lepton']:
            # Merge fsspec config with user kwargs
            aws_kwargs = {**kwargs, **fsspec_config}
            self.fs = fsspec.filesystem('s3', **aws_kwargs)
            self.root = f'{bucket_name}/'
        elif cloud_type == 'azure':
            # Azure requires account_name and account_key at the top level
            # Extract credentials from kwargs
            account_name = kwargs.pop('key', None)
            account_key = kwargs.pop('secret', None)

            # Merge fsspec config with user kwargs
            azure_kwargs = {
                'account_name': account_name,
                'account_key': account_key,
                **kwargs,
                **fsspec_config
            }
            self.fs = fsspec.filesystem('az', **azure_kwargs)
            self.root = f'{bucket_name}/'
        elif cloud_type == 'seaweedfs':
            # SeaweedFS uses S3-compatible API
            fsspec_kwargs = kwargs.copy()
            client_kwargs = fsspec_kwargs.pop('client_kwargs', {})

            # Use client_kwargs approach for maximum compatibility (works in both local and Docker)
            if client_kwargs:
                # Filter out incompatible parameters like 'addressing_style'
                client_kwargs_filtered = {
                    k: v for k, v in client_kwargs.items()
                    if k != 'addressing_style'
                }
                if client_kwargs_filtered:
                    fsspec_kwargs['client_kwargs'] = client_kwargs_filtered

            # Merge fsspec config for SeaweedFS reliability
            seaweedfs_kwargs = {**fsspec_kwargs, **fsspec_config}

            self.fs = fsspec.filesystem('s3', **seaweedfs_kwargs)
            self.root = f'{bucket_name}/'
        else:
            raise ValueError(
                f"Unsupported cloud_type: '{cloud_type}'. "
                "Use 'aws', 'azure', or 'seaweedfs'. "
                "For SLURM, use SlurmCloudStorageAdapter directly via create_cs_instance()."
            )

    def reset_filesystem_state(self):
        """Reset the filesystem state to clear any corrupted caches."""
        try:
            # Clear filesystem-specific caches
            if hasattr(self.fs, 'invalidate_cache'):
                self.fs.invalidate_cache()
            if hasattr(self.fs, '_cache') and hasattr(self.fs._cache, 'clear'):
                self.fs._cache.clear()

            # Clear global fsspec caches
            clear_fsspec_caches()

            logger.info("Filesystem state reset complete")
        except Exception as e:
            logger.warning(f"Error resetting filesystem state: {e}")

    def validate_connection(self):
        """Validate the connection to cloud storage by performing a basic operation.

        Raises:
            CloudStorageCredentialError: If credentials are invalid
            CloudStorageConnectionError: If unable to connect to cloud storage
        """
        try:
            # Test basic connectivity by listing the bucket
            bucket_path = f"{self.bucket_name}/"
            self.fs.ls(bucket_path, detail=False)
            logger.info(f"Successfully validated connection to {self.cloud_type} bucket: {self.bucket_name}")
        except Exception as e:
            error_msg = str(e).lower()
            if "nosuchbucket" in error_msg or "containernotfound" in error_msg or "does not exist" in error_msg:
                raise CloudStorageConnectionError(
                    f"Bucket/container '{self.bucket_name}' does not exist or is not accessible"
                ) from e
            if any(keyword in error_msg for keyword in [
                "invalidaccesskeyid", "signaturedoesnotmatch", "authenticationfailed", "unauthorized", "forbidden"
            ]):
                raise CloudStorageCredentialError(
                    f"Invalid credentials for {self.cloud_type}: {str(e)}"
                ) from e
            raise CloudStorageConnectionError(
                f"Failed to connect to {self.cloud_type}: {str(e)}"
            ) from e

    @retry_method
    def is_file(self, cloud_path):
        """Check if the given cloud path is a file."""
        full_path = self.root + cloud_path.strip('/')
        try:
            return self.fs.isfile(full_path)
        except Exception as e:
            logger.error(f"is_file error: {e}")
            return False

    @retry_method
    def is_folder(self, cloud_path):
        """Check if the given cloud path is a folder."""
        full_path = self.root + cloud_path.strip('/') + '/'
        # Azure specific path handling
        if self.cloud_type == 'azure':
            # For Azure, remove trailing slash and use isdir without it
            full_path = full_path.rstrip('/')
        try:
            return self.fs.isdir(full_path)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"is_folder error: {e}")
            return False

    @retry_method
    def glob_files(self, pattern):
        """Return a list of files matching the pattern."""
        full_pattern = self.root + pattern
        try:
            return [p[len(self.root):] for p in self.fs.glob(full_pattern) if self.fs.isfile(p)]
        except Exception as e:
            logger.error(f"glob_files error: {e}")
            return []

    @retry_method
    def list_files_in_folder(self, folder):
        """Recursively list files in the specified folder and its subfolders."""
        try:
            # Normalize folder path
            folder_normalized = folder.strip('/')
            full_path = self.root + folder_normalized + '/' if folder_normalized else self.root

            # Azure Blob Storage find() doesn't work well with trailing slashes
            if self.cloud_type == 'azure' and full_path.endswith('/'):
                full_path = full_path.rstrip('/')

            # Clear filesystem cache before operation for SeaweedFS
            if self.cloud_type == 'seaweedfs':
                try:
                    if hasattr(self.fs, 'invalidate_cache'):
                        self.fs.invalidate_cache()
                    if hasattr(self.fs, '_cache') and hasattr(self.fs._cache, 'clear'):
                        self.fs._cache.clear()
                except Exception as cache_err:
                    logger.warning(f"Could not clear filesystem cache: {cache_err}")

            # Use fsspec for all cloud providers (unified approach)
            all_paths = self.fs.find(full_path)
            logger.info(f"Found {len(all_paths)} paths total")

            file_names = []
            for p in all_paths:
                try:
                    if self.fs.isfile(p):
                        file_names.append(p[len(self.root):])
                except Exception as file_check_err:
                    logger.warning(f"Error checking if {p} is file: {file_check_err}")

            logger.info(f"Found {len(file_names)} files: {file_names[:5]}{'...' if len(file_names) > 5 else ''}")
            return file_names, []  # Details not available recursively
        except Exception as e:
            traceback.print_exc()
            logger.error(f"list_files_in_folder error: {e}")
            return [], []

    @retry_method
    def download_file(self, cloud_file_path, local_destination, progress_tracker=None):
        """Download a file from cloud storage to local destination.

        Args:
            cloud_file_path (str): Path to file in cloud storage
            local_destination (str): Local destination path
            progress_tracker (ProgressTracker, optional): Progress tracker instance
        """
        from nvidia_tao_core.microservices.handlers.cloud_handlers.progress_tracker import ProgressTracker
        from nvidia_tao_core.microservices.handlers.cloud_handlers.progress_tracker_utils import (
            send_progress_status_callback
        )

        full_path = self.root + cloud_file_path.strip('/')
        try:
            if os.path.exists(local_destination):
                logger.info(f"File {local_destination} already exists, skipping download")
                return

            # Get file size for progress tracking
            try:
                file_info = self.fs.info(full_path)
                file_size_mb = file_info.get('size', 0) / (1024 * 1024) if 'size' in file_info else 0.0
            except Exception:
                file_size_mb = 0.0

            # Only create new progress tracker if none provided AND this is a standalone download
            create_own_tracker = progress_tracker is None
            if create_own_tracker:
                progress_tracker = ProgressTracker("download", total_files=1, total_size_mb=file_size_mb)
            else:
                # Mark that we're starting to download this file
                # Include parent folder for better identification
                path_parts = cloud_file_path.strip('/').split('/')
                if len(path_parts) >= 2:
                    file_name = f"{path_parts[-2]}/{os.path.basename(cloud_file_path)}"
                else:
                    file_name = os.path.basename(cloud_file_path)
                progress_tracker.start_file_download(file_name=file_name, file_size_mb=file_size_mb)

            logger.info(
                f"Starting download: {cloud_file_path} -> {local_destination} ({file_size_mb:.1f} MB)"
            )

            # Use streaming download for files >10MB to provide size-based progress
            if file_size_mb > 10:
                self._download_file_with_streaming_progress(
                    full_path, local_destination, file_size_mb, cloud_file_path, progress_tracker
                )
            else:
                self.fs.download(full_path, local_destination)
                if not create_own_tracker:
                    progress_tracker.update_progress(size_processed_mb=file_size_mb)
                    progress_tracker.complete_file_download()

            if create_own_tracker:
                progress_tracker.complete()

            logger.info(f"Downloaded {cloud_file_path} to {local_destination}")
        except Exception as e:
            error_msg = f"Error downloading file {cloud_file_path} to {local_destination}: {str(e)}"
            logger.error(f"download_file error: {e}")

            # Send error callback
            send_progress_status_callback(error_msg)
            raise

    @retry_method
    def is_file_modified(self, cloud_path, local_path):
        """Check if a cloud storage file has been modified compared to the local file using fsspec.

        Args:
            cloud_path (str): Path to the file in cloud storage (relative to bucket root)
            local_path (str): Path to the local file

        Returns:
            bool: True if the cloud file is newer than the local file, False otherwise
        """
        try:
            # Get local file modification time
            if not os.path.exists(local_path):
                logger.warning(f"Local file {local_path} does not exist")
                return True  # Consider it modified if local file doesn't exist

            current_last_modified = os.path.getmtime(local_path)
            if not cloud_path.startswith(self.root):
                cloud_path = self.root + cloud_path.strip('/')

            # Get cloud file info using fsspec
            logger.info(f"Checking Cloud path: {cloud_path}")
            file_info = self.fs.info(cloud_path)
            last_modified = file_info.get('LastModified', datetime.now()).timestamp()
            if 'LastModified' not in file_info:
                logger.warning(f"Could not get last modified time for cloud file {cloud_path}, "
                               f"defaulting to current time")
            return last_modified > current_last_modified
        except Exception as e:
            logger.error(f"Error checking file modification for {cloud_path}: {e}")
            return False

    @retry_method
    def download_folder(self, cloud_folder, local_destination,
                        maintain_src_folder_structure=False, progress_tracker=None, extensions=[],
                        exclude_filenames=None):
        """Download a folder from cloud storage to local destination with progress tracking."""
        from nvidia_tao_core.microservices.handlers.cloud_handlers.progress_tracker import ProgressTracker
        from nvidia_tao_core.microservices.handlers.cloud_handlers.progress_tracker_utils import (
            send_progress_status_callback
        )
        # Normalize path to avoid double slashes
        cloud_folder_normalized = cloud_folder.strip('/')
        full_path = self.root + cloud_folder_normalized + '/' if cloud_folder_normalized else self.root

        try:
            # Get folder statistics for progress tracking
            files = self.fs.find(full_path)
            file_paths = [f for f in files if self.fs.isfile(f)]
            file_count = len(file_paths)

            # Calculate total size
            total_size_mb = 0.0
            for file_path in file_paths:
                try:
                    file_info = self.fs.info(file_path)
                    total_size_mb += file_info.get('size', 0) / (1024 * 1024) if 'size' in file_info else 0.0
                except Exception:
                    continue

            logger.info(f"Starting folder download: {cloud_folder} -> {local_destination}")
            logger.info(f"Folder contains {file_count} files ({total_size_mb:.1f} MB total)")

            # Create progress tracker only if not provided
            create_own_tracker = progress_tracker is None
            if create_own_tracker:
                progress_tracker = ProgressTracker("download", total_files=file_count, total_size_mb=total_size_mb)

            # Use fsspec for all cloud providers (unified approach)
            if maintain_src_folder_structure:
                if not os.path.exists(local_destination):
                    self.fs.download(full_path, local_destination, recursive=True)
                else:
                    logger.info(f"Folder {local_destination} already exists, skipping download")
                    return

                # For large folders, download with progress tracking
                if file_count > 10 or total_size_mb > 100:
                    self._download_folder_with_progress(
                        file_paths, full_path, local_destination, progress_tracker,
                        maintain_src_folder_structure=True, cloud_folder_normalized=cloud_folder_normalized,
                        extensions=extensions, exclude_filenames=exclude_filenames
                    )
                else:
                    # Download maintaining the source folder structure (small folders)
                    self.fs.download(full_path, local_destination, recursive=True)
                    # Mark all files as downloaded
                    for _ in range(file_count):
                        progress_tracker.start_file_download()
                        progress_tracker.complete_file_download()
                    progress_tracker.update_progress(size_processed_mb=total_size_mb)
            else:
                # Download contents without the source folder structure
                os.makedirs(local_destination, exist_ok=True)
                self._download_folder_with_progress(
                    file_paths, full_path, local_destination, progress_tracker,
                    maintain_src_folder_structure=False, extensions=extensions,
                    exclude_filenames=exclude_filenames
                )

            if create_own_tracker:
                progress_tracker.complete()
            logger.info(f"Downloaded folder {cloud_folder} to {local_destination}")
        except Exception as e:
            error_msg = f"Error downloading folder {cloud_folder} to {local_destination}: {str(e)}"
            logger.error(f"download_folder error: {e}")

            # Send error callback
            send_progress_status_callback(error_msg)
            raise

    def _download_folder_with_progress(self, file_paths, full_path, local_destination, progress_tracker,
                                       maintain_src_folder_structure=False, cloud_folder_normalized="", extensions=[],
                                       exclude_filenames=None):
        """Download folder contents with detailed progress tracking."""
        try:
            for file_path in file_paths:
                try:
                    if exclude_filenames and os.path.basename(file_path) in exclude_filenames:
                        logger.info("Skipping excluded file during folder download: %s", file_path)
                        continue
                    if maintain_src_folder_structure:
                        # Maintain original folder structure
                        relative_path = file_path[len(self.root + cloud_folder_normalized + '/'):]
                        local_file_path = os.path.join(local_destination, cloud_folder_normalized, relative_path)
                    else:
                        # Flatten structure
                        relative_path = file_path[len(full_path):]
                        local_file_path = os.path.join(local_destination, relative_path)
                        if extensions and not any(file_path.endswith(ext) for ext in extensions):
                            continue
                        if self.is_file_modified(file_path, local_file_path):
                            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                            self.fs.download(file_path, local_file_path)
                            logger.info(f"Downloaded {file_path} to {local_file_path}")
                        else:
                            logger.info(
                                f"File {local_file_path} in folder {local_destination} "
                                f"already exists, skipping download"
                            )

                    # Skip if file already exists
                    if os.path.exists(local_file_path):
                        logger.debug(f"File {local_file_path} already exists, skipping download")
                        # Still count it as processed for progress tracking
                        try:
                            file_info = self.fs.info(file_path)
                            file_size_mb = file_info.get('size', 0) / (1024 * 1024) if 'size' in file_info else 0.0
                        except Exception:
                            file_size_mb = 0.0
                        file_name = os.path.basename(file_path)
                        progress_tracker.start_file_download(file_name=file_name, file_size_mb=file_size_mb)
                        progress_tracker.update_progress(size_processed_mb=file_size_mb)
                        progress_tracker.complete_file_download()
                        continue

                    # Create directory if needed
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                    # Get file size
                    try:
                        file_info = self.fs.info(file_path)
                        file_size_mb = file_info.get('size', 0) / (1024 * 1024) if 'size' in file_info else 0.0
                    except Exception:
                        file_size_mb = 0.0

                    # Mark file download as started
                    file_name = os.path.basename(file_path)
                    progress_tracker.start_file_download(file_name=file_name, file_size_mb=file_size_mb)

                    # Use streaming download for large files
                    if file_size_mb > 50:
                        self._download_file_with_streaming_progress(
                            file_path, local_file_path, file_size_mb,
                            file_name, progress_tracker
                        )
                    else:
                        # Download individual file
                        self.fs.download(file_path, local_file_path)
                        # Update progress with the file size (streaming does this automatically)
                        progress_tracker.update_progress(size_processed_mb=file_size_mb)

                    # Mark file download as complete
                    progress_tracker.complete_file_download()

                except Exception as file_err:
                    logger.warning(f"Failed to download {file_path}: {file_err}")
                    continue

        except Exception as e:
            logger.error(f"Error in _download_folder_with_progress: {e}")
            raise

    def _download_file_with_streaming_progress(self, cloud_full_path, local_destination,
                                               file_size_mb, display_name, external_progress_tracker=None):
        """Download a file with real-time streaming progress updates.

        Args:
            cloud_full_path (str): Full path in cloud storage
            local_destination (str): Local destination path
            file_size_mb (float): File size in MB
            display_name (str): Display name for progress messages
            external_progress_tracker (ProgressTracker, optional): External progress tracker to use
        """
        from nvidia_tao_core.microservices.handlers.cloud_handlers.progress_tracker import ProgressTracker
        from nvidia_tao_core.microservices.handlers.cloud_handlers.progress_tracker_utils import (
            send_progress_status_callback
        )

        # Use external progress tracker if provided, otherwise create streaming tracker
        if external_progress_tracker:
            streaming_tracker = external_progress_tracker
            manage_tracker = False  # Don't call complete() on external tracker
        else:
            streaming_tracker = ProgressTracker(
                "download", total_files=1, total_size_mb=file_size_mb, file_name=display_name
            )
            manage_tracker = True

        try:
            # Create destination directory if needed
            os.makedirs(os.path.dirname(local_destination), exist_ok=True)

            # Use fsspec to open the file and read in chunks
            chunk_size = 8 * 1024 * 1024  # 8MB chunks for good progress granularity

            with self.fs.open(cloud_full_path, 'rb') as src_file:
                with open(local_destination, 'wb') as dst_file:
                    while True:
                        chunk = src_file.read(chunk_size)
                        if not chunk:
                            break

                        dst_file.write(chunk)
                        # Update progress with the chunk size
                        chunk_size_mb = len(chunk) / (1024 * 1024)
                        streaming_tracker.update_progress(size_processed_mb=chunk_size_mb)

            # Only call complete() if we created our own tracker (not using external)
            if manage_tracker:
                streaming_tracker.complete()

        except Exception as e:
            error_msg = f"Error in streaming download of {display_name}: {str(e)}"
            logger.error(error_msg)
            send_progress_status_callback(error_msg)
            raise

    def _upload_file_with_streaming_progress(self, local_file_path, cloud_full_path,
                                             file_size_mb, display_name,
                                             external_progress_tracker=None):
        """Upload a file with real-time streaming progress updates.

        Args:
            local_file_path (str): Local file path
            cloud_full_path (str): Full path in cloud storage
            file_size_mb (float): File size in MB
            display_name (str): Display name for progress messages
            external_progress_tracker (ProgressTracker, optional): External progress tracker to use
        """
        from nvidia_tao_core.microservices.handlers.cloud_handlers.progress_tracker import ProgressTracker
        from nvidia_tao_core.microservices.handlers.cloud_handlers.progress_tracker_utils import (
            send_progress_status_callback
        )

        # Use external progress tracker if provided, otherwise create streaming tracker
        if external_progress_tracker:
            streaming_tracker = external_progress_tracker
            manage_tracker = False  # Don't call complete() on external tracker
        else:
            streaming_tracker = ProgressTracker("upload", total_files=1, total_size_mb=file_size_mb,
                                                file_name=display_name)
            manage_tracker = True

        try:
            # Use fsspec to open the destination and write in chunks
            chunk_size = 8 * 1024 * 1024  # 8MB chunks for good progress granularity

            with open(local_file_path, 'rb') as src_file:
                with self.fs.open(cloud_full_path, 'wb') as dst_file:
                    while True:
                        chunk = src_file.read(chunk_size)
                        if not chunk:
                            break

                        dst_file.write(chunk)
                        # Update progress with the chunk size
                        chunk_size_mb = len(chunk) / (1024 * 1024)
                        streaming_tracker.update_progress(size_processed_mb=chunk_size_mb)

            # Mark file as complete when streaming upload finishes (if using external tracker)
            if not manage_tracker and external_progress_tracker:
                external_progress_tracker.complete_file_download()

            if manage_tracker:
                streaming_tracker.complete()

        except Exception as e:
            error_msg = f"Error in streaming upload of {display_name}: {str(e)}"
            logger.error(error_msg)
            send_progress_status_callback(error_msg)
            raise

    @retry_method
    @master_node_only
    def create_folder_in_bucket(self, folder):
        """Create a folder in the bucket."""
        full_path = self.root + folder.rstrip('/') + '/'
        try:
            # Create an empty object to represent the folder
            self.fs.makedirs(full_path, exist_ok=True)
            logger.info(f"Created folder {folder}")
        except Exception as e:
            logger.error(f"create_folder_in_bucket error: {e}")
            raise

    @retry_method
    @master_node_only
    def upload_file(self, local_file_path, cloud_file_path, progress_tracker=None, send_status_callbacks=True):
        """Upload a file from local storage to cloud.

        Args:
            local_file_path (str): Path to local file
            cloud_file_path (str): Path in cloud storage
            progress_tracker (ProgressTracker, optional): Progress tracker instance
            send_status_callbacks (bool): Whether to send status callbacks (False for background uploads)
        """
        from nvidia_tao_core.microservices.handlers.cloud_handlers.progress_tracker import ProgressTracker
        from nvidia_tao_core.microservices.handlers.cloud_handlers.progress_tracker_utils import (
            get_file_size_mb, send_progress_status_callback
        )

        full_path = self.root + cloud_file_path.strip('/')
        try:
            # Get file size for progress tracking
            file_size_mb = get_file_size_mb(local_file_path)

            # Only create new progress tracker if none provided AND this is a standalone upload
            create_own_tracker = progress_tracker is None
            if create_own_tracker:
                progress_tracker = ProgressTracker("upload", total_files=1, total_size_mb=file_size_mb,
                                                   send_callbacks=send_status_callbacks)
            else:
                # Mark that we're starting to upload this file
                # Include parent folder for better identification
                path_parts = local_file_path.strip('/').split('/')
                if len(path_parts) >= 2:
                    file_name = f"{path_parts[-2]}/{os.path.basename(local_file_path)}"
                else:
                    file_name = os.path.basename(local_file_path)
                progress_tracker.start_file_download(file_name=file_name, file_size_mb=file_size_mb)

            logger.info(f"Starting upload: {local_file_path} -> {cloud_file_path} ({file_size_mb:.1f} MB)")

            # Use streaming upload for large files (>50MB)
            if file_size_mb > 50:
                self._upload_file_with_streaming_progress(
                    local_file_path, full_path, file_size_mb, cloud_file_path, progress_tracker
                )
                # File completion will be marked inside the streaming function
            else:
                self.fs.upload(local_file_path, full_path)
                # Update progress for small files
                if create_own_tracker:
                    # For standalone uploads, update progress before completing
                    progress_tracker.update_progress(files_processed=1, size_processed_mb=file_size_mb)
                else:
                    # For batch uploads, update size and mark file complete
                    progress_tracker.update_progress(size_processed_mb=file_size_mb)
                    progress_tracker.complete_file_download()

            # Complete our own tracker, but don't interfere with external ones
            if create_own_tracker:
                progress_tracker.complete()

            logger.info(f"Uploaded {local_file_path} to {cloud_file_path}")
        except Exception as e:
            error_msg = f"Error uploading file {local_file_path} to {cloud_file_path}: {str(e)}"
            logger.error(f"upload_file error: {e}")

            # Send error callback
            send_progress_status_callback(error_msg)
            raise

    @retry_method
    @master_node_only
    def upload_folder(self, local_folder, cloud_subfolder, send_status_callbacks=True):
        """Upload a folder from local storage to cloud with progress tracking.

        Args:
            local_folder (str): Local folder path
            cloud_subfolder (str): Cloud destination path
            send_status_callbacks (bool): Whether to send status callbacks (False for background uploads)
        """
        from nvidia_tao_core.microservices.handlers.cloud_handlers.progress_tracker import ProgressTracker
        from nvidia_tao_core.microservices.handlers.cloud_handlers.progress_tracker_utils import (
            get_folder_stats, send_progress_status_callback
        )

        full_path = self.root + cloud_subfolder.strip('/').rstrip('/') + '/'
        try:
            # Get folder statistics for progress tracking
            file_count, total_size_mb = get_folder_stats(local_folder)

            logger.info(f"Starting folder upload: {local_folder} -> {cloud_subfolder}")
            logger.info(f"Folder contains {file_count} files ({total_size_mb:.1f} MB total)")

            # Create progress tracker with callback control
            progress_tracker = ProgressTracker("upload", total_files=file_count,
                                               total_size_mb=total_size_mb,
                                               send_callbacks=send_status_callbacks)

            # For large folders, upload files individually to track progress
            if file_count > 10 or total_size_mb > 100:
                self._upload_folder_with_progress(local_folder, full_path, progress_tracker)
            else:
                # For small folders, use direct fsspec upload
                self.fs.upload(local_folder, full_path, recursive=True)
                progress_tracker.update_progress(files_processed=file_count, size_processed_mb=total_size_mb)

            progress_tracker.complete()
            logger.info(f"Uploaded folder {local_folder} to {cloud_subfolder}")
        except Exception as e:
            error_msg = f"Error uploading folder {local_folder} to {cloud_subfolder}: {str(e)}"
            logger.error(f"upload_folder error: {e}")

            # Send error callback
            send_progress_status_callback(error_msg)
            raise

    def _upload_folder_with_progress(self, local_folder, cloud_path, progress_tracker):
        """Upload folder contents with detailed progress tracking."""
        from nvidia_tao_core.microservices.handlers.cloud_handlers.progress_tracker_utils import get_file_size_mb

        try:
            for root, _, files in os.walk(local_folder):
                for file in files:
                    local_file_path = os.path.join(root, file)

                    # Calculate relative path from local_folder
                    relative_path = os.path.relpath(local_file_path, local_folder)
                    cloud_file_path = cloud_path + relative_path.replace(os.sep, '/')

                    # Get file size
                    file_size_mb = get_file_size_mb(local_file_path)

                    try:
                        # Mark file as started with file name and size
                        # Include parent folder for better identification
                        path_parts = local_file_path.strip('/').split('/')
                        if len(path_parts) >= 2:
                            file_name = f"{path_parts[-2]}/{os.path.basename(local_file_path)}"
                        else:
                            file_name = os.path.basename(local_file_path)
                        progress_tracker.start_file_download(file_name=file_name, file_size_mb=file_size_mb)
                        # Use streaming upload for large files
                        if file_size_mb > 50:
                            self._upload_file_with_streaming_progress(
                                local_file_path, cloud_file_path, file_size_mb,
                                file_name, progress_tracker
                            )
                            # File completion marked inside streaming function
                        else:
                            # Upload individual file
                            self.fs.upload(local_file_path, cloud_file_path)
                            # Update progress with size and mark file complete
                            progress_tracker.update_progress(size_processed_mb=file_size_mb)
                            progress_tracker.complete_file_download()

                    except Exception as file_err:
                        logger.warning(f"Failed to upload {local_file_path}: {file_err}")
                        continue

        except Exception as e:
            logger.error(f"Error in _upload_folder_with_progress: {e}")
            raise

    @retry_method
    @master_node_only
    def delete_folder(self, folder):
        """Delete a folder and its contents from cloud storage."""
        full_path = self.root + folder.strip('/').rstrip('/') + '/'
        try:
            self.fs.rm(full_path, recursive=True)
            logger.info(f"Deleted folder {folder}")
        except Exception as e:
            logger.error(f"delete_folder error: {e}")
            raise

    @retry_method
    @master_node_only
    def delete_file(self, file_path):
        """Delete a file from cloud storage."""
        full_path = self.root + file_path.strip('/')
        try:
            self.fs.rm(full_path)
            logger.info(f"Deleted file {file_path}")
        except Exception as e:
            logger.error(f"delete_file error: {e}")
            raise

    @retry_method
    @master_node_only
    def move_file(self, source_path, destination_path):
        """Move a file within cloud storage."""
        full_source = self.root + source_path.strip('/')
        full_destination = self.root + destination_path.strip('/')
        try:
            self.fs.mv(full_source, full_destination)
            logger.info(f"Moved {full_source} to {full_destination}")
        except Exception as e:
            logger.error(f"move_file error: {e}")
            raise

    @retry_method
    @master_node_only
    def move_folder(self, source_path, destination_path, job_id=None, exclude_files=None):
        """Move a folder within cloud storage.

        Args:
            source_path: Source folder path
            destination_path: Destination folder path
            job_id: Optional job ID for health beat reporting during long operations
            exclude_files: Optional list of filenames to exclude from move (e.g., ['microservices_log.txt', 'log.txt'])
        """
        full_source = self.root + source_path.strip('/').rstrip('/') + '/'
        full_destination = self.root + destination_path.strip('/').rstrip('/') + '/'
        exclude_files = exclude_files or []

        try:
            logger.info(f"Moving folder {full_source} to {full_destination}")
            if exclude_files:
                logger.info(f"Excluding files from move: {exclude_files}")

            if job_id:
                report_health_beat(job_id, f"Starting folder move: {source_path} -> {destination_path}")

            # Get all files in source
            all_files = self.fs.find(full_source)
            files_only = [f for f in all_files if self.fs.isfile(f)]
            # Filter out excluded files
            if exclude_files:
                filtered_files = []
                for file_path in files_only:
                    filename = file_path.split('/')[-1]
                    if filename not in exclude_files:
                        filtered_files.append(file_path)
                    else:
                        logger.info(f"Excluding file from move: {file_path}")
                files_only = filtered_files

            logger.info(f"Attempting to move {len(files_only)} files individually")

            if job_id and len(files_only) > 0:
                report_health_beat(job_id, f"Moving {len(files_only)} files from {source_path}")

            # Create destination directory
            self.fs.makedirs(full_destination, exist_ok=True)

            # Move files one by one
            moved_files = 0
            for file_path in files_only:
                try:
                    relative_path = file_path[len(full_source):]
                    dest_file = full_destination + relative_path

                    # Create intermediate directories if needed
                    dest_dir = '/'.join(dest_file.split('/')[:-1])
                    if dest_dir:
                        self.fs.makedirs(dest_dir, exist_ok=True)

                    # Copy then delete individual file
                    logger.info(f"Moving file {file_path} to {dest_file}")
                    self.fs.cp(file_path, dest_file)
                    logger.info(f"Deleting file {file_path} after moving")
                    self.fs.rm(file_path)
                    moved_files += 1

                    # Report health beat for every file to prevent timeout during long moves
                    if job_id:
                        report_health_beat(
                            job_id,
                            f"Moved {moved_files}/{len(files_only)} files ({int(moved_files / len(files_only) * 100)}%)"
                        )

                except Exception as file_err:
                    logger.error(
                        "Could not move file %s (will raise after loop if any file failed): %s",
                        file_path, file_err, exc_info=True
                    )

            if moved_files == 0:
                raise Exception("No files could be moved using any method")

            if moved_files < len(files_only):
                failed_count = len(files_only) - moved_files
                msg = (
                    f"Only {moved_files}/{len(files_only)} files were moved; {failed_count} file(s) failed. "
                    "Source folder was not removed so no data was lost. Retry may succeed (e.g. after timeout)."
                )
                logger.error(msg)
                raise Exception(msg)

            if job_id:
                report_health_beat(job_id, f"Cleaning up source directory after moving {moved_files} files")

            try:
                self.fs.rm(full_source, recursive=True)
            except Exception:
                logger.warning("Could not remove source directory after file moves")

            logger.info(f"Successfully moved {moved_files} files using file-by-file approach")

            if job_id:
                report_health_beat(
                    job_id,
                    f"Completed folder move: {moved_files} files moved to {destination_path}"
                )
        except Exception as e:
            logger.error(f"move_folder error: {e}")
            raise

    @retry_method
    @master_node_only
    def copy_file(self, source_object_name, destination_object_name):
        """Copy a file within cloud storage."""
        full_source = self.root + source_object_name.strip('/')
        full_destination = self.root + destination_object_name.strip('/')
        try:
            self.fs.cp(full_source, full_destination)
            logger.info(f"Copied {source_object_name} to {destination_object_name}")
        except Exception as e:
            logger.error(f"copy_file error: {e}")
            raise

    @retry_method
    @master_node_only
    def copy_folder(self, source_path, destination_path):
        """Copy a folder within cloud storage."""
        full_source = self.root + source_path.strip('/').rstrip('/') + '/'
        full_destination = self.root + destination_path.strip('/').rstrip('/') + '/'
        try:
            self.fs.cp(full_source, full_destination, recursive=True)
            logger.info(f"Copied folder {source_path} to {destination_path}")
        except Exception as e:
            logger.error(f"copy_folder error: {e}")
            raise

    @retry_method
    def search_for_ptm(self, root="", network="", parameter_name=""):
        """Return path of the PTM file under the PTM root folder (cloud version)"""
        try:
            # Normalize root path
            root_normalized = root.strip('/')
            search_path = root_normalized + '/' if root_normalized else ''

            # Search for common model file extensions (mimic utils.py)
            models = []
            models.extend(self.glob_files(search_path + "**/*.tlt"))
            models.extend(self.glob_files(search_path + "**/*.hdf5"))
            models.extend(self.glob_files(search_path + "**/*.pth"))
            models.extend(self.glob_files(search_path + "**/*.pth.tar"))
            models.extend(self.glob_files(search_path + "**/*.pt"))

            # Special cases for specific networks (mimic utils.py)
            if network in ("classification_pyt", "visual_changenet_classify", "visual_changenet_segment", "nvdinov2"):
                models.extend(self.glob_files(search_path + "**/*.ckpt"))

            if network == "stylegan_xl":
                if parameter_name == "inception_fid_path":
                    models.extend(self.glob_files(search_path + "**/*Inception*.pth"))
                if parameter_name == "input_embeddings_path":
                    models.extend(self.glob_files(search_path + "**/*tf_efficientnet*.pth"))

            if models:
                model_path = models[0]  # pick one arbitrarily
                logger.info(f"Found valid PTM at {model_path}")
                return model_path

            # If no models found but root exists, handle special cases
            if self.is_folder(root_normalized):
                if network == "vila":
                    # For vila, return first item in directory
                    files, dirs = self.list_files_in_folder(root_normalized)
                    if files:
                        return files[0]
                    if dirs:
                        return dirs[0]
                return root_normalized

            logger.info("PTM can't be found")
            return None

        except Exception as e:
            logger.error(f"search_for_ptm error: {e}")
            return None
