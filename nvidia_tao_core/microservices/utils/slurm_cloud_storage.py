# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SLURM Cloud Storage Adapter - provides CloudStorage-compatible interface for SLURM/Lustre"""

import os
import logging
import subprocess
import time
from functools import wraps

logger = logging.getLogger(__name__)

# Timeout for SSH commands (5 minutes)
TIMEOUT_SECONDS = 5 * 60


def retry_on_connection_error(max_retries=5, initial_delay=30, backoff_factor=2, enable_host_failover=True):
    """Decorator to retry SLURM operations on connection errors with exponential backoff.

    Also supports automatic host failover for SlurmCloudStorageAdapter instances.

    Args:
        max_retries: Maximum number of retry attempts per host (default: 5)
        initial_delay: Initial delay between retries in seconds (default: 30)
        backoff_factor: Multiplier for delay after each retry (default: 2)
        enable_host_failover: Try alternate hosts after exhausting retries on current host (default: True)

    Returns:
        Decorated function that retries on connection errors
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if first arg is a SlurmCloudStorageAdapter instance
            self_instance = args[0] if args and hasattr(args[0], '_switch_to_next_hostname') else None
            hosts_tried = set()

            while True:
                current_host = self_instance.slurm_hostname if self_instance else None
                delay = initial_delay
                last_exception = None

                # Try with current host with retries
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        # Check if this is a connection/timeout error worth retrying
                        error_str = str(e).lower()
                        is_retryable = any(keyword in error_str for keyword in [
                            'timeout', 'connection', 'ssh', 'network', 'timed out',
                            'temporarily unavailable', 'connection refused'
                        ])

                        if not is_retryable or attempt >= max_retries:
                            # Not a connection error or exhausted retries on this host
                            if attempt > 0:
                                logger.error(
                                    f"{func.__name__} failed after {attempt + 1} attempts "
                                    f"on host {current_host}. Last error: {e}"
                                )
                            break  # Break inner loop to try host failover

                        # Log retry attempt
                        logger.warning(
                            f"{func.__name__} failed on attempt {attempt + 1}/{max_retries + 1} "
                            f"on host {current_host} with error: {e}. Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                else:
                    # Inner loop completed without break - success would have returned
                    continue

                # Exhausted retries on current host
                if current_host:
                    hosts_tried.add(current_host)

                # Try host failover if enabled and available
                if enable_host_failover and self_instance and \
                   len(hosts_tried) < len(self_instance.slurm_hostnames):
                    error_str = str(last_exception).lower()
                    is_connection_error = any(keyword in error_str for keyword in [
                        'timeout', 'connection', 'ssh', 'network', 'timed out',
                        'temporarily unavailable', 'connection refused'
                    ])

                    if is_connection_error:
                        logger.warning(
                            f"{func.__name__} exhausted retries on host {current_host}. "
                            f"Attempting host failover..."
                        )
                        if self_instance._switch_to_next_hostname():
                            continue  # Try with next host
                        logger.error("Host failover failed - no more hosts available")

                # No more hosts to try or not a connection error
                if hosts_tried:
                    logger.error(
                        f"{func.__name__} failed on all attempted hosts: {hosts_tried}"
                    )
                raise last_exception

        return wrapper
    return decorator


class SlurmCloudStorageAdapter:
    """Adapter that provides CloudStorage-like interface for SLURM via SSH.

    This adapter wraps SlurmHandler to provide remote path checking
    without needing direct filesystem access to the SLURM cluster.
    It implements all methods from CloudStorage for full compatibility.

    Supports multiple hostnames for automatic failover when a host becomes unreachable.
    """

    def __init__(self, slurm_user, slurm_hostname, base_results_dir, ssh_key_path=None):
        """Initialize SLURM cloud storage adapter.

        Args:
            slurm_user: SSH username for SLURM cluster
            slurm_hostname: List of hostnames for SLURM cluster (for multi-host failover)
            base_results_dir: Base directory path on Lustre filesystem
            ssh_key_path: Path to SSH private key (optional)
        """
        self.cloud_type = "slurm"
        self.bucket_name = base_results_dir
        self.root = base_results_dir.rstrip('/') + '/' if base_results_dir else ''
        self.slurm_user = slurm_user

        # slurm_hostname is always a list of strings
        if not isinstance(slurm_hostname, list):
            raise ValueError("slurm_hostname must be a list of strings")
        if not slurm_hostname:
            raise ValueError("slurm_hostname list cannot be empty")

        self.slurm_hostnames = slurm_hostname

        # Track current hostname index for failover
        self.current_hostname_index = 0
        self.slurm_hostname = self.slurm_hostnames[self.current_hostname_index]

        self.ssh_key_path = ssh_key_path or "/root/.ssh/id_ed25519"
        logger.info(
            f"Initialized SLURM cloud storage adapter with root: {self.root}, "
            f"hostnames: {self.slurm_hostnames} ({len(self.slurm_hostnames)} host(s)), "
            f"current: {self.slurm_hostname}"
        )

    def _switch_to_next_hostname(self):
        """Switch to the next available hostname in the list.

        Returns:
            bool: True if switched to a new hostname, False if all hostnames exhausted
        """
        if len(self.slurm_hostnames) <= 1:
            logger.warning("No alternate hostnames available for failover")
            return False

        # Try next hostname
        self.current_hostname_index = (self.current_hostname_index + 1) % len(self.slurm_hostnames)
        old_hostname = self.slurm_hostname
        self.slurm_hostname = self.slurm_hostnames[self.current_hostname_index]

        if old_hostname == self.slurm_hostname:
            # We've cycled through all hostnames
            logger.error("All SLURM hostnames have been tried and failed")
            return False

        logger.info(f"Switching SLURM hostname from {old_hostname} to {self.slurm_hostname}")
        return True

    def _execute_with_host_failover(self, operation_func, operation_name, *args, **kwargs):
        """Execute an operation with automatic host failover on connection errors.

        Args:
            operation_func: Function to execute
            operation_name: Name of operation for logging
            *args, **kwargs: Arguments to pass to operation_func

        Returns:
            Result of operation_func

        Raises:
            Exception: If all hosts fail
        """
        hosts_tried = set()
        last_exception = None

        while len(hosts_tried) < len(self.slurm_hostnames):
            current_host = self.slurm_hostname
            hosts_tried.add(current_host)

            try:
                return operation_func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                is_connection_error = any(keyword in error_str for keyword in [
                    'timeout', 'connection', 'ssh', 'network', 'timed out',
                    'temporarily unavailable', 'connection refused'
                ])

                if is_connection_error and len(hosts_tried) < len(self.slurm_hostnames):
                    logger.warning(
                        f"{operation_name} failed on host {current_host}: {e}. "
                        f"Trying next hostname..."
                    )
                    if not self._switch_to_next_hostname():
                        break
                else:
                    # Not a connection error or no more hosts to try
                    raise

        # All hosts failed
        logger.error(
            f"{operation_name} failed on all {len(self.slurm_hostnames)} SLURM hosts. "
            f"Hosts tried: {hosts_tried}"
        )
        raise last_exception

    def _get_full_path(self, cloud_path):
        """Convert cloud path to full remote path."""
        if not cloud_path:
            return self.root.rstrip('/')

        # If cloud_path is absolute, use it as-is
        if cloud_path.startswith('/'):
            return cloud_path

        # Otherwise, join with root
        return self.root + cloud_path.strip('/')

    def _run_ssh_command(self, command, timeout=TIMEOUT_SECONDS):
        """Execute SSH command on SLURM cluster.

        Args:
            command: Command to execute
            timeout: Timeout in seconds

        Returns:
            tuple: (success, stdout, stderr)
        """
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={timeout}",
            "-o", f"ServerAliveInterval={timeout}",
            "-o", "ServerAliveCountMax=1"
        ]

        if self.ssh_key_path and os.path.exists(self.ssh_key_path):
            ssh_cmd.extend(["-i", self.ssh_key_path])

        ssh_cmd.extend([f"{self.slurm_user}@{self.slurm_hostname}", command])

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            logger.error(f"SSH command timed out after {timeout}s: {command}")
            return False, "", f"Timeout after {timeout}s"
        except Exception as e:
            logger.error(f"SSH command failed: {e}")
            return False, "", str(e)

    @retry_on_connection_error(max_retries=5, initial_delay=30, backoff_factor=2)
    def is_file(self, cloud_path):
        """Check if the given cloud path is a file."""
        full_path = self._get_full_path(cloud_path)
        command = f"[ -f '{full_path}' ] && echo 'true' || echo 'false'"
        success, stdout, _ = self._run_ssh_command(command)
        if success and stdout == 'true':
            return True
        return False

    @retry_on_connection_error(max_retries=5, initial_delay=30, backoff_factor=2)
    def is_folder(self, cloud_path):
        """Check if the given cloud path is a folder."""
        full_path = self._get_full_path(cloud_path)
        command = f"[ -d '{full_path}' ] && echo 'true' || echo 'false'"
        success, stdout, _ = self._run_ssh_command(command)
        if success and stdout == 'true':
            return True
        return False

    @retry_on_connection_error(max_retries=5, initial_delay=30, backoff_factor=2)
    def glob_files(self, pattern):
        """Return a list of files matching the pattern."""
        # Pattern should be relative to root
        full_pattern = self.root + pattern.strip('/') if not pattern.startswith('/') else pattern
        command = f"find {os.path.dirname(full_pattern)} -name '{os.path.basename(full_pattern)}' -type f 2>/dev/null"
        success, stdout, _ = self._run_ssh_command(command)

        if not success or not stdout:
            return []

        matched_files = [line for line in stdout.split('\n') if line]

        # Return paths relative to root
        if self.root:
            return [f[len(self.root):] if f.startswith(self.root) else f for f in matched_files]
        return matched_files

    @retry_on_connection_error(max_retries=3, initial_delay=5, backoff_factor=2)
    def list_files_in_folder(self, folder, recursive=True):
        """Recursively list files in the specified folder and its subfolders."""
        try:
            # Use _get_full_path to handle both absolute and relative paths correctly
            full_path = self._get_full_path(folder)

            # Build find command
            if recursive:
                command = f"find '{full_path}' -type f 2>/dev/null"
            else:
                command = f"find '{full_path}' -maxdepth 1 -type f 2>/dev/null"

            success, stdout, _ = self._run_ssh_command(command)

            if not success or not stdout:
                logger.info(f"No files found in {full_path}")
                return [], []

            files = [line for line in stdout.split('\n') if line]
            logger.info(f"Found {len(files)} files in {full_path}")
            return files, []  # Return (files, details) - details not available
        except Exception as e:
            logger.error(f"list_files_in_folder error: {e}")
            return [], []

    @retry_on_connection_error(max_retries=5, initial_delay=30, backoff_factor=2)
    def download_file(self, cloud_file_path, local_destination, progress_tracker=None):
        """Download a file from SLURM cluster to local destination via SCP.

        Args:
            cloud_file_path (str): Path to file on SLURM cluster
            local_destination (str): Local destination path
            progress_tracker: Optional progress tracker (unused for SLURM)
        """
        full_path = self._get_full_path(cloud_file_path)

        # Ensure local directory exists
        os.makedirs(os.path.dirname(local_destination), exist_ok=True)

        # Build SCP command
        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={TIMEOUT_SECONDS}"
        ]

        if self.ssh_key_path and os.path.exists(self.ssh_key_path):
            scp_cmd.extend(["-i", self.ssh_key_path])

        scp_cmd.extend([
            f"{self.slurm_user}@{self.slurm_hostname}:{full_path}",
            local_destination
        ])

        try:
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False)
            if result.returncode == 0:
                logger.info(f"Downloaded {cloud_file_path} to {local_destination}")
            else:
                logger.error(f"SCP download failed: {result.stderr}")
                raise Exception(f"Failed to download {cloud_file_path}: {result.stderr}")
        except subprocess.TimeoutExpired as e:
            logger.error(f"SCP download timed out for {cloud_file_path}")
            raise Exception(f"Download timeout for {cloud_file_path}") from e
        except Exception as e:
            logger.error(f"Download error: {e}")
            raise

    @retry_on_connection_error(max_retries=5, initial_delay=30, backoff_factor=2)
    def is_file_modified(self, cloud_path, local_path):
        """Check if a cloud file has been modified compared to the local file.

        Args:
            cloud_path: Path on SLURM cluster
            local_path: Local file path

        Returns:
            bool: True if cloud file is newer or sizes differ
        """
        if not os.path.exists(local_path):
            return True

        full_path = self._get_full_path(cloud_path)
        command = f"stat -c '%Y %s' '{full_path}' 2>/dev/null"
        success, stdout, _ = self._run_ssh_command(command)

        if not success or not stdout:
            logger.warning(f"Could not stat remote file: {full_path}")
            return True

        try:
            remote_mtime, remote_size = stdout.split()
            remote_mtime = int(remote_mtime)
            remote_size = int(remote_size)

            local_stat = os.stat(local_path)
            local_mtime = int(local_stat.st_mtime)
            local_size = local_stat.st_size

            # File is modified if size differs or remote is newer
            return remote_size != local_size or remote_mtime > local_mtime
        except Exception as e:
            logger.error(f"Error comparing file timestamps: {e}")
            return True

    @retry_on_connection_error(max_retries=5, initial_delay=30, backoff_factor=2)
    def download_folder(self, cloud_folder, local_destination,
                        maintain_src_folder_structure=False,
                        progress_tracker=None, extensions=None):
        """Download a folder from SLURM cluster to local destination via SCP.

        Args:
            cloud_folder: Remote folder path
            local_destination: Local destination path
            maintain_src_folder_structure: Whether to maintain source folder structure
            progress_tracker: Optional progress tracker (unused for SLURM)
            extensions: List of file extensions to filter (e.g., ['.txt', '.log'])
        """
        full_path = self._get_full_path(cloud_folder)

        # Ensure local directory exists
        os.makedirs(local_destination, exist_ok=True)

        # If extensions filter is provided, use find command to download only matching files
        if extensions:
            # Build find command to locate files with specific extensions
            ext_patterns = " -o ".join([f"-name '*{ext}'" for ext in extensions])
            find_cmd = f"cd '{full_path}' && find . -type f \\( {ext_patterns} \\)"

            # Get list of matching files via SSH
            ssh_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", f"ConnectTimeout={TIMEOUT_SECONDS}"
            ]

            if self.ssh_key_path and os.path.exists(self.ssh_key_path):
                ssh_cmd.extend(["-i", self.ssh_key_path])

            ssh_cmd.extend([
                f"{self.slurm_user}@{self.slurm_hostname}",
                find_cmd
            ])

            try:
                logger.info(f"Finding files with extensions {extensions} in {cloud_folder}")
                result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=60, check=False)
                if result.returncode != 0:
                    logger.warning(f"Find command failed: {result.stderr}")
                    return

                # Parse the file list
                files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
                if not files:
                    logger.info(f"No files with extensions {extensions} found in {cloud_folder}")
                    return

                logger.info(f"Found {len(files)} files to download from {cloud_folder}")

                # Download each file, maintaining directory structure
                for file_path in files:
                    # Remove leading './' from file path
                    rel_path = file_path.lstrip('./')
                    if not rel_path:
                        continue

                    # Create local directory structure
                    local_file_path = os.path.join(local_destination, rel_path)
                    local_file_dir = os.path.dirname(local_file_path)
                    if local_file_dir:
                        os.makedirs(local_file_dir, exist_ok=True)

                    # Download individual file
                    remote_file = os.path.join(full_path, rel_path)
                    scp_cmd = [
                        "scp",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "UserKnownHostsFile=/dev/null",
                        "-o", "ConnectTimeout=30"
                    ]

                    if self.ssh_key_path and os.path.exists(self.ssh_key_path):
                        scp_cmd.extend(["-i", self.ssh_key_path])

                    scp_cmd.extend([
                        f"{self.slurm_user}@{self.slurm_hostname}:{remote_file}",
                        local_file_path
                    ])

                    file_result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=30, check=False)
                    if file_result.returncode != 0:
                        logger.warning(f"Failed to download {rel_path}: {file_result.stderr}")
                    else:
                        logger.debug(f"Downloaded {rel_path}")

                logger.info(f"Downloaded {len(files)} files from {cloud_folder} to {local_destination}")

            except subprocess.TimeoutExpired:
                logger.error(f"Download timed out for {cloud_folder}")
            except Exception as e:
                logger.error(f"Folder download error: {e}")
        else:
            # Original recursive download if no extension filter
            scp_cmd = [
                "scp",
                "-r",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", f"ConnectTimeout={TIMEOUT_SECONDS}"
            ]

            if self.ssh_key_path and os.path.exists(self.ssh_key_path):
                scp_cmd.extend(["-i", self.ssh_key_path])

            scp_cmd.extend([
                f"{self.slurm_user}@{self.slurm_hostname}:{full_path}/*",
                local_destination
            ])

            try:
                logger.info(f"Downloading folder {cloud_folder} to {local_destination}")
                logger.info(f"SCP command: {scp_cmd}")
                result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False)
                if result.returncode == 0:
                    logger.info(f"Downloaded folder {cloud_folder} to {local_destination}")
                else:
                    logger.warning(f"SCP folder download completed with warnings: {result.stderr}")
            except subprocess.TimeoutExpired:
                logger.error(f"SCP folder download timed out for {cloud_folder}")
            except Exception as e:
                logger.error(f"Folder download error: {e}")

    @retry_on_connection_error(max_retries=5, initial_delay=30, backoff_factor=2)
    def create_folder_in_bucket(self, folder):
        """Create a folder on the SLURM cluster."""
        full_path = self._get_full_path(folder)
        command = f"mkdir -p '{full_path}'"
        success, _, stderr = self._run_ssh_command(command)

        if success:
            logger.info(f"Created folder: {full_path}")
        else:
            logger.error(f"Failed to create folder {full_path}: {stderr}")
            raise Exception(f"Failed to create folder: {stderr}")

    @retry_on_connection_error(max_retries=5, initial_delay=30, backoff_factor=2)
    def upload_file(self, local_file_path, cloud_file_path, progress_tracker=None, send_status_callbacks=True):
        """Upload a file from local storage to SLURM cluster via SCP.

        Args:
            local_file_path: Local file path
            cloud_file_path: Destination path on SLURM cluster
            progress_tracker: Optional progress tracker (unused for SLURM)
            send_status_callbacks: Whether to send status callbacks (unused for SLURM)
        """
        full_path = self._get_full_path(cloud_file_path)

        # Ensure remote directory exists
        remote_dir = os.path.dirname(full_path)
        self.create_folder_in_bucket(remote_dir.replace(self.root, '') if self.root else remote_dir)

        # Build SCP command
        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={TIMEOUT_SECONDS}"
        ]

        if self.ssh_key_path and os.path.exists(self.ssh_key_path):
            scp_cmd.extend(["-i", self.ssh_key_path])

        scp_cmd.extend([
            local_file_path,
            f"{self.slurm_user}@{self.slurm_hostname}:{full_path}"
        ])

        try:
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False)
            if result.returncode == 0:
                logger.info(f"Uploaded {local_file_path} to {cloud_file_path}")
            else:
                logger.error(f"SCP upload failed: {result.stderr}")
                raise Exception(f"Failed to upload {local_file_path}: {result.stderr}")
        except subprocess.TimeoutExpired as e:
            logger.error(f"SCP upload timed out for {local_file_path}")
            raise Exception(f"Upload timeout for {local_file_path}") from e
        except Exception as e:
            logger.error(f"Upload error: {e}")
            raise

    @retry_on_connection_error(max_retries=5, initial_delay=30, backoff_factor=2)
    def upload_folder(self, local_folder, cloud_subfolder, send_status_callbacks=True):
        """Upload a folder from local storage to SLURM cluster via SCP.

        Args:
            local_folder: Local folder path
            cloud_subfolder: Destination path on SLURM cluster
            send_status_callbacks: Whether to send status callbacks (unused for SLURM)
        """
        full_path = self._get_full_path(cloud_subfolder)

        # Ensure remote directory exists
        self.create_folder_in_bucket(cloud_subfolder)

        # Build SCP command for recursive copy
        scp_cmd = [
            "scp",
            "-r",  # Recursive
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={TIMEOUT_SECONDS}"
        ]

        if self.ssh_key_path and os.path.exists(self.ssh_key_path):
            scp_cmd.extend(["-i", self.ssh_key_path])

        scp_cmd.extend([
            f"{local_folder}/*",
            f"{self.slurm_user}@{self.slurm_hostname}:{full_path}/"
        ])

        try:
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False)
            if result.returncode == 0:
                logger.info(f"Uploaded folder {local_folder} to {cloud_subfolder}")
            else:
                logger.error(f"SCP folder upload failed: {result.stderr}")
                raise Exception(f"Failed to upload folder {local_folder}: {result.stderr}")
        except subprocess.TimeoutExpired as e:
            logger.error(f"SCP folder upload timed out for {local_folder}")
            raise Exception(f"Upload timeout for {local_folder}") from e
        except Exception as e:
            logger.error(f"Folder upload error: {e}")
            raise

    @retry_on_connection_error(max_retries=5, initial_delay=30, backoff_factor=2)
    def delete_folder(self, folder):
        """Delete a folder and its contents from SLURM cluster."""
        full_path = self._get_full_path(folder)
        command = f"rm -rf '{full_path}'"
        success, _, stderr = self._run_ssh_command(command)

        if success:
            logger.info(f"Deleted folder: {full_path}")
            return True
        logger.error(f"Failed to delete folder {full_path}: {stderr}")
        raise RuntimeError(
            f"Failed to delete folder {full_path}: "
            f"{stderr or 'remote rm returned a non-zero status'}"
        )

    @retry_on_connection_error(max_retries=2, initial_delay=3, backoff_factor=2)
    def delete_file(self, file_path):
        """Delete a file from SLURM cluster."""
        full_path = self._get_full_path(file_path)
        command = f"rm -f '{full_path}'"
        success, _, stderr = self._run_ssh_command(command)

        if success:
            logger.info(f"Deleted file: {full_path}")
            return True
        logger.error(f"Failed to delete file {full_path}: {stderr}")
        raise RuntimeError(
            f"Failed to delete file {full_path}: "
            f"{stderr or 'remote rm returned a non-zero status'}"
        )

    @retry_on_connection_error(max_retries=5, initial_delay=30, backoff_factor=2)
    def move_file(self, source_path, destination_path):
        """Move a file within SLURM cluster."""
        source_full = self._get_full_path(source_path)
        dest_full = self._get_full_path(destination_path)

        # Ensure destination directory exists
        dest_dir = os.path.dirname(destination_path)
        if dest_dir:
            self.create_folder_in_bucket(dest_dir)

        command = f"mv '{source_full}' '{dest_full}'"
        success, _, stderr = self._run_ssh_command(command)

        if success:
            logger.info(f"Moved file from {source_path} to {destination_path}")
        else:
            logger.error(f"Failed to move file: {stderr}")
            raise Exception(f"Failed to move file: {stderr}")

    @retry_on_connection_error(max_retries=5, initial_delay=30, backoff_factor=2)
    def move_folder(self, source_path, destination_path, job_id=None):
        """Move a folder within SLURM cluster."""
        source_full = self._get_full_path(source_path)
        dest_full = self._get_full_path(destination_path)

        # Ensure parent destination directory exists
        dest_parent = os.path.dirname(destination_path)
        if dest_parent:
            self.create_folder_in_bucket(dest_parent)

        command = f"mv '{source_full}' '{dest_full}'"
        success, _, stderr = self._run_ssh_command(command)

        if success:
            logger.info(f"Moved folder from {source_path} to {destination_path}")
        else:
            logger.error(f"Failed to move folder: {stderr}")
            raise Exception(f"Failed to move folder: {stderr}")

    @retry_on_connection_error(max_retries=5, initial_delay=30, backoff_factor=2)
    def copy_file(self, source_object_name, destination_object_name):
        """Copy a file within SLURM cluster."""
        source_full = self._get_full_path(source_object_name)
        dest_full = self._get_full_path(destination_object_name)

        # Ensure destination directory exists
        dest_dir = os.path.dirname(destination_object_name)
        if dest_dir:
            self.create_folder_in_bucket(dest_dir)

        command = f"cp '{source_full}' '{dest_full}'"
        success, _, stderr = self._run_ssh_command(command)

        if success:
            logger.info(f"Copied file from {source_object_name} to {destination_object_name}")
        else:
            logger.error(f"Failed to copy file: {stderr}")
            raise Exception(f"Failed to copy file: {stderr}")

    @retry_on_connection_error(max_retries=5, initial_delay=30, backoff_factor=2)
    def copy_folder(self, source_path, destination_path):
        """Copy a folder within SLURM cluster."""
        source_full = self._get_full_path(source_path)
        dest_full = self._get_full_path(destination_path)

        # Ensure parent destination directory exists
        dest_parent = os.path.dirname(destination_path)
        if dest_parent:
            self.create_folder_in_bucket(dest_parent)

        command = f"cp -r '{source_full}' '{dest_full}'"
        success, _, stderr = self._run_ssh_command(command)

        if success:
            logger.info(f"Copied folder from {source_path} to {destination_path}")
        else:
            logger.error(f"Failed to copy folder: {stderr}")
            raise Exception(f"Failed to copy folder: {stderr}")

    @retry_on_connection_error(max_retries=5, initial_delay=30, backoff_factor=2)
    def search_for_ptm(self, root="", network="", parameter_name=""):
        """Search for PTM file under the PTM root folder on SLURM cluster.

        Args:
            root: Root directory to search
            network: Network name
            parameter_name: Parameter name

        Returns:
            str: Path to PTM file if found, None otherwise
        """
        search_path = self._get_full_path(root) if root else self.root

        # Build search pattern
        if parameter_name:
            pattern = f"*{network}*{parameter_name}*.hdf5"
        elif network:
            pattern = f"*{network}*.hdf5"
        else:
            pattern = "*.hdf5"

        command = f"find '{search_path}' -name '{pattern}' -type f 2>/dev/null | head -1"
        success, stdout, _ = self._run_ssh_command(command)

        if success and stdout:
            # Return path relative to root
            if self.root and stdout.startswith(self.root):
                return stdout[len(self.root):]
            return stdout

        return None

    def reset_filesystem_state(self):
        """No-op for SLURM - included for CloudStorage interface compatibility."""
        logger.info("SLURM: reset_filesystem_state is a no-op (no cache to clear)")

    @retry_on_connection_error(max_retries=5, initial_delay=30, backoff_factor=2)
    def validate_connection(self):
        """Validate SSH connection to SLURM cluster."""
        command = "echo 'connection_test'"
        success, stdout, stderr = self._run_ssh_command(command, timeout=30)

        if not success or stdout != 'connection_test':
            from nvidia_tao_core.microservices.utils.cloud_utils import CloudStorageConnectionError
            raise CloudStorageConnectionError(
                f"Failed to connect to SLURM cluster {self.slurm_hostname}: {stderr}"
            )

        logger.info(f"Successfully validated connection to SLURM cluster {self.slurm_hostname}")
        return True
