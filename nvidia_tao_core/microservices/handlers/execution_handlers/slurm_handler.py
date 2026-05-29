# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Handler to execute jobs on Slurm"""

import json
import math
import subprocess
import logging
import os
import tempfile
import shlex
import re
import pwd
import time
import traceback
from datetime import datetime, timezone

from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    get_handler_job_metadata,
    write_job_metadata,
    update_job_message,
    get_dnn_status,
    save_dnn_status,
    get_automl_controller_info,
    save_automl_controller_info,
    get_handler_id,
    get_handler_kind,
    get_handler_metadata
)

from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import ExecutionHandler
from nvidia_tao_core.microservices.handlers.mongo_handler import MongoHandler
from nvidia_tao_core.microservices.handlers.cloud_handlers.utils import get_file_path_from_cloud_string
from nvidia_tao_core.microservices.enum_constants import Backend
from typing import Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
TIMEOUT_SECONDS = 5 * 60

# Maximum number of job submission retries when a job fails due to bad nodes
MAX_JOB_RETRIES = 10

# Error patterns that indicate a retriable failure (bad node, driver issues, etc.)
# These patterns are searched in job logs to determine if failure is due to infrastructure
RETRIABLE_ERROR_PATTERNS = [
    r"NVIDIA driver.*too old",
    r"CUDA driver version is insufficient",
    r"no CUDA-capable device",
    r"CUDA initialization.*error",
    r"NCCL.*error",
    r"GPU.*not found",
    r"RuntimeError.*CUDA",
    r"cuInit.*failed",
    r"torch\.cuda\.is_available.*False",
    r"Unable to determine the device handle for GPU",
    r"GPU.*fallen off the bus",
    r"Xid.*error",
    r"ECC.*error",
    r"GPU.*temperature",
    r"InfiniBand.*error",
    r"RDMA.*error",
]


def get_results_dir_for_workspace(workspace_metadata, job_id, specs=None):
    """Determine the results directory based on workspace type and specs.

    This function provides consistent results directory handling across all execution backends.

    Args:
        workspace_metadata (dict): Workspace metadata containing cloud_type and cloud_specific_details
        job_id (str): Job ID for constructing the path
        specs (dict, optional): Job specs that may contain a pre-determined results_dir

    Returns:
        str: The results directory path (e.g., /results, /lustre/..., etc.)
    """
    # First priority: Use results_dir from specs if provided (e.g., by infer_output_dir)
    if specs and specs.get("results_dir"):
        return specs["results_dir"]

    # Second priority: Construct based on workspace cloud_type
    cloud_type = workspace_metadata.get("cloud_type") if workspace_metadata else None

    if cloud_type == "slurm":
        cloud_specific_details = workspace_metadata.get("cloud_specific_details", {})
        base_results_dir = cloud_specific_details.get("base_results_dir")
        slurm_user = cloud_specific_details.get("slurm_user")

        if not base_results_dir:
            # Default SLURM path construction
            base_results_dir = f"/lustre/fsw/portfolios/edgeai/users/{slurm_user}/results"
        elif not base_results_dir.endswith("/results"):
            # Ensure it has /results suffix
            base_results_dir = f"{base_results_dir}/results"

        return f"{base_results_dir}/{job_id}"

    # Default for non-SLURM workspaces
    return f"/results/{job_id}"


def get_slurm_handler_from_workspace(workspace_id):
    """Get the Slurm handler from the workspace"""
    workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
    cloud_specific_details = workspace_metadata.get("cloud_specific_details", {})
    slurm_user = cloud_specific_details.get("slurm_user")
    slurm_hostname = cloud_specific_details.get("slurm_hostname")
    if slurm_user and slurm_hostname:
        try:
            logger.info(f"Instantiating Slurm Handler for workspace {workspace_id}")
            return SlurmHandler(slurm_user, slurm_hostname)
        except Exception as e:
            logger.error(f"Exception instantiating Slurm Handler: {e}")
    return None


class SlurmHandler(ExecutionHandler):
    """Handler to execute jobs on Slurm

    Supports multiple hostnames for automatic failover when a host becomes unreachable.
    """

    def __init__(self, login_user=None, login_hostname=None, ssh_key_path=None):
        """Initialize the Slurm Handler

        Args:
            login_user: Username for SSH connection (can be set via SSH_USER env var)
            login_hostname: List of hostnames for SSH connection (for multi-host failover)
                          (can be set via SSH_HOST env var)
            ssh_key_path: Path to SSH private key (can be set via SSH_KEY_PATH env var,
                         default: auto-detect in ~/.ssh/)
        """
        super().__init__(backend_type=Backend.SLURM)
        # Allow configuration via environment variables for containerized deployments
        self.login_user = login_user or os.getenv('SSH_USER') or os.getenv('USER', 'root')

        # login_hostname is always a list of strings
        hostname_param = login_hostname or os.getenv('SSH_HOST')
        if not hostname_param:
            raise ValueError("SSH hostname must be provided either as parameter or SSH_HOST environment variable")

        if not isinstance(hostname_param, list):
            raise ValueError("login_hostname must be a list of strings")
        if not hostname_param:
            raise ValueError("login_hostname list cannot be empty")

        self.login_hostnames = hostname_param

        # Track current hostname index for failover
        self.current_hostname_index = 0
        self.login_hostname = self.login_hostnames[self.current_hostname_index]

        # Auto-detect SSH key if not provided
        self.ssh_key_path = ssh_key_path or os.getenv('SSH_KEY_PATH') or self._find_ssh_key()

        # Set up SSH options
        self.ssh_options = self._get_ssh_options()

        self.logger.info(
            f"SlurmHandler initialized for {self.login_user}@{self.login_hostname} "
            f"with {len(self.login_hostnames)} hostname(s)"
        )

    def _switch_to_next_hostname(self):
        """Switch to the next available hostname in the list.

        Returns:
            bool: True if switched to a new hostname, False if all hostnames exhausted
        """
        if len(self.login_hostnames) <= 1:
            self.logger.warning("No alternate hostnames available for failover")
            return False

        # Try next hostname
        self.current_hostname_index = (self.current_hostname_index + 1) % len(self.login_hostnames)
        old_hostname = self.login_hostname
        self.login_hostname = self.login_hostnames[self.current_hostname_index]

        if old_hostname == self.login_hostname:
            # We've cycled through all hostnames
            self.logger.error("All SLURM hostnames have been tried and failed")
            return False

        self.logger.info(f"Switching SLURM hostname from {old_hostname} to {self.login_hostname}")
        # Update SSH options with new hostname
        self.ssh_options = self._get_ssh_options()
        return True

    def _find_ssh_key(self):
        """Auto-detect SSH private key directory.

        Checks for common key types in order of preference:
        1. id_ed25519 (modern, secure)
        2. id_ecdsa (elliptic curve)
        3. id_rsa (traditional)
        4. id_dsa (legacy)

        In containerized deployments, SSH keys should be mounted to a location
        accessible by the process user (e.g., /home/www-data/.ssh for www-data user).

        Returns:
            str: Path to the first found SSH key, or None if no key found
        """
        # Try multiple locations in order of preference
        # 1. User's home directory (~/.ssh)
        # 2. /home/www-data/.ssh (common mount point for www-data user)
        # 3. /root/.ssh (if running as root)
        ssh_dirs = []

        # Add user's home .ssh directory
        home_dir = os.path.expanduser('~')
        ssh_dirs.append(os.path.join(home_dir, '.ssh'))

        # For www-data user, also check /home/www-data/.ssh
        # (since ~www-data often expands to /tmp in containers)
        try:
            current_user = pwd.getpwuid(os.getuid()).pw_name
            if current_user == 'www-data' and home_dir != '/home/www-data':
                ssh_dirs.append('/home/www-data/.ssh')
        except Exception:
            pass  # If we can't determine user, skip this check

        # Always check /root/.ssh as fallback
        if '/root/.ssh' not in ssh_dirs:
            ssh_dirs.append('/root/.ssh')

        for ssh_dir in ssh_dirs:
            if not os.path.exists(ssh_dir):
                self.logger.debug(f"SSH directory does not exist: {ssh_dir}")
                continue

            # Check if we can actually access it
            try:
                os.listdir(ssh_dir)
            except PermissionError:
                self.logger.warning(f"SSH directory exists but no permission to access: {ssh_dir}")
                continue

            # If we get here, directory exists and is accessible
            break
        else:
            # No accessible SSH directory found
            self.logger.warning("No accessible SSH directory found")
            return None

        # Common SSH key filenames in order of preference
        key_names = ['id_ed25519', 'id_ecdsa', 'id_rsa', 'id_dsa']

        self.logger.info(f"Searching for SSH keys in: {ssh_dir}")

        # List files for debugging
        try:
            files = os.listdir(ssh_dir)
            self.logger.info(f"Files in {ssh_dir}: {files}")
        except Exception as e:
            self.logger.warning(f"Could not list SSH directory {ssh_dir}: {e}")
            return None

        # Search for SSH keys
        for key_name in key_names:
            key_path = os.path.join(ssh_dir, key_name)
            if os.path.exists(key_path):
                self.logger.info(f"Auto-detected SSH key: {key_path}")
                return key_path
            self.logger.debug(f"SSH key not found: {key_path}")

        self.logger.warning(f"No SSH key found in {ssh_dir}")
        return None

    @staticmethod
    def _to_compact_json(value: Union[str, dict, list]):
        """Return a compact JSON string from dict/list/JSON string.

        Converts numpy/pathlib and other common non-JSON types so specs/env
        from API or datasets serialize without TypeError (root cause of
        'Error when creating microservice pod' for evaluate jobs).
        """
        def _json_serializer(obj):
            if hasattr(obj, "item"):  # numpy scalar
                return obj.item()
            if hasattr(obj, "isoformat"):  # datetime
                return obj.isoformat()
            if isinstance(obj, (bytes, bytearray)):
                return obj.decode("utf-8", errors="replace")
            if hasattr(obj, "__fspath__"):  # pathlib.Path
                return str(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        if isinstance(value, (dict, list)):
            return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=_json_serializer)
        # value is str
        try:
            parsed = json.loads(value)
            return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False, default=_json_serializer)
        except Exception:
            return value

    def _get_ssh_options(self):
        """Get SSH options for passwordless authentication"""
        ssh_options = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",  # Suppress warnings about adding host keys
            "-o", "BatchMode=yes",  # Fail if password is required
            "-o", "PasswordAuthentication=no",
            "-o", "PubkeyAuthentication=yes",  # Explicitly enable public key auth
            "-o", "PreferredAuthentications=publickey",  # Only try public key
            "-o", f"ConnectTimeout={TIMEOUT_SECONDS}",
            "-o", f"ServerAliveInterval={TIMEOUT_SECONDS}",
            "-o", "ServerAliveCountMax=3",
            "-o", "UpdateHostKeys=no"  # Don't try to update host keys
        ]

        # Add SSH key if it exists
        if self.ssh_key_path and os.path.exists(self.ssh_key_path):
            ssh_options.extend(["-i", self.ssh_key_path])
            # Ensure proper permissions on key file
            try:
                os.chmod(self.ssh_key_path, 0o600)
                self.logger.info(f"Using SSH key: {self.ssh_key_path}")
            except (PermissionError, OSError) as e:
                # Read-only filesystem or permission denied - proceed if file is readable
                self.logger.warning(f"Could not set permissions on SSH key: {self.ssh_key_path} ({e})")
                self.logger.info(f"Proceeding with existing permissions for SSH key: {self.ssh_key_path}")
        else:
            if self.ssh_key_path:
                self.logger.warning(f"SSH key not found at {self.ssh_key_path}")
            # Check if SSH agent is available
            if os.getenv('SSH_AUTH_SOCK'):
                self.logger.info("SSH agent detected, will attempt agent-based authentication")
            else:
                self.logger.warning("No SSH key found and no SSH agent available")

        return ssh_options

    def _build_ssh_command(self, remote_command):
        """Build SSH command with proper options"""
        ssh_command = ["ssh"] + self.ssh_options + [
            f"{self.login_user}@{self.login_hostname}",
            remote_command
        ]
        return ssh_command

    def _get_sqsh_filename(self, image):
        """Convert Docker image name to SQSH filename

        Converts: nvcr.io/nvidia/pytorch:24.01-py3
        To:       nvcr.io_nvidia_pytorch_24.01-py3.sqsh
        """
        sqsh_name = image.replace('/', '_').replace(':', '_') + '.sqsh'
        return sqsh_name

    def _check_sqsh_exists(self, sqsh_path):
        """Check if SQSH file already exists on remote"""
        self.logger.debug(f"Checking if SQSH file exists: {sqsh_path}")
        check_command = f"test -f {shlex.quote(sqsh_path)} && echo 'exists' || echo 'not_found'"
        ssh_command = self._build_ssh_command(check_command)

        try:
            result = subprocess.run(
                ssh_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=TIMEOUT_SECONDS
            )
            exists = result.stdout.strip() == 'exists'
            if exists:
                self.logger.debug(f"SQSH file found: {sqsh_path}")
            else:
                self.logger.debug(f"SQSH file not found: {sqsh_path}")
            return exists
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            self.logger.warning(f"Failed to check SQSH file existence: {e}")
            return False

    def _convert_to_sqsh_via_srun(self, image, sqsh_path, sqsh_cache_dir,
                                  partition="cpu", account=None,
                                  timeout_minutes=30, memory_gb=32,
                                  job_id=None):
        """Convert Docker image to SQSH format using srun on Slurm cluster

        Args:
            image: Docker image name (e.g., nvcr.io/nvidia/pytorch:24.01-py3)
            sqsh_path: Full path where SQSH file will be saved
            sqsh_cache_dir: Cache directory
            partition: Slurm partition to use for conversion (default: cpu)
            account: Slurm account (optional)
            timeout_minutes: Maximum time for conversion (default: 30 minutes)
            memory_gb: Memory to allocate in GB (default: 32)
            job_id: TAO job ID for status updates (optional)

        Returns:
            bool: True if conversion successful, False otherwise

        Note:
            Uses 'cpu' partition by default since enroot import doesn't require GPUs.
            Allocates 32GB memory by default for large image extraction.
        """
        self.logger.info("=" * 80)
        self.logger.info("SQSH CONVERSION STARTING")
        self.logger.info(f"Docker image: {image}")
        self.logger.info(f"Target SQSH: {sqsh_path}")
        self.logger.info(f"Cache dir: {sqsh_cache_dir}")
        self.logger.info(f"Partition: {partition}")
        self.logger.info(f"Account: {account}")
        self.logger.info(f"Memory: {memory_gb}GB")
        self.logger.info(f"Timeout: {timeout_minutes} minutes")
        self.logger.info("=" * 80)

        # Update job status if job_id provided
        if job_id:
            try:
                automl_params = self.get_automl_aware_handler_params(job_id)
                lookup_id = automl_params['handler_lookup_id']

                handler_id = get_handler_id(lookup_id)
                handler_metadata = get_handler_metadata(lookup_id, kind=None)
                handler_kind = get_handler_kind(handler_metadata)

                update_params = {
                    "handler_id": handler_id,
                    "job_id": lookup_id,
                    "kind": handler_kind,
                    "message": f"Converting Docker image to SQSH format for SLURM execution (partition: {partition})"
                }
                if automl_params['is_automl_experiment']:
                    update_params["automl_expt_job_id"] = automl_params['experiment_job_id']
                    update_params["update_automl_expt"] = True

                update_job_message(**update_params)
            except Exception as e:
                self.logger.warning(f"Failed to update job status: {e}")

        # Create cache directory if it doesn't exist
        self.logger.info(f"Creating cache directory: {sqsh_cache_dir}")
        mkdir_command = f"mkdir -p {shlex.quote(sqsh_cache_dir)}"
        ssh_command = self._build_ssh_command(mkdir_command)

        try:
            subprocess.run(ssh_command, check=True, capture_output=True, text=True)
            self.logger.info("Cache directory created successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create cache directory: {e.stderr}")
            return False

        # Convert image format for enroot
        # From: nvcr.io/nvidia/pytorch:24.01-py3
        # To: docker://nvcr.io#nvidia/pytorch:24.01-py3
        self.logger.info("Converting image name format for enroot...")
        if image.startswith(('docker://', 'dockerd://')):
            docker_image = image
            self.logger.info(f"Image already has docker:// prefix: {docker_image}")
        else:
            # Split on first slash to separate registry from image path
            if '/' in image:
                parts = image.split('/', 1)
                # Use # between registry and image path as per documentation
                docker_image = f"docker://{parts[0]}#{parts[1]}"
                self.logger.info(f"Converted: {image} → {docker_image}")
            else:
                docker_image = f"docker://{image}"
                self.logger.info(f"Added docker:// prefix: {docker_image}")

        # Build srun command for conversion
        # Format: srun -n1 -p <partition> -A <account> --mem=<memory>G -t <time>
        # enroot import -o <output.sqsh> -- <docker-image>
        srun_parts = [
            "srun",
            "-n1",
            f"-p {partition}",
            f"--mem={memory_gb}G",
            f"-t {timeout_minutes}:00"
        ]

        if account:
            srun_parts.append(f"-A {account}")

        srun_parts.extend([
            "enroot", "import",
            "-o", shlex.quote(sqsh_path),
            "--",
            docker_image
        ])

        srun_command = " ".join(srun_parts)

        self.logger.info("Submitting SQSH conversion job via srun...")
        self.logger.info(f"Command: {srun_command}")
        self.logger.info(f"This will block for up to {timeout_minutes + 1} minutes...")
        ssh_command = self._build_ssh_command(srun_command)

        start_time = time.time()

        try:
            # Submit the conversion job
            result = subprocess.run(
                ssh_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=timeout_minutes * 60 + 60  # Add 1 minute buffer
            )

            elapsed_time = time.time() - start_time
            self.logger.info(
                f"Conversion command completed in {elapsed_time:.1f} seconds ({elapsed_time / 60:.1f} minutes)"
            )

            if result.stdout:
                self.logger.info(f"Stdout: {result.stdout.strip()}")
            if result.stderr:
                self.logger.warning(f"Stderr: {result.stderr.strip()}")

            # Verify the file was created
            self.logger.info("Verifying SQSH file was created...")
            if self._check_sqsh_exists(sqsh_path):
                self.logger.info("=" * 80)
                self.logger.info("SQSH CONVERSION SUCCESSFUL!")
                self.logger.info(f"File: {sqsh_path}")
                self.logger.info(f"Duration: {elapsed_time:.1f}s ({elapsed_time / 60:.1f} min)")
                self.logger.info("=" * 80)

                # Update job status if job_id provided
                if job_id:
                    try:
                        automl_params = self.get_automl_aware_handler_params(job_id)
                        lookup_id = automl_params['handler_lookup_id']

                        handler_id = get_handler_id(lookup_id)
                        handler_metadata = get_handler_metadata(lookup_id, kind=None)
                        handler_kind = get_handler_kind(handler_metadata)

                        update_params = {
                            "handler_id": handler_id,
                            "job_id": lookup_id,
                            "kind": handler_kind,
                            "message": "Container image conversion completed. Submitting job to SLURM..."
                        }
                        if automl_params['is_automl_experiment']:
                            update_params["automl_expt_job_id"] = automl_params['experiment_job_id']
                            update_params["update_automl_expt"] = True

                        update_job_message(**update_params)
                    except Exception as e:
                        self.logger.warning(f"Failed to update job status: {e}")

                return True
            self.logger.error("=" * 80)
            self.logger.error("SQSH CONVERSION FAILED!")
            self.logger.error(f"SQSH file was not created: {sqsh_path}")
            self.logger.error("Command completed but file doesn't exist")
            self.logger.error("=" * 80)
            return False

        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            self.logger.error("=" * 80)
            self.logger.error("SQSH CONVERSION TIMEOUT!")
            self.logger.error(f"Conversion timed out after {timeout_minutes} minutes")
            self.logger.error(f"Elapsed: {elapsed_time:.1f}s ({elapsed_time / 60:.1f} min)")
            self.logger.error(f"Image: {image}")
            self.logger.error("Consider increasing conversion_timeout_minutes")
            self.logger.error("=" * 80)
            return False
        except subprocess.CalledProcessError as e:
            elapsed_time = time.time() - start_time
            self.logger.error("=" * 80)
            self.logger.error("SQSH CONVERSION ERROR!")
            self.logger.error(f"Command failed after {elapsed_time:.1f}s ({elapsed_time / 60:.1f} min)")
            self.logger.error(f"Image: {image}")
            self.logger.error(f"Stderr: {e.stderr}")
            self.logger.error("=" * 80)
            return False

    def _prepare_container_image(self, image, sqsh_cache_dir,
                                 conversion_partition="cpu",
                                 conversion_account=None,
                                 conversion_timeout_minutes=30,
                                 conversion_memory_gb=32,
                                 force_reconvert_latest=False,
                                 job_id=None):
        """Prepare container image in SQSH format using Slurm srun

        Args:
            image: Docker image name (e.g., nvcr.io/nvidia/pytorch:24.01-py3)
            sqsh_cache_dir: Directory to cache SQSH files (required, typically job_dir)
            conversion_partition: Partition to use for conversion job (default: cpu)
            conversion_account: Account to use for conversion (optional, uses job account if None)
            conversion_timeout_minutes: Timeout for conversion in minutes (default: 30)
            conversion_memory_gb: Memory to allocate in GB (default: 32)
            force_reconvert_latest: If True, always reconvert :latest tagged images (default: False)
            job_id: TAO job ID for status updates (optional)

        Returns:
            str: Path to SQSH file or original image format

        Note:
            This method runs SYNCHRONOUSLY. The srun command blocks until conversion
            completes or times out (default: 30 minutes + 1 minute buffer = 31 minutes).
            Uses 'cpu' partition by default since enroot import doesn't need GPUs.
            Allocates 32GB memory by default for large image extraction.

            Caching behavior:
            - By default, ALL images (including :latest) use cached SQSH files if available
            - Set force_reconvert_latest=True to always reconvert :latest tagged images
            - This is useful when you need to ensure the absolute latest image version
        """
        self.logger.info(f"Preparing container image for job: {image}")

        # Initialize job status update handlers (common for all paths)
        # AutoML-aware: use brain job ID for handler lookups
        handler_id = None
        handler_kind = None
        update_job_message = None
        automl_params = None

        if job_id:
            try:
                automl_params = self.get_automl_aware_handler_params(job_id)
                lookup_id = automl_params['handler_lookup_id']

                handler_id = get_handler_id(lookup_id)
                handler_metadata = get_handler_metadata(lookup_id, kind=None)
                handler_kind = get_handler_kind(handler_metadata)
            except Exception as e:
                self.logger.warning(f"Failed to initialize job status handlers: {e}")

        # Use provided cache directory
        cache_dir = sqsh_cache_dir

        # Generate SQSH filename
        sqsh_filename = self._get_sqsh_filename(image)
        sqsh_path = f"{cache_dir}/{sqsh_filename}"

        self.logger.info(f"SQSH cache location: {sqsh_path}")

        # Check if image uses 'latest' tag
        is_latest_tag = ':latest' in image or image.endswith(':latest')

        # Determine if we should force reconversion for :latest tags
        should_force_reconvert = is_latest_tag and force_reconvert_latest

        if should_force_reconvert:
            self.logger.info("⚠ Image uses ':latest' tag with force_reconvert_latest=True - will re-convert")
            self.logger.info("(Skipping cache check and forcing fresh conversion)")
            # Delete existing SQSH file if it exists (enroot import fails on existing files)
            if self._check_sqsh_exists(sqsh_path):
                self.logger.info(f"Deleting existing SQSH file: {sqsh_path}")
                delete_command = f"rm -f {shlex.quote(sqsh_path)}"
                ssh_command = self._build_ssh_command(delete_command)
                try:
                    subprocess.run(ssh_command, check=True, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
                    self.logger.info("Existing SQSH file deleted successfully")
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    self.logger.warning(f"Failed to delete existing SQSH file: {e}")
                    self.logger.warning("Conversion may fail if file cannot be overwritten")
        else:
            # Check if SQSH already exists (use cache for all images including :latest by default)
            self.logger.info("Checking for existing SQSH file in cache...")
            if is_latest_tag:
                self.logger.info("(Note: Using cache for :latest tag. Set force_reconvert_latest=True to force update)")
            if self._check_sqsh_exists(sqsh_path):
                self.logger.info("Found existing SQSH file - using cached version")
                self.logger.info(f"Cached file: {sqsh_path}")
                self.logger.info("Skipping conversion (using cached SQSH)")

                # Update job status
                if update_job_message and handler_id and handler_kind and automl_params:
                    try:
                        update_params = {
                            "handler_id": handler_id,
                            "job_id": lookup_id,
                            "kind": handler_kind,
                            "message": "Using cached SQSH image (skipping conversion)"
                        }
                        if automl_params and automl_params['is_automl_experiment']:
                            update_params["automl_expt_job_id"] = automl_params['experiment_job_id']
                            update_params["update_automl_expt"] = True

                        update_job_message(**update_params)
                    except Exception as e:
                        self.logger.warning(f"Failed to update job status: {e}")

                return sqsh_path  # Return direct path, not file:// prefix

        # Convert to SQSH using srun
        if not should_force_reconvert:
            self.logger.info("✗ SQSH file not found in cache")
        self.logger.info(f"Starting conversion: {image} → SQSH")
        self.logger.info("This may take 5-30 minutes depending on image size...")

        # Update job status - conversion starting
        if update_job_message and handler_id and handler_kind and automl_params:
            try:
                update_params = {
                    "handler_id": handler_id,
                    "job_id": lookup_id,
                    "kind": handler_kind,
                    "message": "Starting SQSH conversion for container image (may take 5-30 minutes)"
                }
                if automl_params and automl_params['is_automl_experiment']:
                    update_params["automl_expt_job_id"] = automl_params['experiment_job_id']
                    update_params["update_automl_expt"] = True

                update_job_message(**update_params)
            except Exception as e:
                self.logger.warning(f"Failed to update job status: {e}")

        try:
            success = self._convert_to_sqsh_via_srun(
                image, sqsh_path, cache_dir,
                partition=conversion_partition,
                account=conversion_account,
                timeout_minutes=conversion_timeout_minutes,
                memory_gb=conversion_memory_gb,
                job_id=job_id
            )

            if success:
                self.logger.info(f"SQSH preparation complete - using: {sqsh_path}")

                # Update job status - conversion complete
                if update_job_message and handler_id and handler_kind and automl_params:
                    try:
                        update_params = {
                            "handler_id": handler_id,
                            "job_id": lookup_id,
                            "kind": handler_kind,
                            "message": "SQSH conversion complete - preparing to submit job to SLURM"
                        }
                        if automl_params and automl_params['is_automl_experiment']:
                            update_params["automl_expt_job_id"] = automl_params['experiment_job_id']
                            update_params["update_automl_expt"] = True

                        update_job_message(**update_params)
                    except Exception as e:
                        self.logger.warning(f"Failed to update job status: {e}")

                return sqsh_path  # Return direct path, not file:// prefix
            self.logger.warning("✗ SQSH conversion failed")
            self.logger.warning(f"Falling back to Docker image: {image}")
            self.logger.warning("Job will use Docker (may have slower startup)")
            return image
        except Exception as e:
            self.logger.error(f"✗ Exception during SQSH conversion: {e}")
            self.logger.warning(f"Falling back to Docker image: {image}")
            self.logger.warning("Job will use Docker (may have slower startup)")
            self.logger.debug(traceback.format_exc())
            return image

    def _scp_text(self, content, remote_path):
        """Copy text content to remote_path using scp with SSH options.

        Creates parent directory on remote if it doesn't exist.
        """
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            local_path = tmp.name
        try:
            # Create parent directory on remote if it doesn't exist
            remote_dir = os.path.dirname(remote_path)
            if remote_dir:
                mkdir_command = f"mkdir -p {shlex.quote(remote_dir)}"
                ssh_command = self._build_ssh_command(mkdir_command)
                try:
                    subprocess.run(
                        ssh_command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True,
                        timeout=TIMEOUT_SECONDS
                    )
                    self.logger.debug(f"Created remote directory: {remote_dir}")
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"Failed to create remote directory {remote_dir}: {e.stderr.strip()}")
                    # Continue anyway - directory might already exist or scp might still work

            # Copy file to remote
            scp_command = [
                "scp",
            ] + self.ssh_options + [
                local_path,
                f"{self.login_user}@{self.login_hostname}:{remote_path}"
            ]
            subprocess.run(
                scp_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            self.logger.debug(f"Successfully copied to {remote_path}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to scp to {remote_path}: {e.stderr.strip()}")
            raise
        finally:
            try:
                os.unlink(local_path)
            except OSError:
                pass

    def check_path_exists(self, remote_path):
        """Check if a path exists on the remote SLURM cluster.

        Args:
            remote_path: Path to check on remote SLURM cluster

        Returns:
            bool: True if path exists, False otherwise
        """
        check_command = f"test -e {shlex.quote(remote_path)} && echo 'exists' || echo 'not_found'"
        ssh_command = self._build_ssh_command(check_command)

        try:
            result = subprocess.run(
                ssh_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=TIMEOUT_SECONDS
            )
            exists = result.stdout.strip() == 'exists'
            self.logger.debug(f"Path existence check for {remote_path}: {exists}")
            return exists
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            self.logger.warning(f"Failed to check path existence for {remote_path}: {e}")
            return False

    def test_ssh_connection(self):
        """Test SSH connection to the Slurm cluster"""
        self.logger.info(f"Testing SSH connection to {self.login_user}@{self.login_hostname}")
        self.logger.info(f"Using SSH key: {self.ssh_key_path}")

        # Check if SSH key exists and is readable
        if self.ssh_key_path:
            if os.path.exists(self.ssh_key_path):
                try:
                    with open(self.ssh_key_path, 'r', encoding='utf-8') as f:
                        f.read(1)  # Try to read first byte
                    self.logger.info(f"SSH key is readable: {self.ssh_key_path}")
                except Exception as e:
                    self.logger.warning(f"SSH key exists but is not readable: {e}")
            else:
                self.logger.warning(f"SSH key does not exist: {self.ssh_key_path}")

        test_command = "echo 'SSH connection successful'"
        ssh_command = self._build_ssh_command(test_command)
        self.logger.info(f"SSH command: {' '.join(ssh_command)}")

        try:
            result = subprocess.run(
                ssh_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=TIMEOUT_SECONDS
            )
            self.logger.info(f"SSH test successful: {result.stdout.strip()}")
            return True, result.stdout.strip()
        except subprocess.CalledProcessError as e:
            error_msg = f"SSH test failed: {e.stderr.strip()}"
            self.logger.error(error_msg)
            self.logger.error(f"SSH command that failed: {' '.join(ssh_command)}")
            return False, error_msg
        except subprocess.TimeoutExpired:
            error_msg = "SSH test timed out"
            self.logger.error(error_msg)
            return False, error_msg

    def get_run_command(self, network, action, job_id, job_dir):
        """Build the Python module argv.

        Uses file path arguments to avoid long command lines.
        """
        # Reference the staged JSON files using -file arguments
        # Include job_id to avoid conflicts when multiple jobs run simultaneously
        specs_file = f"{job_dir}/specs/{job_id}.json"
        env_file = f"{job_dir}/env/{job_id}.json"
        meta_file = f"{job_dir}/meta/{job_id}.json"

        command_list = [
            "-m",
            "nvidia_tao_core.microservices.handlers.container_handler",
            "--neural-network-name",
            shlex.quote(network),
            "--action-name",
            shlex.quote(action),
            "--job-id",
            shlex.quote(job_id),
            "--specs-file",
            shlex.quote(specs_file),
            "--docker-env-vars-file",
            shlex.quote(env_file),
            "--cloud-metadata-file",
            shlex.quote(meta_file),
        ]

        return " ".join(command_list)

    def create_job(
            self,
            image,
            network,
            action,
            cloud_metadata={},
            specs={},
            job_id="",
            docker_env_vars={},
            num_nodes=1,
            num_gpus=4,
            cpus_per_task=16,
            time_hours=4,
            account="edgeai_tao-ptm_image-foundation-model-clip",
            partition=None,
            container_mounts="/lustre",
            log_dir=None,
            mail_user=None,
            use_timeout=True,
            timeout_hours=3.8,
            use_requeue=True,
            use_srun=False,
            job_dir=None,
            # SQSH conversion parameters
            sqsh_cache_dir=None,
            use_sqsh=True,
            conversion_partition="cpu",
            conversion_timeout_minutes=30,
            conversion_memory_gb=32,
            force_reconvert_latest=False,
            max_num_gpus=8,
            **kwargs
    ):
        """Create a comprehensive Slurm job with container support and advanced features.

        Args:
            image: Docker image name (will be converted to SQSH if use_sqsh=True)
            sqsh_cache_dir: Directory to cache SQSH files (defaults to job_dir)
            use_sqsh: Whether to convert Docker images to SQSH format (default: True)
            conversion_partition: Partition to use for SQSH conversion (default: cpu)
            conversion_timeout_minutes: Timeout for SQSH conversion in minutes (default: 30)
            conversion_memory_gb: Memory to allocate for conversion in GB (default: 32)
            force_reconvert_latest: Force reconversion of :latest tagged images (default: False)

        Note on SQSH Conversion:
            - The conversion runs SYNCHRONOUSLY via `srun` command over SSH
            - The code will BLOCK and WAIT for conversion to complete
            - Timeout: conversion_timeout_minutes + 1 minute buffer (default: 31 minutes)
            - Uses 'cpu' partition by default (doesn't need GPUs)
            - Allocates 32GB memory by default for large image extraction
            - If conversion fails, falls back to using Docker image directly

        Caching Behavior:
            - By default, ALL images (including :latest) use cached SQSH files if available
            - This speeds up job submission by avoiding redundant 5-30 minute conversions
            - Set force_reconvert_latest=True to always reconvert :latest tagged images
            - Cached SQSH files are stored in sqsh_cache_dir for reuse across jobs
        """
        if specs is None:
            specs = {}
        if docker_env_vars is None:
            docker_env_vars = {}
        if cloud_metadata is None:
            cloud_metadata = {}
        # Set default timeout if not specified
        if timeout_hours is None:
            timeout_hours = time_hours - 0.2  # 12 minutes buffer

        if num_gpus < max_num_gpus:
            exclusive = False
        else:
            # num_gpus may represent total GPUs across all nodes (e.g., cosmos-rl
            # passes total_gpus=64). Compute the correct num_nodes before capping
            # num_gpus to the per-node limit.
            computed_nodes = math.ceil(num_gpus / max_num_gpus)
            if computed_nodes > num_nodes:
                self.logger.info(
                    f"Auto-computed num_nodes={computed_nodes} from "
                    f"total_gpus={num_gpus} / max_num_gpus={max_num_gpus} "
                    f"(overriding provided num_nodes={num_nodes})"
                )
                num_nodes = computed_nodes
            exclusive = True
            num_gpus = max_num_gpus

        # Detect multi-node after potential num_nodes recalculation above
        is_multi_node = num_nodes > 1
        if is_multi_node:
            self.logger.info(f"Detected multi-node job: {num_nodes} nodes")

        if mail_user is None:
            mail_user = f"{self.login_user}@nvidia.com"
            self.logger.info(f"Setting mail_user to {mail_user}")
        else:
            self.logger.info(f"Using provided mail_user: {mail_user}")

        if job_dir is None:
            job_dir = f"/lustre/fsw/portfolios/edgeai/users/{self.login_user}"
            self.logger.info(f"Setting job_dir to {job_dir}")
        else:
            self.logger.info(f"Using provided job_dir: {job_dir}")

        if log_dir is None:
            log_dir = f"{job_dir}/slurm-logs"
            self.logger.info(f"Setting log_dir to {log_dir}")
        else:
            self.logger.info(f"Using provided log_dir: {log_dir}")

        if sqsh_cache_dir is None:
            sqsh_cache_dir = f"/lustre/fsw/portfolios/edgeai/users/{self.login_user}"
            self.logger.info(f"Setting sqsh_cache_dir to {sqsh_cache_dir}")
        else:
            self.logger.info(f"Using provided sqsh_cache_dir: {sqsh_cache_dir}")

        if not partition:
            partition = "polar,polar3,polar4,grizzly"
            self.logger.info(f"Setting partition to default: {partition}")
        else:
            self.logger.info(f"Using provided partition: {partition}")

        job_name = self.get_slurm_job_name(job_id)

        # Helper function to recursively convert slurm:// URLs to actual Lustre paths
        def convert_slurm_urls(obj, path=""):
            """Recursively convert slurm:// URLs in dicts/lists to actual Lustre paths."""
            if isinstance(obj, dict):
                for key, value in list(obj.items()):
                    new_path = f"{path}.{key}" if path else key
                    if isinstance(value, str) and value.startswith("slurm://"):
                        _, _, actual_path = get_file_path_from_cloud_string(value)
                        obj[key] = actual_path
                        self.logger.debug(f"Converted {new_path}: {value} -> {actual_path}")
                    elif isinstance(value, (dict, list)):
                        convert_slurm_urls(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]"
                    if isinstance(item, str) and item.startswith("slurm://"):
                        _, _, actual_path = get_file_path_from_cloud_string(item)
                        obj[i] = actual_path
                        self.logger.debug(f"Converted {new_path}: {item} -> {actual_path}")
                    elif isinstance(item, (dict, list)):
                        convert_slurm_urls(item, new_path)

        # Convert any slurm:// URLs in docker_env_vars and specs to actual Lustre paths
        # Jobs running INSIDE SLURM should use actual paths, not slurm:// URLs
        self.logger.info("Converting slurm:// URLs to actual Lustre paths...")
        convert_slurm_urls(docker_env_vars, "docker_env_vars")
        convert_slurm_urls(specs, "specs")

        # Determine results directory
        results_dir = specs.get("results_dir")
        if not results_dir:
            # Fallback to constructing it from job_dir
            results_dir = f"{job_dir}/results/{job_id}"
            specs["results_dir"] = results_dir
            self.logger.info(f"Using default results_dir: {results_dir}")

        # Set TAO_API_RESULTS_DIR to the BASE directory (without job_id)
        # Other code appends /{job_id} to this value
        # Extract base by removing the job_id from the end
        results_base_dir = results_dir.rsplit('/', 1)[0] if '/' in results_dir else results_dir
        docker_env_vars["TAO_API_RESULTS_DIR"] = results_base_dir
        self.logger.info(f"Setting TAO_API_RESULTS_DIR (base): {results_base_dir}")
        self.logger.info(f"Full results path for job: {results_dir}")

        if network == "cosmos-rl":
            cosmos_cache_dir = f"{job_dir}/.cache/cosmos"
            docker_env_vars["COSMOS_CACHE"] = cosmos_cache_dir
            self.logger.info(f"Setting COSMOS_CACHE to location: {cosmos_cache_dir}")

            # Set CUDA override for cosmos-rl to prevent flashinfer JIT compilation issues
            # This ensures the container uses system CUDA instead of lustre miniconda paths
            if "CUDA_OVERRIDE_VERSION" not in docker_env_vars:
                docker_env_vars["CUDA_OVERRIDE_VERSION"] = "12.8"
                self.logger.info("Setting CUDA_OVERRIDE_VERSION=12.8 for cosmos-rl (prevents flashinfer JIT issues)")

        # Set Hugging Face cache to lustre to avoid node-local cache issues
        # This ensures all ranks share the same model cache instead of each downloading to local /root/.cache
        if "HF_HOME" not in docker_env_vars and "HUGGINGFACE_HUB_CACHE" not in docker_env_vars:
            hf_cache_dir = f"{job_dir}/.cache/huggingface"
            docker_env_vars["HF_HOME"] = hf_cache_dir
            self.logger.info(f"Setting HF_HOME to lustre location: {hf_cache_dir}")

        # Isolate container's Python environment from host/mounted filesystems
        # This prevents Python import errors when /lustre is mounted and contains Python packages
        docker_env_vars["PYTHONNOUSERSITE"] = "1"  # Ignore user site-packages
        docker_env_vars["PYTHONDONTWRITEBYTECODE"] = "1"  # Don't write .pyc files to mounted FS
        self.logger.info("Set Python isolation variables to prevent import conflicts with mounted filesystems")

        # Prepare container image (convert to SQSH if needed)
        # Use job_dir as the default cache directory
        if use_sqsh:
            self.logger.info(f"Preparing container image: {image}")
            cache_dir = sqsh_cache_dir if sqsh_cache_dir else job_dir
            image = self._prepare_container_image(
                image,
                sqsh_cache_dir=cache_dir,
                conversion_partition=conversion_partition,
                conversion_account=account,
                conversion_timeout_minutes=conversion_timeout_minutes,
                conversion_memory_gb=conversion_memory_gb,
                force_reconvert_latest=force_reconvert_latest,
                job_id=job_id
            )
            self.logger.info(f"Using container image: {image}")

        # Stage JSON payloads as files to avoid huge argv/export
        # Include job_id in filenames to avoid conflicts when multiple jobs run simultaneously
        specs_json = self._to_compact_json(specs)
        env_json = self._to_compact_json(docker_env_vars)
        meta_json = self._to_compact_json(cloud_metadata)

        self._scp_text(specs_json, f"{job_dir}/specs/{job_id}.json")
        self._scp_text(env_json, f"{job_dir}/env/{job_id}.json")
        self._scp_text(meta_json, f"{job_dir}/meta/{job_id}.json")

        command = self.get_run_command(
            network,
            action,
            job_id,
            job_dir,
        )

        self.logger.debug(f"partition in slurm_handler create_job: {partition}")

        # For cosmos-rl multi-node: compute node metadata and inject as env vars
        # so that vlm_entrypoint.py can determine each node's role and launch
        # the appropriate cosmos-rl workers (controller, policy, rollout)
        if network == "cosmos-rl" and is_multi_node:
            node_info = self._compute_cosmos_rl_node_metadata(specs, num_gpus)
            docker_env_vars["COSMOS_RL_MULTINODE"] = "1"
            docker_env_vars["NUM_POLICY_NODES"] = str(node_info["n_policy_nodes"])
            docker_env_vars["NUM_ROLLOUT_NODES"] = str(node_info["n_rollout_nodes"])
            docker_env_vars["TOTAL_NODES"] = str(node_info["total_nodes"])
            docker_env_vars["NODE_LAUNCH_METADATA_POLICY"] = json.dumps(
                node_info["policy_metadata"]
            )
            docker_env_vars["NODE_LAUNCH_METADATA_ROLLOUT"] = json.dumps(
                node_info["rollout_metadata"]
            )
            docker_env_vars["CONTROLLER_PORT"] = "8082"
            self.logger.info(
                f"[COSMOS-RL] Multi-node metadata: "
                f"{node_info['n_policy_nodes']} policy nodes "
                f"({node_info['n_policy_replicas']} replicas × "
                f"{node_info['min_gpus_policy']} GPUs), "
                f"{node_info['n_rollout_nodes']} rollout nodes "
                f"({node_info['n_rollout_replicas']} replicas × "
                f"{node_info['min_gpus_rollout']} GPUs), "
                f"total={node_info['total_nodes']} nodes"
            )
            # Override num_nodes with the precise cosmos-rl total
            num_nodes = node_info["total_nodes"]

        # Build the SLURM script content (unified path for all networks)
        slurm_script = self._build_slurm_script(
            job_name=job_name,
            num_nodes=num_nodes,
            num_gpus=num_gpus,
            cpus_per_task=cpus_per_task,
            time_hours=time_hours,
            account=account,
            partition=partition,
            log_dir=log_dir,
            mail_user=mail_user,
            image=image,
            container_mounts=container_mounts,
            use_timeout=use_timeout,
            timeout_hours=timeout_hours,
            use_requeue=use_requeue,
            use_srun=use_srun,
            command=command,
            docker_env_vars=docker_env_vars,
            exclusive=exclusive,
            network=network,
        )
        # Write script to local temp file, submit it, then clean up
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sbatch', delete=False) as tmp:
            tmp.write(slurm_script)
            tmp.flush()
            local_script = tmp.name

        try:
            # scp the script to remote job_dir
            # Include job_id in filename to avoid conflicts when multiple jobs run simultaneously
            remote_script = f"{job_dir}/sbatch/job_{job_id}.sbatch"
            self._scp_text(slurm_script, remote_script)

            # Submit with sbatch using the remote script path
            remote_command = f"sbatch --export=ALL {shlex.quote(remote_script)}"
            ssh_command = self._build_ssh_command(remote_command)

            result = subprocess.run(
                ssh_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            sbatch_output = result.stdout.strip()
            self.logger.info(f"Slurm job submitted: {sbatch_output}")

            # Extract SLURM job ID and store in metadata
            slurm_job_id = self.get_slurm_job_id_from_output(sbatch_output)
            if slurm_job_id:
                self.logger.info(f"SLURM job ID: {slurm_job_id}")
                try:
                    # Check if this is an AutoML experiment by looking for AUTOML_EXPERIMENT_NUMBER
                    is_automl_experiment = "AUTOML_EXPERIMENT_NUMBER" in docker_env_vars

                    if is_automl_experiment:
                        # For AutoML experiments: store SLURM job ID in controller info
                        experiment_number = docker_env_vars.get("AUTOML_EXPERIMENT_NUMBER", "0")
                        self.logger.info(f"Detected AutoML experiment {experiment_number} with job_id {job_id}")

                        brain_job_id = self.get_automl_brain_job_id(job_id)
                        if brain_job_id:
                            # Store SLURM job ID in controller info
                            controller_info = get_automl_controller_info(brain_job_id)
                            if isinstance(controller_info, list) and int(experiment_number) < len(controller_info):
                                rec = controller_info[int(experiment_number)]
                                if not rec.get("backend_details", {}):
                                    rec["backend_details"] = {'backend_type': 'slurm'}
                                if not rec["backend_details"].get("slurm_metadata", {}):
                                    rec["backend_details"]["slurm_metadata"] = {}

                                rec["backend_details"]["slurm_metadata"]["slurm_job_id"] = slurm_job_id
                                rec["backend_details"]["slurm_metadata"]["job_dir"] = job_dir
                                save_automl_controller_info(brain_job_id, controller_info)
                                self.logger.info(
                                    f"Stored SLURM job ID {slurm_job_id} and job_dir {job_dir} "
                                    f"in AutoML controller info for brain {brain_job_id}, "
                                    f"experiment {experiment_number}"
                                )
                        else:
                            self.logger.warning(f"Could not find brain job for AutoML experiment {job_id}")
                    else:
                        # For regular jobs and brain jobs: store in job metadata
                        job_metadata = get_handler_job_metadata(job_id)
                        if not job_metadata:
                            self.logger.warning(f"No metadata found for job {job_id}, cannot store SLURM job ID")
                        else:
                            if not job_metadata.get("backend_details", {}):
                                job_metadata["backend_details"] = {'backend_type': 'slurm'}
                            if not job_metadata["backend_details"].get("slurm_metadata", {}):
                                job_metadata["backend_details"]["slurm_metadata"] = {}

                            job_metadata["backend_details"]["slurm_metadata"]["slurm_job_id"] = slurm_job_id
                            job_metadata["backend_details"]["slurm_metadata"]["job_dir"] = job_dir
                            write_job_metadata(job_id, job_metadata)
                            self.logger.info(
                                f"Stored SLURM job ID {slurm_job_id} and job_dir {job_dir} "
                                f"in job metadata for {job_id}"
                            )

                    # Update job message with SLURM submission details
                    # For AutoML experiments, use brain_job_id for handler lookups
                    lookup_id = brain_job_id if brain_job_id else job_id
                    handler_id = get_handler_id(lookup_id)
                    handler_metadata = get_handler_metadata(lookup_id, kind=None)
                    handler_kind = get_handler_kind(handler_metadata)

                    update_params = {
                        "handler_id": handler_id,
                        "job_id": lookup_id,
                        "kind": handler_kind,
                        "message": f"Job submitted to SLURM cluster (Job ID: {slurm_job_id}). Waiting for resources..."
                    }
                    if brain_job_id:
                        # AutoML experiment: add experiment-specific params
                        update_params["automl_expt_job_id"] = job_id
                        update_params["update_automl_expt"] = True

                    update_job_message(**update_params)
                except Exception as e:
                    self.logger.warning(f"Failed to store SLURM job ID or update status: {e}")

            return sbatch_output
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to submit Slurm job: {e.stderr.strip()}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to create job {job_id}: {e}")
            raise
        finally:
            # Clean up local temp file
            try:
                os.unlink(local_script)
            except OSError:
                pass

    def get_slurm_job_id(self, job_id):
        """Get the SLURM job ID for a given TAO job ID.

        Args:
            job_id (str): TAO job ID

        Returns:
            str: SLURM job ID, or None if not found
        """
        try:
            # Get AutoML-aware parameters
            automl_params = self.get_automl_aware_handler_params(job_id)
            brain_job_id = automl_params['brain_job_id']
            is_automl_experiment = automl_params['is_automl_experiment']

            # Get job metadata to find SLURM job ID
            job_metadata = get_handler_job_metadata(brain_job_id)
            if not job_metadata:
                self.logger.warning(f"No metadata found for job {job_id}")
                return None

            # Get SLURM job ID
            slurm_job_id = None
            if is_automl_experiment:
                controller_info = get_automl_controller_info(brain_job_id)
                experiment_number = automl_params['experiment_number']
                if controller_info and len(controller_info) > experiment_number:
                    experiment_info = controller_info[experiment_number]
                    backend_details = experiment_info.get('backend_details', {})
                    slurm_metadata = backend_details.get('slurm_metadata', {})
                    slurm_job_id = slurm_metadata.get('slurm_job_id')
                    self.logger.info(f"AutoML experiment {experiment_number}, SLURM job ID: {slurm_job_id}")
            else:
                # Regular job - get SLURM job ID from job metadata
                backend_details = job_metadata.get('backend_details', {})
                slurm_metadata = backend_details.get('slurm_metadata', {})
                slurm_job_id = slurm_metadata.get('slurm_job_id')
                self.logger.info(f"Regular job, SLURM job ID: {slurm_job_id}")

            if not slurm_job_id:
                self.logger.warning(f"No SLURM job ID found for TAO job {job_id}")
                self.logger.info("Job may not have been submitted to SLURM yet")
                self.logger.info("=" * 80)
                return None
            return slurm_job_id
        except Exception as e:
            self.logger.error(f"Error getting SLURM job ID for job {job_id}: {e}")
            self.logger.error(traceback.format_exc())
        return None

    def _get_stored_job_dir(self, job_id):
        """Get job_dir from stored job metadata.

        Args:
            job_id (str): TAO job ID

        Returns:
            str: The stored job_dir, or None if not found
        """
        try:
            # First try regular job metadata
            job_metadata = get_handler_job_metadata(job_id)
            if job_metadata:
                backend_details = job_metadata.get("backend_details") or {}
                slurm_metadata = backend_details.get("slurm_metadata") or {}
                job_dir = slurm_metadata.get("job_dir")
                if job_dir:
                    return job_dir

            # Check if this is an AutoML experiment
            if "_expt" in job_id:
                parts = job_id.split("_expt")
                if len(parts) == 2:
                    potential_brain_id = parts[0]
                    try:
                        experiment_number = int(parts[1])
                        controller_info = get_automl_controller_info(potential_brain_id)
                        if controller_info and len(controller_info) > experiment_number:
                            experiment_info = controller_info[experiment_number]
                            backend_details = experiment_info.get("backend_details") or {}
                            slurm_metadata = backend_details.get("slurm_metadata") or {}
                            return slurm_metadata.get("job_dir")
                    except (ValueError, IndexError):
                        pass
        except Exception as e:
            self.logger.warning(f"Error fetching stored job_dir for job {job_id}: {e}")
        return None

    def get_job_logs(self, job_id, tail_lines=None, log_dir=None):
        """Get the logs of a SLURM job by downloading from remote cluster.

        Args:
            job_id (str): TAO job ID
            tail_lines (int, optional): Number of lines to tail. If None, gets all logs
            log_dir (str): Base log directory on SLURM cluster (default: None, will try to fetch from metadata)

        Returns:
            str: Combined log content (stdout + stderr), or None if logs cannot be retrieved
        """
        self.logger.info("=" * 80)
        self.logger.info(f"SLURM LOG RETRIEVAL: Fetching logs for job {job_id}")
        self.logger.info(f"SSH: {self.login_user}@{self.login_hostname}")

        if log_dir is None:
            # Try to fetch job_dir from stored metadata and derive log_dir
            job_dir = self._get_stored_job_dir(job_id)
            if job_dir:
                log_dir = f"{job_dir}/slurm-logs"
                self.logger.info(f"Using log_dir derived from stored job_dir: {log_dir}")
            else:
                # Fallback to default path
                job_dir = f"/lustre/fsw/portfolios/edgeai/users/{self.login_user}"
                log_dir = f"{job_dir}/slurm-logs"
                self.logger.info(f"No stored job_dir found, using fallback log_dir: {log_dir}")
        else:
            self.logger.info(f"Using provided log_dir: {log_dir}")

        try:
            slurm_job_id = self.get_slurm_job_id(job_id)
            if not slurm_job_id:
                return None

            # Get SLURM job name
            job_name = self.get_slurm_job_name(job_id)

            # Construct log directory path: {log_dir}/{job_name}-{slurm_job_id}/
            log_path = f"{log_dir}/{job_name}-{slurm_job_id}"
            self.logger.info(f"Log directory: {log_path}")

            # Try to read both stdout and stderr files
            logs = []

            # Read main.out (stdout)
            stdout_file = f"{log_path}/main.out"
            self.logger.info(f"Reading stdout: {stdout_file}")
            stdout_content = self._read_remote_log_file(stdout_file, tail_lines)
            if stdout_content:
                logs.append(stdout_content)
            else:
                self.logger.info(f"No stdout found at {stdout_file}")

            # Read main.err (stderr)
            stderr_file = f"{log_path}/main.err"
            self.logger.info(f"Reading stderr: {stderr_file}")
            stderr_content = self._read_remote_log_file(stderr_file, tail_lines)
            if stderr_content:
                if logs:
                    logs.append("\n")
                logs.append(stderr_content)
            else:
                self.logger.info(f"No stderr found at {stderr_file}")

            if not logs:
                self.logger.warning(f"No log files found for SLURM job {slurm_job_id}")
                self.logger.info("Job may not have started yet or log directory doesn't exist")
                self.logger.info("=" * 80)
                return None

            combined_logs = "\n".join(logs)
            log_size_kb = len(combined_logs) / 1024
            self.logger.info(f"Successfully retrieved {log_size_kb:.1f} KB of logs")
            self.logger.info("=" * 80)
            return combined_logs

        except Exception as e:
            self.logger.error(f"Error retrieving logs for job {job_id}: {e}")
            self.logger.error(traceback.format_exc())
            self.logger.info("=" * 80)
            return None

    def _read_remote_log_file(self, remote_file_path, tail_lines=None):
        """Read a log file from the remote SLURM cluster via SSH.

        Args:
            remote_file_path (str): Full path to the log file on remote cluster
            tail_lines (int, optional): Number of lines to tail. If None, reads entire file

        Returns:
            str: File content, or None if file cannot be read
        """
        try:
            # Build command: tail -n X or cat
            if tail_lines is not None:
                read_command = f"tail -n {tail_lines} {shlex.quote(remote_file_path)} 2>/dev/null"
                self.logger.debug(f"Using tail with {tail_lines} lines")
            else:
                read_command = f"cat {shlex.quote(remote_file_path)} 2>/dev/null"
                self.logger.debug("Reading entire file with cat")

            ssh_command = self._build_ssh_command(read_command)

            result = subprocess.run(
                ssh_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=False
            )

            if result.returncode != 0:
                self.logger.debug(f"Failed to read {remote_file_path}: return code {result.returncode}")
                return None

            content = result.stdout
            if not content:
                self.logger.debug(f"File {remote_file_path} is empty")
                return None

            line_count = len(content.splitlines())
            size_kb = len(content) / 1024
            self.logger.debug(f"Read {line_count} lines ({size_kb:.1f} KB) from {remote_file_path}")
            return content

        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout reading file {remote_file_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading remote file {remote_file_path}: {e}")
            return None

    @staticmethod
    def _compute_cosmos_rl_node_metadata(specs, num_gpus_per_node=8):
        """Compute node launch metadata for cosmos-rl multi-node jobs.

        Replicates the logic from cosmos_rl/tools/slurm/dispatch_job.py to
        compute how policy and rollout replicas are distributed across nodes.

        Returns:
            dict with keys:
                n_policy_nodes (int): Number of nodes for policy
                n_rollout_nodes (int): Number of nodes for rollout
                total_nodes (int): Total number of nodes
                policy_metadata (list[dict]): Per-node launch metadata for policy
                rollout_metadata (list[dict]): Per-node launch metadata for rollout
        """
        policy_par = specs.get("policy", {}).get("parallelism", {})
        rollout_par = specs.get("rollout", {}).get("parallelism", {})
        train_policy = specs.get("train", {}).get("train_policy", {})

        # Policy GPUs per replica
        tp = int(policy_par.get("tp_size", 1))
        pp = int(policy_par.get("pp_size", 1))
        cp = int(policy_par.get("cp_size", 1))
        dp_rep = int(policy_par.get("dp_replicate_size", 1))
        dp_shard = int(policy_par.get("dp_shard_size", 1))
        n_policy_replicas = int(policy_par.get("n_init_replicas", 1))

        min_gpus_policy = tp * pp * cp * dp_rep
        if dp_shard >= 1:
            min_gpus_policy *= dp_shard

        # Rollout GPUs per replica
        r_tp = int(rollout_par.get("tp_size", 1))
        r_pp = int(rollout_par.get("pp_size", 1))
        n_rollout_replicas = int(rollout_par.get("n_init_replicas", 1))
        min_gpus_rollout = r_tp * r_pp

        train_type = train_policy.get("type", "grpo")

        def compute_nodes_for_role(gpus_per_replica, n_replicas, role):
            """Compute per-node metadata for a role (policy or rollout)."""
            metadata = []
            rendezvous_port = 29345

            if gpus_per_replica >= num_gpus_per_node:
                # Replica spans multiple nodes
                nodes_per_replica = gpus_per_replica // num_gpus_per_node
                total_nodes_role = n_replicas * nodes_per_replica

                rendezvous_node = 0
                for i_node in range(total_nodes_role):
                    if i_node % nodes_per_replica == 0:
                        rendezvous_node = i_node

                    replica_meta = {
                        "colocation": [{
                            "nnode": nodes_per_replica,
                            "role": role,
                            "rendezvous_node": rendezvous_node,
                            "rendezvous_port": rendezvous_port,
                            "visible_gpus": list(range(num_gpus_per_node)),
                        }]
                    }
                    metadata.append(replica_meta)
            else:
                # Multiple replicas per node
                total_nodes_role = math.ceil(
                    n_replicas * gpus_per_replica / num_gpus_per_node
                )
                replica_counter = 0
                for i_node in range(total_nodes_role):
                    colocation = []
                    local_counter = 0
                    while replica_counter < n_replicas:
                        colocation.append({
                            "nnode": 1,
                            "role": role,
                            "rendezvous_node": i_node,
                            "rendezvous_port": rendezvous_port + replica_counter,
                            "visible_gpus": list(range(
                                local_counter * gpus_per_replica,
                                (local_counter + 1) * gpus_per_replica,
                            )),
                        })
                        replica_counter += 1
                        local_counter += 1
                        if replica_counter == n_replicas:
                            break
                        if local_counter * gpus_per_replica >= num_gpus_per_node:
                            break
                    metadata.append({"colocation": colocation})

            return metadata

        policy_metadata = compute_nodes_for_role(
            min_gpus_policy, n_policy_replicas, "policy"
        )
        n_policy_nodes = len(policy_metadata)

        if train_type == "sft":
            rollout_metadata = []
        else:
            rollout_metadata = compute_nodes_for_role(
                min_gpus_rollout, n_rollout_replicas, "rollout"
            )
        n_rollout_nodes = len(rollout_metadata)

        return {
            "n_policy_nodes": n_policy_nodes,
            "n_rollout_nodes": n_rollout_nodes,
            "total_nodes": n_policy_nodes + n_rollout_nodes,
            "policy_metadata": policy_metadata,
            "rollout_metadata": rollout_metadata,
            "min_gpus_policy": min_gpus_policy,
            "min_gpus_rollout": min_gpus_rollout,
            "n_policy_replicas": n_policy_replicas,
            "n_rollout_replicas": n_rollout_replicas,
        }

    def _build_slurm_script(
            self,
            job_name,
            num_nodes,
            num_gpus,
            cpus_per_task,
            time_hours,
            account,
            partition,
            log_dir,
            mail_user,
            image,
            container_mounts,
            use_timeout,
            timeout_hours,
            use_requeue,
            use_srun,
            command,
            docker_env_vars,
            exclusive,
            network=None,
    ):
        """Build the SLURM script content"""
        # Convert time to HH:MM:SS format
        time_str = f"{int(time_hours):02d}:{int((time_hours % 1) * 60):02d}:00"

        # Detect multi-node
        is_multi_node = num_nodes > 1

        # cosmos-rl uses 1 task per node, others use num_gpus tasks per node
        ntasks_per_node = 1 if network == "cosmos-rl" else num_gpus

        script_lines = [
            "#!/bin/bash -x",
            f"#SBATCH --nodes={num_nodes}",
            f"#SBATCH --gres=gpu:{num_gpus}",
            f"#SBATCH --ntasks-per-node={ntasks_per_node}",
        ]

        # Adjust ntasks for multi-node
        if is_multi_node:
            # Multi-node: one task per node
            script_lines.extend([
                f"#SBATCH --ntasks={num_nodes}",
            ])

        script_lines.extend([
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            "#SBATCH --wait-all-nodes=1",
        ])

        # cosmos-rl needs all available memory for model weights and KV cache
        if network == "cosmos-rl":
            script_lines.append("#SBATCH --mem=0")

        if exclusive:
            script_lines.append("#SBATCH --exclusive")

        script_lines.extend([
            f"#SBATCH --time={time_str}",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --account={account}",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --output={log_dir}/%x-%j/main.out",
            f"#SBATCH --error={log_dir}/%x-%j/main.err",
            "#SBATCH --open-mode=append"
        ])

        if mail_user:
            script_lines.extend([
                "#SBATCH --mail-type=all",
                f"#SBATCH --mail-user={mail_user}"
            ])

        if use_requeue:
            script_lines.append("#SBATCH --requeue")

        script_lines.extend([
            "",
            "export NCCL_DEBUG=INFO",
            "export LOGLEVEL=INFO",
            ""
        ])

        # Export docker_env_vars to be passed through to container
        if docker_env_vars:
            for key, value in docker_env_vars.items():
                # Escape single quotes in value for shell safety
                safe_value = str(value).replace("'", "'\"'\"'")
                script_lines.append(f"export {key}='{safe_value}'")
            script_lines.append("")

        # Add multi-node coordination env vars (mimics Kubernetes statefulset)
        if is_multi_node:
            script_lines.extend([
                f"export WORLD_SIZE={num_nodes}",
                f"export NUM_GPU_PER_NODE={num_gpus}",
                "export MASTER_PORT=29500",
                "",
                "# Compute MASTER_ADDR and node lists from SLURM allocation",
                "NODELIST=$(scontrol show hostname $SLURM_JOB_NODELIST)",
                "export MASTER_ADDR=$(echo $NODELIST | cut -d' ' -f1)",
                "",
                "# NODE_RANK from SLURM node ID (more semantically correct than SLURM_PROCID)",
                "export NODE_RANK=$SLURM_NODEID",
            ])

            # Cosmos-RL multi-node: compute controller host and per-role
            # node lists from SLURM allocation (SLURM_JOB_NODELIST is
            # available in the sbatch context).
            # The per-node role (policy vs rollout) is determined inside
            # the container by vlm_entrypoint.py using SLURM_NODEID,
            # NUM_POLICY_NODES, POLICY_NODES, and ROLLOUT_NODES.
            if network == "cosmos-rl":
                n_policy = docker_env_vars.get("NUM_POLICY_NODES", "1")
                n_total = docker_env_vars.get("TOTAL_NODES", str(num_nodes))
                controller_port = docker_env_vars.get("CONTROLLER_PORT", "8082")
                script_lines.extend([
                    "",
                    "# Cosmos-RL: compute controller host and per-role node lists",
                    "export CONTROLLER_NODE=$(echo $NODELIST | cut -d' ' -f1)",
                    f'export COSMOS_CONTROLLER_HOST="${{CONTROLLER_NODE}}:{controller_port}"',
                    f"export POLICY_NODES=$(echo $NODELIST | cut -d' ' -f1-{n_policy})",
                    f"NUM_ROLLOUT_NODES_VAL={docker_env_vars.get('NUM_ROLLOUT_NODES', '0')}",
                    "if [ $NUM_ROLLOUT_NODES_VAL -gt 0 ]; then",
                    f"    export ROLLOUT_NODES=$(echo $NODELIST | cut -d' ' -f$(({n_policy}+1))-$(({n_total})))",
                    "fi",
                    'echo "Cosmos-RL: POLICY_NODES=$POLICY_NODES, ROLLOUT_NODES=$ROLLOUT_NODES"',
                    'echo "Cosmos-RL: COSMOS_CONTROLLER_HOST=$COSMOS_CONTROLLER_HOST"',
                ])

        # CUDA environment override for jobs that need to force use of specific CUDA version
        # This is useful when:
        # - Container needs to use system CUDA instead of mounted filesystem paths (e.g., lustre miniconda)
        # - JIT compilation (flashinfer, torch extensions) needs specific CUDA libraries
        # Set CUDA_OVERRIDE_VERSION env var (e.g., "12.8", "12.2", "11.8") to enable this
        cuda_override = docker_env_vars.get("CUDA_OVERRIDE_VERSION", "").strip()
        if cuda_override:
            cuda_base_path = f"/usr/local/cuda-{cuda_override}"
            script_lines.extend([
                "# CUDA environment override",
                f"# Using CUDA version: {cuda_override}",
                f"export CUDA_HOME={cuda_base_path}",
                f"export CUDA_PATH={cuda_base_path}",
                f"export LD_LIBRARY_PATH=\"{cuda_base_path}/lib64:{cuda_base_path}/lib:$LD_LIBRARY_PATH\"",
                f"export LIBRARY_PATH=\"{cuda_base_path}/lib64:{cuda_base_path}/lib:${{LIBRARY_PATH:-}}\"",
                f"export CPATH=\"{cuda_base_path}/include:${{CPATH:-}}\"",
                f"echo \"CUDA environment configured: CUDA_HOME=$CUDA_HOME (version {cuda_override})\"",
                ""
            ])

        script_lines.extend([
            "# Job spec",
            f"OUTFILE=\"{log_dir}/%x-%j/%n.out\"",
            f"ERRFILE=\"{log_dir}/%x-%j/%n.err\"",
            ""
        ])

        # Build srun command - always needed for container execution
        # Handle different image formats according to Pyxis documentation:
        # 1. /path/to/image.sqsh - Local SQSH file (direct path)
        # 2. registry#image:tag - Docker image with Pyxis format
        # 3. docker://registry#image:tag - Docker protocol

        if image.startswith('/'):
            # Local SQSH file - use direct path as per documentation
            # Example: /lustre/fsw/.../image.sqsh
            container_image = image
        elif image.startswith(('docker://', 'dockerd://')):
            # Docker protocol - use as-is
            container_image = image
        elif '/' in image and not image.startswith('#'):
            # Convert "registry/image:tag" to "registry#image:tag" for Pyxis
            parts = image.split('/', 1)
            if '.' in parts[0] or ':' in parts[0]:  # First part looks like a registry
                container_image = f"{parts[0]}#{parts[1]}"
            else:
                # No registry specified, use as-is
                container_image = image
        else:
            container_image = image

        # Build container-env list from docker_env_vars keys
        # Add multi-node env vars to container
        container_env_keys = ",".join(docker_env_vars.keys()) if docker_env_vars else ""
        if is_multi_node:
            multi_node_env_keys = [
                "NODE_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                "NUM_GPU_PER_NODE"
            ]
            # Cosmos-RL: also forward runtime-computed vars and Slurm node info
            if network == "cosmos-rl":
                multi_node_env_keys.extend([
                    "COSMOS_CONTROLLER_HOST", "POLICY_NODES", "ROLLOUT_NODES",
                    "CONTROLLER_NODE", "SLURM_NODEID", "SLURMD_NODENAME",
                ])
            multi_node_env_str = ",".join(multi_node_env_keys)
            if container_env_keys:
                container_env_keys = multi_node_env_str + "," + container_env_keys
            else:
                container_env_keys = multi_node_env_str

        # Build base srun (all srun options MUST come before the "--")
        srun_base = (
            f"srun --container-image={container_image} "
            f"--container-mounts={container_mounts}"
        )

        if container_env_keys:
            srun_base += f" --container-env={container_env_keys}"

        if network == "cosmos-rl" and is_multi_node:
            # 3-parallel-srun pattern matching the native cosmos_rl_job_multi_node.sh.
            # Each role (controller, policy, rollout) gets its own srun targeting
            # specific nodes, with separate log files and independent failure handling.
            srun_cosmos = (
                f"srun --container-image={container_image} "
                f"--container-mounts={container_mounts} "
                f"--no-container-mount-home --export=ALL"
            )
            if container_env_keys:
                srun_cosmos += f" --container-env={container_env_keys}"

            n_policy = docker_env_vars.get("NUM_POLICY_NODES", "1")
            n_rollout = docker_env_vars.get("NUM_ROLLOUT_NODES", "0")

            script_lines.extend([
                "",
                "# Create per-role log directories",
                f"mkdir -p {log_dir}/${{SLURM_JOB_NAME}}-${{SLURM_JOB_ID}}/controller",
                f"mkdir -p {log_dir}/${{SLURM_JOB_NAME}}-${{SLURM_JOB_ID}}/policy",
                f"mkdir -p {log_dir}/${{SLURM_JOB_NAME}}-${{SLURM_JOB_ID}}/rollout",
                "",
                "# --- Controller (on first policy node, --overlap) ---",
                f"{srun_cosmos} \\",
                "    --overlap \\",
                "    --nodes=1 \\",
                "    --nodelist=$CONTROLLER_NODE \\",
                f"    -o {log_dir}/%x-%j/controller/%t.out \\",
                f"    -e {log_dir}/%x-%j/controller/%t.err \\",
                "    --export=ALL,COSMOS_NODE_ROLE=controller \\",
                f"    -- python {command} &",
                "pid_controller=$!",
                "",
                "# --- Policy workers ---",
                f"{srun_cosmos} \\",
                "    --overlap \\",
                f"    --nodes={n_policy} \\",
                "    --nodelist=\"$POLICY_NODES\" \\",
                f"    -o {log_dir}/%x-%j/policy/%t.out \\",
                f"    -e {log_dir}/%x-%j/policy/%t.err \\",
                "    --export=ALL,COSMOS_NODE_ROLE=policy \\",
                f"    -- python {command} &",
                "pid_policy=$!",
                "",
                f"if [ {n_rollout} -gt 0 ]; then",
                "    # --- Rollout workers ---",
                f"    {srun_cosmos} \\",
                f"        --nodes={n_rollout} \\",
                "        --nodelist=\"$ROLLOUT_NODES\" \\",
                f"        -o {log_dir}/%x-%j/rollout/%t.out \\",
                f"        -e {log_dir}/%x-%j/rollout/%t.err \\",
                "        --export=ALL,COSMOS_NODE_ROLE=rollout \\",
                f"        -- python {command} &",
                "    pid_rollout=$!",
                "fi",
                "",
                'echo "Waiting for controller, policy, and rollout. Job: $SLURM_JOB_ID"',
                "",
                "# Monitor all roles -- if any fails, kill the others and cancel job",
                "while true; do",
                "    kill -0 $pid_policy 2>/dev/null; pol_alive=$?",
                "",
                f"    if [ {n_rollout} -gt 0 ]; then",
                "        kill -0 $pid_rollout 2>/dev/null; roll_alive=$?",
                "    else",
                "        roll_alive=1; exit_code_rollout=0",
                "    fi",
                "",
                "    kill -0 $pid_controller 2>/dev/null; crl_alive=$?",
                "",
                "    # All done?",
                "    if [ $pol_alive -ne 0 ] && [ $roll_alive -ne 0 ] && [ $crl_alive -ne 0 ]; then",
                "        wait $pid_policy; exit_code_policy=$?",
                f"        if [ {n_rollout} -gt 0 ]; then wait $pid_rollout; exit_code_rollout=$?; fi",
                "        wait $pid_controller; exit_code_controller=$?",
                "        if [ $exit_code_policy -ne 0 ] || [ $exit_code_rollout -ne 0 ]"
                " || [ $exit_code_controller -ne 0 ]; then",
                '            echo "One or more roles failed'
                ' (policy=$exit_code_policy rollout=${exit_code_rollout:-0}'
                ' controller=$exit_code_controller)"',
                "            scancel $SLURM_JOB_ID",
                "            exit 1",
                "        else",
                '            echo "All roles succeeded"',
                "            exit 0",
                "        fi",
                "    fi",
                "",
                "    # Policy failed early?",
                "    if [ $pol_alive -ne 0 ]; then",
                "        wait $pid_policy; ec=$?",
                "        if [ $ec -ne 0 ]; then",
                '            echo "Policy failed ($ec). Killing other roles."',
                f"            if [ {n_rollout} -gt 0 ]; then kill $pid_rollout 2>/dev/null || true; fi",
                "            kill $pid_controller 2>/dev/null || true",
                "            scancel $SLURM_JOB_ID; exit $ec",
                "        fi",
                "    fi",
                "",
                "    # Rollout failed early?",
                f"    if [ {n_rollout} -gt 0 ]; then",
                "        if [ $roll_alive -ne 0 ]; then",
                "            wait $pid_rollout; ec=$?",
                "            if [ $ec -ne 0 ]; then",
                '                echo "Rollout failed ($ec). Killing other roles."',
                "                kill $pid_policy 2>/dev/null || true",
                "                kill $pid_controller 2>/dev/null || true",
                "                scancel $SLURM_JOB_ID; exit $ec",
                "            fi",
                "        fi",
                "    fi",
                "",
                "    # Controller failed early?",
                "    if [ $crl_alive -ne 0 ]; then",
                "        wait $pid_controller; ec=$?",
                "        if [ $ec -ne 0 ]; then",
                '            echo "Controller failed ($ec). Killing other roles."',
                "            kill $pid_policy 2>/dev/null || true",
                f"            if [ {n_rollout} -gt 0 ]; then kill $pid_rollout 2>/dev/null || true; fi",
                "            scancel $SLURM_JOB_ID; exit $ec",
                "        fi",
                "    fi",
                "",
                "    sleep 1",
                "done",
            ])
        else:
            # Non-cosmos-rl or single-node: standard single srun
            # Add output redirection if use_srun is enabled (for advanced features)
            if use_srun:
                srun_base += " -o $OUTFILE -e $ERRFILE --open-mode=append"

            # Prevent home directory conflicts for cosmos-rl single-node
            if network == "cosmos-rl":
                srun_base += " --no-container-mount-home --export=ALL"

            # Build command with timeout if enabled (for auto-requeue support)
            if use_timeout:
                # Convert to minutes for better precision (e.g., 3.8h = 228m)
                timeout_mins = int(timeout_hours * 60) if timeout_hours else int((time_hours - 0.2) * 60)
                script_lines.append(f"timeout {timeout_mins}m {srun_base} -- python {command}")
            else:
                script_lines.append(f"{srun_base} -- python {command}")

            if use_timeout and use_requeue:
                script_lines.extend([
                    "",
                    "# Launch self again",
                    "if [[ $? == 124 ]]; then",
                    "    scontrol requeue $SLURM_JOB_ID",
                    "fi"
                ])

        return "\n".join(script_lines)

    def get_slurm_job_name(self, job_id):
        """Get the name of a job on Slurm"""
        return f"tao-job-{job_id}"

    def get_slurm_job_status(self, slurm_job_id):
        """Get the status of a SLURM job using squeue and sacct.

        Args:
            slurm_job_id (str or int): The SLURM job ID (e.g., "20148330")

        Returns:
            str: Status string that can be:
                - "PENDING": Job is waiting for resources
                - "RUNNING": Job is executing
                - "COMPLETING": Job is finishing
                - "COMPLETED": Job finished successfully
                - "FAILED": Job failed
                - "CANCELLED": Job was cancelled
                - "TIMEOUT": Job hit time limit
                - "NODE_FAIL": Node failed
                - "NOT_FOUND": Job not found
                - "ERROR": Error checking status

        SLURM Status Codes (from squeue/sacct):
            - PENDING (PD): Job is waiting for resource allocation
            - RUNNING (R): Job currently has an allocation
            - SUSPENDED (S): Job has an allocation but execution suspended
            - COMPLETING (CG): Job is in the process of completing
            - COMPLETED (CD): Job has terminated all processes with exit code 0
            - FAILED (F): Job terminated with non-zero exit code
            - TIMEOUT (TO): Job terminated upon reaching time limit
            - CANCELLED (CA): Job was explicitly cancelled
            - NODE_FAIL (NF): Job terminated due to node failure
        """
        if not slurm_job_id:
            return "ERROR"

        # First try squeue for active jobs (pending/running)
        remote_command = f"squeue --job={slurm_job_id} --format='%T' --noheader"
        ssh_command = self._build_ssh_command(remote_command)

        try:
            result = subprocess.run(
                ssh_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=TIMEOUT_SECONDS,  # SLURM clusters can be slow
                check=False
            )

            if result.returncode == 0 and result.stdout.strip():
                status = result.stdout.strip()
                self.logger.info(f"SLURM job {slurm_job_id} active status: {status}")
                return status

            # If not in active queue, check sacct for completed/failed jobs
            remote_command = f"sacct --job={slurm_job_id} --format=State --noheader --parsable2"
            ssh_command = self._build_ssh_command(remote_command)

            result = subprocess.run(
                ssh_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=TIMEOUT_SECONDS,  # SLURM clusters can be slow
                check=False
            )

            if result.returncode == 0 and result.stdout.strip():
                # sacct returns multiple lines (parent job + steps), take first non-empty
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        status = line.strip()
                        self.logger.info(f"SLURM job {slurm_job_id} historical status: {status}")
                        return status

            self.logger.warning(f"SLURM job {slurm_job_id} not found in squeue or sacct")
            return "NOT_FOUND"

        except subprocess.TimeoutExpired:
            self.logger.error(f"SSH timeout ({TIMEOUT_SECONDS}s) while checking SLURM job {slurm_job_id} status")
            self.logger.error("This is an SSH/network issue, not a SLURM job timeout")
            self.logger.error("The SLURM cluster may be slow or unresponsive")
            return "ERROR"  # Changed from "TIMEOUT" to distinguish from actual SLURM job timeouts
        except Exception as e:
            self.logger.error(traceback.format_exc())
            self.logger.error(f"Error checking SLURM job {slurm_job_id} status: {e}")
            return "ERROR"

    def map_slurm_status_to_tao_status(self, slurm_status, stage="main"):
        """Map SLURM job status to TAO API status with stage context.

        Args:
            slurm_status (str): SLURM status code (e.g., "PENDING", "RUNNING", "COMPLETED")
            stage (str): Current stage - "conversion" for SQSH generation, "main" for training job

        Returns:
            str: TAO API status (e.g., "Pending", "Running", "Done", "Error")
        """
        # Map SLURM status to TAO status based on stage
        if slurm_status in ("PENDING", "CONFIGURING", "RESIZING"):
            if stage == "conversion":
                return "Pending"  # "Preparing container image..."
            return "Pending"

        if slurm_status in ("RUNNING", "COMPLETING"):
            if stage == "conversion":
                return "Running"  # "Converting image to SQSH format..."
            return "Running"

        if slurm_status in ("COMPLETED",):
            if stage == "conversion":
                # Conversion complete, but main job not started yet
                return "Pending"  # Will transition to main job
            return "Done"

        if slurm_status in ("FAILED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "NODE_FAIL"):
            return "Error"

        if slurm_status in ("CANCELLED", "PREEMPTED", "REVOKED"):
            return "Canceled"

        if slurm_status in ("TIMEOUT",):
            return "Error"

        if slurm_status in ("SUSPENDED", "STOPPED"):
            return "Paused"

        # Unknown status
        self.logger.warning(f"Unknown SLURM status '{slurm_status}' for stage '{stage}'")
        return "Pending"

    def get_slurm_job_id_from_output(self, sbatch_output):
        """Extract SLURM job ID from sbatch output.

        Args:
            sbatch_output (str): Output from sbatch command (e.g., "Submitted batch job 20148330")

        Returns:
            str: SLURM job ID or None if not found
        """
        match = re.search(r'Submitted batch job (\d+)', sbatch_output)
        if match:
            return match.group(1)
        return None

    def _is_retriable_failure(self, slurm_status, job_id, log_dir=None):
        """Check if a job failure is retriable (e.g., bad node, driver issues).

        This method determines if a SLURM job failure was due to infrastructure
        issues (bad nodes, outdated drivers) rather than user code errors.

        Args:
            slurm_status (str): SLURM status code (FAILED, NODE_FAIL, etc.)
            job_id (str): TAO job ID for log retrieval
            log_dir (str, optional): Log directory path

        Returns:
            bool: True if the failure is retriable, False otherwise
        """
        self.logger.info("=" * 80)
        self.logger.info("RETRY CHECK: Determining if failure is retriable")
        self.logger.info(f"  SLURM Status: {slurm_status}")
        self.logger.info(f"  Job ID: {job_id}")
        self.logger.info(f"  Log Dir: {log_dir}")
        self.logger.info("=" * 80)

        # These SLURM statuses are always retriable (infrastructure failures)
        always_retriable_statuses = {"NODE_FAIL", "BOOT_FAIL"}
        if slurm_status in always_retriable_statuses:
            self.logger.info(f"[RETRY CHECK] SLURM status '{slurm_status}' is in always-retriable list: "
                             f"{always_retriable_statuses}")
            self.logger.info("[RETRY CHECK] RESULT: RETRIABLE (infrastructure failure status)")
            return True

        # For FAILED status, check the logs for known retriable error patterns
        if slurm_status == "FAILED":
            self.logger.info("[RETRY CHECK] Status is FAILED - checking job logs for retriable error patterns...")
            self.logger.info(f"[RETRY CHECK] Will search for {len(RETRIABLE_ERROR_PATTERNS)} known error patterns")
            try:
                logs = self.get_job_logs(job_id, tail_lines=500, log_dir=log_dir)
                if logs:
                    log_preview = logs[:500] if len(logs) > 500 else logs
                    self.logger.info(f"[RETRY CHECK] Retrieved {len(logs)} characters of logs")
                    self.logger.debug(f"[RETRY CHECK] Log preview (first 500 chars): {log_preview}")

                    for pattern in RETRIABLE_ERROR_PATTERNS:
                        match = re.search(pattern, logs, re.IGNORECASE)
                        if match:
                            self.logger.info(f"[RETRY CHECK] ✓ FOUND retriable error pattern: '{pattern}'")
                            self.logger.info(f"[RETRY CHECK] ✓ Matched text: '{match.group()}'")
                            self.logger.info("[RETRY CHECK] RESULT: RETRIABLE (matched error pattern in logs)")
                            return True
                    self.logger.info("[RETRY CHECK] ✗ No retriable error patterns found in logs")
                    self.logger.info("[RETRY CHECK] Checked patterns:")
                    for p in RETRIABLE_ERROR_PATTERNS:
                        self.logger.info(f"  - {p}")
                    self.logger.info("[RETRY CHECK] RESULT: NOT RETRIABLE (no matching patterns)")
                else:
                    # If we can't get logs, be conservative and retry
                    self.logger.warning("[RETRY CHECK] Could not retrieve logs (empty or None)")
                    self.logger.info("[RETRY CHECK] RESULT: RETRIABLE (being conservative - assuming infra failure)")
                    return True
            except Exception as e:
                self.logger.warning(f"[RETRY CHECK] Error checking logs for retriable patterns: {e}")
                self.logger.info("[RETRY CHECK] RESULT: RETRIABLE (being conservative due to log check error)")
                # If we can't check logs, be conservative and retry
                return True

        # TIMEOUT, DEADLINE, OUT_OF_MEMORY are typically not retriable
        # (indicates job needs more resources or is stuck)
        self.logger.info(f"[RETRY CHECK] Status '{slurm_status}' is not in retriable categories")
        self.logger.info("[RETRY CHECK] RESULT: NOT RETRIABLE")
        return False

    def _get_retry_info(self, job_id, is_automl_experiment=False, brain_job_id=None, experiment_number=None):
        """Get retry information for a job.

        Args:
            job_id (str): TAO job ID
            is_automl_experiment (bool): Whether this is an AutoML experiment
            brain_job_id (str): Brain job ID for AutoML experiments
            experiment_number (int): Experiment number for AutoML experiments

        Returns:
            dict: Retry info with keys: retry_count, failed_slurm_job_ids, job_params
        """
        self.logger.info("[GET RETRY INFO] Fetching retry information from metadata")
        self.logger.info(f"[GET RETRY INFO]   Job ID: {job_id}")
        self.logger.info(f"[GET RETRY INFO]   Is AutoML Experiment: {is_automl_experiment}")
        self.logger.info(f"[GET RETRY INFO]   Brain Job ID: {brain_job_id}")
        self.logger.info(f"[GET RETRY INFO]   Experiment Number: {experiment_number}")

        default_retry_info = {
            "retry_count": 0,
            "failed_slurm_job_ids": [],
            "job_params": {}
        }

        try:
            if is_automl_experiment and brain_job_id:
                self.logger.info("[GET RETRY INFO] Looking up AutoML controller info...")
                controller_info = get_automl_controller_info(brain_job_id)
                if controller_info and experiment_number is not None and len(controller_info) > experiment_number:
                    experiment_info = controller_info[experiment_number]
                    backend_details = experiment_info.get("backend_details") or {}
                    slurm_metadata = backend_details.get("slurm_metadata") or {}
                    retry_info = {
                        "retry_count": slurm_metadata.get("retry_count", 0),
                        "failed_slurm_job_ids": slurm_metadata.get("failed_slurm_job_ids", []),
                        "job_params": slurm_metadata.get("job_params", {})
                    }
                    self.logger.info("[GET RETRY INFO] Found AutoML experiment retry info:")
                    self.logger.info(f"[GET RETRY INFO]   Retry Count: {retry_info['retry_count']}")
                    self.logger.info(f"[GET RETRY INFO]   Failed SLURM Job IDs: {retry_info['failed_slurm_job_ids']}")
                    return retry_info
                self.logger.info("[GET RETRY INFO] No controller info found or experiment number out of range")
            else:
                self.logger.info("[GET RETRY INFO] Looking up regular job metadata...")
                job_metadata = get_handler_job_metadata(job_id)
                if job_metadata:
                    backend_details = job_metadata.get("backend_details") or {}
                    slurm_metadata = backend_details.get("slurm_metadata") or {}
                    retry_info = {
                        "retry_count": slurm_metadata.get("retry_count", 0),
                        "failed_slurm_job_ids": slurm_metadata.get("failed_slurm_job_ids", []),
                        "job_params": slurm_metadata.get("job_params", {})
                    }
                    self.logger.info("[GET RETRY INFO] Found job retry info:")
                    self.logger.info(f"[GET RETRY INFO]   Retry Count: {retry_info['retry_count']}")
                    self.logger.info(f"[GET RETRY INFO]   Failed SLURM Job IDs: {retry_info['failed_slurm_job_ids']}")
                    return retry_info

                self.logger.info("[GET RETRY INFO] No job metadata found")
        except Exception as e:
            self.logger.warning(f"[GET RETRY INFO] Error getting retry info for job {job_id}: {e}")
            self.logger.warning(traceback.format_exc())

        self.logger.info(f"[GET RETRY INFO] Returning default retry info: {default_retry_info}")
        return default_retry_info

    def _update_retry_info(self, job_id, retry_count, failed_slurm_job_id, new_slurm_job_id,
                           is_automl_experiment=False, brain_job_id=None, experiment_number=None,
                           job_dir=None):
        """Update retry information for a job.

        Args:
            job_id (str): TAO job ID
            retry_count (int): New retry count
            failed_slurm_job_id (str): The SLURM job ID that failed
            new_slurm_job_id (str): The new SLURM job ID after resubmission
            is_automl_experiment (bool): Whether this is an AutoML experiment
            brain_job_id (str): Brain job ID for AutoML experiments
            experiment_number (int): Experiment number for AutoML experiments
            job_dir (str): Job directory path

        Returns:
            bool: True if update was successful
        """
        self.logger.info("[UPDATE RETRY INFO] Saving retry information to metadata")
        self.logger.info(f"[UPDATE RETRY INFO]   Job ID: {job_id}")
        self.logger.info(f"[UPDATE RETRY INFO]   New Retry Count: {retry_count}")
        self.logger.info(f"[UPDATE RETRY INFO]   Failed SLURM Job ID: {failed_slurm_job_id}")
        self.logger.info(f"[UPDATE RETRY INFO]   New SLURM Job ID: {new_slurm_job_id}")
        self.logger.info(f"[UPDATE RETRY INFO]   Is AutoML Experiment: {is_automl_experiment}")
        self.logger.info(f"[UPDATE RETRY INFO]   Brain Job ID: {brain_job_id}")
        self.logger.info(f"[UPDATE RETRY INFO]   Experiment Number: {experiment_number}")
        self.logger.info(f"[UPDATE RETRY INFO]   Job Dir: {job_dir}")

        try:
            if is_automl_experiment and brain_job_id:
                self.logger.info("[UPDATE RETRY INFO] Updating AutoML controller info...")
                controller_info = get_automl_controller_info(brain_job_id)
                if controller_info and experiment_number is not None and len(controller_info) > experiment_number:
                    rec = controller_info[experiment_number]
                    if not rec.get("backend_details"):
                        rec["backend_details"] = {'backend_type': 'slurm'}
                        self.logger.info("[UPDATE RETRY INFO] Created new backend_details structure")
                    if not rec["backend_details"].get("slurm_metadata"):
                        rec["backend_details"]["slurm_metadata"] = {}
                        self.logger.info("[UPDATE RETRY INFO] Created new slurm_metadata structure")

                    slurm_metadata = rec["backend_details"]["slurm_metadata"]

                    # Update retry tracking fields
                    old_retry_count = slurm_metadata.get("retry_count", 0)
                    old_failed_ids = slurm_metadata.get("failed_slurm_job_ids", [])

                    slurm_metadata["retry_count"] = retry_count
                    if "failed_slurm_job_ids" not in slurm_metadata:
                        slurm_metadata["failed_slurm_job_ids"] = []
                    if failed_slurm_job_id and failed_slurm_job_id not in slurm_metadata["failed_slurm_job_ids"]:
                        slurm_metadata["failed_slurm_job_ids"].append(failed_slurm_job_id)
                    slurm_metadata["slurm_job_id"] = new_slurm_job_id
                    if job_dir:
                        slurm_metadata["job_dir"] = job_dir

                    save_automl_controller_info(brain_job_id, controller_info)
                    self.logger.info(
                        f"[UPDATE RETRY INFO] ✓ Successfully updated AutoML experiment {experiment_number} "
                        f"in brain {brain_job_id}"
                    )
                    self.logger.info(f"[UPDATE RETRY INFO]   Retry count: {old_retry_count} -> {retry_count}")
                    self.logger.info(f"[UPDATE RETRY INFO]   Failed IDs: {old_failed_ids} -> "
                                     f"{slurm_metadata['failed_slurm_job_ids']}")
                    self.logger.info(f"[UPDATE RETRY INFO]   New SLURM Job ID: {new_slurm_job_id}")
                    return True
                self.logger.warning("[UPDATE RETRY INFO] Could not find controller info or experiment")
            else:
                self.logger.info("[UPDATE RETRY INFO] Updating regular job metadata...")
                job_metadata = get_handler_job_metadata(job_id)
                if job_metadata:
                    if not job_metadata.get("backend_details"):
                        job_metadata["backend_details"] = {'backend_type': 'slurm'}
                        self.logger.info("[UPDATE RETRY INFO] Created new backend_details structure")
                    if not job_metadata["backend_details"].get("slurm_metadata"):
                        job_metadata["backend_details"]["slurm_metadata"] = {}
                        self.logger.info("[UPDATE RETRY INFO] Created new slurm_metadata structure")

                    slurm_metadata = job_metadata["backend_details"]["slurm_metadata"]

                    # Update retry tracking fields
                    old_retry_count = slurm_metadata.get("retry_count", 0)
                    old_failed_ids = slurm_metadata.get("failed_slurm_job_ids", [])

                    slurm_metadata["retry_count"] = retry_count
                    if "failed_slurm_job_ids" not in slurm_metadata:
                        slurm_metadata["failed_slurm_job_ids"] = []
                    if failed_slurm_job_id and failed_slurm_job_id not in slurm_metadata["failed_slurm_job_ids"]:
                        slurm_metadata["failed_slurm_job_ids"].append(failed_slurm_job_id)
                    slurm_metadata["slurm_job_id"] = new_slurm_job_id
                    if job_dir:
                        slurm_metadata["job_dir"] = job_dir

                    write_job_metadata(job_id, job_metadata)
                    self.logger.info(f"[UPDATE RETRY INFO] ✓ Successfully updated job {job_id}")
                    self.logger.info(f"[UPDATE RETRY INFO]   Retry count: {old_retry_count} -> {retry_count}")
                    self.logger.info(f"[UPDATE RETRY INFO]   Failed IDs: {old_failed_ids} -> "
                                     f"{slurm_metadata['failed_slurm_job_ids']}")
                    self.logger.info(f"[UPDATE RETRY INFO]   New SLURM Job ID: {new_slurm_job_id}")
                    return True

                self.logger.warning(f"[UPDATE RETRY INFO] No job metadata found for {job_id}")
        except Exception as e:
            self.logger.error(f"Error updating retry info for job {job_id}: {e}")
            self.logger.error(traceback.format_exc())

        return False

    def _resubmit_failed_job(self, job_id, failed_slurm_job_id, retry_count,
                             is_automl_experiment=False, brain_job_id=None, experiment_number=None):
        """Resubmit a failed SLURM job.

        This method reads the stored job parameters and resubmits the job
        using the same configuration. The job will be scheduled on potentially
        a different node.

        Args:
            job_id (str): TAO job ID
            failed_slurm_job_id (str): The SLURM job ID that failed
            retry_count (int): Current retry count (will be incremented)
            is_automl_experiment (bool): Whether this is an AutoML experiment
            brain_job_id (str): Brain job ID for AutoML experiments
            experiment_number (int): Experiment number for AutoML experiments

        Returns:
            tuple: (success: bool, new_slurm_job_id: str or None, error_message: str or None)
        """
        self.logger.info("=" * 80)
        self.logger.info("SLURM JOB RESUBMISSION STARTING")
        self.logger.info("=" * 80)
        self.logger.info(f"[RESUBMIT] Attempt: {retry_count + 1}/{MAX_JOB_RETRIES}")
        self.logger.info(f"[RESUBMIT] TAO Job ID: {job_id}")
        self.logger.info(f"[RESUBMIT] Failed SLURM Job ID: {failed_slurm_job_id}")
        self.logger.info(f"[RESUBMIT] Is AutoML Experiment: {is_automl_experiment}")
        self.logger.info(f"[RESUBMIT] Brain Job ID: {brain_job_id}")
        self.logger.info(f"[RESUBMIT] Experiment Number: {experiment_number}")
        self.logger.info("=" * 80)

        try:
            # Get stored job directory from metadata
            self.logger.info("[RESUBMIT] Step 1: Looking up job directory from metadata...")
            job_dir = self._get_stored_job_dir(job_id)
            self.logger.info(f"[RESUBMIT]   job_dir from _get_stored_job_dir: {job_dir}")

            if not job_dir:
                # Try getting from AutoML controller info
                if is_automl_experiment and brain_job_id:
                    self.logger.info("[RESUBMIT]   Trying to get job_dir from AutoML controller info...")
                    controller_info = get_automl_controller_info(brain_job_id)
                    if controller_info and experiment_number is not None and len(controller_info) > experiment_number:
                        rec = controller_info[experiment_number]
                        backend_details = rec.get("backend_details") or {}
                        slurm_metadata = backend_details.get("slurm_metadata") or {}
                        job_dir = slurm_metadata.get("job_dir")
                        self.logger.info(f"[RESUBMIT]   job_dir from controller info: {job_dir}")

            if not job_dir:
                # Fallback to default path
                job_dir = f"/lustre/fsw/portfolios/edgeai/users/{self.login_user}"
                self.logger.warning(f"[RESUBMIT] ⚠ No stored job_dir found, using fallback: {job_dir}")
            else:
                self.logger.info(f"[RESUBMIT] ✓ Using job_dir: {job_dir}")

            # Read the stored sbatch script for this job
            remote_script = f"{job_dir}/job_{job_id}.sbatch"
            self.logger.info(f"[RESUBMIT] Step 2: Checking for sbatch script at: {remote_script}")

            # Check if the script exists
            check_command = f"test -f {shlex.quote(remote_script)} && echo 'exists' || echo 'not_found'"
            self.logger.info(f"[RESUBMIT]   Check command: {check_command}")
            ssh_command = self._build_ssh_command(check_command)

            result = subprocess.run(
                ssh_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=False
            )
            self.logger.info(f"[RESUBMIT]   Script check result: {result.stdout.strip()}")

            if result.stdout.strip() != 'exists':
                error_msg = f"Sbatch script not found at {remote_script}"
                self.logger.error(f"[RESUBMIT] ✗ {error_msg}")
                self.logger.error("[RESUBMIT] Cannot resubmit without the original sbatch script")
                return False, None, error_msg

            self.logger.info("[RESUBMIT] ✓ Sbatch script exists")

            # Resubmit the job using the existing script
            self.logger.info("[RESUBMIT] Step 3: Submitting job to SLURM...")
            remote_command = f"sbatch --export=ALL {shlex.quote(remote_script)}"
            self.logger.info(f"[RESUBMIT]   Command: {remote_command}")
            ssh_command = self._build_ssh_command(remote_command)

            result = subprocess.run(
                ssh_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=TIMEOUT_SECONDS
            )

            sbatch_output = result.stdout.strip()
            self.logger.info(f"[RESUBMIT] ✓ Sbatch output: {sbatch_output}")
            if result.stderr:
                self.logger.info(f"[RESUBMIT]   Stderr: {result.stderr.strip()}")

            # Extract new SLURM job ID
            self.logger.info("[RESUBMIT] Step 4: Extracting new SLURM job ID...")
            new_slurm_job_id = self.get_slurm_job_id_from_output(sbatch_output)
            if not new_slurm_job_id:
                error_msg = f"Failed to extract SLURM job ID from output: {sbatch_output}"
                self.logger.error(f"[RESUBMIT] ✗ {error_msg}")
                return False, None, error_msg

            self.logger.info(f"[RESUBMIT] ✓ New SLURM job ID: {new_slurm_job_id}")

            # Update retry info in metadata
            self.logger.info("[RESUBMIT] Step 5: Updating retry info in metadata...")
            new_retry_count = retry_count + 1
            self._update_retry_info(
                job_id=job_id,
                retry_count=new_retry_count,
                failed_slurm_job_id=failed_slurm_job_id,
                new_slurm_job_id=new_slurm_job_id,
                is_automl_experiment=is_automl_experiment,
                brain_job_id=brain_job_id,
                experiment_number=experiment_number,
                job_dir=job_dir
            )

            # Update job message
            self.logger.info("[RESUBMIT] Step 6: Updating job message...")
            try:
                lookup_id = brain_job_id if brain_job_id else job_id
                handler_id = get_handler_id(lookup_id)
                handler_metadata = get_handler_metadata(lookup_id, kind=None)
                handler_kind = get_handler_kind(handler_metadata)

                message = (
                    f"Job resubmitted (attempt {new_retry_count}/{MAX_JOB_RETRIES}) after infrastructure failure. "
                    f"New SLURM Job ID: {new_slurm_job_id}. Waiting for resources..."
                )
                self.logger.info(f"[RESUBMIT]   Message: {message}")

                update_params = {
                    "handler_id": handler_id,
                    "job_id": lookup_id,
                    "kind": handler_kind,
                    "message": message
                }
                if is_automl_experiment:
                    update_params["automl_expt_job_id"] = job_id
                    update_params["update_automl_expt"] = True

                update_job_message(**update_params)
                self.logger.info("[RESUBMIT] ✓ Job message updated")
            except Exception as e:
                self.logger.warning(f"[RESUBMIT] ⚠ Failed to update job message: {e}")

            self.logger.info("=" * 80)
            self.logger.info("JOB RESUBMISSION SUCCESSFUL")
            self.logger.info(f"  Previous SLURM Job ID: {failed_slurm_job_id} (FAILED)")
            self.logger.info(f"  New SLURM Job ID: {new_slurm_job_id} (PENDING)")
            self.logger.info(f"  Retry Attempt: {new_retry_count}/{MAX_JOB_RETRIES}")
            self.logger.info("  Job will be scheduled on a (hopefully different) node")
            self.logger.info("=" * 80)

            return True, new_slurm_job_id, None

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to resubmit SLURM job: {e.stderr.strip() if e.stderr else str(e)}"
            self.logger.error(error_msg)
            self.logger.info("=" * 80)
            return False, None, error_msg
        except subprocess.TimeoutExpired:
            error_msg = "SSH timeout while resubmitting job"
            self.logger.error(error_msg)
            self.logger.info("=" * 80)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Exception during job resubmission: {e}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.logger.info("=" * 80)
            return False, None, error_msg

    def get_tao_job_status(self, job_id):
        """Get TAO-mapped job status for a SLURM job with descriptive messages.

        This method checks SLURM job status and maps it to TAO status,
        updating job messages with detailed progress information.

        Args:
            job_id (str): TAO job ID

        Returns:
            str: TAO status ("Pending", "Running", "Done", "Error", "Canceled")
        """
        try:
            # Try to get SLURM job ID from job metadata (for regular jobs and brain jobs)
            job_metadata = get_handler_job_metadata(job_id)
            slurm_job_id = None
            brain_job_id = None

            if job_metadata:
                backend_details = job_metadata.get("backend_details") or {}
                slurm_metadata = backend_details.get("slurm_metadata") or {}
                slurm_job_id = slurm_metadata.get("slurm_job_id")

            # If not found, check if this is an AutoML experiment
            if not slurm_job_id:
                self.logger.info(f"No SLURM job ID in job metadata for {job_id}, checking if AutoML experiment...")

                # Search through brain jobs to find this experiment
                mongo_jobs = MongoHandler("tao", "jobs")
                all_jobs = mongo_jobs.find({'status': {'$exists': True}})

                for job_data in all_jobs:
                    potential_brain_id = job_data.get('id')
                    if potential_brain_id:
                        controller_info = get_automl_controller_info(potential_brain_id)
                        if isinstance(controller_info, list):
                            for idx, rec in enumerate(controller_info):
                                if rec.get("job_id") == job_id:
                                    # Found it! Get SLURM job ID from controller info
                                    backend_details = rec.get("backend_details", {})
                                    slurm_metadata = backend_details.get("slurm_metadata", {})
                                    slurm_job_id = slurm_metadata.get("slurm_job_id")
                                    brain_job_id = potential_brain_id
                                    self.logger.info(
                                        f"Found AutoML experiment {idx} in brain {brain_job_id}, "
                                        f"SLURM job ID: {slurm_job_id}"
                                    )
                                    break
                        if brain_job_id:
                            break

            if not slurm_job_id:
                self.logger.warning(
                    f"No SLURM job ID found for TAO job {job_id} - "
                    f"submission may be in progress or not yet stored"
                )
                return "Pending"

            # Check SLURM job status
            slurm_status = self.get_slurm_job_status(slurm_job_id)
            self.logger.info(f"SLURM job {slurm_job_id} (TAO job {job_id}) status: {slurm_status}")

            # Get handler info for status updates
            # For AutoML experiments, use brain job ID to get handler metadata
            lookup_id_for_handler = brain_job_id if brain_job_id else job_id
            handler_id = get_handler_id(lookup_id_for_handler)
            handler_metadata = get_handler_metadata(lookup_id_for_handler)
            handler_kind = get_handler_kind(handler_metadata)

            # Map SLURM status to TAO status and update descriptive message
            # For AutoML experiments, we need to pass brain_job_id and experiment job_id separately
            update_params = {
                "handler_id": handler_id,
                "job_id": lookup_id_for_handler,
                "kind": handler_kind
            }
            if brain_job_id:
                # AutoML experiment: add experiment-specific params
                update_params["automl_expt_job_id"] = job_id
                update_params["update_automl_expt"] = True

            if slurm_status == "PENDING":
                message = f"SLURM job is pending (Job ID: {slurm_job_id}). Waiting for resources..."
                self.update_job_status(job_id, "RUNNING", message)
                return "Pending"

            if slurm_status == "RUNNING":
                self.logger.info(
                    f"SLURM job {slurm_job_id} is RUNNING - StatusParser will show progress from DNN status"
                )
                return "Running"

            if slurm_status == "COMPLETING":
                self.logger.info(
                    f"SLURM job {slurm_job_id} is COMPLETING - "
                    "StatusParser will show final status from DNN status"
                )
                return "Running"

            if slurm_status == "COMPLETED":
                # Job completed - check status.json for actual result
                # Return None to let the caller check status.json
                return None

            if slurm_status in ("FAILED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "NODE_FAIL"):
                self.logger.info("=" * 80)
                self.logger.info("SLURM JOB FAILURE DETECTED - EVALUATING RETRY")
                self.logger.info("=" * 80)
                self.logger.info(f"[RETRY EVAL] SLURM Job ID: {slurm_job_id}")
                self.logger.info(f"[RETRY EVAL] TAO Job ID: {job_id}")
                self.logger.info(f"[RETRY EVAL] SLURM Status: {slurm_status}")
                self.logger.info(f"[RETRY EVAL] Max Retries Configured: {MAX_JOB_RETRIES}")

                # Check if this is a retriable failure (bad node, driver issues, etc.)
                is_automl_experiment = brain_job_id is not None
                experiment_number = None
                self.logger.info(f"[RETRY EVAL] Is AutoML Experiment: {is_automl_experiment}")
                self.logger.info(f"[RETRY EVAL] Brain Job ID: {brain_job_id}")

                # Get experiment number for AutoML experiments
                if is_automl_experiment:
                    self.logger.info("[RETRY EVAL] Looking up experiment number from controller info...")
                    controller_info = get_automl_controller_info(brain_job_id)
                    if isinstance(controller_info, list):
                        for idx, rec in enumerate(controller_info):
                            if rec.get("job_id") == job_id:
                                experiment_number = idx
                                self.logger.info(f"[RETRY EVAL] Found experiment number: {experiment_number}")
                                break
                        if experiment_number is None:
                            self.logger.warning("[RETRY EVAL] Could not find experiment number in controller info")

                # Get current retry info
                self.logger.info("[RETRY EVAL] Fetching current retry information...")
                retry_info = self._get_retry_info(
                    job_id,
                    is_automl_experiment=is_automl_experiment,
                    brain_job_id=brain_job_id,
                    experiment_number=experiment_number
                )
                current_retry_count = retry_info.get("retry_count", 0)
                failed_job_ids_so_far = retry_info.get("failed_slurm_job_ids", [])

                self.logger.info("[RETRY EVAL] Current retry status:")
                self.logger.info(f"[RETRY EVAL]   Retry count: {current_retry_count}/{MAX_JOB_RETRIES}")
                self.logger.info(f"[RETRY EVAL]   Previously failed SLURM Job IDs: {failed_job_ids_so_far}")
                self.logger.info(f"[RETRY EVAL]   Retries remaining: {MAX_JOB_RETRIES - current_retry_count}")

                # Check if we have retries remaining and if the failure is retriable
                if current_retry_count < MAX_JOB_RETRIES:
                    self.logger.info(f"[RETRY EVAL] ✓ Retries remaining "
                                     f"({MAX_JOB_RETRIES - current_retry_count}) - "
                                     f"checking if failure is retriable...")

                    # Get log_dir for checking error patterns
                    job_dir = self._get_stored_job_dir(job_id)
                    log_dir = f"{job_dir}/slurm-logs" if job_dir else None
                    self.logger.info(f"[RETRY EVAL] Log directory for error checking: {log_dir}")

                    if self._is_retriable_failure(slurm_status, job_id, log_dir):
                        self.logger.info("[RETRY EVAL] ✓ Failure is RETRIABLE - initiating job resubmission...")
                        self.logger.info(f"[RETRY EVAL] This will be retry attempt "
                                         f"{current_retry_count + 1} of {MAX_JOB_RETRIES}")

                        # Attempt to resubmit the job
                        success, new_slurm_job_id, error_msg = self._resubmit_failed_job(
                            job_id=job_id,
                            failed_slurm_job_id=slurm_job_id,
                            retry_count=current_retry_count,
                            is_automl_experiment=is_automl_experiment,
                            brain_job_id=brain_job_id,
                            experiment_number=experiment_number
                        )

                        if success:
                            # Job was resubmitted successfully - return Pending
                            self.logger.info("=" * 80)
                            self.logger.info("[RETRY EVAL] ✓ JOB RESUBMISSION SUCCESSFUL")
                            self.logger.info(f"[RETRY EVAL]   Old SLURM Job ID: {slurm_job_id} (FAILED)")
                            self.logger.info(f"[RETRY EVAL]   New SLURM Job ID: {new_slurm_job_id} (PENDING)")
                            self.logger.info("[RETRY EVAL]   Returning status: Pending")
                            self.logger.info("=" * 80)
                            return "Pending"

                        self.logger.error("=" * 80)
                        self.logger.error(f"[RETRY EVAL] ✗ JOB RESUBMISSION FAILED: {error_msg}")
                        self.logger.error("[RETRY EVAL] Will mark job as failed")
                        self.logger.error("=" * 80)
                        # Fall through to mark as failed
                    else:
                        self.logger.info("[RETRY EVAL] ✗ Failure is NOT retriable based on error pattern analysis")
                        self.logger.info("[RETRY EVAL] This appears to be a user code error, "
                                         "not infrastructure failure")
                        self.logger.info("[RETRY EVAL] Will mark job as failed without retry")
                else:
                    self.logger.warning("=" * 80)
                    self.logger.warning(f"[RETRY EVAL] ✗ MAX RETRIES EXHAUSTED ({MAX_JOB_RETRIES})")
                    self.logger.warning(f"[RETRY EVAL] Job {job_id} has failed {current_retry_count} times")
                    self.logger.warning(f"[RETRY EVAL] Failed SLURM Job IDs: {failed_job_ids_so_far}")
                    self.logger.warning("[RETRY EVAL] No more retries available - marking job as permanently failed")
                    self.logger.warning("=" * 80)

                # No more retries or not retriable - mark as failed
                failed_job_ids = retry_info.get("failed_slurm_job_ids", [])
                if slurm_job_id not in failed_job_ids:
                    failed_job_ids.append(slurm_job_id)

                if current_retry_count >= MAX_JOB_RETRIES:
                    message = (
                        f"SLURM job failed after {current_retry_count} retries. "
                        f"Last status: {slurm_status}. "
                        f"Failed job IDs: {failed_job_ids}"
                    )
                else:
                    message = f"SLURM job {slurm_job_id} failed with status: {slurm_status}"

                self.logger.info(f"[RETRY EVAL] Final failure message: {message}")
                self.logger.info("[RETRY EVAL] Updating job status to FAILURE")
                self.update_job_status(job_id, "FAILURE", message)
                return "Error"

            if slurm_status in ("CANCELLED", "PREEMPTED", "REVOKED"):
                message = f"SLURM job {slurm_job_id} was cancelled (status: {slurm_status})"
                self.update_job_status(job_id, "CANCELLED", message)
                return "Canceled"

            if slurm_status == "TIMEOUT":
                message = f"SLURM job {slurm_job_id} hit time limit and was terminated"
                self.update_job_status(job_id, "FAILURE", message)
                return "Error"

            if slurm_status in ("SUSPENDED", "STOPPED"):
                message = f"SLURM job {slurm_job_id} is suspended (status: {slurm_status})"
                self.update_job_status(job_id, "PAUSED", message)
                return "Paused"

            if slurm_status == "NOT_FOUND":
                self.logger.warning(f"SLURM job {slurm_job_id} not found in queue or history")
                return "Pending"

            if slurm_status == "ERROR":
                self.logger.error(f"Error querying SLURM status for job {slurm_job_id}")
                message = "Unable to check SLURM job status (SSH/network issue). Will retry..."
                self.update_job_status(job_id, "RUNNING", message)
                # Return current metadata status instead of marking as Error
                # This allows the system to retry on next check
                return job_metadata.get("status", "Running")

            # Unknown status
            self.logger.warning(f"Unknown SLURM status '{slurm_status}' for job {job_id}")
            return "Pending"

        except Exception as e:
            self.logger.error(traceback.format_exc())
            self.logger.error(f"Error getting SLURM job status for {job_id}: {e}")
            return "Error"

    def get_job_status(self, job_id, **kwargs):
        """Get job status - unified interface for ExecutionHandler

        Args:
            job_id: Job identifier
            **kwargs: Additional parameters (results_dir and workspace_metadata)

        Returns:
            str: Job status (Pending, Running, Done, Error, etc.)
        """
        self.logger.info(f"Checking SLURM job {job_id} status")

        # Check SLURM batch job status (from squeue/sacct)
        slurm_status = self.get_tao_job_status(job_id)
        self.logger.info(f"SLURM job {job_id} TAO status: {slurm_status}")

        results_dir = kwargs.get("results_dir")
        workspace_metadata = kwargs.get("workspace_metadata")
        if not results_dir or not workspace_metadata:
            self.logger.error(f"No results_dir or workspace_metadata found for SLURM job {job_id}")
            return "Error"

        # Check status.json for terminal states (Done/Error)
        # This handles cases where training completes/fails but SLURM job is still running
        try:
            from nvidia_tao_core.microservices.handlers.container_handler import ContainerJobHandler
            container_job_status = ContainerJobHandler.get_current_job_status(
                results_dir, workspace_metadata, job_id=job_id)
            self.logger.info(f"SLURM job {job_id} status.json status: {container_job_status}")

            # If status.json shows terminal state (Done/Error), use that and cancel SLURM job
            if container_job_status in ("Done", "Error"):
                self.logger.info(
                    f"SLURM job {job_id}: status.json shows terminal state '{container_job_status}' "
                    f"(SLURM status: {slurm_status}) - returning {container_job_status}"
                )
                # Cancel the SLURM job if it's still running since job reached terminal state
                if slurm_status == "Running":
                    self._cancel_slurm_job_by_tao_id(job_id)
                return container_job_status
        except Exception as e:
            self.logger.debug(f"Could not read status.json for {job_id}: {e}")
            # Continue with SLURM status if status.json is not available

        # If SLURM status is definitive (not None) and status.json didn't show terminal,
        # use SLURM status. None means COMPLETED - already handled above via status.json
        if slurm_status is not None:
            return slurm_status

        # SLURM job completed but status.json didn't show terminal state
        # This shouldn't normally happen, but return Pending as fallback
        self.logger.warning(
            f"SLURM job {job_id} COMPLETED but status.json doesn't show terminal state. "
            "Job may have exited abnormally."
        )
        return "Pending"

    def _cancel_slurm_job_by_tao_id(self, job_id):
        """Cancel SLURM job when status.json shows terminal state (SUCCESS/FAILURE).

        This is an internal helper that cancels the SLURM job without updating
        the TAO job status to CANCELLED (since job already reached terminal state).

        Args:
            job_id: TAO job ID
        """
        try:
            # Get AutoML-aware parameters
            automl_params = self._get_automl_aware_handler_params(job_id)
            brain_job_id = automl_params['brain_job_id']
            is_automl_experiment = automl_params['is_automl_experiment']

            # Get job metadata to find SLURM job ID
            job_metadata = get_handler_job_metadata(brain_job_id)
            if not job_metadata:
                self.logger.debug(f"No metadata found for job {job_id}, cannot cancel SLURM job")
                return

            # Get SLURM job ID from backend_details.slurm_metadata structure
            slurm_job_id = None
            if is_automl_experiment:
                controller_info = get_automl_controller_info(brain_job_id)
                experiment_number = automl_params['experiment_number']
                if controller_info and len(controller_info) > experiment_number:
                    experiment_info = controller_info[experiment_number]
                    backend_details = experiment_info.get('backend_details', {})
                    slurm_metadata = backend_details.get('slurm_metadata', {})
                    slurm_job_id = slurm_metadata.get('slurm_job_id')
            else:
                backend_details = job_metadata.get('backend_details', {})
                slurm_metadata = backend_details.get('slurm_metadata', {})
                slurm_job_id = slurm_metadata.get('slurm_job_id')

            if not slurm_job_id:
                self.logger.debug(f"No SLURM job ID found for TAO job {job_id}")
                return

            # Cancel the SLURM job
            self.logger.info(f"Canceling SLURM job {slurm_job_id} (job reached terminal state)")
            scancel_cmd = f"scancel {slurm_job_id}"
            ssh_cmd = self._build_ssh_command(scancel_cmd)

            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )

            if result.returncode == 0:
                self.logger.info(f"Successfully canceled SLURM job {slurm_job_id} after job reached terminal state")
            else:
                error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
                # Not an error if job already terminated
                if "Invalid job id" in error_msg or "does not exist" in error_msg:
                    self.logger.debug(f"SLURM job {slurm_job_id} already terminated")
                else:
                    self.logger.warning(f"scancel for {slurm_job_id} returned: {error_msg}")

        except Exception as e:
            self.logger.warning(f"Failed to cancel SLURM job for TAO job {job_id}: {e}")

    def cancel_job(self, job_id):
        """Cancel a SLURM job using scancel.

        Args:
            job_id: TAO job ID (will lookup SLURM job ID from metadata)

        Returns:
            bool: True if cancellation was successful or job already terminated, False otherwise
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info(f"SLURM JOB CANCEL: Canceling job {job_id}")

            # Get AutoML-aware parameters
            automl_params = self.get_automl_aware_handler_params(job_id)
            brain_job_id = automl_params['brain_job_id']

            # Get job metadata to find SLURM job ID
            job_metadata = get_handler_job_metadata(brain_job_id)
            if not job_metadata:
                self.logger.warning(f"No metadata found for job {job_id}")
                self.logger.info("=" * 80)
                return True  # Consider it success if job doesn't exist

            # Check if this is AutoML experiment - get SLURM job ID from controller info
            slurm_job_id = self.get_slurm_job_id(job_id)

            if not slurm_job_id:
                self.logger.warning(f"No SLURM job ID found for TAO job {job_id}")
                self.logger.info("Job may not have been submitted to SLURM yet, or already completed")
                self.logger.info("=" * 80)
                return True  # No SLURM job to cancel

            # Check if job is already completed
            slurm_status = self.get_slurm_job_status(slurm_job_id)
            if not slurm_status:
                self.logger.info(f"SLURM job {slurm_job_id} not found (may have already completed)")
                self.logger.info("=" * 80)
                return True

            if slurm_status in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
                self.logger.info(f"SLURM job {slurm_job_id} already terminated with status: {slurm_status}")
                self.logger.info("=" * 80)
                return True

            # Cancel the SLURM job
            self.logger.info(f"Canceling SLURM job {slurm_job_id} (current status: {slurm_status})")
            scancel_cmd = f"scancel {slurm_job_id}"
            ssh_cmd = self._build_ssh_command(scancel_cmd)

            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )

            if result.returncode == 0:
                self.logger.info(f"Successfully canceled SLURM job {slurm_job_id}")

                # Update job message to reflect cancellation
                self.update_job_status(job_id, "CANCELLED", f"SLURM job {slurm_job_id} canceled by user")

                self.logger.info("=" * 80)
                return True

            error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
            # scancel returns non-zero if job doesn't exist (already terminated)
            if "Invalid job id specified" in error_msg or "does not exist" in error_msg:
                self.logger.info(f"SLURM job {slurm_job_id} no longer exists (already terminated)")
                self.logger.info("=" * 80)
                return True

            self.logger.error(f"Failed to cancel SLURM job {slurm_job_id}: {error_msg}")
            self.logger.info("=" * 80)
            return False

        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout while canceling SLURM job for {job_id}")
            self.logger.info("=" * 80)
            return False
        except Exception as e:
            self.logger.error(f"Error canceling SLURM job for {job_id}: {e}")
            self.logger.error(traceback.format_exc())
            self.logger.info("=" * 80)
            return False

    def delete(self, job_id):
        """Delete a SLURM job"""
        return self.cancel_job(job_id)

    def fetch_status_json(self, results_dir, **kwargs):
        """Fetch status.json from the SLURM cluster's Lustre storage.

        Searches for status.json in the results directory and its subdirectories.

        Args:
            results_dir: Path to the results directory on Lustre (e.g., /lustre/.../results/job-uuid)

        Returns:
            list: List of status line dictionaries, or empty list if file doesn't exist or can't be read
        """
        self.logger.info("=" * 80)
        self.logger.info("SLURM STATUS FETCH: Searching for status.json")
        self.logger.info(f"Results dir: {results_dir}")
        self.logger.info(f"SSH: {self.login_user}@{self.login_hostname}")

        # Search for status.json in results_dir and subdirectories
        # Use find command to locate the file (limits to 3 directory levels deep for performance)
        find_command = f"find {shlex.quote(results_dir)} -maxdepth 3 -type f -name 'status.json' 2>/dev/null | head -1"
        self.logger.info(f"Find command: {find_command}")

        ssh_command = self._build_ssh_command(find_command)
        self.logger.debug(f"Full SSH command: {' '.join(ssh_command)}")

        try:
            self.logger.info("Executing find command via SSH...")
            result = subprocess.run(
                ssh_command,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=False
            )

            self.logger.debug(f"Find return code: {result.returncode}")
            self.logger.debug(f"Find stdout: {result.stdout[:200] if result.stdout else '(empty)'}")
            self.logger.debug(f"Find stderr: {result.stderr[:200] if result.stderr else '(empty)'}")

            status_file_path = result.stdout.strip()

            if not status_file_path:
                self.logger.info(f"status.json NOT FOUND in {results_dir}")
                self.logger.info("This is normal if the job hasn't started writing status yet")
                self.logger.info("=" * 80)
                return []

            self.logger.info(f"Found status.json at: {status_file_path}")

            # File exists - read it
            cat_command = f"cat {shlex.quote(status_file_path)}"
            self.logger.debug(f"Cat command: {cat_command}")

            ssh_command = self._build_ssh_command(cat_command)

            self.logger.info("Reading status.json via SSH...")
            result = subprocess.run(
                ssh_command,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=False
            )

            self.logger.debug(f"Cat return code: {result.returncode}")

            if result.returncode != 0:
                self.logger.warning(f"Failed to read status.json: {result.stderr}")
                self.logger.info("=" * 80)
                return []

            file_size = len(result.stdout)
            self.logger.info(f"Read {file_size} bytes from status.json")

            # Parse status.json - each line is a separate JSON object
            status_lines = []
            parse_errors = 0

            for line_num, line in enumerate(result.stdout.strip().split('\n'), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    status_entry = json.loads(line)
                    status_lines.append(status_entry)
                except json.JSONDecodeError as e:
                    parse_errors += 1
                    if parse_errors <= 3:  # Only log first 3 errors to avoid spam
                        self.logger.warning(f"⚠ Failed to parse line {line_num}: {line[:100]}... Error: {e}")
                    continue

            if parse_errors > 3:
                self.logger.warning(f"⚠ Total parse errors: {parse_errors} (showing first 3)")

            self.logger.info(f"Successfully parsed {len(status_lines)} status entries")
            self.logger.info("=" * 80)
            return status_lines

        except subprocess.TimeoutExpired as e:
            self.logger.error(f"SSH command timed out after {TIMEOUT_SECONDS} seconds: {e}")
            self.logger.info("=" * 80)
            return []
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"SSH command failed: {e}")
            self.logger.info("=" * 80)
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error fetching status.json: {e}")
            self.logger.error(traceback.format_exc())
            self.logger.info("=" * 80)
            return []

    def sync_status_to_database(self, job_id, results_dir, **kwargs):
        """Fetch status.json from SLURM and sync to database.

        This method replicates what status callbacks do for local GPU jobs:
        - Fetches status.json from Lustre
        - Parses the status lines
        - Saves them to the database using save_dnn_status()

        Args:
            job_id: TAO job ID (can be experiment job ID for AutoML)
            results_dir: Path to results directory on Lustre
            handler_id: Handler ID (optional, for logging)
            kind: Handler kind (optional, for logging)

        Returns:
            bool: True if sync was successful, False otherwise
        """
        self.logger.info("=" * 80)
        self.logger.info("SLURM STATUS SYNC: Syncing status to database")
        self.logger.info(f"Job ID: {job_id}")
        self.logger.info(f"Results dir: {results_dir}")

        # Detect if this is an AutoML experiment
        automl_params = self.get_automl_aware_handler_params(job_id)

        is_automl_experiment = automl_params['is_automl_experiment']
        if is_automl_experiment:
            # For AutoML experiments, find the experiment number from the brain's controller info
            brain_job_id = automl_params['brain_job_id']
            experiment_job_id = automl_params['experiment_job_id']
            controller_info = get_automl_controller_info(brain_job_id)
            experiment_number = None
            for exp in controller_info:
                if exp.get('job_id') == job_id:
                    experiment_number = str(exp.get('id'))
                    break

            if experiment_number is None:
                self.logger.warning(f"⚠ Could not find experiment number for AutoML job {job_id}")
                experiment_number = "0"

            self.logger.info("Detected AutoML experiment")
            self.logger.info(f"Brain Job ID: {brain_job_id}")
            self.logger.info(f"Experiment Number: {experiment_number}")
            self.logger.info(f"Experiment Job ID: {experiment_job_id}")
        else:
            self.logger.info("Not an AutoML experiment")
            brain_job_id = job_id
            experiment_number = "0"

        try:
            # Fetch status lines from Lustre
            self.logger.info("Step 1: Fetching status.json from Lustre...")
            status_lines = self.fetch_status_json(results_dir)

            if not status_lines:
                self.logger.info(f"No status updates found for job {job_id}")
                self.logger.info("This is expected if job hasn't started or hasn't written status yet")
                self.logger.info("=" * 80)
                return False

            self.logger.info(f"Fetched {len(status_lines)} status entries from Lustre")

            # Get existing status from database to avoid duplicates
            self.logger.info("Step 2: Checking for existing status in database...")
            existing_status = get_dnn_status(
                brain_job_id,
                automl=is_automl_experiment,
                experiment_number=experiment_number
            )
            existing_count = len(existing_status) if existing_status else 0

            self.logger.info(f"Database has {existing_count} existing status entries")

            # Only save new status lines (ones we haven't saved yet)
            new_status_count = len(status_lines) - existing_count

            if new_status_count <= 0:
                self.logger.info(
                    "  No new status updates to sync "
                    f"(DB has {existing_count}, file has {len(status_lines)})"
                )
                self.logger.info("Status is already up to date")
                self.logger.info("=" * 80)
                return True

            self.logger.info(f"Found {new_status_count} NEW status entries to sync")

            # Save only the new status lines
            self.logger.info("Step 3: Saving new status entries to database...")

            saved_count = 0
            for idx, status_entry in enumerate(status_lines[existing_count:], start=1):
                try:
                    # Ensure timestamp exists
                    if 'timestamp' not in status_entry:
                        status_entry['timestamp'] = datetime.now(tz=timezone.utc).isoformat()

                    # Create callback_data structure that save_dnn_status expects
                    callback_data = {
                        'status': json.dumps(status_entry),
                        'experiment_number': experiment_number
                    }

                    self.logger.debug(f"Saving status entry {idx}/{new_status_count}: {str(status_entry)[:100]}...")

                    save_dnn_status(
                        job_id=brain_job_id,
                        automl=is_automl_experiment,
                        callback_data=callback_data,
                        experiment_number=experiment_number
                    )
                    saved_count += 1

                except Exception as save_error:
                    self.logger.error(f"Failed to save status entry {idx}: {save_error}")
                    self.logger.error(traceback.format_exc())
                    # Continue with next entry
                    continue

            if saved_count == new_status_count:
                self.logger.info(f"Successfully saved all {saved_count} new status entries to database")
            else:
                self.logger.warning(f"Saved {saved_count}/{new_status_count} status entries (some failed)")

            self.logger.info(f"Database now has {existing_count + saved_count} total status entries")
            self.logger.info("SYNC COMPLETE")
            self.logger.info("=" * 80)
            return True

        except Exception as e:
            self.logger.error(f"Error syncing status to database for job {job_id}: {e}")
            self.logger.error(traceback.format_exc())
            self.logger.info("=" * 80)
            return False
