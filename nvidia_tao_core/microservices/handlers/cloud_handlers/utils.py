# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Cloud Storage handler"""
import ast
import fnmatch
import glob
import json
import logging
import os
import re
import requests
import subprocess
import sys
import tarfile
import time
import traceback

from nvidia_tao_core.microservices.utils.cloud_utils import CloudStorage
from nvidia_tao_core.microservices.utils.ngc_utils import download_ngc_model, split_ngc_path, get_model_size_info
from nvidia_tao_core.microservices.utils.stateless_handler_utils import BACKEND, get_internal_job_status_update_data
from nvidia_tao_core.distributed.decorators import master_node_only
from nvidia_tao_core.microservices.enum_constants import Backend


logger = logging.getLogger(__name__)
NUM_RETRY = 3
REQUESTS_TIMEOUT = 180


def _untar_file(tar_path, dest, strip_components=0):
    """Function to untar a file.

    Args:
        tar_path (str): The path to the tar file to be untarred.
        dest (str): The destination directory where the contents will be extracted.
        strip_components (int, optional): The number of leading directory components to strip (default is 0).
    """
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            # Remove leading directory components using strip_components
            components = member.name.split(os.sep)
            if len(components) > strip_components:
                member.name = os.path.join(*components[strip_components:])
            if member.isdir():
                # Make subdirs ahead because tarfile extracts them with user permissions only
                os.makedirs(os.path.join(dest, member.name), exist_ok=True)
            tar.extract(member, path=dest, set_attrs=False)


def _extract_images(tar_path, dest, remove_tar=True):
    """Function to extract images, other directories on same level as images to root of dataset.

    Args:
        tar_path (str): The path to the tar file.
        dest (str): The destination directory where the contents will be extracted.
        remove_tar (bool): Whether to remove the tar file after extraction. Default: True
    """
    # Infer how many components to strip to get images,labels to top of dataset directory
    # Assumes: images, other necessary directories are in the same level
    with tarfile.open(tar_path) as tar:
        strip_components = 0
        names = [tinfo.name for tinfo in tar.getmembers()]
        for name in names:
            if "/images/" in name:
                strip_components = name.split("/").index("images")
                break
    # Build shell command for untarring
    logger.info("Untarring data started")
    _untar_file(tar_path, dest, strip_components)
    logger.info("Untarring data complete")

    # Remove .tar.gz file (optional)
    if remove_tar:
        logger.info("Removing data tar file")
        os.remove(tar_path)
        logger.info("Deleted data tar file")
    else:
        logger.info("Keeping tar file (remove_tar=False)")


def search_for_ptm(root, network="", parameter_name=""):
    """Return path of the PTM file under the PTM root folder"""
    models = None
    models = (
        glob.glob(root + "/**/*.tlt", recursive=True) +
        glob.glob(root + "/**/*.hdf5", recursive=True) +
        glob.glob(root + "/**/*.pth", recursive=True) +
        glob.glob(root + "/**/*.pth.tar", recursive=True) +
        glob.glob(root + "/**/*.pt", recursive=True)
    )
    # TODO: remove after next nvaie release, Varun and Subha
    if network in ("classification_pyt", "visual_changenet", "nvdinov2"):
        models += glob.glob(root + "/**/*.ckpt", recursive=True)
    if network in ("classification_tf2", "efficientdet_tf2"):
        models = [os.path.join(root, os.listdir(root)[0])]
    if network == "stylegan_xl":
        if parameter_name == "inception_fid_path":
            models = glob.glob(root + "/**/*Inception*.pth", recursive=True)
        if parameter_name == "input_embeddings_path":
            models = glob.glob(root + "/**/*tf_efficientnet*.pth", recursive=True)
    if models:
        model_path = models[0]  # pick one arbitrarily
        logger.info("Found valid PTM at {}".format(model_path)) # noqa pylint: disable=C0209
        return model_path
    if os.path.exists(root):
        if network == "vila":
            return os.path.join(root, os.listdir(root)[0])
        return root
    logger.info("PTM can't be found")
    return None


def run_subprocess_command(command):
    """Run a subprocess command.

    Args:
        command (str): The command to run.
    """
    try:
        # Run the script.
        subprocess.check_call(
            ['/bin/bash', '-c', command],
            shell=False,
            stdout=sys.stdout,
            stderr=sys.stdout
        )
    except (KeyboardInterrupt, SystemExit) as e:
        logger.info("Command was interrupted due to {}".format(e)) # noqa pylint: disable=C0209
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            logger.info("Called process error {}".format(e.output)) # noqa pylint: disable=C0209


def download_from_https_link(download_url, destination_folder):
    """Download a file from an HTTP link.

    Args:
        download_url (str): The URL of the file to download.
        destination_folder (str): The destination folder where the file will be saved.
    """
    cmnd = (
        f"until wget --timeout=1 --tries=1 --retry-connrefused "
        f"--no-verbose --directory-prefix={destination_folder}/ {download_url}; "
        f"do sleep 10; done"
    )
    run_subprocess_command(cmnd)
    file_name = download_url.split("/")[-1]
    if file_name.endswith(".tar") or file_name.endswith(".tar.gz"):
        _extract_images(f'{destination_folder}/{file_name}', f'{destination_folder}')


def download_from_git_link(download_url, destination_folder):
    """Download a repository from a git link.

    Args:
        download_url (str): The URL of the repository to download.
        destination_folder (str): The destination folder where the repository will be saved.
    """
    cmnd = f"git clone {download_url} {destination_folder}"
    run_subprocess_command(cmnd)


def extract_cloud_details(metadata):
    """Extract cloud-related details from metadata.

    Args:
        metadata (dict): Metadata dictionary.

    Returns:
        Tuple: Cloud details (cloud_type, bucket_name, access_key, secret_key, region, cloud_file_path).
    """
    cloud_type = metadata.get('cloud_type', '')

    # AWS, AZURE
    bucket_name = metadata.get('cloud_specific_details', {}).get('cloud_bucket_name', '')
    access_key = metadata.get('cloud_specific_details', {}).get('access_key', '')
    secret_key = metadata.get('cloud_specific_details', {}).get('secret_key', '')
    region = metadata.get('cloud_specific_details', {}).get('cloud_region', '')

    # Self_hosted, HuggingFace
    download_url = metadata.get('cloud_specific_details', {}).get('url', '')

    # HuggingFace
    token = metadata.get('cloud_specific_details', {}).get('token', '')

    return cloud_type, bucket_name, access_key, secret_key, region, download_url, token


def initialize_cloud_storage(cloud_type, bucket_name, region, access_key, secret_key, endpoint_url=None):
    """Initialize CloudStorage instance.

    Args:
        cloud_type (str): Type of cloud storage.
        bucket_name (str): Name of the bucket/container.
        region (str): Region for the cloud storage provider.
        access_key (str): Access key for authentication.
        secret_key (str): Secret key for authentication.
        endpoint_url (str): Endpoint URL for the cloud storage provider.

    Returns:
        CloudStorage: Initialized CloudStorage instance.
    """
    # Prepare client_kwargs - only include endpoint_url if it's provided
    client_kwargs = {}
    if endpoint_url and endpoint_url != "":
        client_kwargs["endpoint_url"] = endpoint_url

    return CloudStorage(
        cloud_type=cloud_type,
        bucket_name=bucket_name,
        region=region,
        key=access_key,
        secret=secret_key,
        client_kwargs=client_kwargs
    )


def search_for_dataset(root):
    """Return path of the dataset file"""
    datasets = (
        glob.glob(root + "/*.tar.gz", recursive=False) +
        glob.glob(root + "/*.tgz", recursive=False) +
        glob.glob(root + "/*.tar", recursive=False)
    )

    if datasets:
        dataset_path = datasets[0]  # pick one arbitrarily
        return dataset_path
    return None


def download_files(cloud_storage, cloud_file_path, local_path):
    """Download files from cloud storage.

    Args:
        cloud_storage (CloudStorage): Initialized CloudStorage instance.
        cloud_file_path (str): Path of the file in cloud storage.
        local_path (str): Local path for downloading the file.
    """
    if not local_path.endswith("/"):
        local_path = f"{local_path}/"
    cloud_storage.download_file(cloud_file_path, local_path)
    if cloud_file_path.endswith(".tar") or cloud_file_path.endswith(".tar.gz"):
        _extract_images(os.path.join(local_path, os.path.basename(cloud_file_path)), local_path)
        dir_name = cloud_file_path.split("/")[-1].split(".")[0]
        destination_path = f"{local_path}{dir_name}"
        if not os.path.exists(destination_path):
            cleanup_cuda_contexts()
            raise ValueError("Folder name not same as the file name")
    else:
        file_name = cloud_file_path.split("/")[-1]
        destination_path = f"{local_path}{file_name}"
        if not os.path.isfile(destination_path):
            cleanup_cuda_contexts()
            raise ValueError("Unable to download the file")
    return destination_path


def get_file_modification_time(local_path):
    """Gets file modification time and ignores any issue in getting so.

    Args:
        local_path (str): The local path to monitor.
        cloud_storage: An instance of the CloudStorage class for uploading files.
        file_last_modified: Dictionary to find modified files
    """
    try:
        return os.path.getmtime(local_path)
    except Exception:
        return 0


@master_node_only
def upload_files(local_path, cloud_storage, file_last_modified=None,
                 selective_tarball_config=None, exclude_patterns=None,
                 file_snapshot=None, progress_tracker=None, exit_event=None,
                 retain_patterns=None):
    """Uploads any detected changes to the specified cloud storage.

    Args:
        local_path (str): The local path to monitor.
        cloud_storage: An instance of the CloudStorage class for uploading files.
        file_last_modified (dict, optional): Dictionary to find modified files. Not used when file_snapshot is provided.
        selective_tarball_config (dict): Configuration for selective tarball patterns to skip.
        exclude_patterns (list, optional): List of regex patterns to exclude files from upload.
        file_snapshot (set, optional): Set of relative file paths to upload. If provided, only these files are uploaded.
        progress_tracker: ProgressTracker instance for batch upload tracking (post-job only)
        exit_event: Threading event to check for early exit during continuous upload
        retain_patterns (list, optional): List of regex patterns to retain files until job completion.
                                         Files matching these patterns won't be removed after upload
                                         during continuous monitoring. Set to None during final upload
                                         or graceful pause to remove all files.
    """
    # If no snapshot provided, create one from directory walk (for continuous monitoring)
    # Track which files need their modification time updated after successful upload
    pending_mod_time_updates = {}

    if not file_snapshot:
        file_snapshot = set()
        for root, _, files in os.walk(local_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                current_last_modified = get_file_modification_time(file_path)
                if current_last_modified:
                    # Check if the file is new or modified
                    if (
                        file_path not in file_last_modified or
                        current_last_modified > file_last_modified[file_path]
                    ):
                        rel_path = os.path.relpath(file_path, local_path)
                        file_snapshot.add(rel_path)
                        # Don't update file_last_modified yet - do it after successful upload
                        # to avoid marking files as uploaded when they weren't
                        pending_mod_time_updates[file_path] = current_last_modified
                else:
                    logger.error("File could not be uploaded: %s", file_path)

    # Filter files to upload (apply all skip conditions)
    files_to_upload = []
    backend = os.getenv("TAO_EXECUTION_BACKEND", "")
    skip_log_files = backend in ("local-k8s", "local-docker")

    for rel_path in file_snapshot:
        file_path = os.path.join(local_path, rel_path)
        filename = os.path.basename(file_path)

        # Skip if file doesn't exist or is not a regular file
        if not (os.path.exists(file_path) and os.path.isfile(file_path)):
            continue

        # Skip log files for k8s/docker backends (server streams logs directly)
        if skip_log_files and filename == "microservices_log.txt":
            logger.debug(
                f"Skipping log file upload for backend={backend}: {file_path}. "
                "Server-side log streaming is enabled."
            )
            continue

        # Skip checkpoint and tmp files
        if "checkpoint-" in file_path or "tmp" in file_path:
            continue

        # Check exclude_patterns
        should_exclude = False
        if exclude_patterns:
            for pattern in exclude_patterns:
                try:
                    if re.search(pattern, rel_path) or re.search(pattern, filename):
                        should_exclude = True
                        logger.debug("Excluding file from upload due to pattern '%s': %s", pattern, rel_path)
                        break
                except re.error:
                    logger.warning("Invalid regex pattern '%s', skipping", pattern)

        if should_exclude:
            continue

        # Skip files that will be included in selective tarball
        if should_skip_file_for_tarball(file_path, local_path, selective_tarball_config):
            continue

        files_to_upload.append((rel_path, file_path))

    # Log summary of files to upload
    if files_to_upload:
        logger.info("Total files to upload: %d", len(files_to_upload))
        logger.info("Files to upload: %s", [rel_path for rel_path, _ in files_to_upload])

    # Process all files to upload
    # Defer large file removals until after all uploads complete to avoid
    # breaking symlinks whose targets are removed mid-batch
    deferred_removals = []
    for idx, (rel_path, file_path) in enumerate(files_to_upload, 1):
        if exit_event and exit_event.is_set() and progress_tracker is None:
            logger.info("Exit event detected during continuous upload, breaking early")
            break
        remaining = len(files_to_upload) - idx
        logger.info("Uploading file %d/%d: %s (remaining: %d)", idx, len(files_to_upload), file_path, remaining)
        try:
            send_callbacks = progress_tracker is not None  # Enable callbacks only for batch uploads
            if "graceful_termination_signal" not in file_path:
                if not file_last_modified or progress_tracker is not None:
                    cloud_storage.upload_file(
                        file_path, file_path, progress_tracker=progress_tracker,
                        send_status_callbacks=send_callbacks)
                else:
                    time.sleep(10)
                    cloud_storage.upload_file(
                        file_path, file_path, progress_tracker=progress_tracker,
                        send_status_callbacks=send_callbacks)

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to upload file: {} - Error: {}".format(file_path, str(e)))  # noqa pylint: disable=C0209
            continue  # Skip updating file_last_modified if upload failed

        # Update file_last_modified after successful upload
        # This ensures files aren't marked as uploaded if exit_event interrupts the loop
        if file_last_modified is not None:
            if file_path in pending_mod_time_updates:
                # Continuous monitoring mode: use the timestamp from when we detected the change
                file_last_modified[file_path] = pending_mod_time_updates[file_path]
                logger.info("Updated modification time for successfully uploaded file: %s", file_path)
            else:
                # Snapshot mode: get current modification time
                current_mod_time = get_file_modification_time(file_path)
                if current_mod_time:
                    file_last_modified[file_path] = current_mod_time
                    logger.info("Updated modification time for snapshot-mode file: %s", file_path)

        # Schedule large files (>50MB) for deferred removal after all uploads
        try:
            if cloud_storage.is_file(file_path):
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                if file_size_mb > 50:
                    should_retain = False
                    if retain_patterns:
                        for pattern in retain_patterns:
                            try:
                                if re.search(pattern, rel_path) or re.search(pattern, os.path.basename(file_path)):
                                    should_retain = True
                                    logger.info(
                                        "File (%.2f MB) matches retain pattern '%s', keeping until job completion: %s",
                                        file_size_mb, pattern, file_path
                                    )
                                    break
                            except re.error:
                                logger.warning("Invalid regex pattern '%s', skipping", pattern)

                    if not should_retain:
                        deferred_removals.append((file_path, file_size_mb))
                    else:
                        logger.info(
                            "Large file (%.2f MB) successfully uploaded but retained (matches retain pattern): %s",
                            file_size_mb, file_path
                        )
                else:
                    logger.info("File (%.2f MB) successfully uploaded but retained (under 50MB): %s",
                                file_size_mb, file_path)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to check file size after upload: {} - Error: {}".format(file_path, str(e)))  # noqa pylint: disable=C0209

    # Remove large files after all uploads complete (prevents breaking symlinks)
    for file_path, file_size_mb in deferred_removals:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info("Large file (%.2f MB) successfully uploaded and removed: %s",
                            file_size_mb, file_path)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to remove file after upload: {} - Error: {}".format(file_path, str(e)))  # noqa pylint: disable=C0209


def get_log_file_name():
    """Return log file name"""
    job_id = os.getenv("JOB_ID")
    # Use TAO_API_RESULTS_DIR for SLURM compatibility, fallback to /results
    logs_dir = os.getenv('TAO_MICROSERVICES_TTY_LOG') or os.getenv('TAO_API_RESULTS_DIR', '/results')
    log_file = f'{logs_dir}/{job_id}/microservices_log.txt'
    return log_file


def send_logs_to_server(seek_position, retry=0):
    """Sends TTY logs back to Hosted API"""
    # Skip log callbacks when DEBUG_ENABLED is set (for local debugging)
    if os.getenv("DEBUG_ENABLED", "").lower() == "true":
        return seek_position

    if os.getenv("CLOUD_BASED") == "True":
        # Skip log upload for k8s and docker backends - server handles it via direct streaming
        # Check TAO_EXECUTION_BACKEND (set by server for job containers)
        backend = os.getenv("TAO_EXECUTION_BACKEND", BACKEND.value)
        backend = Backend(backend)
        if backend in (Backend.LOCAL_K8S, Backend.LOCAL_DOCKER):
            logger.info(
                f"Skipping container-side log upload for backend={backend.value}. "
                "Server-side log streaming is enabled."
            )
            return seek_position

        if retry >= NUM_RETRY:
            cleanup_cuda_contexts()
            raise ValueError("Log Callback was unsuccessfull")

        log_file = get_log_file_name()

        if os.path.isfile(log_file):
            with open(log_file, 'r', encoding='utf-8') as log_file:
                log_file.seek(seek_position)
                log_contents = log_file.read()
                seek_position = log_file.tell()
                ngc_key = os.getenv("TAO_USER_KEY")
                headers = {"Authorization": f"Bearer {ngc_key}"}
                if log_contents and headers:
                    log_callback_url = os.getenv("TAO_LOGGING_SERVER_URL") + ":log_update"
                    if log_callback_url:
                        headers['Content-Type'] = 'application/json'
                        data = {
                            'experiment_number': os.getenv("AUTOML_EXPERIMENT_NUMBER", "0"),
                            'log_contents': log_contents
                        }
                        try:
                            response = requests.post(
                                log_callback_url,
                                json=data,
                                headers=headers,
                                timeout=REQUESTS_TIMEOUT
                            )
                            if response.ok:
                                return seek_position
                            logger.info(
                                "Failed to send logs. Status code: {}".format(response.status_code)  # noqa pylint: disable=C0209
                            )
                            seek_position -= len(log_contents)
                            retry += 1

                        except requests.RequestException as e:
                            logger.info("Exception during log sending: {}".format(e))  # noqa pylint: disable=C0209
                            seek_position -= len(log_contents)
                            retry += 1

                        time.sleep(5)
                        return send_logs_to_server(seek_position, retry)
    return seek_position


def status_callback(data_string, retry=0):
    """Sends status update data back to the server.

    Args:
        data_string (str): The status data to be sent.
        retry (int, optional): The current retry attempt (default is 0).
    """
    # Skip callbacks when DEBUG_ENABLED is set (for local debugging)
    if os.getenv("DEBUG_ENABLED", "").lower() == "true":
        logger.info("DEBUG_ENABLED=True: Skipping status callback")
        return

    # Check for early stopping based on epoch threshold
    early_stop_epoch = os.getenv("EARLY_STOP_EPOCH")
    if early_stop_epoch:
        try:
            early_stop_epoch = int(early_stop_epoch)
            # Parse status data to extract current epoch
            status_data = json.loads(data_string)
            current_epoch = status_data.get("epoch")

            if current_epoch is not None and current_epoch > early_stop_epoch:
                logger.info(f"Early stop triggered: epoch {current_epoch} >= {early_stop_epoch}")
                # Write graceful termination signal
                job_id = os.getenv("JOB_ID")
                if job_id:
                    results_dir = f"/results/{job_id}"
                    signal_file = os.path.join(results_dir, ".graceful_termination_signal")
                    try:
                        os.makedirs(results_dir, exist_ok=True)
                        if not os.path.exists(signal_file):
                            with open(signal_file, 'w', encoding='utf-8') as f:
                                f.write(f"early_stop_epoch_{early_stop_epoch}")
                            logger.info(f"Early stop signal written for job {job_id} at epoch {current_epoch}")
                    except Exception as e:
                        logger.error(f"Failed to write early stop signal: {e}")
        except (ValueError, json.JSONDecodeError) as e:
            logger.debug(f"Could not parse early stop epoch or status data: {e}")

    if os.getenv("CLOUD_BASED") == "True":
        if retry >= NUM_RETRY:
            cleanup_cuda_contexts()
            raise ValueError("Status Callback was unsuccessful after multiple retries")

        ngc_key = os.getenv("TAO_ADMIN_KEY")
        headers = {"Authorization": f"Bearer {ngc_key}"}
        if data_string and headers:
            status_url = os.getenv("TAO_LOGGING_SERVER_URL", "") + ":status_update"
            if status_url:
                data = {
                    "experiment_number": os.getenv("AUTOML_EXPERIMENT_NUMBER", "0"),
                    "status": data_string,
                }
                try:
                    response = requests.post(status_url, json=data, headers=headers, timeout=REQUESTS_TIMEOUT)
                    if response.ok:
                        logger.info(f"Status update with data {data} sent successfully")
                        return
                    logger.error(
                        "Failed to send status update. Status code: {}".format(  # noqa pylint: disable=C0209
                            response.status_code
                        )
                    )
                    retry += 1

                except requests.RequestException as e:
                    logger.error(
                        "Exception during status update sending: {}".format(e)  # noqa pylint: disable=C0209
                    )
                    retry += 1

                time.sleep(5)
                status_callback(data_string, retry)


def should_skip_file_for_tarball(file_path, local_path, selective_tarball_config):
    """Check if a file should be skipped because it will be included in selective tarball.

    Args:
        file_path (str): Full path to the file
        local_path (str): Base local path
        selective_tarball_config (dict): Selective tarball configuration

    Returns:
        bool: True if file should be skipped, False otherwise
    """
    if not selective_tarball_config:
        return False

    patterns = selective_tarball_config.get("patterns", [])
    base_path = selective_tarball_config.get("base_path", "")

    if not patterns:
        return False

    # Calculate the search directory
    search_dir = os.path.join(local_path, base_path) if base_path else local_path

    # Check if file is within the search directory
    if not file_path.startswith(search_dir):
        return False

    # Get relative path from search directory
    rel_path = os.path.relpath(file_path, search_dir)

    # Check if file matches any pattern
    for pattern in patterns:
        if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(file_path, os.path.join(search_dir, pattern)):
            logger.debug("Skipping continuous upload for tarball file: %s (matches pattern: %s)", rel_path, pattern)
            return True

    return False


@master_node_only
def monitor_and_upload(local_path, cloud_storage, exit_event, seek_position=0,
                       selective_tarball_config=None, exclude_patterns=None, retain_patterns=None):
    """Monitors the specified local path and its subdirectories for new or modified files.

    Args:
        local_path (str): The local path to monitor.
        cloud_storage: An instance of the CloudStorage class for uploading files.
        exit_event (threading.Event): An event to signal the thread to exit.
        seek_position (int): Initial seek position for log reading.
        selective_tarball_config (dict): Configuration for selective tarball patterns to skip.
        exclude_patterns (list, optional): List of regex patterns to exclude files from upload.
        retain_patterns (list, optional): List of regex patterns to retain files until job completion.

    Returns:
        None
    """
    logger.info("monitor_and_upload :: Entering")

    # For k8s and docker backends, skip log file uploads (server handles via streaming)
    # But still handle send_logs_to_server for status updates
    # Check TAO_EXECUTION_BACKEND (set by server for job containers)
    backend = os.getenv("TAO_EXECUTION_BACKEND", BACKEND.value)
    backend = Backend(backend)
    skip_log_upload = backend in (Backend.LOCAL_K8S, Backend.LOCAL_DOCKER)
    if skip_log_upload:
        logger.info(
            f"Backend={backend.value}: Server-side log streaming enabled. "
            "Log files will not be uploaded from container."
        )

    if selective_tarball_config:
        patterns = selective_tarball_config.get("patterns", [])
        base_path = selective_tarball_config.get("base_path", "")
        logger.info("Selective tarball enabled - skipping patterns: %s in base_path: %s", patterns, base_path)
    if exclude_patterns:
        logger.info("Exclude patterns enabled for continuous upload: %s", exclude_patterns)
    if retain_patterns:
        logger.info("Retain patterns enabled - files matching these will be kept until job completion: %s",
                    retain_patterns)
    file_last_modified = {}

    # Initialize file_last_modified with files that are already part of results dir
    for root, _, files in os.walk(local_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_last_modified[file_path] = os.path.getmtime(file_path)

    try:
        while True:
            if exit_event.is_set():
                # Check if this is a graceful pause (signal file exists) or normal completion
                signal_file = os.path.join(local_path, ".graceful_termination_signal")
                # Read signal file content if it exists to determine if it's from early stop
                is_early_stop_signal = False
                if os.path.exists(signal_file):
                    try:
                        with open(signal_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Early stop signals have format "early_stop_epoch_N"
                            is_early_stop_signal = "early_stop_epoch_" in content
                    except Exception:
                        pass

                # If signal exists but it's just an early stop marker (not graceful pause),
                # we should still do final upload since training completed
                if os.path.exists(signal_file) and not is_early_stop_signal:
                    # True graceful pause: exit immediately, snapshot upload will handle remaining files
                    logger.info("[CLOUD-HANDLERS][UTILS] Continuous upload monitor stopped by graceful pause signal")
                else:
                    # Normal completion or early stop completion: do final upload to ensure all files are uploaded
                    if is_early_stop_signal:
                        logger.info(
                            "[CLOUD-HANDLERS][UTILS] Continuous upload monitor stopped "
                            "(training completed with early stop), performing final upload"
                        )
                    else:
                        logger.info(
                            "[CLOUD-HANDLERS][UTILS] Continuous upload monitor stopped, "
                            "performing final upload"
                        )

                    # Count ALL files that need to be uploaded in this final pass
                    files_to_upload = []
                    for root, _, files in os.walk(local_path):
                        for filename in files:
                            file_path = os.path.join(root, filename)
                            current_last_modified = get_file_modification_time(file_path)

                            if not current_last_modified:
                                continue

                            # Skip checkpoint and tmp files
                            if "checkpoint-" in file_path or "tmp" in file_path:
                                continue

                            # Check if file needs upload (new or modified)
                            if (file_path not in file_last_modified or
                                    current_last_modified > file_last_modified[file_path]):

                                # Check exclude patterns
                                should_exclude = False
                                if exclude_patterns:
                                    rel_path = os.path.relpath(file_path, local_path)
                                    for pattern in exclude_patterns:
                                        try:
                                            if re.search(pattern, rel_path) or re.search(pattern, filename):
                                                should_exclude = True
                                                break
                                        except re.error:
                                            pass

                                if should_exclude:
                                    continue

                                # Check if file will be in selective tarball
                                if should_skip_file_for_tarball(file_path, local_path, selective_tarball_config):
                                    continue

                                files_to_upload.append(file_path)

                    # Create final progress tracker with accurate file count and size
                    final_progress_tracker = None
                    if files_to_upload:
                        from nvidia_tao_core.microservices.handlers.cloud_handlers.progress_tracker import (
                            ProgressTracker
                        )
                        total_size_mb = sum(
                            os.path.getsize(f) / (1024 * 1024) for f in files_to_upload if os.path.exists(f))
                        logger.info("Final upload: %d files (%.1f MB) to upload", len(files_to_upload), total_size_mb)

                        final_progress_tracker = ProgressTracker(
                            "upload",
                            total_files=len(files_to_upload),
                            total_size_mb=total_size_mb,
                            send_callbacks=True  # Enable callbacks for post-job uploads
                        )
                    else:
                        logger.info("No files need final upload")

                    # Perform final upload with progress tracker (no exit_event check during final upload)
                    # Set retain_patterns=None to allow removal of all large files after final upload
                    upload_files(local_path, cloud_storage, file_last_modified,
                                 selective_tarball_config, exclude_patterns,
                                 progress_tracker=final_progress_tracker, exit_event=None,
                                 retain_patterns=None)

                    # Complete the progress tracker
                    if final_progress_tracker:
                        final_progress_tracker.complete()

                    seek_position = send_logs_to_server(seek_position)

                break
            # During continuous monitoring, pass retain_patterns to keep files until job completes
            upload_files(local_path, cloud_storage, file_last_modified, selective_tarball_config,
                         exclude_patterns, progress_tracker=None, exit_event=exit_event,
                         retain_patterns=retain_patterns)

            seek_position = send_logs_to_server(seek_position)
            exit_event.wait(30)  # Wakes immediately when exit_event is set

    except (KeyboardInterrupt, SystemExit, Exception):
        logger.error("traceback: %s", traceback.format_exc())
        exit_event.set()


def get_file_path_from_cloud_string(value):
    """Get the cloud storage class object from the value

    Args:
        value (str): Path string with protocol (e.g., aws://bucket/path, lustre://path)

    Returns:
        tuple: (csp_provider, bucket_name, cloud_file_path)
               For local mounts (lustre, file, local), bucket_name is empty string
    """
    csp_provider = value.split(":")[0].lower()

    # Normalize s3:// to aws://
    if csp_provider == "s3":
        csp_provider = "aws"

    # Handle local filesystem protocols (no bucket concept)
    if csp_provider in ['lustre', 'file', 'local', 'slurm']:
        # Format: protocol://path
        cloud_file_path = value.split("://", 1)[1] if "://" in value else value

        # For SLURM, skip the "None" placeholder bucket name
        # Format: slurm://None/actual/path -> /actual/path
        if csp_provider == 'slurm' and cloud_file_path.startswith('None/'):
            cloud_file_path = cloud_file_path[5:]  # Skip "None/"

        # Ensure path starts with /
        if not cloud_file_path.startswith('/'):
            cloud_file_path = '/' + cloud_file_path
        return csp_provider, '', cloud_file_path

    # Handle cloud protocols with bucket (aws, azure, seaweedfs)
    bucket_name = value.split("//")[1].split("/")[0]
    cloud_file_path = value[value.find(bucket_name) + len(bucket_name):]
    return csp_provider, bucket_name, cloud_file_path


def get_cloud_storage_class_object(cloud_data, cloud_string):
    """Initalize Apache LibCloud class"""
    csp_provider, bucket_name, cloud_file_path = get_file_path_from_cloud_string(cloud_string)

    # Handle local filesystem and SLURM protocols (no cloud credentials needed)
    if csp_provider in ['lustre', 'file', 'local', 'slurm']:
        # For SLURM, files are directly accessible on Lustre mount - no cloud storage needed
        # For other local protocols (lustre, file, local), also skip cloud storage
        # These are used when jobs run on nodes with direct filesystem access
        while cloud_file_path.find("//") != -1:
            cloud_file_path = cloud_file_path.replace("//", "/")
        return None, cloud_file_path

    # Handle cloud protocols with credentials
    account_name = cloud_data[csp_provider][bucket_name].get("account_name")
    access_key = cloud_data[csp_provider][bucket_name].get("access_key")
    secret_key = cloud_data[csp_provider][bucket_name].get("secret_key")
    region = cloud_data[csp_provider][bucket_name].get("cloud_region")
    endpoint_url = cloud_data[csp_provider][bucket_name].get("endpoint_url")
    if account_name:
        csp_provider = "azure"
        access_key = account_name

    cloud_storage = initialize_cloud_storage(
        cloud_type=csp_provider,
        bucket_name=bucket_name,
        region=region,
        access_key=access_key,
        secret_key=secret_key,
        endpoint_url=endpoint_url
    )
    while cloud_file_path.find("//") != -1:
        cloud_file_path = cloud_file_path.replace("//", "/")
    return cloud_storage, cloud_file_path


def download_from_user_storage(
    cloud_storage=None, job_id="", cloud_data={}, value="", dictionary={}, key="",
    preserve_source_path=False, reset_value=False, progress_tracker=None, exclude_filenames=None
):
    """Download a file/folder from user storage"""
    try:
        # Check if this is a local filesystem path (SLURM, Lustre, file, local)
        # These paths are directly accessible - no download needed
        if value.startswith(("slurm://", "lustre://", "file://", "local://")):
            protocol, _, cloud_file_path = get_file_path_from_cloud_string(value)
            # Clean up double slashes
            while cloud_file_path.find("//") != -1:
                cloud_file_path = cloud_file_path.replace("//", "/")

            logger.info(f"{protocol.upper()} path detected - skipping download, using direct path: {cloud_file_path}")

            # Check if it's a tar file that needs extraction
            if cloud_file_path.endswith(".tar.gz") or cloud_file_path.endswith(".tar"):
                extraction_dir = cloud_file_path.replace(".tar.gz", "").replace(".tar", "")

                # Check if tar file exists and extracted directory doesn't
                if os.path.exists(cloud_file_path) and not os.path.exists(extraction_dir):
                    logger.info(f"Extracting SLURM tar file (keeping original): {cloud_file_path}")
                    _extract_images(cloud_file_path, os.path.dirname(cloud_file_path), remove_tar=False)
                    logger.info(f"Extracted to: {extraction_dir}")
                elif os.path.exists(extraction_dir):
                    logger.info(f"SLURM extracted directory already exists: {extraction_dir}")
                else:
                    logger.warning(f"SLURM tar file not found: {cloud_file_path}")

                # Return the extracted directory path
                if reset_value and dictionary and key:
                    dictionary[key] = extraction_dir
                return extraction_dir

            # Not a tar file - return as-is
            if reset_value and dictionary and key:
                dictionary[key] = cloud_file_path
            return cloud_file_path

        if not cloud_storage:
            cloud_storage, cloud_file_path = get_cloud_storage_class_object(cloud_data, value)
        else:
            cloud_file_path = value

        local_path_of_dataset_file = f"/results/{job_id}/{cloud_file_path}"
        if preserve_source_path:
            local_path_of_dataset_file = cloud_file_path
        if reset_value:
            # Update the dictionary value with the local path
            if dictionary and key:
                dictionary[key] = local_path_of_dataset_file.replace(".tar.gz", "")
        destination_path = local_path_of_dataset_file
        if cloud_file_path.startswith("/"):
            cloud_file_path = cloud_file_path[1:]

        # Create destination directory
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        if cloud_storage.is_file(cloud_file_path):
            cloud_storage.download_file(cloud_file_path, destination_path, progress_tracker=progress_tracker)
            if cloud_file_path.endswith(".tar") or cloud_file_path.endswith(".tar.gz"):
                _extract_images(destination_path, os.path.dirname(destination_path))
        else:
            cloud_storage.download_folder(cloud_file_path, destination_path, progress_tracker=progress_tracker,
                                          exclude_filenames=exclude_filenames)
            for root, _, files in os.walk(destination_path):
                for file in files:
                    abs_filepath = os.path.join(root, file)
                    if abs_filepath.endswith(".tar") or abs_filepath.endswith(".tar.gz"):
                        _extract_images(abs_filepath, os.path.dirname(abs_filepath))

        logger.info("Downloaded: {}".format(cloud_file_path))  # noqa pylint: disable=C0209
        return local_path_of_dataset_file.replace(".tar.gz", "")
    except Exception as e:
        logger.error("Error downloading cloud file: %s", str(e))
        logger.error(traceback.format_exc())
        callback_data = get_internal_job_status_update_data(
            automl_experiment_number=os.getenv("AUTOML_EXPERIMENT_NUMBER", "0"),
            message=f"Error downloading cloud file {value}"
        )
        status_callback(callback_data)
        raise e


def download_files_from_cloud(
    cloud_data,
    dictionary,
    key,
    value,
    job_id,
    network_arch,
    ngc_key,
    reset_value=False,
    preserve_source_path=False,
    progress_tracker=None
):
    """Based on the cloud dype, download the file"""
    if "'link': 'https://" in value:
        try:
            https_dictionary = ast.literal_eval(value)
            link = https_dictionary.get("link", "")
            destination_path = https_dictionary.get("destination_path", f"/ptm/{network_arch}/download")
            destination_folder = os.path.dirname(destination_path)
            download_from_https_link(link, destination_folder)
            dictionary[key] = destination_path
            return destination_path
        except Exception as e:
            logger.error("Error downloading public hosted file: %s", str(e))
            logger.error(traceback.format_exc())
            callback_data = get_internal_job_status_update_data(
                automl_experiment_number=os.getenv("AUTOML_EXPERIMENT_NUMBER", "0"),
                message=f"Error downloading public hosted file {value}"
            )
            status_callback(callback_data)
            raise e

    if value.startswith("ngc://"):
        try:
            if not ngc_key:
                cleanup_cuda_contexts()
                raise ValueError("NGC Personal key has not been provided")
            ngc_model = value.split("ngc://")[-1]
            org, team, model_name, model_version = split_ngc_path(ngc_model)
            destination_path = f"/ptm/{org}/{team}/{model_name}/{model_version}/model"

            total_size_bytes, _ = get_model_size_info(ngc_model, ngc_key)
            total_size_mb = total_size_bytes / (1024 * 1024) if total_size_bytes else 0

            # Update progress tracker with PTM file info
            if progress_tracker:
                # Include model name and organization for better identification
                display_name = f"NGC: {model_name} ({org}/{team})"
                progress_tracker.start_file_download(
                    file_name=display_name,
                    file_size_mb=total_size_mb
                )

            logger.info(f"Starting NGC model download: {ngc_model} ({total_size_mb:.1f} MB)")

            # Download the model with progress monitoring integrated into main tracker
            if progress_tracker:
                from nvidia_tao_core.microservices.handlers.cloud_handlers.ptm_download_monitor import (
                    download_ptm_with_progress
                )
                display_name = f"NGC: {model_name} ({org}/{team})"
                success = download_ptm_with_progress(
                    download_ngc_model, destination_path, display_name, total_size_mb,
                    progress_tracker, ngc_model, destination_path, ngc_key
                )
            else:
                # Fallback to direct download without progress monitoring
                success = download_ngc_model(ngc_model, destination_path, ngc_key)

            if not success:
                cleanup_cuda_contexts()
                raise ValueError("Unable to download the PTM")

            logger.info(f"NGC model download completed: {ngc_model}")

            ptm_path = search_for_ptm(destination_path, network_arch, key)
            dictionary[key] = ptm_path
            return ptm_path
        except Exception as e:
            logger.error("Error downloading NGC model: %s", str(e))
            logger.error(traceback.format_exc())
            callback_data = get_internal_job_status_update_data(
                automl_experiment_number=os.getenv("AUTOML_EXPERIMENT_NUMBER", "0"),
                message=f"Error downloading NGC model {value}"
            )
            status_callback(callback_data)
            raise e

    if value.startswith("hf_model://"):
        try:
            huggingface_model = value.split("hf_model://")[-1]
            destination_path = "/ptm/huggingface_models"

            # Get model size for progress tracking
            from nvidia_tao_core.microservices.handlers.cloud_handlers.huggingface import (
                get_huggingface_model_size_info, download_huggingface_model
            )

            hf_token = os.getenv("HF_TOKEN", "")
            total_size_bytes, _ = get_huggingface_model_size_info(huggingface_model, token=hf_token)
            total_size_mb = total_size_bytes / (1024 * 1024) if total_size_bytes else 0

            # Update progress tracker with PTM file info
            if progress_tracker:
                # Include model name for better identification
                display_name = f"HF: {huggingface_model}"
                progress_tracker.start_file_download(
                    file_name=display_name,
                    file_size_mb=total_size_mb
                )

            logger.info(f"Starting HuggingFace model download: {huggingface_model} ({total_size_mb:.1f} MB)")

            # Download the model with progress monitoring integrated into main tracker
            if progress_tracker:
                from nvidia_tao_core.microservices.handlers.cloud_handlers.ptm_download_monitor import (
                    download_ptm_with_progress
                )
                display_name = f"HF: {huggingface_model}"
                download_ptm_with_progress(
                    download_huggingface_model, destination_path, display_name, total_size_mb,
                    progress_tracker, huggingface_model, destination_path, hf_token
                )
            else:
                # Fallback to direct download without progress monitoring
                download_huggingface_model(huggingface_model, destination_path, hf_token)

            logger.info(f"HuggingFace model download completed: {huggingface_model}")

            dictionary[key] = destination_path
            return destination_path
        except Exception as e:
            logger.error("Error downloading Hugging Face model: %s", str(e))
            logger.error(traceback.format_exc())
            callback_data = get_internal_job_status_update_data(
                automl_experiment_number=os.getenv("AUTOML_EXPERIMENT_NUMBER", "0"),
                message=f"Error downloading Hugging Face model {value}"
            )
            status_callback(callback_data)
            raise e

    if "://" in value:
        return download_from_user_storage(
            cloud_data=cloud_data,
            value=value,
            job_id=job_id,
            dictionary=dictionary,
            key=key,
            preserve_source_path=preserve_source_path,
            reset_value=reset_value,
            progress_tracker=progress_tracker
        )

    # Handle plain absolute paths for tar files (SLURM case)
    # This runs in the job container where Lustre is mounted, so we can extract if needed
    if isinstance(value, str) and os.path.isabs(value):
        if value.endswith(".tar.gz") or value.endswith(".tar"):
            extraction_dir = value.replace(".tar.gz", "").replace(".tar", "")

            # Check if tar file exists and needs extraction
            if os.path.exists(value):
                if not os.path.exists(extraction_dir):
                    try:
                        _extract_images(value, os.path.dirname(value), remove_tar=False)
                    except Exception as e:
                        logger.error(f"Failed to extract {value}: {e}")
                        return value
                else:
                    logger.info(f"Extracted directory already exists: {extraction_dir}")
            else:
                # Tar file doesn't exist, but maybe directory does
                if os.path.exists(extraction_dir):
                    logger.info(f"Tar file not found, but directory exists: {extraction_dir}")
                else:
                    logger.warning(f"Neither tar file nor extracted directory found: {value}")
                    return value

            # Update dictionary to use extracted directory
            if reset_value and dictionary and key:
                dictionary[key] = extraction_dir

            return extraction_dir

    return None


def count_files_in_spec(data, cloud_data=None):
    """Count the total number of files that will be downloaded from a spec.

    Args:
        data (dict): The spec data structure to analyze
        cloud_data (dict): Cloud metadata for accessing cloud storage

    Returns:
        int: Total number of downloadable files found in the spec

    Note:
        - For cloud storage folders, lists actual files in the folder
        - HuggingFace models (hf_model://) are counted as 1 logical download operation
    """
    file_count = 0

    if isinstance(data, dict):
        for _, value in data.items():
            if isinstance(value, dict):
                file_count += count_files_in_spec(value, cloud_data)
            elif isinstance(value, list):
                for list_element in value:
                    if isinstance(list_element, str):
                        # Check if this looks like a downloadable file path
                        if _is_downloadable_path(list_element):
                            # Count actual files (especially for cloud folders)
                            file_count += _count_downloadable_path(list_element, cloud_data)
                    elif isinstance(list_element, dict):
                        file_count += count_files_in_spec(list_element, cloud_data)
            elif isinstance(value, str):
                # Check if this looks like a downloadable file path
                if _is_downloadable_path(value):
                    # Count actual files (especially for cloud folders)
                    file_count += _count_downloadable_path(value, cloud_data)

    return file_count


def _count_downloadable_path(path, cloud_data=None):
    """Count how many files will be downloaded for a given path.

    Args:
        path (str): Path to check (could be file, folder, or HF model)
        cloud_data (dict): Cloud metadata for accessing cloud storage

    Returns:
        int: Number of files that will be downloaded (1 for single file, N for folders)
    """
    try:
        # Handle HuggingFace models
        if path.startswith("hf_model://"):
            # HF models downloaded as single operation via snapshot_download
            # Even though they contain multiple files, we count as 1 logical operation
            # because we can't track per-file progress with snapshot_download
            return 1

        # Handle cloud storage paths (s3://, azure://, etc.)
        if "://" in path:
            try:
                # Get cloud storage object
                cloud_storage, cloud_file_path = get_cloud_storage_class_object(cloud_data, path)

                if cloud_storage:
                    # Check if it's a folder or file
                    if cloud_storage.is_file(cloud_file_path):
                        return 1
                    # It's a folder - list actual files
                    files, _ = cloud_storage.list_files_in_folder(cloud_file_path)  # Returns (files, details) tuple
                    file_count = len(files) if files else 1
                    if file_count > 1:
                        logger.info(f"Cloud folder {path} contains {file_count} files")
                    return file_count
            except Exception as e:
                logger.warning(f"Could not list files in cloud path {path}: {e}")
                return 1

        # Default: treat as single file
        return 1

    except Exception as e:
        logger.warning(f"Error counting files for {path}: {e}")
        return 1


def _is_downloadable_path(path):
    """Check if a path string represents a downloadable file.

    Args:
        path (str): Path string to check

    Returns:
        bool: True if the path appears to be downloadable
    """
    if not isinstance(path, str) or not path.strip():
        return False

    # Consider it downloadable if it contains cloud storage indicators or file extensions
    downloadable_indicators = [
        "://",  # Cloud storage URLs
        ".tar", ".tar.gz", ".zip",  # Archive files
        ".pth", ".ckpt", ".h5", ".hdf5", ".tlt",  # Model files
        ".json", ".yaml", ".yml",  # Config files
        ".txt", ".csv",  # Data files
        "ngc://",  # NGC paths
        "hf_model://",  # Hugging Face models
    ]

    return any(indicator in path.lower() for indicator in downloadable_indicators)


def calculate_total_download_size(cloud_data, data, job_id):
    """Calculate total size of all downloadable files in the spec.

    Args:
        cloud_data: Cloud storage metadata
        data: Spec data structure to analyze
        job_id: Job identifier

    Returns:
        float: Total size in MB of all downloadable files
    """
    total_size_mb = 0.0

    def _get_file_size(file_path):
        """Get size of a single file."""
        try:
            # Handle NGC models
            if file_path.startswith("ngc://"):
                ngc_key = os.getenv("TAO_USER_KEY")
                if ngc_key:
                    ngc_model = file_path.split("ngc://")[-1]
                    total_size_bytes, _ = get_model_size_info(ngc_model, ngc_key)
                    if total_size_bytes:
                        return total_size_bytes / (1024 * 1024)  # Convert to MB
                return 0.0

            # Handle Hugging Face models
            if file_path.startswith("hf_model://"):
                from nvidia_tao_core.microservices.handlers.cloud_handlers.huggingface import (
                    get_huggingface_model_size_info
                )
                hf_model = file_path.split("hf_model://")[-1]
                hf_token = os.getenv("HF_TOKEN", "")
                total_size_bytes, _ = get_huggingface_model_size_info(hf_model, token=hf_token)
                if total_size_bytes:
                    return total_size_bytes / (1024 * 1024)  # Convert to MB
                return 0.0

            # Handle cloud storage files
            if "://" in file_path and _is_downloadable_path(file_path):
                # Try to get cloud file size
                cloud_storage, cloud_file_path = get_cloud_storage_class_object(cloud_data, file_path)
                if cloud_storage:
                    if cloud_storage.is_file(cloud_file_path):
                        # Single file - get its size
                        full_path = cloud_storage.root + cloud_file_path.strip('/')
                        try:
                            file_info = cloud_storage.fs.info(full_path)
                            return file_info.get('size', 0) / (1024 * 1024)  # Convert to MB
                        except Exception:
                            logger.warning(f"Could not get size for {file_path}")
                            return 0.0
                    else:
                        # Folder - sum sizes of all files inside
                        try:
                            # Returns (files, details) tuple
                            files, _ = cloud_storage.list_files_in_folder(cloud_file_path)
                            if files:
                                folder_size_mb = 0.0
                                for file in files:
                                    try:
                                        # list_files_in_folder returns relative paths, construct full path
                                        file_full_path = cloud_storage.root + file.strip('/')
                                        file_info = cloud_storage.fs.info(file_full_path)
                                        file_size_mb = file_info.get('size', 0) / (1024 * 1024)
                                        folder_size_mb += file_size_mb
                                    except Exception:
                                        continue
                                if folder_size_mb > 0:
                                    logger.info(f"Cloud folder {file_path} total size: {folder_size_mb:.1f} MB")
                                return folder_size_mb
                        except Exception as e:
                            logger.warning(f"Could not calculate folder size for {file_path}: {e}")
                            return 0.0
        except Exception:
            logger.warning(f"Could not determine size for {file_path}")
        return 0.0

    def _process_spec_for_size(spec_data):
        """Recursively process spec to calculate total size."""
        nonlocal total_size_mb

        if isinstance(spec_data, dict):
            for _, value in spec_data.items():
                if isinstance(value, dict):
                    _process_spec_for_size(value)
                elif isinstance(value, list):
                    for list_element in value:
                        if isinstance(list_element, str) and _is_downloadable_path(list_element):
                            file_size = _get_file_size(list_element)
                            total_size_mb += file_size
                        elif isinstance(list_element, dict):
                            _process_spec_for_size(list_element)
                elif isinstance(value, str) and _is_downloadable_path(value):
                    file_size = _get_file_size(value)
                    total_size_mb += file_size

    try:
        _process_spec_for_size(data)
        logger.info(f"Calculated total download size: {total_size_mb:.1f} MB")
        return total_size_mb
    except Exception as e:
        logger.warning(f"Error calculating total download size: {e}")
        return 0.0


def download_files_from_spec(
    cloud_data,
    data,
    job_id,
    network_arch=None,
    ngc_key=None,
    reprocess_files=None,
    preserve_source_path=False,
    preserve_source_path_params=None,
    current_path="",
    progress_tracker=None
):
    """Recursively download files from a nested dictionary."""
    if preserve_source_path_params is None:
        preserve_source_path_params = set()
    if isinstance(data, dict):
        for key, value in data.items():
            # Build the current parameter path
            new_path = f"{current_path}.{key}" if current_path else key
            if isinstance(value, dict):
                download_files_from_spec(
                    cloud_data,
                    value,
                    job_id,
                    network_arch=network_arch,
                    ngc_key=ngc_key,
                    reprocess_files=reprocess_files,
                    preserve_source_path=preserve_source_path,
                    preserve_source_path_params=preserve_source_path_params,
                    current_path=new_path,
                    progress_tracker=progress_tracker
                )
            elif isinstance(value, list):
                override_list = []
                for list_element in value:
                    if isinstance(list_element, str):
                        # Check if this parameter should preserve source path
                        param_preserve_source_path = preserve_source_path or new_path in preserve_source_path_params
                        override_value = download_files_from_cloud(
                            cloud_data,
                            data,
                            key,
                            list_element,
                            job_id,
                            network_arch,
                            ngc_key,
                            preserve_source_path=param_preserve_source_path,
                            progress_tracker=progress_tracker
                        )
                        if not override_value:
                            override_value = list_element
                        if (reprocess_files is not None and override_value and
                                (list_element.endswith(".yaml") or list_element.endswith(".json"))):
                            reprocess_files.append(override_value)
                        override_list.append(override_value)
                    elif isinstance(list_element, dict):
                        override_dict = {}
                        for list_dict_key, list_dict_value in list_element.items():
                            if isinstance(list_dict_value, str):
                                # Check if this parameter should preserve source path
                                param_preserve_source_path = (
                                    preserve_source_path or new_path in preserve_source_path_params
                                )
                                override_value = download_files_from_cloud(
                                    cloud_data,
                                    data,
                                    key,
                                    list_dict_value,
                                    job_id,
                                    network_arch,
                                    ngc_key,
                                    preserve_source_path=param_preserve_source_path,
                                    progress_tracker=progress_tracker
                                )
                                if (reprocess_files is not None and override_value and
                                        (list_dict_value.endswith(".yaml") or list_dict_value.endswith(".json"))):
                                    reprocess_files.append(override_value)
                                if not override_value:
                                    override_value = list_dict_value
                            else:
                                override_value = list_dict_value
                            override_dict[list_dict_key] = override_value
                        override_list.append(override_dict)
                    else:
                        override_list.append(list_element)
                data[key] = override_list
            else:
                if isinstance(value, str):
                    # Check if this parameter should preserve source path
                    param_preserve_source_path = preserve_source_path or new_path in preserve_source_path_params
                    override_value = download_files_from_cloud(
                        cloud_data,
                        data,
                        key,
                        value,
                        job_id,
                        network_arch,
                        ngc_key,
                        reset_value=True,
                        preserve_source_path=param_preserve_source_path,
                        progress_tracker=progress_tracker
                    )
                    if (reprocess_files is not None and override_value and
                            (value.endswith(".yaml") or value.endswith(".json"))):
                        reprocess_files.append(override_value)

                    # File was successfully downloaded, progress already updated by cloud_storage


def get_results_cloud_data(cloud_data, spec_data, dest_dir=None):
    """Obtain the CloudStorage instance for uploading the results.

    Args:
    cloud_data: Cloud storage metadata.
    spec_data: Model spec.
    dest_dir: Directory where the results directorys is present.

    Returns:
    cloud_storage: CloudStorage instance.
    spec_data: Updated model spec.
    """
    results_dir = spec_data["results_dir"]
    if "://" in results_dir:
        cloud_storage, cloud_file_path = get_cloud_storage_class_object(cloud_data, results_dir)
        spec_data["results_dir"] = cloud_file_path
        return cloud_storage, spec_data

    # Check if results_dir is an absolute path (e.g., Slurm Lustre paths)
    if os.path.isabs(results_dir):
        # Use the absolute path as-is, don't prepend dest_dir
        return None, spec_data

    if not dest_dir:
        cleanup_cuda_contexts()
        raise ValueError("Destination directory is not provided")
    spec_data["results_dir"] = f'{dest_dir}/{spec_data["results_dir"]}'
    return None, spec_data


def create_tarball(source_dir, tarball_path, exclude_paths=None, exclude_patterns=None):
    """Create a tarball from the source directory.

    Args:
        source_dir (str): Directory to tarball
        tarball_path (str): Path where the tarball will be created
        exclude_paths (set, optional): Set of relative paths to exclude from tarball
        exclude_patterns (list, optional): List of regex patterns to exclude files from tarball

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        exclusion_info = []
        if exclude_paths:
            exclusion_info.append(f"{len(exclude_paths)} pre-existing items")
        if exclude_patterns:
            exclusion_info.append(f"files matching patterns: {exclude_patterns}")
        if exclusion_info:
            logger.info("Creating tarball excluding %s: %s from source: %s",
                        ", ".join(exclusion_info), tarball_path, source_dir)
        else:
            logger.info("Creating tarball: %s from source: %s", tarball_path, source_dir)

        with tarfile.open(tarball_path, 'w:gz') as tar:
            if exclude_paths or exclude_patterns:
                # Walk through directory and selectively add files
                files_added = 0
                files_excluded = 0

                for root, dirs, files in os.walk(source_dir):
                    # Process files
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, source_dir)
                        # Check if file should be excluded
                        should_exclude = False
                        # Check exclude_paths
                        if exclude_paths and rel_path in exclude_paths:
                            should_exclude = True
                        # Check exclude_patterns
                        if exclude_patterns and not should_exclude:
                            for pattern in exclude_patterns:
                                try:
                                    if re.search(pattern, rel_path) or re.search(pattern, file):
                                        should_exclude = True
                                        break
                                except re.error:
                                    logger.warning("Invalid regex pattern '%s', skipping", pattern)

                        if not should_exclude:
                            tar.add(file_path, arcname=rel_path)
                            files_added += 1
                        else:
                            files_excluded += 1

                    # Process directories (only add empty directories that weren't in exclusion set)
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        rel_path = os.path.relpath(dir_path, source_dir)

                        # Check if directory is empty and wasn't in exclusion set
                        # Empty directory not in exclusion set
                        if (not exclude_paths or rel_path not in exclude_paths) and not os.listdir(dir_path):
                            tar.add(dir_path, arcname=rel_path)

                logger.info("Tarball created successfully: %s (%d files added, %d files excluded)",
                            tarball_path, files_added, files_excluded)
            else:
                # Original behavior - add entire directory
                tar.add(source_dir, arcname=os.path.basename(source_dir))
                logger.info("Tarball created successfully: %s", tarball_path)

        return True
    except Exception as e:
        logger.error("Error creating tarball: %s", str(e))
        logger.error("Traceback: %s", traceback.format_exc())
        return False


@master_node_only
def upload_tarball_to_cloud(cloud_storage, tarball_path, remove_after_upload=True):
    """Upload a tarball to cloud storage.

    Args:
        cloud_storage: CloudStorage instance
        tarball_path (str): Path to the tarball to upload
        remove_after_upload (bool): Whether to remove the local tarball after upload

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Uploading tarball to cloud: %s", tarball_path)

        # Post-job tarball upload - enable status callbacks for visibility
        cloud_storage.upload_file(tarball_path, tarball_path, send_status_callbacks=True)
        logger.info("Tarball uploaded successfully: %s", tarball_path)

        if remove_after_upload and os.path.exists(tarball_path):
            os.remove(tarball_path)
            logger.info("Local tarball removed: %s", tarball_path)

        return True
    except Exception as e:
        logger.error("Error uploading tarball: %s", str(e))
        logger.error("Traceback: %s", traceback.format_exc())
        return False


def cleanup_cuda_contexts():
    """Clean up any stale CUDA contexts.

    Call this function in cleanup routines or when throwing exceptions.
    """
    try:
        # Check if PyCUDA is imported in this environment
        pycuda_imported = "pycuda" in sys.modules
        if pycuda_imported:
            import pycuda.driver as cuda

            # Clean up any active contexts
            if cuda.Context.get_current() is not None:
                try:
                    # Pop the current context
                    logger.info("Found active CUDA context. Cleaning up...")
                    cuda.Context.pop()
                    logger.info("Active CUDA context cleaned up!")
                except Exception as e:
                    logger.warning(f"Error cleaning up CUDA context: {str(e)}")
    except Exception as e:
        logger.warning(f"Unexpected error during CUDA cleanup: {str(e)}")
