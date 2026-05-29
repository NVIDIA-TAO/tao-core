# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Original source taken from https://github.com/NVIDIA/NeMo

"""Job queue handling."""


import argparse
import json
import glob
import importlib
import os
import sys
import threading
import time
import traceback
import uuid
import yaml
import logging
import tarfile
from tqdm import tqdm

from nvidia_tao_core.api_utils import module_utils
from nvidia_tao_core.api_utils.entrypoint_mimicker import vlm_entrypoint
from nvidia_tao_core.microservices.handlers.cloud_handlers.utils import (
    download_files_from_spec,
    count_files_in_spec,
    calculate_total_download_size,
    get_results_cloud_data,
    monitor_and_upload,
    cleanup_cuda_contexts,
    create_tarball,
    upload_tarball_to_cloud,
    upload_files,
    download_from_user_storage
)
from nvidia_tao_core.microservices.handlers.cloud_handlers.progress_tracker import ProgressTracker
import nvidia_tao_core.loggers.logging as status_logging
from nvidia_tao_core.api_utils.module_utils import entrypoint_paths, entry_points
from nvidia_tao_core.microservices.utils.cloud_utils import create_cs_instance, NUM_RETRY
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    get_handler_id,
    get_handler_kind,
    get_handler_metadata,
    save_dnn_status
)
from nvidia_tao_core.microservices.utils.core_utils import (
    safe_load_file,
    safe_dump_file,
    read_network_config,
    get_spec_backend_info
)
from nvidia_tao_core.microservices.utils.specs_utils import json_to_kitti, json_to_yaml, json_to_toml
from nvidia_tao_core.microservices.utils.handler_utils import write_nested_dict
# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)

# Spec backend to conversion functions mapping
SPEC_BACKEND_TO_FUNCTIONS = {
    "protobuf": json_to_kitti.kitti,
    "yaml": json_to_yaml.yml,
    "toml": json_to_toml.toml_format
}

# IS_MASTER: Only true for rank-0 task on rank-0 node
# For SLURM: Check both NODE_RANK (which node) and SLURM_LOCALID (which task on that node)
# This ensures only ONE task per node does shared operations like downloads
IS_MASTER = int(os.environ.get("NODE_RANK", 0)) == 0
IS_LOCAL_MASTER = int(os.environ.get("SLURM_LOCALID", 0)) == 0  # Master task on current node

# Graceful termination signal file name
GRACEFUL_TERMINATION_SIGNAL_FILE = ".graceful_termination_signal"


def prepare_data_before_job_run(job, docker_env_vars):
    """Prepare data before job run.

    For SLURM single-node multi-GPU jobs, only the local master task (SLURM_LOCALID=0)
    performs downloads. Other tasks wait for downloads to complete.

    Note: This issue is SLURM-specific because:
    - SLURM single-node: Sets --ntasks-per-node=N, launching N parallel container_handler instances
    - SLURM multi-node: Sets --ntasks=num_nodes, launching 1 task per node (already deduplicated)
    - Kubernetes/Docker: Launch 1 container → 1 container_handler.py → torchrun spawns workers internally
    """
    if docker_env_vars:
        os.environ.update(docker_env_vars)

    # Use TAO_API_RESULTS_DIR for SLURM compatibility, fallback to /results
    results_base = os.getenv('TAO_API_RESULTS_DIR', '/results')
    cloud_storage, specs = get_results_cloud_data(
        job.get("cloud_metadata"),
        job["specs"],
        f'{results_base}/{job["job_id"]}'
    )

    ngc_key = docker_env_vars.get("TAO_API_KEY")
    if not ngc_key:
        ngc_key = docker_env_vars.get("TAO_USER_KEY")

    # Create results directory (all tasks need this)
    os.makedirs(specs["results_dir"], exist_ok=True)
    reprocess_files = []

    # Check if this is the local master task (for multi-GPU SLURM jobs)
    # Only local master (SLURM_LOCALID=0) on each node should download files
    is_local_master = int(os.environ.get("SLURM_LOCALID", 0)) == 0

    if not is_local_master:
        logger.info(f"Non-master task (SLURM_LOCALID={os.environ.get('SLURM_LOCALID')}) - skipping downloads")
        logger.info("Waiting for master task to complete downloads...")

        # Wait for master task to finish downloads by checking for spec file
        spec_backend, file_extension = get_spec_backend_info(job["neural_network_name"])
        spec_path = os.path.join(specs["results_dir"], f"spec.{file_extension}")

        # Wait up to 10 minutes for downloads to complete
        max_wait_seconds = 600
        wait_interval = 5
        elapsed = 0

        while not os.path.exists(spec_path) and elapsed < max_wait_seconds:
            time.sleep(wait_interval)
            elapsed += wait_interval
            if elapsed % 30 == 0:  # Log every 30 seconds
                logger.info(f"Still waiting for downloads... ({elapsed}s elapsed)")

        if os.path.exists(spec_path):
            logger.info(f"Downloads complete, proceeding with training (waited {elapsed}s)")
        else:
            logger.warning(f"Timeout waiting for downloads after {max_wait_seconds}s")
            logger.warning("Proceeding anyway - downloads may still be in progress")

        return cloud_storage, specs, spec_path

    # This is the local master task - perform downloads
    node_rank = os.environ.get('NODE_RANK', 0)
    logger.info(f"Local master task (SLURM_LOCALID=0, NODE_RANK={node_rank}) - performing downloads")

    # Pop companion ONNX folder before counting/downloading so it can be handled separately
    # with status.json excluded — prevents the parent export job's status.json from being
    # mistaken as the current job's status (causing false "Done" during download).
    companion_onnx_folder = specs.pop("_companion_onnx_folder", None)

    # Count total files to download before starting
    logger.info("Analyzing spec for download requirements...")

    # Count files in main spec (pass cloud_metadata to count actual files in cloud folders)
    main_spec_files = count_files_in_spec(specs, cloud_data=job.get("cloud_metadata"))

    # Count additional downloads
    additional_downloads = specs.pop("additional_downloads", [])
    additional_files_count = len(additional_downloads) if additional_downloads else 0

    total_files_to_download = main_spec_files + additional_files_count

    if total_files_to_download > 0:
        logger.info("Found %d files to download before job launch:", total_files_to_download)
        if main_spec_files > 0:
            logger.info("  - Main spec files: %d", main_spec_files)
        if additional_files_count > 0:
            logger.info("  - Additional downloads: %d", additional_files_count)
    else:
        logger.info("No files to download - job will start immediately")

    # Handle additional downloads
    if additional_downloads:
        logger.info("Processing additional downloads: %s", additional_downloads)
        ContainerJobHandler._handle_additional_downloads(
            additional_downloads,
            job.get("cloud_metadata"),
            job["job_id"],
            job["neural_network_name"],
            ngc_key,
        )

    # Remove internal parameters that should not be passed to CLI
    preserve_source_path_params = specs.pop("preserve_source_path_params", set())
    if isinstance(preserve_source_path_params, list):
        preserve_source_path_params = set(preserve_source_path_params)

    # Download main spec files
    if main_spec_files > 0:
        logger.info("Downloading files from normal spec (preserve_source_path_params=%s)", preserve_source_path_params)

        # Calculate total size of all files upfront
        logger.info("Calculating total download size...")
        total_size_mb = calculate_total_download_size(
            cloud_data=job.get("cloud_metadata"),
            data=specs,
            job_id=job["job_id"]
        )

        # Create progress tracker for main spec downloads with known total size
        main_progress_tracker = ProgressTracker(
            "download",
            total_files=main_spec_files,
            total_size_mb=total_size_mb,  # Now we know the total size upfront
            send_callbacks=True
        )

        download_files_from_spec(
            cloud_data=job.get("cloud_metadata"),
            data=specs,
            job_id=job["job_id"],
            network_arch=job["neural_network_name"],
            ngc_key=ngc_key,
            reprocess_files=reprocess_files,
            preserve_source_path_params=preserve_source_path_params,
            progress_tracker=main_progress_tracker
        )

        main_progress_tracker.complete()
        logger.info("Main spec file downloads completed")
    else:
        logger.info("No files to download from main spec")

    custom_script = specs.pop("custom_script", None)

    # Download companion ONNX folder (e.g. CLIP large-model exports that produce model.onnx +
    # model.bin). Exclude status.json to prevent the parent export job's status from being
    # mistaken as the current job's status, which would cause a false "Done" during download.
    if companion_onnx_folder:
        logger.info("Downloading companion ONNX folder (excluding status.json): %s", companion_onnx_folder)
        download_from_user_storage(
            cloud_data=job.get("cloud_metadata"),
            value=companion_onnx_folder,
            job_id=job["job_id"],
            exclude_filenames=["status.json"],
        )
        logger.info("Companion ONNX folder download completed")

    # Save spec file with dynamic backend
    network_arch = job["neural_network_name"]
    spec_backend, file_extension = get_spec_backend_info(network_arch)
    spec_path = os.path.join(specs["results_dir"], f"spec.{file_extension}")

    if spec_backend == "yaml":
        # Use yaml format
        with open(spec_path, 'w+', encoding='utf-8') as spec_file:
            yaml.dump(specs, spec_file, default_flow_style=False)
    elif spec_backend in SPEC_BACKEND_TO_FUNCTIONS:
        # Use appropriate conversion function
        conversion_func = SPEC_BACKEND_TO_FUNCTIONS[spec_backend]
        converted_specs = conversion_func(specs)
        with open(spec_path, 'w+', encoding='utf-8') as spec_file:
            spec_file.write(converted_specs)
    else:
        # Fallback to yaml if unknown backend
        logger.warning(f"Unknown spec backend '{spec_backend}', falling back to yaml")
        with open(spec_path, 'w+', encoding='utf-8') as spec_file:
            yaml.dump(specs, spec_file, default_flow_style=False)

    if docker_env_vars.get("RECURSIVE_DATASET_FILE_DOWNLOAD", "False") == "True":
        logger.info("reprocess_files: %s", reprocess_files)
        if reprocess_files:
            for file_name in reprocess_files:
                file_type = file_name.split(".")[-1]
                reprocess_file_data = safe_load_file(file_name, file_type=file_type)
                if reprocess_file_data:
                    # Count files in reprocess data for progress tracking
                    reprocess_file_count = count_files_in_spec(reprocess_file_data)
                    if reprocess_file_count > 0:
                        logger.info(
                            "Reprocessing %s: found %d additional files to download",
                            file_name, reprocess_file_count
                        )

                        # Create progress tracker for reprocessing
                        reprocess_progress_tracker = ProgressTracker(
                            "download",
                            total_files=reprocess_file_count,
                            total_size_mb=0,
                            send_callbacks=True
                        )

                        download_files_from_spec(
                            cloud_data=job.get("cloud_metadata"),
                            data=reprocess_file_data,
                            job_id=job["job_id"],
                            network_arch=job["neural_network_name"],
                            ngc_key=ngc_key,
                            progress_tracker=reprocess_progress_tracker
                        )

                        reprocess_progress_tracker.complete()
                    else:
                        download_files_from_spec(
                            cloud_data=job.get("cloud_metadata"),
                            data=reprocess_file_data,
                            job_id=job["job_id"],
                            network_arch=job["neural_network_name"],
                            ngc_key=ngc_key,
                        )
                    if reprocess_file_data:
                        safe_dump_file(file_name, reprocess_file_data, file_type=file_type)

    # Add custom_script back to specs for vlm_entrypoint to handle
    if custom_script:
        specs["custom_script"] = custom_script

    return cloud_storage, specs, spec_path


class ContainerJobHandler:
    """Handler for processing jobs in a containerized environment."""

    @staticmethod
    def check_graceful_termination_signal(results_dir):
        """Check if a graceful termination signal file exists.

        Args:
            results_dir (str): Results directory to check for signal file

        Returns:
            bool: True if graceful termination signal exists, False otherwise
        """
        try:
            signal_file = os.path.join(results_dir, GRACEFUL_TERMINATION_SIGNAL_FILE)
            return os.path.exists(signal_file)
        except Exception as e:
            logger.error("Error checking graceful termination signal: %s", str(e))
            return False

    @staticmethod
    def write_graceful_termination_signal(job_id):
        """Write a graceful termination signal file.

        Args:
            job_id (str): Job ID being terminated (results_dir is inferred from
                TAO_API_RESULTS_DIR or /results/{job_id})

        Returns:
            bool: True if signal was written successfully, False otherwise
        """
        try:
            # Use TAO_API_RESULTS_DIR for SLURM compatibility, fallback to /results
            results_base = os.getenv('TAO_API_RESULTS_DIR', '/results')
            # Infer results directory from job_id
            results_dir = f"{results_base}/{job_id}"
            signal_file = os.path.join(results_dir, GRACEFUL_TERMINATION_SIGNAL_FILE)

            # Ensure results directory exists
            os.makedirs(results_dir, exist_ok=True)

            # Create signal file (content doesn't matter, just existence)
            with open(signal_file, 'w', encoding='utf-8') as f:
                f.write(job_id)
            logger.info("Graceful termination signal written for job %s at %s", job_id, signal_file)
            return True
        except Exception as e:
            logger.error("Error writing graceful termination signal: %s", str(e))
            logger.error("Traceback: %s", traceback.format_exc())
            return False

    @staticmethod
    def remove_graceful_termination_signal(results_dir):
        """Remove the graceful termination signal file.

        Args:
            results_dir (str): Results directory containing signal file
        """
        try:
            signal_file = os.path.join(results_dir, GRACEFUL_TERMINATION_SIGNAL_FILE)
            if os.path.exists(signal_file):
                os.remove(signal_file)
                logger.info("Graceful termination signal file removed")
        except Exception as e:
            logger.error("Error removing graceful termination signal: %s", str(e))

    @staticmethod
    def capture_directory_snapshot(directory):
        """Capture a snapshot of all files and directories in the given directory.

        Args:
            directory (str): Directory to snapshot

        Returns:
            set: Set of relative file paths from the directory
        """
        try:
            if not os.path.exists(directory):
                logger.warning("Directory does not exist for snapshot: %s", directory)
                return set()

            snapshot = set()
            for root, dirs, files in os.walk(directory):
                # Add all files
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, directory)
                    snapshot.add(rel_path)

                # Add all directories
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    rel_path = os.path.relpath(dir_path, directory)
                    snapshot.add(rel_path)

            logger.info("Captured snapshot of %d items in directory: %s", len(snapshot), directory)
            return snapshot

        except Exception as e:
            logger.error("Error capturing directory snapshot: %s", str(e))
            logger.error("Traceback: %s", traceback.format_exc())
            return set()

    @staticmethod
    def create_and_upload_tarball(results_dir, cloud_storage, job_id, action_name,
                                  exclude_snapshot=None, exclude_patterns=None):
        """Create a tarball of the results directory and upload it.

        Args:
            results_dir (str): Directory to tarball
            cloud_storage: CloudStorage instance for uploading
            job_id (str): Job ID for naming the tarball
            action_name (str): Action name for naming the tarball
            exclude_snapshot (set, optional): Set of relative paths to exclude from tarball
            exclude_patterns (list, optional): List of regex patterns to exclude files from tarball
        """
        try:
            tarball_name = f"{action_name}_results.tar.gz"
            tarball_path = os.path.join(results_dir, tarball_name)

            # Create tarball using utility function
            if create_tarball(results_dir, tarball_path, exclude_snapshot, exclude_patterns):
                # Upload tarball using utility function
                if cloud_storage and os.path.exists(tarball_path):
                    upload_tarball_to_cloud(cloud_storage, tarball_path, remove_after_upload=True)
                else:
                    logger.warning("No cloud storage configured, tarball created but not uploaded: %s", tarball_path)
            else:
                logger.error("Failed to create tarball for job %s action %s", job_id, action_name)

        except Exception as e:
            logger.error("Error in create_and_upload_tarball: %s", str(e))
            logger.error("Traceback: %s", traceback.format_exc())

    @staticmethod
    def setup_and_run(job, docker_env_vars, sync=False):
        """Setup and run a container job."""
        cloud_storage = None
        exit_event = None
        upload_thread = None
        status_logger = None

        try:
            # Setup cloud storage and specs

            cloud_storage, specs, spec_path = prepare_data_before_job_run(job, docker_env_vars)

            # Get upload strategy, exclude patterns, and retain patterns by reading network config directly
            network = docker_env_vars.get("ORCHESTRATION_API_NETWORK", job.get("neural_network_name", ""))
            action = docker_env_vars.get("ORCHESTRATION_API_ACTION", job.get("action_name", ""))
            retain_checkpoints_for_resume = (
                docker_env_vars.get("RETAIN_CHECKPOINTS_FOR_RESUME", "false").lower() == "true"
            )
            upload_strategy, exclude_patterns, retain_patterns = (
                ContainerJobHandler.get_upload_strategy_from_config(
                    network, action, retain_checkpoints_for_resume
                )
            )
            logger.info("Using upload strategy for %s %s: %s", network, action, upload_strategy)
            if exclude_patterns:
                logger.info("Excluding patterns for %s %s: %s", network, action, exclude_patterns)
            if retain_patterns:
                logger.info("Retaining files matching patterns for %s %s until job completion: %s",
                            network, action, retain_patterns)

            # Determine if we should start continuous monitoring
            should_start_continuous = True
            selective_tarball_config = None

            if isinstance(upload_strategy, dict):
                # Complex upload strategy
                default_strategy = upload_strategy.get("default", "continuous")
                selective_tarball_config = upload_strategy.get("selective_tarball")

                if default_strategy != "continuous":
                    should_start_continuous = False

                logger.info("Complex upload strategy - default: %s, selective_tarball: %s",
                            default_strategy, bool(selective_tarball_config))
            elif upload_strategy != "continuous":
                # Simple non-continuous strategy
                should_start_continuous = False

            if cloud_storage and should_start_continuous:
                logger.info("Starting continuous upload monitor thread (sync mode)")
                exit_event = threading.Event()
                upload_thread = threading.Thread(
                    target=monitor_and_upload,
                    args=(specs["results_dir"], cloud_storage, exit_event, 0,
                          selective_tarball_config, exclude_patterns, retain_patterns),
                    daemon=True
                )
                upload_thread.start()
                logger.info("Upload monitor thread started successfully")
            else:
                logger.info(
                    "Not starting continuous upload monitor: cloud_storage=%s, should_start_continuous=%s",
                    bool(cloud_storage), should_start_continuous)
                # For tarball_after_completion or complex strategies, we'll handle upload after job completion
                exit_event = None
                upload_thread = None

            # Prepare entrypoint arguments
            args = {
                "subtask": job["action_name"],
                "experiment_spec_file": spec_path,
                "results_dir": specs["results_dir"]
            }

            module = entry_points[0].module_name.split('.')[0] if entry_points else None
            entrypoint = importlib.import_module(entrypoint_paths[module]) if module else None

            def run_entrypoint():
                nonlocal status_logger
                is_completed = False
                status_file = None

                def initialize_status_logger(status_file):
                    """Initialize or get existing status logger."""
                    nonlocal status_logger
                    if not status_logger:
                        status_logger = status_logging.StatusLogger(
                            filename=status_file,
                            is_master=True,
                            verbosity=1,
                            append=True
                        )
                        status_logging.set_status_logger(status_logger)
                    return status_logger

                try:
                    # Launch entrypoint
                    if entrypoint:
                        try:
                            _, actions = module_utils.get_neural_network_actions(job["neural_network_name"])
                            entrypoint.launch(args, "", actions, network=job["neural_network_name"])
                            is_completed = True
                        except SystemExit as e:
                            is_completed = e.code == 0
                    else:
                        is_completed = vlm_entrypoint.vlm_launch(
                            job["neural_network_name"],
                            job["action_name"],
                            specs,
                            job["job_id"]
                        )

                except Exception:
                    logger.error("Traceback")
                    logger.error(traceback.format_exc())
                    status_file = ContainerJobHandler.get_status_file(specs["results_dir"], job["action_name"])
                    status_logger = initialize_status_logger(status_file)
                    ContainerJobHandler._handle_failure(job, status_logger, status_file)
                finally:
                    status_file = status_file or ContainerJobHandler.get_status_file(
                        specs["results_dir"],
                        job["action_name"]
                    )
                    status_logger = initialize_status_logger(status_file)
                    ContainerJobHandler._cleanup(
                        exit_event,
                        upload_thread,
                        job,
                        is_completed,
                        status_logger,
                        status_file,
                        cloud_storage,
                        upload_strategy,
                        specs["results_dir"],
                        selective_tarball_config
                    )

            if sync:
                run_entrypoint()
            else:
                # Launch job asynchronously
                entrypoint_thread = threading.Thread(target=run_entrypoint, daemon=True)
                entrypoint_thread.start()

        except Exception:
            logger.error("Traceback")
            logger.error(traceback.format_exc())
            if status_logger:
                status_logging.get_status_logger().write(
                    message=(
                        f"{job['action_name']} action couldn't be launched "
                        f"for {job['neural_network_name']}"
                    ),
                    status_level=status_logging.Status.FAILURE
                )
            ContainerJobHandler._cleanup(
                exit_event=exit_event,
                upload_thread=upload_thread
            )

    @staticmethod
    def entrypoint_wrapper(job):
        """Starts job asynchronously and handles cleanup after completion.

        Args:
            job (dict): Job configuration containing action, neural network and other metadata

        Returns:
            str: Job ID if launch successful, None otherwise
        """
        try:
            docker_env_vars = job.get('docker_env_vars', {})

            def async_setup_and_run():
                cloud_storage = None
                exit_event = None
                upload_thread = None
                status_logger = None
                results_dir_snapshot = None

                try:
                    # Setup cloud storage and specs
                    cloud_storage, specs, spec_path = prepare_data_before_job_run(job, docker_env_vars)

                    # Capture snapshot of results directory after downloads but before job execution
                    results_dir_snapshot = ContainerJobHandler.capture_directory_snapshot(specs["results_dir"])

                    # Get upload strategy, exclude patterns, and retain patterns by reading network config directly
                    network = docker_env_vars.get("ORCHESTRATION_API_NETWORK",
                                                  job.get("neural_network_name", ""))
                    action = docker_env_vars.get("ORCHESTRATION_API_ACTION", job.get("action_name", ""))
                    retain_checkpoints_for_resume = (
                        docker_env_vars.get("RETAIN_CHECKPOINTS_FOR_RESUME", "false").lower() == "true"
                    )
                    upload_strategy, exclude_patterns, retain_patterns = (
                        ContainerJobHandler.get_upload_strategy_from_config(
                            network, action, retain_checkpoints_for_resume
                        )
                    )
                    logger.info("Using upload strategy for %s %s: %s", network, action, upload_strategy)
                    if exclude_patterns:
                        logger.info("Excluding patterns for %s %s: %s", network, action, exclude_patterns)
                    if retain_patterns:
                        logger.info("Retaining files matching patterns for %s %s until job completion: %s",
                                    network, action, retain_patterns)

                    # Determine if we should start continuous monitoring
                    should_start_continuous = True
                    selective_tarball_config = None

                    if isinstance(upload_strategy, dict):
                        # Complex upload strategy
                        default_strategy = upload_strategy.get("default", "continuous")
                        selective_tarball_config = upload_strategy.get("selective_tarball")

                        if default_strategy != "continuous":
                            should_start_continuous = False

                        logger.info("Complex upload strategy - default: %s, selective_tarball: %s",
                                    default_strategy, bool(selective_tarball_config))
                    elif upload_strategy != "continuous":
                        # Simple non-continuous strategy
                        should_start_continuous = False

                    if cloud_storage and should_start_continuous:
                        exit_event = threading.Event()
                        upload_thread = threading.Thread(
                            target=monitor_and_upload,
                            args=(specs["results_dir"], cloud_storage, exit_event, 0,
                                  selective_tarball_config, exclude_patterns, retain_patterns),
                            daemon=True
                        )
                        upload_thread.start()
                    else:
                        # For tarball_after_completion or complex strategies, we'll handle upload after job completion
                        exit_event = None
                        upload_thread = None

                    # Prepare entrypoint arguments
                    args = {
                        "subtask": job["action_name"],
                        "experiment_spec_file": spec_path,
                        "results_dir": specs["results_dir"]
                    }

                    module = entry_points[0].module_name.split('.')[0] if entry_points else None
                    entrypoint = importlib.import_module(entrypoint_paths[module]) if module else None

                    def run_entrypoint():
                        nonlocal status_logger
                        is_completed = False
                        status_file = None
                        entrypoint_running = threading.Event()
                        cleanup_already_done = threading.Event()

                        def initialize_status_logger(status_file):
                            """Initialize or get existing status logger."""
                            nonlocal status_logger
                            if not status_logger:
                                status_logger = status_logging.StatusLogger(
                                    filename=status_file,
                                    is_master=IS_MASTER,
                                    verbosity=1,
                                    append=True
                                )
                                status_logging.set_status_logger(status_logger)
                            return status_logger

                        def monitor_graceful_termination():
                            """Monitor for graceful termination signal and trigger shutdown if detected."""
                            nonlocal is_completed, status_logger, status_file
                            check_interval = 5  # Check every 5 seconds
                            logger.debug(
                                f"[GRACEFUL-PAUSE] Starting graceful termination monitor: "
                                f"job_id={job['job_id']}, check_interval={check_interval}s"
                            )
                            while entrypoint_running.is_set():
                                if ContainerJobHandler.check_graceful_termination_signal(specs["results_dir"]):
                                    logger.debug(
                                        f"[GRACEFUL-PAUSE] Graceful termination signal detected: "
                                        f"job_id={job['job_id']}, results_dir={specs['results_dir']}"
                                    )
                                    entrypoint_running.clear()  # Stop monitoring
                                    logger.debug(
                                        f"[GRACEFUL-PAUSE] Cleared entrypoint running flag: "
                                        f"job_id={job['job_id']}"
                                    )
                                    is_completed = False  # Mark as paused, not completed
                                    logger.debug(
                                        f"[GRACEFUL-PAUSE] Marked job as not completed (paused): "
                                        f"job_id={job['job_id']}"
                                    )

                                    cleanup_already_done.set()  # Prevent duplicate cleanup in finally block
                                    logger.debug(
                                        f"[GRACEFUL-PAUSE] Set cleanup_already_done flag: "
                                        f"job_id={job['job_id']}"
                                    )

                                    # Snapshot files to upload (prevents uploading files generated during upload wait)
                                    logger.debug(
                                        f"[GRACEFUL-PAUSE] Capturing directory snapshot for upload: "
                                        f"job_id={job['job_id']}, results_dir={specs['results_dir']}"
                                    )
                                    current_snapshot = ContainerJobHandler.capture_directory_snapshot(
                                        specs["results_dir"]
                                    )
                                    files_to_upload = (
                                        current_snapshot - results_dir_snapshot
                                        if results_dir_snapshot else current_snapshot
                                    )
                                    logger.debug(
                                        f"[GRACEFUL-PAUSE] Snapshot captured: "
                                        f"job_id={job['job_id']}, total_files={len(files_to_upload)}"
                                    )
                                    logger.debug(
                                        f"[GRACEFUL-PAUSE] Files to upload: "
                                        f"job_id={job['job_id']}, files={files_to_upload}"
                                    )

                                    # Stop continuous upload monitor and wait for current uploads
                                    if exit_event:
                                        logger.debug("Stopping continuous upload monitor")
                                        exit_event.set()
                                    if upload_thread:
                                        logger.debug("Waiting for current upload to complete")
                                        upload_thread.join()
                                        logger.debug("Upload thread joined successfully")

                                    # Upload snapshot files using common upload function
                                    if cloud_storage and files_to_upload:
                                        logger.debug(
                                            f"[GRACEFUL-PAUSE] Starting snapshot upload: "
                                            f"job_id={job['job_id']}, num_files={len(files_to_upload)}"
                                        )
                                        logger.debug(f"[GRACEFUL-PAUSE] Files to upload: {files_to_upload}")

                                        # Create progress tracker for snapshot upload
                                        snapshot_progress_tracker = None
                                        try:
                                            # Calculate total size of files to upload
                                            total_size_mb = 0.0
                                            valid_files = []
                                            logger.debug(
                                                f"[GRACEFUL-PAUSE] Calculating file sizes: "
                                                f"job_id={job['job_id']}"
                                            )
                                            for rel_path in files_to_upload:
                                                file_path = os.path.join(specs["results_dir"], rel_path)
                                                if os.path.exists(file_path) and os.path.isfile(file_path):
                                                    total_size_mb += os.path.getsize(file_path) / (1024 * 1024)
                                                    valid_files.append(rel_path)
                                            logger.debug(
                                                f"[GRACEFUL-PAUSE] Valid files: "
                                                f"job_id={job['job_id']}, files={valid_files}"
                                            )
                                            if valid_files:
                                                logger.debug(
                                                    f"[GRACEFUL-PAUSE] Snapshot upload details: "
                                                    f"job_id={job['job_id']}, files={len(valid_files)}, "
                                                    f"size_mb={total_size_mb:.1f}"
                                                )

                                                snapshot_progress_tracker = ProgressTracker(
                                                    "upload",
                                                    total_files=len(valid_files),
                                                    total_size_mb=total_size_mb,
                                                    send_callbacks=True  # Enable callbacks for snapshot uploads
                                                )
                                                logger.debug(
                                                    f"[GRACEFUL-PAUSE] Progress tracker created: "
                                                    f"job_id={job['job_id']}"
                                                )

                                                logger.debug(
                                                    f"[GRACEFUL-PAUSE] Uploading files to cloud storage: "
                                                    f"job_id={job['job_id']}"
                                                )
                                                # Don't retain during graceful pause - remove all files
                                                upload_files(
                                                    specs["results_dir"],
                                                    cloud_storage,
                                                    file_snapshot=valid_files,
                                                    selective_tarball_config=selective_tarball_config,
                                                    exclude_patterns=exclude_patterns,
                                                    progress_tracker=snapshot_progress_tracker,
                                                    retain_patterns=None
                                                )

                                                # Complete the progress tracker
                                                snapshot_progress_tracker.complete()
                                                logger.debug(
                                                    f"[GRACEFUL-PAUSE] Snapshot upload completed "
                                                    f"successfully: job_id={job['job_id']}"
                                                )
                                            else:
                                                logger.warning(
                                                    f"[GRACEFUL-PAUSE] No valid files found in snapshot "
                                                    f"to upload: job_id={job['job_id']}"
                                                )
                                        except Exception as e:
                                            logger.error(
                                                f"[GRACEFUL-PAUSE] Error during snapshot upload: "
                                                f"job_id={job['job_id']}, error={str(e)}"
                                            )
                                            logger.error(f"[GRACEFUL-PAUSE] Traceback: {traceback.format_exc()}")
                                            if snapshot_progress_tracker:
                                                # Mark as complete even on error to send final status
                                                snapshot_progress_tracker.complete()
                                            raise
                                    else:
                                        logger.warning(
                                            f"[GRACEFUL-PAUSE] Snapshot upload skipped: "
                                            f"job_id={job['job_id']}, cloud_storage={bool(cloud_storage)}, "
                                            f"files_to_upload={len(files_to_upload) if files_to_upload else 0}"
                                        )

                                    # Initialize status logger and run cleanup
                                    if not status_logger:
                                        status_file = ContainerJobHandler.get_status_file(
                                            specs["results_dir"], job["action_name"]
                                        )
                                        status_logger = initialize_status_logger(status_file)

                                    ContainerJobHandler._cleanup(
                                        exit_event=None,
                                        upload_thread=None,
                                        job=job,
                                        is_completed=False,
                                        status_logger=status_logger,
                                        status_file=status_file,
                                        cloud_storage=cloud_storage,
                                        upload_strategy=upload_strategy,
                                        results_dir=specs["results_dir"],
                                        selective_tarball_config=selective_tarball_config,
                                        results_dir_snapshot=results_dir_snapshot,
                                        exclude_patterns=exclude_patterns,
                                        is_graceful_pause=True
                                    )
                                    logger.debug(
                                        f"[GRACEFUL-PAUSE] Checkpoint upload completed: "
                                        f"job_id={job['job_id']}"
                                    )

                                    ContainerJobHandler.remove_graceful_termination_signal(specs["results_dir"])
                                    logger.debug(
                                        f"[GRACEFUL-PAUSE] Removed termination signal file: "
                                        f"job_id={job['job_id']}"
                                    )
                                    logger.debug(
                                        f"[GRACEFUL-PAUSE] Graceful pause complete, exiting: "
                                        f"job_id={job['job_id']}"
                                    )
                                    sys.exit(0)
                                logger.debug(
                                    f"[GRACEFUL-PAUSE] Monitoring - no signal detected, continuing: "
                                    f"job_id={job['job_id']}"
                                )
                                threading.Event().wait(check_interval)

                        # Start graceful termination monitor (non-daemon to ensure cleanup completes)
                        entrypoint_running.set()
                        monitor_thread = threading.Thread(target=monitor_graceful_termination, daemon=False)
                        monitor_thread.start()

                        try:
                            # Launch entrypoint
                            if entrypoint:
                                try:
                                    _, actions = module_utils.get_neural_network_actions(job["neural_network_name"])
                                    entrypoint.launch(args, "", actions, network=job["neural_network_name"])
                                    is_completed = True
                                except SystemExit as e:
                                    is_completed = e.code == 0
                            else:
                                is_completed = vlm_entrypoint.vlm_launch(
                                    job["neural_network_name"],
                                    job["action_name"],
                                    specs,
                                    job["job_id"]
                                )

                        except Exception:
                            logger.error("Traceback")
                            logger.error(traceback.format_exc())
                            status_file = ContainerJobHandler.get_status_file(specs["results_dir"], job["action_name"])
                            status_logger = initialize_status_logger(status_file)
                            ContainerJobHandler._handle_failure(job, status_logger, status_file)
                        finally:
                            # Stop monitor
                            entrypoint_running.clear()

                            # Only proceed with cleanup if graceful pause didn't already handle it
                            if not cleanup_already_done.is_set():
                                monitor_thread.join(timeout=2)

                                # Remove graceful termination signal if training completed naturally
                                # This MUST happen BEFORE setting exit_event to avoid race condition
                                # where upload thread checks signal before removal completes
                                is_terminated = ContainerJobHandler.check_graceful_termination_signal(
                                    specs["results_dir"]
                                )
                                if is_completed and is_terminated:
                                    logger.info(
                                        "Training completed naturally with early stop signal present. "
                                        f"Removing signal to allow final uploads to complete for job {job['job_id']}"
                                    )
                                    ContainerJobHandler.remove_graceful_termination_signal(specs["results_dir"])
                                    # Small delay to ensure file removal is complete before upload thread checks
                                    time.sleep(5)

                                status_file = status_file or ContainerJobHandler.get_status_file(
                                    specs["results_dir"],
                                    job["action_name"]
                                )
                                status_logger = initialize_status_logger(status_file)
                                ContainerJobHandler._cleanup(
                                    exit_event,
                                    upload_thread,
                                    job,
                                    is_completed,
                                    status_logger,
                                    status_file,
                                    cloud_storage,
                                    upload_strategy,
                                    specs["results_dir"],
                                    selective_tarball_config,
                                    results_dir_snapshot,
                                    exclude_patterns
                                )

                    # Launch job asynchronously
                    entrypoint_thread = threading.Thread(target=run_entrypoint, daemon=True)
                    entrypoint_thread.start()

                except Exception:
                    logger.error("Traceback")
                    logger.error(traceback.format_exc())
                    if status_logger:
                        status_logging.get_status_logger().write(
                            message=(
                                f"{job['action_name']} action couldn't be launched "
                                f"for {job['neural_network_name']}"
                            ),
                            status_level=status_logging.Status.FAILURE
                        )
                    ContainerJobHandler._cleanup(
                        exit_event=exit_event,
                        upload_thread=upload_thread,
                        results_dir_snapshot=results_dir_snapshot
                    )

            # Launch the async setup and execution
            setup_thread = threading.Thread(target=async_setup_and_run, daemon=True)
            setup_thread.start()

            return job["job_id"]

        except Exception:
            logger.error("Traceback")
            logger.error(traceback.format_exc())
            return None

    @staticmethod
    def _handle_failure(job, status_logger, status_file):
        """Handle failure cases by creating status logger if needed and logging failure."""
        if not status_logger:
            try:
                status_logger = status_logging.StatusLogger(
                    filename=status_file,
                    is_master=IS_MASTER,
                    verbosity=1,
                    append=True
                )
                status_logging.set_status_logger(status_logger)
            except Exception:
                logger.error("Failed to create status logger")
        status_logging.get_status_logger().write(
            message=f"{job['action_name']} action failed for {job['neural_network_name']}",
            status_level=status_logging.Status.FAILURE
        )

    @staticmethod
    def _cleanup(
        exit_event=None,
        upload_thread=None,
        job=None,
        is_completed=None,
        status_logger=None,
        status_file=None,
        cloud_storage=None,
        upload_strategy="continuous",
        results_dir=None,
        selective_tarball_config=None,
        results_dir_snapshot=None,
        exclude_patterns=None,
        is_graceful_pause=False
    ):
        """Clean up resources and log final status."""
        if exit_event:
            exit_event.set()
        if upload_thread:
            upload_thread.join()

        # Determine if tarball should be created
        should_create_tarball = is_completed or is_graceful_pause

        # Handle tarball upload strategy
        should_upload_tarball = (
            job and should_create_tarball and cloud_storage and
            upload_strategy == "tarball_after_completion" and results_dir
        )
        if should_upload_tarball:
            status = "completed" if is_completed else "paused"
            logger.info("Job %s, creating and uploading tarball", status)
            ContainerJobHandler.create_and_upload_tarball(
                results_dir,
                cloud_storage,
                job["job_id"],
                job["action_name"],
                results_dir_snapshot,
                exclude_patterns
            )

        # Handle selective tarball creation
        if selective_tarball_config and job and should_create_tarball and cloud_storage:
            status = "completed" if is_completed else "paused"
            logger.info("Job %s, creating and uploading selective tarball", status)
            tarball_path = ContainerJobHandler.create_selective_tarball(
                results_dir,
                selective_tarball_config.get("patterns", []),
                selective_tarball_config.get("base_path", ""),
                job["job_id"],
                job["action_name"]
            )

            if tarball_path:
                upload_tarball_to_cloud(cloud_storage, tarball_path, remove_after_upload=True)

        if job and is_completed is not None:
            # Determine status and result message
            if is_graceful_pause:
                status = status_logging.Status.SUCCESS
                result = "paused gracefully"
            elif is_completed:
                status = status_logging.Status.SUCCESS
                result = "completed successfully"
            else:
                status = status_logging.Status.FAILURE
                result = "failed"

            if not status_logger:
                try:
                    status_logger = status_logging.StatusLogger(
                        filename=status_file,
                        is_master=IS_MASTER,
                        verbosity=1,
                        append=True
                    )
                    status_logging.set_status_logger(status_logger)
                except Exception:
                    logger.error("Failed to create status logger")

            status_logging.get_status_logger().write(
                message=f"{job['action_name']} action {result} for {job['neural_network_name']}",
                status_level=status
            )
            if cloud_storage:
                cloud_storage.upload_file(status_file, status_file, send_status_callbacks=False)
        # Clean up any stale CUDA contexts
        cleanup_cuda_contexts()

    @staticmethod
    def get_status_file(results_dir, action_name=""):
        """Get the path to the status file.

        Args:
            results_dir (str): Directory containing results and status files
            action_name (str, optional): Name of action for creating new status file path. Defaults to "".

        Returns:
            str: Path to existing status.json file if found, otherwise constructs new path
        """
        status_files = glob.glob(os.path.join(results_dir, "**", "status.json"), recursive=True)
        if not status_files:
            return os.path.join(results_dir, action_name, "status.json")
        return status_files[0]

    @staticmethod
    def _handle_additional_downloads(
        additional_downloads,
        cloud_metadata,
        job_id,
        network_arch,
        ngc_key,
    ):
        """Handle downloading additional files specified in additional_downloads.

        Args:
            additional_downloads (list): List of additional file/directory paths to download
            cloud_metadata (dict): Cloud storage metadata
            job_id (str): Current job ID
            network_arch (str): Network architecture name
            ngc_key (str): NGC API key
        """
        try:
            if not additional_downloads:
                return

            logger.info("Starting additional downloads (%d files)...", len(additional_downloads))

            # Create progress tracker for additional downloads
            additional_progress_tracker = ProgressTracker(
                "download",
                total_files=len(additional_downloads),
                total_size_mb=0,  # Size unknown for additional downloads
                send_callbacks=True
            )

            # Create a spec structure that includes all additional downloads
            # Use preserve_source_path=True to maintain original path structure
            additional_spec = {}

            for i, download_path in enumerate(additional_downloads, 1):
                logger.info("Processing additional download %d/%d: %s", i, len(additional_downloads), download_path)
                additional_spec[f"additional_download_{i - 1}"] = download_path

            # Use the existing download utility with preserve_source_path=True
            download_files_from_spec(
                cloud_data=cloud_metadata,
                data=additional_spec,
                job_id=job_id,
                network_arch=network_arch,
                ngc_key=ngc_key,
                reprocess_files=[],
                preserve_source_path=True,
                progress_tracker=additional_progress_tracker
            )

            additional_progress_tracker.complete()
            logger.info("All %d additional downloads completed successfully", len(additional_downloads))

        except Exception as e:
            logger.error("Error handling additional downloads: %s", str(e))
            logger.error("Traceback: %s", traceback.format_exc())

    @staticmethod
    def get_current_job_status(results_dir, workspace_metadata={}, job_id=None):
        """Finds 'status.json' under specs['results_dir'] and returns the last entry's status."""
        if "://" in results_dir:
            bucket_name = results_dir.split("//")[1].split("/")[0]
            results_dir = results_dir[results_dir.find(bucket_name) + len(bucket_name):]

        if not results_dir:
            cleanup_cuda_contexts()
            raise ValueError("Empty 'results_dir' in specs.")

        # Only download from cloud if workspace_metadata is provided
        # For Lustre/Slurm jobs, no workspace_metadata means direct filesystem access
        local_results_dir = results_dir
        if workspace_metadata:
            cs_instance, _ = create_cs_instance(workspace_metadata)
            if cs_instance:
                cloud_type = workspace_metadata.get('cloud_type', '')
                # For SLURM, download to a temporary directory since Lustre path won't exist in workflow container
                if cloud_type == 'slurm':
                    import tempfile
                    local_results_dir = tempfile.mkdtemp(prefix=f"status_{job_id}_")
                    logger.info(f"Downloading SLURM status files from {results_dir} to {local_results_dir}")
                cs_instance.download_folder(results_dir, local_results_dir, extensions=[".json"])
            else:
                logger.error(
                    "Failed to create cloud storage instance for cloud type: %s",
                    workspace_metadata.get('cloud_type')
                )
                return "Pending"

        if not os.path.isdir(local_results_dir):
            logger.error("results_dir directory %s does not exist", local_results_dir)
            return "Pending"

        file_path = ContainerJobHandler.get_status_file(local_results_dir)
        last_status, data = None, {}

        if not os.path.exists(file_path):
            # Clean up temp directory if created for SLURM
            cloud_type = workspace_metadata.get('cloud_type', '') if workspace_metadata else ''
            if cloud_type == 'slurm' and local_results_dir != results_dir:
                import shutil
                try:
                    shutil.rmtree(local_results_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory {local_results_dir}: {e}")
            return "Pending"

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    last_status = data.get("status")
                except json.JSONDecodeError:
                    continue
        if workspace_metadata and data:
            handler_id = get_handler_id(job_id)
            handler_metadata = get_handler_metadata(handler_id)
            handler_kind = get_handler_kind(handler_metadata)
            if os.getenv("DEBUG_MODE", "False") == "True":
                logger.info(f"Saving data for: {data} for job: {job_id} with handler_id: "
                            f"{handler_id} and kind: {handler_kind}")
            callback_data = {
                "experiment_number": os.getenv("AUTOML_EXPERIMENT_NUMBER", "0"),
                "status": json.dumps(data),
            }
            save_dnn_status(job_id, callback_data=callback_data, handler_id=handler_id, kind=handler_kind)

        # Clean up temp directory if created for SLURM
        cloud_type = workspace_metadata.get('cloud_type', '') if workspace_metadata else ''
        if cloud_type == 'slurm' and local_results_dir != results_dir:
            import shutil
            try:
                shutil.rmtree(local_results_dir)
                logger.info(f"Cleaned up temp directory {local_results_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {local_results_dir}: {e}")

        return {
            "STARTED": "Running",
            "RUNNING": "Running",
            "SUCCESS": "Done",
            "FAILURE": "Error",
        }.get(last_status, "Pending")

    @staticmethod
    def find_files_by_patterns(base_dir, patterns):
        """Find files and directories matching the given patterns.

        Args:
            base_dir (str): Base directory to search in
            patterns (list): List of glob patterns to match

        Returns:
            list: List of file/directory paths that match the patterns
        """
        matched_paths = set()

        for pattern in patterns:
            # Convert the pattern to work with os.walk
            search_path = os.path.join(base_dir, pattern)

            # Use glob to find matching paths
            for match in glob.glob(search_path, recursive=True):
                if os.path.exists(match):
                    matched_paths.add(match)

        return list(matched_paths)

    @staticmethod
    def create_selective_tarball(results_dir, patterns, base_path, job_id, action_name):
        """Create a tarball containing only files matching the specified patterns.

        Args:
            results_dir (str): Results directory
            patterns (list): List of glob patterns to include in tarball
            base_path (str): Base path within results_dir to apply patterns
            job_id (str): Job ID for naming the tarball
            action_name (str): Action name for naming the tarball

        Returns:
            str: Path to created tarball or None if failed
        """
        try:
            search_dir = os.path.join(results_dir, base_path) if base_path else results_dir
            if not os.path.exists(search_dir):
                logger.warning("Search directory does not exist: %s", search_dir)
                return None

            # Find files matching patterns
            matched_files = ContainerJobHandler.find_files_by_patterns(search_dir, patterns)

            if not matched_files:
                logger.info("No files found matching patterns: %s", patterns)
                return None

            tarball_name = f"{action_name}_selective.tar.gz"
            tarball_path = os.path.join(results_dir, tarball_name)

            logger.info("Creating selective tarball with %d matching files/directories", len(matched_files))

            # Create tarball with only matching files
            with tarfile.open(tarball_path, 'w:gz') as tar:
                for file_path in tqdm(matched_files, desc="Adding files to tarball"):
                    # Calculate relative path for archive
                    if os.path.isfile(file_path):
                        rel_path = os.path.relpath(file_path, results_dir)
                        tar.add(file_path, arcname=rel_path)

            logger.info("Selective tarball created successfully: %s", tarball_path)
            return tarball_path

        except Exception as e:
            logger.error("Error creating selective tarball: %s", str(e))
            logger.error("Traceback: %s", traceback.format_exc())
            return None

    @staticmethod
    def get_upload_strategy_from_config(network, action, retain_checkpoints_for_resume=False):
        """Get upload strategy, exclude patterns, and retain patterns by reading network config directly.

        Args:
            network (str): Network name
            action (str): Action name
            retain_checkpoints_for_resume (bool): Whether to retain .pth checkpoints for training resume

        Returns:
            tuple: (upload_strategy, exclude_patterns, retain_patterns) where upload_strategy is dict or str,
                   exclude_patterns is list or None, and retain_patterns is list or None
        """
        try:
            network_config = read_network_config(network)
            if network_config and "cloud_upload" in network_config:
                cloud_upload_config = network_config["cloud_upload"]
                strategy = cloud_upload_config.get("upload_strategy", {}).get(action, "continuous")
                exclude_patterns = cloud_upload_config.get("exclude_patterns", {}).get(action)
                retain_patterns = cloud_upload_config.get("retain_patterns", {}).get(action)

                # If retaining for resume, remove .pth exclusion patterns
                if retain_checkpoints_for_resume and exclude_patterns:
                    # Filter out patterns that exclude .pth files
                    exclude_patterns = [
                        pattern for pattern in exclude_patterns
                        if not ("pth" in pattern.lower() or r"\\.pth" in pattern)
                    ]
                    logger.info(
                        "retain_checkpoints_for_resume is enabled, allowing .pth files to be uploaded for %s %s",
                        network, action
                    )
                    # Return None if all patterns were removed
                    exclude_patterns = exclude_patterns if exclude_patterns else None

                return strategy, exclude_patterns, retain_patterns
            return "continuous", None, None  # Default to continuous if not specified
        except Exception as e:
            logger.error("Error reading upload strategy from network config: %s", str(e))
            return "continuous", None, None


def auto_resume_checkpoint_inference(specs, job_id, network_name, action_name):
    """Auto-detect and infer resume checkpoint parameters for SLURM requeue support.

    This function works WITHOUT MongoDB access (container-level code):
    - Uses direct filesystem operations to find checkpoints
    - Handles both regular train jobs and AutoML train jobs
    - Updates resume-related spec parameters based on found checkpoints
    - Handles boolean flags (like cosmos-rl resume: true)

    Args:
        specs (dict): Job specs to update
        job_id (str): Job ID
        network_name (str): Network architecture name
        action_name (str): Action name (should be 'train')

    Returns:
        dict: Updated specs with resume checkpoint paths if found
    """
    if action_name != "train":
        logger.debug(f"Action is {action_name}, not train - skipping resume checkpoint detection")
        return specs

    logger.info("=" * 80)
    logger.info("AUTO-RESUME CHECKPOINT DETECTION (Container-Level, No MongoDB)")
    logger.info(f"Network: {network_name}, Job ID: {job_id}")
    logger.info("=" * 80)

    try:
        # Get results directory from specs
        results_dir = specs.get("results_dir")
        if not results_dir:
            logger.warning("No results_dir in specs - cannot detect checkpoints")
            return specs

        if not os.path.exists(results_dir):
            logger.info(f"Results directory doesn't exist yet: {results_dir}")
            logger.info("This is normal for first run - no checkpoint to resume")
            return specs

        logger.info(f"Searching for checkpoints in: {results_dir}")

        # Check if this is an AutoML job
        automl_experiment_number = os.environ.get("AUTOML_EXPERIMENT_NUMBER")
        is_automl_job = automl_experiment_number is not None

        logger.info(f"Job type: {'AutoML' if is_automl_job else 'Regular'}")
        if is_automl_job:
            logger.info(f"AutoML experiment number: {automl_experiment_number}")

        # Read network config to get resume parameter names
        network_config = read_network_config(network_name)

        # Determine which config section to use
        if is_automl_job:
            if "automl_spec_params" not in network_config:
                logger.info(f"No automl_spec_params in {network_name} config")
                return specs
            spec_params = network_config["automl_spec_params"]
            param_section = "automl_spec_params"
        else:
            if "spec_params" not in network_config or "train" not in network_config["spec_params"]:
                logger.info(f"No spec_params.train in {network_name} config")
                return specs
            spec_params = network_config["spec_params"]["train"]
            param_section = "spec_params.train"

        logger.info(f"Using config section: {param_section}")

        # Find checkpoint files using glob patterns
        # Common checkpoint patterns for TAO networks
        checkpoint_patterns = [
            "*.pth",
            "*.ckpt",
            "*.tlt",
            "**/checkpoint*.pth",
            "**/model*.pth",
            "**/latest*.pth",
            "**/best*.pth",
            "**/epoch*.pth"
        ]

        found_checkpoints = []
        for pattern in checkpoint_patterns:
            search_pattern = os.path.join(results_dir, pattern)
            matches = glob.glob(search_pattern, recursive=True)
            found_checkpoints.extend(matches)

        # Remove duplicates and sort by modification time (newest first)
        found_checkpoints = list(set(found_checkpoints))
        if found_checkpoints:
            found_checkpoints.sort(key=os.path.getmtime, reverse=True)
            logger.info(f"Found {len(found_checkpoints)} checkpoint file(s)")
            for idx, ckpt in enumerate(found_checkpoints[:5]):  # Show first 5
                logger.info(f"  [{idx + 1}] {os.path.relpath(ckpt, results_dir)}")
            if len(found_checkpoints) > 5:
                logger.info(f"  ... and {len(found_checkpoints) - 5} more")

            # Use the newest checkpoint
            latest_checkpoint = found_checkpoints[0]
            logger.info(f"Selected checkpoint: {os.path.relpath(latest_checkpoint, results_dir)}")

            # Update specs with checkpoint path
            # Find resume-related parameters in spec_params
            resume_path_params = []
            resume_bool_params = []

            for field_name, inference_fn in spec_params.items():
                if isinstance(inference_fn, str):
                    # Check if this looks like a resume parameter
                    if "resume" in field_name.lower() and "path" in field_name.lower():
                        resume_path_params.append(field_name)
                    elif "resume" in field_name.lower() and ("bool" in inference_fn or field_name.endswith("resume")):
                        resume_bool_params.append(field_name)

            updated_params = {}

            # Update path parameters
            for param in resume_path_params:
                write_nested_dict(specs, param, latest_checkpoint)
                updated_params[param] = latest_checkpoint
                logger.info(f"✓ Updated {param} = {latest_checkpoint}")

            # Update boolean parameters (set to True when checkpoint exists)
            for param in resume_bool_params:
                write_nested_dict(specs, param, True)
                updated_params[param] = True
                logger.info(f"✓ Updated {param} = True")

            if updated_params:
                logger.info("=" * 80)
                logger.info("✓ RESUME CHECKPOINT DETECTED")
                logger.info(f"Job type: {'AutoML' if is_automl_job else 'Regular'}")
                logger.info(f"Checkpoint: {latest_checkpoint}")
                logger.info(f"Updated {len(updated_params)} parameter(s):")
                for param, value in updated_params.items():
                    logger.info(f"  - {param}: {value}")
                logger.info("Training will resume from existing checkpoint")
                logger.info("=" * 80)
            else:
                logger.warning("No resume parameters found in network config to update")
                logger.warning(f"Check {param_section} in network config")
        else:
            logger.info("=" * 80)
            logger.info("✗ No checkpoints found")
            logger.info(f"Searched in: {results_dir}")
            logger.info(f"Patterns: {checkpoint_patterns}")
            logger.info("Training will start from scratch or use PTM")
            logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during auto-resume checkpoint detection: {e}")
        logger.error(traceback.format_exc())
        logger.warning("Proceeding with original specs (no resume applied)")

    return specs


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Run container jobs directly without microservices")
    parser.add_argument(
        '--neural-network-name',
        type=str,
        required=True,
        help='Name of the neural network to run'
    )
    parser.add_argument(
        '--action-name',
        type=str,
        required=True,
        help='Name of the action to run'
    )
    parser.add_argument(
        '--job-id',
        type=str,
        default=str(uuid.uuid4()),
        required=True,
        help='Job ID'
    )
    parser.add_argument(
        '--specs',
        type=str,
        help='JSON string containing specs'
    )
    parser.add_argument(
        '--specs-file',
        type=str,
        help='Path to JSON file containing specs'
    )
    parser.add_argument(
        '--cloud-metadata',
        help='JSON string containing cloud metadata'
    )
    parser.add_argument(
        '--cloud-metadata-file',
        type=str,
        help='Path to JSON file containing cloud metadata'
    )
    parser.add_argument(
        '--docker-env-vars',
        type=str,
        help='JSON string containing docker environment variables'
    )
    parser.add_argument(
        '--docker-env-vars-file',
        type=str,
        help='Path to JSON file containing docker environment variables'
    )
    args = parser.parse_args()

    def load_json_from_arg_or_file(json_str, file_path, default='{}'):
        """Load JSON from string argument or file path with retry mechanism."""
        logger.info("Path exists: %s", os.path.exists(file_path) if file_path else False)
        logger.info("File path: %s", file_path)
        logger.info("JSON string: %s", json_str)
        logger.info("Default: %s", default)
        if file_path:
            # Load from file with retry mechanism for distributed filesystems (e.g., Lustre/NFS)
            last_exception = None
            for attempt in range(NUM_RETRY):
                try:
                    if attempt > 0:
                        logger.warning(f"Retrying JSON file read (attempt {attempt + 1}/{NUM_RETRY}): {file_path}")
                        time.sleep(10)  # Wait 10 seconds between retries for filesystem sync
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if not content.strip():
                            raise ValueError("File is empty or contains only whitespace")
                        return json.loads(content)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Failed to read JSON file {file_path} (attempt {attempt + 1}/{NUM_RETRY}): {e}")
                    if attempt == NUM_RETRY - 1:
                        # Last attempt - log full error and raise
                        logger.error(f"All {NUM_RETRY} retry attempts failed for JSON file {file_path}: {e}")
                        logger.error(traceback.format_exc())
                        raise
            # This should not be reached, but just in case
            raise last_exception
        if json_str:
            # Parse as JSON string
            return json.loads(json_str)
        # Use default
        return json.loads(default)

    try:
        specs = load_json_from_arg_or_file(args.specs, args.specs_file)
        docker_env_vars = load_json_from_arg_or_file(args.docker_env_vars, args.docker_env_vars_file, '{}')
        cloud_metadata = load_json_from_arg_or_file(args.cloud_metadata, args.cloud_metadata_file,
                                                    os.environ.get("CLOUD_METADATA", "{}"))

        # Auto-detect resume checkpoint for SLURM requeue support
        # Note: MongoDB is not available in containers, so we use filesystem-based detection
        if args.action_name == "train":
            try:
                logger.info("Applying auto-resume checkpoint detection...")
                specs = auto_resume_checkpoint_inference(
                    specs,
                    args.job_id,
                    args.neural_network_name,
                    args.action_name
                )

                # Update the specs file if it was loaded from file
                # This makes the resume visible in logs and for debugging
                if args.specs_file:
                    try:
                        with open(args.specs_file, 'w', encoding='utf-8') as f:
                            json.dump(specs, f, indent=2)
                        logger.info(f"Updated specs file with resume checkpoint: {args.specs_file}")
                    except Exception as e:
                        logger.warning(f"Could not update specs file: {e}")
            except Exception as e:
                logger.warning(f"Auto-resume detection failed: {e}")
                logger.warning("Proceeding with original specs")
                logger.debug(traceback.format_exc())

        job = {
            'job_id': args.job_id,
            'neural_network_name': args.neural_network_name,
            'action_name': args.action_name,
            'specs': specs,
            'cloud_metadata': cloud_metadata,
            'docker_env_vars': docker_env_vars,
        }
        logger.info(f"Starting container job: {args.job_id}")
        docker_env_vars = job.get("docker_env_vars", {})
        if docker_env_vars:
            os.environ.update(docker_env_vars)
        ContainerJobHandler.setup_and_run(job, docker_env_vars, sync=True)
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"Error running container job: {e}")
        sys.exit(1)

    logger.info(f"Job {args.job_id} completed successfully")
    sys.exit(0)


if __name__ == "__main__":
    main()
