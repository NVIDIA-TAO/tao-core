# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for progress tracking."""

import os
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


def format_size(size_mb):
    """Format size for display (MB or GB)."""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f} GB"
    return f"{size_mb:.1f} MB"


def format_eta(seconds):
    """Format ETA as timedelta string."""
    if seconds <= 0:
        return "0:00:00"
    return str(timedelta(seconds=int(seconds)))


def get_file_size_mb(file_path):
    """Get file size in MB.

    Args:
        file_path (str): Path to the file

    Returns:
        float: File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)  # Convert to MB
    except (OSError, IOError):
        return 0.0


def get_folder_stats(folder_path):
    """Get folder statistics (file count and total size).

    Args:
        folder_path (str): Path to the folder

    Returns:
        tuple: (file_count, total_size_mb)
    """
    file_count = 0
    total_size = 0

    try:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                except (OSError, IOError):
                    continue

        return file_count, total_size / (1024 * 1024)  # Convert to MB
    except Exception:
        return 0, 0.0


def send_progress_status_callback(message):
    """Send progress status callback using HTTP (job pods don't have MongoDB access).

    For cloud-based jobs: sends HTTP callback to server
    For SLURM jobs: writes to status.json so microservices can pick it up
    Also notifies inference microservice progress bridge for real-time progress updates.

    Args:
        message (str): Progress message to send
    """
    # Skip callbacks when DEBUG_ENABLED is set (for local debugging)
    if os.getenv("DEBUG_ENABLED", "").lower() == "true":
        logger.info("DEBUG_ENABLED=True: Skipping progress status callback. Message: %s", message)
        return

    try:
        # Notify inference microservice progress bridge (if registered)
        try:
            job_id = os.getenv("TAO_API_JOB_ID")
            if job_id:
                from nvidia_tao_core.microservices.handlers.inference_progress_bridge import notify_progress
                notify_progress(job_id, message)
        except Exception as e:
            logger.debug(f"Could not notify inference progress bridge: {e}")

        if os.getenv("CLOUD_BASED") == "True":
            # Use status_callback (HTTP) instead of internal_job_status_update (MongoDB)
            # Job pods don't have MongoDB access, so we send via HTTP to the server
            from nvidia_tao_core.microservices.handlers.cloud_handlers.utils import status_callback
            from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_internal_job_status_update_data

            # Create status update data with RUNNING status for progress updates
            callback_data = get_internal_job_status_update_data(
                automl_experiment_number=os.getenv("AUTOML_EXPERIMENT_NUMBER", "0"),
                message=message,
                status="RUNNING"  # Progress updates use RUNNING status
            )

            # Send via HTTP callback (server will save to MongoDB)
            status_callback(callback_data)
        else:
            # For non-cloud-based (SLURM), write to status.json
            from nvidia_tao_core.loggers.logging import StatusLogger, Status, Verbosity

            job_id = os.getenv('TAO_API_JOB_ID')
            if not job_id:
                logger.debug("TAO_API_JOB_ID not set, skipping status.json write")
                return

            # Use TAO_API_RESULTS_DIR for SLURM compatibility, fallback to /results
            results_base = os.getenv('TAO_API_RESULTS_DIR', '/results')
            results_dir = os.path.join(results_base, job_id)
            status_file = os.path.join(results_dir, "status.json")
            logger.info(f"Writing progress status to {status_file}")

            # Check if we're the master process (only master should write)
            is_master = int(os.environ.get("NODE_RANK", 0)) == 0

            if is_master:
                # Create status logger and write progress message
                status_logger = StatusLogger(
                    filename=status_file,
                    is_master=True,
                    verbosity=Verbosity.INFO,
                    append=True
                )
                status_logger.write(
                    data={},
                    status_level=Status.RUNNING,
                    verbosity_level=Verbosity.INFO,
                    message=message
                )

    except Exception as e:
        # Don't let callback failures break the main operation
        logger.warning(f"Error sending progress status callback: {e}")
