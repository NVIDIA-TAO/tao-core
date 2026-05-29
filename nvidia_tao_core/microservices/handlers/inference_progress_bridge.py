# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge to capture download progress and update inference microservice progress"""

import re
import logging
import threading

logger = logging.getLogger(__name__)

# Global registry for progress callbacks
_progress_callbacks = {}
_callbacks_lock = threading.Lock()


def register_progress_callback(job_id: str, callback_func):
    """Register a progress callback for a specific job

    Args:
        job_id: Job ID to register callback for
        callback_func: Function to call with progress info (percentage, message)
    """
    with _callbacks_lock:
        _progress_callbacks[job_id] = callback_func
        logger.info(f"Registered progress callback for job {job_id}")


def unregister_progress_callback(job_id: str):
    """Unregister a progress callback

    Args:
        job_id: Job ID to unregister
    """
    with _callbacks_lock:
        if job_id in _progress_callbacks:
            del _progress_callbacks[job_id]
            logger.info(f"Unregistered progress callback for job {job_id}")


def notify_progress(job_id: str, message: str):
    """Notify registered callbacks about progress

    Args:
        job_id: Job ID
        message: Progress message from ProgressTracker
    """
    with _callbacks_lock:
        callback = _progress_callbacks.get(job_id)

    if callback:
        try:
            # Parse percentage from message for informational purposes
            # Example: "Total Download Progress: 0/1 files (0.0%), 2.1 GB/3.9 GB (53.7%)"
            percentage = _extract_percentage(message)
            # Pass the percentage (for potential use) and the full message with details
            callback(percentage or 0, message)
        except Exception as e:
            logger.debug(f"Error in progress callback: {e}")


def _extract_percentage(message: str) -> float:
    """Extract percentage from progress message

    Args:
        message: Progress message

    Returns:
        Percentage as float (0-100) or None if not found
    """
    # Look for patterns like "2.1 GB/3.9 GB (53.7%)"
    match = re.search(r'(\d+\.?\d*)\s*GB/(\d+\.?\d*)\s*GB\s*\((\d+\.?\d*)%\)', message)
    if match:
        return float(match.group(3))

    # Look for patterns like "3/10 files (30.0%)"
    match = re.search(r'(\d+)/(\d+)\s+files\s*\((\d+\.?\d*)%\)', message)
    if match:
        return float(match.group(3))

    return None
