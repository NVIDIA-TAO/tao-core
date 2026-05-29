# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Monitor PTM downloads and send progress updates."""

import os
import logging

logger = logging.getLogger(__name__)


def get_folder_size_mb(folder_path):
    """Get current size of a folder in MB.

    Args:
        folder_path (str): Path to folder

    Returns:
        float: Folder size in MB
    """
    total_size = 0
    try:
        if os.path.exists(folder_path):
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                    except (OSError, IOError):
                        continue
        return total_size / (1024 * 1024)  # Convert to MB
    except Exception:
        return 0.0


def download_ptm_with_progress(download_func, destination_path, file_name,
                               total_size_mb, progress_tracker, *args,
                               **kwargs):
    """Execute a download function with progress monitoring integrated into main progress tracker.

    Args:
        download_func: Function to call for downloading
        destination_path (str): Destination folder path
        file_name (str): Display name for the file
        total_size_mb (float): Expected total size in MB
        progress_tracker: Main progress tracker instance
        *args, **kwargs: Arguments to pass to download_func

    Returns:
        Result of download_func
    """
    import threading
    import time

    # Create stop event for monitoring thread
    stop_event = threading.Event()

    def monitor_ptm_progress():
        """Monitor PTM download progress and update main progress tracker."""
        try:
            start_time = time.time()
            last_update_time = start_time

            while not stop_event.is_set():
                current_time = time.time()

                # Send updates every 2 seconds
                if current_time - last_update_time >= 1.0:
                    downloaded_mb = get_folder_size_mb(destination_path)

                    # Update the main progress tracker with current progress
                    if progress_tracker:
                        # Calculate how much progress to add since last update
                        size_diff = downloaded_mb - getattr(monitor_ptm_progress, 'last_downloaded_mb', 0)
                        if size_diff > 0:
                            progress_tracker.update_progress(size_processed_mb=size_diff)
                        monitor_ptm_progress.last_downloaded_mb = downloaded_mb

                    last_update_time = current_time

                # Sleep briefly before checking again
                time.sleep(0.5)

        except Exception as e:
            logger.warning(f"Error in PTM progress monitoring: {e}")

    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_ptm_progress, daemon=True)
    monitor_ptm_progress.last_downloaded_mb = 0  # Initialize tracking
    monitor_thread.start()

    try:
        # Execute the actual download
        result = download_func(*args, **kwargs)

        # Send final progress update
        final_downloaded_mb = get_folder_size_mb(destination_path)
        remaining_size = final_downloaded_mb - getattr(monitor_ptm_progress, 'last_downloaded_mb', 0)
        if progress_tracker and remaining_size > 0:
            progress_tracker.update_progress(size_processed_mb=remaining_size)

        # Mark file as complete
        if progress_tracker:
            progress_tracker.complete_file_download()

        return result
    finally:
        # Stop monitoring thread
        stop_event.set()
        monitor_thread.join(timeout=2)
