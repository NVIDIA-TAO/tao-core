# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Progress tracking for file upload/download operations."""

import time
import logging
import threading
from datetime import timedelta

from nvidia_tao_core.microservices.handlers.cloud_handlers.progress_tracker_utils import (
    format_size,
    format_eta,
    send_progress_status_callback
)

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Unified progress tracking utility for file operations with ETA estimation.

    Handles both file-level progress (multiple files) and streaming progress (single large file).
    """

    def __init__(self, operation_type, total_files=0, total_size_mb=0, file_name=None, send_callbacks=True):
        """Initialize progress tracker.

        Args:
            operation_type (str): Type of operation ('upload' or 'download')
            total_files (int): Total number of files to process
            total_size_mb (float): Total size in MB to process
            file_name (str, optional): Single file name for streaming progress
            send_callbacks (bool): Whether to send status callbacks
        """
        self.operation_type = operation_type
        self.total_files = total_files
        self.total_size_mb = total_size_mb
        self.file_name = file_name
        self.completed_files = 0
        self.in_progress_files = 0
        self.processed_size_mb = 0.0
        self.current_file_size_mb = 0.0  # Track current file's size
        self.current_file_processed_mb = 0.0  # Track current file's progress
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_callback_time = self.start_time
        self.send_callbacks = send_callbacks
        self.lock = threading.Lock()

        # Determine if this is streaming mode (single large file)
        self.is_streaming = file_name is not None and total_files <= 1

    def start_file_download(self, file_name=None, file_size_mb=0.0):
        """Mark that a file download has started.

        Args:
            file_name (str, optional): Name of the file being downloaded
            file_size_mb (float, optional): Size of the current file
        """
        with self.lock:
            current_total = self.completed_files + self.in_progress_files
            if current_total < self.total_files:
                self.in_progress_files += 1
                # Update current file info for batch operations
                if file_name:
                    self.file_name = file_name
                self.current_file_size_mb = file_size_mb
                self.current_file_processed_mb = 0.0  # Reset for new file

    def complete_file_download(self):
        """Mark that a file download has completed."""
        with self.lock:
            if self.completed_files < self.total_files:
                self.completed_files += 1
                self.in_progress_files = max(0, self.in_progress_files - 1)
                # Clear file name for batch operations (will be set for next file)
                if not self.is_streaming:
                    self.file_name = None

    def get_current_file_count(self):
        """Get the current file count for display (completed + in_progress)."""
        current_count = self.completed_files + self.in_progress_files
        return min(current_count, self.total_files)

    def get_remaining_files(self):
        """Get the number of remaining files (not yet started)."""
        return max(0, self.total_files - self.completed_files - self.in_progress_files)

    def update_progress(self, files_processed=0, size_processed_mb=0.0):
        """Update progress with size information.

        Args:
            files_processed (int): Number of files completed (for simple cases)
            size_processed_mb (float): Size in MB processed in this update
        """
        with self.lock:
            # Simple file completion tracking
            if files_processed > 0:
                self.completed_files = min(self.completed_files + files_processed, self.total_files)
                self.in_progress_files = max(0, self.in_progress_files - files_processed)

            self.processed_size_mb += size_processed_mb
            # Also track current file progress
            self.current_file_processed_mb += size_processed_mb
            current_time = time.time()

            # Different update intervals for streaming vs batch operations
            update_interval = 3.0 if self.is_streaming else 2.0
            callback_interval = 5.0

            # Check if we should update
            should_update = (
                current_time - self.last_update_time >= update_interval or
                self.completed_files >= self.total_files or
                (self.is_streaming and self.processed_size_mb >= self.total_size_mb)
            )

            if should_update:
                self.last_update_time = current_time

                # Send callback less frequently
                if (self.send_callbacks and
                    (current_time - self.last_callback_time >= callback_interval or
                     self._is_complete())):
                    self._send_progress_callback()
                    self.last_callback_time = current_time

    def _is_complete(self):
        """Check if operation is complete."""
        if self.is_streaming:
            return self.processed_size_mb >= self.total_size_mb
        return self.completed_files >= self.total_files

    def _send_progress_callback(self):
        """Send progress status callback and log to terminal."""
        try:
            # Use helper methods for consistent logic
            current_file_count = self.get_current_file_count()
            remaining_files = self.get_remaining_files()
            remaining_size_mb = max(0, self.total_size_mb - self.processed_size_mb)

            # Calculate progress percentages
            file_progress = (current_file_count / max(1, self.total_files)) * 100 if self.total_files > 0 else 0
            size_progress = (self.processed_size_mb / max(1, self.total_size_mb)) * 100 if self.total_size_mb > 0 else 0

            # Calculate ETA
            elapsed_time = time.time() - self.start_time
            eta_str = "calculating..."
            if elapsed_time > 1 and self.processed_size_mb > 0 and self.total_size_mb > 0:
                mb_per_second = self.processed_size_mb / elapsed_time
                if mb_per_second > 0 and remaining_size_mb > 0:
                    eta_seconds = remaining_size_mb / mb_per_second
                    eta_str = format_eta(eta_seconds)
                elif remaining_size_mb <= 0:
                    eta_str = "0:00:00"

            # Create progress message with better format
            if self.total_files > 0 and self.total_size_mb > 0:
                # Add current file info with its progress if available
                if self.file_name:
                    current_file_percent = (
                        (self.current_file_processed_mb / max(1, self.current_file_size_mb)) * 100
                        if self.current_file_size_mb > 0 else 0
                    )
                    current_file_info = (
                        f"Current file {self.operation_type}: {self.file_name}; "
                        f"Current file {self.operation_type} Progress: {format_size(self.current_file_processed_mb)}"
                    )
                    if self.current_file_size_mb > 0:
                        current_file_info += f"/{format_size(self.current_file_size_mb)} ({current_file_percent:.1f}%)"
                    current_file_info += "; "
                else:
                    current_file_info = ""

                # Prioritize size-based progress for single large downloads (like HF models)
                # to avoid showing misleading "0/1 files" when size shows significant progress
                if self.total_files == 1 and size_progress > 5.0:
                    message = (
                        f"{current_file_info}"
                        f"Total {self.operation_type.capitalize()} Progress: "
                        f"{format_size(self.processed_size_mb)}/{format_size(self.total_size_mb)} "
                        f"({size_progress:.1f}%), Remaining: {format_size(remaining_size_mb)}, ETA: {eta_str}"
                    )
                else:
                    message = (
                        f"{current_file_info}"
                        f"Total {self.operation_type.capitalize()} Progress: "
                        f"{current_file_count}/{self.total_files} files ({file_progress:.1f}%), "
                        f"{format_size(self.processed_size_mb)}/{format_size(self.total_size_mb)} "
                        f"({size_progress:.1f}%), Remaining: {remaining_files} files, "
                        f"{format_size(remaining_size_mb)}, ETA: {eta_str}"
                    )
            elif self.total_files > 0:
                current_file_info = f"Current file {self.operation_type}: {self.file_name}; " if self.file_name else ""
                if self.processed_size_mb > 0:
                    message = (
                        f"{current_file_info}"
                        f"Total {self.operation_type.capitalize()} Progress: "
                        f"{current_file_count}/{self.total_files} files ({file_progress:.1f}%), "
                        f"{format_size(self.processed_size_mb)} downloaded, "
                        f"Remaining: {remaining_files} files"
                    )
                else:
                    message = (
                        f"{current_file_info}"
                        f"Total {self.operation_type.capitalize()} Progress: "
                        f"{current_file_count}/{self.total_files} files ({file_progress:.1f}%), "
                        f"Remaining: {remaining_files} files"
                    )
            elif self.total_size_mb > 0:
                message = (
                    f"{self.operation_type.capitalize()} Progress: "
                    f"{format_size(self.processed_size_mb)}/{format_size(self.total_size_mb)} "
                    f"({size_progress:.1f}%), Remaining: {format_size(remaining_size_mb)}"
                )
            else:
                message = f"{self.operation_type.capitalize()} in progress..."

            # Log to terminal
            logger.info(message)

            # Send callback to server if enabled
            if self.send_callbacks:
                send_progress_status_callback(message)

        except Exception as e:
            logger.warning(f"Failed to send progress callback: {e}")

    def complete(self):
        """Mark operation as complete and log final stats."""
        elapsed_time = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))

        # Log completion
        if self.total_files > 0 and self.total_size_mb > 0:
            completion_msg = (
                f"{self.operation_type.capitalize()} completed: "
                f"{self.completed_files} files, {format_size(self.processed_size_mb)} "
                f"in {elapsed_str}"
            )
        elif self.total_files > 0:
            completion_msg = (
                f"{self.operation_type.capitalize()} completed: "
                f"{self.completed_files} files in {elapsed_str}"
            )
        elif self.total_size_mb > 0:
            completion_msg = (
                f"{self.operation_type.capitalize()} completed: "
                f"{format_size(self.processed_size_mb)} in {elapsed_str}"
            )
        else:
            completion_msg = f"{self.operation_type.capitalize()} completed in {elapsed_str}"

        logger.info(completion_msg)

        # Send completion callback
        if self.send_callbacks:
            send_progress_status_callback(completion_msg)
