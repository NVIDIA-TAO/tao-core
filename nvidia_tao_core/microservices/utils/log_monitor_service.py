# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Log monitoring service for Kubernetes and Docker backends.

This service runs on the server side and periodically fetches logs from
K8s pods or Docker containers, sending them back as status updates. This
is more reliable than having each container manage its own log uploads.
"""

import os
import time
import logging
import threading
from typing import Dict, Optional
from collections import defaultdict

from nvidia_tao_core.microservices.utils.log_streaming_utils import (
    get_job_logs_from_backend
)
from nvidia_tao_core.microservices.utils.handler_utils import upload_log_to_cloud
from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_handler_metadata

logger = logging.getLogger(__name__)


class LogMonitor:
    """Monitors logs from running jobs and sends them as callbacks to the server.

    This class provides server-side log monitoring for K8s and Docker backends,
    eliminating the need for containers to manage their own log uploads.
    """

    def __init__(self, backend: Optional[str] = None, poll_interval: int = 10):
        """Initialize the log monitor.

        Args:
            backend (str, optional): Backend type ('local-k8s', 'local-docker', etc.)
                                    If None, reads from BACKEND env var
            poll_interval (int): Interval in seconds between log fetches
        """
        self.backend = backend or os.getenv("BACKEND", "local-k8s")
        self.poll_interval = poll_interval
        self.monitored_jobs: Dict[str, dict] = {}
        self.monitor_threads: Dict[str, threading.Thread] = {}
        self.stop_events: Dict[str, threading.Event] = {}
        self.log_positions: Dict[str, int] = defaultdict(int)  # Track byte position for each job
        self.lock = threading.Lock()

    def add_job(self, job_id: str, callback_url: Optional[str] = None,
                namespace: Optional[str] = None, metadata: Optional[dict] = None):
        """Add a job to be monitored for logs.

        Args:
            job_id (str): The job ID
            callback_url (str, optional): URL to send log updates to
            namespace (str, optional): Kubernetes namespace (for k8s backend)
            metadata (dict, optional): Additional job metadata
        """
        logger.debug(
            f"[LOG_MONITOR] add_job called: job_id={job_id}, callback_url={callback_url}, "
            f"namespace={namespace}, metadata={metadata}"
        )
        with self.lock:
            if job_id in self.monitored_jobs:
                logger.warning(f"[LOG_MONITOR] Job {job_id} is already being monitored, skipping add")
                return

            self.monitored_jobs[job_id] = {
                'callback_url': callback_url,
                'namespace': namespace,
                'metadata': metadata or {},
                'start_time': time.time(),
                'last_update': time.time()
            }
            logger.debug(f"[LOG_MONITOR] Added job {job_id} to monitored_jobs dict")

            # Create and start monitoring thread
            stop_event = threading.Event()
            self.stop_events[job_id] = stop_event
            logger.debug(f"[LOG_MONITOR] Created stop_event for job {job_id}")

            monitor_thread = threading.Thread(
                target=self._monitor_job_logs,
                args=(job_id, stop_event),
                daemon=True,
                name=f"LogMonitor-{job_id[:8]}"
            )
            self.monitor_threads[job_id] = monitor_thread
            logger.debug(f"[LOG_MONITOR] Created monitoring thread for job {job_id}")

            monitor_thread.start()
            logger.info(f"[LOG_MONITOR] Started log monitoring thread for job {job_id}")

    def remove_job(self, job_id: str):
        """Stop monitoring a job.

        Args:
            job_id (str): The job ID to stop monitoring
        """
        logger.debug(f"[LOG_MONITOR] remove_job called for job_id={job_id}")
        with self.lock:
            if job_id not in self.monitored_jobs:
                logger.warning(f"[LOG_MONITOR] Job {job_id} is not being monitored, skipping removal")
                return

            logger.debug(f"[LOG_MONITOR] Signaling monitoring thread to stop for job {job_id}")
            # Signal the monitoring thread to stop
            if job_id in self.stop_events:
                self.stop_events[job_id].set()
                logger.debug(f"[LOG_MONITOR] Stop event set for job {job_id}")

            # Wait for thread to finish (with timeout)
            if job_id in self.monitor_threads:
                logger.debug(f"[LOG_MONITOR] Waiting for monitoring thread to finish (timeout=5s) for job {job_id}")
                self.monitor_threads[job_id].join(timeout=5)
                if self.monitor_threads[job_id].is_alive():
                    logger.warning(f"[LOG_MONITOR] Monitoring thread did not stop within timeout for job {job_id}")
                else:
                    logger.debug(f"[LOG_MONITOR] Monitoring thread stopped successfully for job {job_id}")
                del self.monitor_threads[job_id]

            # Clean up
            logger.debug(f"[LOG_MONITOR] Cleaning up resources for job {job_id}")
            del self.monitored_jobs[job_id]
            if job_id in self.stop_events:
                del self.stop_events[job_id]
            if job_id in self.log_positions:
                final_position = self.log_positions[job_id]
                logger.debug(f"[LOG_MONITOR] Final log position for job {job_id}: {final_position} bytes")
                del self.log_positions[job_id]

            logger.info(f"[LOG_MONITOR] Successfully stopped log monitoring for job {job_id}")

    def _monitor_job_logs(self, job_id: str, stop_event: threading.Event):
        """Monitor logs for a specific job in a separate thread.

        Args:
            job_id (str): The job ID to monitor
            stop_event (threading.Event): Event to signal thread to stop
        """
        logger.debug(f"[LOG_MONITOR] _monitor_job_logs thread starting for job_id={job_id}")
        job_info = self.monitored_jobs.get(job_id)
        if not job_info:
            logger.error(f"[LOG_MONITOR] Job info not found for job_id={job_id}, thread exiting")
            return

        namespace = job_info.get('namespace')
        callback_url = job_info.get('callback_url')

        logger.info(
            f"[LOG_MONITOR] Log monitoring thread started for job {job_id}, "
            f"namespace={namespace}, callback_url={callback_url}, "
            f"poll_interval={self.poll_interval}s"
        )

        poll_count = 0
        try:
            while not stop_event.is_set():
                poll_count += 1
                logger.debug(f"[LOG_MONITOR] Poll #{poll_count} for job_id={job_id}")
                try:
                    # Get logs from backend
                    logger.debug(
                        f"[LOG_MONITOR] Calling get_job_logs_from_backend for job_id={job_id}"
                    )
                    logs = get_job_logs_from_backend(
                        job_id
                    )

                    if logs:
                        # Track what we've already sent
                        last_position = self.log_positions[job_id]
                        new_logs = logs[last_position:]

                        logger.debug(
                            f"[LOG_MONITOR] Got {len(logs)} bytes total, {len(new_logs)} bytes new "
                            f"for job_id={job_id} (last_position={last_position})"
                        )

                        if new_logs:
                            # Send new logs as callback
                            if callback_url:
                                logger.debug(
                                    f"[LOG_MONITOR] Sending {len(new_logs)} bytes to callback "
                                    f"for job_id={job_id}"
                                )
                                self._send_log_callback(job_id, new_logs, callback_url)
                            else:
                                logger.debug(f"[LOG_MONITOR] No callback_url for job_id={job_id}, skipping callback")

                            # Update position
                            self.log_positions[job_id] = len(logs)
                            job_info['last_update'] = time.time()

                            logger.debug(f"[LOG_MONITOR] Updated position to {len(logs)} for job_id={job_id}")

                            # Write logs to local filesystem immediately (for both brain and recommendations)
                            cached_log_file_path = None
                            try:
                                handler_id = job_info.get('metadata', {}).get('handler_id')
                                if handler_id:
                                    handler_metadata = get_handler_metadata(
                                        handler_id,
                                        kind=job_info.get('metadata', {}).get('handler_kind', 'experiment')
                                    )
                                    if handler_metadata:
                                        user_id = handler_metadata.get("user_id")
                                        org_name = handler_metadata.get("org_name")

                                        if user_id and org_name:
                                            # Determine log file path
                                            from nvidia_tao_core.microservices.utils.stateless_handler_utils import (  # noqa: E501
                                                get_handler_log_root
                                            )
                                            log_dir = get_handler_log_root(user_id, org_name, handler_id)
                                            cached_log_file_path = os.path.join(log_dir, f"{job_id}.txt")

                                            # Ensure directory exists
                                            os.makedirs(os.path.dirname(cached_log_file_path), exist_ok=True)
                                            # Write logs to file
                                            with open(cached_log_file_path, 'w', encoding='utf-8') as f:
                                                f.write(logs)
                                            logger.debug(
                                                f"[LOG_MONITOR] Wrote {len(logs)} bytes to local file: "
                                                f"{cached_log_file_path}"
                                            )
                            except Exception as e:
                                logger.warning(
                                    f"[LOG_MONITOR] Failed to write logs to local filesystem for "
                                    f"job_id={job_id}: {type(e).__name__}: {e}"
                                )
                                cached_log_file_path = None

                            # Upload to cloud storage for persistence (periodically)
                            # Only upload every N polls to avoid excessive uploads
                        else:
                            logger.debug(f"[LOG_MONITOR] No new logs for job_id={job_id}")

                        if poll_count % 6 == 0:  # Every 6 polls (~60 seconds at 10s interval)
                            logger.info(
                                f"[LOG_MONITOR] Poll #{poll_count} - "
                                f"Triggering cloud upload for job_id={job_id}"
                            )
                            try:
                                # Use the cached log file path if available, otherwise create temp file
                                if cached_log_file_path and os.path.exists(cached_log_file_path):
                                    log_path_to_upload = cached_log_file_path
                                    should_delete_after = False
                                    logger.info(f"[LOG_MONITOR] Using cached log file for upload: {log_path_to_upload}")
                                else:
                                    # Fallback: create temp file (shouldn't normally happen)
                                    logger.warning(
                                        "[LOG_MONITOR] Cached log file not available, "
                                        "creating temp file for upload"
                                    )
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(
                                        mode='w', delete=False, suffix='.txt'
                                    ) as tmp_file:
                                        tmp_file.write(logs)
                                        log_path_to_upload = tmp_file.name
                                    should_delete_after = True
                                handler_id = job_info.get('metadata', {}).get('handler_id')
                                logger.info(f"[LOG_MONITOR] handler_id from metadata: {handler_id}")
                                if handler_id:
                                    handler_kind = job_info.get('metadata', {}).get('handler_kind', 'experiment')
                                    logger.info(
                                        f"[LOG_MONITOR] Fetching handler metadata for "
                                        f"handler_id={handler_id}, kind={handler_kind}"
                                    )
                                    handler_metadata = get_handler_metadata(handler_id, kind=handler_kind)

                                    if handler_metadata:
                                        logger.info(
                                            f"[LOG_MONITOR] Got handler metadata, "
                                            f"uploading logs to cloud for job_id={job_id}"
                                        )
                                        try:
                                            upload_success = upload_log_to_cloud(
                                                handler_metadata, job_id, log_path_to_upload, automl_index=None
                                            )
                                            if upload_success:
                                                logger.info(
                                                    f"[LOG_MONITOR] ✅ Successfully uploaded logs to cloud "
                                                    f"for job_id={job_id}"
                                                )
                                            else:
                                                logger.warning(
                                                    f"[LOG_MONITOR] ❌ Failed to upload logs to cloud "
                                                    f"for job_id={job_id} (non-fatal)"
                                                )
                                        finally:
                                            # Only delete if we created a temp file
                                            if should_delete_after and os.path.exists(log_path_to_upload):
                                                os.remove(log_path_to_upload)
                                    else:
                                        logger.warning(
                                            f"[LOG_MONITOR] Handler metadata not found "
                                            f"for handler_id={handler_id}"
                                        )
                                        if should_delete_after and os.path.exists(log_path_to_upload):
                                            os.remove(log_path_to_upload)
                                else:
                                    logger.warning(
                                        f"[LOG_MONITOR] No handler_id in metadata for "
                                        f"job_id={job_id}, skipping cloud upload"
                                    )
                                    logger.warning(
                                        f"[LOG_MONITOR] Available metadata keys: "
                                        f"{list(job_info.get('metadata', {}).keys())}"
                                    )
                                    if should_delete_after and os.path.exists(log_path_to_upload):
                                        os.remove(log_path_to_upload)
                            except Exception as e:
                                logger.error(
                                    f"[LOG_MONITOR] Exception uploading logs to cloud for job_id={job_id}: "
                                    f"{type(e).__name__}: {e}"
                                )
                                logger.debug("[LOG_MONITOR] Full exception:", exc_info=True)
                    else:
                        logger.debug(
                            f"[LOG_MONITOR] No logs retrieved for job_id={job_id} "
                            f"(pod/container may not be ready yet)"
                        )

                except Exception as e:
                    logger.error(
                        f"[LOG_MONITOR] Error fetching logs for job {job_id} (poll #{poll_count}): "
                        f"{type(e).__name__}: {e}"
                    )
                    logger.debug("[LOG_MONITOR] Exception details:", exc_info=True)

                # Wait before next poll (or until stop event)
                logger.debug(
                    f"[LOG_MONITOR] Waiting {self.poll_interval}s before next poll "
                    f"for job_id={job_id}"
                )
                stop_event.wait(self.poll_interval)

        except Exception as e:
            logger.error(
                f"[LOG_MONITOR] Fatal error in log monitoring thread for job {job_id}: "
                f"{type(e).__name__}: {e}"
            )
            logger.debug("[LOG_MONITOR] Fatal exception details:", exc_info=True)
        finally:
            logger.info(f"[LOG_MONITOR] Log monitoring thread stopped for job {job_id} after {poll_count} polls")

    def _send_log_callback(self, job_id: str, log_content: str, callback_url: str):
        """Send log content to callback URL.

        Args:
            job_id (str): The job ID
            log_content (str): The log content to send
            callback_url (str): URL to send logs to
        """
        try:
            import requests

            # Prepare callback data
            data = {
                'job_id': job_id,
                'log_contents': log_content,
                'timestamp': time.time()
            }

            # Get auth headers if available
            headers = {}
            ngc_key = os.getenv("TAO_ADMIN_KEY") or os.getenv("TAO_USER_KEY")
            if ngc_key:
                headers["Authorization"] = f"Bearer {ngc_key}"

            headers['Content-Type'] = 'application/json'

            # Send callback
            response = requests.post(
                callback_url,
                json=data,
                headers=headers,
                timeout=30
            )

            if response.ok:
                logger.debug(f"Successfully sent log callback for job {job_id}")
            else:
                logger.warning(
                    f"Log callback failed for job {job_id}: "
                    f"status={response.status_code}"
                )

        except Exception as e:
            logger.error(f"Error sending log callback for job {job_id}: {e}")

    def get_monitored_jobs(self):
        """Get list of currently monitored jobs.

        Returns:
            list: List of job IDs being monitored
        """
        with self.lock:
            return list(self.monitored_jobs.keys())

    def shutdown(self):
        """Shutdown the log monitor and stop all monitoring threads."""
        logger.info("Shutting down log monitor...")

        # Get list of jobs to remove
        with self.lock:
            jobs_to_remove = list(self.monitored_jobs.keys())

        # Remove all jobs
        for job_id in jobs_to_remove:
            self.remove_job(job_id)

        logger.info("Log monitor shutdown complete")


# Global log monitor instance
_log_monitor: Optional[LogMonitor] = None
_log_monitor_lock = threading.Lock()


def get_log_monitor() -> LogMonitor:
    """Get the global log monitor instance (singleton pattern).

    Returns:
        LogMonitor: The global log monitor instance
    """
    global _log_monitor  # pylint: disable=global-statement

    logger.debug("[LOG_MONITOR] get_log_monitor called")
    if _log_monitor is None:
        logger.debug("[LOG_MONITOR] Log monitor not initialized, acquiring lock")
        with _log_monitor_lock:
            if _log_monitor is None:
                backend = os.getenv("BACKEND", "local-k8s")
                logger.debug(f"[LOG_MONITOR] Checking backend: {backend}")
                # Only create log monitor for k8s and docker backends
                if backend in ("local-k8s", "local-docker"):
                    logger.info(f"[LOG_MONITOR] Creating global log monitor for backend: {backend}")
                    _log_monitor = LogMonitor(backend=backend)
                    logger.info(f"[LOG_MONITOR] Created global log monitor for backend: {backend}")
                else:
                    logger.info(f"[LOG_MONITOR] Log monitor not needed for backend: {backend}")
    else:
        logger.debug("[LOG_MONITOR] Returning existing log monitor instance")

    return _log_monitor


def start_monitoring_job(job_id: str, callback_url: Optional[str] = None,
                         namespace: Optional[str] = None, metadata: Optional[dict] = None):
    """Start monitoring logs for a job.

    Args:
        job_id (str): The job ID
        callback_url (str, optional): URL to send log updates to
        namespace (str, optional): Kubernetes namespace (for k8s backend)
        metadata (dict, optional): Additional job metadata
    """
    logger.debug(
        f"[LOG_MONITOR] start_monitoring_job called for job_id={job_id}, "
        f"callback_url={callback_url}, namespace={namespace}"
    )
    monitor = get_log_monitor()
    if monitor:
        logger.debug(f"[LOG_MONITOR] Monitor available, calling add_job for job_id={job_id}")
        monitor.add_job(job_id, callback_url, namespace, metadata)
    else:
        logger.debug(f"[LOG_MONITOR] Log monitor not available for job {job_id} (backend may not support monitoring)")


def stop_monitoring_job(job_id: str):
    """Stop monitoring logs for a job.

    Args:
        job_id (str): The job ID
    """
    logger.debug(f"[LOG_MONITOR] stop_monitoring_job called for job_id={job_id}")
    monitor = get_log_monitor()
    if monitor:
        logger.debug(f"[LOG_MONITOR] Monitor available, calling remove_job for job_id={job_id}")
        monitor.remove_job(job_id)
    else:
        logger.debug(f"[LOG_MONITOR] Log monitor not available for job {job_id}")
