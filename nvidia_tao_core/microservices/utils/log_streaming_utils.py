# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Log streaming utilities for Kubernetes and Docker.

This module provides direct log streaming from Kubernetes pods and Docker containers
without relying on the container's internal log capture mechanism. This makes log
retrieval more reliable and real-time.
"""

import os
import logging
from typing import Optional, Generator
from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import ExecutionHandler
from nvidia_tao_core.microservices.utils.stateless_handler_utils import BACKEND, get_handler_metadata

logger = logging.getLogger(__name__)


def get_k8s_pod_logs(job_id: str, namespace: Optional[str] = None, tail_lines: Optional[int] = None) -> Optional[str]:
    """Get logs directly from Kubernetes pod using Kubernetes Python client.

    For multi-node jobs (StatefulSet), this automatically identifies and retrieves
    logs from the master node (identified by -0 suffix).

    Args:
        job_id (str): The job ID (used as pod/job name)
        namespace (str, optional): Kubernetes namespace. If None, uses default from config
        tail_lines (int, optional): Number of lines to tail. If None, gets all logs

    Returns:
        str: The log content, or None if logs cannot be retrieved
    """
    logger.debug(
        f"[K8S_LOGS] Starting get_k8s_pod_logs for job_id={job_id}, "
        f"namespace={namespace}, tail_lines={tail_lines}"
    )
    try:
        from kubernetes import client, config as k8s_config
        from kubernetes.client.rest import ApiException

        # Load Kubernetes configuration
        logger.debug(f"[K8S_LOGS] Loading Kubernetes configuration for job_id={job_id}")
        try:
            k8s_config.load_incluster_config()
            logger.debug(f"[K8S_LOGS] Successfully loaded in-cluster config for job_id={job_id}")
        except Exception as e1:
            logger.debug(f"[K8S_LOGS] In-cluster config failed: {e1}, trying kube config")
            try:
                k8s_config.load_kube_config()
                logger.debug(f"[K8S_LOGS] Successfully loaded kube config for job_id={job_id}")
            except Exception as e2:
                logger.error(
                    f"[K8S_LOGS] Failed to load any Kubernetes config for job_id={job_id}: "
                    f"in-cluster={e1}, kube={e2}"
                )
                return None

        # Get namespace
        if namespace is None:
            try:
                namespace_file = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
                logger.debug(f"[K8S_LOGS] Reading namespace from {namespace_file}")
                with open(namespace_file, 'r', encoding='utf-8') as f:
                    namespace = f.read().strip()
                logger.debug(f"[K8S_LOGS] Got namespace from service account: {namespace}")
            except Exception as e:
                namespace = os.getenv("NAMESPACE", "default")
                logger.debug(f"[K8S_LOGS] Using fallback namespace: {namespace} (error: {e})")
        else:
            logger.debug(f"[K8S_LOGS] Using provided namespace: {namespace}")

        v1 = client.CoreV1Api()
        logger.debug(f"[K8S_LOGS] Created CoreV1Api client for job_id={job_id}, namespace={namespace}")

        # Try to find the pod - could be a Job pod or StatefulSet pod
        pod_name = None
        logger.debug(f"[K8S_LOGS] Searching for pod for job_id={job_id} in namespace={namespace}")
        try:
            # First, try to list pods with the job_id as a label selector
            label_selector = f"job-name={job_id}"
            logger.debug(f"[K8S_LOGS] Trying label selector: {label_selector}")
            pods = v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)

            if pods.items:
                # Found pods with job-name label (regular Kubernetes Job)
                pod_name = pods.items[0].metadata.name
                pod_status = pods.items[0].status.phase
                logger.info(
                    f"[K8S_LOGS] Found Job pod: {pod_name} (status={pod_status}) "
                    f"for job_id: {job_id}"
                )
                logger.debug(f"[K8S_LOGS] Pod details: namespace={namespace}, labels={pods.items[0].metadata.labels}")
            else:
                logger.debug(
                    f"[K8S_LOGS] No pods found with label selector {label_selector}, "
                    f"trying StatefulSet naming"
                )
                # Try StatefulSet naming convention
                # For StatefulSet, pods are named as <statefulset-name>-<ordinal>
                # For multi-node, master is -0, workers are -1, -2, etc.
                master_pod_name = f"{job_id}-0"
                logger.debug(f"[K8S_LOGS] Trying StatefulSet master pod: {master_pod_name}")
                try:
                    pod = v1.read_namespaced_pod(name=master_pod_name, namespace=namespace)
                    pod_name = master_pod_name
                    pod_status = pod.status.phase
                    logger.info(
                        f"[K8S_LOGS] Found StatefulSet master pod: {pod_name} "
                        f"(status={pod_status}) for job_id: {job_id}"
                    )
                    logger.debug(f"[K8S_LOGS] Pod details: namespace={namespace}, labels={pod.metadata.labels}")
                except ApiException as e1:
                    logger.debug(
                        f"[K8S_LOGS] Master pod {master_pod_name} not found "
                        f"(status={e1.status}), trying single pod name"
                    )
                    # Try single pod name (might be a single-node job)
                    try:
                        pod = v1.read_namespaced_pod(name=job_id, namespace=namespace)
                        pod_name = job_id
                        pod_status = pod.status.phase
                        logger.info(
                            f"[K8S_LOGS] Found single pod: {pod_name} (status={pod_status}) "
                            f"for job_id: {job_id}"
                        )
                        logger.debug(f"[K8S_LOGS] Pod details: namespace={namespace}, labels={pod.metadata.labels}")
                    except ApiException as e2:
                        logger.debug("[K8S_LOGS] Direct pod names failed, trying partial name match")
                        # Try partial name match - list all pods and find one containing the job_id
                        try:
                            all_pods = v1.list_namespaced_pod(namespace=namespace)
                            matching_pods = [p for p in all_pods.items if job_id in p.metadata.name]
                            if matching_pods:
                                # Prefer pods ending with -0 (master node) if multiple matches
                                master_pods = [p for p in matching_pods if p.metadata.name.endswith('-0')]
                                selected_pod = master_pods[0] if master_pods else matching_pods[0]
                                pod_name = selected_pod.metadata.name
                                pod_status = selected_pod.status.phase
                                logger.info(
                                    f"[K8S_LOGS] Found pod by partial name match: {pod_name} "
                                    f"(status={pod_status}) for job_id: {job_id}"
                                )
                                logger.debug(
                                    f"[K8S_LOGS] Pod details: namespace={namespace}, "
                                    f"labels={selected_pod.metadata.labels}"
                                )
                            else:
                                logger.error(
                                    f"[K8S_LOGS] Could not find pod for job_id={job_id}: "
                                    f"master_pod={e1.status}, single_pod={e2.status}, no partial matches"
                                )
                                logger.debug(f"[K8S_LOGS] Tried names: {master_pod_name}, {job_id}, and partial match")
                                return None
                        except Exception as e3:
                            logger.error(
                                f"[K8S_LOGS] Error during partial name match for job_id={job_id}: "
                                f"{type(e3).__name__}: {e3}"
                            )
                            logger.debug(f"[K8S_LOGS] Tried names: {master_pod_name}, {job_id}")
                            return None

        except Exception as e:
            logger.error(f"[K8S_LOGS] Exception finding pod for job_id={job_id}: {type(e).__name__}: {e}")
            logger.debug("[K8S_LOGS] Full traceback:", exc_info=True)
            return None

        if not pod_name:
            logger.error(f"[K8S_LOGS] No pod found for job_id={job_id} after all attempts")
            return None

        # Get logs from the pod
        logger.debug(f"[K8S_LOGS] Retrieving logs from pod={pod_name}, namespace={namespace}, tail_lines={tail_lines}")
        try:
            log_kwargs = {
                'name': pod_name,
                'namespace': namespace,
                'timestamps': False,
            }

            if tail_lines is not None:
                log_kwargs['tail_lines'] = tail_lines
                logger.debug(f"[K8S_LOGS] Using tail_lines={tail_lines}")

            logger.debug(f"[K8S_LOGS] Calling read_namespaced_pod_log with kwargs: {log_kwargs}")
            logs = v1.read_namespaced_pod_log(**log_kwargs)
            log_line_count = len(logs.splitlines())
            log_size_kb = len(logs) / 1024
            logger.info(
                f"[K8S_LOGS] Successfully retrieved {log_line_count} lines "
                f"({log_size_kb:.1f} KB) from pod {pod_name}"
            )
            return logs

        except ApiException as e:
            if e.status == 400:
                # Pod might not have started yet or container not ready
                logger.warning(
                    f"[K8S_LOGS] Pod {pod_name} logs not available yet: "
                    f"status={e.status}, reason={e.reason}"
                )
                logger.debug(f"[K8S_LOGS] ApiException details: {e.body}")
                return None
            logger.error(
                f"[K8S_LOGS] Error retrieving logs from pod {pod_name}: "
                f"status={e.status}, reason={e.reason}"
            )
            logger.debug(f"[K8S_LOGS] ApiException details: {e.body}")
            return None

    except ImportError as e:
        logger.error(
            f"[K8S_LOGS] Kubernetes Python client not installed: {e}. "
            f"Install with: pip install kubernetes"
        )
        return None
    except Exception as e:
        logger.error(
            f"[K8S_LOGS] Unexpected error getting Kubernetes pod logs for "
            f"job_id={job_id}: {type(e).__name__}: {e}"
        )
        logger.debug("[K8S_LOGS] Full traceback:", exc_info=True)
        return None


def stream_k8s_pod_logs(job_id: str, namespace: Optional[str] = None,
                        follow: bool = True) -> Optional[Generator[str, None, None]]:
    """Stream logs from Kubernetes pod in real-time.

    For multi-node jobs (StatefulSet), this automatically identifies and streams
    logs from the master node (identified by -0 suffix).

    Args:
        job_id (str): The job ID (used as pod/job name)
        namespace (str, optional): Kubernetes namespace. If None, uses default from config
        follow (bool): Whether to follow the log stream (tail -f behavior)

    Yields:
        str: Log lines as they are produced
    """
    try:
        from kubernetes import client, config as k8s_config, watch
        from kubernetes.client.rest import ApiException

        # Load Kubernetes configuration
        try:
            k8s_config.load_incluster_config()
        except Exception:
            try:
                k8s_config.load_kube_config()
            except Exception as e:
                logger.error(f"Failed to load Kubernetes config: {e}")
                return None

        # Get namespace
        if namespace is None:
            try:
                namespace_file = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
                with open(namespace_file, 'r', encoding='utf-8') as f:
                    namespace = f.read().strip()
            except Exception:
                namespace = os.getenv("NAMESPACE", "default")

        v1 = client.CoreV1Api()

        # Find the pod (same logic as get_k8s_pod_logs)
        pod_name = None
        try:
            label_selector = f"job-name={job_id}"
            pods = v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)

            if pods.items:
                pod_name = pods.items[0].metadata.name
            else:
                master_pod_name = f"{job_id}-0"
                try:
                    v1.read_namespaced_pod(name=master_pod_name, namespace=namespace)
                    pod_name = master_pod_name
                except ApiException:
                    try:
                        v1.read_namespaced_pod(name=job_id, namespace=namespace)
                        pod_name = job_id
                    except ApiException:
                        logger.error(f"Could not find pod for job_id: {job_id}")
                        return None

        except Exception as e:
            logger.error(f"Error finding pod for job_id {job_id}: {e}")
            return None

        if not pod_name:
            return None

        # Stream logs
        w = watch.Watch()
        try:
            for line in w.stream(
                v1.read_namespaced_pod_log,
                name=pod_name,
                namespace=namespace,
                follow=follow,
                timestamps=False
            ):
                yield line + '\n'
        except ApiException as e:
            if e.status == 400:
                logger.warning(f"Pod {pod_name} logs not available for streaming yet: {e.reason}")
            else:
                logger.error(f"Error streaming logs from pod {pod_name}: {e}")
            return None
        finally:
            w.stop()
        return None

    except ImportError:
        logger.error("Kubernetes Python client not installed")
        return None
    except Exception as e:
        logger.error(f"Unexpected error streaming Kubernetes pod logs: {e}")
        return None


def get_job_logs_from_backend(
    job_id: str, tail_lines: Optional[int] = None
) -> Optional[str]:
    """Get logs from the appropriate backend (Kubernetes, Docker, SLURM, or Lepton).

    This is a unified interface that automatically determines the backend
    and retrieves logs accordingly.

    Args:
        job_id (str): The job ID
        tail_lines (int, optional): Number of lines to tail

    Returns:
        str: The log content, or None if logs cannot be retrieved or backend not supported
    """
    handler_metadata = get_handler_metadata(job_id)
    workspace_id = handler_metadata.get("workspace", "")
    workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
    handler = ExecutionHandler.create_handler(workspace_metadata=workspace_metadata, backend=BACKEND, job_id=job_id)
    if not handler:
        logger.error(f"Unable to determine appropriate handler for backend '{BACKEND}' and job_id '{job_id}'")
        return None
    logs = handler.get_job_logs(job_id, tail_lines)

    if not logs:
        logger.info(
            f"[LOG_BACKEND] Backend {handler.backend_type} not supported for direct log streaming, "
            f"will use fallback method"
        )
        return None

    logger.info(
        f"[LOG_BACKEND] Successfully retrieved logs for job_id={job_id} from backend={handler.backend_type}"
    )
    return logs
