# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes handler"""

import os
import time
import traceback
import inspect
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    get_handler_job_metadata,
    write_job_metadata,
    get_dnn_status,
    BACKEND
)
from nvidia_tao_core.microservices.utils.executor_utils import (
    get_owner_reference,
    release_name,
    get_available_local_k8s_gpus
)
from nvidia_tao_core.microservices.enum_constants import Backend
from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import ExecutionHandler


class KubernetesHandler(ExecutionHandler):
    """Kubernetes handler"""

    def __init__(self):
        """Initialize Kubernetes handler"""
        super().__init__(backend_type=Backend.LOCAL_K8S)
        self.release_name = release_name

    def get_owner_reference(self):
        """Get the owner reference for K8s resources"""
        return get_owner_reference()

    def get_available_instances(self):
        """Get available GPUs in the K8s cluster"""
        local_k8s_gpus = get_available_local_k8s_gpus()
        if local_k8s_gpus:
            return local_k8s_gpus
        self.logger.error("No available GPUs found in the K8s cluster")
        return {}

    def get_namespace(self):
        """Get the namespace of the environment"""
        return os.getenv('NAMESPACE', default='default')

    def create_job(self, org_name, job_name, image, command, num_gpu=-1, num_nodes=1, accelerator=None,
                   docker_env_vars=None, port=False, nv_job_metadata=None, automl_brain=False,
                   automl_exp_job=False, local_cluster=False):
        """Creates a kubernetes job

        Args:
            org_name: Organization name
            job_name: Job name
            image: Container image
            command: Command to run
            num_gpu: Number of GPUs (-1 for default)
            num_nodes: Number of nodes
            accelerator: Accelerator type
            docker_env_vars: Docker environment variables
            port: Port flag
            nv_job_metadata: job metadata
            automl_brain: Whether this is an AutoML brain job
            automl_exp_job: Whether this is an AutoML experiment job
            local_cluster: Whether this is a local cluster
        """
        name_space = self.get_namespace()
        host_base_url = os.getenv("HOSTBASEURL", "no_url")
        if host_base_url == "no_url":
            raise ValueError(
                f"Base URL not set in values yaml. Please set it as "
                f"http(s)://<ip_address>:{self.release_name}-ingress-nginx-controller service's port number>"
            )

        command = 'umask 0 && ' + command
        if num_gpu == -1:
            num_gpu = int(os.getenv('NUM_GPU_PER_NODE', default='1'))

        node_selector = None
        if accelerator:
            available_gpus = self.get_available_local_k8s_gpus()
            gpu_to_be_run_on = None
            if available_gpus:
                gpu_to_be_run_on = available_gpus.get(accelerator, "")
            node_selector = {'accelerator': gpu_to_be_run_on}

        image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')
        name_space = self.get_namespace()
        api_instance = client.BatchV1Api()

        volume_mounts = []
        if local_cluster:
            if os.getenv("INGRESSENABLED", "false") == "false":
                from nvidia_tao_core.microservices.utils.executor_utils import get_cluster_ip
                in_cluster_ip, cluster_port = get_cluster_ip()
            else:
                service_name = f"{self.release_name}-ingress-nginx-controller"
                from nvidia_tao_core.microservices.utils.executor_utils import get_service_in_cluster_ip
                in_cluster_ip = get_service_in_cluster_ip(
                    service_name, namespace=name_space
                )
                cluster_port = 80
            # change the host_base_url to the in-cluster ip
            in_cluster_url = f"http://{in_cluster_ip}:{cluster_port}" if nv_job_metadata is None else None
            if "TAO_API_SERVER" in docker_env_vars:
                docker_env_vars["TAO_API_SERVER"] = docker_env_vars[
                    "TAO_API_SERVER"
                ].replace(host_base_url, in_cluster_url)
            docker_env_vars["TAO_LOGGING_SERVER_URL"] = (
                docker_env_vars["TAO_LOGGING_SERVER_URL"].replace(host_base_url, in_cluster_url)
            )
        dshm_volume_mount = client.V1VolumeMount(
            name="dshm",
            mount_path="/dev/shm")
        volume_mounts.append(dshm_volume_mount)

        # Add SSH volume mount for AutoML brain jobs
        if automl_brain:
            host_ssh_path = os.getenv('HOST_SSH_PATH')
            if host_ssh_path:
                ssh_volume_mount = client.V1VolumeMount(
                    name="ssh-keys",
                    mount_path="/root/.ssh",
                    read_only=True)
                volume_mounts.append(ssh_volume_mount)

        resources = client.V1ResourceRequirements(
            limits={
                'nvidia.com/gpu': str(num_gpu)
            })
        capabilities = client.V1Capabilities(
            add=['SYS_PTRACE']
        )
        security_context = client.V1SecurityContext(
            capabilities=capabilities
        )

        # Get mongo configuration if backend is set
        backend_env = client.V1EnvVar(
            name="BACKEND",
            value=BACKEND.value)
        # CL job needs to set the environment variable to pass GPU checks (validate_num_gpu) for training jobs
        num_gpu_env = client.V1EnvVar(
            name="NUM_GPU_PER_NODE",
            value=str(num_gpu))

        dynamic_docker_envs = []
        if os.getenv("BACKEND"):
            from nvidia_tao_core.microservices.utils.mongo_utils import (
                mongo_secret,
                mongo_operator_enabled,
                mongo_namespace
            )
            mongo_secret_env = client.V1EnvVar(
                name="MONGOSECRET",
                value=mongo_secret
            )
            mongo_operator_enabled_env = client.V1EnvVar(
                name="MONGO_OPERATOR_ENABLED",
                value=str(mongo_operator_enabled).lower()
            )
            mongo_namespace_env = client.V1EnvVar(
                name="NAMESPACE",
                value=mongo_namespace
            )
            dynamic_docker_envs.extend([mongo_secret_env, mongo_operator_enabled_env, mongo_namespace_env])

        if docker_env_vars:
            for docker_env_var_key, docker_env_var_value in docker_env_vars.items():
                kubernetes_env = client.V1EnvVar(
                    name=docker_env_var_key,
                    value=docker_env_var_value)
                dynamic_docker_envs.append(kubernetes_env)

        container = client.V1Container(
            name="container",
            image=image,
            env=[backend_env, num_gpu_env] + dynamic_docker_envs,
            command=["/bin/bash", "-c"],
            args=[command],
            resources=resources,
            volume_mounts=volume_mounts,
            security_context=security_context)
        dshm_volume = client.V1Volume(
            name="dshm",
            empty_dir=client.V1EmptyDirVolumeSource(medium='Memory'))

        # Define volumes list with dshm
        volumes = [dshm_volume]

        # Add SSH volume for AutoML brain jobs
        if automl_brain:
            host_ssh_path = os.getenv('HOST_SSH_PATH')
            if host_ssh_path:
                ssh_volume = client.V1Volume(
                    name="ssh-keys",
                    host_path=client.V1HostPathVolumeSource(
                        path=host_ssh_path,
                        type="Directory"))
                volumes.append(ssh_volume)

        restart_policy = "Always"
        if automl_brain:
            restart_policy = "Never"
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={"purpose": "tao-toolkit-job"}
            ),
            spec=client.V1PodSpec(
                image_pull_secrets=[client.V1LocalObjectReference(name=image_pull_secret)],
                containers=[container],
                volumes=volumes,
                node_selector=node_selector,
                restart_policy=restart_policy))
        spec = client.V1JobSpec(
            ttl_seconds_after_finished=100,
            template=template,
            backoff_limit=0)

        # Create metadata with Helm annotations for proper lifecycle management
        # This ensures the Job is deleted when helm delete is run
        job_metadata = {
            "name": job_name,
            "owner_references": [self.get_owner_reference()]
        }

        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(**job_metadata),
            spec=spec)

        try:
            api_instance.create_namespaced_job(
                body=job,
                namespace=name_space)

            # Update job message with initial status (checking for image)
            if job_name:
                self.update_image_pull_status(job_name, image, "checking")

            return
        except Exception as e:
            self.logger.error(f"Exception thrown in executor create is {str(e)}")
            self.logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to create K8s job '{job_name}': {e}") from e

    def check_and_update_job_image_pull_status(self, job_name, namespace=None):
        """Check image pull status for a K8s Job and update job message.

        This method can be called from the workflow to monitor image pull status
        during job execution.

        Args:
            job_name (str): Name of the K8s Job
            namespace (str, optional): Kubernetes namespace

        Returns:
            str: Current status - "pulling", "complete", "error", "waiting", or "unknown"
        """
        try:
            if namespace is None:
                namespace = self.get_namespace()

            core_v1 = client.CoreV1Api()

            # Find pods belonging to this job
            pods = core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"job-name={job_name}"
            )

            if not pods.items:
                return "waiting"

            # Check first pod
            pod = pods.items[0]
            status, image, error_msg = self._get_pod_image_pull_status(
                pod.metadata.name, namespace
            )

            # Update job message based on status
            if status in ("pulling", "extracting", "complete", "error", "auth_error", "not_exists_in_registry"):
                self.update_image_pull_status(job_name, image, status, error_message=error_msg)

            return status

        except Exception as e:
            self.logger.warning(f"Error checking job image pull status: {e}")
            return "unknown"

    def delete_job(self, job_name):
        """Deletes a kubernetes job

        Args:
            job_name: Job name
            use_ngc: Whether to use NGC (unused for K8s)
        """
        name_space = self.get_namespace()
        if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
            config.load_kube_config()
        else:
            config.load_incluster_config()

        api_instance = client.BatchV1Api()
        try:
            # Delete associated flask service
            self.delete_service(job_id=job_name, service_type="flask")

            api_response = api_instance.delete_namespaced_job(
                name=job_name,
                namespace=name_space,
                body=client.V1DeleteOptions(
                    propagation_policy='Foreground',
                    grace_period_seconds=5))
            self.logger.info(f"Job deleted. status='{str(api_response.status)}'")
            return
        except Exception as e:
            self.logger.error(f"Exception caught in delete_job {str(e)}")
            self.logger.error("Job failed to delete.")
            return

    def _delete_flask_service(self, job_id):
        """Delete flask service associated with a job

        Args:
            job_id: Job identifier
        """
        service_name = f"flask-service-{job_id}"
        namespace = self.get_namespace()
        api_instance = client.CoreV1Api()
        try:
            api_instance.delete_namespaced_service(
                name=service_name,
                namespace=namespace
            )
            self.logger.info(f"Flask service {service_name} deleted")
        except Exception as e:
            self.logger.debug(f"Flask service {service_name} not found or already deleted: {e}")

    def list_namespace_jobs(self):
        """List kubernetes jobs in a namespace

        Returns:
            API response with list of jobs
        """
        name_space = self.get_namespace()
        api_instance = client.BatchV1Api()
        api_response = None
        try:
            api_response = api_instance.list_namespaced_job(
                namespace=name_space,
                label_selector="purpose=tao-toolkit-job",
                watch=False,
                limit=1000
            )
        except Exception as e:
            self.logger.error(f"Exception thrown in list_namespace_jobs is {str(e)}")
            pass
        return api_response

    def create_microservice(
        self,
        job_id,
        api_port=8000,
        num_gpu=1,
        num_nodes=1,
        image=None,
        accelerator=None,
        custom_command=None,
        inference_microservice=False,
        docker_env_vars={}
    ):
        """Create a Kubernetes StatefulSet microservice

        Args:
            job_id: Job/microservice identifier
            api_port: Port for the API service
            num_gpu: Number of GPUs per node
            num_nodes: Number of nodes/replicas
            image: Container image
            accelerator: Accelerator type
            custom_command: Custom command to run

        Returns:
            bool: True if StatefulSet created successfully
        """
        try:
            if inference_microservice and api_port == 8000:
                api_port = 8080  # Default port for inference microservices

            # Use the handler's own create_statefulset method
            self.create_statefulset(
                job_id=job_id,
                num_gpu_per_node=num_gpu,
                num_nodes=num_nodes,
                image=image,
                api_port=api_port,
                accelerator=accelerator,
                statefulset_type="multinode" if not inference_microservice else "inference_microservice",
                custom_command=custom_command,
                custom_env_vars=docker_env_vars
            )

            # StatefulSet creation is successful if no exception was raised
            self.logger.info(f"K8s microservice {job_id} created successfully")

            # Transition normal jobs from Pending → Started so that
            # _reclaim_stale_gpus knows a pod has been created and can
            # safely apply its "no container = stale" heuristic.
            # AutoML experiment sub-jobs are handled separately via the
            # controller recommendation (pending → started) in AutoMLPipeline.run().
            try:
                _meta = get_handler_job_metadata(job_id)
                if _meta and _meta.get("status") in ("Pending", "pending"):
                    _meta["status"] = "Started"
                    write_job_metadata(job_id, _meta)
                    self.logger.info(f"[LIFECYCLE] Job {job_id}: Pending → Started (pod created)")
            except Exception as status_err:
                self.logger.warning(f"[LIFECYCLE] Could not update status to Started for {job_id}: {status_err}")

            return True
        except Exception as e:
            self.logger.error(f"Failed to create K8s microservice: {e}")
            # Re-raise exception with clear message so caller can handle it
            raise RuntimeError(f"Failed to create K8s microservice: {e}") from e

    def send_request_to_microservice(
        self,
        api_endpoint,
        network,
        action,
        cloud_metadata={},
        specs={},
        job_id="",
        docker_env_vars={},
        port=8000,
        statefulset_replicas=1
    ):
        """Send a request to the K8s StatefulSet microservice

        Returns:
            requests.Response: Response from the microservice
        """
        # Construct base URL using StatefulSet FQDN
        statefulset_name = self.get_statefulset_name(job_id)
        statefulset_service_name = self.get_statefulset_service_name(job_id)
        statefulset_namespace = self.get_namespace()

        # StatefulSet pod FQDN format: {statefulset-name}-{replica-index}.{service-name}.{namespace}.svc.cluster.local
        base_url = (
            f"http://{statefulset_name}-0."
            f"{statefulset_service_name}.{statefulset_namespace}."
            f"svc.cluster.local:{port}"
        )

        # Use the base class's send_microservice_request with retry logic
        return self.send_microservice_request(
            base_url=base_url,
            api_endpoint=api_endpoint,
            network=network,
            action=action,
            cloud_metadata=cloud_metadata,
            specs=specs,
            job_id=job_id,
            docker_env_vars=docker_env_vars,
            statefulset_replicas=statefulset_replicas
        )

    def delete(self, job_id, **kwargs):
        """Delete K8s StatefulSet microservice

        Args:
            job_id: Job/microservice identifier
        """
        try:
            resource_type = kwargs.get("resource_type", "multinode")
            deleted = self.delete_statefulset(
                job_id, use_ngc=False, resource_type=resource_type
            )
            if deleted is False:
                return False

            # Foreground deletion is asynchronous at the Kubernetes API
            # boundary. Confirm that no labeled pod remains before callers
            # remove bind-mounted checkpoints.
            core_api = client.CoreV1Api()
            namespace = self.get_namespace()
            deadline = time.monotonic() + 30
            while True:
                pods = core_api.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=f"job-id={job_id}",
                ).items
                if not pods:
                    break
                if time.monotonic() >= deadline:
                    self.logger.error(
                        "Timed out waiting for pods of %s to terminate", job_id
                    )
                    return False
                time.sleep(0.5)
            self.logger.info(f"Deleted K8s microservice {job_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete K8s microservice: {e}")
            return False

    def _get_pod_image_pull_status(self, pod_name, namespace):
        """Get the image pull status from a pod's events and container statuses.

        Args:
            pod_name (str): Name of the pod
            namespace (str): Kubernetes namespace

        Returns:
            tuple: (status, image, error_message)
                status: "pulling", "extracting", "complete", "error", "waiting", "unknown"
                image: Docker image being pulled
                error_message: Error message if any
        """
        try:
            core_v1 = client.CoreV1Api()
            pod = core_v1.read_namespaced_pod(name=pod_name, namespace=namespace)

            # Get image from pod spec
            image = None
            if pod.spec.containers:
                image = pod.spec.containers[0].image

            # Check pod events first to determine if image was already pulled
            events = core_v1.list_namespaced_event(
                namespace=namespace,
                field_selector=f"involvedObject.name={pod_name}"
            )

            has_pulled_event = False
            has_pulling_event = False
            for event in events.items:
                reason = event.reason
                message = event.message or ""

                if reason == "Pulled":
                    has_pulled_event = True
                elif reason == "Pulling":
                    has_pulling_event = True
                elif reason == "Failed":
                    if "ImagePullBackOff" in message or "ErrImagePull" in message:
                        return ("error", image, message)
                    if "unauthorized" in message.lower():
                        return ("auth_error", image, message)

            # Check container statuses for image pull state
            if pod.status.container_statuses:
                for container_status in pod.status.container_statuses:
                    if container_status.state.waiting:
                        reason = container_status.state.waiting.reason
                        message = container_status.state.waiting.message or ""

                        if reason == "ContainerCreating":
                            # If image was already pulled, we're now extracting
                            if has_pulled_event:
                                return ("extracting", image, None)
                            return ("pulling", image, None)
                        if reason == "ImagePullBackOff":
                            return ("error", image, f"Image pull failed: {message}")
                        if reason == "ErrImagePull":
                            return ("error", image, f"Failed to pull image: {message}")
                        if reason == "ImageInspectError":
                            return ("error", image, f"Image inspection failed: {message}")
                        if reason == "InvalidImageName":
                            return ("not_exists_in_registry", image, f"Invalid image name: {message}")
                        if reason in ["ErrImageNeverPull", "RegistryUnavailable"]:
                            return ("error", image, f"{reason}: {message}")

                    elif container_status.state.running:
                        return ("complete", image, None)

            # Use events to determine status if container status wasn't conclusive
            if has_pulled_event:
                return ("complete", image, None)
            if has_pulling_event:
                return ("pulling", image, None)

            return ("waiting", image, None)

        except ApiException as e:
            if e.status == 404:
                return ("waiting", None, None)
            self.logger.warning(f"Error getting pod status: {e}")
            return ("unknown", None, str(e))
        except Exception as e:
            self.logger.warning(f"Error getting pod image pull status: {e}")
            return ("unknown", None, str(e))

    def _monitor_statefulset_image_pull(self, statefulset_name, namespace, job_id, timeout_seconds=600):
        """Monitor a StatefulSet's pods for image pull events and update job status.

        Args:
            statefulset_name (str): Name of the StatefulSet
            namespace (str): Kubernetes namespace
            job_id (str): Job ID for status updates
            timeout_seconds (int): Maximum time to wait for image pull

        Returns:
            bool: True if image pull completed successfully, False otherwise
        """
        core_v1 = client.CoreV1Api()
        start_time = time.time()
        last_status = None
        image = None

        self.logger.info(f"Monitoring image pull for StatefulSet {statefulset_name} (job_id={job_id})")

        while time.time() - start_time < timeout_seconds:
            try:
                # Find pods belonging to this StatefulSet
                pods = core_v1.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=f"job-id={job_id}"
                )

                if not pods.items:
                    # No pods yet, might still be creating
                    if last_status != "waiting":
                        last_status = "waiting"
                        self.update_image_pull_status(job_id, image or "unknown", "checking")
                    time.sleep(5)
                    continue

                # Check first pod (master node)
                pod = pods.items[0]
                pod_name = pod.metadata.name
                status, pod_image, error_msg = self._get_pod_image_pull_status(pod_name, namespace)

                if pod_image:
                    image = pod_image

                # Only update if status changed
                if status != last_status:
                    last_status = status

                    if status == "pulling":
                        self.update_image_pull_status(job_id, image, "pulling")
                    elif status == "extracting":
                        self.update_image_pull_status(job_id, image, "extracting")
                    elif status == "complete":
                        self.update_image_pull_status(job_id, image, "complete")
                        return True
                    elif status == "error":
                        self.update_image_pull_status(job_id, image, "error", error_message=error_msg)
                        return False
                    elif status == "auth_error":
                        self.update_image_pull_status(job_id, image, "auth_error", error_message=error_msg)
                        return False
                    elif status == "not_exists_in_registry":
                        self.update_image_pull_status(job_id, image, "not_exists_in_registry", error_message=error_msg)
                        return False

                # If complete or error, return
                if status in ("complete", "error", "auth_error", "not_exists_in_registry"):
                    return status == "complete"

                time.sleep(5)

            except Exception as e:
                self.logger.warning(f"Error monitoring image pull: {e}")
                time.sleep(5)

        # Timeout
        self.logger.warning(f"Timeout waiting for image pull for job {job_id}")
        self.update_image_pull_status(job_id, image, "error", error_message="Image pull timeout")
        return False

    def wait_for_statefulset_ready(self, statefulset_name, name_space, job_id=None, image=None):
        """Wait for the statefulset to be ready.

        Args:
            statefulset_name (str): Name of the StatefulSet
            name_space (str): Kubernetes namespace
            job_id (str, optional): Job ID for status updates
            image (str, optional): Docker image being used
        """
        api_instance = client.AppsV1Api()
        stateful_set_ready = False
        image_pull_monitored = False

        # Start monitoring image pull if job_id provided
        if job_id and image:
            self.update_image_pull_status(job_id, image, "checking")

        while not stateful_set_ready:
            statefulset_response = api_instance.read_namespaced_stateful_set(
                statefulset_name,
                name_space
            )
            if statefulset_response:
                desired_replicas = statefulset_response.spec.replicas
                ready_replicas = statefulset_response.status.ready_replicas or 0

                # If job_id provided and image not yet pulled, monitor image pull
                if job_id and not image_pull_monitored:
                    # Check pod status for image pulling
                    # Use job-id label which is set during StatefulSet creation
                    pods = client.CoreV1Api().list_namespaced_pod(
                        namespace=name_space,
                        label_selector=f"job-id={job_id}"
                    )
                    if pods.items:
                        pod = pods.items[0]
                        status, _, error_msg = self._get_pod_image_pull_status(
                            pod.metadata.name, name_space
                        )
                        if status == "pulling":
                            self.update_image_pull_status(job_id, image, "pulling")
                        elif status == "extracting":
                            self.update_image_pull_status(job_id, image, "extracting")
                        elif status == "complete":
                            self.update_image_pull_status(job_id, image, "complete")
                            image_pull_monitored = True
                        elif status in ("error", "auth_error", "not_exists_in_registry"):
                            self.update_image_pull_status(job_id, image, status, error_message=error_msg)

                if desired_replicas == ready_replicas:
                    self.logger.info(f"Statefulset {statefulset_name} is ready with {ready_replicas} replicas")
                    stateful_set_ready = True
                else:
                    # Detect CrashLoopBackOff / repeated container failures early
                    if job_id:
                        is_crashed, crash_reason = self._check_pod_crash_loop(job_id, name_space)
                        if is_crashed:
                            raise RuntimeError(
                                f"Pod for job {job_id} crashed during startup "
                                f"({crash_reason}). Check pod logs with: "
                                f"kubectl logs ims-{job_id}-0"
                            )
                    self.logger.info(
                        f"Statefulset {statefulset_name} pending with "
                        f"{ready_replicas}/{desired_replicas} ready"
                    )
                    time.sleep(10)
            else:
                self.logger.info(f"{statefulset_name} not found.")
                time.sleep(10)

    def create_statefulset(
            self, job_id, num_gpu_per_node, num_nodes, image, api_port=8000,
            master_port=29500, accelerator=None, statefulset_type="multinode",
            custom_command=None, custom_env_vars=None, custom_ports=None,
            org_name=None, experiment_id=None, is_long_lived=False):
        """Create statefulset with flexible configuration for different types"""
        try:
            # Use api_port for multinode services
            if statefulset_type == "inference_microservice":
                service_ports = custom_ports or [(8080, 8080), (8081, 8081)]
            else:
                service_ports = custom_ports or [(api_port, api_port)]

            self.create_statefulset_service(job_id, statefulset_type=statefulset_type, ports=service_ports)

            name_space = self.get_namespace()
            api_instance = client.AppsV1Api()

            # Set statefulset name and service name based on type
            if statefulset_type == "inference_microservice":
                statefulset_name = f"ims-{job_id}"
                service_name = f"ims-svc-{job_id}"
                app_label = "ims"
            else:
                statefulset_name = self.get_statefulset_name(job_id)
                service_name = self.get_statefulset_service_name(job_id)
                app_label = "multinode"

            # Configure labels based on type
            labels = {
                "app": app_label,
                "job-id": job_id
            }

            if statefulset_type == "inference_microservice":
                labels.update({
                    "service-type": "long-lived" if is_long_lived else "temporary",
                    "auto-cleanup": "false" if is_long_lived else "true",
                    "statefulset": statefulset_name,
                    "org": org_name or "default",
                    "experiment": experiment_id or "",
                    "job": job_id or ""
                })

            # Configure environment variables based on type
            env_vars = []
            if statefulset_type == "multinode":
                # Original multinode environment variables
                release_name_env_var = client.V1EnvVar(name="RELEASE_NAME", value=self.release_name)
                namespace_env_var = client.V1EnvVar(name="NAMESPACE", value=name_space)
                num_gpu_env_var = client.V1EnvVar(name="NUM_GPU_PER_NODE", value=str(num_gpu_per_node))
                world_size_env_var = client.V1EnvVar(name="WORLD_SIZE", value=str(num_nodes))
                node_rank_env_var = client.V1EnvVar(name="NODE_RANK", value_from=client.V1EnvVarSource(
                    field_ref=client.V1ObjectFieldSelector(
                        field_path="metadata.labels['apps.kubernetes.io/pod-index']"
                    )
                ))
                master_address_env_var = client.V1EnvVar(
                    name="MASTER_ADDR",
                    value=f"{statefulset_name}-0.{service_name}.{name_space}.svc.cluster.local"
                )
                master_port_env_var = client.V1EnvVar(name="MASTER_PORT", value=str(master_port))
                save_on_each_node_env_var = client.V1EnvVar(name="SAVE_ON_EACH_NODE", value="True")
                nccl_ib_disable_env_var = client.V1EnvVar(
                    name="NCCL_IB_DISABLE",
                    value=os.getenv("NCCL_IB_DISABLE", default="0")
                )
                nccl_ib_ext_disable_env_var = client.V1EnvVar(
                    name="NCCL_IBEXT_DISABLE",
                    value=os.getenv("NCCL_IBEXT_DISABLE", default="0")
                )
                service_prefix_env_var = client.V1EnvVar(name="JOB_SERVICE_PREFIX", value=statefulset_name)
                service_name_env_var = client.V1EnvVar(name="JOB_SERVICE_NAME", value=service_name)
                env_vars = [namespace_env_var, num_gpu_env_var, world_size_env_var, node_rank_env_var,
                            master_address_env_var, master_port_env_var, save_on_each_node_env_var,
                            nccl_ib_disable_env_var, nccl_ib_ext_disable_env_var, release_name_env_var,
                            service_prefix_env_var, service_name_env_var]
            elif statefulset_type == "inference_microservice":
                # Automatic environment variables for inference microservice
                env_vars = [client.V1EnvVar(name="JOB_ID", value=job_id or "")]

            # Add custom environment variables if provided
            if custom_env_vars and isinstance(custom_env_vars, dict):
                for key, value in custom_env_vars.items():
                    env_vars.append(client.V1EnvVar(name=key, value=str(value)))

            # Configure ports
            if statefulset_type == "inference_microservice":
                # Default inference microservice ports: HTTP API (8080) and health check (8081)
                if custom_ports:
                    container_ports = [
                        client.V1ContainerPort(
                            container_port=port[0],
                            name=port[2] if len(port) > 2 else f"port-{port[0]}"
                        )
                        for port in custom_ports
                    ]
                else:
                    container_ports = [
                        client.V1ContainerPort(container_port=8080, name="http-ims"),
                        client.V1ContainerPort(container_port=8081, name="health-ims")
                    ]
            elif custom_ports:
                container_ports = [
                    client.V1ContainerPort(
                        container_port=port[0],
                        name=port[2] if len(port) > 2 else f"port-{port[0]}"
                    )
                    for port in custom_ports
                ]
            else:
                container_ports = [
                    client.V1ContainerPort(container_port=api_port),
                    client.V1ContainerPort(container_port=8080)]

            # Configure command
            # custom_command may arrive as a list ["/bin/bash","-c","<shell>"]
            # (from Docker-compatible callers) or as a plain string.
            # Extract the shell string so it can be safely interpolated.
            if isinstance(custom_command, (list, tuple)):
                shell_cmd = custom_command[-1] if len(custom_command) > 1 else custom_command[0]
            else:
                shell_cmd = custom_command

            if statefulset_type == "inference_microservice" and custom_command:
                container_command = ["/bin/bash", "-c"]
                inference_microservice_command = f"""
umask 0 &&


echo "Starting Inference Microservice..." &&
{shell_cmd}
"""
                container_args = [inference_microservice_command]
            elif custom_command:
                container_command = ["/bin/bash", "-c"]
                container_args = [shell_cmd]
            else:
                container_command = ["/bin/bash", "-c"]
                container_args = ["flask run --host 0.0.0.0 --port 8000"]

            # Configure container name
            container_name = f"{app_label}-container"

            dshm_volume_mount = client.V1VolumeMount(name="dshm", mount_path="/dev/shm")
            dshm_volume = client.V1Volume(
                name="dshm",
                empty_dir=client.V1EmptyDirVolumeSource(medium="Memory")
            )
            capabilities = client.V1Capabilities(
                add=['SYS_PTRACE']
            )
            security_context = client.V1SecurityContext(
                capabilities=capabilities
            )
            image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')
            node_selector = None
            if accelerator:
                available_gpus = self.get_available_instances()
                gpu_to_be_run_on = None
                if available_gpus:
                    gpu_to_be_run_on = available_gpus.get(accelerator, {}).get("gpu_type")
                node_selector = {'accelerator': gpu_to_be_run_on}

            # Configure probes (only for multinode, not for inference microservice)
            probes = {}
            if statefulset_type == "multinode":
                probes = {
                    "readiness_probe": client.V1Probe(
                        http_get=client.V1HTTPGetAction(
                            path="/api/v1/health/readiness",
                            port=8000
                        ),
                        initial_delay_seconds=10,
                        period_seconds=10,
                        timeout_seconds=5,
                        failure_threshold=3
                    ),
                    "liveness_probe": client.V1Probe(
                        http_get=client.V1HTTPGetAction(
                            path="/api/v1/health/liveness",
                            port=8000
                        ),
                        initial_delay_seconds=10,
                        period_seconds=10,
                        timeout_seconds=5,
                        failure_threshold=3
                    )
                }

            # Create container
            container_spec = {
                "name": container_name,
                "image": image,
                "command": container_command,
                "args": container_args,
                "resources": client.V1ResourceRequirements(
                    limits={
                        "nvidia.com/gpu": (
                            num_gpu_per_node if statefulset_type == "multinode"
                            else (num_gpu_per_node if num_gpu_per_node > 0 else 1)
                        )
                    }
                ),
                "env": env_vars,
                "ports": container_ports,
                "volume_mounts": [dshm_volume_mount],
                "security_context": security_context
            }

            # Add probes if they exist
            if probes:
                container_spec.update(probes)

            container = client.V1Container(**container_spec)

            # Configure affinity (only for multinode)
            affinity = None
            if statefulset_type == "multinode":
                affinity = client.V1Affinity(
                    pod_anti_affinity=client.V1PodAntiAffinity(
                        preferred_during_scheduling_ignored_during_execution=[
                            client.V1WeightedPodAffinityTerm(
                                weight=100,
                                pod_affinity_term=client.V1PodAffinityTerm(
                                    label_selector=client.V1LabelSelector(
                                        match_expressions=[
                                            client.V1LabelSelectorRequirement(
                                                key="app",
                                                operator="In",
                                                values=["multinode"]
                                            )
                                        ]
                                    ),
                                    topology_key="kubernetes.io/hostname"
                                )
                            )
                        ]
                    )
                )

            # Create metadata with owner references (only for multinode)
            metadata_spec = {"name": statefulset_name}
            if statefulset_type == "multinode":
                metadata_spec["owner_references"] = [self.get_owner_reference()]

            # Add labels to metadata
            if statefulset_type == "inference_microservice":
                metadata_spec["labels"] = {
                    "app": app_label,
                    "service-type": "long-lived" if is_long_lived else "temporary",
                    "auto-cleanup": "false" if is_long_lived else "true",
                    "org": org_name or "default",
                    "experiment": experiment_id or "",
                    "job": job_id or ""
                }

            stateful_set = client.V1StatefulSet(
                api_version="apps/v1",
                kind="StatefulSet",
                metadata=client.V1ObjectMeta(**metadata_spec),
                spec=client.V1StatefulSetSpec(
                    replicas=num_nodes,
                    pod_management_policy="Parallel",
                    selector=client.V1LabelSelector(
                        match_labels=(
                            {"statefulset": statefulset_name}
                            if statefulset_type == "inference_microservice"
                            else labels
                        )
                    ),
                    service_name=service_name,
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels=labels
                        ),
                        spec=client.V1PodSpec(
                            image_pull_secrets=[client.V1LocalObjectReference(name=image_pull_secret)],
                            containers=[container],
                            volumes=[dshm_volume],
                            node_selector=node_selector,
                            restart_policy="Always",
                            affinity=affinity
                        )
                    )
                )
            )
            api_instance.create_namespaced_stateful_set(
                namespace=name_space,
                body=stateful_set
            )
            # Ensure the statefulset is ready, monitoring image pull status
            self.wait_for_statefulset_ready(statefulset_name, name_space, job_id=job_id, image=image)
            return True
        except RuntimeError:
            raise
        except Exception as e:
            self.logger.error(f"Exception thrown in create_statefulset is {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def delete_statefulset(self, job_name, use_ngc=True, resource_type="multinode", workspace_metadata={}):
        """Deletes a Job or StatefulSet"""
        # Enhanced logging with call stack to understand WHY deletion was triggered
        caller_frame = inspect.currentframe().f_back
        caller_info = inspect.getframeinfo(caller_frame) if caller_frame else None
        caller_location = f"{caller_info.filename}:{caller_info.lineno}" if caller_info else "unknown"

        self.logger.debug(
            f"{'-' * 80}\n"
            f"DELETE_STATEFULSET CALLED\n"
            f"Job Name: {job_name}\n"
            f"Backend: {BACKEND}\n"
            f"Use NGC: {use_ngc}\n"
            f"Resource Type: {resource_type}\n"
            f"Called From: {caller_location}\n"
            f"Call Stack (top 5):\n"
        )

        # Log call stack to understand the termination trigger
        stack_lines = traceback.format_stack(limit=6)
        for line in stack_lines[-5:]:  # Last 5 frames
            self.logger.warning(f"  {line.strip()}")

        self.logger.debug(f"{'-' * 80}")

        name_space = self.get_namespace()
        if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
            config.load_kube_config()
        else:
            config.load_incluster_config()

        api_instance = client.AppsV1Api()
        try:
            # Configure naming and service type based on resource type
            if resource_type == "inference_microservice":
                stateful_set_name = f"ims-{job_name}"
                service_type = "inference_microservice"
            else:
                stateful_set_name = self.get_statefulset_name(job_name)
                service_type = "statefulset"

            # Delete service first, then statefulset
            self.delete_service(job_id=job_name, service_type=service_type)
            stateful_set = api_instance.read_namespaced_stateful_set(
                name=stateful_set_name,
                namespace=name_space
            )
            if not stateful_set:
                self.logger.info(f"Statefulset {stateful_set_name} not found in namespace {name_space}")
                return True  # Deletion goal achieved - resource doesn't exist
            api_response = api_instance.delete_namespaced_stateful_set(
                name=stateful_set_name,
                namespace=name_space,
                body=client.V1DeleteOptions(
                    propagation_policy='Foreground',
                    grace_period_seconds=5
                )
            )
            self.logger.info(f"Statefulset deleted. status='{str(api_response.status)}'")
            return True
        except ApiException as e:
            if e.status == 404:
                # Resource not found - deletion goal already achieved (likely already deleted by another process)
                self.logger.info(
                    f"Statefulset {stateful_set_name} not found (404) - already deleted. Deletion successful."
                )
                return True
            self.logger.error(f"ApiException caught in delete_statefulset {str(e)}")
            self.logger.error("Statefulset failed to delete.")
            return False
        except Exception as e:
            self.logger.error(f"Exception caught in delete_statefulset {str(e)}")
            self.logger.error("Statefulset failed to delete.")
            return False

    def get_statefulset_status(self, statefulset_name, replicas=1, resource_type="StatefulSet"):
        """General function to get status of any StatefulSet"""
        name_space = self.get_namespace()
        api_instance = client.AppsV1Api()
        try:
            api_response = api_instance.read_namespaced_stateful_set_status(
                name=statefulset_name,
                namespace=name_space)
            ready_replicas = api_response.status.ready_replicas or 0
            if ready_replicas < replicas:
                return {"status": "ReplicaNotReady", "replicas": {"ready": ready_replicas, "desired": replicas}}
            return {"status": "Running", "replicas": {"ready": ready_replicas, "desired": replicas}}
        except ApiException as e:
            if e.status == 404:
                self.logger.info(f"{resource_type} StatefulSet not found.")
                return {"status": "NotFound"}
            self.logger.error(f"Got other ApiException error: {e}")
            return {"status": "Error"}
        except Exception as e:
            self.logger.error(f"Got {type(e)} error: {e}")
            return {"status": "Error"}

    def create_service_unified(self, service_name, selector, ports, service_type="ClusterIP", labels=None,
                               return_info=False, add_owner_reference=True):
        """Unified function to create Kubernetes services with flexible configuration"""
        try:
            name_space = self.get_namespace()
            api_instance = client.CoreV1Api()

            # Set default labels
            if labels is None:
                labels = {}

            # Handle different port formats
            if isinstance(ports, int):
                # Single port number
                service_ports = [client.V1ServicePort(port=ports, target_port=ports)]
            elif isinstance(ports, tuple) and len(ports) == 2:
                # Single (port, target_port) tuple
                service_ports = [client.V1ServicePort(port=ports[0], target_port=ports[1])]
            elif isinstance(ports, list):
                # List of port configurations
                service_ports = []
                for port_config in ports:
                    if isinstance(port_config, tuple):
                        if len(port_config) == 2:
                            # (port, target_port)
                            service_ports.append(client.V1ServicePort(
                                port=port_config[0],
                                target_port=port_config[1],
                                name=f"port-{port_config[0]}"
                            ))
                        elif len(port_config) == 3:
                            # (port, target_port, name)
                            service_ports.append(client.V1ServicePort(
                                port=port_config[0],
                                target_port=port_config[1],
                                name=port_config[2],
                                protocol="TCP"
                            ))
                    else:
                        # Just port number
                        service_ports.append(client.V1ServicePort(
                            port=port_config,
                            target_port=port_config,
                            name=f"port-{port_config}"
                        ))
            else:
                raise ValueError(f"Unsupported ports format: {ports}")

            # Configure service spec
            cluster_ip = "None" if service_type == "Headless" else None
            spec = client.V1ServiceSpec(
                type=service_type if service_type != "Headless" else "ClusterIP",
                ports=service_ports,
                selector=selector,
                cluster_ip=cluster_ip
            )

            # Configure metadata
            metadata_kwargs = {"name": service_name, "labels": labels}
            if add_owner_reference:
                metadata_kwargs["owner_references"] = [self.get_owner_reference()]

            # Create service object
            service = client.V1Service(
                api_version="v1",
                kind="Service",
                metadata=client.V1ObjectMeta(**metadata_kwargs),
                spec=spec
            )

            # Create the service
            api_response = api_instance.create_namespaced_service(
                namespace=name_space,
                body=service
            )
            self.logger.info(f"Service created: {service_name}")

            if return_info:
                return {
                    "service_name": service_name,
                    "cluster_ip": api_response.spec.cluster_ip,
                    "ports": ports
                }
            return None

        except Exception as e:
            self.logger.error(f"Failed to create service {service_name}: {e}")
            if return_info:
                raise
            # For backward compatibility with create_service, don't raise for non-return cases
            self.logger.error(traceback.format_exc())
            return None

    def create_service(self, service_name, selector, service_port, target_port, labels=None):
        """Legacy function - delegates to unified service creation"""
        if labels is None:
            labels = {}
        return self.create_service_unified(
            service_name=service_name,
            selector=selector,
            ports=(service_port, target_port),
            service_type="Headless",
            labels=labels,
            return_info=False,
            add_owner_reference=True
        )

    def create_flask_service(self, job_id):
        """Create a service for a microservice pod"""
        service_name = f"flask-service-{job_id}"
        selector = {
            "app": "flask",
            "job-id": job_id
        }
        return self.create_service_unified(
            service_name=service_name,
            selector=selector,
            ports=(8000, 8000),
            service_type="Headless",
            labels={},
            return_info=False,
            add_owner_reference=True
        )

    def create_statefulset_service(self, job_id, statefulset_type="multinode", ports=None, service_type="ClusterIP"):
        """Create a service for a statefulset with flexible configuration"""
        if statefulset_type == "inference_microservice":
            # Inference microservice service configuration
            service_name = f"ims-svc-{job_id}"
            statefulset_name = f"ims-{job_id}"
            selector = {"statefulset": statefulset_name}
            labels = {"app": "ims"}

            # Handle ports format for inference microservices
            if isinstance(ports, tuple) and len(ports) == 2:
                port_list = [
                    (ports[0], ports[0], "http-port"),
                    (ports[1], ports[1], "health-port")
                ]
            else:
                port_list = ports or [(8080, 8080, "http-port"), (8081, 8081, "health-port")]

            return self.create_service_unified(
                service_name=service_name,
                selector=selector,
                ports=port_list,
                service_type=service_type,
                labels=labels,
                return_info=True,
                add_owner_reference=False
            )

        # Multinode service configuration
        service_name = self.get_statefulset_service_name(job_id)
        selector = {
            "app": "multinode",
            "job-id": job_id
        }
        labels = {
            "app": "multinode",
            "job-id": job_id
        }
        # Default to single port if not specified
        port_info = ports[0] if ports else (8000, 8000)

        return self.create_service_unified(
            service_name=service_name,
            selector=selector,
            ports=port_info,
            service_type="Headless",
            labels=labels,
            return_info=False,
            add_owner_reference=True
        )

    def delete_service(self, job_id=None, service_name=None, service_type="default"):
        """Delete a microservice pod's service with flexible service name handling"""
        try:
            # Handle different service naming patterns
            if service_name is None:
                if job_id is None:
                    raise ValueError("Either job_id or service_name must be provided")

                if service_type == "flask":
                    service_name = f"flask-service-{job_id}"
                elif service_type == "statefulset":
                    service_name = self.get_statefulset_service_name(job_id)
                else:
                    # Default case
                    service_name = job_id
            elif service_type == "inference_microservice" and job_id is None:
                # Extract job_id from inference microservice service name pattern
                if service_name.startswith("ims-svc-"):
                    job_id = service_name.replace("ims-svc-", "")
                else:
                    job_id = service_name  # fallback

            name_space = self.get_namespace()
            core_v1 = client.CoreV1Api()
            service = core_v1.read_namespaced_service(name=service_name, namespace=name_space)
            if not service:
                self.logger.info(f"Service {service_name} not found in namespace {name_space}")
                return True  # Return True since the goal (service not existing) is achieved

            core_v1.delete_namespaced_service(name=service_name, namespace=name_space)
            self.logger.info(f"Successfully deleted service: {service_name}")
            return True
        except ApiException as e:
            if e.status == 404:
                # Service not found - deletion goal already achieved (likely already deleted by another process)
                self.logger.info(f"Service {service_name} not found (404) - already deleted. Deletion successful.")
                return True
            self.logger.error(f"ApiException thrown in delete_service is {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
        except Exception as e:
            self.logger.error(f"Exception thrown in delete_service is {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def _check_pod_crash_loop(self, job_id, namespace):
        """Check if the IMS pod is in CrashLoopBackOff or has repeatedly failed.

        Uses read_namespaced_pod with the deterministic StatefulSet pod name
        (ims-{job_id}-0) instead of list_namespaced_pod, because the default
        service account typically lacks the pods list permission.
        """
        terminal_waiting_reasons = (
            "CrashLoopBackOff", "ErrImagePull", "ImagePullBackOff",
            "CreateContainerError", "InvalidImageName", "CreateContainerConfigError",
        )
        restart_threshold = 2
        pod_name = f"ims-{job_id}-0"
        try:
            pod = client.CoreV1Api().read_namespaced_pod(
                name=pod_name, namespace=namespace
            )
            if not pod.status or not pod.status.container_statuses:
                return False, None
            for cs in pod.status.container_statuses:
                if cs.state and cs.state.waiting:
                    reason = cs.state.waiting.reason
                    if reason in terminal_waiting_reasons:
                        return True, reason
                if cs.restart_count is not None and cs.restart_count >= restart_threshold:
                    last_exit = ""
                    if cs.last_state and cs.last_state.terminated:
                        last_exit = f", last exit_code={cs.last_state.terminated.exit_code}"
                    return True, f"restart_count={cs.restart_count}{last_exit}"
            return False, None
        except ApiException as e:
            if e.status == 404:
                return False, None
            self.logger.warning(f"Could not check pod crash status for {pod_name}: {e.reason}")
            return False, None
        except Exception as e:
            self.logger.warning(f"Could not check pod crash status for {pod_name}: {e}")
            return False, None

    def wait_for_service(self, job_id, service_name=None):
        """Wait until the specified service is ready or timeout is reached."""
        service_name = service_name or self.get_statefulset_service_name(job_id)
        namespace = self.get_namespace()
        start_time = time.time()
        while time.time() - start_time < 300:
            # Check job metadata status - if job is already terminated, exit early
            job_metadata = get_handler_job_metadata(job_id)
            if job_metadata:
                metadata_status = job_metadata.get("status")
                # Check if job has been terminated (by timeout, cancellation, or completion)
                if metadata_status in ("Canceled", "Canceling", "Paused", "Pausing", "Error", "Done"):
                    self.logger.info(
                        f"Job {job_id} has status '{metadata_status}'. "
                        f"Exiting wait_for_service early (no need to wait for service that won't come up)."
                    )
                    return metadata_status

            # Also check DNN status to catch timeout terminations early
            try:
                dnn_status = get_dnn_status(job_id, automl=False)
                if dnn_status:
                    # Check the most recent status entry
                    latest_status = dnn_status[-1] if isinstance(dnn_status, list) and len(dnn_status) > 0 else {}
                    if isinstance(latest_status, dict):
                        status_msg = latest_status.get('status', '')
                        if isinstance(status_msg, str):
                            import json
                            try:
                                status_data = json.loads(status_msg)
                                job_status = status_data.get('status', '')
                                # If job has FAILURE status, it's been terminated
                                if job_status == 'FAILURE':
                                    self.logger.info(
                                        f"Job {job_id} has DNN status 'FAILURE'. "
                                        f"Exiting wait_for_service early (job terminated)."
                                    )
                                    return "Error"
                            except (json.JSONDecodeError, AttributeError):
                                pass
            except Exception as e:
                # Don't fail the wait if we can't check DNN status
                self.logger.debug(f"Could not check DNN status for {job_id}: {e}")

            # Detect CrashLoopBackOff / repeated container failures early
            is_crashed, crash_reason = self._check_pod_crash_loop(job_id, namespace)
            if is_crashed:
                self.logger.error(
                    f"Pod for job {job_id} is in a crash loop ({crash_reason}). "
                    f"Exiting wait_for_service early."
                )
                return "Error"

            # Check if service is ready
            if (self.check_service_ready(service_name, namespace) and
                    self.check_endpoints_ready(service_name, namespace)):
                self.logger.info(f"Service '{service_name}' is ready.")
                return "Running"
            self.logger.info(f"Waiting for service '{service_name}' to be ready...")
            time.sleep(10)
        self.logger.error(f"Timed out waiting for service '{service_name}' to be ready.")
        return "Error"

    def get_job_status(self, job_id, **kwargs):
        """Get job status - unified interface for ExecutionHandler

        For K8s jobs, checks service status and sends a request to get job status.

        Args:
            job_name: Job identifier
            **kwargs: Additional parameters

        Returns:
            str: Job status (Pending, Running, Done, Error, Canceled, Paused, etc.)
        """
        network = kwargs.get("network")
        action = kwargs.get("action")
        specs = kwargs.get("specs")
        if not network or not action or not specs:
            self.logger.error(
                f"Missing required parameters for K8s job status check: "
                f"network={network}, action={action}, specs={specs}"
            )
            return "Error"
        from nvidia_tao_core.microservices.utils.handler_utils import send_statefulset_request

        service_status = self.wait_for_service(job_id)
        if service_status == "Running":
            response = send_statefulset_request(
                api_endpoint="get_job_status",
                network=network,
                action=action,
                job_id=job_id,
                specs=specs,
            )
            if response and response.ok:
                job_status = response.json()
                status = job_status.get("status")
                return status
        elif service_status in ("Canceled", "Canceling", "Paused", "Pausing"):
            return service_status
        return "Error"

    def get_statefulset_name(self, job_id):
        """Get the statefulset name for the given job id"""
        if os.getenv('STATEFULSET_NAME'):
            return os.getenv('STATEFULSET_NAME')
        release_name = os.getenv('RELEASE_NAME', default='tao-api')
        return f"{release_name}-sts-{job_id}"

    def get_statefulset_service_name(self, job_id):
        """Get the statefulset service name for the given job id"""
        if os.getenv('STATEFULSET_SERVICE_NAME'):
            return os.getenv('STATEFULSET_SERVICE_NAME')
        release_name = os.getenv('RELEASE_NAME', default='tao-api')
        return f"{release_name}-sts-svc-{job_id}"

    def check_service_ready(self, service_name, namespace):
        """Check if the specified service is ready."""
        try:
            _ = client.CoreV1Api().read_namespaced_service(name=service_name, namespace=namespace)
            return True
        except ApiException as e:
            if e.status == 404:
                return False
            raise e

    def check_endpoints_ready(self, service_name, namespace):
        """Check if the specified service has ready endpoints."""
        try:
            endpoints = client.CoreV1Api().read_namespaced_endpoints(name=service_name, namespace=namespace)
            if not endpoints.subsets:
                return False
            for subset in endpoints.subsets:
                if subset.addresses:
                    return True
            return False
        except ApiException as e:
            if e.status == 404:
                return False
            raise e

    def get_job_logs(self, job_id: str, tail_lines=None, namespace=None):
        """Get logs directly from Kubernetes pod using Kubernetes Python client.

        For multi-node jobs (StatefulSet), this automatically identifies and retrieves
        logs from the master node (identified by -0 suffix).

        Args:
            job_id (str): The job ID (used as pod/job name)
            tail_lines (int, optional): Number of lines to tail. If None, gets all logs

        Returns:
            str: The log content, or None if logs cannot be retrieved
        """
        self.logger.debug(
            f"[K8S_LOGS] Starting get_k8s_pod_logs for job_id={job_id}, "
            f"namespace={namespace}, tail_lines={tail_lines}"
        )
        try:
            # Load Kubernetes configuration
            self.logger.debug(f"[K8S_LOGS] Loading Kubernetes configuration for job_id={job_id}")
            try:
                config.load_incluster_config()
                self.logger.debug(f"[K8S_LOGS] Successfully loaded in-cluster config for job_id={job_id}")
            except Exception as e1:
                self.logger.debug(f"[K8S_LOGS] In-cluster config failed: {e1}, trying kube config")
                try:
                    config.load_kube_config()
                    self.logger.debug(f"[K8S_LOGS] Successfully loaded kube config for job_id={job_id}")
                except Exception as e2:
                    self.logger.error(
                        f"[K8S_LOGS] Failed to load any Kubernetes config for job_id={job_id}: "
                        f"in-cluster={e1}, kube={e2}"
                    )
                    return None

            # Get namespace
            if namespace is None:
                try:
                    namespace_file = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
                    self.logger.debug(f"[K8S_LOGS] Reading namespace from {namespace_file}")
                    with open(namespace_file, 'r', encoding='utf-8') as f:
                        namespace = f.read().strip()
                    self.logger.debug(f"[K8S_LOGS] Got namespace from service account: {namespace}")
                except Exception as e:
                    namespace = os.getenv("NAMESPACE", "default")
                    self.logger.debug(f"[K8S_LOGS] Using fallback namespace: {namespace} (error: {e})")
            else:
                self.logger.debug(f"[K8S_LOGS] Using provided namespace: {namespace}")

            v1 = client.CoreV1Api()
            self.logger.debug(f"[K8S_LOGS] Created CoreV1Api client for job_id={job_id}, namespace={namespace}")

            # Try to find the pod - could be a Job pod or StatefulSet pod
            pod_name = None
            self.logger.debug(f"[K8S_LOGS] Searching for pod for job_id={job_id} in namespace={namespace}")
            try:
                # First, try to list pods with the job_id as a label selector
                label_selector = f"job-name={job_id}"
                self.logger.debug(f"[K8S_LOGS] Trying label selector: {label_selector}")
                pods = v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)

                if pods.items:
                    # Found pods with job-name label (regular Kubernetes Job)
                    pod_name = pods.items[0].metadata.name
                    pod_status = pods.items[0].status.phase
                    self.logger.info(
                        f"[K8S_LOGS] Found Job pod: {pod_name} (status={pod_status}) "
                        f"for job_id: {job_id}"
                    )
                    self.logger.debug(
                        f"[K8S_LOGS] Pod details: namespace={namespace}, "
                        f"labels={pods.items[0].metadata.labels}"
                    )
                else:
                    self.logger.debug(
                        f"[K8S_LOGS] No pods found with label selector {label_selector}, "
                        f"trying StatefulSet naming"
                    )
                    # Try StatefulSet naming convention
                    # For StatefulSet, pods are named as <statefulset-name>-<ordinal>
                    # For multi-node, master is -0, workers are -1, -2, etc.
                    master_pod_name = f"{job_id}-0"
                    self.logger.debug(f"[K8S_LOGS] Trying StatefulSet master pod: {master_pod_name}")
                    try:
                        pod = v1.read_namespaced_pod(name=master_pod_name, namespace=namespace)
                        pod_name = master_pod_name
                        pod_status = pod.status.phase
                        self.logger.info(
                            f"[K8S_LOGS] Found StatefulSet master pod: {pod_name} "
                            f"(status={pod_status}) for job_id: {job_id}"
                        )
                        self.logger.debug(
                            f"[K8S_LOGS] Pod details: namespace={namespace}, "
                            f"labels={pod.metadata.labels}"
                        )
                    except ApiException as e1:
                        self.logger.debug(
                            f"[K8S_LOGS] Master pod {master_pod_name} not found "
                            f"(status={e1.status}), trying single pod name"
                        )
                        # Try single pod name (might be a single-node job)
                        try:
                            pod = v1.read_namespaced_pod(name=job_id, namespace=namespace)
                            pod_name = job_id
                            pod_status = pod.status.phase
                            self.logger.info(
                                f"[K8S_LOGS] Found single pod: {pod_name} (status={pod_status}) "
                                f"for job_id: {job_id}"
                            )
                            self.logger.debug(
                                f"[K8S_LOGS] Pod details: namespace={namespace}, "
                                f"labels={pod.metadata.labels}"
                            )
                        except ApiException as e2:
                            self.logger.debug("[K8S_LOGS] Direct pod names failed, trying partial name match")
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
                                    self.logger.info(
                                        f"[K8S_LOGS] Found pod by partial name match: {pod_name} "
                                        f"(status={pod_status}) for job_id: {job_id}"
                                    )
                                    self.logger.debug(
                                        f"[K8S_LOGS] Pod details: namespace={namespace}, "
                                        f"labels={selected_pod.metadata.labels}"
                                    )
                                else:
                                    self.logger.error(
                                        f"[K8S_LOGS] Could not find pod for job_id={job_id}: "
                                        f"master_pod={e1.status}, single_pod={e2.status}, no partial matches"
                                    )
                                    self.logger.debug(
                                        f"[K8S_LOGS] Tried names: {master_pod_name}, "
                                        f"{job_id}, and partial match"
                                    )
                                    return None
                            except Exception as e3:
                                self.logger.error(
                                    f"[K8S_LOGS] Error during partial name match for job_id={job_id}: "
                                    f"{type(e3).__name__}: {e3}"
                                )
                                self.logger.debug(f"[K8S_LOGS] Tried names: {master_pod_name}, {job_id}")
                                return None

            except Exception as e:
                self.logger.error(f"[K8S_LOGS] Exception finding pod for job_id={job_id}: {type(e).__name__}: {e}")
                self.logger.debug("[K8S_LOGS] Full traceback:", exc_info=True)
                return None

            if not pod_name:
                self.logger.error(f"[K8S_LOGS] No pod found for job_id={job_id} after all attempts")
                return None

            # Get logs from the pod
            self.logger.debug(
                f"[K8S_LOGS] Retrieving logs from "
                f"pod={pod_name}, namespace={namespace}, tail_lines={tail_lines}"
            )
            try:
                log_kwargs = {
                    'name': pod_name,
                    'namespace': namespace,
                    'timestamps': False,
                }

                if tail_lines is not None:
                    log_kwargs['tail_lines'] = tail_lines
                    self.logger.debug(f"[K8S_LOGS] Using tail_lines={tail_lines}")

                self.logger.debug(f"[K8S_LOGS] Calling read_namespaced_pod_log with kwargs: {log_kwargs}")
                logs = v1.read_namespaced_pod_log(**log_kwargs)
                log_line_count = len(logs.splitlines())
                log_size_kb = len(logs) / 1024
                self.logger.info(
                    f"[K8S_LOGS] Successfully retrieved {log_line_count} lines "
                    f"({log_size_kb:.1f} KB) from pod {pod_name}"
                )
                return logs

            except ApiException as e:
                if e.status == 400:
                    # Pod might not have started yet or container not ready
                    self.logger.warning(
                        f"[K8S_LOGS] Pod {pod_name} logs not available yet: "
                        f"status={e.status}, reason={e.reason}"
                    )
                    self.logger.debug(f"[K8S_LOGS] ApiException details: {e.body}")
                    return None
                self.logger.error(
                    f"[K8S_LOGS] Error retrieving logs from pod {pod_name}: "
                    f"status={e.status}, reason={e.reason}"
                )
                self.logger.debug(f"[K8S_LOGS] ApiException details: {e.body}")
                return None

        except ImportError as e:
            self.logger.error(
                f"[K8S_LOGS] Kubernetes Python client not installed: {e}. "
                f"Install with: pip install kubernetes"
            )
            return None
        except Exception as e:
            self.logger.error(
                f"[K8S_LOGS] Unexpected error getting Kubernetes pod logs for "
                f"job_id={job_id}: {type(e).__name__}: {e}"
            )
            self.logger.debug("[K8S_LOGS] Full traceback:", exc_info=True)
            return None
