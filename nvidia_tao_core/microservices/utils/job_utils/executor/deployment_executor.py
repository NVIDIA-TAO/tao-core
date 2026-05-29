# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment executor for Tensorboard deployments"""
import os
from kubernetes import client
from kubernetes.client.rest import ApiException

from nvidia_tao_core.microservices.handlers.execution_handlers.kubernetes_handler import KubernetesHandler
from nvidia_tao_core.microservices.utils.stateless_handler_utils import BACKEND

if os.getenv("BACKEND"):  # To see if the container is going to be used for Service pods or network jobs
    from nvidia_tao_core.microservices.utils import mongo_utils


class DeploymentExecutor(KubernetesHandler):
    """Handles Kubernetes Deployment operations for Tensorboard"""

    def __init__(self):
        """Initialize the deployment executor"""
        super().__init__()
        self.backend = BACKEND

    def create_tensorboard_deployment(self, deployment_name, image, command, logs_image, logs_command, replicas):
        """Creates Tensorboard Deployment"""
        name_space = self.get_namespace()
        api_instance = client.AppsV1Api()
        logs_volume_mount = client.V1VolumeMount(
            name="tb-data",
            mount_path="/tfevents")
        capabilities = client.V1Capabilities(
            add=['SYS_PTRACE']
        )
        security_context = client.V1SecurityContext(
            capabilities=capabilities
        )

        tb_port = [
            client.V1ContainerPort(container_port=6006)
        ]
        resources = client.V1ResourceRequirements(
            limits={
                'memory': "600Mi",
                'cpu': "10m",
            },
            requests={
                'memory': '300Mi',
                'cpu': "5m"
            }
        )
        no_gpu = client.V1EnvVar(
            name="NVIDIA_VISIBLE_DEVICES",
            value="none")
        mongo_secret_env = client.V1EnvVar(
            name="MONGOSECRET",
            value=mongo_utils.mongo_secret  # pylint: disable=E0606
        )
        mongo_operator_enabled_env = client.V1EnvVar(
            name="MONGO_OPERATOR_ENABLED",
            value=str(mongo_utils.mongo_operator_enabled).lower()
        )
        mongo_namespace_env = client.V1EnvVar(
            name="NAMESPACE",
            value=mongo_utils.mongo_namespace
        )
        backend_env = client.V1EnvVar(
            name="BACKEND",
            value=self.backend.value,
        )
        image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')
        tb_container = client.V1Container(
            name="tb-container",
            image=image,
            env=[no_gpu],
            command=["/bin/sh", "-c"],
            args=[command],
            resources=resources,
            volume_mounts=[logs_volume_mount],
            ports=tb_port,
            security_context=security_context)

        tb_logs_container = client.V1Container(
            name="tb-logs-container",
            image=logs_image,
            env=[no_gpu,
                 mongo_secret_env,
                 mongo_operator_enabled_env,
                 mongo_namespace_env,
                 backend_env],
            command=["/bin/sh", "-c"],
            resources=resources,
            args=[logs_command],
            volume_mounts=[logs_volume_mount],
            security_context=security_context,
        )

        logs_volume = client.V1Volume(
            name="tb-data",
            empty_dir=client.V1EmptyDirVolumeSource())

        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={
                    "purpose": "tao-toolkit-tensorboard-job",
                    "resource-type": "tensorboard",
                    "app": deployment_name,  # use deployment_name as the selector name
                }
            ),
            spec=client.V1PodSpec(
                containers=[tb_container, tb_logs_container],
                volumes=[logs_volume],
                image_pull_secrets=[client.V1LocalObjectReference(name=image_pull_secret)]
            ))

        spec = client.V1DeploymentSpec(
            replicas=replicas,
            template=template,
            selector={"matchLabels": {"app": deployment_name}})

        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(name=deployment_name, labels={
                "resource-type": "tensorboard",
            }, owner_references=[self.get_owner_reference()]),
            spec=spec)

        self.logger.info("Prepared deployment configs")
        try:
            api_instance.create_namespaced_deployment(
                body=deployment,
                namespace=name_space)
            self.logger.info("Start create deployment")
            return
        except Exception as e:
            self.logger.error(f"Create deployment got error: {e}")
            return

    def create_tensorboard_service(self, tb_service_name, deploy_label):
        """Creates Tensorboard Service"""
        name_space = self.get_namespace()
        tb_port = [
            client.V1ServicePort(name='tb-default-port', port=6006, target_port=6006, protocol="TCP")
        ]
        spec = client.V1ServiceSpec(ports=tb_port, selector={"app": deploy_label})
        # add annotation, it will only works in Azure, but will not affect other cloud
        annotation = {
            "service.beta.kubernetes.io/azure-load-balancer-internal": "true",
        }
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=tb_service_name,
                labels={
                    "app": tb_service_name,
                    "resource-type": "tensorboard",
                },
                owner_references=[self.get_owner_reference()],
                annotations=annotation),
            spec=spec,
        )
        api_instance = client.CoreV1Api()

        self.logger.info("Prepared Tensorboard Service configs")
        try:
            api_instance.create_namespaced_service(
                body=service,
                namespace=name_space)
            self.logger.info("Start create Tensorboard Service")
            return
        except Exception as e:
            self.logger.error(f"Create Tensorboard Service got error: {e}")
            return

    def create_tensorboard_ingress(self, tb_service_name, tb_ingress_name, tb_ingress_path):
        """Creates Tensorboard Ingress"""
        name_space = self.get_namespace()
        networking_v1_api = client.NetworkingV1Api()
        ingress = client.V1Ingress(
            api_version="networking.k8s.io/v1",
            kind="Ingress",
            metadata=client.V1ObjectMeta(
                name=tb_ingress_name,
                namespace=name_space,
                labels={
                    "resource-type": "tensorboard",
                },
                annotations={
                    "kubernetes.io/ingress.class": "nginx",
                    "nginx.ingress.kubernetes.io/client-max-body-size": "0m",
                    "nginx.ingress.kubernetes.io/proxy-body-size": "0m",
                    "nginx.ingress.kubernetes.io/body-size": "0m",
                    "nginx.ingress.kubernetes.io/client-body-buffer-size": "50m",
                    "nginx.ingress.kubernetes.io/proxy-buffer-size": "128k",
                    "nginx.ingress.kubernetes.io/proxy-buffers-number": "4",
                    "nginx.ingress.kubernetes.io/proxy-connect-timeout": "3600",
                    "nginx.ingress.kubernetes.io/proxy-read-timeout": "3600",
                    "nginx.ingress.kubernetes.io/proxy-send-timeout": "3600",
                    "meta.helm.sh/release-name": self.release_name,
                    "meta.helm.sh/release-namespace": name_space
                },
                owner_references=[self.get_owner_reference()]),
            spec=client.V1IngressSpec(
                rules=[client.V1IngressRule(
                    http=client.V1HTTPIngressRuleValue(
                        paths=[client.V1HTTPIngressPath(
                            path=tb_ingress_path,
                            path_type="Prefix",
                            backend=client.V1IngressBackend(
                                service=client.V1IngressServiceBackend(
                                    port=client.V1ServiceBackendPort(
                                        name='tb-default-port'
                                    ),
                                    name=tb_service_name
                                )
                            )
                        )]
                    )
                )]
            )
        )

        try:
            networking_v1_api.create_namespaced_ingress(
                body=ingress,
                namespace=name_space
            )
            self.logger.info("Created Tensorboard Ingress")
            return
        except Exception as e:
            self.logger.error(f"Create Tensorboard Ingress got error: {e}")
            return

    def get_tensorboard_deployment_status(self, deployment_name, replicas=1):
        """Returns status of Tensorboard deployment"""
        name_space = self.get_namespace()
        api_instance = client.AppsV1Api()
        try:
            api_response = api_instance.read_namespaced_deployment_status(
                name=deployment_name,
                namespace=name_space)
            available_replicas = api_response.status.available_replicas
            if not isinstance(available_replicas, int) or available_replicas < replicas:
                return {"status": "ReplicaNotReady"}
            return {"status": "Running"}
        except ApiException as e:
            if e.status == 404:
                self.logger.info("Tensorboard Deployment not found.")
                return {"status": "NotFound"}
            self.logger.error(f"Got other ApiException error: {e}")
            return {"status": "Error"}
        except Exception as e:
            self.logger.error(f"Got {type(e)} error: {e}")
            return {"status": "Error"}

    def get_tensorboard_service_status(self, tb_service_name, port=6006):
        """Returns status of TB Service"""
        name_space = self.get_namespace()
        api_instance = client.CoreV1Api()

        try:
            api_response = api_instance.read_namespaced_service(
                name=tb_service_name,
                namespace=name_space,
            )
            self.logger.info(f'TB Service API Response: {api_response}')
            tb_service_ip = api_response.spec.cluster_ip
            return {"status": "Running", "tb_service_ip": tb_service_ip}
        except ApiException as e:
            if e.status == 404:
                self.logger.info("Tensorboard Service not found.")
                return {"status": "NotFound"}
            self.logger.error(f"Got other ApiException error: {e}")
            return {"status": "Error"}
        except Exception as e:
            self.logger.error(f"Got {type(e)} error: {e}")
            return {"status": "Error"}

    def delete_tensorboard_deployment(self, deployment_name):
        """Deletes Tensorboard Deployment"""
        name_space = self.get_namespace()
        api_instance = client.AppsV1Api()
        try:
            api_response = api_instance.delete_namespaced_deployment(
                name=deployment_name,
                namespace=name_space,
                body=client.V1DeleteOptions(
                    propagation_policy='Foreground',
                    grace_period_seconds=5))
            self.logger.info(f"Tensorboard Deployment deleted. status='{str(api_response.status)}'")
            return
        except Exception as e:
            self.logger.error(f"Tensorboard Deployment failed to delete, got error: {e}")
            return

    def delete_tensorboard_service(self, tb_service_name):
        """Deletes Tensorboard service"""
        name_space = self.get_namespace()
        api_instance = client.CoreV1Api()
        try:
            api_response = api_instance.delete_namespaced_service(
                name=tb_service_name,
                namespace=name_space,
            )
            self.logger.info(f"Tensorboard Service deleted. status='{str(api_response.status)}'")
            return
        except Exception as e:
            self.logger.error(f"Tensorboard Service failed to delete, got error: {e}")
            return

    def delete_tensorboard_ingress(self, tb_ingress_name):
        """Delete Tensorboard Ingress"""
        name_space = self.get_namespace()
        networking_v1_api = client.NetworkingV1Api()
        try:
            api_response = networking_v1_api.delete_namespaced_ingress(
                name=tb_ingress_name,
                namespace=name_space
            )
            self.logger.info(f"Tensorboard Ingress deleted. status='{str(api_response.status)}'")
            return
        except Exception as e:
            self.logger.error(f"Tensorboard Ingress failed to delete, got error: {e}")
            return
