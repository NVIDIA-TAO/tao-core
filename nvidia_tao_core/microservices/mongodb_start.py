# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MongoDB init script"""
import time
from kubernetes import client, config
import os
from time import sleep
from pymongo import MongoClient
from urllib import parse
import logging

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)

mongodb_crd_group = 'mongodbcommunity.mongodb.com'
mongodb_crd_version = 'v1'
mongodb_crd_plural = 'mongodbcommunity'
mongodb_crd_name = 'mongodb'
mongodb_desired_replica_count = int(os.getenv('MONGODESIREDREPLICAS', '3'))
mongodb_storage_class = os.getenv('MONGOSTORAGECLASS', 'nfs-client')
mongodb_storage_size = os.getenv('MONGOSTORAGESIZE', '100Gi')
mongodb_storage_access_mode = os.getenv('MONGOSTORAGEACCESSMODE', 'ReadWriteOnce')
mongod_memory_request = os.getenv('MONGODMEMORYREQUEST', '5000M')
mongod_cpu_request = os.getenv('MONGODCPUREQUEST', '1000m')
mongod_cpu_limit = os.getenv('MONGODCPULIMIT', '4000m')
mongod_memory_limit = os.getenv('MONGODMEMORYLIMIT', '8000M')
mongo_agent_memory_request = os.getenv('MONGOAGENTMEMORYREQUEST', '400M')
mongo_agent_cpu_request = os.getenv('MONGOAGENTCPUREQUEST', '500m')
mongo_agent_cpu_limit = os.getenv('MONGOAGENTCPULIMIT', '1000m')
mongo_agent_memory_limit = os.getenv('MONGOAGENTMEMORYLIMIT', '500M')
mongodb_namespace = os.getenv("NAMESPACE", "default")
mongo_operator_enabled = os.getenv('MONGO_OPERATOR_ENABLED', 'true') == 'true'
mongo_secret = os.getenv("MONGOSECRET")
encoded_secret = parse.quote(mongo_secret, safe='')


def create_mongodb_replicaset():
    """Creates a MongoDB replicaset"""
    config.load_incluster_config()

    # Initialize the custom objects API
    api_instance = client.CustomObjectsApi()

    mongodb_body = {
        "apiVersion": "mongodbcommunity.mongodb.com/v1",
        "kind": "MongoDBCommunity",
        "metadata": {
            "name": mongodb_crd_name,
        },
        "spec": {
            "members": mongodb_desired_replica_count,
            "type": "ReplicaSet",
            "version": "6.0.5",
            "security": {
                "authentication": {
                    "modes": ["SCRAM"]
                }
            },
            "users": [
                {
                    "name": "default-user",
                    "db": "admin",
                    "passwordSecretRef": {
                        "name": "mongodb-keyfile-secret",
                        "key": "mongodb-keyfile"
                    },
                    "roles": [
                        {
                            "name": "root",
                            "db": "admin"
                        }
                    ],
                    "scramCredentialsSecretName": "scram-secret",
                }
            ],
            "additionalMongodConfig": {
                "storage.wiredTiger.engineConfig.journalCompressor": "zlib"
            },
            "statefulSet": {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "mongodb-agent",
                                    "resources": {
                                        "requests": {
                                            "cpu": mongo_agent_cpu_request,
                                            "memory": mongo_agent_memory_request
                                        },
                                        "limits": {
                                            "cpu": mongo_agent_cpu_limit,
                                            "memory": mongo_agent_memory_limit
                                        }
                                    }
                                },
                                {
                                    "name": "mongod",
                                    "resources": {
                                        "requests": {
                                            "cpu": mongod_cpu_request,
                                            "memory": mongod_memory_request
                                        },
                                        "limits": {
                                            "cpu": mongod_cpu_limit,
                                            "memory": mongod_memory_limit
                                        }
                                    }
                                },
                            ]
                        }
                    },
                    "volumeClaimTemplates": [
                        {
                            "metadata": {
                                "name": "data-volume"
                            },
                            "spec": {
                                "storageClassName": mongodb_storage_class,
                                "accessModes": [mongodb_storage_access_mode],
                                "resources": {
                                    "requests": {
                                        "storage": mongodb_storage_size
                                    }
                                }
                            }
                        },
                        {
                            "metadata": {
                                "name": "logs-volume"
                            },
                            "spec": {
                                "storageClassName": mongodb_storage_class,
                                "accessModes": [mongodb_storage_access_mode],
                                "resources": {
                                    "requests": {
                                        "storage": mongodb_storage_size
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        },
    }

    # Create the custom resource in the specified namespace
    try:
        api_instance.create_namespaced_custom_object(
            mongodb_crd_group,
            mongodb_crd_version,
            mongodb_namespace,
            mongodb_crd_plural,
            mongodb_body
        )
        logger.info("MongoDB replicaset created successfully.")
    except Exception as e:
        logger.error("Failed to create MongoDB replicaset: %s", e)
        raise e


def patch_mongodb_replicaset():
    """Patch MongoDB Replicaset"""
    config.load_incluster_config()

    # Initialize the custom objects API
    api_instance = client.CustomObjectsApi()

    updated_spec = {
        "spec": {
            "members": mongodb_desired_replica_count
        }
    }

    try:
        api_instance.patch_namespaced_custom_object(
            mongodb_crd_group,
            mongodb_crd_version,
            mongodb_namespace,
            mongodb_crd_plural,
            name=mongodb_crd_name,
            body=updated_spec
        )
        logger.info("MongoDB replicaset patched successfully.")
    except Exception as e:
        logger.error("Failed to patch MongoDB replicaset: %s", e)
        raise e


if __name__ == "__main__":
    logger.info("Mongo Operator Enabled: %s", mongo_operator_enabled)
    if not mongo_operator_enabled:
        try:
            from .utils.mongo_utils import MongoHandler
            # try to make a test connection to DB replicaset
            mongo_experiments = MongoHandler(
                "tao",
                "experiments"
            )
        except Exception as e:  # if error, initialize replicaset
            logger.error("Exception caught in mongodb start with message %s", str(e))
            retry = 0
            while retry <= 10:
                try:
                    logger.info("Retrying attempt: %s", retry)
                    init_time = 20 * mongodb_desired_replica_count
                    time.sleep(init_time)
                    connection_string = (
                        f'mongodb://default-user:{encoded_secret}@'
                        f'mongodb-0.mongodb-svc.{mongodb_namespace}.svc.cluster.local'
                        '?authSource=admin'
                    )
                    c = MongoClient(connection_string, directConnection=True)
                    rs_members = []
                    for rs_id in range(mongodb_desired_replica_count):
                        rs_members.append({
                            '_id': rs_id,
                            'host': f'mongodb-{rs_id}.mongodb-svc.{mongodb_namespace}.svc.cluster.local:27017'
                        })
                    mongo_config = {
                        '_id': 'mongodb',
                        'members': rs_members
                    }
                    logger.info("Going to run replSetInitiate command")
                    out = c.admin.command("replSetInitiate", mongo_config)
                    logger.info("replSetInitiate output: %s", out)
                    logger.debug("Breaking while")
                    logger.debug("Output: %s", out)
                    break
                except Exception as e:
                    if 'already initialized' in str(e):
                        logger.info("Replicaset already initialized")
                        break
                    logger.error("Error initializing replicaset! %s", e)
                    retry += 1
                    logger.info("Retrying %s", retry)
                    if retry > 10:
                        import traceback
                        logger.error("Traceback: %s", traceback.format_exc())
                        logger.error("Raising exception")
                        raise e
    else:
        config.load_incluster_config()

        api_instance = client.CustomObjectsApi()

        try:
            api_response = api_instance.list_namespaced_custom_object(
                group=mongodb_crd_group,
                version=mongodb_crd_version,
                namespace=mongodb_namespace,
                plural=mongodb_crd_plural
            )
            logger.info("MongoDB response: %s", api_response)
            items = api_response['items']
            if len(items) == 0:
                create_mongodb_replicaset()
            else:
                status = items[0].get('status', {})
                phase = status.get('phase', '')
                if phase == 'Running' and mongodb_desired_replica_count != status.get('currentStatefulSetReplicas', 0):
                    patch_mongodb_replicaset()
                elif phase != 'Running':
                    logger.warning("Unknown MongoDB Replicaset status: %s", status)
                    raise Exception("Unknown MongoDB Replicaset status")

            # Wait for all replicas to be ready
            replica_count = 0
            phase = 'Pending'
            while replica_count != mongodb_desired_replica_count and phase != 'Running':
                api_response = api_instance.list_namespaced_custom_object(
                    group=mongodb_crd_group,
                    version=mongodb_crd_version,
                    namespace=mongodb_namespace,
                    plural=mongodb_crd_plural
                )
                logger.info("MongoDB response: %s", api_response)
                items = api_response.get('items', [])
                if items:
                    status = items[0].get('status', {})
                    phase = status.get('phase', 'Pending')
                    replica_count = status.get('currentStatefulSetReplicas', 0)
                    logger.info(
                        f"Current Replica Count: {replica_count}, waiting for "
                        f"{mongodb_desired_replica_count - replica_count} more replicas to be ready")
                    logger.info(f"Current ReplicaSet Phase {phase}")
                sleep(20)

        except Exception as e:
            logger.error("Exception when calling CustomObjectsApi: %s", str(e))
            raise e
