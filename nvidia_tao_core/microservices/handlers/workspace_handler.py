# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workspace handler module for managing workspace operations"""
import copy
import os
import uuid
import logging
import traceback
from datetime import datetime, timezone

from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    get_user,
    resolve_metadata,
    check_read_access,
    check_write_access,
    get_handler_metadata,
    printc,
    write_handler_metadata
)
from nvidia_tao_core.microservices.utils.cloud_utils import create_cs_instance
from nvidia_tao_core.microservices.utils.dataset_utils import validate_dataset
from nvidia_tao_core.microservices.utils.encrypt_utils import NVVaultEncryption
from nvidia_tao_core.microservices.utils.handler_utils import Code
from nvidia_tao_core.microservices.utils.deduplication_utils import find_duplicate_workspace

if os.getenv("BACKEND"):
    from .mongo_handler import MongoHandler
else:
    MongoHandler = None  # type: ignore

from ..utils.basic_utils import (
    get_org_workspaces,
    get_user_workspaces,
    get_user_experiments,
    get_experiment,
    get_dataset
)

# Configure logging
logger = logging.getLogger(__name__)


class WorkspaceHandler:
    """Handles workspace creation, updating, deletion and retrieval."""

    @staticmethod
    def list_workspaces(user_id, org_name):
        """Retrieve a list of workspaces accessible by the given user.

        Args:
            user_id (str): UUID representing the user.
            org_name (str): Name of the organization.

        Returns:
            list[dict]: A list of workspace metadata dictionaries.
        """
        # Collect all metadatas
        metadatas = []
        for workspace_id in list(set(get_org_workspaces(org_name))):
            handler_metadata = get_handler_metadata(workspace_id, 'workspaces')
            shared_workspace = handler_metadata.get("shared", False)
            if handler_metadata:
                if shared_workspace or handler_metadata.get("user_id") == user_id:
                    metadatas.append(handler_metadata)
            else:
                # Something is wrong. The user metadata has a workspace that doesn't exist in the system.
                contexts = {"user_id": user_id, "org_name": org_name, "handler_id": workspace_id}
                printc("Workspace not found. Skipping.", contexts)
        return metadatas

    @staticmethod
    def retrieve_workspace(user_id, org_name, workspace_id):
        """Retrieve metadata for a specific workspace if the user has access.

        Args:
            user_id (str): UUID representing the user.
            org_name (str): Name of the organization.
            workspace_id (str): UUID of the workspace.

        Returns:
            Code:
                - 200 with metadata if successful.
                - 404 if the workspace is not found or access is denied.
        """
        handler_metadata = resolve_metadata("workspace", workspace_id)
        if not handler_metadata:
            return Code(404, {}, "Workspace not found")

        if not check_read_access(user_id, org_name, workspace_id, kind="workspaces"):
            return Code(404, {}, "Workspace not found")

        return Code(200, handler_metadata, "Workspace retrieved")

    @staticmethod
    def retrieve_cloud_datasets(user_id, org_name, workspace_id, dataset_type, dataset_format, dataset_intention):
        """Retrieve paths of cloud datasets accessible within a workspace.

        Args:
            user_id (str): UUID representing the user.
            org_name (str): Name of the organization.
            workspace_id (str): UUID of the workspace.
            dataset_type (str): Type of the dataset.
            dataset_format (str): Format of the dataset.
            dataset_intention (str): Purpose of dataset usage.

        Returns:
            Code:
                - 200 with dataset paths if successful.
                - 404 if the workspace is not found or access is denied.
        """
        handler_metadata = resolve_metadata("workspace", workspace_id)
        if not handler_metadata:
            return Code(404, {}, "Workspace not found")

        if not check_read_access(user_id, org_name, workspace_id, kind="workspaces"):
            return Code(404, {}, "Workspace not found")

        cloud_instance, _ = create_cs_instance(handler_metadata)
        cloud_files, _ = cloud_instance.list_files_in_folder("data")
        suggestions = set([])
        for cloud_file_path in cloud_files:
            cloud_folder = os.path.dirname(cloud_file_path)
            if dataset_type in ("segmentation", "pose_classification"):
                cloud_folder = os.path.dirname(cloud_folder)
            if dataset_type == "ml_recog":
                index_of_folder = cloud_folder.find("metric_learning_recognition")
                if index_of_folder != "-1":
                    cloud_folder = cloud_folder[0:index_of_folder]
            dataset_handler_metadata = {
                "type": dataset_type,
                "format": dataset_format,
                "use_for": dataset_intention
            }
            is_cloud_dataset_present, _ = validate_dataset(
                org_name,
                dataset_handler_metadata,
                temp_dir=f"/{cloud_folder}",
                workspace_metadata=handler_metadata
            )
            if is_cloud_dataset_present:
                suggestions.add(f"/{cloud_folder}")
        suggestions = list(suggestions)
        return_response_data = {"dataset_paths": suggestions}
        if suggestions:
            return Code(200, return_response_data, "Dataset folder path suggestions retrieved")
        return Code(200, return_response_data, "Dataset folder path suggestion couldn't be retrieved")

    @staticmethod
    def create_workspace(user_id, org_name, request_dict):
        """Create a new workspace with specified parameters.

        Args:
            user_id (str): UUID representing the user.
            org_name (str): Name of the organization.
            request_dict (dict): Workspace request following WorkspaceRspSchema.
                - "type" (required)
                - "format" (required)
                - "force_create" (optional): If False, return existing workspace with same params

        Returns:
            Code:
                - 200 with metadata if creation is successful or duplicate found.
                - 400 if there are errors in the request.
        """
        # Check for duplicate workspace unless force_create is True
        force_create = request_dict.get("force_create", False)
        cloud_type = request_dict.get("cloud_type", "")
        cloud_specific_details = request_dict.get("cloud_specific_details", {})

        if not force_create and cloud_type and cloud_specific_details:
            duplicate_id = find_duplicate_workspace(user_id, org_name, cloud_type, cloud_specific_details)
            if duplicate_id:
                # Return existing workspace
                handler_metadata = resolve_metadata("workspace", duplicate_id)
                if handler_metadata:
                    logger.info(
                        "Returning existing workspace %s for user %s with matching cloud config",
                        duplicate_id, user_id
                    )
                    return Code(200, handler_metadata, "Workspace with same parameters already exists")

        workspace_id = str(uuid.uuid4())
        # Create metadata dict and create some initial folders
        metadata = {"id": workspace_id,
                    "user_id": user_id,
                    "org_name": org_name,
                    "created_on": datetime.now(tz=timezone.utc),
                    "last_modified": datetime.now(tz=timezone.utc),
                    "name": request_dict.get("name", "My Workspace"),
                    "shared": request_dict.get("shared", False),
                    "version": request_dict.get("version", "1.0.0"),
                    "cloud_type": request_dict.get("cloud_type", ""),
                    "cloud_specific_details": request_dict.get("cloud_specific_details", {}),
                    }

        encrypted_metadata = copy.deepcopy(metadata)
        cloud_specific_details = encrypted_metadata.get("cloud_specific_details", {})
        lepton_workspace_id = cloud_specific_details.get('lepton_workspace_id')
        lepton_auth_token = cloud_specific_details.get('lepton_auth_token')
        if lepton_workspace_id and lepton_auth_token:
            try:
                from nvidia_tao_core.microservices.handlers.execution_handlers.lepton_handler import LeptonHandler
                lepton_handler = LeptonHandler(lepton_workspace_id, lepton_auth_token)
                lepton_handler.api_client.info()
            except Exception as e:
                logger.error(f"Exception instantiating Lepton Handler: {e}")
                return Code(400, {}, "Provided Lepton cloud credentials are invalid")
        # Encrypt Cloud details
        config_path = os.getenv("VAULT_SECRET_PATH", None)
        if encrypted_metadata["cloud_specific_details"]:
            cloud_specific_details = encrypted_metadata["cloud_specific_details"]
            if config_path and cloud_specific_details:
                encryption = NVVaultEncryption(config_path)
                for key, value in cloud_specific_details.items():
                    if encryption.check_config()[0]:
                        encrypted_metadata["cloud_specific_details"][key] = encryption.encrypt(value)
                    elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                        return Code(400, {}, "Vault service does not work, can't save cloud workspace")

        try:
            if encrypted_metadata["cloud_type"] in ("aws", "azure", "lepton"):
                create_cs_instance(encrypted_metadata)
        except Exception as e:
            logger.error("Exception thrown in create workspace is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(400, {}, "Provided cloud credentials are invalid")

        write_handler_metadata(workspace_id, encrypted_metadata, "workspace")
        mongo_users = MongoHandler("tao", "users")
        workspaces = get_user_workspaces(user_id, mongo_users)
        workspaces.append(workspace_id)
        mongo_users.upsert({'id': user_id}, {'id': user_id, 'workspaces': workspaces})

        ret_Code = Code(200, metadata, "Workspace created")
        return ret_Code

    @staticmethod
    def update_workspace(user_id, org_name, workspace_id, request_dict):
        """Update an existing workspace with new metadata.

        Args:
            user_id (str): UUID representing the user.
            org_name (str): Name of the organization.
            workspace_id (str): UUID of the workspace.
            request_dict (dict): Update request following WorkspaceRspSchema.
                - "type" (required)
                - "format" (required)

        Returns:
            Code:
                - 200 with updated metadata if successful.
                - 404 if the workspace is not found or access is denied.
                - 400 if the update request is invalid.
        """
        metadata = resolve_metadata("workspace", workspace_id)
        if not metadata:
            return Code(404, {}, "Workspace not found")

        if not check_write_access(user_id, org_name, workspace_id, kind="workspaces"):
            return Code(404, {}, "Workspace not available")

        update_keys = request_dict.keys()
        for key in ["name", "version", "cloud_type", "shared"]:
            if key in update_keys:
                requested_value = request_dict[key]
                if requested_value is not None:
                    metadata[key] = requested_value
                    metadata["last_modified"] = datetime.now(tz=timezone.utc)

        encrypted_metadata = copy.deepcopy(metadata)
        if "cloud_specific_details" in request_dict.keys():
            # Encrypt Cloud details
            for key, value in request_dict["cloud_specific_details"].items():
                if key == "cloud_type":
                    encrypted_metadata["cloud_type"] = value
                if key == "cloud_specific_details":
                    config_path = os.getenv("VAULT_SECRET_PATH", None)
                    if config_path:
                        if not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                            return Code(400, {}, "Vault service does not work, can't save cloud workspace")
                        encryption = NVVaultEncryption(config_path)
                        for cloud_key, cloud_value in request_dict["cloud_specific_details"].items():
                            encrypted_metadata["cloud_specific_details"][cloud_key] = cloud_value
                            if encryption.check_config()[0]:
                                encrypted_metadata["cloud_specific_details"][cloud_key] = (
                                    encryption.encrypt(cloud_value)
                                )

        if encrypted_metadata["cloud_type"] in ("aws", "azure", "lepton"):
            try:
                if "cloud_type" in request_dict.keys() or "cloud_specific_details" in request_dict.keys():
                    create_cs_instance(encrypted_metadata)
            except Exception as e:
                logger.error("Exception thrown in update_workspace is %s", str(e))
                logger.error(traceback.format_exc())
                return Code(400, {}, "Provided cloud credentials are invalid")

        write_handler_metadata(workspace_id, encrypted_metadata, "workspace")
        ret_Code = Code(200, metadata, "Workspace updated")
        return ret_Code

    @staticmethod
    def delete_workspace(org_name, workspace_id):
        """Delete a workspace if it belongs to the user.

        Args:
            org_name (str): Name of the organization.
            workspace_id (str): UUID of the workspace.

        Returns:
            Code:
                - 200 if deletion is successful.
                - 404 if the workspace cannot be accessed or deleted.
        """
        handler_metadata = resolve_metadata("workspace", workspace_id)
        if not handler_metadata:
            return Code(200, {}, "Workspace deleted")
        user_id = handler_metadata.get("user_id")

        if workspace_id not in get_user_workspaces(user_id):
            return Code(404, {}, "Workspace cannot be deleted as it's doesn't belong to user")

        # If workspace is being used by user's experiments or datasets.
        experiments = get_user_experiments(user_id)

        for experiment_id in experiments:
            experiment_metadata = get_experiment(experiment_id)
            experiment_workspace = experiment_metadata.get("workspace", "")
            if experiment_workspace and workspace_id in experiment_workspace:
                return Code(
                    400,
                    {},
                    f"Experiment {experiment_metadata['id']} "
                    f"({experiment_metadata['id']}) in use; Delete experiment first"
                )

            train_datasets = experiment_metadata.get("train_datasets", [])
            if not isinstance(train_datasets, list):
                train_datasets = [train_datasets]
            for dataset_id in train_datasets:
                dataset_metadata = get_dataset(dataset_id)
                dataset_workspace = dataset_metadata.get("workspace", "")
                if workspace_id == dataset_workspace:
                    return Code(
                        400,
                        {},
                        f"Dataset {dataset_metadata['id']} "
                        f"({dataset_metadata['id']}) in use; Delete dataset first"
                    )

            for key in ["eval_dataset", "inference_dataset", "calibration_dataset"]:
                additional_dataset_id = experiment_metadata.get(key)
                if additional_dataset_id:
                    dataset_metadata = get_dataset(additional_dataset_id)
                    dataset_workspace = dataset_metadata.get("workspace", "")
                    if workspace_id == dataset_workspace:
                        return Code(
                            400,
                            {},
                            f"Dataset {dataset_metadata['id']} "
                            f"({dataset_metadata['id']}) in use; Delete dataset first"
                        )

        mongo_users = MongoHandler("tao", "users")
        user = get_user(user_id, mongo_users)
        workspaces = user.get("workspaces", [])
        if workspace_id in workspaces:
            workspaces.remove(workspace_id)
            mongo_users.upsert({'id': user_id}, {'id': user_id, 'workspaces': workspaces})
        mongo_workspaces = MongoHandler("tao", "workspaces")
        mongo_workspaces.delete_one({'id': workspace_id})
        return Code(200, {"message": "Workspace deleted"}, "")
