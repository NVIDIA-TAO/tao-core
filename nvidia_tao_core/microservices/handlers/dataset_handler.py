# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset handler module for managing dataset operations"""
import copy
import os
import shutil
import threading
import traceback
import uuid
import logging
from datetime import datetime, timezone

from nvidia_tao_core.microservices.enum_constants import DatasetType
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    resolve_metadata,
    check_read_access,
    check_write_access,
    get_handler_metadata,
    get_public_datasets,
    printc,
    sanitize_handler_metadata,
    write_handler_metadata,
    add_public_dataset,
    remove_public_dataset
)
from nvidia_tao_core.microservices.utils.encrypt_utils import NVVaultEncryption
from nvidia_tao_core.microservices.utils.dataset_utils import validate_dataset
from nvidia_tao_core.microservices.utils.handler_utils import (
    Code,
    download_dataset
)
from nvidia_tao_core.microservices.utils.core_utils import (
    read_network_config,
)
from nvidia_tao_core.microservices.utils.deduplication_utils import find_duplicate_dataset

if os.getenv("BACKEND"):
    from .mongo_handler import MongoHandler
else:
    MongoHandler = None  # type: ignore

from ..utils.basic_utils import (
    get_org_datasets,
    get_user_datasets,
    get_dataset_actions,
)

# Configure logging
logger = logging.getLogger(__name__)


class DatasetHandler:
    """Handles dataset creation, updating, deletion and retrieval."""

    @staticmethod
    def list_datasets(user_id, org_name):
        """Retrieve a list of datasets accessible by the given user.

        Args:
            user_id (str): UUID representing the user.
            org_name (str): Name of the organization.

        Returns:
            list[dict]: A list of dataset metadata dictionaries.
        """
        # Collect all metadatas
        metadatas = []
        for dataset_id in list(set(get_org_datasets(org_name) + get_public_datasets())):
            handler_metadata = get_handler_metadata(dataset_id, 'datasets')
            shared_dataset = handler_metadata.get("shared", False)
            if handler_metadata:
                if shared_dataset or handler_metadata.get("user_id") == user_id:
                    handler_metadata = sanitize_handler_metadata(handler_metadata)
                    metadatas.append(handler_metadata)
            else:
                # Something is wrong. The user metadata has a dataset that doesn't exist in the system.
                contexts = {"user_id": user_id, "org_name": org_name, "handler_id": dataset_id}
                printc("Dataset not found. Skipping.", contexts)
        return metadatas

    @staticmethod
    def get_dataset_formats(dataset_type):
        """Retrieve available dataset formats for a given dataset type.

        Args:
            dataset_type (str): The type of dataset.

        Returns:
            list[str]: A list of supported dataset formats.
        """
        try:
            dataset_formats = []
            accepted_dataset_intents = []
            api_params = read_network_config(dataset_type).get("api_params", {})
            if api_params:
                if api_params.get("formats", []):
                    dataset_formats += api_params.get("formats", [])
                if api_params.get("accepted_ds_intents", []):
                    accepted_dataset_intents += api_params.get("accepted_ds_intents", [])
            return Code(
                200,
                {
                    "dataset_formats": dataset_formats,
                    "accepted_dataset_intents": accepted_dataset_intents
                },
                ""
            )
        except Exception:
            logger.error("Exception caught during getting dataset formats: %s", traceback.format_exc())
            return Code(404, [], "Exception caught during getting dataset formats")

    @staticmethod
    def create_dataset(user_id, org_name, request_dict, dataset_id=None, from_ui=False):
        """Creates a new dataset with the given parameters.

        Args:
            user_id (str): The unique identifier of the user.
            org_name (str): The name of the organization.
            request_dict (dict): Dictionary containing dataset creation parameters.
                - "type" (str): Required dataset type.
                - "format" (str): Required dataset format.
                - "force_create" (optional): If False, return existing dataset with same params
            dataset_id (str, optional): A predefined dataset ID. Defaults to None.
            from_ui (bool, optional): Flag indicating if the request is from UI. Defaults to False.

        Returns:
            Code: Response object containing status and metadata of the created dataset.
        """
        # Check for duplicate dataset unless force_create is True
        force_create = request_dict.get("force_create", False)
        if not force_create:
            dataset_params = {
                "type": request_dict.get("type"),
                "format": request_dict.get("format"),
                "workspace": request_dict.get("workspace"),
                "cloud_file_path": request_dict.get("cloud_file_path"),
                "url": request_dict.get("url"),
                "use_for": request_dict.get("use_for")
            }
            duplicate_id = find_duplicate_dataset(user_id, org_name, dataset_params)
            if duplicate_id:
                # Return existing dataset
                handler_metadata = get_handler_metadata(duplicate_id, 'datasets')
                if handler_metadata:
                    handler_metadata = sanitize_handler_metadata(handler_metadata)
                    logger.info(
                        "Returning existing dataset %s for user %s with matching params",
                        duplicate_id, user_id
                    )
                    return Code(200, handler_metadata, "Dataset with same parameters already exists")

        workspace_id = request_dict.get("workspace", None)
        if workspace_id and not check_read_access(user_id, org_name, workspace_id, kind="workspaces"):
            return Code(404, None, f"Workspace {workspace_id} not found")

        # Gather type,format fields from request
        ds_type = request_dict.get("type", None)
        if ds_type == "ocrnet":
            intention = request_dict.get("use_for", [])
            if not (intention in (["training"], ["evaluation"])):
                return Code(
                    400,
                    {},
                    "Use_for in dataset metadata is not set ['training'] or ['evaluation']. "
                    "Please set use_for appropriately"
                )

        ds_format = request_dict.get("format", None)
        # Perform basic checks - valid type and format?
        if ds_type not in DatasetType.__members__.values():
            msg = "Invalid dataset type"
            return Code(400, {}, msg)

        if ds_format not in read_network_config(ds_type)["api_params"]["formats"]:
            msg = "Incompatible dataset format and type"
            return Code(400, {}, msg)

        intention = request_dict.get("use_for", [])
        if ds_format in ("raw", "coco_raw") and intention:
            if intention != ["testing"]:
                msg = "raw or coco_raw's format should be associated with ['testing'] intent"
                return Code(400, {}, msg)

        # Create a dataset ID and its root
        pull = False
        if not dataset_id:
            pull = True
            dataset_id = str(uuid.uuid4())

        if request_dict.get("public", False):
            add_public_dataset(dataset_id)

        dataset_actions = get_dataset_actions(ds_type, ds_format)

        # Create metadata dict and create some initial folders
        metadata = {"id": dataset_id,
                    "user_id": user_id,
                    "org_name": org_name,
                    "authorized_party_nca_id": request_dict.get("authorized_party_nca_id", ""),
                    "created_on": datetime.now(tz=timezone.utc),
                    "last_modified": datetime.now(tz=timezone.utc),
                    "name": request_dict.get("name", "My Dataset"),
                    "shared": request_dict.get("shared", False),
                    "description": request_dict.get("description", "My TAO Dataset"),
                    "version": request_dict.get("version", "1.0.0"),
                    "docker_env_vars": request_dict.get("docker_env_vars", {}),
                    "logo": request_dict.get("logo", "https://www.nvidia.com"),
                    "type": ds_type,
                    "format": ds_format,
                    "actions": dataset_actions,
                    "client_url": request_dict.get("client_url", None),
                    "client_id": request_dict.get("client_id", None),
                    "client_secret": request_dict.get("client_secret", None),  # TODO:: Store Secrets in Vault
                    "filters": request_dict.get("filters", None),
                    "cloud_file_path": request_dict.get("cloud_file_path"),
                    "url": request_dict.get("url"),
                    "workspace": request_dict.get("workspace"),
                    "use_for": intention,
                    "base_experiment_ids": request_dict.get("base_experiment_ids", []),
                    }

        # Set status based on skip_validation flag
        skip_validation = request_dict.get("skip_validation", False)
        if skip_validation:
            metadata["status"] = "pull_complete"
        else:
            metadata["status"] = request_dict.get("status", "starting")

        if metadata.get("url", ""):
            if not metadata.get("url").startswith("https"):
                return Code(400, {}, "Invalid pull URL passed")

        # Encrypt the MLOPs keys
        config_path = os.getenv("VAULT_SECRET_PATH", None)
        if config_path and metadata["docker_env_vars"]:
            encryption = NVVaultEncryption(config_path)
            for key, value in metadata["docker_env_vars"].items():
                if encryption.check_config()[0]:
                    metadata["docker_env_vars"][key] = encryption.encrypt(value)
                elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                    return Code(400, {}, "Vault service does not work, can't enable MLOPs services")

        write_handler_metadata(dataset_id, metadata, "dataset")
        mongo_users = MongoHandler("tao", "users")
        user_query = {'id': user_id}
        datasets = get_user_datasets(user_id, mongo_users)
        datasets.append(dataset_id)
        mongo_users.upsert(user_query, {'id': user_id, 'datasets': datasets})

        # Pull dataset in background if known URL and not skipping validation
        if pull and not skip_validation:
            job_run_thread = threading.Thread(target=DatasetHandler.pull_dataset, args=(user_id, org_name, dataset_id,))
            job_run_thread.start()

        # Read this metadata from saved file...
        return_metadata = sanitize_handler_metadata(metadata)
        ret_Code = Code(200, return_metadata, "Dataset created")
        return ret_Code

    @staticmethod
    def create_dataset_dict_from_experiment_metadata(dataset_id, action, handler_metadata):
        """Generates a dataset request dictionary from existing experiment metadata.

        Args:
            dataset_id (str): The unique identifier of the new dataset.
            action (str): The action performed that triggered dataset creation.
            handler_metadata (dict): Metadata from the source dataset or experiment.

        Returns:
            dict: A request dictionary containing dataset creation parameters.
        """
        infer_ds = handler_metadata.get("inference_dataset", None)
        if infer_ds:
            dataset_metadata = get_handler_metadata(infer_ds, "datasets")
        else:
            dataset_metadata = copy.deepcopy(handler_metadata)
        request_dict = {}
        output_dataset_type = dataset_metadata.get("type")
        output_dataset_format = dataset_metadata.get("format")
        use_for = dataset_metadata.get("use_for")
        request_dict["type"] = output_dataset_type
        request_dict["status"] = dataset_metadata.get("status", "pull_complete")
        request_dict["format"] = output_dataset_format
        request_dict["use_for"] = use_for
        request_dict["workspace"] = dataset_metadata.get("workspace")
        request_dict["cloud_file_path"] = os.path.join("/results/", dataset_id)
        request_dict["name"] = f"{dataset_metadata.get('name')} (created from Data services {action} action)"
        request_dict["shared"] = dataset_metadata.get("shared", False)
        request_dict["use_for"] = dataset_metadata.get("use_for", [])
        request_dict["docker_env_vars"] = dataset_metadata.get("docker_env_vars", {})
        return request_dict

    @staticmethod
    def update_dataset(org_name, dataset_id, request_dict):
        """Updates an existing dataset with new metadata.

        Args:
            org_name (str): The name of the organization.
            dataset_id (str): The unique identifier of the dataset.
            request_dict (dict): Dictionary containing update parameters.
                - "type" (str): Required dataset type (unchangeable).
                - "format" (str): Required dataset format (unchangeable).

        Returns:
            Code: Response object indicating success (200) or failure (404 or 400).
        """
        metadata = resolve_metadata("dataset", dataset_id)
        if not metadata:
            return Code(404, {}, "Dataset not found")

        user_id = metadata.get("user_id")
        if not check_write_access(user_id, org_name, dataset_id, kind="datasets"):
            return Code(404, {}, "Dataset not available")
        if request_dict.get("public", None):
            if request_dict["public"]:
                add_public_dataset(dataset_id)
            else:
                remove_public_dataset(dataset_id)
        pull = False
        for key in request_dict.keys():

            # Cannot process the update, so return 400
            if key in ["type", "format"]:
                if request_dict[key] != metadata.get(key):
                    msg = f"Cannot change dataset {key}"
                    return Code(400, {}, msg)

            if key in [
                "name", "description", "version", "logo", "shared",
                "base_experiment_ids", "authorized_party_nca_id"
            ]:
                requested_value = request_dict[key]
                if requested_value:
                    metadata[key] = requested_value
                    metadata["last_modified"] = datetime.now(tz=timezone.utc)

            if key == "cloud_file_path":
                if metadata["status"] not in ("pull_complete", "invalid_pull"):
                    return Code(
                        400,
                        {},
                        f"Cloud file_path can be updated only when status is pull_complete or "
                        f"invalid_pull, the current status is {metadata['status']}. Try again after sometime"
                    )
                pull = True
                metadata["status"] = "starting"
                metadata["cloud_file_path"] = request_dict[key]

            if key == "docker_env_vars":
                # Encrypt the MLOPs keys
                requested_value = request_dict[key]
                config_path = os.getenv("VAULT_SECRET_PATH", None)
                if config_path:
                    encryption = NVVaultEncryption(config_path)
                    for mlops_key, value in requested_value.items():
                        if encryption.check_config()[0]:
                            metadata["docker_env_vars"][mlops_key] = encryption.encrypt(value)
                        elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                            return Code(400, {}, "Vault service does not work, can't enable MLOPs services")
                else:
                    metadata["docker_env_vars"] = requested_value

        # Pull dataset in background if known URL
        if pull:
            job_run_thread = threading.Thread(target=DatasetHandler.pull_dataset, args=(user_id, org_name, dataset_id,))
            job_run_thread.start()

        write_handler_metadata(dataset_id, metadata, "dataset")
        # Read this metadata from saved file...
        return_metadata = sanitize_handler_metadata(metadata)
        ret_Code = Code(200, return_metadata, "Dataset updated")
        return ret_Code

    @staticmethod
    def retrieve_dataset(org_name, dataset_id):
        """Retrieves metadata for an existing dataset.

        Args:
            org_name (str): The name of the organization.
            dataset_id (str): The unique identifier of the dataset.

        Returns:
            Code: Response object containing dataset metadata if found (200) or an error (404).
        """
        handler_metadata = resolve_metadata("dataset", dataset_id)
        if not handler_metadata:
            return Code(404, {}, "Dataset not found")

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, dataset_id, kind="datasets"):
            return Code(404, {}, "Dataset not found")

        return_metadata = sanitize_handler_metadata(handler_metadata)
        if return_metadata.get("status") == "invalid_pull":
            # Include detailed validation error information if available
            validation_details = return_metadata.get("validation_details", {})
            if validation_details:
                error_msg = validation_details.get("error_details", "Dataset validation failed")
                return Code(404, return_metadata, error_msg, use_data_as_response=True)
            return Code(404, return_metadata, "Dataset pulled from cloud doesn't match folder structure required")
        return Code(200, return_metadata, "Dataset retrieved")

    @staticmethod
    def delete_dataset(org_name, dataset_id):
        """Deletes a dataset if it is not in use or restricted.

        Args:
            org_name (str): Name of the organization requesting the deletion.
            dataset_id (str): UUID of the dataset to be deleted.

        Returns:
            Code: Response object containing:
                - 200 with metadata of the deleted dataset if successful.
                - 404 if the user lacks access to the dataset.
                - 400 if the dataset is in use by a running job or an active experiment.
        """
        handler_metadata = resolve_metadata("dataset", dataset_id)
        if not handler_metadata:
            return Code(200, {}, f"Dataset {dataset_id} not exists, should have been deleted already")

        user_id = handler_metadata.get("user_id")
        if not check_write_access(user_id, org_name, dataset_id, kind="datasets"):
            return Code(404, {}, f"Dataset {dataset_id} is not owned by you")

        # Check if any job running
        for job in handler_metadata.get("jobs", {}):
            if handler_metadata["jobs"][job]["status"] == "Running":
                return Code(400, {}, f"Dataset {dataset_id} in use by job {job}")

        # Check if dataset is public, then someone could be running it
        if handler_metadata.get("public", False):
            return Code(400, {}, f"Dataset {dataset_id} is Public. Cannot delete")

        # Check if dataset is read only, if yes, cannot delete
        if handler_metadata.get("read_only", False):
            return Code(400, {}, f"Dataset {dataset_id} is read only. Cannot delete")

        mongo_users = MongoHandler("tao", "users")
        datasets = get_user_datasets(user_id, mongo_users)
        if dataset_id in datasets:
            datasets.remove(dataset_id)
        user_query = {'id': user_id}
        mongo_users.upsert(user_query, {'id': user_id, 'datasets': datasets})

        mongo_datasets = MongoHandler("tao", "datasets")
        dataset_query = {'id': dataset_id}
        mongo_datasets.delete_one(dataset_query)
        from ..utils.basic_utils import delete_jobs_for_handler
        delete_jobs_for_handler(dataset_id, "dataset")
        # TODO: Delete logs for dataset
        return_metadata = sanitize_handler_metadata(handler_metadata)
        return Code(200, return_metadata, "Dataset deleted")

    @staticmethod
    def validate_dataset(user_id, org_name, dataset_id, temp_dir=None, file_path=None):
        """Validates a dataset and updates its status accordingly.

        Args:
            user_id (str): UUID of the user requesting validation.
            org_name (str): Name of the organization.
            dataset_id (str): UUID of the dataset to be validated.
            temp_dir (str, optional): Path to the temporary directory for dataset processing.
            file_path (str, optional): Path to the dataset file or folder.

        Returns:
            Code: Response object containing:
                - 200 with an empty dictionary if validation starts successfully.
                - 404 if the dataset is not found or access is denied.
                - 400 if validation fails due to structural issues.
        """
        metadata = resolve_metadata("dataset", dataset_id)
        if not metadata:
            return Code(404, {}, "Dataset not found")

        if not check_write_access(user_id, org_name, dataset_id, kind="datasets"):
            return Code(404, {}, "Dataset not available")

        try:
            metadata["status"] = "in_progress"
            write_handler_metadata(dataset_id, metadata, "dataset")

            def validate_dataset_thread():
                try:
                    # For cloud-based validation, resolve workspace metadata
                    workspace_metadata = None
                    if metadata.get("workspace"):
                        workspace_metadata = resolve_metadata("workspace", metadata.get("workspace"))

                    valid_dataset_structure, validation_result = validate_dataset(
                        org_name,
                        metadata,
                        temp_dir=temp_dir,
                        workspace_metadata=workspace_metadata
                    )
                    # Only remove temp_dir if it was actually created (not empty for cloud validation)
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)

                    if valid_dataset_structure:
                        metadata["status"] = "pull_complete"
                    else:
                        metadata["status"] = "invalid_pull"
                        # Store detailed validation information in metadata for user feedback
                        metadata["validation_details"] = {
                            "error_details": validation_result.get("error_details", "Unknown validation error"),
                            "expected_structure": validation_result.get("expected_structure", {}),
                            "actual_structure": validation_result.get("actual_structure", []),
                            "missing_files": validation_result.get("missing_files", []),
                            "network_type": validation_result.get("network_type", ""),
                            "dataset_format": validation_result.get("dataset_format", ""),
                            "dataset_intent": validation_result.get("dataset_intent", [])
                        }
                        logger.error(
                            "Dataset structure validation failed for dataset %s. "
                            "Expected structure: %s. Actual files: %s. Missing files: %s. Error: %s",
                            dataset_id,
                            validation_result.get("expected_structure", {}),
                            validation_result.get("actual_structure", []),
                            validation_result.get("missing_files", []),
                            validation_result.get("error_details", ""))

                    write_handler_metadata(dataset_id, metadata, "dataset")
                except Exception as e:
                    logger.error("Exception thrown in validate_dataset_thread is %s", str(e))
                    logger.error(traceback.format_exc())
                    metadata["status"] = "invalid_pull"
                    metadata["validation_details"] = {
                        "error_details": f"Validation process failed: {str(e)}",
                        "expected_structure": {},
                        "actual_structure": [],
                        "missing_files": [],
                        "network_type": metadata.get("type", ""),
                        "dataset_format": metadata.get("format", ""),
                        "dataset_intent": metadata.get("use_for", [])
                    }
                    write_handler_metadata(dataset_id, metadata, "dataset")

            thread = threading.Thread(target=validate_dataset_thread)
            thread.start()
            return Code(200, {}, "Server recieved file and upload process started")
        except Exception as e:
            logger.error("Exception thrown in validate_dataset is %s", str(e))
            logger.error(traceback.format_exc())
            metadata["status"] = "invalid_pull"
            write_handler_metadata(dataset_id, metadata, "dataset")
            return Code(404, [], "Exception caught during upload")

    @staticmethod
    def pull_dataset(user_id, org_name, dataset_id):
        """Initiates the process of validating a dataset, optimizing for cloud-based datasets.

        Args:
            user_id (str): UUID of the user requesting the dataset pull.
            org_name (str): Name of the organization.
            dataset_id (str): UUID of the dataset to be pulled.

        Notes:
            - For cloud-based datasets: validates structure directly without downloading.
            - For public URLs/HuggingFace: downloads first then validates.
            - Updates dataset status upon failure.
        """
        try:
            metadata = resolve_metadata("dataset", dataset_id)
            if not metadata:
                logger.error("Dataset metadata not found for %s", dataset_id)
                return

            # Check if this is a cloud-based dataset that can use cloud peek validation
            cloud_file_path = metadata.get("cloud_file_path")
            workspace_id = metadata.get("workspace")
            dataset_url = metadata.get("url")

            # Determine if we can use cloud peek validation (avoid download)
            can_use_cloud_peek = (
                cloud_file_path and
                workspace_id and
                not dataset_url  # No external URL means it's cloud storage based
            )

            if can_use_cloud_peek:
                # Validate directly from cloud without downloading
                DatasetHandler.validate_dataset(user_id, org_name, dataset_id, temp_dir="", file_path="")
            else:
                logger.info("Using download validation for dataset %s (url: %s)", dataset_id, dataset_url)
                temp_dir, file_path = download_dataset(dataset_id)
                DatasetHandler.validate_dataset(user_id, org_name, dataset_id, temp_dir=temp_dir, file_path=file_path)
        except Exception as e:
            logger.error("Exception thrown in pull_dataset is %s", str(e))
            logger.error(traceback.format_exc())
            metadata = resolve_metadata("dataset", dataset_id)
            metadata["status"] = "invalid_pull"
            write_handler_metadata(dataset_id, metadata, "dataset")
