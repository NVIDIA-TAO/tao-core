# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experiment handler module for managing experiment operations"""
import os
import uuid
import logging
import traceback
from datetime import datetime, timezone

from nvidia_tao_core.microservices.constants import (
    AUTOML_DISABLED_NETWORKS,
    TENSORBOARD_DISABLED_NETWORKS,
    TENSORBOARD_EXPERIMENT_LIMIT,
    TAO_NETWORKS,
)
from nvidia_tao_core.microservices.utils.airgapped_utils import AirgappedExperimentLoader
from nvidia_tao_core.microservices.enum_constants import ExperimentNetworkArch
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    check_read_access,
    check_write_access,
    get_handler_metadata,
    get_handler_metadata_with_jobs,
    get_handler_status,
    get_public_experiments,
    infer_action_from_job,
    sanitize_handler_metadata,
    validate_chained_actions,
    write_handler_metadata,
    add_public_experiment,
    remove_public_experiment,
    validate_automl_settings,
    resolve_metadata,
    get_automl_controller_info,
    get_automl_current_rec,
    get_automl_best_rec_info,
    get_handler_job_metadata,
    is_request_automl,
    delete_dnn_status,
)
from nvidia_tao_core.microservices.utils.encrypt_utils import NVVaultEncryption
from nvidia_tao_core.microservices.handlers.tensorboard_handler import TensorboardHandler
from nvidia_tao_core.microservices.utils.handler_utils import (
    Code,
    validate_and_update_experiment_metadata
)
from nvidia_tao_core.microservices.utils.core_utils import (
    read_network_config,
)

if os.getenv("BACKEND"):
    from nvidia_tao_core.microservices.utils.mongo_utils import MongoHandler
else:
    MongoHandler = None  # type: ignore

from ..utils.basic_utils import (
    get_org_experiments,
    get_user_experiments,
    get_experiment,
)

# Configure logging
logger = logging.getLogger(__name__)

# Identify if nginx-ingress is enabled
ingress_enabled = os.getenv("INGRESSENABLED", "false") == "true"


class ExperimentHandler:
    """Handles experiment creation, updating, deletion and retrieval."""

    @staticmethod
    def list_experiments(user_id, org_name, user_only=False):
        """Lists experiments accessible by the user.

        Args:
            user_id (str): The UUID of the user.
            org_name (str): The name of the organization.
            user_only (bool): Flag to indicate whether to list only experiments owned by the user.

        Returns:
            list(dict): A list of dictionaries containing metadata of experiments accessible by the user.
        """
        # Collect all metadatas
        metadatas = []
        for experiment_id in list(set(get_org_experiments(org_name))):
            handler_metadata = get_handler_metadata(experiment_id, "experiments")
            shared_experiment = handler_metadata.get("shared", False)
            if handler_metadata:
                if shared_experiment or handler_metadata.get("user_id") == user_id:
                    handler_metadata = sanitize_handler_metadata(handler_metadata)
                    handler_metadata["status"] = get_handler_status(handler_metadata)
                    metadatas.append(handler_metadata)
        if not user_only:
            public_experiments_metadata = get_public_experiments()
            metadatas += public_experiments_metadata
        return metadatas

    @staticmethod
    def list_base_experiments(user_id, org_name):
        """Lists public base experiments.

        Returns:
            list(dict): A list of dictionaries containing metadata of publicly accessible base experiments.
        """
        # Collect all metadatas
        metadatas = []
        public_experiments_metadata = get_public_experiments()
        metadatas += public_experiments_metadata
        return metadatas

    @staticmethod
    def load_airgapped_experiments(user_id, org_name, workspace_id):
        """Load airgapped experiments from cloud storage using workspace credentials.

        Args:
            user_id (str): The UUID of the user.
            org_name (str): The name of the organization.
            workspace_id (str): The UUID of the workspace containing cloud credentials.
            models_base_dir (str, optional): Base directory for searching model files.

        Returns:
            Code: A response object indicating the result of the operation.
                - 200 if experiments are loaded successfully.
                - 400 if there's an error with the request or configuration.
                - 403 if access denied to workspace.
                - 404 if workspace not found.
        """
        # Get workspace metadata
        workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
        if not workspace_metadata:
            return Code(404, {"error_desc": f"Workspace {workspace_id} not found", "error_code": 1},
                        f"Workspace {workspace_id} not found")

        # Check workspace access
        workspace_user_id = workspace_metadata.get('user_id')
        if workspace_user_id != user_id:
            return Code(403, {"error_desc": "Access denied to workspace", "error_code": 1},
                        "Access denied to workspace")

        # Map workspace cloud credentials to cloud_config format
        cloud_type = workspace_metadata.get("cloud_type", "seaweedfs")
        cloud_specific_details = workspace_metadata.get("cloud_specific_details", {})

        cloud_config = {
            "cloud_type": cloud_type,
            "bucket_name": cloud_specific_details.get("cloud_bucket_name", "tao-storage"),
            "region": cloud_specific_details.get("cloud_region"),
            "access_key": cloud_specific_details.get("access_key"),
            "secret_key": cloud_specific_details.get("secret_key"),
            "endpoint_url": cloud_specific_details.get("endpoint_url")
        }

        # Initialize and run airgapped loader (dry_run=False to save to MongoDB)
        try:
            loader = AirgappedExperimentLoader(
                cloud_config=cloud_config
            )

            # Load and import experiments to MongoDB
            success = loader.load_and_import()

            if success:
                return_metadata = {
                    "success": True,
                    "message": "Successfully loaded airgapped experiments to MongoDB",
                    "experiments_loaded": 1,  # We don't have exact counts from the loader
                    "experiments_failed": 0
                }
                return Code(200, return_metadata, "Successfully loaded airgapped experiments to MongoDB")
            return_metadata = {
                "success": False,
                "message": "Failed to load airgapped experiments",
                "experiments_loaded": 0,
                "experiments_failed": 1
            }
            return Code(400, return_metadata, "Failed to load airgapped experiments")

        except Exception as e:
            return_metadata = {
                "success": False,
                "message": f"Error loading airgapped experiments: {str(e)}",
                "experiments_loaded": 0,
                "experiments_failed": 1,
                "error_desc": str(e),
                "error_code": 1
            }
            return Code(400, return_metadata, f"Error loading airgapped experiments: {str(e)}")
        finally:
            # Clean up
            try:
                if 'loader' in locals():
                    loader.cleanup()
            except Exception:
                pass  # Ignore cleanup errors

    @staticmethod
    def create_experiment(user_id, org_name, request_dict, experiment_id=None, from_ui=False):
        """Creates a new experiment with the specified metadata.

        Args:
            user_id (str): The UUID of the user.
            org_name (str): The name of the organization.
            request_dict (dict): A dictionary containing the experiment details, adhering to the `ExperimentReqSchema`.
            experiment_id (str, optional): The ID of the experiment to be created (auto-generated if not provided).
            from_ui (bool): Flag indicating if the experiment creation request originated from the UI.

        Returns:
            Code: A response object indicating the result of the experiment creation.
                - 200 if the experiment is created successfully with metadata.
                - 400 if invalid data is provided in the request (e.g., missing or incorrect fields).
        """
        # Create a dataset ID and its root
        experiment_id = experiment_id or str(uuid.uuid4())

        workspace_id = request_dict.get("workspace", None)
        if workspace_id and not check_read_access(user_id, org_name, workspace_id, kind="workspaces"):
            return Code(404, None, f"Workspace {workspace_id} not found")

        # Gather type,format fields from request
        mdl_nw = request_dict.get("network_arch", None)
        # Perform basic checks - valid type and format?
        if mdl_nw not in ExperimentNetworkArch.__members__:
            msg = "Invalid network arch"
            return Code(400, {}, msg)

        if request_dict.get("public", False):
            add_public_experiment(experiment_id)

        # Create metadata dict and create some initial folders
        # Initially make datasets, base_experiment None
        metadata = {"id": experiment_id,
                    "user_id": user_id,
                    "org_name": org_name,
                    "authorized_party_nca_id": request_dict.get("authorized_party_nca_id", ""),
                    "created_on": datetime.now(tz=timezone.utc),
                    "last_modified": datetime.now(tz=timezone.utc),
                    "name": request_dict.get("name", "My Experiment"),
                    "shared": request_dict.get("shared", False),
                    "description": request_dict.get("description", "My Experiments"),
                    "version": request_dict.get("version", "1.0.0"),
                    "logo": request_dict.get("logo", "https://www.nvidia.com"),
                    "ngc_path": request_dict.get("ngc_path", ""),
                    "encryption_key": request_dict.get("encryption_key", "tlt_encode"),
                    "read_only": request_dict.get("read_only", False),
                    "public": request_dict.get("public", False),
                    "network_arch": mdl_nw,
                    "dataset_type": read_network_config(mdl_nw)["api_params"]["dataset_type"],
                    "dataset_formats": read_network_config(mdl_nw)["api_params"].get(
                        "formats",
                        read_network_config(
                            read_network_config(mdl_nw)["api_params"]["dataset_type"]
                        ).get("api_params", {}).get("formats", None)
                    ),
                    "accepted_dataset_intents": read_network_config(mdl_nw)["api_params"].get(
                        "accepted_ds_intents",
                        []
                    ),
                    "actions": read_network_config(mdl_nw)["api_params"]["actions"],
                    "docker_env_vars": request_dict.get("docker_env_vars", {}),
                    "train_datasets": [],
                    "eval_dataset": None,
                    "inference_dataset": None,
                    "additional_id_info": None,
                    "checkpoint_choose_method": request_dict.get("checkpoint_choose_method", "best_model"),
                    "checkpoint_epoch_number": request_dict.get("checkpoint_epoch_number", {}),
                    "calibration_dataset": None,
                    # New fields for direct dataset paths
                    "train_dataset_uris": request_dict.get("train_dataset_uris"),
                    "eval_dataset_uri": request_dict.get("eval_dataset_uri"),
                    "inference_dataset_uri": request_dict.get("inference_dataset_uri"),
                    "calibration_dataset_uri": request_dict.get("calibration_dataset_uri"),
                    "dataset_format": request_dict.get("dataset_format"),
                    "base_experiment_ids": [],
                    "automl_settings": request_dict.get("automl_settings", {}),
                    "metric": request_dict.get("metric", "kpi"),
                    "model_params": request_dict.get("model_params", {}),
                    "tensorboard_enabled": request_dict.get("tensorboard_enabled", False),
                    "workspace": request_dict.get("workspace", None),
                    "experiment_actions": request_dict.get('experiment_actions', []),
                    "tags": list({t.lower(): t for t in request_dict.get("tags", [])}.values()),
                    }

        if metadata.get("automl_settings", {}).get("automl_enabled") and mdl_nw in AUTOML_DISABLED_NETWORKS:
            return Code(400, {}, "automl_enabled cannot be True for unsupported network")
        if mdl_nw in TAO_NETWORKS and (not metadata.get("workspace")):
            return Code(400, {}, "Workspace must be provided for experiment creation")
        if not ingress_enabled and metadata.get("tensorboard_enabled", False):
            return Code(400, {}, "Tensorboard not available without ingress-nginx")

        automl_validation_error = validate_automl_settings(metadata.get("automl_settings", {}))
        if automl_validation_error:
            return Code(400, {}, automl_validation_error)

        # Encrypt the MLOPs keys
        config_path = os.getenv("VAULT_SECRET_PATH", None)
        if config_path and metadata["docker_env_vars"]:
            encryption = NVVaultEncryption(config_path)
            for key, value in metadata["docker_env_vars"].items():
                if encryption.check_config()[0]:
                    metadata["docker_env_vars"][key] = encryption.encrypt(value)
                elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                    return Code(400, {}, "Vault service does not work, can't enable MLOPs services")

        # Update datasets and base_experiments if given.
        # need to prepare base_experiment first
        metadata, error_code = validate_and_update_experiment_metadata(
            user_id,
            org_name,
            request_dict,
            metadata,
            [
                "train_datasets",
                "eval_dataset",
                "inference_dataset",
                "calibration_dataset",
                "base_experiment_ids"
            ]
        )
        if error_code:
            return error_code

        # Dataset structure validation (checks for required files like annotations.json)
        train_dataset_uris = request_dict.get("train_dataset_uris")
        eval_dataset_uri = request_dict.get("eval_dataset_uri")
        inference_dataset_uri = request_dict.get("inference_dataset_uri")
        calibration_dataset_uri = request_dict.get("calibration_dataset_uri")
        skip_validation = request_dict.get("skip_dataset_validation", False)

        if any([
            train_dataset_uris,
            eval_dataset_uri,
            inference_dataset_uri,
            calibration_dataset_uri
        ]) and not skip_validation:
            from nvidia_tao_core.microservices.utils.runtime_dataset_validator import (
                validate_all_dataset_uris_structure
            )

            network_arch = request_dict.get("network_arch")
            if not network_arch:
                return Code(400, {}, "network_arch is required for dataset validation")

            # Prepare metadata for validation
            validation_metadata = {
                "train_dataset_uris": train_dataset_uris,
                "eval_dataset_uri": eval_dataset_uri,
                "inference_dataset_uri": inference_dataset_uri,
                "calibration_dataset_uri": calibration_dataset_uri,
                "dataset_format": request_dict.get("dataset_format"),
                "dataset_type": request_dict.get("dataset_type"),
                "workspace": request_dict.get("workspace")
            }

            is_valid, error_msg, validation_details = validate_all_dataset_uris_structure(
                validation_metadata,
                network_arch,
                skip_validation=False
            )

            if not is_valid:
                # Return detailed validation error
                return Code(400, validation_details, error_msg)

            logger.info(
                f"Dataset validation passed for experiment: "
                f"{validation_details.get('message')}"
            )

        def clean_on_error(experiment_id=experiment_id):
            mongo_experiments = MongoHandler("tao", "experiments")
            mongo_experiments.delete_one({'id': experiment_id})

        # Create Tensorboard deployment if enabled
        if metadata.get("tensorboard_enabled", False):
            if mdl_nw in TENSORBOARD_DISABLED_NETWORKS:
                clean_on_error(experiment_id)
                return Code(400, {}, f"Network {mdl_nw} not supported for Tensorboard")
            workspace_id = request_dict.get("workspace", None)
            if TensorboardHandler.check_user_metadata(user_id) and workspace_id:
                response = TensorboardHandler.start(org_name, experiment_id, user_id, workspace_id)
                if response.code != 200:
                    TensorboardHandler.stop(experiment_id, user_id)
                    clean_on_error(experiment_id)
                    return response
            else:
                clean_on_error(experiment_id)
                return Code(
                    400,
                    {},
                    f"Maximum of {TENSORBOARD_EXPERIMENT_LIMIT} Tensorboard Experiments allowed per user."
                )

        # Actual "creation" happens here...
        write_handler_metadata(experiment_id, metadata, "experiment")

        mongo_users = MongoHandler("tao", "users")
        experiments = get_user_experiments(user_id, mongo_users)
        experiments.append(experiment_id)
        mongo_users.upsert({'id': user_id}, {'id': user_id, 'experiments': experiments})

        experiment_actions = request_dict.get('experiment_actions', [])
        retry_experiment_id = request_dict.get('retry_experiment_id', None)
        error_response = None
        if retry_experiment_id:
            error_response = ExperimentHandler.retry_experiment(
                org_name, user_id, retry_experiment_id, experiment_id, from_ui
            )
        elif experiment_actions:
            error_response = ExperimentHandler.retry_experiment_actions(
                user_id,
                org_name,
                experiment_id,
                experiment_actions,
                from_ui
            )
        if error_response:
            clean_on_error(experiment_id)
            return error_response

        # Read this metadata from saved file...
        return_metadata = sanitize_handler_metadata(metadata)
        ret_Code = Code(200, return_metadata, "Experiment created")

        return ret_Code

    @staticmethod
    def retry_experiment(org_name, user_id, retry_experiment_id, new_experiment_id, from_ui):
        """Retries all jobs within an experiment by reusing the specs from already run jobs.

        Args:
            org_name (str): Organization name.
            user_id (str): User ID initiating the retry.
            retry_experiment_id (str): ID of the experiment whose jobs are to be retried.
            new_experiment_id (str): ID of the new experiment to create jobs for.
            from_ui (bool): Indicates whether the retry was triggered from the UI.

        Returns:
            Response: A response indicating the outcome of the operation (201 for success, error response otherwise).
        """
        from .job_handler import JobHandler
        from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_job_specs
        from ..utils.basic_utils import get_job
        handler_metadata = get_handler_metadata_with_jobs(retry_experiment_id, "experiments")
        handler_jobs = handler_metadata.get("jobs", [])
        job_map = {}
        for job in handler_jobs:
            job_id = job.get('id')
            job_action = job.get('action')
            if job_id and job_action:
                logger.info("Loading existing specs from job %s", job_id)
                specs = get_job_specs(job_id)
                name = job.get('name')
                description = job.get('description')
                retry_parent_job_id = job.get('parent_id', None)
                parent_job_id = None
                if retry_parent_job_id:
                    retry_parent_job = get_job(retry_parent_job_id)
                    parent_action = retry_parent_job.get('action')
                    parent_job_id = job_map.get(parent_action, None)
                response = JobHandler.job_run(
                    org_name=org_name,
                    handler_id=new_experiment_id,
                    parent_job_id=parent_job_id,
                    action=job_action,
                    kind='experiment',
                    specs=specs,
                    name=name,
                    description=description,
                    from_ui=from_ui
                )
                if response.code == 200:
                    job_id = response.data
                    job_map[job_action] = job_id
                    logger.info(
                        f"Created {job_action} job with id {job_id} for experiment {new_experiment_id}"
                    )
                else:
                    return response
        return None

    @staticmethod
    def retry_experiment_actions(user_id, org_name, experiment_id, experiment_actions, from_ui):
        """Retries specific jobs within an experiment based on the provided actions.

        Args:
            user_id (str): User ID initiating the retry.
            org_name (str): Organization name.
            experiment_id (str): ID of the experiment to retry jobs within.
            experiment_actions (list): List of actions with details to be retried.
            from_ui (bool): Indicates whether the retry was triggered from the UI.

        Returns:
            Response: A response indicating the outcome of the operation (201 for success, error response otherwise).
        """
        from .job_handler import JobHandler
        from .spec_handler import SpecHandler
        raw_actions = []
        action_lookup = {}
        for action_dict in experiment_actions:
            action = action_dict.get('action')
            specs = action_dict.get('specs', {})
            name = action_dict.get('name')
            description = action_dict.get('description')
            num_gpu = action_dict.get('num_gpu', -1)
            platform_id = action_dict.get('platform_id', None)
            action_data = {
                'specs': specs,
                'name': name,
                'description': description,
                'num_gpu': num_gpu,
                'platform_id': platform_id
            }
            if action:
                raw_actions.append(action)
                action_lookup[action] = action_data

        if raw_actions and action_lookup:
            job_mapping = validate_chained_actions(raw_actions)
            if not job_mapping:
                return Code(400, {}, "Invalid workflow chaining")

            job_action_to_id = {}
            for mapping in job_mapping:
                child_action = mapping.get('child')
                parent_action = mapping.get('parent', None)
                if child_action in action_lookup:
                    lookup_data = action_lookup[child_action]
                    specs = {}
                    if not specs and not lookup_data.get('specs', {}):
                        specs_response = SpecHandler.get_spec_schema(
                            user_id,
                            org_name,
                            experiment_id,
                            child_action,
                            'experiment'
                        )
                        if specs_response.code == 200:
                            spec_schema = specs_response.data
                            specs = spec_schema["default"]
                            logger.info("Retrieved specs from DNN: %s", specs)
                        else:
                            return specs_response
                    else:
                        specs = action_lookup[child_action].get('specs', {})
                    name = lookup_data.get('name')
                    description = lookup_data.get('description')
                    num_gpu = lookup_data.get('num_gpu', -1)
                    backend_details = lookup_data.get('backend_details', None)
                    parent_job_id = job_action_to_id.get(parent_action, None)
                    response = JobHandler.job_run(
                        org_name=org_name,
                        handler_id=experiment_id,
                        parent_job_id=parent_job_id,
                        action=child_action,
                        kind='experiment',
                        specs=specs,
                        name=name,
                        description=description,
                        num_gpu=num_gpu,
                        backend_details=backend_details,
                        from_ui=from_ui
                    )
                    if response.code == 200:
                        job_id = response.data
                        logger.info(
                            f"Created {child_action} job with id {job_id} for experiment {experiment_id}"
                        )
                        job_action_to_id[child_action] = job_id
                    else:
                        return response
        return None

    @staticmethod
    def update_experiment(org_name, experiment_id, request_dict):
        """Updates the metadata of an existing experiment based on the provided request data.

        Args:
            org_name (str): Organization name.
            experiment_id (str): ID of the experiment to update.
            request_dict (dict): Dictionary containing the fields to update, following the ExperimentReqSchema.

        Returns:
            Response: A response indicating the outcome of the operation (200 for success, error responses for failure).
        """
        metadata = resolve_metadata("experiment", experiment_id)
        if not metadata:
            return Code(400, {}, "Experiment does not exist")

        user_id = metadata.get("user_id")
        if not check_write_access(user_id, org_name, experiment_id, kind="experiments"):
            return Code(400, {}, "User doesn't have write access to experiment")

        # if public is set to True => add it to public_experiments, if it is set to False => take it down
        # if public is not there, do nothing
        if request_dict.get("public", None):
            if request_dict["public"]:
                add_public_experiment(experiment_id)
            else:
                remove_public_experiment(experiment_id)

        user_id = metadata.get("user_id")
        workspace_id = metadata.get('workspace')
        for key in request_dict.keys():

            # Cannot process the update, so return 400
            if key in ["network_arch", "experiment_params", "base_experiment_metadata"]:
                if request_dict[key] != metadata.get(key):
                    msg = f"Cannot change experiment {key}"
                    return Code(400, {}, msg)

            if key in ["name", "description", "version", "logo",
                       "ngc_path", "encryption_key", "read_only",
                       "metric", "public", "shared", "tags", "authorized_party_nca_id"]:
                requested_value = request_dict[key]
                if requested_value is not None:
                    metadata[key] = requested_value
                    metadata["last_modified"] = datetime.now(tz=timezone.utc)
                    if key == "tags":
                        metadata[key] = list({t.lower(): t for t in requested_value}.values())

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

            metadata, error_code = validate_and_update_experiment_metadata(
                user_id,
                org_name,
                request_dict,
                metadata,
                [
                    "train_datasets",
                    "eval_dataset",
                    "inference_dataset",
                    "calibration_dataset",
                    "train_dataset_uris",
                    "eval_dataset_uri",
                    "inference_dataset_uri",
                    "calibration_dataset_uri",
                    "base_experiment_ids",
                    "checkpoint_choose_method",
                    "checkpoint_epoch_number"
                ]
            )
            if error_code:
                return error_code

            automl_enabled = metadata.get("automl_settings", {}).get("automl_enabled", False)
            tensorboard_enabled = metadata.get("tensorboard_enabled", False)
            if key == "automl_settings":
                value = request_dict[key]
                automl_enabled = value.get('automl_enabled', False)
                # If False, can set. If True, need to check if AutoML is supported
                if value:
                    mdl_nw = metadata.get("network_arch", "")
                    if mdl_nw not in AUTOML_DISABLED_NETWORKS:
                        metadata[key] = request_dict.get(key, {})
                    else:
                        return Code(400, {}, "automl_enabled cannot be True for unsupported network")
                    automl_validation_error = validate_automl_settings(value)
                    if automl_validation_error:
                        return Code(400, {}, automl_validation_error)
                else:
                    metadata[key] = value

            if key == "tensorboard_enabled":
                value = request_dict[key]
                mdl_nw = metadata.get("network_arch", "")
                if not tensorboard_enabled and value:  # Enable Tensorboard
                    if mdl_nw in TENSORBOARD_DISABLED_NETWORKS:
                        return Code(400, {}, f"Network {mdl_nw} not supported for Tensorboard")
                    if automl_enabled:
                        return Code(400, {}, "AutoML not supported yet for Tensorboard")
                    if TensorboardHandler.check_user_metadata(user_id) and workspace_id:
                        response = TensorboardHandler.start(org_name, experiment_id, user_id, workspace_id)
                        if response.code != 200:
                            TensorboardHandler.stop(experiment_id, user_id)
                            return response
                    else:
                        return Code(
                            400,
                            {},
                            f"Maximum of {TENSORBOARD_EXPERIMENT_LIMIT} Tensorboard Experiments allowed per user. "
                        )
                elif tensorboard_enabled and not value:  # Disable Tensorboard
                    response = TensorboardHandler.stop(experiment_id, user_id)
                    if response.code != 200:
                        return response
                metadata[key] = value

        write_handler_metadata(experiment_id, metadata, "experiment")
        # Read this metadata from saved file...
        return_metadata = sanitize_handler_metadata(metadata)
        ret_Code = Code(200, return_metadata, "Experiment updated")
        return ret_Code

    @staticmethod
    def retrieve_experiment(org_name, experiment_id):
        """Retrieves experiment metadata.

        Args:
            org_name (str): Organization name.
            experiment_id (str): ID of the experiment to retrieve.

        Returns:
            Response: A response indicating the outcome of the operation (200 with metadata for success,
                     404 if not found).
        """
        handler_metadata = resolve_metadata("experiment", experiment_id)
        if experiment_id not in ("*", "all") and not handler_metadata:
            return Code(404, {}, "Experiment not found")

        user_id = handler_metadata.get("user_id")
        if (experiment_id not in ("*", "all") and
                not check_read_access(user_id, org_name, experiment_id, kind="experiments")):
            return Code(404, {}, "Experiment not found")

        handler_metadata["status"] = get_handler_status(handler_metadata)
        return_metadata = sanitize_handler_metadata(handler_metadata)
        return Code(200, return_metadata, "Experiment retrieved")

    @staticmethod
    def delete_experiment(org_name, experiment_id):
        """Deletes an experiment if it is not in use by any job or other experiments.

        Args:
            org_name (str): Organization name.
            experiment_id (str): ID of the experiment to delete.

        Returns:
            Response: A response indicating the outcome of the operation (200 for success, error responses for failure).
        """
        from ..utils.basic_utils import delete_jobs_for_handler
        handler_metadata = resolve_metadata("experiment", experiment_id)
        if not handler_metadata:
            return Code(200, {}, "Experiment deleted")
        user_id = handler_metadata.get("user_id")
        if not check_write_access(user_id, org_name, experiment_id, kind="experiments"):
            return Code(404, {}, "User doesn't have write access to experiment")

        # If experiment is being used by user's experiments.
        experiments = get_user_experiments(user_id)

        if experiment_id not in experiments:
            return Code(404, {}, f"Experiment {experiment_id} cannot be deleted")

        for handler_id in experiments:
            metadata = get_experiment(handler_id)
            if experiment_id in metadata.get("base_experiment_ids", []):
                return Code(400, {}, f"Experiment {experiment_id} in use as a base_experiment")

        for job in handler_metadata.get("jobs", {}):
            if handler_metadata["jobs"][job]["status"] in ("Pending", "Running"):
                return Code(400, {}, f"Experiment {experiment_id} in use by job {job}")

        # Check if experiment is public, then someone could be running it
        if handler_metadata.get("public", False):
            return Code(400, {}, f"Experiment {experiment_id} is Public. Cannot delete")

        # Check if experiment is read only, if yes, cannot delete
        if handler_metadata.get("read_only", False):
            return Code(400, {}, f"Experiment {experiment_id} is read only. Cannot delete")

        if handler_metadata.get("tensorboard_enabled", False):
            response = TensorboardHandler.stop(experiment_id, user_id)
            if response.code != 200:
                return response

        if experiment_id in experiments:
            experiments.remove(experiment_id)
            mongo_users = MongoHandler("tao", "users")
            mongo_users.upsert({'id': user_id}, {'id': user_id, 'experiments': experiments})

        delete_jobs_for_handler(experiment_id, "experiment")
        mongo_experiments = MongoHandler("tao", "experiments")
        mongo_experiments.delete_one({'id': experiment_id})
        return_metadata = sanitize_handler_metadata(handler_metadata)
        return Code(200, return_metadata, "Experiment deleted")

    @staticmethod
    def resume_experiment_job(
        org_name,
        experiment_id,
        job_id,
        kind,
        parent_job_id=None,
        specs=None,
        name=None,
        description=None,
        num_gpu=-1,
        timeout_minutes=None,
        backend_details=None
    ):
        """Resumes a paused experiment job, adding it back to the queue for processing.

        Args:
            org_name (str): Organization name.
            experiment_id (str): ID of the experiment to which the job belongs.
            job_id (str): ID of the job to resume.
            kind (str): Type of the experiment (e.g., "experiment").
            parent_job_id (str, optional): ID of the parent job if applicable.
            specs (dict, optional): Specifications for the resumed job.
            name (str, optional): Name of the job.
            description (str, optional): Description of the job.
            num_gpu (int, optional): Number of GPUs to allocate.
            platform_id (str, optional): Platform ID for the job.
            timeout_minutes (int, optional): The job-specific timeout in minutes. Defaults to None.
        Returns:
            Response: A response indicating the outcome of the operation (200 for success, error responses for failure).
        """
        from nvidia_tao_core.microservices.handlers.automl_handler import AutoMLHandler
        from nvidia_tao_core.microservices.utils.job_utils.workflow_driver import create_job_context, on_new_job
        from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_job_specs

        logger.debug(
            f"[RESUME] Starting resume operation: job_id={job_id}, "
            f"experiment_id={experiment_id}, kind={kind}, org_name={org_name}"
        )

        handler_metadata = resolve_metadata(kind, experiment_id)
        if not handler_metadata:
            logger.debug(f"[RESUME] Experiment not found: experiment_id={experiment_id}")
            return Code(404, [], "Experiment not found")

        user_id = handler_metadata.get("user_id")
        if not check_write_access(user_id, org_name, experiment_id, kind="experiments"):
            logger.debug(f"[RESUME] Write access denied: user_id={user_id}, experiment_id={experiment_id}")
            return Code(404, [], "Experiment not found")

        job_metadata = get_handler_job_metadata(job_id)
        if not job_metadata:
            logger.debug(f"[RESUME] Job metadata not found: job_id={job_id}")
            return Code(404, None, "job trying to resume not found")
        action = job_metadata.get("action", "")
        action = infer_action_from_job(experiment_id, job_id)
        status = job_metadata.get("status", "")
        if not timeout_minutes:
            timeout_minutes = job_metadata.get("timeout_minutes", 60)

        logger.debug(
            f"[RESUME] Job metadata retrieved: job_id={job_id}, action={action}, "
            f"status={status}, timeout_minutes={timeout_minutes}"
        )

        if status != "Paused":
            logger.debug(f"[RESUME] Job status is not Paused: job_id={job_id}, status={status}")
            return Code(400, [], f"Job status should be paused, not {status}")
        if action not in ("train", "distill", "quantize", "retrain"):
            logger.debug(f"[RESUME] Action not resumable: job_id={job_id}, action={action}")
            return Code(400, [], f"Action should be train, distill, quantize, retrain, not {action}")
        if not user_id:
            logger.debug(f"[RESUME] User ID not found in metadata: experiment_id={experiment_id}")
            return Code(
                404,
                [],
                "User ID couldn't be found in the experiment metadata. Try creating the experiment again"
            )

        msg = ""
        try:
            from nvidia_tao_core.microservices.utils.stateless_handler_utils import update_job_status
            logger.debug(f"[RESUME] Updating job status to Resuming: job_id={job_id}")
            update_job_status(experiment_id, job_id, status="Resuming", kind=kind + "s")
            # Reset timeout timer by clearing old status history
            delete_dnn_status(job_id, automl=False)
            logger.debug(f"[RESUME] Cleared status history for resumed job to reset timeout timer: job_id={job_id}")
            if not name:
                name = job_metadata.get("name", "")
                logger.debug(f"[RESUME] Using existing name from metadata: job_id={job_id}, name={name}")
            if not backend_details:
                logger.info("Loading existing backend_details from paused job")
                backend_details = job_metadata.get("backend_details", None)

            if is_request_automl(experiment_id, action, kind):
                msg = "AutoML "
                logger.debug(f"[RESUME] Resuming AutoML job: job_id={job_id}, experiment_id={experiment_id}")
                AutoMLHandler.resume(
                    user_id,
                    org_name,
                    experiment_id,
                    job_id,
                    handler_metadata,
                    name=name,
                    backend_details=backend_details,
                    timeout_minutes=timeout_minutes
                )
                logger.debug(f"[RESUME] AutoML resume handler called: job_id={job_id}")
            else:
                # Create a job and run it
                logger.debug(f"[RESUME] Resuming non-AutoML job: job_id={job_id}")
                if not specs:
                    specs = get_job_specs(job_id)
                    logger.debug(
                        f"[RESUME] Loading existing specs from paused job: "
                        f"job_id={job_id}, num_spec_keys={len(specs) if specs else 0}"
                    )
                if not parent_job_id:
                    parent_job_id = handler_metadata.get('parent_job_id', None)
                    logger.debug(
                        f"[RESUME] Loading existing parent_job_id from metadata: "
                        f"job_id={job_id}, parent_job_id={parent_job_id}"
                    )
                if not description:
                    description = job_metadata.get("description", "")
                if num_gpu == -1:
                    num_gpu = job_metadata.get("num_gpu", -1)
                retain_checkpoints_for_resume = job_metadata.get("retain_checkpoints_for_resume", False)
                early_stop_epoch = job_metadata.get("early_stop_epoch", None)
                timeout_minutes = job_metadata.get("timeout_minutes", None)
                logger.debug(
                    f"[RESUME] Job configuration: job_id={job_id}, num_gpu={num_gpu}, "
                    f"retain_checkpoints={retain_checkpoints_for_resume}, "
                    f"early_stop_epoch={early_stop_epoch}"
                )

                logger.debug(f"[RESUME] Creating job context: job_id={job_id}")
                job_context = create_job_context(
                    parent_job_id,
                    "train",
                    job_id,
                    experiment_id,
                    user_id,
                    org_name,
                    kind,
                    handler_metadata=handler_metadata,
                    specs=specs,
                    name=name,
                    description=description,
                    num_gpu=num_gpu,
                    retain_checkpoints_for_resume=retain_checkpoints_for_resume,
                    early_stop_epoch=early_stop_epoch,
                    timeout_minutes=timeout_minutes,
                    backend_details=backend_details
                )
                logger.debug(f"[RESUME] Calling on_new_job to queue resumed job: job_id={job_id}")
                on_new_job(job_context)
                logger.debug(f"[RESUME] Job queued successfully: job_id={job_id}")
            logger.debug(f"[RESUME] Resume operation completed successfully: job_id={job_id}")
            return Code(200, {"message": f"{msg}Action for job {job_id} resumed"})
        except Exception as e:
            logger.error(f"[RESUME] Exception thrown in resume_experiment_job: job_id={job_id}, error={str(e)}")
            logger.error(f"[RESUME] Traceback: {traceback.format_exc()}")
            try:
                from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
                    update_job_status as _update_status,
                    write_job_metadata
                )
                _update_status(experiment_id, job_id, status="Error", kind=kind + "s")
                err_meta = get_handler_job_metadata(job_id)
                if err_meta:
                    err_meta["status"] = "Error"
                    err_meta.setdefault("job_details", {})[job_id] = {
                        "detailed_status": {
                            "message": f"Job resume failed: {e}"
                        }
                    }
                    write_job_metadata(job_id, err_meta)
            except Exception as status_err:
                logger.error(f"[RESUME] Failed to update job {job_id} status to Error: {status_err}")
            return Code(400, [], f"Action cannot be resumed: {e}")

    @staticmethod
    def automl_details(org_name, experiment_id, job_id):
        """Retrieves AutoML details for a specific experiment and job.

        Args:
            org_name (str): Organization name.
            experiment_id (str): ID of the experiment.
            job_id (str): ID of the job.

        Returns:
            - Code(200, details, None) if successful.
            - Code(404, {}, "AutoML details not found") if the experiment does not have AutoML details
              or the experiment is not found.
        """
        try:
            handler_metadata = resolve_metadata("experiment", experiment_id)
            if not handler_metadata:
                return Code(404, [], "Experiment not found")

            user_id = handler_metadata.get("user_id")
            if not check_write_access(user_id, org_name, experiment_id, kind="experiments"):
                return Code(404, [], "Experiment not found")
            automl_controller_data = get_automl_controller_info(job_id)

            automl_interpretable_result = {}

            # Get current experiment id
            current_rec = get_automl_current_rec(job_id)
            if not current_rec:
                current_rec = 0
            automl_interpretable_result["current_experiment_id"] = current_rec

            # Get per experiment result and status
            automl_interpretable_result["experiments"] = {}
            for experiment_details in automl_controller_data:
                automl_interpretable_result["metric"] = experiment_details.get("metric")
                exp_id = experiment_details.get("id")
                # Convert exp_id to string for MongoDB compatibility (MongoDB requires string keys)
                exp_id_str = str(exp_id)
                automl_interpretable_result["experiments"][exp_id_str] = {}
                automl_interpretable_result["experiments"][exp_id_str]["result"] = experiment_details.get("result")
                automl_interpretable_result["experiments"][exp_id_str]["status"] = experiment_details.get("status")
                automl_interpretable_result["experiments"][exp_id_str]["specs"] = experiment_details.get("specs", {})
                automl_interpretable_result["experiments"][exp_id_str]["job_id"] = experiment_details.get("job_id", "")

            # Get the best experiment id from the automl_jobs table
            best_rec_number, _, _ = get_automl_best_rec_info(job_id)
            if best_rec_number and best_rec_number != "-1":
                automl_interpretable_result["best_experiment_id"] = int(best_rec_number)
            return Code(200, automl_interpretable_result, "AutoML results compiled")
        except Exception as e:
            logger.error("Exception thrown in automl_details fetch is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(400, [], "Error in constructing AutoML results")
