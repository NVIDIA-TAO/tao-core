# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spec handler module for managing specification schemas"""
import os
import logging
import traceback

from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    check_read_access,
    get_base_experiment_metadata,
    get_handler_job_metadata,
    get_handler_metadata,
    get_job_specs,
    is_request_automl
)
from nvidia_tao_core.microservices.utils.handler_utils import Code
from nvidia_tao_core.microservices.utils.specs_utils import csv_to_json_schema
from nvidia_tao_core.microservices.utils.stateless_handler_utils import BACKEND
from nvidia_tao_core.microservices.utils.core_utils import (
    merge_nested_dicts,
    override_dicts,
    get_microservices_network_and_action
)
from nvidia_tao_core.scripts.generate_schema import generate_schema, validate_and_clean_merged_spec
from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import ExecutionHandler

from ..utils.basic_utils import resolve_metadata

# Configure logging
logger = logging.getLogger(__name__)


class SpecHandler:
    """Handles specification schema operations."""

    @staticmethod
    def get_spec_schema(user_id, org_name, handler_id, action, kind):
        """Retrieves the specification schema for a dataset or experiment.

        Args:
            user_id (str): UUID of the user requesting the schema.
            org_name (str): Name of the organization.
            handler_id (str): UUID of the dataset or experiment.
            action (str): The specific action for which the schema is required.
            kind (str): Type of entity, either "experiment" or "dataset".

        Returns:
            Code: Response object containing:
                - 200 with the schema in JSON format if retrieval is successful.
                - 404 if the dataset/experiment is not found or access is denied.
                - 404 if the requested action is invalid.
        """
        metadata = resolve_metadata(kind, handler_id)
        if not metadata:
            return Code(404, {}, "Spec schema not found")

        if not check_read_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, {}, "Spec schema not available")

        # Action not available
        if action not in metadata.get("actions", []):
            if not (kind == "dataset" and action == "validate_images"):
                return Code(404, {}, "Action not found")

        base_experiment_spec = {}
        if metadata.get("base_experiment_ids", []):
            for base_experiment_id in metadata["base_experiment_ids"]:
                base_experiment_metadata = get_base_experiment_metadata(base_experiment_id)
                base_exp_meta = base_experiment_metadata.get("base_experiment_metadata", {})
                if base_experiment_metadata and base_exp_meta.get("spec_file_present"):
                    base_experiment_spec = base_exp_meta.get("specs", {})
                    if not base_experiment_spec:
                        return Code(404, {}, "Base specs not present.")

        # Read csv from utils/spec_utils/specs/<network_name>/action.csv
        # Convert to json schema
        json_schema = {}

        network = metadata.get("network_arch", None)
        if not network:
            # Used for dataset jobs
            network = metadata.get("type", None)

        microservices_network, microservices_action = get_microservices_network_and_action(network, action)

        try:
            json_schema = generate_schema(microservices_network, microservices_action)
        except Exception as e:
            logger.error("Exception thrown in get_spec_schema is %s", str(e))
            logger.error("Unable to fetch schema from tao_core")

        if not json_schema:
            DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            # Try regular format for CSV_PATH => "<network> - <action>.csv"
            CSV_PATH = os.path.join(DIR_PATH, "utils", "specs_utils", "specs", network, f"{network} - {action}.csv")
            if not os.path.exists(CSV_PATH):
                # Try secondary format for CSV_PATH => "<network> - <action>__<dataset-format>.csv"
                fmt = metadata.get("format", "_")
                CSV_PATH = os.path.join(
                    DIR_PATH,
                    "utils",
                    "specs_utils",
                    "specs",
                    network,
                    f"{network} - {action}__{fmt}.csv"
                )
                if not os.path.exists(CSV_PATH):
                    return Code(404, {}, "Default specs do not exist for action")
            json_schema = csv_to_json_schema.convert(CSV_PATH)

        if "default" in json_schema and base_experiment_spec:
            # Merge the base experiment spec with the default schema
            merged_default = merge_nested_dicts(json_schema["default"], base_experiment_spec)
            # Validate and clean the merged spec to remove any invalid keys from corrupt base_experiment_spec
            logger.info("Validating merged base_experiment_spec")
            json_schema["default"] = validate_and_clean_merged_spec(json_schema, merged_default)
        return Code(200, json_schema, "Schema retrieved")

    @staticmethod
    def get_spec_schema_for_job(user_id, org_name, handler_id, job_id, kind):
        """Retrieves the specification schema for a specific job within an experiment or dataset.

        Args:
            user_id (str): UUID of the user requesting the schema.
            org_name (str): Name of the organization.
            handler_id (str): UUID of the dataset or experiment.
            job_id (str): UUID of the job.
            kind (str): Type of entity, either "experiment" or "dataset".

        Returns:
            Code: Response object containing:
                - 200 with the schema in JSON format if retrieval is successful.
                - 404 if the dataset/experiment or job is not found or access is denied.
        """
        metadata = resolve_metadata(kind, handler_id)
        if not metadata:
            return Code(404, {}, "Spec schema not found")

        if not check_read_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, {}, "Spec schema not available")

        job_metadata = get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, [], "Job trying to get schema for not found")
        action = job_metadata.get("action", "")

        job_specs = get_job_specs(job_id)

        json_schema = {}

        network = metadata.get("network_arch", None)
        if not network:
            # Used for dataset jobs
            network = metadata.get("type", None)

        microservices_network, microservices_action = get_microservices_network_and_action(network, action)

        try:
            json_schema = generate_schema(microservices_network, microservices_action)
        except Exception as e:
            logger.error("Exception thrown in get_spec_schema_for_job is %s", str(e))
            logger.error("Unable to fetch schema from tao_core")

        if not json_schema:
            DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            # Try regular format for CSV_PATH => "<network> - <action>.csv"
            CSV_PATH = os.path.join(DIR_PATH, "utils", "specs_utils", "specs", network, f"{network} - {action}.csv")
            if not os.path.exists(CSV_PATH):
                # Try secondary format for CSV_PATH => "<network> - <action>__<dataset-format>.csv"
                fmt = metadata.get("format", "_")
                CSV_PATH = os.path.join(
                    DIR_PATH,
                    "utils",
                    "specs_utils",
                    "specs",
                    network,
                    f"{network} - {action}__{fmt}.csv"
                )
                if not os.path.exists(CSV_PATH):
                    return Code(404, {}, "Default specs do not exist for action")
            json_schema = csv_to_json_schema.convert(CSV_PATH)

        json_schema["default"] = job_specs
        if "popular" in json_schema and job_specs:
            json_schema["popular"] = override_dicts(json_schema["popular"], job_specs)
        if is_request_automl(handler_id, action, kind):
            json_schema["automl_default_parameters"] = (
                metadata.get("automl_settings", {}).get("automl_hyperparameters", "[]")
            )
        return Code(200, json_schema, "Schema retrieved")

    @staticmethod
    def get_base_experiment_spec_schema(experiment_id, action):
        """Retrieves the base experiment specification schema.

        This method fetches the JSON schema for a base experiment spec based on the
        experiment ID and action. It checks the availability of the schema either
        through microservices or locally stored spec files.

        Args:
            experiment_id (str): UUID corresponding to the base experiment.
            action (str): A valid action for a dataset.

        Returns:
            Code: A response code object containing the status and the JSON schema
                  (if found) or error details:
                  - 200: JSON schema in a schema format.
                  - 404: If experiment/action not found or user cannot access.
        """
        base_experiment_spec = {}
        base_experiment_metadata = get_base_experiment_metadata(experiment_id)
        base_experiment_network = base_experiment_metadata.get("network_arch", "")
        if action not in base_experiment_metadata.get("actions", []):
            return Code(404, {}, "Action not found")

        base_exp_meta = base_experiment_metadata.get("base_experiment_metadata", {})
        if base_experiment_metadata and base_exp_meta.get("spec_file_present"):
            base_experiment_spec = base_exp_meta.get("specs", {})
            if not base_experiment_spec:
                return Code(404, {}, "Base specs not present.")

        # Map network name through actions_mapping (e.g. visual_changenet_segment -> visual_changenet)
        mapped_network, mapped_action = get_microservices_network_and_action(base_experiment_network, action)

        # Read csv from utils/spec_utils/specs/<network_name>/action.csv
        # Convert to json schema
        json_schema = {}
        try:
            json_schema = generate_schema(mapped_network, mapped_action)
        except Exception as e:
            logger.error("Exception thrown in get_base_experiment_spec_schema is %s", str(e))
            logger.error("Unable to fetch schema from tao_core")

        if not json_schema:
            DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

            # Try regular format for CSV_PATH => "<network> - <action>.csv"
            CSV_PATH = os.path.join(DIR_PATH, "utils", "specs_utils", "specs", base_experiment_network,
                                    f"{base_experiment_network} - {action}.csv")
            if not os.path.exists(CSV_PATH):
                return Code(404, {}, "Default specs do not exist for action")
            json_schema = csv_to_json_schema.convert(CSV_PATH)
        if "default" in json_schema and base_experiment_spec:
            # Merge the base experiment spec with the default schema
            merged_default = merge_nested_dicts(json_schema["default"], base_experiment_spec)
            # Validate and clean the merged spec to remove any invalid keys from corrupt base_experiment_spec
            logger.info("Validating merged base_experiment_spec for base_experiment_id: %s, action: %s",
                        experiment_id, action)
            json_schema["default"] = validate_and_clean_merged_spec(json_schema, merged_default)
            if (base_experiment_network == "visual_changenet_segment" and
                    "train" in json_schema["default"]):
                json_schema["default"]["train"].pop("tensorboard", None)
        if "popular" in json_schema and base_experiment_spec:
            # Merge the base experiment spec with the popular schema
            merged_popular = override_dicts(json_schema["popular"], base_experiment_spec)
            # Validate and clean the merged spec to remove any invalid keys from corrupt base_experiment_spec
            logger.info("Validating merged base_experiment_spec (popular) for base_experiment_id: %s, action: %s",
                        experiment_id, action)
            json_schema["popular"] = validate_and_clean_merged_spec(json_schema, merged_popular)
            if (base_experiment_network == "visual_changenet_segment" and
                    "train" in json_schema["popular"]):
                json_schema["popular"]["train"].pop("tensorboard", None)
        return Code(200, json_schema, "Schema retrieved")

    @staticmethod
    def get_spec_schema_without_handler_id(org_name, network, dataset_format, action, train_datasets):
        """Retrieves the specification schema without handler ID.

        This method generates a JSON schema for a given network and action, either
        through direct retrieval or by reading from CSV files stored locally.

        Args:
            org_name (str): The organization name.
            network (str): A valid network architecture name supported.
            dataset_format (str): A valid format of the architecture, if necessary.
            action (str): A valid action for a dataset.
            train_datasets (list): A list of UUIDs corresponding to training datasets.

        Returns:
            Code: A response code object containing the status and the JSON schema
                  (if found) or error details:
                  - 200: JSON schema in a schema format.
                  - 404: If experiment/dataset not found or user cannot access.
        """
        # Action not available
        if not network:
            return Code(404, {}, "Pass network name to the request")
        if not action:
            return Code(404, {}, "Pass action name to the request")

        # Read csv from utils/spec_utils/specs/<network_name>/action.csv
        # Convert to json schema
        json_schema = {}
        microservices_network, microservices_action = get_microservices_network_and_action(network, action)
        try:
            json_schema = generate_schema(microservices_network, microservices_action)
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("Exception thrown in get_spec_schema_without_handler_id is %s", str(e))
            logger.error("Unable to fetch schema from tao_core")

        if not json_schema:
            DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

            # Try regular format for CSV_PATH => "<network> - <action>.csv"
            CSV_PATH = os.path.join(DIR_PATH, "utils", "specs_utils", "specs", network, f"{network} - {action}.csv")
            if not os.path.exists(CSV_PATH):
                # Try secondary format for CSV_PATH => "<network> - <action>__<dataset-format>.csv"
                CSV_PATH = os.path.join(
                    DIR_PATH,
                    "utils",
                    "specs_utils",
                    "specs",
                    network,
                    f"{network} - {action}__{dataset_format}.csv"
                )
                if not os.path.exists(CSV_PATH):
                    return Code(404, {}, "Default specs do not exist for action")

            json_schema = csv_to_json_schema.convert(CSV_PATH)
        return Code(200, json_schema, "Schema retrieved")

    @staticmethod
    def get_gpu_types(user_id, org_name, workspace_id=None):
        """Retrieves available GPU types for the given user and organization.

        This method checks the backend for available GPU resources based on the
        provided user and organization information.

        Args:
            user_id (str): The user's UUID.
            org_name (str): The organization's name.

        Returns:
            Code: A response code object containing the status and available GPU
                  types or error details:
                  - 200: A list of available GPUs.
                  - 404: If GPUs cannot be retrieved for the specified backend.
        """
        workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
        try:
            handler = ExecutionHandler.create_handler(
                workspace_metadata=workspace_metadata,
                backend=BACKEND,
            )
            if handler:
                available_instances = handler.get_available_instances()
                if available_instances:
                    return Code(200, available_instances, "Retrieved available GPU info")
            else:
                logger.error(f"Unable to determine appropriate handler for backend '{BACKEND}' and workspace_metadata")
        except Exception as e:
            logger.error("Exception thrown in get_gpu_types is %s", str(e))
            logger.error("Unable to get GPU types for the deployed backend")
            logger.error(traceback.format_exc())
        return Code(404, [], f"GPU types can't be retrieved for deployed Backend {BACKEND}")
