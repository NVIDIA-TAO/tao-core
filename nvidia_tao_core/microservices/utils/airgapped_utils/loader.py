# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Airgapped experiment loader for TAO microservices"""

import os
import logging
import yaml
import glob
import tempfile
import uuid
import shutil

from nvidia_tao_core.microservices.utils.mongo_utils import MongoHandler
from nvidia_tao_core.microservices.utils.core_utils import safe_load_file
from nvidia_tao_core.microservices.handlers.cloud_handlers.utils import initialize_cloud_storage

logger = logging.getLogger(__name__)


class AirgappedExperimentLoader:
    """Loader for air-gapped experiment metadata"""

    def __init__(self, cloud_config: dict = None):
        """Initialize the loader

        Args:
            cloud_config (dict, optional): Cloud storage configuration.
        """
        self.mongo_handler = MongoHandler("tao", "jobs")
        self.local_json_file = None

        if not cloud_config:
            raise ValueError("cloud_config is required when using cloud_storage")

        # Use fixed path "index.json" for cloud storage
        self.json_file_path = "ptm_metadatas.json"

        # Initialize cloud storage
        self.cloud_storage = initialize_cloud_storage(
            cloud_type=cloud_config.get("cloud_type", "seaweedfs"),
            bucket_name=cloud_config.get("bucket_name", "tao-storage"),
            region=cloud_config.get("region"),
            access_key=cloud_config.get("access_key"),
            secret_key=cloud_config.get("secret_key"),
            endpoint_url=cloud_config.get("endpoint_url")
        )
        logger.info("Initialized cloud storage for downloading JSON and experiment.yaml files")

        # Download JSON file from cloud storage
        self.local_json_file = self._download_json_file_from_cloud()
        if not self.local_json_file:
            raise FileNotFoundError(f"Failed to download JSON file from cloud storage: {self.json_file_path}")

    def _download_json_file_from_cloud(self):
        """Download JSON file from cloud storage under LOCAL_MODEL_REGISTRY folder

        Returns:
            str: Path to local temporary JSON file, or None if download failed
        """
        try:
            # Construct cloud path using LOCAL_MODEL_REGISTRY (e.g. /shared-storage/models or shared-storage/models)
            local_model_registry = (os.getenv('LOCAL_MODEL_REGISTRY') or "").strip().strip('/')
            if not local_model_registry:
                raise ValueError(
                    "LOCAL_MODEL_REGISTRY environment variable is not set. "
                    "Set it to the path where ptm_metadatas.json was uploaded (e.g. shared-storage/models)."
                )

            cloud_json_path = f"{local_model_registry}/{self.json_file_path}"
            logger.info(
                "Downloading JSON file from cloud storage: %s (LOCAL_MODEL_REGISTRY=%s)",
                cloud_json_path, os.getenv('LOCAL_MODEL_REGISTRY')
            )

            # Download directly so connection errors (e.g. workspace endpoint_url=localhost
            # from Docker) surface in the response
            temp_dir = tempfile.mkdtemp()
            temp_filename = f"airgapped_models_{uuid.uuid4().hex}.json"
            temp_path = os.path.join(temp_dir, temp_filename)
            try:
                self.cloud_storage.download_file(cloud_json_path, temp_path)
                logger.info("Successfully downloaded JSON file to: %s", temp_path)
                return temp_path
            except Exception as e:
                logger.exception("Download failed for %s", cloud_json_path)
                try:
                    shutil.rmtree(temp_dir)
                except OSError:
                    pass
                err_str = str(e).lower()
                if "connection" in err_str or "refused" in err_str or "resolve" in err_str or "timeout" in err_str:
                    raise RuntimeError(
                        f"Cannot reach Seaweed at workspace endpoint: {e}. "
                        "When TAO API runs in Docker, workspace endpoint_url must be "
                        "http://seaweedfs-s3:8333 (not localhost). "
                        "Re-run the 'Create Cloud Workspace' cell so the workspace has the "
                        "correct endpoint, or use a new workspace."
                    ) from e
                raise FileNotFoundError(
                    f"JSON file not found at {cloud_json_path}: {e}. "
                    "Ensure the 'copy to Seaweed' step ran and TAO API has "
                    "LOCAL_MODEL_REGISTRY=shared-storage/models."
                ) from e

        except (ValueError, RuntimeError, FileNotFoundError):
            raise
        except Exception as e:
            logger.error("Error setting up JSON file download from cloud storage: %s", e)
            return None

    def load_experiments_from_json(self):
        """Load experiments from JSON file"""
        json_file_to_use = self.local_json_file

        if not os.path.isfile(json_file_to_use):
            raise FileNotFoundError(f"JSON file not found: {json_file_to_use}")

        logger.info("Loading experiments from JSON file: %s", json_file_to_use)

        try:
            experiments = safe_load_file(json_file_to_use)
            return experiments
        except Exception as e:
            logger.error("Failed to load experiments from JSON: %s", e)
            raise

    def find_and_read_experiment_yaml(self, experiment):
        """Find and read experiment.yaml file for a given experiment

        Args:
            experiment (dict): Experiment metadata containing ngc_path

        Returns:
            dict or None: Parsed YAML content if file found and valid, None otherwise
        """
        ngc_path = experiment.get("ngc_path")
        if not ngc_path:
            logger.warning("No ngc_path found for experiment: %s", experiment.get("name", "unknown"))
            return None

        # Split ngc_path: "nvstaging/tao/ocrnet:trainable_v2.0" -> "nvstaging/tao/ocrnet", "trainable_v2.0"
        if ":" not in ngc_path:
            logger.warning("Invalid ngc_path format (missing version): %s", ngc_path)
            return None

        path_part, version = ngc_path.split(":", 1)

        return self._download_experiment_yaml_from_cloud(experiment, path_part, version)

    def _download_experiment_yaml_from_cloud(self, experiment, path_part, version):
        """Download experiment.yaml from cloud storage"""
        try:
            local_model_registry = (os.getenv('LOCAL_MODEL_REGISTRY') or "").strip().strip('/')
            cloud_dir = f"{local_model_registry}/{path_part}/{version}/"
            logger.debug("Searching for model directories in cloud path: %s", cloud_dir)

            # List directories to find the model subdirectory
            files, _ = self.cloud_storage.list_files_in_folder(cloud_dir)
            logger.debug("Found files: %s", files)

            # Look for experiment.yaml in each subdirectory
            for file in files:
                if not file.endswith("experiment.yaml"):
                    continue

                if self.cloud_storage.is_file(file):
                    # Create temporary directory and filename for download
                    temp_dir = tempfile.mkdtemp()
                    temp_filename = f"experiment_{uuid.uuid4().hex}.yaml"
                    temp_path = os.path.join(temp_dir, temp_filename)

                    try:
                        self.cloud_storage.download_file(file, temp_path)
                        logger.debug("Downloaded experiment.yaml to temporary file: %s", temp_path)

                        # Read and parse YAML content
                        with open(temp_path, 'r', encoding='utf-8') as f:
                            yaml_content = yaml.safe_load(f)
                            logger.debug("Successfully loaded YAML content from cloud storage")
                            return yaml_content

                    except Exception as e:
                        logger.error("Failed to download/parse YAML file %s: %s", file, e)
                        return None

                    finally:
                        # Clean up temporary directory
                        try:
                            shutil.rmtree(temp_dir)
                        except Exception as e:
                            logger.error("Failed to clean up temporary directory %s: %s", temp_dir, e)
                            pass

            logger.warning("No experiment.yaml file found in cloud storage for experiment: %s (searched in %s)",
                           experiment.get("name", "unknown"), cloud_dir)
            return None

        except Exception as e:
            logger.error("Error accessing cloud storage for experiment %s: %s",
                         experiment.get("name", "unknown"), e)
            return None

    def _read_experiment_yaml_from_local(self, experiment, path_part, version):
        """Read experiment.yaml from local filesystem"""
        # Construct search pattern: base_dir/ngc_path_part/version/*/experiment.yaml
        search_pattern = os.path.join(self.base_dir, path_part, version, "*", "experiment.yaml")
        logger.debug("Searching for spec file with pattern: %s", search_pattern)

        matches = glob.glob(search_pattern)
        if matches:
            yaml_file = matches[0]  # Use first match
            logger.info("Found experiment.yaml for %s: %s", experiment.get("name", "unknown"), yaml_file)

            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    yaml_content = yaml.safe_load(f)
                    logger.debug("Successfully loaded YAML content from %s", yaml_file)
                    return yaml_content
            except Exception as e:
                logger.error("Failed to read YAML file %s: %s", yaml_file, e)
                return None

        logger.warning("No experiment.yaml file found for experiment: %s (pattern: %s)",
                       experiment.get("name", "unknown"), search_pattern)
        return None

    def validate_experiment_data(self, experiments):
        """Validate experiment data before database import"""
        logger.info("Validating experiment data...")

        required_fields = [
            "id", "name", "network_arch", "ngc_path", "actions",
            "accepted_dataset_intents", "dataset_type"
        ]

        valid_experiments = {}
        invalid_count = 0

        for exp_id, experiment in experiments.items():
            try:
                # Check required fields
                missing_fields = [field for field in required_fields if field not in experiment]
                if missing_fields:
                    logger.warning("Experiment %s missing required fields: %s", exp_id, missing_fields)
                    invalid_count += 1
                    continue

                # Validate experiment ID matches
                if experiment["id"] != exp_id:
                    logger.warning("Experiment ID mismatch: key=%s, id=%s", exp_id, experiment["id"])
                    invalid_count += 1
                    continue

                # Ensure base_experiment_metadata exists
                if "base_experiment_metadata" not in experiment:
                    logger.warning("Experiment %s missing base_experiment_metadata", exp_id)
                    experiment["base_experiment_metadata"] = {}

                # Handle spec file reading if spec_file_present is true
                base_meta = experiment["base_experiment_metadata"]
                if base_meta.get("spec_file_present"):
                    logger.info("Reading experiment.yaml for experiment: %s", exp_id)
                    yaml_content = self.find_and_read_experiment_yaml(experiment)
                    if yaml_content:
                        base_meta["specs"] = yaml_content
                        logger.info("Successfully loaded specs for experiment: %s", exp_id)
                    else:
                        logger.warning(
                            "Failed to load specs for experiment %s despite spec_file_present=true, "
                            "setting spec_file_present=false", exp_id
                        )
                        base_meta["spec_file_present"] = False
                        base_meta["specs"] = {}
                else:
                    # Ensure specs field exists even when spec_file_present is false
                    if "specs" not in base_meta:
                        base_meta["specs"] = {}

                valid_experiments[exp_id] = experiment

            except Exception as e:
                logger.error("Error validating experiment %s: %s", exp_id, e)
                invalid_count += 1
                continue

        logger.info("Validation complete: %d valid, %d invalid experiments",
                    len(valid_experiments), invalid_count)
        return valid_experiments

    def import_to_database(self, experiments):
        """Import experiments to database"""
        logger.info("Importing %d experiments to database...", len(experiments))

        success_count = 0
        error_count = 0

        for exp_id, experiment in experiments.items():
            try:
                experiment['public'] = True
                self.mongo_handler.upsert({'id': exp_id}, experiment)
                success_count += 1
                logger.debug("Successfully imported experiment: %s", exp_id)
            except Exception as e:
                logger.error("Failed to import experiment %s: %s", exp_id, e)
                error_count += 1

        logger.info("Import complete: %d successful, %d failed", success_count, error_count)

    def load_and_import(self):
        """Load experiments from JSON and import to database"""
        try:
            # Load experiments from JSON
            experiments = self.load_experiments_from_json()

            # Validate data
            valid_experiments = self.validate_experiment_data(experiments)

            if not valid_experiments:
                logger.error("No valid experiments found to import")
                return False

            # Import to database
            self.import_to_database(valid_experiments)

            logger.info("Air-gapped experiment import completed successfully")
            return True

        except Exception as e:
            logger.error("Failed to load and import experiments: %s", e)
            return False
        finally:
            # Clean up temporary files
            self.cleanup()

    def cleanup(self):
        """Clean up temporary files"""
        if self.local_json_file and os.path.exists(self.local_json_file):
            try:
                os.unlink(self.local_json_file)
                logger.debug("Cleaned up temporary JSON file: %s", self.local_json_file)
            except Exception as e:
                logger.error("Failed to clean up temporary JSON file %s: %s", self.local_json_file, e)
