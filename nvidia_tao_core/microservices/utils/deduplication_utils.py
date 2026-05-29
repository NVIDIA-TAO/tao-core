# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deduplication utilities for workspace, dataset, and job creation."""

import logging
import os

if os.getenv("BACKEND"):
    from nvidia_tao_core.microservices.utils.mongo_utils import MongoHandler
else:
    MongoHandler = None

# Configure logging
logger = logging.getLogger(__name__)


def normalize_cloud_details(cloud_details):
    """Normalize cloud details for comparison by removing order-dependent variations.

    Args:
        cloud_details (dict): Cloud-specific details to normalize.

    Returns:
        dict: Normalized cloud details dictionary.
    """
    if not cloud_details:
        return {}

    # Create a sorted dictionary for consistent comparison
    return {k: v for k, v in sorted(cloud_details.items()) if v is not None}


def find_duplicate_workspace(user_id, org_name, cloud_type, cloud_specific_details):
    """Find an existing workspace with identical cloud configuration.

    Args:
        user_id (str): UUID of the user.
        org_name (str): Organization name.
        cloud_type (str): Type of cloud storage (aws, azure, huggingface, etc.).
        cloud_specific_details (dict): Cloud-specific configuration details.

    Returns:
        str or None: ID of the duplicate workspace if found, None otherwise.
    """
    if not os.getenv("BACKEND"):
        return None

    try:
        mongo_workspaces = MongoHandler("tao", "workspaces")

        # Normalize cloud details for consistent comparison
        normalized_details = normalize_cloud_details(cloud_specific_details)

        # Query for workspaces matching user, org, and cloud type
        query = {
            "user_id": user_id,
            "org_name": org_name,
            "cloud_type": cloud_type
        }

        workspaces = mongo_workspaces.find(query)

        # Check each workspace for matching cloud_specific_details
        for workspace in workspaces:
            existing_details = workspace.get("cloud_specific_details", {})
            existing_normalized = normalize_cloud_details(existing_details)

            # Compare normalized cloud details
            if normalized_details == existing_normalized:
                logger.debug(
                    "Found duplicate workspace %s for user %s with matching cloud config",
                    workspace.get("id"), user_id
                )
                return workspace.get("id")

        return None

    except Exception as e:
        logger.error("Error finding duplicate workspace: %s", str(e))
        return None


def find_duplicate_dataset(user_id, org_name, dataset_params):
    """Find an existing dataset with identical parameters.

    Args:
        user_id (str): UUID of the user.
        org_name (str): Organization name.
        dataset_params (dict): Dataset creation parameters including:
            - type: Dataset type
            - format: Dataset format
            - cloud_file_path: Path in cloud storage
            - workspace: Workspace ID
            - use_for: Intended use (training, evaluation, etc.)
            - url: Source URL if applicable

    Returns:
        str or None: ID of the duplicate dataset if found, None otherwise.
    """
    if not os.getenv("BACKEND"):
        return None

    try:
        mongo_datasets = MongoHandler("tao", "datasets")

        # Build query for matching datasets
        query = {
            "user_id": user_id,
            "org_name": org_name,
            "type": dataset_params.get("type"),
            "format": dataset_params.get("format")
        }

        # Add optional fields to query if present
        if dataset_params.get("workspace"):
            query["workspace"] = dataset_params.get("workspace")

        if dataset_params.get("cloud_file_path"):
            query["cloud_file_path"] = dataset_params.get("cloud_file_path")

        if dataset_params.get("url"):
            query["url"] = dataset_params.get("url")

        if dataset_params.get("use_for"):
            query["use_for"] = dataset_params.get("use_for")

        datasets = mongo_datasets.find(query)

        # Return first matching dataset
        if datasets:
            dataset_id = datasets[0].get("id")
            logger.debug(
                "Found duplicate dataset %s for user %s with matching params",
                dataset_id, user_id
            )
            return dataset_id

        return None

    except Exception as e:
        logger.error("Error finding duplicate dataset: %s", str(e))
        return None


def normalize_specs(specs):
    """Normalize job specs for comparison.

    Args:
        specs (dict): Job specifications.

    Returns:
        dict: Normalized specs dictionary.
    """
    if not specs:
        return {}

    # Create a sorted dictionary for consistent comparison
    # Remove fields that shouldn't affect duplication (like output paths)
    exclude_keys = {"results_dir", "checkpoint_dir", "log_file"}
    return {k: v for k, v in sorted(specs.items()) if k not in exclude_keys and v is not None}


def normalize_automl_settings(automl_settings):
    """Normalize AutoML settings for comparison.

    Args:
        automl_settings (dict): AutoML configuration settings.

    Returns:
        dict: Normalized AutoML settings dictionary.
    """
    if not automl_settings:
        return {}

    # Create a sorted dictionary for consistent comparison
    # Remove fields that shouldn't affect duplication
    exclude_keys = {"run_id", "timestamp"}
    return {k: v for k, v in sorted(automl_settings.items()) if k not in exclude_keys and v is not None}


def find_duplicate_job(user_id, org_name, job_params):
    """Find an existing job with identical parameters across ALL user's experiments/datasets.

    Args:
        user_id (str): UUID of the user (derived from experiment/dataset).
        org_name (str): Organization name.
        job_params (dict): Job creation parameters including:
            - kind: 'experiment' or 'dataset'
            - handler_id: experiment_id or dataset_id (IGNORED - searches all)
            - action: Job action (train, evaluate, export, etc.)
            - specs: Job specifications
            - parent_job_id: Parent job ID if applicable
            - base_experiment_ids: List of base experiment IDs
            - network_arch: Network architecture
            - automl_settings: AutoML configuration (if applicable)

    Returns:
        str or None: ID of the duplicate job if found, None otherwise.
    """
    if not os.getenv("BACKEND"):
        return None

    try:
        kind = job_params.get("kind")
        network_arch = job_params.get("network_arch")

        # Get ALL experiments/datasets for this user with matching network_arch
        all_jobs = {}
        experiments_checked = []

        if kind == "experiment":
            mongo_experiments = MongoHandler("tao", "experiments")
            # Find all experiments for this user with matching network_arch
            experiments = mongo_experiments.find({
                "user_id": user_id,
                "network_arch": network_arch
            })

            if not experiments:
                logger.debug("No experiments found for user %s with network_arch %s", user_id, network_arch)
                return None

            logger.debug("Found %d experiments to check", len(experiments))

            # Collect jobs from all experiments
            for exp in experiments:
                exp_id = exp.get("id")
                experiments_checked.append(exp_id)
                exp_jobs = exp.get("jobs", {})
                for job_id, job_data in exp_jobs.items():
                    # Store job with its experiment context
                    all_jobs[job_id] = {
                        "job_data": job_data,
                        "experiment": exp,
                        "experiment_id": exp_id
                    }

            logger.debug("Found %d total jobs across %d experiments", len(all_jobs), len(experiments))

        elif kind == "dataset":
            mongo_datasets = MongoHandler("tao", "datasets")
            dataset = mongo_datasets.find_one({"id": job_params.get("handler_id")})
            if not dataset:
                return None

            jobs = dataset.get("jobs", {})
            for job_id, job_data in jobs.items():
                all_jobs[job_id] = {
                    "job_data": job_data,
                    "dataset": dataset
                }
        else:
            return None

        # Normalize specs and automl_settings for comparison
        normalized_specs = normalize_specs(job_params.get("specs", {}))
        normalized_automl = normalize_automl_settings(job_params.get("automl_settings", {}))
        action = job_params.get("action")
        parent_job_id = job_params.get("parent_job_id")

        # Get the jobs collection to fetch full job specs
        mongo_jobs = MongoHandler("tao", "jobs")

        # Check each job for matching parameters
        for job_id, job_info in all_jobs.items():
            job_data = job_info["job_data"]
            # Get experiment context for experiment jobs
            if kind == "experiment":
                experiment = job_info["experiment"]
            else:
                dataset = job_info.get("dataset")
            logger.debug("Checking job %s (status=%s, action=%s)",
                         job_id, job_data.get("status"), job_data.get("action"))

            # Match action
            if job_data.get("action") != action:
                logger.debug("Skipping job %s due to action mismatch", job_id)
                continue

            # Match parent_job_id if specified
            if parent_job_id and job_data.get("parent_id") != parent_job_id:
                continue

            # Fetch full job details from jobs collection (has specs)
            full_job = mongo_jobs.find_one({"id": job_id})
            if not full_job:
                logger.debug("Skipping job %s - not found in jobs collection", job_id)
                continue

            # Match specs
            existing_specs = normalize_specs(full_job.get("specs", {}))

            # Skip if existing job has no specs (not fully initialized yet)
            if not existing_specs and normalized_specs:
                logger.debug("Skipping job %s - existing job has no specs (not initialized)", job_id)
                continue

            if normalized_specs != existing_specs:
                logger.debug("Skipping job %s due to specs mismatch", job_id)
                # Check which keys have different values
                all_keys = set(normalized_specs.keys()) | set(existing_specs.keys())
                differing_keys = [k for k in all_keys if normalized_specs.get(k) != existing_specs.get(k)]
                logger.debug("Top-level keys with different values: %s", differing_keys)

                # Show detailed differences for dict-type keys (like 'train')
                for key in differing_keys[:2]:  # Check first 2 differing keys
                    new_val = normalized_specs.get(key)
                    existing_val = existing_specs.get(key)
                    if isinstance(new_val, dict) and isinstance(existing_val, dict):
                        # Both are dicts, find which sub-keys differ
                        sub_keys = set(new_val.keys()) | set(existing_val.keys())
                        diff_sub_keys = [sk for sk in sub_keys if new_val.get(sk) != existing_val.get(sk)]
                        logger.debug("  '%s' has %d differing sub-fields: %s", key, len(diff_sub_keys), diff_sub_keys)
                        # Show values for first 2 differing sub-keys
                        for sk in diff_sub_keys[:2]:
                            logger.debug(
                                "    %s.%s: NEW=%s vs EXISTING=%s",
                                key, sk, new_val.get(sk), existing_val.get(sk)
                            )
                continue

            # For experiment jobs, check additional parameters
            # Note: base_experiment_ids and automl_settings are experiment-level fields
            # stored in the experiment document, not in the jobs collection
            if kind == "experiment":
                # Get experiment-level fields
                experiment_base_ids = experiment.get("base_experiment_ids", [])
                experiment_automl = experiment.get("automl_settings", {})

                # Check base_experiment_ids if provided
                base_experiments = job_params.get("base_experiment_ids")
                if base_experiments:
                    if set(base_experiments) != set(experiment_base_ids):
                        logger.debug("Skipping job %s due to base_experiment_ids mismatch", job_id)
                        logger.debug("New base_experiment_ids: %s", base_experiments)
                        logger.debug("Existing base_experiment_ids: %s", experiment_base_ids)
                        continue

                # Check AutoML settings - CRITICAL for AutoML jobs
                if normalized_automl:
                    # If the new request has automl_settings, compare them
                    existing_automl = normalize_automl_settings(experiment_automl)
                    if normalized_automl != existing_automl:
                        logger.debug(
                            "Skipping job %s - AutoML settings differ",
                            job_id
                        )
                        logger.debug("New AutoML: %s", normalized_automl)
                        logger.debug("Existing AutoML: %s", existing_automl)
                        continue
                else:
                    # If new request has no automl_settings, ensure existing job also doesn't
                    existing_automl = normalize_automl_settings(experiment_automl)
                    if existing_automl:
                        logger.debug(
                            "Skipping job %s - has AutoML settings but new request doesn't",
                            job_id
                        )
                        continue

            logger.info(
                "Found duplicate job %s (kind=%s, action=%s) with matching params",
                job_id, kind, action
            )
            return job_id

        return None

    except Exception as e:
        logger.error("Error finding duplicate job: %s", str(e))
        return None


def find_duplicate_experiment(user_id, org_name, name, network_arch,
                              workspace_id, base_experiment_ids,
                              train_datasets, eval_dataset):
    """Find an existing experiment with identical parameters.

    Args:
        user_id (str): UUID of the user.
        org_name (str): Organization name.
        name (str): Experiment name.
        network_arch (str): Network architecture.
        workspace_id (str): Workspace ID.
        base_experiment_ids (list): List of base experiment IDs.
        train_datasets (list): List of training dataset IDs.
        eval_dataset (str): Evaluation dataset ID.

    Returns:
        str or None: ID of the duplicate experiment if found, None otherwise.
    """
    if not os.getenv("BACKEND"):
        return None

    try:
        logger.debug(
            "Starting duplicate experiment search for user %s, name=%s, network_arch=%s",
            user_id, name, network_arch
        )

        mongo_experiments = MongoHandler("tao", "experiments")

        # Query for experiments by user and network_arch
        experiments = mongo_experiments.find({
            "user_id": user_id,
            "network_arch": network_arch
        })

        if not experiments:
            logger.debug("No experiments found for user %s with network_arch %s", user_id, network_arch)
            return None

        logger.debug("Found %d experiments to check", len(experiments))

        # Normalize the new experiment parameters for comparison
        normalized_base_experiments = sorted(base_experiment_ids) if base_experiment_ids else []
        normalized_train_datasets = sorted(train_datasets) if train_datasets else []

        for exp in experiments:
            exp_id = exp.get("id")
            logger.debug("Checking experiment %s (name=%s)", exp_id, exp.get("name"))

            # Match workspace
            if exp.get("workspace") != workspace_id:
                logger.debug("Skipping experiment %s due to workspace mismatch", exp_id)
                continue

            # Match base_experiment_ids
            existing_base = sorted(exp.get("base_experiment_ids", []))
            if normalized_base_experiments != existing_base:
                logger.debug("Skipping experiment %s due to base_experiment_ids mismatch", exp_id)
                continue

            # Match train_datasets
            existing_train = sorted(exp.get("train_datasets", []))
            if normalized_train_datasets != existing_train:
                logger.debug("Skipping experiment %s due to train_datasets mismatch", exp_id)
                continue

            # Match eval_dataset
            if exp.get("eval_dataset") != eval_dataset:
                logger.debug("Skipping experiment %s due to eval_dataset mismatch", exp_id)
                continue

            logger.debug(
                "Found duplicate experiment %s for user %s with matching params",
                exp_id, user_id
            )
            return exp_id

        logger.debug("No duplicate experiment found")
        return None

    except Exception as e:
        logger.error("Error finding duplicate experiment: %s", str(e))
        return None


def create_indexes_for_deduplication():
    """Create MongoDB indexes to optimize deduplication queries.

    This function creates compound indexes on frequently queried fields
    to improve the performance of duplicate detection.
    """
    if not os.getenv("BACKEND"):
        return

    try:
        # Index for workspace deduplication
        mongo_workspaces = MongoHandler("tao", "workspaces")
        mongo_workspaces.collection.create_index([
            ("user_id", 1),
            ("org_name", 1),
            ("cloud_type", 1)
        ], name="workspace_dedup_idx")

        # Index for dataset deduplication
        mongo_datasets = MongoHandler("tao", "datasets")
        mongo_datasets.collection.create_index([
            ("user_id", 1),
            ("org_name", 1),
            ("type", 1),
            ("format", 1),
            ("workspace", 1)
        ], name="dataset_dedup_idx")

        logger.info("Successfully created deduplication indexes")

    except Exception as e:
        logger.warning("Could not create deduplication indexes: %s", str(e))


# Create indexes on module import
if os.getenv("BACKEND"):
    try:
        create_indexes_for_deduplication()
    except Exception as e:
        logger.debug("Deduplication indexes will be created when first used: %s", str(e))
