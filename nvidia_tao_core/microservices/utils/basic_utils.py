# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utility functions for app handlers"""
import os
import logging

from . import stateless_handler_utils
from .stateless_handler_utils import (
    resolve_metadata,
    get_user
)

if os.getenv("BACKEND"):  # To see if the container is going to be used for Service pods or network jobs
    from .mongo_utils import MongoHandler

# Configure logging
logger = logging.getLogger(__name__)


def resolve_job_existence(job_id):
    """Return whether job exists or not"""
    mongo_jobs = MongoHandler("tao", "jobs")
    job = mongo_jobs.find_one({'id': job_id})
    if job:
        return True
    return False


def delete_jobs_for_handler(handler_id, kind):
    """Deletes job metadatas associated with handler_id"""
    mongo_jobs = MongoHandler("tao", "jobs")
    mongo_jobs.delete_many({f"{kind}_id": handler_id})


def resolve_metadata_with_jobs(user_id, org_name, kind, handler_id):
    """Reads job_id.json in jobs_metadata folder and return it's contents"""
    if not user_id:
        logger.error("Can't resolve job metadata without user information")
        return {}
    handler_id = "*" if handler_id in ("*", "all") else handler_id
    metadata = {} if handler_id == "*" else resolve_metadata(kind, handler_id)
    if metadata or handler_id == "*":
        metadata["jobs"] = []
        jobs = stateless_handler_utils.get_jobs_for_handler(handler_id, kind)
        for job_meta in jobs:
            job_meta.pop('num_gpu', None)
            metadata["jobs"].append(job_meta)
        return metadata
    return {}


def get_org_experiments(org_name):
    """Returns a list of experiment IDs that are available for the given org_name"""
    mongo_experiments = MongoHandler("tao", "experiments")
    org_experiments = mongo_experiments.find({'org_name': org_name})
    experiments = []
    for experiment in org_experiments:
        experiment_id = experiment.get('id')
        experiments.append(experiment_id)
    return experiments


def get_org_datasets(org_name):
    """Returns a list of dataset IDs that are available for the given org_name"""
    mongo_datasets = MongoHandler("tao", "datasets")
    org_datasets = mongo_datasets.find({'org_name': org_name})
    datasets = []
    for dataset in org_datasets:
        dataset_id = dataset.get('id')
        datasets.append(dataset_id)
    return datasets


def get_org_workspaces(org_name):
    """Returns a list of workspace IDs that are available in given org_name"""
    mongo_workspaces = MongoHandler("tao", "workspaces")
    org_workspaces = mongo_workspaces.find({'org_name': org_name})
    workspaces = []
    for workspace in org_workspaces:
        workspace_id = workspace.get('id')
        workspaces.append(workspace_id)
    return workspaces


def get_user_experiments(user_id, mongo_users=None):
    """Returns a list of experiments that are available for the user"""
    user = get_user(user_id, mongo_users)
    experiments = user.get("experiments", [])
    return experiments


def get_user_datasets(user_id, mongo_users=None):
    """Returns a list of datasets that are available for the user"""
    user = get_user(user_id, mongo_users)
    datasets = user.get("datasets", [])
    return datasets


def get_user_workspaces(user_id, mongo_users=None):
    """Returns a list of datasets that are available for the user in given org_name"""
    user = get_user(user_id, mongo_users)
    workspaces = user.get("workspaces", [])
    return workspaces


def get_job(job_id):
    """Returns job from DB (converted to dict for schema compatibility)"""
    mongo_jobs = MongoHandler("tao", "jobs")
    job_query = {'id': job_id}
    job = mongo_jobs.find_one(job_query)
    return job


def get_experiment(experiment_id):
    """Returns experiment from DB (converted to dict for schema compatibility)"""
    mongo_experiments = MongoHandler("tao", "experiments")
    experiment_query = {'id': experiment_id}
    experiment = mongo_experiments.find_one(experiment_query)
    return experiment


def get_dataset(dataset_id):
    """Returns dataset from DB (converted to dict for schema compatibility)"""
    mongo_datasets = MongoHandler("tao", "datasets")
    dataset_query = {'id': dataset_id}
    dataset = mongo_datasets.find_one(dataset_query)
    return dataset


def get_workspace(workspace_id):
    """Returns workspace from DB"""
    mongo_workspaces = MongoHandler("tao", "workspaces")
    workspace_query = {'id': workspace_id}
    workspace = mongo_workspaces.find_one(workspace_query)
    return workspace


def create_blob_dataset(org_name, kind, handler_id):
    """Creates a blob dataset"""
    # Make a placeholder for S3 blob dataset
    msg = "Doesn't support the blob dataset for now."
    from .handler_utils import Code
    return Code(400, {}, msg)


def get_dataset_actions(ds_type, ds_format):
    """Reads the dataset's network config and returns the valid actions of the given dataset type and format"""
    from .core_utils import read_network_config
    actions_default = read_network_config(ds_type)["api_params"]["actions"]

    # Define all anamolous formats where actions are not same as ones listed in the network config
    TYPE_FORMAT_ACTIONS_MAP = {("object_detection", "raw"): [],
                               ("object_detection", "coco_raw"): [],
                               ("segmentation", "raw"): [],
                               }

    actions_override = TYPE_FORMAT_ACTIONS_MAP.get((ds_type, ds_format), actions_default)
    return actions_override


def nested_update(source, additions, allow_overwrite=True):
    """Merge one dictionary(additions) into another(source)"""
    if not isinstance(additions, dict):
        return source
    for key, value in additions.items():
        if isinstance(value, dict):
            # Initialize key in source if not present
            if key not in source:
                source[key] = {}
            source[key] = nested_update(source[key], value, allow_overwrite=allow_overwrite)
        else:
            source[key] = value if allow_overwrite else source.get(key, value)
    return source


def get_job_logs(log_file_path):
    """Yield lines from job log file"""
    with open(log_file_path, 'r', encoding="utf-8") as log_file:
        while True:
            log_line = log_file.readline()
            if not log_line:
                break
            yield log_line
