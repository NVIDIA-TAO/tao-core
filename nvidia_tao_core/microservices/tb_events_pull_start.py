#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start Tensorboard Events pulling from cloud storage"""
import argparse
import json
from time import sleep
import os
import logging

from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_handler_metadata_with_jobs
from nvidia_tao_core.microservices.utils.handler_utils import filter_file_objects
from nvidia_tao_core.microservices.utils.cloud_utils import create_cs_instance_with_decrypted_metadata

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


def pull_tf_events(results_dir, action, cs_instance, experiment_id):
    """Function to pull .tfevents from cloud to local path"""
    tf_events_path = results_dir + "/" + action
    tf_events_path = tf_events_path.lstrip('/')
    if cs_instance.is_folder(tf_events_path):
        _, objects = cs_instance.list_files_in_folder(tf_events_path)
        tf_events_objects = filter_file_objects(objects, regex_pattern=r'.*\.tfevents.+$')
        if len(tf_events_objects) == 0:
            logger.info("No tfevents files present in %s", tf_events_path)
        for obj in tf_events_objects:
            file = obj.name
            basename = os.path.basename(file)
            destination = f'/tfevents/{action}/{basename}'
            if not os.path.exists(destination):
                cs_instance.download_file(file, destination)
                logger.info("Downloaded tfevents file to %s", destination)
            elif cs_instance.is_file_modified(file, destination):
                logger.info("File has been modified, downloading file now")
                cs_instance.download_file(file, destination)
                logger.info("Downloaded tfevents file to %s", destination)
    else:
        logger.warning(
            "Path %s does not exist in cloud storage for experiment %s",
            tf_events_path,
            experiment_id
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Tensorboard Events Controller',
        description='Periodically pull tfevents files from cloud storage'
    )
    parser.add_argument(
        '--experiment_id',
        type=str,
    )
    parser.add_argument(
        '--org_name',
        type=str,
    )
    parser.add_argument(
        '--decrypted_workspace_metadata',
        type=json.loads,
    )
    args = parser.parse_args()
    experiment_id = args.experiment_id
    org_name = args.org_name
    decrypted_workspace_metadata = args.decrypted_workspace_metadata
    cs_instance, _ = create_cs_instance_with_decrypted_metadata(decrypted_workspace_metadata)

    if not cs_instance:
        logger.error(
            "Unable to create cloud storage instance for Tensorboard Events Pull for experiment %s",
            experiment_id
        )
    else:
        logger.info("Starting Tensorboard Events Pull for experiment %s", experiment_id)
    while cs_instance is not None:
        sleep(30)
        handler_metadata = get_handler_metadata_with_jobs(experiment_id, "experiment")
        automl_enabled = handler_metadata.get("automl_settings", {}).get("automl_enabled", False)
        # Use TAO_API_RESULTS_DIR for SLURM compatibility, fallback to /results
        results_root = os.getenv('TAO_API_RESULTS_DIR', '/results')
        jobs = handler_metadata.get("jobs", [])
        if len(jobs) == 0:
            logger.info("No jobs found for experiment %s", experiment_id)
            continue
        for job in jobs:
            action = job.get('action', None)
            job_id = job.get('id', None)
            if action and job_id:
                if automl_enabled:
                    job_details = job.get('job_details', {})
                    if not job_details:
                        logger.info("No jobs found for AutoML experiment %s", experiment_id)
                        continue
                    for automl_job_id in job_details:
                        results_dir = os.path.join(results_root, automl_job_id)
                        pull_tf_events(results_dir, action, cs_instance, experiment_id)
                else:
                    results_dir = os.path.join(results_root, job_id)
                    pull_tf_events(results_dir, action, cs_instance, experiment_id)
