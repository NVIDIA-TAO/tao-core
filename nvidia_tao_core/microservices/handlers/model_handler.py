# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model publishing and management handler module"""
import logging
import traceback

from nvidia_tao_core.microservices.utils import ngc_utils
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    check_read_access,
    get_handler_job_metadata,
    resolve_metadata
)
from nvidia_tao_core.microservices.utils.handler_utils import (
    Code,
    resolve_checkpoint_root_and_search
)

# Configure logging
logger = logging.getLogger(__name__)


class ModelHandler:
    """Handles model publishing and removal operations."""

    @staticmethod
    def publish_model(org_name, team_name, experiment_id, job_id, display_name, description):
        """Publish a model with the specified details after validating the job status.

        Parameters:
        org_name (str): The name of the organization.
        team_name (str): The name of the team.
        experiment_id (str): UUID corresponding to the experiment.
        job_id (str): UUID corresponding to the job.
        display_name (str): Display name for the model.
        description (str): Description for the model.

        Returns:
        Code: A response code (200 if the model is successfully published, 404 or 403 for errors).
              - 200: Model successfully created and uploaded.
              - 404: If experiment, job, or relevant files are not found.
              - 403: If the user does not have permission to publish the model.
        """
        handler_metadata = resolve_metadata("experiment", experiment_id)
        if not handler_metadata:
            return Code(404, {"error_desc": "Experiment not found", "error_code": 404})

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, experiment_id, kind="experiments"):
            return Code(404, {"error_desc": "Experiment cant be read", "error_code": 404})

        job_metadata = get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, {"error_desc": "Job trying to retrieve not found", "error_code": 404})

        job_status = job_metadata.get("status", "Error")
        if job_status not in ("Success", "Done"):
            return Code(
                404,
                {"error_desc": f"Job is not in Success or Done state (current: {job_status})", "error_code": 404}
            )
        job_action = job_metadata.get("action", "")
        if job_action not in ("train", "distill", "quantize", "prune", "retrain", "export", "gen_trt_engine"):
            return Code(
                404,
                {"error_desc": "Publish model is available only for train, distill, quantize, prune, retrain, "
                 "export, gen_trt_engine actions", "error_code": 404}
            )

        try:
            source_files = []
            source_file = resolve_checkpoint_root_and_search(handler_metadata, job_id)
            source_files.append(source_file)
            if not source_files:
                return Code(404, {"error_desc": "Unable to find a model for the given job", "error_code": 404})

            # Create NGC model
            ngc_key = ngc_utils.get_user_key(user_id, org_name)
            if not ngc_key:
                return Code(403, {"error_desc": "User does not have access to publish model", "error_code": 403})

            code, message = ngc_utils.create_model(
                org_name, team_name, handler_metadata, source_files[0], ngc_key, display_name, description
            )
            if code not in [200, 201]:
                logger.error("Error while creating NGC model")
                return Code(code, {"error_desc": message, "error_code": code})

            # Upload model version
            response_code, response_message = ngc_utils.upload_model(
                org_name, team_name, handler_metadata, source_files, ngc_key, job_id, job_action
            )
            if "already exists" in response_message:
                response_message = (
                    "Version trying to upload already exists, use remove_published_model endpoint to reupload the model"
                )
            if response_code == 200:
                return Code(response_code, {"message": response_message})
            return Code(response_code, {"error_desc": response_message, "error_code": response_code})
        except Exception as e:
            logger.error("Exception thrown in publish_model is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(404, {"error_desc": f"Unable to publish model: {str(e)}", "error_code": 404})

    @staticmethod
    def remove_published_model(org_name, team_name, experiment_id, job_id):
        """Remove a previously published model.

        Parameters:
        org_name (str): The name of the organization.
        team_name (str): The name of the team.
        experiment_id (str): UUID corresponding to the experiment.
        job_id (str): UUID corresponding to the job.

        Returns:
        Code: A response code (200 if the model is successfully removed, 404 for errors).
              - 200: Successfully deleted the model.
              - 404: If experiment, job, or the published model is not found.
        """
        handler_metadata = resolve_metadata("experiment", experiment_id)
        if not handler_metadata:
            return Code(404, {"error_desc": "Experiment not found", "error_code": 404})

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, experiment_id, kind="experiments"):
            return Code(404, {"error_desc": "Experiment cant be read", "error_code": 404})

        job_metadata = get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, {"error_desc": "Job trying to retrieve not found", "error_code": 404})

        job_status = job_metadata.get("status", "Error")
        if job_status not in ("Success", "Done"):
            return Code(
                404,
                {"error_desc": f"Job is not in Success or Done state (current: {job_status})", "error_code": 404}
            )
        job_action = job_metadata.get("action", "")
        if job_action not in ("train", "distill", "quantize", "prune", "retrain", "export", "gen_trt_engine"):
            return Code(
                404,
                {"error_desc": "Delete published model is available only for train, distill, "
                 "quantize, prune, retrain, export, gen_trt_engine actions", "error_code": 404}
            )

        try:
            ngc_key = ngc_utils.get_user_key(user_id, org_name)
            if not ngc_key:
                return Code(
                    403,
                    {"error_desc": "User does not have access to remove published model", "error_code": 403}
                )

            response = ngc_utils.delete_model(
                org_name, team_name, handler_metadata, ngc_key, job_id, job_action
            )
            if response.ok:
                return Code(response.status_code, {"message": "Successfully deleted model"})
            return Code(
                response.status_code,
                {"error_desc": "Unable to delete published model", "error_code": response.status_code}
            )
        except Exception as e:
            logger.error("Exception thrown in remove_published_model is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(404, {"error_desc": f"Unable to delete published model: {str(e)}", "error_code": 404})
