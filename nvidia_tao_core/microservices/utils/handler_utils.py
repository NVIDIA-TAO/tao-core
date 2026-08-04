# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper classes, functions

Classes:
- Code
- JobContext
- StatusParser

Functions:
- search_for_base_experiment
- search_for_dataset
- get_dataset_download_command
- get_model_results_path
- write_nested_dict
- build_cli_command
- get_num_nodes_from_spec

"""
import os
import re
import copy
import glob
import json
import math
import uuid
import requests
import tempfile
import traceback
import subprocess
from datetime import datetime, timezone, timedelta
import logging

from nvidia_tao_core.microservices.constants import (
    _ITER_MODELS,
    CONTINUOUS_STATUS_KEYS,
    _PYT_TAO_NETWORKS,
    STATUS_CALLBACK_MISMATCH_WITH_CHECKPOINT_EPOCH,
    MISSING_EPOCH_FORMAT_NETWORKS,
)
from .network_utils.network_constants import gpu_mapper, node_mapper
from .stateless_handler_utils import (
    get_handler_job_metadata,
    get_job_specs,
    get_automl_brain_info,
    get_automl_best_rec_info,
    get_automl_controller_info,
    get_dnn_status,
    write_job_metadata,
    resolve_metadata,
    write_handler_metadata,
    update_base_experiment_metadata,
    experiment_update_handler_attributes,
    update_handler_with_jobs_info,
    get_workspace_string_identifier,
    get_automl_experiment_job_id
)
from .ngc_utils import validate_ptm_download
from .core_utils import create_folder_with_permissions, get_monitoring_metric, normalize_metric_config

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


# Helper Classes


class TAOResponse:
    """Helper class for API response"""

    def __init__(self, code, data, message=""):
        """Initialize TAOResponse helper class"""
        self.code = code
        self.data = data
        self.message = message
        self.attachment_key = None


def Code(code, data={}, msg="", use_data_as_response=False):
    """Wraps TAOResponse and returns appropriate responses

    Args:
        code (int): HTTP status code
        data (dict): Response data
        msg (str): Error message or informational message
        use_data_as_response (bool): If True, use data directly as response (like 200 responses)
                                   If False, format as standard error response
    """
    if code == 200:
        return TAOResponse(code, data, msg)

    if code in [400, 404, 409]:
        if use_data_as_response:
            # Use data directly as response, just like 200 responses
            return TAOResponse(code, data, msg)
        # Standard error formatting
        error_data = {"error_desc": msg, "error_code": code}
        return TAOResponse(code, error_data, msg)

    error_data = {"error_desc": msg, "error_code": code}
    return TAOResponse(404, error_data, msg)


class JobContext:
    """Class for holding job related information"""

    # Initialize Job Related fields
    # Contains API related parameters
    # ActionPipeline interacts with Toolkit and uses this JobContext
    def __init__(
        self,
        job_id,
        parent_id,
        network,
        action,
        handler_id,
        user_id,
        org_name,
        kind,
        created_on=None,
        specs=None,
        name=None,
        description=None,
        num_gpu=-1,
        backend_details=None,
        retain_checkpoints_for_resume=False,
        early_stop_epoch=None,
        timeout_minutes=None
    ):
        """Initialize JobContext class"""
        # Non-state variables
        self.id = job_id
        self.parent_id = parent_id
        self.network = network
        self.action = action
        self.handler_id = handler_id
        self.user_id = user_id
        self.org_name = org_name
        self.kind = kind
        self.created_on = created_on
        if not self.created_on:
            self.created_on = datetime.now(tz=timezone.utc)

        # State variables
        self.last_modified = datetime.now(tz=timezone.utc)
        self.status = "Pending"  # Starts off like this
        self.job_details = {}
        self.specs = specs
        # validate and update num_gpu
        self.name = name
        self.description = description
        self.num_gpu = num_gpu
        self.backend_details = backend_details
        # Extract platform_id from backend_details for backward compatibility
        self.platform_id = None
        if backend_details and backend_details.get('backend_type') == "lepton":
            self.platform_id = backend_details.get('platform_id')
        self.retain_checkpoints_for_resume = retain_checkpoints_for_resume
        self.early_stop_epoch = early_stop_epoch
        self.timeout_minutes = timeout_minutes
        self.write()

    def write(self):
        """Write the schema dict to jobs_metadata/job_id.json file"""
        # Create a job metadata
        write_job_metadata(self.id, self.schema())
        update_handler_with_jobs_info(self.schema(), self.handler_id, self.id, self.kind + "s")

    def __repr__(self):
        """Returns the schema dict"""
        return self.schema().__repr__()

    # ModelHandler / DatasetHandler interacts with this function
    def schema(self):
        """Creates schema dict based on the member variables"""
        _schema = {  # Cannot modify
            "name": self.name,
            "description": self.description,
            "id": self.id,
            "user_id": self.user_id,
            "org_name": self.org_name,
            "parent_id": self.parent_id,
            "backend_details": self.backend_details,
            "network_arch": self.network,
            "action": self.action,
            "created_on": self.created_on,
            "specs": self.specs,
            f"{self.kind}_id": self.handler_id,
            # Can modify
            "last_modified": self.last_modified,
            "status": self.status,
            "job_details": self.job_details,
            "retain_checkpoints_for_resume": self.retain_checkpoints_for_resume,
            "early_stop_epoch": self.early_stop_epoch,
            "timeout_minutes": self.timeout_minutes,
        }
        return _schema


class StatusParser:
    """Class for parsing DNN status callback info"""

    def __init__(self, network, results_dir, first_epoch_number=-1):
        """Intialize StatusParser class"""
        self.network = network
        self.results_dir = results_dir
        self.cur_line = 0
        # Initialize results
        self.results = {}
        # Logging fields
        self.results["date"] = ""
        self.results["time"] = ""
        self.results["status"] = ""
        self.results["message"] = ""
        # Categorical
        self.results["categorical"] = {}
        # KPI
        self.results["kpi"] = {}
        # Graphical
        self.results["graphical"] = {}
        # Key metric for continual learning
        self.results["key_metric"] = 0.0

        self.last_seen_epoch = 0
        self.best_epoch_number = 0
        self.latest_epoch_number = 0
        self.first_epoch_number = first_epoch_number
        #
        self.gr_dict_cache = []

    def _update_first_epoch_number(self, epoch_number):
        if self.first_epoch_number == -1:
            self.first_epoch_number = epoch_number

    def _update_categorical(self, status_dict):
        """Update categorical key of status line"""
        if "epoch" in status_dict:
            self.last_seen_epoch = status_dict["epoch"]
        if "cur_iter" in status_dict and self.network in _ITER_MODELS:
            self.last_seen_epoch = status_dict["cur_iter"]

        # Categorical
        if "categorical" in status_dict:
            cat_dict = status_dict["categorical"]
            if type(cat_dict) is not dict:
                return
            for _, value_dict in cat_dict.items():
                if type(value_dict) is not dict:
                    return
            self.results["categorical"].update(cat_dict)

    def _process_object_count_kpi(self, kpi_dict):
        """Process object count KPI from DS Analyze action"""
        index_to_object = kpi_dict.get('index', {})
        count_num = kpi_dict.get('count_num', {})
        percent = kpi_dict.get('percent', {})
        self.results['kpi']['object_count_index'] = {"values": {}}
        self.results['kpi']['object_count_num'] = {"values": {}}
        self.results['kpi']['object_count_percent'] = {"values": {}}
        for key, value in index_to_object.items():
            self.results['kpi']['object_count_index']['values'][str(key)] = value
        for key, value in count_num.items():
            self.results['kpi']['object_count_num']['values'][str(key)] = value
        for key, value in percent.items():
            self.results['kpi']['object_count_percent']['values'][str(key)] = value

    def _process_bbox_area_kpi(self, kpi_dict):
        """Process Bounding Box Area KPI from DS Analyze action"""
        type_to_object = kpi_dict.get('type', {})
        mean = kpi_dict.get('mean', {})
        self.results['kpi']['bbox_area_type'] = {"values": {}}
        self.results['kpi']['bbox_area_mean'] = {"values": {}}
        for key, value in type_to_object.items():
            self.results['kpi']['bbox_area_type']['values'][str(key)] = value
        for key, value in mean.items():
            self.results['kpi']['bbox_area_mean']['values'][str(key)] = value

    def _update_kpi(self, status_dict):
        """Update kpi key of status line"""
        if "epoch" in status_dict:
            self.last_seen_epoch = status_dict["epoch"]
            if "kpi" in status_dict:
                self._update_first_epoch_number(status_dict["epoch"])
        if "cur_iter" in status_dict and self.network in _ITER_MODELS:
            self.last_seen_epoch = status_dict["cur_iter"]
            if "kpi" in status_dict:
                self._update_first_epoch_number(status_dict["cur_iter"])
        if "mode" in status_dict and status_dict["mode"] == "train":
            return

        if "kpi" in status_dict:
            kpi_dict = status_dict["kpi"]
            if type(kpi_dict) is not dict:
                return
            analyze_type = kpi_dict.get('analyze_type', '')
            if analyze_type == 'object_count':  # DS Analyze KPI
                self._process_object_count_kpi(kpi_dict)
            elif analyze_type == 'bbox_area':
                self._process_bbox_area_kpi(kpi_dict)
            else:
                for key, value in kpi_dict.items():
                    if type(value) is dict:
                        # Process it differently
                        float_value = StatusParser.force_float(value.get("value", None))
                    else:
                        float_value = StatusParser.force_float(value)
                    # Simple append to "values" if the list exists
                    if key in self.results["kpi"]:
                        if float_value is not None:
                            if self.last_seen_epoch not in self.results["kpi"][key]["values"].keys():
                                self.results["kpi"][key]["values"][str(self.last_seen_epoch)] = float_value
                    else:
                        if float_value is not None:
                            self.results["kpi"][key] = {"values": {str(self.last_seen_epoch): float_value}}

    def _update_graphical(self, status_dict):
        """Update graphical key of status line"""
        if "epoch" in status_dict:
            self.last_seen_epoch = status_dict["epoch"]
        if "cur_iter" in status_dict and self.network in _ITER_MODELS:
            self.last_seen_epoch = status_dict["cur_iter"]

        if "graphical" in status_dict:
            gr_dict = status_dict["graphical"]
            # If the exact same dict was seen before, skip (an artefact of how status logger is written)
            if gr_dict in self.gr_dict_cache:
                return
            self.gr_dict_cache.append(gr_dict)
            if type(gr_dict) is not dict:
                return
            for key, value in gr_dict.items():
                plot_helper_dict = {}
                if type(value) is dict:
                    # Process it differently
                    float_value = StatusParser.force_float(value.get("value", None))
                    # Store x_min, x_max, etc... if given
                    for plot_helper_key in ["x_min", "x_max", "y_min", "y_max", "units"]:
                        if value.get(plot_helper_key):
                            plot_helper_dict[plot_helper_key] = value.get(plot_helper_key)
                else:
                    float_value = StatusParser.force_float(value)
                # Simple append to "values" if the list exists
                if key in self.results["graphical"]:
                    if float_value is not None:
                        if self.last_seen_epoch not in self.results["graphical"][key]["values"].keys():
                            self.results["graphical"][key]["values"][str(self.last_seen_epoch)] = float_value
                else:
                    if float_value is not None:
                        self.results["graphical"][key] = {"values": {str(self.last_seen_epoch): float_value}}

                if key in self.results["graphical"]:
                    # Put together x_min, x_max, y_min, y_max
                    graph_key_vals = self.results["graphical"][key]["values"]
                    self.results["graphical"][key].update({
                        "x_min": 0,
                        "x_max": len(graph_key_vals),
                        "y_min": 0,
                        "y_max": StatusParser.force_max([
                            val for key, val in graph_key_vals.items()
                        ]),
                        "units": None
                    })
                    # If given in value, then update x_min, x_max, etc...
                    self.results["graphical"][key].update(plot_helper_dict)

    @staticmethod
    def force_float(value):
        """Convert str to float"""
        try:
            if isinstance(value, str):
                # Check for special float values first
                if value.lower() in {"nan", "infinity", "inf"}:
                    return None
                # Silently ignore non-numeric strings (like status strings)
                try:
                    return float(value)
                except ValueError:
                    return None
            elif isinstance(value, float):
                if math.isnan(value) or value in {float("inf"), float("-inf")}:
                    return None
            return float(value)
        except Exception:
            # Only log unexpected errors, not conversion failures
            return None

    @staticmethod
    def force_min(values):
        """Return min elements in the list"""
        values_no_none = [val for val in values if val is not None]
        if values_no_none != []:
            return min(values_no_none)
        return 0

    @staticmethod
    def force_max(values):
        """Return max elements in the list"""
        values_no_none = [val for val in values if val is not None]
        if values_no_none != []:
            return max(values_no_none)
        return 1e10

    def post_process_results(
        self,
        total_epochs=0,
        eta="",
        last_seen_epoch=0,
        automl=False,
        job_id="",
        processed_results=None,
        automl_brain=False
    ):
        """Post process the status from DNN callbacks to be compatible with defined schema's in app.py"""
        # Copy the results
        if processed_results is None:
            processed_results = {}
        if job_id not in processed_results:
            processed_results[job_id] = {}
        # Detailed results
        if "detailed_status" not in processed_results[job_id]:
            processed_results[job_id]["detailed_status"] = {}

        if automl_brain or (not automl):
            processed_results[job_id]["starting_epoch"] = int(self.first_epoch_number)
            processed_results[job_id]["max_epoch"] = int(total_epochs)
            processed_results[job_id]["epoch"] = int(self.last_seen_epoch)
        if automl_brain:
            if automl and eta != "":
                processed_results[job_id]["epoch"] = int(last_seen_epoch)
                if type(eta) is float:
                    eta = str(timedelta(seconds=eta))
                processed_results[job_id]["eta"] = str(eta)
                return processed_results

        for key in ["date", "time", "status", "message"]:
            if self.results[key]:
                processed_results[job_id]["detailed_status"][key] = self.results[key]
        # Categorical
        processed_results[job_id]["categorical"] = []
        for key, value_dict in self.results["categorical"].items():
            value_dict_unwrapped = [
                {"category": cat, "value": StatusParser.force_float(val)}
                for cat, val in value_dict.items()
            ]
            processed_results[job_id]["categorical"].append({
                "metric": key,
                "category_wise_values": value_dict_unwrapped
            })

        # KPI and Graphical
        for result_type in ("kpi", "graphical"):
            processed_results[job_id][result_type] = []
            for key, value_dict in self.results[result_type].items():
                dict_schema = {"metric": key}
                dict_schema.update(value_dict)
                processed_results[job_id][result_type].append(dict_schema)

        # Continuous remain the same
        for key in CONTINUOUS_STATUS_KEYS:
            dnn_value = self.results.get(key, None)
            api_value = processed_results.get(job_id, {}).get(key, None)
            if (not dnn_value) and api_value:
                dnn_value = api_value
            processed_results[job_id][key] = dnn_value

        common_keys = ["automl_stats", "automl_result"]
        for result_key in common_keys:
            if result_key in processed_results[job_id]:
                processed_results[result_key] = processed_results[job_id].pop(result_key)
        return processed_results

    def update_results(
        self,
        experiment_number="0",
        total_epochs=0,
        eta="",
        last_seen_epoch=0,
        automl=False,
        job_id="",
        rec_job_id="",
        previous_result_metadata=None,
        automl_brain=False,
        workspace_metadata=None,
        cloud_results_dir=None
    ):
        """Update results in status DB

        Args:
            workspace_metadata: Optional workspace metadata for SLURM sync
            cloud_results_dir: Optional remote results directory (e.g., Lustre path) for SLURM sync
        """
        # For SLURM jobs, sync status.json from Lustre to DB before reading
        # This ensures StatusParser reads the latest status updates
        if workspace_metadata and cloud_results_dir:
            cloud_type = workspace_metadata.get('cloud_type', '')
            if cloud_type in ["slurm", "lepton"]:
                from nvidia_tao_core.microservices.handlers.execution_handlers.slurm_handler import (
                    get_slurm_handler_from_workspace
                )
                from nvidia_tao_core.microservices.handlers.execution_handlers.lepton_handler import (
                    get_lepton_handler_from_workspace
                )
                workspace_id = workspace_metadata.get('id')
                if workspace_id:
                    slurm_handler = get_slurm_handler_from_workspace(workspace_id) if cloud_type == "slurm" else None
                    lepton_handler = get_lepton_handler_from_workspace(workspace_id) if cloud_type == "lepton" else None
                    if slurm_handler or lepton_handler:
                        try:
                            # Determine which job_id to use for sync
                            sync_job_id = rec_job_id if rec_job_id else job_id

                            logger.debug(
                                "StatusParser: Syncing cloud status for job_id=%s, automl=%s, exp=%s",
                                sync_job_id, automl, experiment_number
                            )
                            if slurm_handler:
                                slurm_handler.sync_status_to_database(
                                    job_id=sync_job_id,
                                    results_dir=cloud_results_dir,
                                )
                            if lepton_handler:
                                lepton_handler.sync_status_to_database(
                                    job_id=sync_job_id,
                                    results_dir=cloud_results_dir,
                                    workspace_metadata=workspace_metadata,
                                )
                        except Exception as e:
                            logger.warning(
                                "StatusParser: Failed to sync status for job %s: %s",
                                sync_job_id, str(e)
                            )
                            # Continue - work with whatever is in DB

        # Read all the status lines in status DB till now
        good_statuses = []
        lines_to_process = get_dnn_status(job_id, automl=automl, experiment_number=experiment_number)[self.cur_line:]
        for status_dict in lines_to_process:
            try:
                good_statuses.append(status_dict)
            except Exception as e:
                logger.error("Exception thrown in while adding good statuses in update_results is %s", str(e))
                continue
            self.cur_line += 1

        for status_dict in good_statuses:
            # Logging fields
            for key in ["date", "time", "status", "message"]:
                if key in status_dict:
                    self.results[key] = status_dict[key]

            # Categorical
            self._update_categorical(status_dict)

            # KPI
            self._update_kpi(status_dict)

            # Graphical
            self._update_graphical(status_dict)

            # Continuous
            for key in status_dict:
                if key in CONTINUOUS_STATUS_KEYS:
                    # verbosity is an additional status DB variable API does not process
                    self.results[key] = status_dict[key]
        post_process_results_job_id = job_id
        if rec_job_id:
            post_process_results_job_id = rec_job_id
        return self.post_process_results(
            total_epochs,
            eta,
            last_seen_epoch,
            automl,
            post_process_results_job_id,
            previous_result_metadata,
            automl_brain
        )

    def trim_list(self, metric_list, automl_algorithm, brain_epoch_number):
        """Retains only the tuples whose epoch numbers are <= required epochs"""
        trimmed_list = []
        for tuple_var in metric_list:
            epoch, value = (int(tuple_var[0]), tuple_var[1])
            if epoch >= 0:
                if automl_algorithm in ("bayesian", "b", ""):
                    excluded_networks = set(["pointpillars", "bevfusion", "ml_recog"])
                    if self.network in (_PYT_TAO_NETWORKS - excluded_networks):
                        # epoch number in checkpoint starts from 0 or models whose validation logs
                        # are generated before the training logs
                        if epoch < brain_epoch_number:
                            trimmed_list.append((epoch, value))
                    else:
                        trimmed_list.append((epoch, value))
                elif (self.network in ("bevfusion", "ml_recog", "cosmos-rl") and epoch <= brain_epoch_number):
                    trimmed_list.append((epoch, value))
                elif epoch < brain_epoch_number:
                    trimmed_list.append((epoch, value))
        return trimmed_list

    def read_metric(self, results, metric="loss", automl_algorithm="", automl_brain_job_id="", brain_epoch_number=0):
        """Parses the status parser object and returns the metric of interest

        result: value from status_parser.update_results()
        returns: the metric requested in normalized float
        """
        logger.debug(f"results: {results}")
        logger.debug(
            f"metric: {metric}, automl_algorithm: {automl_algorithm}, "
            f"automl_brain_job_id: {automl_brain_job_id}, "
            f"brain_epoch_number: {brain_epoch_number}"
        )
        metric_value = 0.0
        try:
            for result_type in ("graphical", "kpi"):
                for log in results[result_type]:
                    if metric == "kpi":
                        criterion = get_monitoring_metric(self.network)
                    else:
                        criterion = metric

                    # Normalize criterion to list for consistent handling
                    criterion_list = normalize_metric_config(criterion)
                    reverse_sort = True
                    logger.info("Metric: %s, Criterion(s): %s in read_metric", metric, criterion_list)

                    # Check if criterion contains loss-related metrics
                    criterion_str = str(criterion_list[0]) if criterion_list else ""
                    if metric == "loss" or criterion_str in ("loss", "evaluation_cost") or "loss" in criterion_str:
                        reverse_sort = False

                    # Check if log metric matches any criterion in the list
                    if log["metric"] in criterion_list:
                        if log["values"]:
                            logger.debug(f"log['values']: {log['values']}")
                            values_to_search = self.trim_list(
                                metric_list=log["values"].items(),
                                automl_algorithm=automl_algorithm,
                                brain_epoch_number=brain_epoch_number
                            )
                            logger.debug(f"values_to_search: {values_to_search}")
                            # Hyperband-like algorithms use bracket/sh_iter structure
                            if automl_algorithm in ("hyperband", "h", "bohb", "asha", "dehb", "hyperband_es", "hes"):
                                brain_dict = get_automl_brain_info(automl_brain_job_id)
                                bracket_key = str(brain_dict.get("bracket", 0))
                                ni_list = brain_dict.get("ni", [str(float('-inf'))])[bracket_key]
                                sh_iter = brain_dict.get("sh_iter", float('inf'))
                                if len(ni_list) != (sh_iter + 1):
                                    self.best_epoch_number, metric_value = values_to_search[-1]
                                else:
                                    self.best_epoch_number, metric_value = sorted(
                                        sorted(values_to_search, key=lambda x: x[0], reverse=False),
                                        key=lambda x: x[1],
                                        reverse=reverse_sort
                                    )[0]
                            else:
                                self.best_epoch_number, metric_value = sorted(
                                    sorted(values_to_search, key=lambda x: x[0], reverse=True),
                                    key=lambda x: x[1],
                                    reverse=reverse_sort
                                )[0]
                            self.latest_epoch_number, _ = sorted(values_to_search, key=lambda x: x[0], reverse=True)[0]
                            metric_value = float(metric_value)
                            break
        except Exception:
            # Something went wrong inside...
            logger.error(traceback.format_exc())
            logger.warning("Requested metric not found, defaulting to 0.0")
            monitoring_metric = get_monitoring_metric(self.network)
            monitoring_metric_list = normalize_metric_config(monitoring_metric)
            # Check if any of the monitoring metrics indicate a loss-type metric
            is_loss_metric = (
                metric in ("loss", "evaluation_cost") or
                any(m in ("loss", "evaluation_cost") or "loss" in str(m) for m in monitoring_metric_list)
            )
            if is_loss_metric:
                metric_value = 1e7
            else:
                metric_value = 1e-7

        if self.network in STATUS_CALLBACK_MISMATCH_WITH_CHECKPOINT_EPOCH:
            self.best_epoch_number += 1
            self.latest_epoch_number += 1
        logger.info(
            f"Metric returned is {metric_value} at best epoch/iter {self.best_epoch_number} "
            f"while latest epoch/iter is {self.latest_epoch_number}",
        )
        return metric_value, self.best_epoch_number, self.latest_epoch_number


def search_for_dataset(root):
    """Return path of the dataset file"""
    datasets = (
        glob.glob(root + "/*.tar.gz", recursive=False) +
        glob.glob(root + "/*.tgz", recursive=False) +
        glob.glob(root + "/*.tar", recursive=False)
    )

    if datasets:
        dataset_path = datasets[0]  # pick one arbitrarily
        return dataset_path
    return None


def search_for_base_experiment(root, network=""):
    """Return path of the Base-experiment file or spec file for TAO under the Base-experiment root folder"""
    artifacts = (
        glob.glob(root + "/**/*.tlt", recursive=True) +
        glob.glob(root + "/**/*.hdf5", recursive=True) +
        glob.glob(root + "/**/*.pth", recursive=True) +
        glob.glob(root + "/**/*.pth.tar", recursive=True) +
        glob.glob(root + "/**/*.pt", recursive=True)
    )
    if artifacts:
        artifact_path = artifacts[0]  # pick one arbitrarily
        return artifact_path
    return None


def get_dataset_download_command(dataset_metadata):
    """Frames a wget and untar commands to download the dataset"""
    from .stateless_handler_utils import get_handler_metadata

    workspace_id = dataset_metadata.get("workspace")
    workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
    meta_data = copy.deepcopy(workspace_metadata)

    cloud_type = meta_data.get("cloud_type")
    cloud_specific_details = meta_data.get("cloud_specific_details")

    if cloud_specific_details:
        from .cloud_utils import create_cs_instance
        cs_instance, cloud_specific_details = create_cs_instance(meta_data)

    cloud_file_path = dataset_metadata.get("cloud_file_path")

    cloud_download_url = dataset_metadata.get("url", "")
    if not cloud_type:
        cloud_type = "self_hosted"
        if "huggingface" in cloud_download_url:
            cloud_type = "huggingface"

    temp_dir = tempfile.TemporaryDirectory().name  # pylint: disable=R1732
    create_folder_with_permissions(temp_dir)

    # if pull url, then download the dataset into some place inside root
    cmnd = ""
    if cloud_type == "self_hosted":
        cmnd = (
            f"until wget --timeout=1 --tries=1 --retry-connrefused --no-verbose "
            f"--directory-prefix={temp_dir}/ {cloud_download_url}; do sleep 10; done"
        )
    elif cloud_type in ("aws", "azure", "seaweedfs", "lepton"):
        if cloud_file_path.startswith("/"):
            cloud_file_path = cloud_file_path[1:]
        logger.info("Downloading to %s", os.path.join(temp_dir, cloud_file_path))
        if cloud_specific_details:
            cs_instance.download_folder(cloud_file_path, temp_dir)
    elif cloud_type == "huggingface":
        if cloud_specific_details:
            hf_token = cloud_specific_details.get("token", "")
            match = re.match(r"https://huggingface.co/datasets/([^/]+)/", cloud_download_url)
            username = ""
            if match:
                username = match.group(1)
            cmnd = f"git clone https://{username}:{hf_token}@{cloud_download_url.replace('https://', '')} {temp_dir}"
        else:
            cmnd = f"git clone {cloud_download_url} {temp_dir}"
    # run and wait till it finishes / run in background
    if cmnd:
        logger.info("Executing command: %s", cmnd)
    return cmnd, temp_dir


def download_dataset(handler_dataset):
    """Calls wget and untar"""
    if handler_dataset is None:
        return None, None
    tar_file_path = None
    metadata = resolve_metadata("dataset", handler_dataset)
    status = metadata.get("status")
    temp_dir = ""
    if status == "starting":
        metadata["status"] = "in_progress"
        write_handler_metadata(handler_dataset, metadata, "datasets")

        # Get download command - guaranteed to be non-None based on earlier checks
        dataset_download_command, temp_dir = get_dataset_download_command(metadata)  # Non-None (checked earlier)
        if dataset_download_command:
            if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                # In dev setting, we don't need to set HOME
                result = subprocess.run(
                    ['/bin/bash', '-c', dataset_download_command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False
                )
            else:
                cmd = 'HOME=/var/www/ && ' + dataset_download_command
                result = subprocess.run(
                    ['/bin/bash', '-c', cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False
                )
            if result.stdout:
                logger.info("Dataset pull stdout: %s", result.stdout.decode("utf-8"))
            if result.stderr:
                logger.info("Dataset pull stderr: %s", result.stderr.decode("utf-8"))

        tar_file_path = search_for_dataset(temp_dir)
        if not tar_file_path:  # If dataset downloaded is of folder type
            tar_file_path = temp_dir
        metadata["status"] = "pull_complete"
        write_handler_metadata(handler_dataset, metadata, "datasets")
    return temp_dir, tar_file_path


def validate_and_update_experiment_metadata(user_id, org_name, request_dict, meta_data, key_list):
    """Updates experiment metadata with specified keys from the request.

    Args:
        user_id (str): UUID of the user requesting the update.
        org_name (str): Name of the organization.
        request_dict (dict): Dictionary containing the update request.
        meta_data (dict): Existing metadata of the experiment.
        key_list (list): List of keys to be updated if present in `request_dict`.

    Returns:
        tuple:
            - dict: Updated metadata if the update is successful.
            - Code or None: Returns `Code(400)` with error message if key validation fails,
                           otherwise returns `None`.

    Notes:
        - Only updates keys present in `key_list` and `request_dict`.
        - Uses `experiment_update_handler_attributes` to validate values before updating.
        - If any validation fails, the function returns the current metadata along with an error code.
    """
    for key in key_list:
        if key in request_dict.keys():
            value = request_dict[key]
            if experiment_update_handler_attributes(user_id, org_name, meta_data, key, value):
                meta_data[key] = value
            else:
                return meta_data, Code(400, {}, f"Provided {key} cannot be added")
    return meta_data, None


def validate_and_update_base_experiment_metadata(base_experiment_file, base_experiment_id, meta_data):
    """Checks downloaded file hash and updates status in metadata"""
    sha256_digest = meta_data.get("sha256_digest", "")
    logger.info(f"File {base_experiment_file} already exists, validating")
    sha256_digest_matched = validate_ptm_download(base_experiment_file, sha256_digest)
    msg = "complete" if sha256_digest_matched else "in-complete"
    logger.info(f"Download of {base_experiment_id} is {msg}")
    meta_data["base_experiment_pull_complete"] = "pull_complete" if sha256_digest_matched else "starting"
    update_base_experiment_metadata(base_experiment_id, meta_data)


def write_nested_dict(dictionary, key_dotted, value):
    """Merge 2 dicitonaries"""
    ptr = dictionary
    keys = key_dotted.split(".")
    for key in keys[:-1]:
        if type(ptr) is not dict:
            temp = {}
            for ptr_dic in ptr:
                temp.update(ptr_dic)
            ptr = temp
        ptr = ptr.setdefault(key, {})
    ptr[keys[-1]] = value


def write_nested_dict_if_exists(target_dict, nested_key, source_dict, key):
    """Merge 2 dicitonaries if given key exists in the source dictionary"""
    if key in source_dict:
        write_nested_dict(target_dict, nested_key, source_dict[key])
    # if key is not there, no update


def read_nested_dict(dictionary, flattened_key):
    """Returns the value of a flattened key separated by dots"""
    for key in flattened_key.split("."):
        value = dictionary[key]
        dictionary = value
    return value


def build_cli_command(config_data):
    """Generate cli command from the values of config_data"""
    # data is a dict
    # cmnd generates --<field_name> <value> for all key,value in data
    # Usage: To generate detectnet_v2 train --<> <> --<> <>,
    # The part after detectnet_v2 train is generated by this
    cmnd = ""
    for key, value in config_data.items():
        assert (type(value) is not dict), f"Config value for '{key}' cannot be a dictionary"
        assert (type(value) is not list), f"Config value for '{key}' cannot be a list"
        if type(value) is bool:
            if value:
                cmnd += f"--{key} "
        else:
            cmnd += f"--{key}={value} "
    return cmnd


def get_flatten_specs(dict_spec, flat_specs, parent=""):
    """Flatten nested dictionary"""
    for key, value in dict_spec.items():
        if isinstance(value, dict):
            get_flatten_specs(value, flat_specs, parent + key + ".")
        else:
            flat_key = parent + key
            flat_specs[flat_key] = value


def get_total_epochs(job_context, handler_root, automl=False, automl_experiment_id=None):
    """Get the epoch/iter number from specs of train action"""
    job_id = job_context if type(job_context) is str else job_context.id
    spec = get_job_specs(job_id, automl=automl, automl_experiment_id=automl_experiment_id)
    max_epoch = 100.0
    for key1 in spec:
        if key1 in ("training_config", "train_config", "train"):
            for key2 in spec[key1]:
                if key2 in ("num_epochs", "epochs", "n_epochs", "max_iters", "epoch"):
                    max_epoch = int(spec[key1][key2])
                elif key2 in ("train_config"):
                    for key3 in spec[key1][key2]:
                        if key3 == "runner":
                            for key4 in spec[key1][key2][key3]:
                                if key4 == "max_epochs":
                                    max_epoch = int(spec[key1][key2][key3][key4])
        elif key1 in ("num_epochs"):
            max_epoch = int(spec[key1])

    return max_epoch


def _check_gpu_conditions(field_name, field_value):
    if not field_value:
        raise ValueError("GPU related value not set")
    available_gpus = int(os.getenv("NUM_GPU_PER_NODE", "0"))
    if field_name in ("gpus", "num_gpus"):
        if int(field_value) < 0:
            raise ValueError("GPU related value requested is negative")
        if int(field_value) > available_gpus:
            raise ValueError(
                f"GPUs requested count of {field_value} is greater than "
                f"gpus made available during deployment {available_gpus}"
            )
    if field_name in ("gpu_ids", "gpu_id"):
        available_gpu_ids = set(range(0, available_gpus))
        requested_gpu_ids = set(field_value)
        if not requested_gpu_ids.issubset(available_gpu_ids):
            raise ValueError(
                f"GPU ids requested is {str(requested_gpu_ids)} but available gpu ids are {str(available_gpu_ids)}"
            )


def is_remote_backend(backend_details):
    """Check if the backend is a remote scheduler (SLURM or Lepton) from backend_details dict."""
    if not backend_details or not isinstance(backend_details, dict):
        return False
    return backend_details.get('backend_type') in ('slurm', 'lepton')


def get_nested_dict_value(data, key_path):
    """Get value from nested dictionary using dot notation key path.

    Args:
        data (dict): The dictionary to search in
        key_path (str): Dot-separated key path (e.g., "policy.parallelism.dp_shard_size")

    Returns:
        The value at the key path, or None if not found
    """
    if not key_path or not isinstance(data, dict):
        return None

    keys = key_path.split('.')
    current = data

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None

    return current


def _get_cosmos_rl_total_gpus(spec):
    """Calculate total GPUs needed for cosmos-rl based on training mode.

    For SFT mode: policy_tp_size × policy_dp_shard_size
    For GRPO/RL mode: policy GPUs + rollout GPUs

    Args:
        spec: The specification dictionary

    Returns:
        int: Total number of GPUs required, or None if calculation fails
    """
    try:
        # Get training policy type (sft or grpo/rl)
        train_policy_type = get_nested_dict_value(spec, "train.train_policy.type")
        if train_policy_type:
            train_policy_type = train_policy_type.lower()
        else:
            train_policy_type = "sft"  # Default to SFT

        # Get policy parallelism parameters
        policy_tp_size = get_nested_dict_value(spec, "policy.parallelism.tp_size") or 1
        policy_dp_shard_size = get_nested_dict_value(spec, "policy.parallelism.dp_shard_size") or 1
        policy_n_init_replicas = get_nested_dict_value(spec, "policy.parallelism.n_init_replicas") or 1

        # Calculate policy GPUs
        policy_gpus = policy_tp_size * policy_dp_shard_size * policy_n_init_replicas

        # For GRPO/RL, also add rollout GPUs
        if train_policy_type in ("grpo", "rl"):
            rollout_tp_size = get_nested_dict_value(spec, "rollout.parallelism.tp_size") or 1
            rollout_n_init_replicas = get_nested_dict_value(spec, "rollout.parallelism.n_init_replicas") or 1
            rollout_gpus = rollout_tp_size * rollout_n_init_replicas

            total_gpus = policy_gpus + rollout_gpus
            logger.debug(
                f"[COSMOS-RL] GRPO mode: policy_gpus={policy_gpus} "
                f"(tp={policy_tp_size} × dp_shard={policy_dp_shard_size} × replicas={policy_n_init_replicas}), "
                f"rollout_gpus={rollout_gpus} (tp={rollout_tp_size} × replicas={rollout_n_init_replicas}), "
                f"total={total_gpus}"
            )
        else:
            total_gpus = policy_gpus
            logger.debug(
                f"[COSMOS-RL] SFT mode: policy_gpus={policy_gpus} "
                f"(tp={policy_tp_size} × dp_shard={policy_dp_shard_size} × replicas={policy_n_init_replicas})"
            )

        return total_gpus
    except Exception as e:
        logger.warning(f"[COSMOS-RL] Failed to calculate total GPUs: {e}")
        return None


def _get_cosmos_rl_num_nodes(spec, gpus_per_node=8):
    """Calculate the number of SLURM nodes needed for cosmos-rl.

    Computes total GPUs from the parallelism config and divides by
    gpus_per_node to get the required number of nodes.

    Args:
        spec: The specification dictionary
        gpus_per_node: Number of GPUs per node (default: 8)

    Returns:
        int: Number of nodes required, or None if calculation fails
    """
    total_gpus = _get_cosmos_rl_total_gpus(spec)
    if total_gpus is None or total_gpus <= 0:
        return None
    num_nodes = math.ceil(total_gpus / gpus_per_node)
    logger.debug(
        f"[COSMOS-RL] num_nodes={num_nodes} "
        f"(total_gpus={total_gpus} / gpus_per_node={gpus_per_node})"
    )
    return num_nodes


def get_num_gpus_from_spec(spec, action, network=None, default=0, skip_gpu_conditions_check=False):
    """Validate the gpus requested.

    Args:
        skip_gpu_conditions_check: If True, skip per-node GPU validation.
            Set this for SLURM backends where total GPUs may exceed per-node
            count because SLURM handles multi-node GPU scheduling itself.
    """
    if not isinstance(spec, dict):
        return default

    if action == "retrain":
        action = "train"

    gpu_set_values = []

    # Special handling for cosmos-rl to calculate total GPUs correctly.
    # Only use training-specific calculation for train/retrain actions;
    # evaluate/inference use the top-level num_gpus from specs.
    if network == "cosmos-rl" and action not in ("evaluate", "inference"):
        cosmos_rl_gpus = _get_cosmos_rl_total_gpus(spec)
        if cosmos_rl_gpus is not None and cosmos_rl_gpus > 0:
            if not skip_gpu_conditions_check:
                _check_gpu_conditions("num_gpus", cosmos_rl_gpus)
            gpu_set_values.append(cosmos_rl_gpus)
            return cosmos_rl_gpus

    # First check for network-specific GPU parameter using gpu_mapper
    if network and network in gpu_mapper:
        gpu_param_path = gpu_mapper[network]
        if gpu_param_path:
            network_gpu_value = get_nested_dict_value(spec, gpu_param_path)
            if network_gpu_value is not None and network_gpu_value != 0:
                if isinstance(network_gpu_value, (int, float)):
                    if not skip_gpu_conditions_check:
                        _check_gpu_conditions("num_gpus", network_gpu_value)
                    gpu_set_values.append(int(network_gpu_value))
                elif isinstance(network_gpu_value, list):
                    if not skip_gpu_conditions_check:
                        _check_gpu_conditions("gpu_ids", network_gpu_value)
                    gpu_set_values.append(len(set(network_gpu_value)))

    # Fall back to original logic for standard GPU parameters
    for gpu_param_name in ("gpus", "num_gpus", "gpu_ids", "gpu_id"):
        field_value = 0
        field_name = ""
        if gpu_param_name in spec.keys():
            field_name = gpu_param_name
            field_value = spec[gpu_param_name]
            if field_value != 0 and not skip_gpu_conditions_check:
                _check_gpu_conditions(field_name, field_value)
        if action in spec and gpu_param_name in spec[action]:
            field_name = gpu_param_name
            field_value = spec[action][gpu_param_name]
            if field_value != 0 and not skip_gpu_conditions_check:
                _check_gpu_conditions(field_name, field_value)
        if action in spec and "system" in spec[action]:
            gpu_set_values.append(int(spec[action]["system"].get(gpu_param_name, 0)))
        if field_name in ("gpus", "num_gpus"):
            gpu_set_values.append(int(field_value))
        if field_name in ("gpu_ids", "gpu_id"):
            if type(field_value) is int:
                gpu_set_values.append(1)
            elif type(field_value) is list:
                gpu_set_values.append(len(set(field_value)))
    if gpu_set_values:
        return max(gpu_set_values)
    return 1


def get_num_nodes_from_spec(spec, action, network=None, default=1):
    """Validate the nodes requested

    Args:
        spec (dict): The specification dictionary containing configuration details.
        action (str): The action key to look for in the spec dictionary.
        default (int): The default number of nodes to return if none are specified.

    Returns:
        int: The maximum number of nodes specified in the spec, or the default value if none are found.
    """
    if not isinstance(spec, dict):
        return default

    if action == "retrain":
        action = "train"

    node_set_values = []

    # First check for network-specific node parameter using node_mapper
    if network and network in node_mapper:
        node_param_path = node_mapper[network]
        if node_param_path:  # Only check if there's a non-empty path defined
            network_node_value = get_nested_dict_value(spec, node_param_path)
            if network_node_value is not None and network_node_value != 0:
                if isinstance(network_node_value, (int, float)):
                    node_set_values.append(int(network_node_value))

    # Accessing num_nodes under train['system']
    node_param_name = "num_nodes"

    if action in spec and "system" in spec[action]:
        field_value = int(spec[action]["system"].get(node_param_name, 0))
        if field_value != 0:
            node_set_values.append(field_value)

    if node_param_name in spec:
        field_value = int(spec.get(node_param_name, 0))
        if field_value != 0:
            node_set_values.append(field_value)

    if action in spec and node_param_name in spec[action]:
        field_value = int(spec[action][node_param_name])
        if field_value != 0:
            node_set_values.append(field_value)

    if node_set_values:
        return max(node_set_values)
    return default


def validate_num_gpu(num_gpu=None, action: str = ""):
    """Validate the requested number of GPUs and return the validated number of GPUs.

    Args:
        num_gpu (str | None): Number of GPUs.
        action (str): Action to be performed.

    Returns:
        int: Validated number of GPUs.
        str: Error message indicating why validation fails.
    """
    # No gpu if num_gpu is not provided
    if num_gpu is None or num_gpu == 0:
        return 0, ""  # No GPU is requested. No need to validate further.
    # Convert num_gpu to int if it is a string
    if not isinstance(num_gpu, int):
        try:
            num_gpu = int(num_gpu)
        except ValueError:
            return 0, f"Requested number of GPUs ({num_gpu}) is not a valid number."
    # Check if num_gpu is a valid number
    if num_gpu < -1:
        return 0, f"Requested number of GPUs ({num_gpu}) is invalid negative number."

    # Get maximum available number of GPUs
    num_gpu_per_node = os.getenv("NUM_GPU_PER_NODE")
    if num_gpu_per_node is None:
        return 0, "NUM_GPU_PER_NODE is not set in the environment. Assuming no GPU is available!"
    max_num_gpu = int(num_gpu_per_node)

    # Use all maximum number of GPUs if num_gpu is -1
    if num_gpu == -1:
        return max_num_gpu, f"Requested number of GPUs is -1. Using all maximum number of GPUs ({max_num_gpu})."

    # Limit number of GPUs to the available number of GPUs
    if num_gpu > max_num_gpu:
        return 0, f"Requested number of GPUs ({num_gpu}) is larger than available number of GPUs ({max_num_gpu}). "

    # Use single GPU for actions not supporting multi-GPU
    multi_gpu_supported_actions = [
        "train", "distill", "retrain", "finetune",
        "evaluate", "inference"
    ]
    if action not in multi_gpu_supported_actions:
        if num_gpu > 1:
            return 0, f"Multi-GPU is not supported for {action}."

    return num_gpu, ""


def validate_uuid(dataset_id=None, job_id=None, experiment_id=None, workspace_id=None):
    """Validate possible UUIDs"""
    if dataset_id:
        try:
            uuid.UUID(dataset_id)
        except Exception as e:
            logger.error("Exception thrown in validate_uuid for dataset_id is %s", str(e))
            return "Dataset ID passed is not a valid UUID"
    if job_id:
        try:
            uuid.UUID(job_id)
        except Exception as e:
            logger.error("Exception thrown in validate_uuid for job_id is %s", str(e))
            return "Job ID passed is not a valid UUID"
    if experiment_id:
        try:
            uuid.UUID(experiment_id)
        except Exception as e:
            logger.error("Exception thrown in validate_uuid for experiment_id is %s", str(e))
            return "Experiment ID passed is not a valid UUID"
    if workspace_id:
        try:
            uuid.UUID(workspace_id)
        except Exception as e:
            logger.error("Exception thrown in validate_uuid for workspace_id is %s", str(e))
            return "Workspace ID passed is not a valid UUID"
    return ""


def decrypt_handler_metadata(workspace_metadata):
    """Decrypt NvVault encrypted values"""
    cloud_specific_details = workspace_metadata.get("cloud_specific_details")
    config_path = os.getenv("VAULT_SECRET_PATH", None)
    if config_path and cloud_specific_details:
        from .encrypt_utils import NVVaultEncryption
        encryption = NVVaultEncryption(config_path)
        for key, value in cloud_specific_details.items():
            if encryption.check_config()[0]:
                workspace_metadata["cloud_specific_details"][key] = encryption.decrypt(value)
            else:
                logger.info("deencryption not possible")


def add_workspace_to_cloud_metadata(workspace_metadata, cloud_metadata):
    """Add microservices needed cloud info to cloud_metadata"""
    cloud_type = workspace_metadata.get('cloud_type', '')

    # AWS, AZURE
    cloud_specific_details = workspace_metadata.get('cloud_specific_details', {})
    bucket_name = cloud_specific_details.get('cloud_bucket_name', '')
    access_key = cloud_specific_details.get('access_key', '')
    account_name = cloud_specific_details.get('account_name', '')
    secret_key = cloud_specific_details.get('secret_key', '')
    if not secret_key and account_name and access_key:
        secret_key = access_key
    cloud_region = cloud_specific_details.get('cloud_region', '')
    endpoint_url = cloud_specific_details.get('endpoint_url', '')
    cloud_type = workspace_metadata.get("cloud_type")
    lepton_workspace_id = cloud_specific_details.get('lepton_workspace_id')
    lepton_auth_token = cloud_specific_details.get('lepton_auth_token')
    slurm_user = cloud_specific_details.get('slurm_user')
    slurm_hostname = cloud_specific_details.get('slurm_hostname')
    base_results_dir = cloud_specific_details.get('base_results_dir')
    if lepton_workspace_id and lepton_auth_token:
        cloud_metadata["lepton_workspace_id"] = lepton_workspace_id
        cloud_metadata["lepton_auth_token"] = lepton_auth_token
    if slurm_user and slurm_hostname:
        cloud_metadata["slurm_user"] = slurm_user
        # slurm_hostname is a list of strings for multi-host failover
        cloud_metadata["slurm_hostname"] = slurm_hostname
        if base_results_dir:
            cloud_metadata["base_results_dir"] = base_results_dir
        else:
            cloud_metadata["base_results_dir"] = f"/lustre/fsw/portfolios/edgeai/users/{slurm_user}"

    if cloud_type not in cloud_metadata:
        cloud_metadata[cloud_type] = {}
    cloud_metadata[cloud_type][bucket_name] = {
        "cloud_region": cloud_region,
        "access_key": access_key,
        "account_name": account_name,
        "secret_key": secret_key,
        "endpoint_url": endpoint_url,
    }
    logger.info(f"Generated cloud metadata: {cloud_metadata}")


def get_cloud_metadata(workspace_ids, cloud_metadata):
    """For each workspace_id provided, fetch the necessary cloud info"""
    from .stateless_handler_utils import get_handler_metadata

    workspace_ids = list(set(workspace_ids))
    for workspace_id in workspace_ids:
        workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
        decrypt_handler_metadata(workspace_metadata)
        add_workspace_to_cloud_metadata(workspace_metadata, cloud_metadata)


def get_statefulset_name(job_id):
    """Get the statefulset name for the given job id"""
    if os.getenv('STATEFULSET_NAME'):
        return os.getenv('STATEFULSET_NAME')
    release_name = os.getenv('RELEASE_NAME', default='tao-api')
    return f"{release_name}-sts-{job_id}"


def get_statefulset_service_name(job_id):
    """Get the statefulset service name for the given job id"""
    if os.getenv('STATEFULSET_SERVICE_NAME'):
        return os.getenv('STATEFULSET_SERVICE_NAME')
    release_name = os.getenv('RELEASE_NAME', default='tao-api')
    return f"{release_name}-sts-svc-{job_id}"


def send_microservice_request(
        base_url,
        api_endpoint,
        network,
        action,
        cloud_metadata={},
        specs={},
        job_id="",
        docker_env_vars={},
        cloud_based=True,
        statefulset_replicas=1
):
    """Send a request to an endpoint"""
    if not docker_env_vars.get("TAO_API_SERVER"):
        docker_env_vars["TAO_API_SERVER"] = "https://nvidia.com"
    if not docker_env_vars.get("TAO_LOGGING_SERVER_URL"):
        docker_env_vars["TAO_LOGGING_SERVER_URL"] = "https://nvidia.com"

    if action == "retrain":
        action = "train"

    if cloud_based:
        docker_env_vars["CLOUD_BASED"] = "True"
    else:
        docker_env_vars["CLOUD_BASED"] = "False"  # Disable status callback

    request_metadata = {
        "neural_network_name": network,
        "action_name": action,
        "specs": specs,
        "cloud_metadata": cloud_metadata,
        "docker_env_vars": docker_env_vars,
    }

    if job_id:
        request_metadata["job_id"] = job_id
        request_metadata["docker_env_vars"]["JOB_ID"] = job_id

    endpoint = f"{base_url}/api/v1/internal/container_job"

    if api_endpoint == "get_job_status":
        endpoint = f"{base_url}/api/v1/internal/container_job:status"
        request_metadata = {"results_dir": specs.get("results_dir", "")}
    elif api_endpoint == "post_action" and statefulset_replicas > 1:
        request_metadata["statefulset_replicas"] = statefulset_replicas
    # Send request
    if os.getenv("DEBUG_MODE", "false").lower() == "true":
        logger.info(f"Sending request to {endpoint} with request_metadata {request_metadata}")
    else:
        logger.info(f"Sending request to {endpoint}")
    try:
        if api_endpoint == "get_job_status":
            response = requests.get(endpoint, params=request_metadata, timeout=120)
        else:
            data = json.dumps(request_metadata)
            response = requests.post(endpoint, data=data, timeout=120)
    except Exception as e:
        logger.error("Exception caught during sending a microservice request %s", e)
        raise e
    return response


def send_statefulset_request(
    api_endpoint,
    network,
    action,
    cloud_metadata={},
    specs={},
    job_id="",
    docker_env_vars={},
    statefulset_replica_index=0,
    statefulset_replicas=1,
):
    """Make a requests call to the microservice within the statefulset

    Args:
        api_endpoint (str): The API endpoint to call, e.g. "get_job_status"
        network (str): The neural network name
        action (str): The action to perform, e.g. "train", "retrain"
        cloud_metadata (dict, optional): Cloud metadata. Defaults to {}.
        specs (dict, optional): Job specifications. Defaults to {}.
        job_id (str, optional): Job ID. Defaults to "".
        docker_env_vars (dict, optional): Docker environment variables. Defaults to {}.
        statefulset_replicas (int, optional): StatefulSet replicas. Defaults to 1.
        statefulset_replica_index (int, optional): StatefulSet replica index. Defaults to 0.
    Returns:
        requests.Response: The response from the microservice pod
    """
    # Construct base URL and endpoint using StatefulSet FQDN
    statefulset_name = get_statefulset_name(job_id)
    statefulset_service_name = get_statefulset_service_name(job_id)
    statefulset_namespace = os.getenv("NAMESPACE", "default")
    base_url = (
        f"http://{statefulset_name}-{statefulset_replica_index}."
        f"{statefulset_service_name}.{statefulset_namespace}."
        "svc.cluster.local:8000"
    )
    if statefulset_replica_index != 0:
        statefulset_replicas = 0
    try:
        response = send_microservice_request(
            base_url,
            api_endpoint,
            network,
            action,
            cloud_metadata,
            specs,
            job_id,
            docker_env_vars,
            statefulset_replicas)
    except Exception as e:
        logger.error("Exception caught during sending a microservice request %s", e)
        raise e
    return response


def sanitize_metadata(metadata):
    """Convert metadata datetime objects to strings.

    MongoDB natively supports datetime objects. However, we pass a dict string as env variable to DNN container.
    DNN Container uses ast.literal_eval to safely reconstruct this string into a dict.
    However, ast.literal_eval doesn't support datetime objects, so we convert datetime
    objects to string here before passing the dict as a string in the job env variables.
    """
    if 'last_modified' in metadata and isinstance(metadata['last_modified'], datetime):
        date_string = metadata['last_modified'].isoformat()
        metadata['last_modified'] = date_string

    if 'created_on' in metadata and isinstance(metadata['created_on'], datetime):
        date_string = metadata['created_on'].isoformat()
        metadata['created_on'] = date_string

    metadata.pop('_id', None)


def latest_model(files, delimiters="_", epoch_number="000", extensions=[".tlt", ".hdf5", ".pth"], network_name=""):
    """Returns the latest generated model file based on epoch number"""
    # Update extensions based on network config if available
    if network_name:
        network_config = get_network_config(network_name)
        checkpoint_config = network_config.get("checkpoint", {})
        if checkpoint_config:
            checkpoint_format = checkpoint_config.get("format", "")
            if checkpoint_format:
                extensions = [f".{checkpoint_format}"]
    cur_best = 0
    best_model = None
    for file in files:
        _, file_extension = os.path.splitext(file)
        if file_extension not in extensions:
            continue
        if "_latest" in os.path.basename(file):
            continue
        model_name = file
        for extension in extensions:
            model_name = re.sub(f"{extension}$", "", model_name)
        delimiters_list = delimiters.split(",")
        if len(delimiters_list) > 1:
            delimiters_list = delimiters_list[0:-1]
        for delimiter in delimiters_list:
            epoch_num = model_name.split(delimiter)[-1]
            model_name = epoch_num
        if len(delimiters) > 1:
            epoch_num = model_name.split(delimiters[-1])[0]
        try:
            epoch_num = int(epoch_num)
        except Exception as e:
            logger.error("Exception thrown in latest_model is %s", str(e))
            epoch_num = 0
        if epoch_num >= cur_best:
            cur_best = epoch_num
            best_model = file
    checkpoint_name = None
    if best_model:
        checkpoint_name = f"/{best_model}"
    return checkpoint_name


def filter_files(files, regex_pattern="", network_name=""):
    """Filter file list based on regex provided

    Args:
        files: List of file paths to filter
        regex_pattern: Custom regex pattern to use for filtering
        network_name: Network name to read checkpoint config from network_configs

    Returns:
        List of filtered file paths
    """
    # Try to get checkpoint config from network configuration
    if not regex_pattern:
        regex_pattern = r'^(?!.*lightning_logs).*\.(pth|tlt|hdf5)$'

    if network_name:
        network_config = get_network_config(network_name)
        checkpoint_config = network_config.get("checkpoint", {})

        if checkpoint_config:
            checkpoint_format = checkpoint_config.get("format", "")
            # Build regex pattern based on format
            if checkpoint_format:
                regex_pattern = rf'.*\.{re.escape(checkpoint_format)}$'

    checkpoints = [path for path in files if re.match(regex_pattern, path)]
    return checkpoints


def get_network_config(network_name):
    """Read network configuration from network_configs directory"""
    if not network_name:
        return {}

    # Get the directory where this utilities.py file is located
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file = os.path.join(current_dir, "handlers", "network_configs", f"{network_name}.config.json")
    if not os.path.exists(config_file):
        logger.error("Network config file not found: %s", config_file)
        return {}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error("Error reading network config for %s: %s", network_name, str(e))
        return {}


def filter_file_objects(file_objects, regex_pattern="", network_name=""):
    """Based on regex provided filter file_objects list

    Args:
        file_objects: List of file objects to filter
        regex_pattern: Custom regex pattern to use for filtering
        network_name: Network name to read checkpoint config from network_configs

    Returns:
        List of filtered file objects
    """
    # Try to get checkpoint config from network configuration
    if not regex_pattern:
        regex_pattern = r'.*\.(pth|tlt|hdf5)$'

    if network_name:
        network_config = get_network_config(network_name)
        checkpoint_config = network_config.get("checkpoint", {})

        if checkpoint_config:
            checkpoint_format = checkpoint_config.get("format", "")
            # Build regex pattern based on format
            if checkpoint_format:
                regex_pattern = rf'.*\.{re.escape(checkpoint_format)}$'

    filtered_objects = [file_object for file_object in file_objects if re.match(regex_pattern, file_object.name)]
    return filtered_objects


def format_checkpoints_path(checkpoints):
    """Add formatting to the checkpoint name"""
    checkpoint_name = None
    if checkpoints:
        # Don't add leading slash if path already starts with one (e.g., absolute paths from SLURM)
        if checkpoints[0].startswith('/'):
            checkpoint_name = checkpoints[0]
        else:
            checkpoint_name = f"/{checkpoints[0]}"
    return checkpoint_name


def from_epoch_number(files, delimiters="", epoch_number="000", network_name=""):
    """Based on the epoch number string passed, returns the path of the checkpoint.

    If a checkpoint with the epoch info is not present, raises an exception.

    Args:
        files: List of files to search through
        delimiters: String of delimiters to use for parsing
        epoch_number: Epoch number to search for

    Returns:
        str: Path to the checkpoint file, or None if not found
    """
    regex_pattern = fr'''
    ^(?!.*lightning_logs)               # Exclude files with 'lightning_logs' in the path
    .*                                  # Match any preceding text
    (?:_({epoch_number})          # Match '_epoch_<epoch_number>' for general networks
    _step_\d+)?                         # Optionally match '_step_<digits>' for general networks
    |                                   # OR
    (?:_({epoch_number}))               # Match '_<epoch_number>' for MISSING_EPOCH_FORMAT_NETWORKS
    \.(pth|tlt|hdf5)$                   # Match file extensions
    '''
    checkpoints = filter_files(files, regex_pattern, network_name=network_name)
    checkpoint_name = format_checkpoints_path(checkpoints)
    return checkpoint_name


def _get_result_file_path(checkpoint_function, files, format_epoch_number, network_name=""):
    result_file = checkpoint_function(files, delimiters="_", epoch_number=format_epoch_number,
                                      network_name=network_name)
    return result_file


def get_file_list_from_cloud_storage(workspace_metadata, res_root):
    """Return files present in res_root in cloud storage - Enhanced with storage fix"""
    # Validate workspace metadata
    if not workspace_metadata:
        logger.error("No workspace metadata provided")
        return []

    if not workspace_metadata.get("cloud_specific_details"):
        logger.error("No cloud_specific_details in workspace metadata")
        return []

    # Create storage client with cleaned metadata
    from .cloud_utils import create_cs_instance
    cs_instance, _ = create_cs_instance(workspace_metadata)

    # Clear any cached state
    if hasattr(cs_instance, '_fs') and cs_instance._fs:
        if hasattr(cs_instance._fs, 'clear_instance_cache'):
            cs_instance._fs.clear_instance_cache()
        if hasattr(cs_instance._fs, 'invalidate_cache'):
            cs_instance._fs.invalidate_cache()

    # For SLURM, paths are already absolute Lustre paths - keep them as-is
    # For other cloud types (AWS, Azure), strip leading '/' to convert to bucket keys
    cloud_type = workspace_metadata.get('cloud_type', '')
    if cloud_type == 'slurm':
        folder_path = res_root  # Keep absolute path for SLURM
    else:
        folder_path = res_root[1:] if res_root.startswith('/') else res_root

    files, _ = cs_instance.list_files_in_folder(folder_path)

    return files


def upload_log_to_cloud(handler_metadata, job_id, log_file_path, automl_index=None):
    """Upload log file from local pod storage to cloud"""
    if not os.path.exists(log_file_path):
        logger.warning(f"Cannot upload log file - does not exist: {log_file_path}")
        return False

    lookup_job_id = job_id
    if "experiment_" in log_file_path:
        controller_list = get_automl_controller_info(job_id)
        for rec_info in controller_list:
            rec_id = rec_info.get("id", "")
            if (rec_id != "" and automl_index is not None) and (int(rec_id) == int(automl_index)):
                lookup_job_id = rec_info.get("job_id", "")
                break

    workspace_id = handler_metadata.get("workspace", "")
    if not workspace_id:
        logger.warning("No workspace assigned, cannot upload log to cloud")
        return False

    try:
        from .stateless_handler_utils import get_handler_metadata
        from .cloud_utils import create_cs_instance
        workspace_metadata = get_handler_metadata(workspace_id, "workspace")
        cs_instance, _ = create_cs_instance(workspace_metadata)

        # Upload to cloud storage at /results/{job_id}/microservices_log.txt
        cloud_path = f"/results/{lookup_job_id}/microservices_log.txt"
        logger.info(f"Uploading log file to cloud: {cloud_path}")
        cs_instance.upload_file(log_file_path, cloud_path, send_status_callbacks=False)
        logger.info(f"Successfully uploaded log file to cloud: {cloud_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload log file to cloud: {type(e).__name__}: {e}")
        logger.debug("Upload exception details:", exc_info=True)
        return False


def download_log_from_cloud(handler_metadata, job_id, log_file_path, automl_index=None):
    """Download log file from cloud onto local pod storage"""
    lookup_job_id = job_id
    if "experiment_" in log_file_path:
        controller_list = get_automl_controller_info(job_id)
        for rec_info in controller_list:
            rec_id = rec_info.get("id", "")
            if (rec_id != "" and automl_index is not None) and (int(rec_id) == int(automl_index)):
                lookup_job_id = rec_info.get("job_id", "")
                break
    workspace_id = handler_metadata.get("workspace", "")
    from .stateless_handler_utils import get_handler_metadata
    from .cloud_utils import create_cs_instance
    workspace_metadata = get_handler_metadata(workspace_id, "workspace")
    cs_instance, _ = create_cs_instance(workspace_metadata)
    if cs_instance.is_file(f"/results/{lookup_job_id}/microservices_log.txt"):
        cs_instance.download_file(f"/results/{lookup_job_id}/microservices_log.txt", log_file_path)
    else:
        logger.error("Log file not found at %s", f"/results/{lookup_job_id}/microservices_log.txt")
        # Best model files are moved under /results/brain_job_id
        log_path = f"/results/{job_id}/microservices_log.txt"
        if cs_instance.is_file(log_path):
            cs_instance.download_file(log_path, log_file_path)
        else:
            logger.error("Log file not found at %s", log_path)


def format_epoch(network, epoch_number):
    """Based on the network returns the epoch number formatted"""
    if network in MISSING_EPOCH_FORMAT_NETWORKS:
        format_epoch_number = str(epoch_number)
    else:
        format_epoch_number = f"{epoch_number:03}"
    return format_epoch_number


def search_for_checkpoint(handler_metadata, job_id, res_root, files, checkpoint_choose_method):
    """Based onf the choice of choosing checkpoint, handle different function calls and return the path found"""
    network = handler_metadata.get("network_arch")
    if network == "vila":
        parent_specs = get_job_specs(job_id)
        if parent_specs:
            llm_mode = parent_specs.get("train", {}).get("llm_mode", "lora")
            vision_mode = parent_specs.get("train", {}).get("vision_mode", "ft")
            result_file = f'results/{job_id}/{vision_mode}_{llm_mode}'
            logger.info("result_file: %s", result_file)
    else:
        epoch_number_dictionary = handler_metadata.get("checkpoint_epoch_number", {})
        epoch_number = epoch_number_dictionary.get(f"{checkpoint_choose_method}_{job_id}", 0)

        if checkpoint_choose_method == "latest_model" or "/best_model" in res_root:
            checkpoint_function = latest_model
        elif checkpoint_choose_method in ("best_model", "from_epoch_number"):
            checkpoint_function = from_epoch_number
        else:
            raise ValueError(f"Chosen method to pick checkpoint not valid: {checkpoint_choose_method}")

        format_epoch_number = format_epoch(network, epoch_number)
        result_file = _get_result_file_path(
            checkpoint_function=checkpoint_function,
            files=files,
            format_epoch_number=format_epoch_number,
            network_name=network
        )
        if (not result_file) and (checkpoint_choose_method in ("best_model", "from_epoch_number")):
            logger.warning(
                "Couldn't find the epoch number requested or the checkpointed "
                "associated with the best metric value, defaulting to latest_model"
            )
            checkpoint_function = latest_model
            result_file = _get_result_file_path(
                checkpoint_function=checkpoint_function,
                files=files,
                format_epoch_number=format_epoch_number,
                network_name=network
            )

    return result_file


def get_files_from_cloud(handler_metadata, job_id, automl=False, automl_experiment_id="0"):
    """Get filelist of a job from cloud - Enhanced with storage fix"""
    if job_id is None:
        return None

    action = get_handler_job_metadata(job_id).get("action")
    lookup_job_id = job_id
    if automl:
        lookup_job_id = get_automl_experiment_job_id(job_id, automl_experiment_id)
        if not lookup_job_id:
            lookup_job_id = job_id
    else:
        # Parent may be an AutoML brain; best model may live in brain or in best experiment folder
        best_rec_number, _, best_model_results_job_id = get_automl_best_rec_info(job_id)
        if best_rec_number != "-1":
            lookup_job_id = best_model_results_job_id
            action = get_handler_job_metadata(lookup_job_id).get("action") or action
    logger.info("lookup_job_id: %s", lookup_job_id)

    # Build results path - for SLURM, use full Lustre path
    workspace_id = handler_metadata.get("workspace")
    workspace_metadata = resolve_metadata("workspace", workspace_id)
    cloud_type = workspace_metadata.get('cloud_type', '') if workspace_metadata else ''

    if cloud_type == 'slurm':
        # For SLURM, construct full Lustre path
        base_results_dir = workspace_metadata.get('cloud_specific_details', {}).get('base_results_dir', '')
        if not base_results_dir:
            # Infer base_results_dir from slurm_user if not set in workspace
            slurm_user = workspace_metadata.get('cloud_specific_details', {}).get('slurm_user', '')
            if slurm_user:
                base_results_dir = f"/lustre/fsw/portfolios/edgeai/users/{slurm_user}"

        if base_results_dir:
            # Append /results to base_results_dir to get full results path
            res_root = os.path.join(base_results_dir, 'results', str(lookup_job_id))
        else:
            # Fallback to TAO_API_RESULTS_DIR
            results_base = os.getenv('TAO_API_RESULTS_DIR', '/results')
            res_root = os.path.join(results_base, str(lookup_job_id))
    else:
        # Use TAO_API_RESULTS_DIR for local/cloud, fallback to /results
        results_base = os.getenv('TAO_API_RESULTS_DIR', '/results')
        res_root = os.path.join(results_base, str(lookup_job_id))

    files = get_file_list_from_cloud_storage(workspace_metadata, res_root)
    return files, action, res_root, workspace_id


def resolve_checkpoint_root_and_search(handler_metadata, job_id, folder=False, regex=None,
                                       automl=False, automl_experiment_id="0"):
    """Returns path of the model based on the action of the job"""
    if job_id is None:
        return None
    files, action, res_root, workspace_id = get_files_from_cloud(
        handler_metadata, job_id, automl=automl, automl_experiment_id=automl_experiment_id
    )
    logger.info("files for job %s: %s", job_id, files)
    network = handler_metadata.get("network_arch", "")

    if action == "retrain":
        action = "train"

    result_file = None
    if action in ("train", "distill", "quantize"):
        checkpoint_choose_method = handler_metadata.get("checkpoint_choose_method", "best_model")
        result_file = search_for_checkpoint(
            handler_metadata=handler_metadata,
            job_id=job_id,
            res_root=res_root,
            files=files,
            checkpoint_choose_method=checkpoint_choose_method
        )

    if action == "export" or (action == "quantize" and not result_file):
        regex_pattern = regex if regex else r'.*\.(onnx|uff)$'
        result_file = filter_files(files, regex_pattern=regex_pattern, network_name=network)
        result_file = format_checkpoints_path(result_file)

    elif action == "prune":
        result_file = filter_files(files, network_name=network)
        result_file = format_checkpoints_path(result_file)

    elif action in ("trtexec", "gen_trt_engine"):
        regex_pattern = regex if regex else r'.*\.(engine)$'
        result_file = filter_files(files, regex_pattern=regex_pattern, network_name=network)
        result_file = format_checkpoints_path(result_file)

    if result_file:
        workspace_identifier = get_workspace_string_identifier(workspace_id, workspace_cache={})
        workspace_metadata = resolve_metadata("workspace", workspace_id)
        cloud_type = workspace_metadata.get('cloud_type', '') if workspace_metadata else ''

        if folder:
            result_file = f"{os.path.dirname(result_file)}"

            # Check if both merged and safetensors folders exist, prefer merged
            parent_dir = os.path.dirname(result_file)
            logger.info("parent_dir: %s", parent_dir)
            checkpoint_folder_name = os.path.basename(result_file)
            logger.info("checkpoint_folder_name: %s", checkpoint_folder_name)

            # If current folder is safetensors, check if merged folder exists
            if checkpoint_folder_name == 'safetensors':
                merged_folder = os.path.join(parent_dir, 'merged')
                logger.info("merged_folder: %s", merged_folder)

                # Check if merged folder exists by listing files
                try:
                    files_in_parent = get_file_list_from_cloud_storage(workspace_metadata, parent_dir)
                    logger.info("files_in_parent: %s", files_in_parent)
                    # Normalize paths for comparison - check if any file is under merged/
                    merged_prefix = f"{parent_dir}/merged/"
                    logger.info("merged_prefix: %s", merged_prefix)
                    if not merged_prefix.startswith('/') and cloud_type != 'slurm':
                        # For cloud storage, paths might not have leading slash
                        merged_prefix = merged_prefix.lstrip('/')
                    elif merged_prefix.startswith('//'):
                        merged_prefix = merged_prefix[1:]

                    merged_exists = any(
                        f.startswith(merged_prefix) or f.startswith(merged_prefix.lstrip('/'))
                        for f in files_in_parent
                    )
                    logger.info("merged_exists: %s", merged_exists)
                    if merged_exists:
                        logger.info(
                            "Both merged and safetensors folders exist. Using merged folder: %s",
                            merged_folder
                        )
                        result_file = merged_folder
                except Exception as e:
                    logger.warning(
                        "Could not check for merged folder existence: %s. Using safetensors folder.",
                        str(e)
                    )

        # For SLURM, if result_file already has absolute path, don't add workspace_identifier
        # (it would cause path duplication since result_file already has full Lustre path)
        if not (cloud_type == 'slurm' and result_file.startswith('/')) and workspace_identifier not in result_file:
            result_file = f"{workspace_identifier}{result_file}"

    return result_file


def get_model_results_path(handler_metadata, job_id, folder=False, automl=False,
                           automl_experiment_id="0"):
    """Return the model file for the job context and handler metadata passes"""
    return resolve_checkpoint_root_and_search(
        handler_metadata, job_id, folder=folder, automl=automl,
        automl_experiment_id=automl_experiment_id
    )
