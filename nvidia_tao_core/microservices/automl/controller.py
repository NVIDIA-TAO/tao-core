# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoML controller modules"""
import os
import re
import glob
import time
import uuid
import traceback
import logging
from copy import deepcopy
from datetime import timedelta

from nvidia_tao_core.microservices.utils.automl_utils import Recommendation, ResumeRecommendation, JobStates
from nvidia_tao_core.microservices.utils.core_utils import get_monitoring_metric, normalize_metric_config
from nvidia_tao_core.microservices.constants import (
    _ITER_MODELS,
    NO_VAL_METRICS_DURING_TRAINING_NETWORKS,
    MISSING_EPOCH_FORMAT_NETWORKS
)
from nvidia_tao_core.microservices.utils.cloud_utils import create_cs_instance_with_decrypted_metadata
from nvidia_tao_core.microservices.utils.handler_utils import (
    StatusParser,
    get_total_epochs,
    get_file_list_from_cloud_storage,
    filter_files,
    format_epoch,
    get_network_config
)
from nvidia_tao_core.microservices.handlers.execution_handlers.slurm_handler import (
    get_results_dir_for_workspace
)
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    update_job_status,
    get_handler_metadata,
    write_handler_metadata,
    get_handler_job_metadata,
    update_job_metadata,
    update_job_message,
    write_job_metadata,
    get_job_specs,
    save_job_specs,
    get_automl_controller_info,
    save_automl_controller_info,
    get_automl_current_rec,
    save_automl_current_rec,
    save_automl_best_rec_info,
    get_automl_brain_info,
    delete_dnn_status,
    update_automl_stats,
    report_health_beat,
    delete_health_beat,
    get_automl_child_job_ids,
    reset_automl_child_lifecycle_for_resume,
    set_automl_checkpoint_cleanup_state,
)
from nvidia_tao_core.microservices.utils.automl_job_utils import (
    on_new_automl_job,
    on_delete_automl_job,
    cancel_automl_child_job,
)
# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)

time_per_epoch = 0
time_per_epoch_counter = 0


class Controller:
    """Abstractly, just a collection of threads and a switch to start and stop them

    - start(): Start all threads needed to run AutoML
    - stop(): Stop all threads started by start()
    - generate_recommendations(): Runs the automl algorithm to generate and analyze recommendations
    - read_results(): Listens to experiments
    - write_results(): Routinely updates a controller_data.json to help Handlers
    """

    def __init__(
        self,
        root,
        network,
        brain,
        automl_context,
        automl_algorithm_settings,
        delete_intermediate_ckpt,
        metric,
        automl_algorithm,
        decrypted_workspace_metadata,
        parameter_names=None
    ):
        """Initialize the Automl Controller class

        Args:
            root: handler root
            network: model name
            brain: Bayesian/Hyperband class object
            automl_context: job context with regards to automl
            automl_algorithm_settings: automl algorithm settings
            delete_intermediate_ckpt: boolean value to delete/not-delete checkpoints which don't correspond to the
            best model
            metric: metric name which will be used to choose best models
            automl_algorithm: automl algorithm name
            decrypted_workspace_metadata: decrypted workspace metadata
            parameter_names: list of parameter names being varied by AutoML
        """
        self.brain = brain

        self.recommendations = []
        self.automl_context = automl_context
        logger.debug("automl_context.id: %s", self.automl_context.id)
        logger.debug("automl_context: %s", self.automl_context)

        self.root = root
        self.network = network
        self.checkpoint_delimiter = ""
        if self.network in MISSING_EPOCH_FORMAT_NETWORKS:
            self.checkpoint_delimiter = "_"
        self.completed_recommendations = 0
        self.automl_algorithm_settings = automl_algorithm_settings
        self.delete_intermediate_ckpt = bool(delete_intermediate_ckpt.lower() == "true")
        self.automl_algorithm = automl_algorithm
        self.decrypted_workspace_metadata = decrypted_workspace_metadata
        self.metric = metric
        hyperband_like_algos = ("hyperband", "h", "bohb", "asha", "dehb", "hyperband_es", "hes")
        if self.automl_algorithm in hyperband_like_algos and self.network in NO_VAL_METRICS_DURING_TRAINING_NETWORKS:
            self.metric_key = "loss"
            self.metric = "loss"
        elif self.metric == "kpi":
            # Get monitoring metric(s) - may be string or list
            monitoring_metric = get_monitoring_metric(self.network)
            # Normalize to list for consistent handling, use first as primary key
            metric_list = normalize_metric_config(monitoring_metric)
            self.metric_key = metric_list[0] if metric_list else monitoring_metric
            # Store full list for matching against available metrics
            self.metric_key_list = metric_list
        else:
            self.metric_key = self.metric
            self.metric_key_list = [self.metric]

        self.brain.reverse_sort = True
        self.min_max = max
        metric_key_str = str(self.metric_key) if self.metric_key else ""
        if self.metric == "loss" or metric_key_str in ("loss", "evaluation_cost") or "loss" in metric_key_str:
            self.brain.reverse_sort = False
            self.min_max = min

        self.total_epochs = 0
        self.first_epoch_number = -1
        self.best_epoch_number = {}
        self.brain_epoch_number = -2
        self.best_model_copied = False
        self.best_rec_id = -1
        self.ckpt_path = {}

        self.old_bracket = "0"
        self.hyperband_cancel_condition_seen = False
        job_metadata = get_handler_job_metadata(self.automl_context.id) or {}
        self._checkpoint_cleanup_pending = bool(
            job_metadata.get("checkpoint_cleanup_pending", False)
        )

        self.wandb_group_name = f"automl_{self.automl_context.id}"
        self.parameter_names = list(parameter_names) if parameter_names else []
        self.wandb_initialized = False
        self.wandb_table = None

        self.eta = "Will be updated after completing one epoch"
        self.remaining_epochs_in_experiment = float("inf")
        self.average_time_per_epoch = float("inf")

        self.on_new_automl_job = lambda jc: on_new_automl_job(self.automl_context, jc)

        self.cs_instance, _ = create_cs_instance_with_decrypted_metadata(self.decrypted_workspace_metadata)
        self.retain_checkpoints_for_resume = automl_context.retain_checkpoints_for_resume

    def _get_checkpoint_config(self):
        """Get checkpoint config from network config"""
        network_config = get_network_config(self.network)
        return network_config.get("checkpoint", {})

    def _uses_folder_lookup(self):
        """Check if network config specifies folder lookup method"""
        checkpoint_config = self._get_checkpoint_config()
        return checkpoint_config.get("folder", "false") == "true"

    def _inject_automl_env_vars(self):
        """Inject AutoML-specific environment variables into experiment docker_env_vars."""
        # Get current experiment metadata
        experiment_metadata = get_handler_metadata(self.automl_context.handler_id, "experiments")

        # Initialize docker_env_vars if not present
        if "docker_env_vars" not in experiment_metadata:
            experiment_metadata["docker_env_vars"] = {}

        # Set AutoML triggered flag and client type
        experiment_metadata["docker_env_vars"]["TAO_AUTOML_TRIGGERED"] = "true"
        if "TAO_CLIENT_TYPE" not in experiment_metadata["docker_env_vars"]:
            # Set to api since AutoML is triggered by API server
            experiment_metadata["docker_env_vars"]["TAO_CLIENT_TYPE"] = "api"
        # Pass brain's log level to train jobs so logging is consistent (brain sets TAO_LOG_LEVEL when started)
        brain_log_level = os.getenv("TAO_LOG_LEVEL", "INFO")
        experiment_metadata["docker_env_vars"]["TAO_LOG_LEVEL"] = brain_log_level

        # Save updated metadata
        write_handler_metadata(self.automl_context.handler_id, experiment_metadata, "experiments")
        logger.info(
            "Injected TAO_AUTOML_TRIGGERED=true and TAO_LOG_LEVEL=%s (from brain) for experiment %s",
            brain_log_level, self.automl_context.handler_id
        )

    def _initialize_wandb_for_automl(self):
        """Initialize WandB for AutoML tracking using API key from experiment metadata.

        This method:
        1. Retrieves WANDB_API_KEY from experiment metadata's docker_env_vars
        2. Logs into WandB using the API key
        3. Initializes a WandB run for the AutoML controller
        4. Creates a table to track all AutoML experiments with their hyperparameters and results
        5. Groups all child experiment runs under self.wandb_group_name for easy visualization

        The WandB table contains:
        - experiment_id: Sequential ID of the recommendation
        - job_id: Unique job identifier
        - status: Current status (pending/running/success/failure)
        - metric_key: The optimization metric value (e.g., mAP, loss)
        - best_epoch_number: The epoch number with the best metric
        - All varying hyperparameters from automl_hyperparameters (parameter_names)
        """
        if self.wandb_initialized:
            return

        try:
            # Get experiment metadata to access docker_env_vars
            experiment_metadata = get_handler_metadata(self.automl_context.handler_id, "experiments")
            docker_env_vars = experiment_metadata.get("docker_env_vars", {})
            wandb_api_key = docker_env_vars.get("WANDB_API_KEY")

            if not wandb_api_key:
                logger.info("WANDB_API_KEY not found in experiment metadata, skipping WandB initialization")
                return

            # Check if wandb is logged in
            try:
                import wandb
                wandb_logged_in = wandb.login(key=wandb_api_key)
                if not wandb_logged_in:
                    logger.warning("Failed to login to WandB, skipping table logging")
                    return
            except Exception as e:
                logger.warning("WandB login failed: %s", str(e))
                return

            # Initialize wandb run for AutoML controller
            specs = get_job_specs(self.automl_context.id)
            wandb_config = specs.get("wandb", {})

            config = {
                "network": self.network,
                "metric": self.metric,
                "algorithm": self.automl_algorithm,
            }
            if self.automl_algorithm in ("hyperband", "h", "bohb", "asha", "dehb", "hyperband_es", "hes", "pbt"):
                config["max_epochs"] = self.automl_algorithm_settings.automl_max_epochs
                config["reduction_factor"] = self.automl_algorithm_settings.automl_reduction_factor
                config["epoch_multiplier"] = self.automl_algorithm_settings.epoch_multiplier
            elif self.automl_algorithm in ("bayesian", "b", "bfbo"):
                config["max_recommendations"] = self.automl_algorithm_settings.automl_max_recommendations
            else:
                raise ValueError(f"AutoML Algorithm {self.automl_algorithm} is not valid")

            wandb.init(
                project=wandb_config.get("project", "TAO Toolkit"),
                entity=wandb_config.get("entity"),
                name="automl_brain",
                group=self.wandb_group_name,
                config=config,
                dir=os.path.join(self.root, "wandb"),
                reinit=True
            )

            self.wandb_initialized = True
            logger.info("WandB initialized for AutoML controller with group: %s", self.wandb_group_name)

            # Create the table
            self._create_wandb_table()

        except Exception as e:
            logger.warning("Failed to initialize WandB for AutoML: %s", str(e))
            self.wandb_initialized = False

    def _create_wandb_table(self):
        """Create WandB table with columns for hyperparameters and metrics."""
        if not self.wandb_initialized:
            return

        try:
            import wandb

            # Define table columns: experiment_id, job_id, status, result, best_epoch_number,
            # plus all varying hyperparameters
            columns = ["experiment_id", "job_id", "status", self.metric_key, "best_epoch_number"]
            columns.extend(self.parameter_names)

            self.wandb_table = wandb.Table(columns=columns)
            logger.info("Created WandB table with columns: %s", columns)

        except Exception as e:
            logger.warning("Failed to create WandB table: %s", str(e))
            self.wandb_table = None

    def _update_wandb_table(self):
        """Update WandB table with current state of all recommendations."""
        if not self.wandb_initialized or self.wandb_table is None:
            return

        try:
            import wandb

            # Recreate the table with all current recommendations
            columns = ["experiment_id", "job_id", "status", self.metric_key, "best_epoch_number"]
            columns.extend(self.parameter_names)
            self.wandb_table = wandb.Table(columns=columns)

            # Add all recommendations as rows
            for rec in self.recommendations:
                # Format result with higher precision to preserve decimal places in WandB display
                # WandB defaults to 4 decimal places, so format as string with more precision
                result_value = rec.result
                if isinstance(result_value, float):
                    # Format with 10 decimal places to preserve precision in WandB display
                    # Preserve at least one decimal place to avoid removing all digits
                    formatted = f"{result_value:.10f}".rstrip('0')
                    if formatted.endswith('.'):
                        formatted = formatted[:-1] + '.0'
                    result_value = formatted

                row_data = [
                    rec.id,
                    rec.job_id or "pending",
                    rec.status,
                    result_value,
                    self.best_epoch_number.get(rec.id, "")
                ]

                # Add hyperparameter values (formatted, e.g., "train.optim.lr")
                # Note: rec.specs is a flat dict with keys like "train.optim.lr", not nested
                for param_name in self.parameter_names:
                    value = rec.specs.get(param_name, "N/A")
                    row_data.append(value)

                self.wandb_table.add_data(*row_data)

            # Log the updated table
            wandb.log({"automl_experiments": self.wandb_table})
            logger.debug("Updated WandB table with %d recommendations", len(self.recommendations))

        except Exception as e:
            logger.warning("Failed to update WandB table: %s", str(e))

    def _get_checkpoint_format(self):
        """Get checkpoint format from network config"""
        checkpoint_config = self._get_checkpoint_config()
        return checkpoint_config.get("format", "")

    def _get_experiment_results_path(self, job_id):
        """Get the results path for an experiment based on workspace type.

        For SLURM workspaces, returns the actual Lustre path.
        For non-SLURM workspaces, returns the local container path.

        Args:
            job_id: The job ID of the experiment

        Returns:
            str: The full path to the experiment results directory
        """
        cloud_type = ''
        if self.decrypted_workspace_metadata:
            cloud_type = self.decrypted_workspace_metadata.get('cloud_type', '')

        if cloud_type == 'slurm':
            return get_results_dir_for_workspace(
                workspace_metadata=self.decrypted_workspace_metadata,
                job_id=job_id
            )

        results_base = os.getenv('TAO_API_RESULTS_DIR', '/results')
        return os.path.join(results_base, job_id)

    def _select_best_epoch_folder(self, epoch_folder_files, all_files, specific_format=None):
        """Select the best epoch folder that contains the required checkpoint format

        Args:
            epoch_folder_files: Files matching the epoch pattern
            all_files: All available files
            specific_format: Optional specific format to look for (e.g., 'safetensors', 'pth')

        Returns:
            str: Path to the selected folder, or None if not found
        """
        checkpoint_format = specific_format or self._get_checkpoint_format()

        # Extract unique folder paths
        folder_paths = list(set(
            "/".join(f.split("/")[:-1]) for f in epoch_folder_files
        ))
        logger.info("Found potential epoch folders: %s", folder_paths)

        # Find the folder that contains files with the correct format
        for folder_path in folder_paths:
            folder_files = [f for f in all_files if f.startswith(folder_path + "/")]

            if checkpoint_format:
                # Check for specific format files
                format_files = [f for f in folder_files if f.endswith(f".{checkpoint_format}")]
                if format_files:
                    logger.info("Selected folder with %s files: %s", checkpoint_format, folder_path)
                    return folder_path
            else:
                # Check for any checkpoint files if no specific format
                checkpoint_files_in_folder = filter_files(
                    folder_files, network_name=self.network
                )
                if checkpoint_files_in_folder:
                    logger.info("Selected folder with checkpoint files: %s", folder_path)
                    return folder_path

        logger.info("No folder found containing required format files")
        return None

    def cancel_recommendation_jobs(self):
        """Cleanup recommendation jobs"""
        backend = os.getenv("TAO_EXECUTION_BACKEND", "local-k8s")
        all_children_stopped = True
        known_child_ids = set()
        for rec in self.recommendations:
            job_name = rec.job_id
            logger.info("Deleting %s", job_name)
            if not job_name:
                continue
            known_child_ids.add(job_name)
            if not os.getenv("CI_PROJECT_DIR", None):
                logger.info("Cancelling automl job at end of controller %s", job_name)
                cancel_result = cancel_automl_child_job(rec.job_id)
                if cancel_result is not True:
                    all_children_stopped = False
                    logger.warning(
                        "Could not confirm termination of AutoML child job %s; "
                        "retaining its checkpoint artifacts",
                        job_name,
                    )
            else:
                # A test/CI shortcut is not proof that a real writer is absent.
                all_children_stopped = False

            # Stop log monitoring for this experiment
            if backend in ("local-k8s", "local-docker"):
                try:
                    from nvidia_tao_core.microservices.utils.log_monitor_service import stop_monitoring_job
                    logger.info(f"[AUTOML-CONTROLLER] Stopping log monitoring for experiment job {job_name}")
                    stop_monitoring_job(job_name)
                    logger.info(f"[AUTOML-CONTROLLER] Stopped log monitoring for experiment job {job_name}")
                except Exception as e:
                    logger.warning(f"[AUTOML-CONTROLLER] Failed to stop log monitoring for {job_name}: {e}")

        # Include workflow children that were durably enqueued but were not yet
        # present in the controller's in-memory recommendation snapshot.
        for child_job_id in set(
            get_automl_child_job_ids(self.automl_context.id)
        ) - known_child_ids:
            if cancel_automl_child_job(child_job_id) is not True:
                all_children_stopped = False

        cleanup_complete = not self._checkpoint_cleanup_pending
        if self._checkpoint_cleanup_pending and all_children_stopped:
            try:
                # Child jobs have now been stopped, so bind-mounted artifacts are no longer
                # being written and platform ownership repair has had a chance to run.
                set_automl_checkpoint_cleanup_state(
                    self.automl_context.id, "cleaning"
                )
                cleanup_complete = self._cleanup_checkpoints_on_cancellation()
            except Exception as err:
                cleanup_complete = False
                logger.warning(
                    "Unexpected checkpoint cleanup failure for canceled AutoML job %s: %s",
                    self.automl_context.id, err
                )
                set_automl_checkpoint_cleanup_state(
                    self.automl_context.id, "pending", error=err
                )
            if cleanup_complete and set_automl_checkpoint_cleanup_state(
                self.automl_context.id, "complete"
            ):
                self._checkpoint_cleanup_pending = False
            else:
                cleanup_complete = False
                self._checkpoint_cleanup_pending = True
                set_automl_checkpoint_cleanup_state(
                    self.automl_context.id,
                    "pending",
                    error="one or more checkpoint artifacts could not be removed",
                )
        elif self._checkpoint_cleanup_pending:
            logger.warning(
                "Skipping checkpoint cleanup for AutoML job %s because one or "
                "more child writers were not confirmed stopped",
                self.automl_context.id,
            )

        if not all_children_stopped or not cleanup_complete:
            logger.warning(
                "Retaining AutoML controller %s so child cancellation and "
                "artifact cleanup can be retried",
                self.automl_context.id,
            )
            return False

        logger.info("Deleting automl job %s", self.automl_context.id)
        on_delete_automl_job(self.automl_context.id)
        return all_children_stopped

    def _schedule_checkpoint_cleanup(self):
        """Durably request cancellation cleanup before stopping any child."""
        self._checkpoint_cleanup_pending = True
        if not set_automl_checkpoint_cleanup_state(
            self.automl_context.id, "pending"
        ):
            logger.error(
                "Unable to persist checkpoint cleanup intent for AutoML job %s",
                self.automl_context.id,
            )
            return False
        return True

    def _finish_pending_checkpoint_cleanup(self):
        """Retry cancellation until all launchers and artifact deletes are quiescent."""
        while True:
            if self.cancel_recommendation_jobs() is not False:
                return True
            report_health_beat(
                self.automl_context.id,
                "Waiting for AutoML child quiescence and checkpoint cleanup retry",
            )
            time.sleep(5)

    def _cleanup_checkpoints_on_cancellation(self):
        """Delete checkpoint artifacts that cannot be used after AutoML cancellation.

        The controller can observe the parent job's canceled state before ``read_results``
        gets another opportunity to run its normal checkpoint cleanup. Preserve the best
        completed recommendation while removing artifacts from the remaining recommendations.
        Paused jobs never enter this terminal cleanup path, so their resume sources remain intact.
        """
        if not self.delete_intermediate_ckpt:
            logger.info(
                "Skipping checkpoint cleanup for canceled AutoML job %s because "
                "automl_delete_intermediate_ckpt is disabled",
                self.automl_context.id
            )
            return True

        try:
            self.refresh_recommendations()
        except Exception as err:
            logger.warning(
                "Unable to refresh recommendations before cancellation cleanup for AutoML job %s: %s",
                self.automl_context.id, err
            )
            return False

        successful_recs = [
            rec for rec in self.recommendations if rec.status == JobStates.success
        ]
        best_rec_id = None
        if successful_recs:
            try:
                best_rec_id = self.min_max(successful_recs, key=lambda rec: rec.result).id
            except Exception as err:
                # Failing closed here avoids deleting a potentially best checkpoint.
                logger.warning(
                    "Unable to identify the best recommendation during cancellation cleanup "
                    "for AutoML job %s; preserving completed recommendations: %s",
                    self.automl_context.id, err
                )

        # Pause deliberately skips terminal cleanup and is the supported resume
        # path. Once the parent is canceled, errored, or completed, promotion
        # sources are no longer live dependencies; retaining every successful
        # Hyperband/ASHA/PBT trial would recreate the disk-exhaustion issue.

        all_artifacts_cleaned = True
        for rec in self.recommendations:
            if not rec.job_id:
                continue

            try:
                expt_root = self._get_experiment_results_path(rec.job_id)
                if rec.id == best_rec_id:
                    # Retain the selected best checkpoint, but discard its intermediate saves.
                    if self.delete_checkpoint_files(
                        expt_root, rec, retain_resume_checkpoint=False
                    ) is False:
                        all_artifacts_cleaned = False
                elif best_rec_id is None and rec.status == JobStates.success:
                    # If best selection failed, preserve all successful recommendations.
                    logger.info(
                        "Preserving successful recommendation %s because the current best "
                        "could not be determined",
                        rec.id
                    )
                else:
                    self.delete_not_best_model_checkpoints(expt_root, rec, True)
                    if not getattr(
                        self, "_last_checkpoint_delete_verified", True
                    ):
                        all_artifacts_cleaned = False
            except Exception as err:
                all_artifacts_cleaned = False
                logger.warning(
                    "Failed to clean checkpoint artifacts for canceled AutoML recommendation %s "
                    "(job %s): %s",
                    rec.id, rec.job_id, err
                )
        return all_artifacts_cleaned

    def start(self):
        """Starts the automl controller"""
        try:
            # Report initial health beat
            report_health_beat(self.automl_context.id, "AutoML controller starting")

            # Set WandB group in specs for trial linking so all trials in dashboard are grouped together
            specs = get_job_specs(self.automl_context.id)
            if "wandb" in specs and specs["wandb"]:
                specs["wandb"]["group"] = self.wandb_group_name
                save_job_specs(self.automl_context.id, specs)
                logger.info("Updated WandB group in automl job specs: %s", self.wandb_group_name)

            # Initialize WandB for AutoML controller
            self._initialize_wandb_for_automl()

            update_job_message(
                self.automl_context.handler_id,
                self.automl_context.id,
                "experiments",
                "AutoML train started, more details in automl_brain_info of response"
            )
            self._execute_loop()
            result_metadata = get_handler_job_metadata(self.automl_context.id) or {}
            current_status = str(result_metadata.get("status", "")).lower()
            if self._checkpoint_cleanup_pending or current_status in ("canceled", "canceling"):
                # Preserve the externally managed cancellation state. Child shutdown also
                # performs the deferred, ownership-safe checkpoint cleanup.
                self._schedule_checkpoint_cleanup()
                self._finish_pending_checkpoint_cleanup()
                delete_health_beat(self.automl_context.id)
                return

            status = "Error"
            best_model_path = self._get_experiment_results_path(self.automl_context.id)
            result_metadata["job_details"][self.automl_context.id] = {
                "detailed_status": {
                    "message": f"Checkpoint file doesn't exist in best model folder {best_model_path}",
                    "status": "FAILURE"
                }
            }
            if self.best_model_copied:
                status = "Done"
                result_metadata["job_details"][self.automl_context.id] = {
                    "detailed_status": {
                        "message": (
                            f"AutoML run is successful with best checkpoints under {best_model_path}"
                        ),
                        "status": "SUCCESS"
                    }
                }
            elif self.best_rec_id >= 0:
                # Fallback: find_best_model failed to move (or match) but write_results
                # identified the best successful experiment. Verify checkpoints exist and
                # persist the mapping so downstream export/inference can resolve the path.
                best_rec = next(
                    (r for r in self.recommendations if r.id == self.best_rec_id),
                    None
                )
                has_checkpoints = False
                if best_rec is not None:
                    expt_folder = self._get_experiment_results_path(best_rec.job_id)
                    (find_trained_tlt, find_trained_hdf5, find_trained_pth, _,
                     find_trained_safetensors) = self.get_checkpoint_paths_matching_epoch_number(
                        expt_folder, self.best_rec_id
                    )
                    has_checkpoints = bool(
                        find_trained_tlt or find_trained_hdf5 or find_trained_pth or find_trained_safetensors
                    )
                if has_checkpoints:
                    save_automl_best_rec_info(
                        self.automl_context.id, best_rec.id, best_rec.job_id,
                        best_model_results_job_id=best_rec.job_id
                    )
                    best_specs = get_job_specs(
                        best_rec.job_id, automl=True,
                        automl_experiment_id=str(best_rec.id)
                    )
                    save_job_specs(
                        self.automl_context.id, specs=best_specs,
                        automl=True, automl_experiment_id="-1"
                    )
                    handler_metadata = get_handler_metadata(
                        self.automl_context.handler_id, "experiments"
                    )
                    best_epoch = self.best_epoch_number.get(self.best_rec_id, 0)
                    handler_metadata["checkpoint_epoch_number"][
                        f"best_model_{self.automl_context.id}"
                    ] = best_epoch
                    handler_metadata["checkpoint_epoch_number"][
                        f"latest_model_{self.automl_context.id}"
                    ] = best_epoch
                    write_handler_metadata(
                        self.automl_context.handler_id, handler_metadata, "experiments"
                    )
                    logger.info(
                        "Fallback: saved best_rec_info for experiment %s (job %s) "
                        "with checkpoint_epoch=%s",
                        best_rec.id, best_rec.job_id, best_epoch
                    )
                    status = "Done"
                    result_metadata["job_details"][self.automl_context.id] = {
                        "detailed_status": {
                            "message": (
                                "AutoML run completed; best model checkpoints remain in experiment "
                                "folder (copy to brain folder failed). Export/inference will use them."
                            ),
                            "status": "SUCCESS"
                        }
                    }
                else:
                    no_ckpt_path = expt_folder if best_rec is not None else best_model_path
                    result_metadata["job_details"][self.automl_context.id] = {
                        "detailed_status": {
                            "message": (
                                f"No valid checkpoint files in best experiment folder {no_ckpt_path}; "
                                "cannot use fallback."
                            ),
                            "status": "FAILURE"
                        }
                    }

            # Final pruning is a durable phase, separate from best-model
            # handoff. This covers parallel algorithms such as ASHA whose trial
            # checkpoints may not have been incrementally pruned.
            # Do not let the older metadata snapshot overwrite the fresh
            # cleanup handshake fields written below.
            for cleanup_key in (
                "checkpoint_cleanup_pending",
                "checkpoint_cleanup_status",
                "checkpoint_cleanup_error",
                "checkpoint_cleanup_updated_at",
            ):
                result_metadata.pop(cleanup_key, None)
            if not self._schedule_checkpoint_cleanup():
                raise RuntimeError(
                    "Unable to persist terminal AutoML checkpoint cleanup intent"
                )
            write_job_metadata(self.automl_context.id, result_metadata)
            update_job_status(self.automl_context.handler_id, self.automl_context.id, status=status, kind="experiments")
            self._finish_pending_checkpoint_cleanup()

            # Final WandB table update
            if self.wandb_initialized:
                self._update_wandb_table()
                try:
                    import wandb
                    wandb.finish()
                    logger.info("Closed WandB run for AutoML controller")
                except Exception as e:
                    logger.warning("Failed to close WandB run: %s", str(e))

            # Clean up health beat on completion
            delete_health_beat(self.automl_context.id)

        except Exception:
            logger.error(
                "AutoMLpipeline loop for network %s with job id %s failed due to exception %s",
                self.network, self.automl_context.id, traceback.format_exc()
            )
            result_metadata = get_handler_job_metadata(self.automl_context.id) or {}
            current_status = str(result_metadata.get("status", "")).lower()
            cancellation_in_progress = (
                self._checkpoint_cleanup_pending or current_status in ("canceled", "canceling")
            )
            if not cancellation_in_progress:
                result_metadata["job_details"][self.automl_context.id] = {
                    "detailed_status": {
                        "message": "AutoML train failed due to run-time exception",
                        "status": "FAILURE"
                    }
                }
                write_job_metadata(self.automl_context.id, result_metadata)

            # Abnormal exits bypass read_results just like cancellation exits. Schedule
            # eligible cleanup before stopping children; the deletion flag still controls it.
            self._schedule_checkpoint_cleanup()
            self._finish_pending_checkpoint_cleanup()
            if not cancellation_in_progress:
                update_job_status(
                    self.automl_context.handler_id,
                    self.automl_context.id,
                    status="Error",
                    kind="experiments"
                )

            # Clean up WandB on error
            if self.wandb_initialized:
                try:
                    import wandb
                    wandb.finish()
                except Exception as e:
                    logger.warning("Failed to close WandB run: %s", str(e))

            # Clean up health beat on error
            delete_health_beat(self.automl_context.id)

    def refresh_recommendations(self):
        """Refresh the recommendations"""
        self.recommendations = []
        recs_dict = get_automl_controller_info(self.automl_context.id)
        for rec_dict in recs_dict:
            rec = Recommendation(rec_dict["id"], rec_dict["specs"], self.metric_key)
            rec.update_result(rec_dict["result"])
            rec.update_status(rec_dict["status"])
            rec.assign_job_id(rec_dict["job_id"])
            # Restore PBT resume_from_job_id if present
            if "resume_from_job_id" in rec_dict:
                rec.resume_from_job_id = rec_dict["resume_from_job_id"]
            # Restore early_stop_epoch if present (for PBT/Hyperband metric trimming)
            if "early_stop_epoch" in rec_dict:
                rec.early_stop_epoch = rec_dict["early_stop_epoch"]
            self.recommendations.append(rec)

    def save_state(self):
        """Save the self.recommendations into automl brain DB"""
        recs_dict = [ele.__dict__ for ele in self.recommendations]
        metadata = get_handler_job_metadata(self.automl_context.id)
        current_status = metadata.get("status", "")
        logger.debug(
            f"[CONTROLLER-SAVE-STATE] About to save state: automl_job_id={self.automl_context.id}, "
            f"brain_status={current_status}, num_recs={len(recs_dict)}"
        )

        # Check if brain is being stopped
        if current_status in ("canceled", "canceling", "pausing", "paused"):
            logger.debug(
                f"[CONTROLLER-SAVE-STATE] Skipping save due to brain status: "
                f"automl_job_id={self.automl_context.id}, status={current_status}"
            )
            return

        # Check if any recommendations in MongoDB are already marked as canceled (race condition prevention)
        existing_recs = get_automl_controller_info(self.automl_context.id)
        if existing_recs:
            for existing_rec in existing_recs:
                if existing_rec.get("status") in ("canceled", "canceling"):
                    logger.debug(
                        f"[CONTROLLER-SAVE-STATE] Skipping save due to canceled recommendation in MongoDB: "
                        f"automl_job_id={self.automl_context.id}, rec_id={existing_rec.get('id')}, "
                        f"status={existing_rec.get('status')}"
                    )
                    return

            # Preserve backend_details from existing records (added by SLURM handler)
            # The Recommendation class doesn't have backend_details, so we need to merge it
            existing_by_id = {rec.get("id"): rec for rec in existing_recs}
            for rec in recs_dict:
                rec_id = rec.get("id")
                if rec_id in existing_by_id:
                    existing_rec = existing_by_id[rec_id]
                    if "backend_details" in existing_rec and "backend_details" not in rec:
                        rec["backend_details"] = existing_rec["backend_details"]
                        logger.debug(
                            f"[CONTROLLER-SAVE-STATE] Preserved backend_details for rec {rec_id}"
                        )

        save_automl_controller_info(self.automl_context.id, recs_dict)
        logger.debug(
            f"[CONTROLLER-SAVE-STATE] Saved state to MongoDB: automl_job_id={self.automl_context.id}, "
            f"num_recs={len(recs_dict)}"
        )

    @staticmethod
    def load_state(
        root,
        network,
        brain,
        automl_context,
        automl_algorithm_settings,
        delete_intermediate_ckpt,
        metric,
        automl_algorithm,
        decrypted_workspace_metadata,
        parameter_names=None
    ):
        """Loads a Controller object from pre-existing root"""
        logger.debug(
            f"[CONTROLLER-LOAD-STATE] Starting controller load_state: "
            f"automl_job_id={automl_context.id}, algorithm={automl_algorithm}"
        )
        ctrl = Controller(
            root,
            network,
            brain,
            automl_context,
            automl_algorithm_settings,
            delete_intermediate_ckpt,
            metric,
            automl_algorithm,
            decrypted_workspace_metadata,
            parameter_names
        )
        ctrl.recommendations = []
        # Restore the recommendations
        recs_dict = get_automl_controller_info(automl_context.id)
        logger.debug(
            f"[CONTROLLER-LOAD-STATE] Retrieved recommendations from controller info: "
            f"automl_job_id={automl_context.id}, num_recs={len(recs_dict) if recs_dict else 0}"
        )
        if recs_dict:
            logger.debug(
                f"[CONTROLLER-LOAD-STATE] Raw recommendations from MongoDB: "
                f"automl_job_id={automl_context.id}, "
                f"recs={[{'id': r['id'], 'status': r['status'], 'job_id': r['job_id']} for r in recs_dict]}"
            )

        for idx, rec_dict in enumerate(recs_dict):
            rec = Recommendation(rec_dict["id"], rec_dict["specs"], ctrl.metric_key)
            rec.update_result(rec_dict["result"])
            logger.debug(
                f"[CONTROLLER-LOAD-STATE] Before update_status: automl_job_id={automl_context.id}, "
                f"rec_id={rec_dict['id']}, rec.status={rec.status}, db_status={rec_dict['status']}"
            )
            rec.update_status(rec_dict["status"])
            logger.debug(
                f"[CONTROLLER-LOAD-STATE] After update_status: automl_job_id={automl_context.id}, "
                f"rec_id={rec_dict['id']}, rec.status={rec.status}"
            )
            rec.assign_job_id(rec_dict["job_id"])
            ctrl.recommendations.append(rec)
            ctrl.best_epoch_number[rec_dict["id"]] = (
                rec_dict.get("best_epoch_number") if rec_dict.get("best_epoch_number") else 0
            )
            logger.debug(
                f"[CONTROLLER-LOAD-STATE] Restored recommendation {idx}: automl_job_id={automl_context.id}, "
                f"rec_id={rec_dict['id']}, status={rec_dict['status']}, job_id={rec_dict['job_id']}"
            )

        # Handle temp_rec
        # temp_rec is a recommendation that started, but never ended
        # Usually, if the controller is stopped before a recommendation is done,
        # it might have to be started / resumed again
        temp_rec = get_automl_current_rec(automl_context.id)
        logger.debug(
            f"[CONTROLLER-LOAD-STATE] Retrieved current recommendation index: "
            f"automl_job_id={automl_context.id}, temp_rec={temp_rec}, "
            f"num_recommendations={len(ctrl.recommendations)}"
        )
        ctrl.save_state()

        if temp_rec is not None and temp_rec < len(ctrl.recommendations):
            rec_status = ctrl.recommendations[temp_rec].status
            logger.debug(
                f"[CONTROLLER-LOAD-STATE] Checking if should resume recommendation: "
                f"automl_job_id={automl_context.id}, temp_rec={temp_rec}, status={rec_status}, "
                f"is_canceled={rec_status == JobStates.canceled}"
            )
            if rec_status == JobStates.canceled:
                child_job_id = ctrl.recommendations[temp_rec].job_id
                if not reset_automl_child_lifecycle_for_resume(child_job_id):
                    raise RuntimeError(
                        "AutoML child cancellation fence could not be reset for "
                        f"paused recommendation {child_job_id}"
                    )
                logger.debug(
                    f"[CONTROLLER-LOAD-STATE] Resuming stopped automl sub-experiment: "
                    f"automl_job_id={automl_context.id}, rec_id={temp_rec}, "
                    f"job_id={ctrl.recommendations[temp_rec].job_id}"
                )
                if ctrl.automl_algorithm == "hyperband":
                    ctrl.brain.track_id = temp_rec
                    logger.debug(
                        f"[CONTROLLER-LOAD-STATE] Set Hyperband brain track_id: "
                        f"automl_job_id={automl_context.id}, track_id={temp_rec}"
                    )
                ctrl.on_new_automl_job(ctrl.recommendations[temp_rec])
                logger.debug(
                    f"[CONTROLLER-LOAD-STATE] Queued recommendation for resume: "
                    f"automl_job_id={automl_context.id}, rec_id={temp_rec}"
                )
            else:
                logger.debug(
                    f"[CONTROLLER-LOAD-STATE] NOT resuming recommendation (status is not canceled): "
                    f"automl_job_id={automl_context.id}, temp_rec={temp_rec}, status={rec_status}"
                )
        else:
            logger.warning(
                f"[CONTROLLER-LOAD-STATE] Invalid temp_rec or no recommendations to resume: "
                f"automl_job_id={automl_context.id}, temp_rec={temp_rec}, "
                f"num_recommendations={len(ctrl.recommendations)}"
            )

        logger.debug(
            f"[CONTROLLER-LOAD-STATE] Controller load_state completed: automl_job_id={automl_context.id}"
        )
        return ctrl

    def _execute_loop(self):
        """A loop that does the 3 things in order

        1.See if any new recommendation is up to execute
        2.Reads results of newly done experiments
        3.Writes AutoML status into a file which can be shown to the end user
        """
        update_job_status(self.automl_context.handler_id, self.automl_context.id, status="Running", kind="experiments")

        # Report health beat at start of loop
        report_health_beat(self.automl_context.id, "AutoML execute loop started")

        while True:
            # Report health beat on each iteration
            report_health_beat(
                self.automl_context.id,
                f"AutoML loop iteration (completed: {self.completed_recommendations}/"
                f"{self.automl_algorithm_settings.automl_max_recommendations})"
            )

            metadata = get_handler_job_metadata(self.automl_context.id) or {}
            current_status = str(metadata.get("status", "")).lower()
            automl_status = get_automl_controller_info(self.automl_context.id)
            if current_status in ("canceled", "canceling"):
                # Defer storage cleanup until cancel_recommendation_jobs has stopped every
                # child process. This avoids racing checkpoint writers and ownership repair.
                self._schedule_checkpoint_cleanup()
                return
            if automl_status:
                self.completed_recommendations = len(automl_status)
                logger.info(
                    f"Controller loop: completed_recommendations={self.completed_recommendations}, "
                    f"max_recommendations={self.automl_algorithm_settings.automl_max_recommendations}"
                )

                # Sequential algorithms: check max_recommendations
                sequential_algos = ("bayesian", "b", "bfbo")
                # Parallel algorithms with done() method
                parallel_algos = ("hyperband", "h", "bohb", "asha", "dehb", "hyperband_es", "hes", "pbt")

                # Check sequential condition
                max_rec_idx = self.automl_algorithm_settings.automl_max_recommendations - 1
                sequential_done = (
                    self.completed_recommendations == self.automl_algorithm_settings.automl_max_recommendations and
                    automl_status[max_rec_idx]['status'] in ('success', 'failure') and
                    self.automl_algorithm in sequential_algos
                )

                # Check parallel condition
                if self.automl_algorithm in parallel_algos:
                    brain_done = self.brain.done()
                    logger.info(
                        f"Controller: checking termination for parallel algo '{self.automl_algorithm}': "
                        f"brain.done()={brain_done}"
                    )
                    parallel_done = brain_done
                else:
                    parallel_done = False

                logger.info(f"Controller: sequential_done={sequential_done}, parallel_done={parallel_done}")

                if sequential_done or parallel_done:
                    # Find best model based on mAP
                    logger.info("Finding best model - TERMINATION TRIGGERED")
                    self.best_rec_id = self.find_best_model()
                    logger.info("best_model_copied result %s", self.best_model_copied)

                    if self.best_model_copied:
                        # Trial pruning happens after final status construction via
                        # the durable cleanup barrier in ``start``. Do not delete
                        # here: child writers and remote storage acknowledgements
                        # have not yet been joined into one terminal barrier.
                        handler_metadata = get_handler_metadata(
                            self.automl_context.handler_id, "experiments"
                        )
                        handler_metadata["checkpoint_epoch_number"][
                            f"best_model_{self.automl_context.id}"
                        ] = self.best_epoch_number[self.best_rec_id]
                        handler_metadata["checkpoint_epoch_number"][
                            f"latest_model_{self.automl_context.id}"
                        ] = self.best_epoch_number[self.best_rec_id]
                        write_handler_metadata(self.automl_context.handler_id, handler_metadata, "experiments")

                    self.eta = 0.0
                    self.remaining_epochs_in_experiment = 0.0
                    self.write_results(final=True)
                    return

            self.run_experiments()
            self.read_results()
            self.write_results()
            if not os.getenv("CI_PROJECT_DIR", None):
                time.sleep(10)

    def run_experiments(self):
        """Generate recommendation from brain

        if a new job is requested, add it to self.recommendations and execute it (add it to workflow)
        if a resume is requested, add the relevant recommendation to the workflow
        Supports parallel execution for algorithms like ASHA and PBT
        """
        report_health_beat(self.automl_context.id, "Running experiments")

        # Sequential algorithms (Bayesian, BFBO) are limited by max_recommendations
        sequential_algos = ("bayesian", "b", "bfbo")
        max_recs = self.automl_algorithm_settings.automl_max_recommendations
        if self.automl_algorithm in sequential_algos and len(self.recommendations) == max_recs:
            return

        # Check current running jobs for capacity management
        running_jobs = sum(
            1 for rec in self.recommendations
            if rec.status in [JobStates.pending, JobStates.started, JobStates.running]
        )

        # Get max concurrent limit based on algorithm
        max_concurrent = getattr(self.brain, 'max_concurrent', None)
        if max_concurrent is None:
            max_concurrent = getattr(self.brain, 'population_size', 1)

        # If we're at capacity, don't request more recommendations
        if running_jobs >= max_concurrent:
            logger.info(
                f"Controller: at capacity (running: {running_jobs}/{max_concurrent}), "
                f"not requesting new recommendations"
            )
            return

        logger.info(
            f"Controller: calling brain.generate_recommendations with history of "
            f"{len(self.recommendations)} experiments"
        )
        for i, rec in enumerate(self.recommendations):
            logger.info(
                f"  recommendations[{i}]: id={rec.id}, job_id={rec.job_id}, "
                f"status={rec.status}, result={rec.result}"
            )

        history = deepcopy(self.recommendations)
        recommended_specs = self.brain.generate_recommendations(history)

        # Support both single and multiple recommendations
        if not isinstance(recommended_specs, list):
            recommended_specs = [recommended_specs] if recommended_specs else []

        # Limit to available capacity
        available_slots = max_concurrent - running_jobs
        recommended_specs = recommended_specs[:available_slots]

        logger.info(
            f"Received {len(recommended_specs)} recommendation(s) for {self.network} "
            f"(running: {running_jobs}/{max_concurrent})"
        )

        for spec in recommended_specs:
            logger.info("Recommendation received for %s", self.network)
            if type(spec) is dict:
                # Save brain state and update current recommendation
                self.hyperband_cancel_condition_seen = False
                self.brain.save_state()
                if self.automl_algorithm in ("hyperband", "h", "bohb", "asha", "dehb", "hyperband_es", "hes", "pbt"):
                    self.automl_context.early_stop_epoch = self.brain.epoch_number
                # update temp_rec
                new_id = len(self.recommendations)
                self.best_epoch_number[new_id] = 0
                save_automl_current_rec(self.automl_context.id, new_id)

                # Run new recommendation
                rec = Recommendation(new_id, spec, self.metric_key)
                job_id = str(uuid.uuid4())  # Assign job_id for this recommendation
                rec.assign_job_id(job_id)
                logger.debug(f"[AUTOML-CONTROLLER] Assigned job_id: {job_id} to recommendation: {rec.id}")

                # Store early_stop_epoch on the recommendation for later metric trimming
                if self.automl_algorithm in ("hyperband", "h", "bohb", "asha", "dehb", "hyperband_es", "hes", "pbt"):
                    rec.early_stop_epoch = self.brain.epoch_number
                    logger.debug(
                        f"[AUTOML-CONTROLLER] Set early_stop_epoch for new recommendation: "
                        f"rec_id={rec.id}, early_stop_epoch={self.brain.epoch_number}"
                    )

                self.recommendations.append(rec)
                self.save_state()

                # Inject TAO_AUTOML_TRIGGERED flag into experiment docker_env_vars
                self._inject_automl_env_vars()

                self.on_new_automl_job(rec)

            elif type(spec) is ResumeRecommendation:
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Resume recommendation received: "
                    f"automl_job_id={self.automl_context.id}, rec_id={spec.id}, job_id={spec.job_id}"
                )
                self.hyperband_cancel_condition_seen = False
                rec_id = spec.id
                self.best_epoch_number[rec_id] = 0
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Reset best epoch number for recommendation: "
                    f"automl_job_id={self.automl_context.id}, rec_id={rec_id}"
                )

                # Save brain state and update current recommendation
                self.brain.save_state()
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Saved brain state: "
                    f"automl_job_id={self.automl_context.id}"
                )
                if self.automl_algorithm in ("hyperband", "h", "bohb", "asha", "dehb", "hyperband_es", "hes", "pbt"):
                    self.automl_context.early_stop_epoch = self.brain.epoch_number
                    # Store early_stop_epoch on the recommendation for later metric trimming
                    self.recommendations[rec_id].early_stop_epoch = self.brain.epoch_number
                    logger.debug(
                        f"[AUTOML-CONTROLLER-RESUME] Set early_stop_epoch for Hyperband: "
                        f"automl_job_id={self.automl_context.id}, rec_id={rec_id}, "
                        f"early_stop_epoch={self.brain.epoch_number}"
                    )

                # update temp_rec
                save_automl_current_rec(self.automl_context.id, rec_id)
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Updated current recommendation: "
                    f"automl_job_id={self.automl_context.id}, rec_id={rec_id}"
                )

                assert (self.recommendations[rec_id].id == rec_id), (
                    f"Recommendation ID mismatch: expected {rec_id} but got "
                    f"{self.recommendations[rec_id].id}"
                )

                # Store resume_from_job_id directly from ResumeRecommendation (PBT)
                self.recommendations[rec_id].resume_from_job_id = spec.resume_from_job_id
                if spec.resume_from_job_id:
                    logger.info(
                        f"[AUTOML-CONTROLLER-RESUME] PBT: Recommendation {rec_id} will resume "
                        f"from checkpoint of job {spec.resume_from_job_id}"
                    )

                self.recommendations[rec_id].specs = spec.specs.copy()
                self.recommendations[rec_id].update_status(JobStates.pending)
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Updated recommendation specs and status to pending: "
                    f"automl_job_id={self.automl_context.id}, rec_id={rec_id}"
                )

                # Save updated specs to MongoDB
                save_job_specs(
                    self.automl_context.id,
                    self.recommendations[rec_id].specs,
                    automl=True,
                    automl_experiment_id=str(rec_id)
                )

                # Remove previous files (except checkpoints) from experiment folder.
                def remove_files(local_expt_path, cloud_expt_path, rec_id):
                    logger.debug(
                        f"[AUTOML-CONTROLLER-RESUME] Cleaning up previous experiment files: "
                        f"automl_job_id={self.automl_context.id}, rec_id={rec_id}, "
                        f"local_path={local_expt_path}, cloud_path={cloud_expt_path}"
                    )
                    expt_file_name = get_file_list_from_cloud_storage(
                        self.decrypted_workspace_metadata, cloud_expt_path)
                    regex_pattern = r'.*(?:lightning_logs|events).*$|.*\.(json)$'
                    expt_file_name = filter_files(expt_file_name, regex_pattern)
                    logger.debug(
                        f"[AUTOML-CONTROLLER-RESUME] Found {len(expt_file_name)} files to delete from cloud: "
                        f"automl_job_id={self.automl_context.id}, rec_id={rec_id}"
                    )
                    for file_name in expt_file_name:
                        self.cs_instance.delete_file(file_name)
                    if os.path.exists(local_expt_path):
                        expt_file_name = glob.glob(local_expt_path + "/**/*.txt", recursive=True)
                        # Filter out microservices_log.txt - we want to keep logs from log streaming
                        expt_file_name = [f for f in expt_file_name if not f.endswith('microservices_log.txt')]
                        logger.debug(
                            f"[AUTOML-CONTROLLER-RESUME] Removing {len(expt_file_name)} log files "
                            f"(excluding microservices_log.txt): "
                            f"automl_job_id={self.automl_context.id}, rec_id={rec_id}, files={expt_file_name}"
                        )
                        for file_name in expt_file_name:
                            if os.path.isfile(file_name):
                                os.remove(file_name)
                    else:
                        logger.debug(
                            f"[AUTOML-CONTROLLER-RESUME] Local experiment path does not exist: "
                            f"automl_job_id={self.automl_context.id}, rec_id={rec_id}, path={local_expt_path}"
                        )
                    delete_dnn_status(self.automl_context.id, automl=True, experiment_number=str(rec_id))
                    logger.debug(
                        f"[AUTOML-CONTROLLER-RESUME] Deleted DNN status for experiment: "
                        f"automl_job_id={self.automl_context.id}, rec_id={rec_id}"
                    )

                expt_name = "experiment_" + str(rec_id)
                cloud_expt_path = self._get_experiment_results_path(self.recommendations[rec_id].job_id)
                remove_files(
                    os.path.join(self.root, expt_name),
                    cloud_expt_path,
                    rec_id
                )
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Completed cleanup for resumed experiment: "
                    f"automl_job_id={self.automl_context.id}, rec_id={rec_id}"
                )

                self.save_state()
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Saved controller state: "
                    f"automl_job_id={self.automl_context.id}"
                )

                # Inject TAO_AUTOML_TRIGGERED flag into experiment docker_env_vars
                self._inject_automl_env_vars()
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Injected AutoML environment variables: "
                    f"automl_job_id={self.automl_context.id}"
                )

                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Queueing resumed recommendation job: "
                    f"automl_job_id={self.automl_context.id}, rec_id={rec_id}, "
                    f"rec_job_id={self.recommendations[rec_id].job_id}"
                )
                # Multi-fidelity promotions intentionally reuse the same child
                # job ID and checkpoint directory. read_results fenced and
                # quiesced the prior writer before inspecting its checkpoint;
                # explicitly re-arm that identity for this one promoted launch.
                # A concurrent parent Canceling/Paused state is still checked by
                # AutoMLPipeline before and after its atomic launch claim.
                child_job_id = self.recommendations[rec_id].job_id
                if not reset_automl_child_lifecycle_for_resume(child_job_id):
                    raise RuntimeError(
                        "AutoML child cancellation fence could not be reset for "
                        f"promoted recommendation {child_job_id}"
                    )
                self.on_new_automl_job(self.recommendations[rec_id])
                logger.debug(
                    f"[AUTOML-CONTROLLER-RESUME] Resume recommendation processing completed: "
                    f"automl_job_id={self.automl_context.id}, rec_id={rec_id}"
                )

    def read_results(self):
        """Update results for each recommendation"""
        report_health_beat(self.automl_context.id, "Reading results")

        flag = False
        self.refresh_recommendations()
        for rec in self.recommendations:
            child_quiescent = True
            # Report health beat for each experiment being processed
            report_health_beat(
                self.automl_context.id,
                f"Processing experiment {rec.id} (status: {rec.status})"
            )

            old_status = rec.status
            logger.info(f"old_status: {old_status}")
            logger.info(f"rec: {rec.status}")

            job_name = rec.job_id
            if not job_name:
                continue

            expt_name = "experiment_" + str(rec.id)
            local_expt_root = os.path.join(self.root, expt_name)
            cloud_expt_root = self._get_experiment_results_path(rec.job_id)

            # If rec already changed to Success, no need to check
            if rec.status in [JobStates.success, JobStates.failure]:
                child_quiescent = cancel_automl_child_job(rec.job_id) is True
                if not child_quiescent:
                    logger.warning(
                        "Deferring checkpoint cleanup for completed AutoML child %s "
                        "until backend quiescence is confirmed",
                        rec.job_id,
                    )
                    continue
                # Apply penalty for failed experiments if result is still 0.0
                if rec.status == JobStates.failure and rec.result == 0.0:
                    if self.brain.reverse_sort:
                        penalty_value = 1e-7  # Low penalty for metrics where higher is better
                    else:
                        penalty_value = 1e7  # High penalty for metrics where lower is better
                    logger.warning(
                        f"AutoML experiment {rec.id} (job {rec.job_id}) failed with result 0.0. "
                        f"Assigning penalty value {penalty_value} based on metric '{self.metric_key}' "
                        f"(reverse_sort={self.brain.reverse_sort}) to enable AutoML optimization to continue."
                    )
                    rec.update_result(penalty_value)
                    self.save_state()

                if self.delete_intermediate_ckpt:
                    self.delete_checkpoint_files(
                        cloud_expt_root, rec, retain_resume_checkpoint=True
                    )
                    # Remove the checkpoints from not best model
                    brain_dict = get_automl_brain_info(self.automl_context.id)
                    if brain_dict:
                        # Checkpoint deletion strategy by algorithm:
                        # - Bayesian/BFBO: Delete immediately (experiments are independent)
                        # - Hyperband/BOHB/DEHB: Delete when bracket changes AND all configs complete
                        #   (prevents deletion before resume in next rung)
                        # - PBT: NO intermediate deletion (ID reuse means checkpoints naturally overwrite)
                        can_delete_checkpoints = False

                        if self.automl_algorithm in ("bayesian", "b", "bfbo"):
                            # Non-resuming algorithms: safe to delete as soon as we know it's not best
                            can_delete_checkpoints = True
                            logger.debug(
                                f"AutoML ({self.automl_algorithm}): Non-resuming algorithm, "
                                f"safe to delete non-best checkpoints"
                            )
                        elif self.automl_algorithm in (
                            "hyperband", "h", "bohb", "asha", "dehb",
                            "hyperband_es", "hes"
                        ):
                            # Hyperband-based: Only delete when bracket changes AND all configs complete
                            # This ensures configs needed for next rung's resume are preserved
                            if self.old_bracket != brain_dict.get("bracket", "0"):
                                all_complete = all(
                                    r.status in [JobStates.success, JobStates.failure, JobStates.canceled]
                                    for r in self.recommendations
                                )
                                if all_complete:
                                    can_delete_checkpoints = True
                                    logger.info(
                                        f"AutoML ({self.automl_algorithm}): All configs complete in bracket "
                                        f"{self.old_bracket}, safe to delete non-best checkpoints"
                                    )
                                else:
                                    logger.info(
                                        f"AutoML ({self.automl_algorithm}): Bracket changed but not all configs "
                                        f"complete yet, deferring checkpoint deletion to preserve resume capability"
                                    )
                        if can_delete_checkpoints:
                            flag = self.delete_not_best_model_checkpoints(cloud_expt_root, rec, flag)
                continue

            status_parser = StatusParser(self.network, local_expt_root, self.first_epoch_number)

            # Pass workspace_metadata and cloud_results_dir so StatusParser can sync SLURM status
            new_results = status_parser.update_results(
                experiment_number=str(rec.id),
                automl=True,
                job_id=self.automl_context.id,
                rec_job_id=rec.job_id,
                workspace_metadata=self.decrypted_workspace_metadata,
                cloud_results_dir=cloud_expt_root
            )
            logger.info(f"new_results 0: {new_results}")
            report_health_beat(self.automl_context.id, f"Recieved updated results for experiment {rec.id}")
            self.calculate_eta(new_results, rec.job_id, rec.id)
            metadata = get_handler_job_metadata(self.automl_context.id)
            results = metadata.get("job_details", {})
            brain_dict = get_automl_brain_info(self.automl_context.id)
            self.brain_epoch_number = float(brain_dict.get("epoch_number", float('inf')))
            # Calculate last_seen_epoch and ensure it's non-negative
            last_seen_epoch_value = max(0, self.total_epochs - self.remaining_epochs_in_experiment)
            new_results = status_parser.update_results(
                experiment_number=str(rec.id),
                total_epochs=self.total_epochs,
                eta=self.eta,
                last_seen_epoch=last_seen_epoch_value,
                automl=True,
                job_id=self.automl_context.id,
                previous_result_metadata=results,
                automl_brain=True
            )
            logger.info(f"new_results 1: {new_results}")
            new_results = status_parser.update_results(
                experiment_number=str(rec.id),
                total_epochs=self.total_epochs,
                last_seen_epoch=last_seen_epoch_value,
                automl=True,
                job_id=self.automl_context.id,
                rec_job_id=rec.job_id,
                previous_result_metadata=results
            )
            logger.info(f"new_results 2: {new_results}")
            if status_parser.first_epoch_number != -1:
                self.first_epoch_number = status_parser.first_epoch_number
            detailed_status_message = (
                metadata.get("job_details", {})
                .get(self.automl_context.id, {})
                .get("detailed_status", {})
                .get("message", "")
            )
            if "Invalid schema" not in detailed_status_message:
                update_job_metadata(
                    self.automl_context.handler_id,
                    self.automl_context.id,
                    metadata_key="job_details",
                    data=new_results,
                    kind="experiments"
                )

            # Status is read from the status DB and not from K8s
            status = ""
            if rec.status == JobStates.success:
                status = JobStates.success
            elif new_results[rec.job_id].get("detailed_status"):
                status = new_results[rec.job_id]["detailed_status"].get("status", JobStates.pending).lower()
                logger.info(f"status updated from new_results: {status}")
            if not status:
                status = JobStates.pending
            if status in [JobStates.success, JobStates.failure]:
                logger.info("Post processing of job %s under automl algorithm %s", rec.job_id, self.automl_algorithm)
                brain_epoch_number = self.brain_epoch_number
                if self.automl_algorithm in ("bayesian", "b", "bfbo"):
                    self.brain.num_epochs_per_experiment = get_total_epochs(
                        self.automl_context,
                        os.path.dirname(self.root),
                    )
                    brain_epoch_number = self.brain.num_epochs_per_experiment
                elif self.automl_algorithm in ("hyperband", "h", "bohb", "asha", "dehb", "hyperband_es", "hes", "pbt"):
                    # For resuming algorithms, use the early_stop_epoch that was set when this job launched
                    # (not the current brain.epoch_number which points to the NEXT generation/rung)
                    if rec.early_stop_epoch is not None:
                        brain_epoch_number = rec.early_stop_epoch
                        logger.info(
                            f"AutoML ({self.automl_algorithm}): Using early_stop_epoch={rec.early_stop_epoch} "
                            f"for experiment {rec.id} (job {rec.job_id}), not current brain.epoch_number="
                            f"{self.brain_epoch_number}"
                        )
                report_health_beat(
                    self.automl_context.id,
                    f"Reading final metrics for experiment {rec.id}"
                )
                validation_map, self.best_epoch_number[rec.id], _ = status_parser.read_metric(
                    results=new_results[rec.job_id],
                    metric=self.metric,
                    automl_algorithm=self.automl_algorithm,
                    automl_brain_job_id=self.automl_context.id,
                    brain_epoch_number=brain_epoch_number
                )
                if status == JobStates.failure:
                    if self.brain.reverse_sort:
                        validation_map = 1e-7
                    else:
                        validation_map = 1e7
                    logger.warning(
                        f"AutoML experiment {rec.id} (job {rec.job_id}) failed. "
                        f"Assigning penalty value {validation_map} to enable Bayesian optimization to continue."
                    )
                rec.update_result(validation_map)
                self.save_state()

                # Enhanced logging for job cancellation with full context
                logger.debug(
                    f"{'-' * 80}\n"
                    f"AUTOML CONTROLLER: CANCELLING EXPERIMENT JOB\n"
                    f"Brain Job ID: {self.automl_context.id}\n"
                    f"Experiment ID: {rec.id}\n"
                    f"Experiment Job ID: {rec.job_id}\n"
                    f"Final Status: {status}\n"
                    f"Final Result: {validation_map}\n"
                    f"Reason: Experiment completed with status={status}\n"
                    f"Action: Fencing and deleting the child backend\n"
                    f"{'-' * 80}"
                )

                report_health_beat(
                    self.automl_context.id,
                    f"Cancelling completed job {rec.job_id} (experiment {rec.id})"
                )
                child_quiescent = cancel_automl_child_job(rec.job_id) is True
                if not child_quiescent:
                    logger.warning(
                        "AutoML child %s reached %s but backend quiescence is "
                        "not yet confirmed; checkpoint cleanup will retry",
                        rec.job_id, status,
                    )
            if old_status != status:
                rec.update_status(status)
                self.save_state()
                if status == JobStates.success:
                    container_log_file = f"{self.root}/{rec.job_id}/log.txt"
                    if os.path.exists(container_log_file):
                        with open(container_log_file, "a", encoding='utf-8') as f:
                            f.write("\nEOF\n")

            if (
                rec.status in [JobStates.success, JobStates.failure] and
                self.delete_intermediate_ckpt and
                child_quiescent
            ):
                # Retain the latest checkpoint and remove others in experiment folder
                self.delete_checkpoint_files(
                    cloud_expt_root, rec, retain_resume_checkpoint=True
                )

        if self.automl_algorithm in ("hyperband", "h", "bohb", "dehb", "hyperband_es", "hes"):
            brain_dict = get_automl_brain_info(self.automl_context.id)
            if brain_dict:
                self.old_bracket = brain_dict.get("bracket", "0")

    def calculate_eta(self, new_results, rec_job_id, rec_id):
        """Calculate estimated time remaining for automl job"""
        report_health_beat(self.automl_context.id, f"Calculating ETA for experiment {rec_id}")

        global time_per_epoch  # pylint: disable=global-statement
        global time_per_epoch_counter  # pylint: disable=global-statement
        self.total_epochs = 0
        if self.automl_algorithm in ("bayesian", "b", "bfbo"):
            max_recs = self.automl_algorithm_settings.automl_max_recommendations
            self.total_epochs = max_recs * self.brain.num_epochs_per_experiment
        elif self.automl_algorithm == "pbt":
            # PBT: population_size * max_generations * eval_interval
            population_size = getattr(self.brain, 'population_size', 10)
            max_generations = getattr(self.brain, 'max_generations', 20)
            eval_interval = getattr(self.brain, 'eval_interval', 10)
            self.total_epochs = population_size * max_generations * eval_interval
        elif self.automl_algorithm in ("hyperband", "h", "bohb", "asha", "dehb", "hyperband_es", "hes"):
            for key in self.brain.ni:
                experiments = self.brain.ni[key]
                epochs = self.brain.ri[key]
                for i, num_epochs in enumerate(epochs):
                    if i == 0:
                        self.total_epochs += experiments[i] * num_epochs
                    else:
                        self.total_epochs += experiments[i] * (epochs[i] - epochs[i - 1])
            self.total_epochs *= self.brain.epoch_multiplier

        for result_key in new_results.get(rec_job_id, {}).keys():
            if result_key in ("epoch", "cur_iter") and new_results[rec_job_id].get(result_key) is not None:
                current_epoch = new_results[rec_job_id].get(result_key)
                if result_key == "cur_iter":
                    time_per_key = "time_per_iter"
                else:
                    time_per_key = "time_per_epoch"
                time_per_epoch_string = new_results[rec_job_id].get(time_per_key, "0:0:0.0")
                if time_per_epoch_string:
                    format_time_per_epoch = time.strptime(time_per_epoch_string.split(".")[0], '%H:%M:%S')
                    time_per_epoch += (
                        format_time_per_epoch.tm_hour * 60 * 60 +
                        format_time_per_epoch.tm_min * 60 +
                        format_time_per_epoch.tm_sec
                    )
                else:
                    time_per_epoch = 0
                time_per_epoch_counter += 1
                self.average_time_per_epoch = time_per_epoch / time_per_epoch_counter

                if self.automl_algorithm in ("bayesian", "b", "bfbo"):
                    current_experiment_epoch = get_total_epochs(
                        rec_job_id,
                        os.path.dirname(self.root),
                        automl=True,
                        automl_experiment_id=str(rec_id)
                    )
                    remaining_epochs = current_experiment_epoch - current_epoch
                    self.remaining_epochs_in_experiment = (
                        remaining_epochs +
                        (self.automl_algorithm_settings.automl_max_recommendations - self.completed_recommendations) *
                        (self.brain.num_epochs_per_experiment)
                    )
                    self.eta = self.remaining_epochs_in_experiment * self.average_time_per_epoch

                elif self.automl_algorithm == "pbt":
                    # PBT: Estimate based on population progress
                    current_generation = getattr(self.brain, 'generation', 0)
                    max_generations = getattr(self.brain, 'max_generations', 20)
                    eval_interval = getattr(self.brain, 'eval_interval', 10)
                    population_size = getattr(self.brain, 'population_size', 10)

                    remaining_generations = max(0, max_generations - current_generation)
                    self.remaining_epochs_in_experiment = (
                        remaining_generations * eval_interval * population_size +
                        (eval_interval - current_epoch % eval_interval if current_epoch % eval_interval != 0 else 0)
                    )
                    self.eta = self.remaining_epochs_in_experiment * self.average_time_per_epoch

                elif self.automl_algorithm in ("hyperband", "h", "bohb", "asha", "dehb", "hyperband_es", "hes"):
                    # Calculate completed epochs for completed sh sessions
                    completed_epochs = 0
                    # Only iterate through brackets that actually exist in ni
                    for bracket_str in sorted(self.brain.ni.keys(), key=int):
                        bracket = int(bracket_str)
                        # Skip future brackets beyond current
                        if bracket > int(self.brain.bracket):
                            continue
                        local_sh_iter = len(self.brain.ni[bracket_str])
                        if bracket == int(self.brain.bracket):
                            local_sh_iter = self.brain.sh_iter
                        for sh in range(0, local_sh_iter):
                            if (sh == 0):
                                completed_epochs += (
                                    self.brain.ni[bracket_str][sh] *
                                    self.brain.ri[bracket_str][sh]
                                )
                            else:
                                completed_epochs += (
                                    self.brain.ni[bracket_str][sh] *
                                    (self.brain.ri[bracket_str][sh] -
                                     self.brain.ri[bracket_str][sh - 1])
                                )

                    # Calculate completed epochs for current sh session
                    current_sh_allowed_epochs = 0
                    # Only calculate if current bracket exists in ri
                    if self.brain.bracket in self.brain.ri:
                        if self.brain.sh_iter < len(self.brain.ri[self.brain.bracket]):
                            current_sh_allowed_epochs = (
                                self.brain.ri[self.brain.bracket][self.brain.sh_iter] *
                                self.brain.epoch_multiplier
                            )
                            if self.brain.sh_iter > 0:
                                current_sh_allowed_epochs = (
                                    (self.brain.ri[self.brain.bracket][self.brain.sh_iter] -
                                     self.brain.ri[self.brain.bracket][self.brain.sh_iter - 1]) *
                                    self.brain.epoch_multiplier
                                )
                            completed_epochs += self.brain.expt_iter * current_sh_allowed_epochs

                    self.remaining_epochs_in_experiment = max(0, self.total_epochs - completed_epochs)
                    self.eta = self.remaining_epochs_in_experiment * self.average_time_per_epoch

        if self.remaining_epochs_in_experiment == float("inf") or self.remaining_epochs_in_experiment == float("-inf"):
            self.remaining_epochs_in_experiment = self.total_epochs

    def write_results(self, final=False):
        """Update stats value and write to job metadata"""
        report_health_beat(self.automl_context.id, "Writing results" if not final else "Writing final results")

        # Best mAP seen till now
        result_dict = {}
        try:
            # Filter recommendations to only those with completed status
            valid_recs = [r for r in self.recommendations
                          if r.status in (JobStates.success, JobStates.failure)]

            if valid_recs:
                best_metric_value = self.min_max(valid_recs, key=lambda rec: rec.result).result
            else:
                best_metric_value = 0.0

            result_dict[f"best_{self.metric_key}"] = best_metric_value
        except Exception as e:
            logger.error("Exception thrown in write_results is %s", str(e))
            result_dict[f"best_{self.metric_key}"] = 0.0

        if type(self.eta) is float:
            self.eta = str(timedelta(seconds=self.eta))
        result_dict["Estimated time for automl completion"] = str(self.eta)
        result_dict["Current experiment id"] = len(self.recommendations)

        if self.network in _ITER_MODELS:
            result_dict["Number of iters yet to start"] = self.remaining_epochs_in_experiment
            result_dict["Time per iter in seconds"] = round(self.average_time_per_epoch, 2)
        else:
            result_dict["Number of epochs yet to start"] = self.remaining_epochs_in_experiment
            result_dict["Time per epoch in seconds"] = round(self.average_time_per_epoch, 2)

        # Only consider successful recommendations (not failures with penalties)
        completed_recs = [
            rec for rec in self.recommendations
            if rec.status == JobStates.success
        ]
        if completed_recs:
            try:
                best_rec = self.min_max(completed_recs, key=lambda rec: rec.result)
                self.best_rec_id = best_rec.id
                logger.debug(
                    f"Updated best_rec_id to {self.best_rec_id} with "
                    f"{self.metric_key}={best_rec.result}"
                )
            except Exception as e:
                logger.error("Exception while updating best_rec_id: %s", str(e))

        # Add best experiment id (always, not just at the end)
        if self.best_rec_id != -1:
            result_dict["Best experiment id"] = self.best_rec_id

        update_automl_stats(self.automl_context.id, result_dict)

        # Update WandB table with current state
        self._update_wandb_table()

    def find_best_model(self):
        """Find best model based on metric value chosen and move those artifacts to best_model folder"""
        report_health_beat(self.automl_context.id, "Finding best model")

        logger.info("Finding best recommendation config")
        try:
            successful_recs = [
                r for r in self.recommendations if r.status == JobStates.success
            ]
            if not successful_recs:
                logger.warning("No successful experiments found for best model selection")
                return -1
            best_mAP = self.min_max(successful_recs, key=lambda rec: rec.result).result
        except Exception as e:
            logger.error("Exception thrown in find_best_model is %s", str(e))
            best_mAP = 0.0
            return -1

        logger.info("Best metric value %s", best_mAP)
        for rec in self.recommendations:
            # Report health beat while processing each recommendation
            report_health_beat(self.automl_context.id, f"Checking experiment {rec.id} for best model")
            logger.info("\nRecommendation in function find_best_model %s", rec)
            job_name = rec.job_id
            if not job_name:
                continue
            expt_folder = self._get_experiment_results_path(rec.job_id)
            checkpoint_files = get_file_list_from_cloud_storage(self.decrypted_workspace_metadata, expt_folder)

            # Use network config for filtering if available
            if self._uses_folder_lookup():
                checkpoint_files = filter_files(checkpoint_files, network_name=self.network)
                # For folder lookup, we want to find the folder containing checkpoints
                if checkpoint_files:
                    # Get unique folder paths
                    checkpoint_folders = list(set(os.path.dirname(f) for f in checkpoint_files))
                    checkpoint_files = checkpoint_folders
            else:
                regex_pattern = r'^(?!.*lightning_logs).*\.(pth|tlt|hdf5)$'
                checkpoint_files = filter_files(checkpoint_files, regex_pattern)
            logger.info("Experiment folder %s", expt_folder)
            logger.info("Checkpoints in find best_model %s", checkpoint_files)

            if checkpoint_files and (rec.status == JobStates.success and rec.result == best_mAP):
                cloud_best_model_folder = self._get_experiment_results_path(self.automl_context.id)
                logger.info("cloud_best_model_folder %s chosen for rec %s", cloud_best_model_folder, rec.id)

                # Clean up invalid checkpoint folders before moving
                self.delete_checkpoint_files(expt_folder, rec, filter_by_format=True)

                report_health_beat(
                    self.automl_context.id,
                    f"Moving best model folder for experiment {rec.id} to {cloud_best_model_folder}"
                )
                best_specs = get_job_specs(job_name, automl=True, automl_experiment_id=str(rec.id))
                exclude_log_files = ['microservices_log.txt', 'log.txt']
                logger.info(f"Moving best experiment folder excluding log files: {exclude_log_files}")
                try:
                    self.cs_instance.move_folder(
                        expt_folder[1:],
                        cloud_best_model_folder,
                        job_id=self.automl_context.id,
                        exclude_files=exclude_log_files
                    )
                    report_health_beat(
                        self.automl_context.id,
                        f"Completed moving best model folder for experiment {rec.id}"
                    )
                    save_automl_best_rec_info(self.automl_context.id, rec.id, rec.job_id)
                    save_job_specs(self.automl_context.id, specs=best_specs, automl=True, automl_experiment_id="-1")
                    (find_trained_tlt,
                     find_trained_hdf5,
                     find_trained_pth,
                     _,
                     find_trained_safetensors) = self.get_checkpoint_paths_matching_epoch_number(
                        cloud_best_model_folder,
                        rec.id
                    )
                    if find_trained_tlt or find_trained_hdf5 or find_trained_pth or find_trained_safetensors:
                        self.best_model_copied = True
                        return rec.id
                    logger.info("Best model checkpoints couldn't be moved")
                    return -1
                except Exception as move_err:
                    logger.warning(
                        "Move of best experiment folder to brain folder failed (%s); "
                        "storing best_model_results_job_id=experiment so downstream use experiment folder: %s",
                        rec.job_id, move_err
                    )
                    save_automl_best_rec_info(
                        self.automl_context.id, rec.id, rec.job_id,
                        best_model_results_job_id=rec.job_id
                    )
                    save_job_specs(self.automl_context.id, specs=best_specs, automl=True, automl_experiment_id="-1")
                    (find_trained_tlt,
                     find_trained_hdf5,
                     find_trained_pth,
                     _,
                     find_trained_safetensors) = self.get_checkpoint_paths_matching_epoch_number(
                        expt_folder, rec.id
                    )
                    if find_trained_tlt or find_trained_hdf5 or find_trained_pth or find_trained_safetensors:
                        self.best_model_copied = False
                        return rec.id
                    logger.warning("Best model checkpoints not found in experiment folder after move failure")
                    return -1
        return -1

    def get_checkpoint_paths_matching_epoch_number(
        self, path, rec_id, epoch_number=None
    ):
        """Get checkpoints from cloud_path and filter based on epoch number

        For networks with folder-based checkpoints (like cosmos-rl), this will find
        folders for different formats (.pth, .safetensors) separately.
        """
        checkpoint_files = get_file_list_from_cloud_storage(self.decrypted_workspace_metadata, path)
        selected_epoch = (
            self.best_epoch_number[rec_id]
            if epoch_number is None else epoch_number
        )
        format_epoch_number = format_epoch(self.network, selected_epoch)
        if self._uses_folder_lookup():
            # For folder lookup, look for epoch-numbered folders (e.g., epoch_1/, epoch_2/)
            if self.network in MISSING_EPOCH_FORMAT_NETWORKS:
                folder_pattern = fr".*{self.checkpoint_delimiter}{format_epoch_number}/"
            else:
                folder_pattern = fr".*epoch_{format_epoch_number}(?:_step_\d+)?/"

            # Find files inside the epoch folder to identify the folder exists
            epoch_folder_files = [f for f in checkpoint_files if re.match(folder_pattern, f)]

            if epoch_folder_files:
                # For cosmos-rl and similar networks, try to find both safetensors and pth folders
                safetensors_folder = self._select_best_epoch_folder(
                    epoch_folder_files, checkpoint_files, specific_format="safetensors"
                )
                pth_folder = self._select_best_epoch_folder(
                    epoch_folder_files, checkpoint_files, specific_format="pth"
                )

                # If specific formats not found, fall back to default format
                if not safetensors_folder and not pth_folder:
                    selected_folder = self._select_best_epoch_folder(
                        epoch_folder_files, checkpoint_files
                    )
                    if selected_folder:
                        return (
                            [selected_folder], [selected_folder], [selected_folder],
                            [selected_folder], [selected_folder]
                        )
                    return [], [], [], [], []

                # Return format-specific folders
                return (
                    [safetensors_folder] if safetensors_folder else [],  # tlt
                    [safetensors_folder] if safetensors_folder else [],  # hdf5
                    [pth_folder] if pth_folder else [],  # pth
                    [safetensors_folder] if safetensors_folder else [],  # ckzip
                    [safetensors_folder] if safetensors_folder else []   # safetensors
                )
            logger.info("No epoch folder found for pattern: %s", folder_pattern)
            return [], [], [], [], []
        # Traditional epoch-based filtering for files
        if self.network in MISSING_EPOCH_FORMAT_NETWORKS:
            regex_pattern = fr".*{self.checkpoint_delimiter}{format_epoch_number}"
        else:
            regex_pattern = fr".*epoch_{format_epoch_number}(?:_step_\d+)?"
        find_trained_tlt = filter_files(checkpoint_files, regex_pattern=fr'.*{regex_pattern}\.tlt$')
        find_trained_hdf5 = filter_files(checkpoint_files, regex_pattern=fr'.*{regex_pattern}\.hdf5$')
        find_trained_pth = filter_files(checkpoint_files, regex_pattern=fr'.*{regex_pattern}\.pth$')
        find_trained_ckzip = filter_files(checkpoint_files, regex_pattern=fr'.*{regex_pattern}\.ckzip$')
        find_trained_safetensors = filter_files(checkpoint_files, regex_pattern=fr'.*{regex_pattern}\.safetensors$')
        return find_trained_tlt, find_trained_hdf5, find_trained_pth, find_trained_ckzip, find_trained_safetensors

    def get_best_checkpoint_path(self, path, recommendation, filter_by_format=False):
        """Get the path to the best checkpoint.

        Args:
            path: Path to search for checkpoints
            recommendation: Recommendation object containing experiment info
            filter_by_format: If True and using folder lookup, only save the configured checkpoint format.
                            Used in best model workflow to clean up non-matching formats.

        Returns:
            None: Updates internal checkpoint path mapping
        """
        self.ckpt_path[path] = {}
        format_epoch_number = format_epoch(self.network, self.best_epoch_number[recommendation.id])
        recommendation.best_epoch_number = format_epoch_number
        self.save_state()

        if self._uses_folder_lookup():
            logger.info(
                "Using folder-based checkpoint lookup for epoch %s at %s",
                recommendation.best_epoch_number, path
            )
        else:
            logger.info("Best epoch number %s %s", recommendation.best_epoch_number, path)
        (find_trained_tlt,
         find_trained_hdf5,
         find_trained_pth,
         find_trained_ckzip,
         find_trained_safetensors) = self.get_checkpoint_paths_matching_epoch_number(
            path, recommendation.id
        )

        # Check if a specific checkpoint format is configured
        checkpoint_format = self._get_checkpoint_format()

        # Map format names to their corresponding found paths
        format_map = {
            "tlt": find_trained_tlt,
            "hdf5": find_trained_hdf5,
            "pth": find_trained_pth,
            "ckzip": find_trained_ckzip,
            "safetensors": find_trained_safetensors
        }

        if filter_by_format and checkpoint_format and self._uses_folder_lookup():
            # Only save the configured checkpoint format for folder-based checkpoints
            logger.info("Filtering: saving only %s format to ckpt_path", checkpoint_format)
            if checkpoint_format in format_map and format_map[checkpoint_format]:
                self.ckpt_path[path][checkpoint_format] = format_map[checkpoint_format][0]
        else:
            # Save all available formats (default behavior)
            if find_trained_tlt:
                self.ckpt_path[path]["tlt"] = find_trained_tlt[0]
            if find_trained_hdf5:
                self.ckpt_path[path]["hdf5"] = find_trained_hdf5[0]
            if find_trained_pth:
                self.ckpt_path[path]["pth"] = find_trained_pth[0]
            if find_trained_ckzip:
                self.ckpt_path[path]["ckzip"] = find_trained_ckzip[0]
            if find_trained_safetensors:
                self.ckpt_path[path]["safetensors"] = find_trained_safetensors[0]

    def delete_checkpoint_files(
        self,
        path,
        rec,
        filter_by_format=False,
        retain_resume_checkpoint=False,
    ):
        """Remove the extra checkpoints generated after the on_cancel_automl_job"""
        report_health_beat(
            self.automl_context.id,
            f"Starting checkpoint cleanup for experiment {rec.id}"
        )
        if not os.getenv("CI_PROJECT_DIR", None):
            time.sleep(30)  # Mounted paths can take time to reflect files generated on remote locally
        trained_files = get_file_list_from_cloud_storage(self.decrypted_workspace_metadata, path)

        regex_pattern = r'.*\.(tlt|hdf5|pth|ckzip|safetensors|resume|lightning_logs)$'

        trained_files = filter_files(trained_files, regex_pattern)
        logger.info("Available checkpoints in delete_checkpoint_files function %s", trained_files)
        self.get_best_checkpoint_path(path, rec, filter_by_format=filter_by_format)
        if (
            retain_resume_checkpoint and
            self.automl_algorithm in (
                "hyperband", "h", "bohb", "asha", "dehb",
                "hyperband_es", "hes", "pbt",
            ) and
            getattr(rec, "early_stop_epoch", None) is not None
        ):
            resume_paths = self.get_checkpoint_paths_matching_epoch_number(
                path, rec.id, epoch_number=rec.early_stop_epoch
            )
            for format_name, candidates in zip(
                ("tlt", "hdf5", "pth", "ckzip", "safetensors"),
                resume_paths,
            ):
                if candidates:
                    candidate = candidates[0]
                    if candidate not in self.ckpt_path[path].values():
                        self.ckpt_path[path][f"resume_{format_name}"] = candidate
            logger.info(
                "Retaining rung-boundary checkpoint at epoch %s for possible "
                "promotion of recommendation %s",
                rec.early_stop_epoch, rec.id,
            )
        logger.info("self.ckpt_path in delete_checkpoint_files function %s", self.ckpt_path)
        logger.info("RETAIN_CHECKPOINTS_FOR_RESUME setting: %s", self.retain_checkpoints_for_resume)

        deletion_accepted = True
        for files in trained_files:
            should_delete = True

            if self._uses_folder_lookup():
                # For folder-based checkpointing, check if file has prefix of any checkpoint folder
                for checkpoint_folder in self.ckpt_path[path].values():
                    if files.startswith(checkpoint_folder + "/"):
                        should_delete = False
                        break
            else:
                # For file-based checkpointing, use exact match
                if files in self.ckpt_path[path].values():
                    should_delete = False

            if should_delete:
                logger.info("Removing item in delete_checkpoint_files function %s", files)
                if self.cs_instance.is_file(files):
                    logger.info("Removing file in delete_checkpoint_files function %s", files)
                    if self.cs_instance.delete_file(files) is False:
                        deletion_accepted = False
                    report_health_beat(
                        self.automl_context.id,
                        f"Deleting checkpoint file {files} for experiment {rec.id}"
                    )
                elif self._uses_folder_lookup():
                    logger.info("Removing folder in delete_checkpoint_files function %s", files)
                    if self.cs_instance.delete_folder(files[1:]) is False:
                        deletion_accepted = False
                    report_health_beat(
                        self.automl_context.id,
                        f"Deleting checkpoint file {files} for experiment {rec.id}"
                    )

        # Storage APIs can acknowledge a delete while leaving a prefix behind.
        # Re-list and require every remaining checkpoint to belong to the one
        # protected best checkpoint. A stale remote listing simply leaves the
        # durable cleanup phase pending for retry.
        remaining = filter_files(
            get_file_list_from_cloud_storage(
                self.decrypted_workspace_metadata, path
            ),
            regex_pattern,
        )
        protected = tuple(self.ckpt_path[path].values())
        if self._uses_folder_lookup():
            unprotected = [
                item for item in remaining
                if not any(
                    item == checkpoint or item.startswith(checkpoint + "/")
                    for checkpoint in protected
                )
            ]
        else:
            unprotected = [item for item in remaining if item not in protected]
        if unprotected:
            logger.warning(
                "Checkpoint cleanup for experiment %s is not yet visible; "
                "remaining unprotected artifacts: %s",
                rec.id, unprotected,
            )
        return deletion_accepted and not unprotected

    def delete_not_best_model_checkpoints(self, path, rec, flag):
        """Remove the checkpoints which don't correspond to the best result"""
        self._last_checkpoint_delete_verified = True
        try:
            valid_recs = [r for r in self.recommendations
                          if r.status in (JobStates.success, JobStates.failure)]

            if not valid_recs:
                # No completed experiments yet
                logger.warning(
                    "No completed experiment results available yet for best model selection. "
                    "All experiments are still pending or running."
                )
                best_mAP = 0.0
            else:
                # Find the best result from completed recommendations
                best_mAP = self.min_max(valid_recs, key=lambda rec: rec.result).result
        except Exception as e:
            logger.error("Exception thrown in delete_not_best_model_checkpoints is %s", str(e))
            best_mAP = 0.0

        logger.info("delete_not_best_model_checkpoints function arguments %s %s %s", path, rec, flag)
        logger.debug("rec.result %s best_mAP %s", rec.result, best_mAP)
        if rec.result != best_mAP or bool(flag):
            trained_files = get_file_list_from_cloud_storage(self.decrypted_workspace_metadata, path)
            regex_pattern = r'.*(?:lightning_logs|events).*$|.*\.(tlt|hdf5|pth|ckzip|safetensors|resume)$'
            trained_files = filter_files(trained_files, regex_pattern)
            logger.info("Available checkpoints in delete_not_best_model_checkpoints function %s", trained_files)
            if self._uses_folder_lookup():
                # Folder checkpoints contain marker/config files without a recognized
                # extension (for example Cosmos .rank_*_complete and cosmos_config).
                # Delete each checkpoint leaf recursively so those sidecars do not remain.
                result_root = os.path.normpath(path).lstrip("/")
                checkpoint_folders = []
                checkpoint_folder_keys = []
                flat_checkpoint_files = []
                for files in sorted(
                    trained_files, key=lambda item: os.path.dirname(item).count("/")
                ):
                    folder = os.path.dirname(files)
                    folder_key = os.path.normpath(folder).lstrip("/")
                    if folder_key == result_root:
                        # Never recursively delete the experiment root merely because a
                        # folder-based network wrote a checkpoint directly into it.
                        flat_checkpoint_files.append(files)
                    elif not folder_key.startswith(result_root + "/"):
                        logger.warning(
                            "Skipping checkpoint path outside experiment root %s: %s",
                            path, files
                        )
                    elif not any(
                        folder_key == existing or folder_key.startswith(existing + "/")
                        for existing in checkpoint_folder_keys
                    ):
                        checkpoint_folders.append(folder)
                        checkpoint_folder_keys.append(folder_key)

                for folder in checkpoint_folders:
                    logger.info(
                        "Removing folder in delete_not_best_model_checkpoints function %s",
                        folder
                    )
                    if self.cs_instance.delete_folder(folder) is False:
                        self._last_checkpoint_delete_verified = False
                for files in flat_checkpoint_files:
                    if self.cs_instance.is_file(files):
                        logger.info(
                            "Removing file in delete_not_best_model_checkpoints function %s",
                            files
                        )
                        if self.cs_instance.delete_file(files) is False:
                            self._last_checkpoint_delete_verified = False
            else:
                for files in trained_files:
                    if self.cs_instance.is_file(files):
                        logger.info("Removing files in delete_not_best_model_checkpoints function %s", files)
                        if self.cs_instance.delete_file(files) is False:
                            self._last_checkpoint_delete_verified = False

            remaining = filter_files(
                get_file_list_from_cloud_storage(
                    self.decrypted_workspace_metadata, path
                ),
                regex_pattern,
            )
            if remaining:
                self._last_checkpoint_delete_verified = False
                logger.warning(
                    "Checkpoint deletion for recommendation %s is not yet "
                    "visible; retaining cleanup intent for: %s",
                    rec.id, remaining,
                )
        else:
            flag = True
        return flag
