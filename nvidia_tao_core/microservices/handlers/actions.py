# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline construction for all experiment actions"""
import copy
import json
import os
import threading
import time
import traceback
import uuid
import logging
import re

from nvidia_tao_core.microservices.utils.automl_utils import (
    delete_lingering_checkpoints,
    wait_for_job_completion,
    update_automl_details_metadata
)
from nvidia_tao_core.microservices.constants import (
    _DATA_GENERATE_ACTIONS,
    _DATA_SERVICES_ACTIONS,
    COPY_MODEL_PARAMS_FROM_TRAIN_NETWORKS
)
from nvidia_tao_core.microservices.utils.cloud_utils import create_cs_instance
from nvidia_tao_core.microservices.utils.handler_utils import get_files_from_cloud
from nvidia_tao_core.microservices.utils.ngc_utils import get_user_key
from .docker_images import DOCKER_IMAGE_MAPPER, DOCKER_IMAGE_VERSION
from .infer_data_sources import apply_data_source_config
from .infer_params import CLI_CONFIG_TO_FUNCTIONS
from .execution_handlers.slurm_handler import get_results_dir_for_workspace
from .execution_handlers.execution_handler import ExecutionHandler
from nvidia_tao_core.microservices.utils.encrypt_utils import NVVaultEncryption
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    BACKEND,
    base_exp_uuid,
    get_base_experiment_metadata,
    get_handler_job_metadata,
    get_handler_log_root,
    get_handler_metadata,
    get_jobs_root,
    get_handler_root,
    get_toolkit_status,
    resolve_metadata,
    get_job_specs,
    save_job_specs,
    get_automl_brain_info,
    get_automl_best_rec_info,
    get_automl_controller_info,
    claim_automl_child_launch,
    claim_automl_child_relaunch,
    get_automl_child_lifecycle,
    save_automl_controller_info,
    set_automl_child_launch_state,
    get_dnn_status,
    update_job_metadata,
    update_job_status,
    write_handler_metadata,
    update_job_details_with_microservices_response,
    get_user_telemetry_opt_out
)
from nvidia_tao_core.microservices.utils.handler_utils import (
    StatusParser,
    build_cli_command,
    get_num_nodes_from_spec,
    get_total_epochs,
    read_nested_dict,
    search_for_base_experiment,
    get_num_gpus_from_spec,
    is_remote_backend,
    write_nested_dict,
    get_cloud_metadata
)
from nvidia_tao_core.microservices.utils.core_utils import (
    remove_key_by_flattened_string,
    read_network_config,
    get_admin_key,
    find_differences,
    merge_nested_dicts,
    get_monitoring_metric,
    get_microservices_network_and_action
)
from nvidia_tao_core.microservices.utils.job_utils.executor import (
    MicroserviceExecutor
)
from nvidia_tao_core.microservices.utils.executor_utils import get_cluster_ip
from nvidia_tao_core.microservices.utils.network_utils.network_constants import ptm_mapper
from nvidia_tao_core.microservices.utils.specs_utils import json_to_kitti, json_to_yaml, json_to_toml
from nvidia_tao_core.microservices.enum_constants import Backend
from nvidia_tao_core.microservices.utils.log_monitor_service import start_monitoring_job, stop_monitoring_job

SPEC_BACKEND_TO_FUNCTIONS = {
    "protobuf": json_to_kitti.kitti,
    "yaml": json_to_yaml.yml,
    "toml": json_to_toml.toml_format
}
HOST_PLATFORM = os.getenv("HOST_PLATFORM", "local-k8s")

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


class ActionPipeline:
    """ActionPipeline - Train, Evaluate, Retrain, Prune, Export, Gen_trt_engine (Model),

    To spawn a job by handling all dependencies, monitor and close a job end-to-end
    - Inputs:
        - JobContext: To communicate with the Model / Dataset handler
        - Requires Handler & AppHandler to run()
    - Processes spec requirements
        - Prepares specs (generate_specs step)
            - dataset config (defined for each network's train, evaluate, retrain)
            - base_experiment config (for train, evaluate)
            - parent model and load graph (for retrain)
            - CLI paramters (for all actions)
            - Classwise configs (for all applicable train, evaluate, retrain) => currently for OD networks only
            - Converts json to spec backend

        - Prepare command (generate_run_command)
            - Generate run command
        - Infers image from config.json and platform information (if applicable) and sends it to K8s
        - Interacts with status callbacks from DNN container (ETA: TBD) and communicated to Handlers through JobContext
        - Supports delete job
        - Supports resume for train
    - Exposed functions:
        - run():
        - delete():
        - resume(): Same as run()
    - Internal functions:
        - parse_status():
        - generate_config(): Assumes <action>.json exists
        - generate_run_command():
    - Helper functions():
        - __init__():
        - _read_api_params()
    """

    def __init__(self, job_context):
        """Initialize the ActionPipeline class"""
        # Job Context - bridge between Action and Handler
        self.job_context = job_context
        # Get some handler related data
        self.network = self.job_context.network
        self.action = self.job_context.action
        self.network_config = read_network_config(self.network)
        self.api_params = self._read_api_params()
        self.handler_kind = self.job_context.kind
        self.handler_metadata = get_handler_metadata(self.job_context.handler_id, self.handler_kind)
        self.workspace_id = self.handler_metadata.get("workspace")
        self.workspace_metadata = get_handler_metadata(self.workspace_id, "workspaces")
        self.handler_root = get_handler_root(self.job_context.org_name, None, self.job_context.handler_id, None)
        self.jobs_root = get_jobs_root(self.job_context.user_id, self.job_context.org_name)
        self.handler_log_root = get_handler_log_root(
            self.job_context.user_id,
            self.job_context.org_name,
            self.job_context.handler_id
        )
        self.handler_id = self.job_context.handler_id
        self.tao_deploy_actions = False
        self.parent_job_action = get_handler_job_metadata(self.job_context.parent_id).get("action")
        if (self.job_context.action in ("gen_trt_engine", "trtexec") or
           self.parent_job_action in ("gen_trt_engine", "trtexec")):
            self.tao_deploy_actions = True
        self.image = DOCKER_IMAGE_MAPPER[self.api_params.get("image", "")]
        if self.job_context.action in _DATA_SERVICES_ACTIONS:
            self.image = DOCKER_IMAGE_MAPPER["TAO_DS"]
        # If current or parent action is gen_trt_engine or trtexec, then it'a a tao-deploy container action
        # Override version of image specific for networks
        if self.tao_deploy_actions:
            self.image = DOCKER_IMAGE_MAPPER["TAO_DEPLOY"]
        using_previous_version = False
        if self.network in DOCKER_IMAGE_VERSION.keys():
            self.tao_framework_version, self.tao_model_override_version = DOCKER_IMAGE_VERSION[self.network]
            if self.tao_model_override_version not in self.image:
                image = self.image.replace(self.tao_framework_version, self.tao_model_override_version)
                if image != self.image:
                    self.image = image
                    using_previous_version = True
        if self.action in self.network_config.get("api_params", {}).get("image_override_per_action", {}):
            image_override_per_action = self.api_params.get("image_override_per_action", {})
            override_key = image_override_per_action.get(self.action)
            if override_key:
                self.image = DOCKER_IMAGE_MAPPER[override_key]
        # This will be run inside a thread
        self.thread = None

        # Parameters to launch a job and monitor status
        self.job_name = str(self.job_context.id)

        self.spec = {}
        self.config = {}
        self.job_env_variables = {
            "ORCHESTRATION_API_NETWORK": self.network,
            "ORCHESTRATION_API_ACTION": self.action
        }
        if using_previous_version:
            self.job_env_variables = {}
        # Extract platform_id from backend_details
        self.platform_id = None
        if (self.job_context.backend_details and
                self.job_context.backend_details.get('backend_type') == "lepton"):
            self.platform_id = self.job_context.backend_details.get('platform_id')

        self.run_command = ""
        self.logfile = os.path.join(self.handler_log_root, str(self.job_context.id) + ".txt")
        self.cloud_metadata = {}
        self.cs_instance = None  # initialized in run()
        self.ngc_runner = False
        self.local_cluster = False
        self.num_gpu = (
            self.job_context.specs.get("num_gpu", self.job_context.num_gpu)
            if self.job_context.specs else self.job_context.num_gpu
        )
        self.num_nodes = get_num_nodes_from_spec(self.job_context.specs, self.action, network=self.network)
        self.recursive_dataset_file_download = self.api_params.get("recursive_dataset_file_download", False)
        self.retain_checkpoints_for_resume = self.job_context.retain_checkpoints_for_resume
        self.early_stop_epoch = self.job_context.early_stop_epoch
        self.workspace_ids = []

    def _read_api_params(self):
        """Read network config json file and return api_params key"""
        return self.network_config.get("api_params", {})

    def generate_config(self):
        """Generate config for this action; Actions may override"""
        return {}, {}

    def generate_run_command(self):
        """Generate run command for this action; Actions may override"""
        return "", None

    def post_run(self):
        """Run & modify internal variables after toolkit job is done; Actions may override"""
        return

    def decrypt_docker_env_vars(self, docker_env_vars):
        """Decrypt NvVault encrypted values"""
        config_path = os.getenv("VAULT_SECRET_PATH", None)
        if config_path:
            encryption = NVVaultEncryption(config_path)
            for docker_env_var_key, docker_env_var_value in docker_env_vars.items():
                if encryption.check_config()[0]:
                    docker_env_vars[docker_env_var_key] = encryption.decrypt(docker_env_var_value)

    def get_epoch_numbers_from_job(self):
        """Extract epoch numbers from checkpoint files and store in job metadata"""
        try:
            job_files, _, _, _ = get_files_from_cloud(self.handler_metadata, self.job_name)
            epoch_numbers = []
            for job_file in job_files:
                # Extract numbers before the extension using regex
                match = re.search(r'(\d+)(?=\.(pth|hdf5|tlt)$)', job_file)
                if match:
                    epoch_number = match.group(1)
                    if epoch_number not in epoch_numbers:
                        epoch_numbers.append(epoch_number)

            # Sort epoch numbers as integers
            if epoch_numbers:
                epoch_numbers = sorted([int(e) for e in epoch_numbers])
                # Update job metadata with epoch numbers
                update_job_metadata(
                    self.handler_id,
                    self.job_name,
                    metadata_key="epoch_numbers",
                    data=epoch_numbers,
                    kind=self.handler_kind
                )
                self.detailed_print(f"Stored epoch numbers for job {self.job_name}: {epoch_numbers}")
        except Exception as e:
            logger.error("Exception while extracting epoch numbers: %s", str(e))
            self.detailed_print(f"Failed to extract epoch numbers: {str(e)}")

    def generate_env_variables(self, automl_brain_job_id=None, experiment_number=None, automl_exp_job_id=None):
        """Generate env variables required for a job"""
        host_base_url = os.getenv("HOSTBASEURL", "no_url")
        log_callback_job_id = self.job_context.id
        if automl_exp_job_id:
            log_callback_job_id = automl_exp_job_id

        org_name = self.job_context.org_name
        if BACKEND == Backend.LOCAL_K8S:
            cluster_ip, cluster_port = get_cluster_ip()
            if cluster_ip and cluster_port:
                host_base_url = f"http://{cluster_ip}:{cluster_port}"

        # Pluralize handler_kind for API endpoint (experiment -> experiments, dataset -> datasets)
        handler_kind_plural = f"{self.handler_kind}s" if not self.handler_kind.endswith('s') else self.handler_kind

        status_url = (
            f"{host_base_url}/api/v1/orgs/{org_name}/{handler_kind_plural}/"
            f"{self.handler_id}/jobs/{self.job_context.id}"
        )
        if automl_brain_job_id:
            status_url = (
                f"{host_base_url}/api/v1/orgs/{org_name}/{handler_kind_plural}/"
                f"{self.handler_id}/jobs/{automl_brain_job_id}"
            )
            if experiment_number:
                self.job_env_variables["AUTOML_EXPERIMENT_NUMBER"] = experiment_number

        self.job_env_variables["TELEMETRY_OPT_OUT"] = get_user_telemetry_opt_out(
            self.job_context.user_id,
            self.job_context.org_name
        )
        self.job_env_variables["CLOUD_BASED"] = "True"
        # Pass BACKEND to container so it knows whether to use server-side log streaming
        # Set TAO_EXECUTION_BACKEND env variable so container knows its execution environment
        # NOTE: We use TAO_EXECUTION_BACKEND instead of BACKEND to avoid conflicts
        # BACKEND is used to detect if code is running in service pods vs job containers
        self.job_env_variables["TAO_EXECUTION_BACKEND"] = BACKEND.value
        user_key = get_user_key(
            self.job_context.user_id,
            self.job_context.org_name,
            admin_key_override=True
        )
        self.job_env_variables["TAO_USER_KEY"] = user_key
        self.job_env_variables["RECURSIVE_DATASET_FILE_DOWNLOAD"] = str(self.recursive_dataset_file_download)
        self.job_env_variables["TAO_ADMIN_KEY"] = get_admin_key()
        self.job_env_variables["TAO_API_KEY"] = get_admin_key(legacy_key=True)
        self.job_env_variables["TAO_API_SERVER"] = host_base_url
        self.job_env_variables["TAO_API_JOB_ID"] = log_callback_job_id
        self.job_env_variables["TAO_LOGGING_SERVER_URL"] = status_url
        self.job_env_variables["RETAIN_CHECKPOINTS_FOR_RESUME"] = str(self.retain_checkpoints_for_resume)
        if self.early_stop_epoch is not None:
            self.job_env_variables["EARLY_STOP_EPOCH"] = str(self.early_stop_epoch)

        # Set TAO_API_RESULTS_DIR if not already set (e.g., by SLURM handler)
        # This ensures consistent results directory handling across all execution backends
        if "TAO_API_RESULTS_DIR" not in self.job_env_variables:
            # Use the shared utility function to determine results directory
            results_dir = get_results_dir_for_workspace(
                workspace_metadata=self.workspace_metadata,
                job_id=self.job_context.id,
                specs=None  # specs not available in this context, will use workspace metadata
            )
            # Extract just the base directory (without job_id) for the env var
            # The full path with job_id will be constructed where needed
            self.job_env_variables["TAO_API_RESULTS_DIR"] = results_dir.rsplit('/', 1)[0]

    def generate_nv_job_metadata(self, nv_job_metadata, workspace_metadata=None):
        """Convert run command generated into format that"""
        nv_job_metadata["dockerImageName"] = self.image

        if workspace_metadata:
            handler = ExecutionHandler.create_handler(
                workspace_metadata=workspace_metadata,
                backend=BACKEND,
                job_id=self.job_name
            )
            if handler:
                available_instances = handler.get_available_instances()
                if available_instances:
                    nv_job_metadata["workspace_ids"] = list(self.workspace_ids)
                if available_instances and self.platform_id in available_instances:
                    nv_job_metadata["backend_details"] = {
                        "cluster": available_instances[self.platform_id]["cluster"],
                        "gpu_type": available_instances[self.platform_id]["gpu_type"],
                    }
                else:
                    logger.error(f"No available instances found for platform {self.platform_id}")

    def get_handler_cloud_details(self):
        """Gather cloud details from various handlers associated for the job"""
        self.workspace_ids = []
        workspace_cache = {}

        def process_metadata(data_type, dataset_id=None, metadata=None, workspace_cache={}):
            """Process metadata for datasets, workspaces, etc."""
            if not metadata:
                metadata = resolve_metadata(data_type, dataset_id)
            else:
                metadata = copy.deepcopy(metadata)
            workspace_id = metadata.get("workspace", "")
            if not workspace_id:
                return
            self.workspace_ids.append(workspace_id)

        if self.handler_metadata.get("train_datasets", []):
            for train_ds in self.handler_metadata.get("train_datasets", []):
                process_metadata("dataset", dataset_id=train_ds, workspace_cache={})
        elif (self.job_context.network not in ["auto_label", "image"]):
            process_metadata("dataset", metadata=self.handler_metadata, workspace_cache=workspace_cache)

        eval_ds = self.handler_metadata.get("eval_dataset", None)
        if eval_ds:
            process_metadata("dataset", dataset_id=eval_ds, workspace_cache=workspace_cache)

        infer_ds = self.handler_metadata.get("inference_dataset", None)
        if infer_ds:
            process_metadata("dataset", dataset_id=infer_ds, workspace_cache=workspace_cache)

        experiment_metadata = copy.deepcopy(self.handler_metadata)
        exp_workspace_id = experiment_metadata.get("workspace")
        self.workspace_ids.append(exp_workspace_id)

        # Populate cloud_metadata with workspace cloud details
        get_cloud_metadata(self.workspace_ids, self.cloud_metadata)

    def handle_ptm_anomalies(self):
        """Remove one of end-end or backbone related PTM field based on the Handler metadata info"""
        for base_experiment_id in self.handler_metadata.get("base_experiment_ids", []):
            base_experiment_metadata = get_base_experiment_metadata(base_experiment_id)
            if base_experiment_metadata.get("base_experiment_metadata", {}).get("is_backbone"):
                # if ptm is a backbone remove end_to_end field from config and spec
                parameter_to_remove = ptm_mapper.get("end_to_end", {}).get(base_experiment_metadata.get("network_arch"))
            else:
                # if ptm is not a backbone remove it field from config and spec
                parameter_to_remove = ptm_mapper.get("backbone", {}).get(base_experiment_metadata.get("network_arch"))
            if parameter_to_remove:
                remove_key_by_flattened_string(self.spec, parameter_to_remove)
                remove_key_by_flattened_string(self.config, parameter_to_remove)

        if not self.handler_metadata.get("base_experiment_ids", []):
            parameters_to_remove = [
                ptm_mapper.get("end_to_end", {}).get(self.network),
                ptm_mapper.get("backbone", {}).get(self.network),
                ptm_mapper.get("default", {}).get(self.network),
            ]
            for parameter_to_remove in parameters_to_remove:
                if parameter_to_remove:
                    remove_key_by_flattened_string(self.spec, parameter_to_remove)
                    remove_key_by_flattened_string(self.config, parameter_to_remove)

    def detailed_print(self, *args, **kwargs):
        """Print with job context"""
        # Remove 'file' from kwargs since logger.info() doesn't accept it
        if 'file' in kwargs:
            del kwargs['file']
        # Join all args and kwargs into a single string message
        message = ' '.join(str(arg) for arg in args)
        kwargs_str = ' '.join(f'{k}={v}' for k, v in kwargs.items())
        if kwargs_str:
            message = f'{message} {kwargs_str}'
        logger.info(message)

    def create_microservice_action_job(self, job_id, nv_job_metadata={}):
        """Call executor function to create microservice pod and then invoke it"""
        logger.info("Creating microservices job_action ms pod")
        if nv_job_metadata:
            logger.info(f"NV job metadata: {nv_job_metadata}")
            resource_shape = nv_job_metadata.get("backend_details", {}).get("gpu_type", 'gpu.1xh200')
            dedicated_node_group = nv_job_metadata.get(
                "backend_details", {}).get("cluster", 'nv-int-multiteam-nebius-h200-01-mjgbgffo')
        else:
            resource_shape = None
            dedicated_node_group = None
        microservice_executor = MicroserviceExecutor()
        response = microservice_executor.create_microservice_and_send_request(
            api_endpoint="post_action",
            network=self.network,
            action=self.action,
            cloud_metadata=self.cloud_metadata,
            specs=self.spec,
            microservice_pod_id=self.job_name,
            num_gpu=self.num_gpu,
            microservice_container=self.image,
            org_name=self.job_context.org_name,
            handler_id=self.handler_id,
            handler_kind=self.handler_kind,
            accelerator=self.platform_id,
            docker_env_vars=self.job_env_variables,
            num_nodes=self.num_nodes,
            resource_shape=resource_shape,
            dedicated_node_group=dedicated_node_group,
            backend_details=self.job_context.backend_details,
        )
        if response and not response.ok:
            update_job_details_with_microservices_response(response.json().get("error", ""), job_id, self.job_name)

    def monitor_job(self):
        """Monitors the job status and updates job metadata"""
        _, outdir = self.generate_run_command()
        if not outdir:
            outdir = CLI_CONFIG_TO_FUNCTIONS["output_dir"](self.job_context, self.handler_metadata)

        status_parser = StatusParser(self.job_context.network, outdir)

        total_epochs = 1
        if self.job_context.action in ['train', 'distill', 'retrain', 'quantize']:
            total_epochs = get_total_epochs(self.job_context, self.job_context.specs)

        metric = self.handler_metadata.get("metric", "")
        if not metric:
            metric = get_monitoring_metric(self.network)

        # Detect if this is a SLURM job (no Docker container/statefulset to manage)
        cloud_type = self.workspace_metadata.get("cloud_type")
        is_cloud_job = cloud_type in ["slurm", "lepton"]
        if is_cloud_job:
            logger.info(f"Job {self.job_name} is a cloud job - skipping Docker/K8s container operations")

        # Get results directory for cloud job status checking
        from nvidia_tao_core.microservices.handlers.execution_handlers.slurm_handler import (
            get_results_dir_for_workspace
        )
        results_dir = get_results_dir_for_workspace(
            workspace_metadata=self.workspace_metadata,
            job_id=self.job_name,
            specs=None
        )

        k8s_status = ExecutionHandler.get_job_status_with_handler(
            job_name=self.job_name,
            workspace_metadata=self.workspace_metadata,
            handler_id=self.handler_id,
            handler_kind=self.handler_kind,
            network=self.network,
            action=self.action,
            specs=None,
            docker_env_vars=self.job_env_variables,
            results_dir=results_dir,
            automl_exp_job=False,
            org_name=self.job_context.org_name
        )
        logger.info(f"Init Job status: {k8s_status}")

        # Delete job if is canceled/paused during pod creation
        metadata_status = get_handler_job_metadata(self.job_name).get("status", "Error")
        if metadata_status in ("Canceling", "Canceled", "Pausing", "Paused"):
            self.detailed_print(f"Terminating job {self.job_name}")
            if not is_cloud_job:
                ExecutionHandler.delete_with_handler(
                    self.job_name,
                    workspace_metadata=self.workspace_metadata,
                )
            else:
                logger.info(f"Cloud job {self.job_name} canceled/paused - handled by cloud cluster")

        # Monitor job status
        cur_status_line = 0
        while k8s_status in ["Done", "Error", "Running", "Pending", "Pausing"]:
            # If Done, try running self.post_run()
            # Poll every 30 seconds
            time.sleep(30)

            metadata_status = get_handler_job_metadata(self.job_name).get("status", "Error")
            logger.info(f"Metadata status: {metadata_status}")
            if metadata_status in ("Canceled", "Paused") and k8s_status == "Running":
                self.detailed_print(f"Terminating job {self.job_name}")
                if not is_cloud_job:
                    ExecutionHandler.delete_with_handler(
                        self.job_name,
                        workspace_metadata=self.workspace_metadata,
                    )
                else:
                    logger.info(f"Cloud job {self.job_name} canceled/paused - handled by cloud cluster")
            if k8s_status == "Done":
                update_job_status(self.handler_id, self.job_name, status="Running", kind=self.handler_kind)
                # Retrieve status one last time!
                new_results = status_parser.update_results(
                    total_epochs=total_epochs,
                    job_id=self.job_name,
                    workspace_metadata=self.workspace_metadata,
                    cloud_results_dir=results_dir
                )
                update_job_metadata(
                    self.handler_id,
                    self.job_name,
                    metadata_key="job_details",
                    data=new_results,
                    kind=self.handler_kind
                )
                try:
                    self.detailed_print("Post running")
                    # If post run is done, make it done
                    self.post_run()
                    if self.job_context.action in ['train', 'distill', 'retrain', 'quantize']:
                        _, best_checkpoint_epoch_number, latest_checkpoint_epoch_number = status_parser.read_metric(
                            results=new_results[self.job_name],
                            metric=metric,
                            brain_epoch_number=total_epochs
                        )
                        self.handler_metadata["checkpoint_epoch_number"][
                            f"best_model_{self.job_name}"
                        ] = best_checkpoint_epoch_number
                        self.handler_metadata["checkpoint_epoch_number"][
                            f"latest_model_{self.job_name}"
                        ] = latest_checkpoint_epoch_number
                        write_handler_metadata(self.handler_id, self.handler_metadata, self.handler_kind)
                        # Extract and store epoch numbers in job metadata
                        self.get_epoch_numbers_from_job()
                    if not os.path.exists(f"{self.jobs_root}/{self.job_name}"):
                        os.makedirs(f"{self.jobs_root}/{self.job_name}")
                    update_job_status(self.handler_id, self.job_name, status="Done", kind=self.handler_kind)

                    # Stop log monitoring when job is done
                    logger.debug(f"[ACTIONS] Job {self.job_name} completed, checking if should stop log monitoring")
                    if BACKEND in (Backend.LOCAL_K8S, Backend.LOCAL_DOCKER):
                        try:
                            logger.info(
                                f"[ACTIONS] Stopping log monitoring for completed job {self.job_name}"
                            )
                            stop_monitoring_job(self.job_name)
                            logger.info(
                                f"[ACTIONS] Successfully stopped log monitoring for completed job "
                                f"{self.job_name}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"[ACTIONS] Failed to stop log monitoring for job {self.job_name}: "
                                f"{type(e).__name__}: {e}"
                            )

                    break
                except Exception as e:
                    # If post run fails, call it Error
                    logger.error("Exception thrown in post run after done status %s", str(e))
                    self.detailed_print(traceback.format_exc())
                    update_job_status(self.handler_id, self.job_name, status="Error", kind=self.handler_kind)

                    # Stop log monitoring when job errors
                    if BACKEND in (Backend.LOCAL_K8S, Backend.LOCAL_DOCKER):
                        try:
                            stop_monitoring_job(self.job_name)
                            logger.info(f"Stopped log monitoring for errored job {self.job_name}")
                        except Exception as stop_err:
                            logger.warning(f"Failed to stop log monitoring for job {self.job_name}: {stop_err}")

                    break
            # If running in K8s, update results to job_context
            elif k8s_status == "Running":
                update_job_status(self.handler_id, self.job_name, status="Running", kind=self.handler_kind)
                # Update results
                new_results = status_parser.update_results(
                    total_epochs=total_epochs,
                    job_id=self.job_name,
                    workspace_metadata=self.workspace_metadata,
                    cloud_results_dir=results_dir
                )
                update_job_metadata(
                    self.handler_id,
                    self.job_name,
                    metadata_key="job_details",
                    data=new_results,
                    kind=self.handler_kind
                )

            # Pending is if we have queueing systems down the road
            elif k8s_status == "Pending":
                k8s_status = ExecutionHandler.get_job_status_with_handler(
                    job_name=self.job_name,
                    workspace_metadata=self.workspace_metadata,
                    handler_id=self.handler_id,
                    handler_kind=self.handler_kind,
                    network=self.network,
                    action=self.action,
                    specs=None,
                    docker_env_vars=self.job_env_variables,
                    results_dir=results_dir,
                    automl_exp_job=False,
                    org_name=self.job_context.org_name
                )
                continue

            elif k8s_status == "Pausing":
                lines_to_process = get_dnn_status(self.job_name, automl=False, experiment_number=None)[cur_status_line:]
                for status_dict in lines_to_process:
                    cur_status_line += 1
                    toolkit_job_completed = False
                    if status_dict.get("status") in ("SUCCESS", "FAILURE"):
                        k8s_status = "Error"
                        if status_dict.get("status") == "SUCCESS":
                            k8s_status = "Done"
                        toolkit_job_completed = True
                        logger.info("toolkit job completed with status %s", status_dict.get("status"))
                        # update_job_status(self.handler_id, self.job_name, status=k8s_status, kind=self.handler_kind)
                        break
                if toolkit_job_completed:
                    break
            # If the job never submitted or errored out!
            if k8s_status == "Error":
                logger.info("K8s error status")
                new_results = status_parser.update_results(
                    total_epochs=total_epochs,
                    job_id=self.job_name,
                    workspace_metadata=self.workspace_metadata,
                    cloud_results_dir=results_dir
                )
                update_job_metadata(
                    self.handler_id,
                    self.job_name,
                    metadata_key="job_details",
                    data=new_results,
                    kind=self.handler_kind
                )
                update_job_status(self.handler_id, self.job_name, status="Error", kind=self.handler_kind)
                break
            k8s_status = ExecutionHandler.get_job_status_with_handler(
                job_name=self.job_name,
                workspace_metadata=self.workspace_metadata,
                handler_id=self.handler_id,
                handler_kind=self.handler_kind,
                network=self.network,
                action=self.action,
                specs=None,
                docker_env_vars=self.job_env_variables,
                results_dir=results_dir,
                automl_exp_job=False,
                org_name=self.job_context.org_name
            )
            logger.info(f"Updated K8s status: {k8s_status}")

            # For SLURM/Lepton jobs, the status from get_job_status is already the final status
            # (Done/Error from status.json), not just the K8s pod status
            if is_cloud_job:
                logger.info(f"Cloud job {self.job_name} current status: {k8s_status}")

        metadata_status = get_handler_job_metadata(self.job_name).get("status", "Error")

        toolkit_status = get_toolkit_status(self.job_name)
        self.detailed_print(f"Toolkit status for {self.job_name} is {toolkit_status}")
        if (metadata_status not in ("Canceled", "Canceling", "Paused", "Pausing") and
                toolkit_status != "SUCCESS" and
                self.job_context.action != "trtexec"):
            update_job_status(self.handler_id, self.job_name, status="Error", kind=self.handler_kind)
            metadata_status = "Error"

        self.detailed_print(f"Job Done: {self.job_name} Final status: {metadata_status}")
        if self.ngc_runner or BACKEND in (Backend.LOCAL_K8S, Backend.LOCAL_DOCKER):
            logger.info(f"Metadata status is {metadata_status}")
            logger.info(f'Bool is {metadata_status not in ("Canceled", "Canceling", "Paused")}')
            if metadata_status not in ("Canceled", "Canceling", "Paused"):
                if not is_cloud_job:
                    ExecutionHandler.delete_with_handler(
                        self.job_name,
                        workspace_metadata=self.workspace_metadata,
                    )
                else:
                    logger.info(f"SLURM job {self.job_name} completed - skipping statefulset deletion")
            if metadata_status == "Pausing":
                update_job_status(self.handler_id, self.job_name, status="Paused", kind=self.handler_kind)

    def run(self):
        """Calls necessary setup functions and calls job creation"""
        # Set up
        self.thread = threading.current_thread()
        try:
            # Generate config
            self.spec, self.config = self.generate_config()
            self.cs_instance, _ = create_cs_instance(self.workspace_metadata)
            self.handle_ptm_anomalies()
            # Populate the cloud metadata for the job
            self.get_handler_cloud_details()
            # Generate run command
            self.run_command, outdir = self.generate_run_command()
            if self.spec:
                self.num_gpu = get_num_gpus_from_spec(
                    self.spec, self.job_context.action, network=self.network, default=self.num_gpu,
                    skip_gpu_conditions_check=is_remote_backend(self.job_context.backend_details)
                )
                self.num_nodes = get_num_nodes_from_spec(
                    self.spec,
                    self.job_context.action,
                    network=self.network,
                    default=self.num_nodes)
                self.detailed_print(f"Job {self.job_name} running with {self.num_gpu} GPUs and {self.num_nodes} nodes")
            if not outdir:
                # Use TAO_API_RESULTS_DIR for SLURM compatibility, fallback to /results
                results_base = os.getenv('TAO_API_RESULTS_DIR', '/results')
                outdir = f"{results_base}/{self.job_name}"
            # Pipe stdout and stderr to logfile
            self.run_command += f" 2>&1 | tee /{self.job_name}.txt"
            # After command runs, make sure subdirs permission allows anyone to enter and delete
            self.run_command += f"; find {outdir} -type d | xargs chmod 777"
            # After command runs, make sure artifact files permission allows anyone to delete
            if self.local_cluster:
                outdir = os.path.normpath(outdir) + os.sep  # remove double trailing slashes for filepath
            self.run_command += f"; find {outdir} -type f | xargs chmod 666"
            # Optionally, pipe self.run_command into a log file
            self.detailed_print(self.run_command)
            self.detailed_print(self.image)

            nv_job_metadata = {}
            # Convert self.spec to a backend and post it into a <self.handler_spec_root><job_id>.txt file
            if self.spec:
                file_type = self.api_params["spec_backend"]
                if file_type == "json":
                    kitti_out = self.spec
                    kitti_out = json.dumps(kitti_out)
                else:
                    kitti_out = SPEC_BACKEND_TO_FUNCTIONS[file_type](self.spec)
                # Save specs to DB
                save_job_specs(self.job_name, kitti_out)

            # Submit to K8s
            # Platform is None, but might be updated in self.generate_config() or self.generate_run_command()
            # If platform is indeed None, JobExecutor.create_job would take care of it.
            docker_env_vars = self.handler_metadata.get("docker_env_vars", {})
            self.decrypt_docker_env_vars(docker_env_vars)
            self.job_env_variables.update(copy.deepcopy(docker_env_vars))
            self.generate_env_variables()
            self.generate_nv_job_metadata(nv_job_metadata, workspace_metadata=self.workspace_metadata)

            if BACKEND in (Backend.LOCAL_K8S, Backend.LOCAL_DOCKER):
                self.create_microservice_action_job(self.job_name, nv_job_metadata=nv_job_metadata)
            else:
                ExecutionHandler.create_job_with_handler(
                    org_name=self.job_context.org_name,
                    job_name=self.job_name,
                    image=self.image,
                    command=self.run_command,
                    workspace_metadata=self.workspace_metadata,
                    num_gpu=self.num_gpu,
                    num_nodes=self.num_nodes,
                    accelerator=self.platform_id,
                    docker_env_vars=self.job_env_variables,
                    nv_job_metadata=nv_job_metadata,
                    local_cluster=self.local_cluster,
                    automl_brain=False,
                    automl_exp_job=False,
                    backend_details=self.job_context.backend_details
                )
            self.detailed_print("Job created", self.job_name)

            # Start log monitoring for K8s and Docker backends
            logger.debug(
                f"[ACTIONS] Checking if log monitoring should start for job {self.job_name}, "
                f"BACKEND={BACKEND}"
            )
            if BACKEND in (Backend.LOCAL_K8S, Backend.LOCAL_DOCKER):
                try:
                    logger.debug(f"[ACTIONS] Starting log monitoring setup for job {self.job_name}")
                    # Get callback URL if available
                    callback_url = self.job_env_variables.get("TAO_LOGGING_SERVER_URL")
                    logger.debug(f"[ACTIONS] Callback URL from env: {callback_url}")
                    if callback_url:
                        callback_url = callback_url + ":log_update"
                        logger.debug(f"[ACTIONS] Full callback URL: {callback_url}")

                    # Get namespace for K8s
                    namespace = None
                    if BACKEND == Backend.LOCAL_K8S:
                        namespace = os.getenv("NAMESPACE")
                        logger.debug(f"[ACTIONS] K8s namespace from env: {namespace}")
                        if not namespace:
                            try:
                                namespace_file = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
                                logger.debug(f"[ACTIONS] Reading namespace from {namespace_file}")
                                with open(namespace_file, 'r', encoding='utf-8') as f:
                                    namespace = f.read().strip()
                                logger.debug(f"[ACTIONS] Got namespace from service account: {namespace}")
                            except Exception as e:
                                namespace = "default"
                                logger.debug(f"[ACTIONS] Using default namespace (error: {e})")

                    # Start monitoring
                    logger.info(f"[ACTIONS] Starting log monitoring for job {self.job_name}, namespace={namespace}")
                    start_monitoring_job(
                        self.job_name,
                        callback_url=callback_url,
                        namespace=namespace,
                        metadata={
                            'handler_id': self.handler_id,
                            'handler_kind': 'experiment',
                            'action': self.action,
                            'network': self.network
                        }
                    )
                    logger.info(f"[ACTIONS] Successfully started log monitoring for job {self.job_name}")
                except Exception as e:
                    logger.warning(
                        f"[ACTIONS] Failed to start log monitoring for job {self.job_name}: "
                        f"{type(e).__name__}: {e}"
                    )
                    logger.debug("[ACTIONS] Exception details:", exc_info=True)
            else:
                logger.debug(f"[ACTIONS] Skipping log monitoring for backend {BACKEND}")

            self.monitor_job()
            return

        except Exception as e:
            # Something went wrong inside...
            self.detailed_print(traceback.format_exc())
            self.detailed_print(f"Job {self.job_name} did not start")
            update_job_status(self.handler_id, self.job_name, status="Error", kind=self.handler_kind)
            result_dict = {
                self.job_name: {
                    "detailed_status": {
                        "message": "Error due to unmet dependencies",
                        "status": "FAILURE"
                    }
                }
            }
            if isinstance(e, ValueError):
                result_dict = {self.job_name: {"detailed_status": {"message": str(e), "status": "FAILURE"}}}
            if isinstance(e, TimeoutError):
                result_dict = {
                    self.job_name: {
                        "detailed_status": {
                            "message": "Data downloading from cloud storage failed.",
                            "status": "FAILURE"
                        }
                    }
                }
            update_job_metadata(
                self.handler_id,
                self.job_name,
                metadata_key="job_details",
                data=result_dict,
                kind=self.handler_kind
            )
            return


class CLIPipeline(ActionPipeline):
    """CLIPipeline for actions involve only cli params"""

    def __init__(self, job_context):
        """Initialize the CLIPipeline class"""
        super().__init__(job_context)

        self.network = job_context.network
        self.action = job_context.action

        # Handle anomalies in network action names
        if self.action == "retrain":
            self.action = "train"

        # Use the centralized function to map network and action
        self.network, self.action = get_microservices_network_and_action(self.network, self.action)

    def generate_config(self):
        """Generate config dictionary"""
        # Get some variables
        action = self.job_context.action
        # User stored CLI param in a json file
        config = get_job_specs(self.job_context.id)
        network = self.job_context.network
        # Get CLI params from config json
        network_config = read_network_config(network)
        if action in network_config.get("cli_params", {}).keys():
            for field_name, inference_fn in network_config["cli_params"][action].items():
                if inference_fn in CLI_CONFIG_TO_FUNCTIONS:
                    field_value = CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
                    if field_value:
                        config[field_name] = field_value
        return {}, config

    def generate_run_command(self):
        """Generate run command"""
        if self.action == "dataset_convert":
            if self.network not in ("ocrnet", "pointpillars"):
                self.config["results_dir"] = CLI_CONFIG_TO_FUNCTIONS["output_dir"](
                    self.job_context,
                    self.handler_metadata
                )

        params_to_cli = build_cli_command(self.config)
        run_command = f"{self.network} {self.action} {params_to_cli}"
        if self.action == "trtexec":
            run_command = f"{self.action} {params_to_cli}"

        return run_command, None


# Specs are modified as well => Train, Evaluate, Retrain Actions


class TrainVal(CLIPipeline):
    """Class for experiment actions which involves both spec file as well as cli params"""

    def generate_config(self):
        """Generates spec and cli params

        Returns:
        spec: contains the network's spec file parameters
        config: contains cli params
        """
        network = self.job_context.network
        action = self.job_context.action
        # Infer CLI params
        config = {}
        network_config = read_network_config(network)
        if action in network_config.get("cli_params", {}).keys():
            for field_name, inference_fn in network_config["cli_params"][action].items():
                if inference_fn in CLI_CONFIG_TO_FUNCTIONS:
                    field_value = CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
                    if field_value:
                        config[field_name] = field_value

        # Read spec from <action>.json for train, resume train, evaluate, retrain. If not there, use train.json
        spec = get_job_specs(self.job_context.id)
        if network in COPY_MODEL_PARAMS_FROM_TRAIN_NETWORKS:
            cur_job_id = self.job_context.id
            while True:
                cur_job_meta = get_handler_job_metadata(cur_job_id)
                if not cur_job_meta:
                    break
                parent_job_id = cur_job_meta.get("parent_id", "")
                if not parent_job_id:
                    break
                parent_job_metadata = get_handler_job_metadata(parent_job_id)
                parent_action = parent_job_metadata.get("action", "")
                if not parent_action:
                    break
                if parent_action in ("train", "distill", "quantize"):
                    # pylint: disable=C0415
                    from .spec_handler import SpecHandler
                    default_spec_schema_response = SpecHandler.get_spec_schema(
                        self.job_context.user_id,
                        self.job_context.org_name,
                        self.job_context.handler_id,
                        action,
                        self.handler_kind
                    )
                    user_modified_values = {}
                    if default_spec_schema_response.code == 200:
                        spec_schema = default_spec_schema_response.data
                        default_spec = spec_schema["default"]
                        user_modified_values = find_differences(spec, default_spec)
                    automl = False
                    best_rec_number, best_rec_job_id, _ = get_automl_best_rec_info(parent_job_id)
                    logger.info(f"Best rec number: {best_rec_number}, Best rec job id: {best_rec_job_id}")
                    if best_rec_number != "-1":
                        automl = True
                        parent_spec = get_job_specs(
                            best_rec_job_id, automl=automl, automl_experiment_id=best_rec_number
                        )
                    else:
                        parent_spec = get_job_specs(parent_job_id)
                    train_specs_passed_in_req_body = get_job_specs(parent_job_id)
                    modified_values = find_differences(parent_spec, train_specs_passed_in_req_body)
                    spec = merge_nested_dicts(spec, modified_values)
                    spec = merge_nested_dicts(spec, user_modified_values)
                    break
                cur_job_id = parent_job_id
            save_job_specs(self.job_context.id, spec)

        # Take .json file, read in spec params, infer spec params
        # Fall back to the mapped network's config if the action isn't in the primary config
        spec_params_action = action
        spec_params_config = network_config
        if action not in network_config.get("spec_params", {}):
            mapped_net, mapped_act = get_microservices_network_and_action(network, action)
            if mapped_net != network:
                mapped_config = read_network_config(mapped_net)
                if mapped_act in mapped_config.get("spec_params", {}):
                    spec_params_action = mapped_act
                    spec_params_config = mapped_config

        if spec_params_action in spec_params_config.get("spec_params", {}):
            for field_name, inference_fn in spec_params_config["spec_params"][spec_params_action].items():
                field_value = (
                    CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
                    if inference_fn in CLI_CONFIG_TO_FUNCTIONS
                    else inference_fn
                )
                if field_value:
                    write_nested_dict(spec, field_name, field_value)

        # Move CLI params from spec to config
        spec_keys_all = copy.deepcopy(list(spec.keys()))
        if "cli_params" in network_config and action in network_config["cli_params"]:
            for field_name in spec_keys_all:
                cnd1 = field_name in network_config["cli_params"][action].keys()
                cnd2 = network_config["cli_params"][action].get(field_name, None) == "from_csv"
                cnd3 = type(spec[field_name]) in [str, float, int, bool]
                if cnd1 and cnd2 and cnd3:
                    config[field_name] = spec.pop(field_name)
        self.detailed_print("Loaded specs")

        # Infer dataset config
        spec = apply_data_source_config(spec, self.job_context, self.handler_metadata)
        self.detailed_print("Loaded dataset")

        return spec, config

    def post_run(self):
        """Carry's out functions after the job is executed"""
        # copy pruned model so that evaluate can access via parent relation
        action = self.job_context.action
        if self.network in ("ocdnet", "ocrnet") and action == "retrain":
            inference_fn = "parent_model"
            pruned_model_path = CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
            bucket_name = pruned_model_path.split("//")[1].split("/")[0]
            pruned_model_path = pruned_model_path[pruned_model_path.find(bucket_name) + len(bucket_name):]
            _, file_extension = os.path.splitext(pruned_model_path)
            # Use TAO_API_RESULTS_DIR for SLURM compatibility, fallback to /results
            results_base = os.getenv('TAO_API_RESULTS_DIR', '/results')
            self.detailed_print(
                f"Copying pruned model {pruned_model_path} after retrain to "
                f"{results_base}/{self.job_name}/pruned_model{file_extension}\n"
            )
            destination_path = f"{results_base}/{self.job_name}/pruned_model{file_extension}"
            self.cs_instance.copy_file(pruned_model_path, destination_path)
        if self.job_context.action == "annotation_format_convert":
            handler_metadata = get_handler_metadata(self.handler_id, self.handler_kind)
            if self.spec["data"]["input_format"] == "KITTI":
                handler_metadata["format"] = "coco"
            elif self.spec["data"]["input_format"] == "COCO":
                handler_metadata["format"] = "kitti"
            write_handler_metadata(self.handler_id, handler_metadata, self.handler_kind)
        # Create dataset for data service actions that generate new dataset
        # These actions create a new dataset as part of their actions
        if action in _DATA_GENERATE_ACTIONS:
            # pylint: disable=C0415
            from .dataset_handler import DatasetHandler
            handler_metadata = get_handler_metadata(self.handler_id, self.handler_kind)
            request_dict = DatasetHandler.create_dataset_dict_from_experiment_metadata(
                self.job_context.id,
                self.action,
                handler_metadata
            )
            response = DatasetHandler.create_dataset(
                self.job_context.user_id,
                self.job_context.org_name,
                request_dict,
                dataset_id=self.job_context.id
            )
            if response.code != 200:
                self.detailed_print(
                    f"Failed to create dataset from {self.action} job. "
                    f"Response code: {response.code}"
                )
                update_job_status(self.handler_id, self.job_context.id, status="Error", kind=self.handler_kind)


class AutoMLPipeline(ActionPipeline):
    """Class for handling AutoML pipeline operations.

    This class contains methods for managing and executing AutoML pipeline tasks.
    """

    def __init__(self, job_context):
        """Initialize the AutoMLPipeline class"""
        super().__init__(job_context)
        self.network, self.action = get_microservices_network_and_action(self.network, self.action)
        # For AutoML experiments: job_context.id is the experiment job ID
        # and job_context.parent_id is the brain job ID
        self.automl_brain_job_id = self.job_context.parent_id if self.job_context.parent_id else self.job_context.id
        self.job_root = os.path.join(
            get_jobs_root(self.job_context.user_id, self.job_context.org_name),
            self.automl_brain_job_id
        )
        self.rec_number = self.get_recommendation_number()
        self.expt_root = f"{self.job_root}/experiment_{self.rec_number}"
        self.recs_dict = get_automl_controller_info(self.automl_brain_job_id)
        self.brain_dict = get_automl_brain_info(self.automl_brain_job_id)
        # Assign a new job id if not assigned already
        self.job_name = self.recs_dict[self.rec_number].get("job_id", None)
        if not self.job_name:
            self.job_name = str(uuid.uuid4())
            self.detailed_print("New job id being assigned to recommendation", self.job_name)
            self.recs_dict[self.rec_number]["job_id"] = self.job_name
            save_automl_controller_info(self.automl_brain_job_id, self.recs_dict)
            update_automl_details_metadata(self.automl_brain_job_id, self.handler_id, self.handler_kind)

        if not os.path.exists(self.expt_root):
            os.makedirs(self.expt_root)

    def add_ptm_dependency(self, recommended_values):
        """Add PTM as a dependency if backbone or num_layers is part of hyperparameter sweep"""
        # See if a ptm is needed (if not searching num_layers / backbone, no PTM), just take default
        ptm_id = None
        if "backbone" in recommended_values.keys() or "num_layers" in recommended_values.keys():
            for dep in self.job_context.dependencies:
                if dep.type == "automl_ptm":
                    ptm_id = dep.name
                    break
        if ptm_id:
            recommended_values["base_experiment_ids"] = search_for_base_experiment(
                get_handler_root(base_exp_uuid, "experiments", base_exp_uuid, ptm_id)
            )

    def generate_config(self, recommended_values):
        """Generate config for AutoML experiment"""
        spec = get_job_specs(self.automl_brain_job_id)

        epoch_multiplier = self.brain_dict.get("epoch_multiplier", None)
        if epoch_multiplier is not None:
            current_ri = int(
                self.brain_dict.get("ri", {"0": [float('-inf')]})[
                    str(self.brain_dict.get("bracket", 0))
                ][0]
            )

        for param_name, param_value in recommended_values.items():
            write_nested_dict(spec, param_name, param_value)

        for field_name, inference_fn in self.network_config["automl_spec_params"].items():
            if "automl_" in inference_fn:
                field_value = CLI_CONFIG_TO_FUNCTIONS[inference_fn](
                    self.job_context,
                    self.handler_metadata,
                    self.job_root,
                    self.rec_number,
                    self.job_name
                )
            elif "assign_const_value" in inference_fn:
                if epoch_multiplier is not None:
                    field_value = int(epoch_multiplier * current_ri)
                else:
                    field_value = int(read_nested_dict(spec, field_name))
                    if "assign_const_value," in inference_fn:
                        dependent_parameter_names = inference_fn.split(",")
                        dependent_field_value = int(read_nested_dict(spec, dependent_parameter_names[1]))
                        if dependent_parameter_names[1] in recommended_values:
                            field_value = recommended_values[dependent_parameter_names[1]]
                        elif len(dependent_parameter_names) == 2:
                            field_value = min(field_value, dependent_field_value)
                        elif len(dependent_parameter_names) == 3:
                            field_value = int(read_nested_dict(spec, dependent_parameter_names[2]))
            else:
                field_value = (
                    CLI_CONFIG_TO_FUNCTIONS[inference_fn](self.job_context, self.handler_metadata)
                    if inference_fn in CLI_CONFIG_TO_FUNCTIONS
                    else inference_fn
                )
            if field_value:
                write_nested_dict(spec, field_name, field_value)

        spec = apply_data_source_config(spec, self.job_context, self.handler_metadata)
        self.detailed_print("Loaded AutoML specs")

        self.num_gpu = get_num_gpus_from_spec(
            spec, "train", network=self.network, default=self.num_gpu,
            skip_gpu_conditions_check=is_remote_backend(self.job_context.backend_details)
        )
        self.num_nodes = get_num_nodes_from_spec(spec, "train", network=self.network, default=self.num_nodes)

        return spec

    def save_recommendation_specs(self):
        """Save recommendation specs to AutoML brain DB"""
        updated_spec = SPEC_BACKEND_TO_FUNCTIONS[self.api_params["spec_backend"]](self.spec)
        save_job_specs(job_id=self.job_name, specs=updated_spec, automl=True, automl_experiment_id=str(self.rec_number))

    def generate_run_command(self):
        """Generate the command to be run inside docker for AutoML experiment"""
        params_to_cli = build_cli_command(self.config)
        run_command = f"{self.network} train {params_to_cli}"
        logfile = f'/{self.job_name}.txt'
        run_command += f"  2>&1 | tee {logfile}"
        return run_command

    def get_recommendation_number(self):
        """Return the current recommendation number"""
        rec_number = None
        for dep in self.job_context.dependencies:
            if dep.type == "automl":
                rec_number = int(dep.name)
                break
        return rec_number

    def _cancellation_requested(self):
        """Check the durable parent/child cancellation fences."""
        lifecycle = get_automl_child_lifecycle(self.job_name)
        if lifecycle["cancel_requested"]:
            return True
        parent_metadata = get_handler_job_metadata(self.automl_brain_job_id) or {}
        if str(parent_metadata.get("status", "")).lower() in (
            "canceled", "canceling", "paused", "pausing"
        ):
            return True
        recommendations = get_automl_controller_info(self.automl_brain_job_id) or []
        if self.rec_number is None or self.rec_number >= len(recommendations):
            return True
        return str(recommendations[self.rec_number].get("status", "")).lower() in (
            "canceled", "canceling"
        )

    def _finish_canceled_launch(self, backend_may_exist=False):
        """Abort a fenced launch and persist quiescence only after confirmation."""
        if not backend_may_exist:
            return set_automl_child_launch_state(self.job_name, "quiescent")
        deleted = ExecutionHandler.delete_with_handler(
            self.job_name, workspace_metadata=self.workspace_metadata
        )
        if deleted is True:
            return set_automl_child_launch_state(self.job_name, "quiescent")
        set_automl_child_launch_state(
            self.job_name, "cancel_failed", "backend quiescence was not confirmed"
        )
        return False

    def monitor_job(self, nv_job_metadata=None):
        """Monitors the job status and updates job metadata"""
        if not self.spec:
            recommended_values = self.recs_dict[self.rec_number].get("specs", {})
            self.spec = self.generate_config(recommended_values)
            self.handle_ptm_anomalies()
            self.get_handler_cloud_details()
            self.save_recommendation_specs()

        if not self.job_env_variables:
            docker_env_vars = self.handler_metadata.get("docker_env_vars", {})
            self.decrypt_docker_env_vars(docker_env_vars)
            self.job_env_variables = copy.deepcopy(docker_env_vars)
            self.generate_env_variables(
                automl_brain_job_id=self.automl_brain_job_id,
                experiment_number=str(self.rec_number)
            )

        run_command = self.generate_run_command()
        if not nv_job_metadata:
            nv_job_metadata = {}
            self.generate_nv_job_metadata(nv_job_metadata, workspace_metadata=self.workspace_metadata)

        # Get results directory for SLURM support
        from nvidia_tao_core.microservices.handlers.execution_handlers.slurm_handler import (
            get_results_dir_for_workspace
        )
        results_dir = get_results_dir_for_workspace(
            workspace_metadata=self.workspace_metadata,
            job_id=self.job_name,
            specs=None
        )

        k8s_status = ExecutionHandler.get_job_status_with_handler(
            job_name=self.job_name,
            workspace_metadata=self.workspace_metadata,
            handler_id=self.handler_id,
            handler_kind=self.handler_kind,
            network=self.network,
            action=self.action,
            specs=None,
            docker_env_vars=self.job_env_variables,
            results_dir=results_dir,
            automl_exp_job=True,
            automl_experiment_id=str(self.rec_number),
            org_name=self.job_context.org_name
        )
        while k8s_status in ["Done", "Error", "Running", "Pending", "Creating"]:
            time.sleep(5)
            if get_dnn_status(
                self.automl_brain_job_id,
                automl=True,
                experiment_number=str(self.rec_number)
            ):
                break
            job_metadata = get_handler_job_metadata(self.automl_brain_job_id) or {}
            detailed_message = (
                (job_metadata.get("job_details") or {})
                .get(self.job_name, {})
            )
            detailed_message = (
                (detailed_message.get("detailed_status") or {})
                .get("message", "")
            )
            if "Invalid schema" in detailed_message:
                break
            if k8s_status == "Error":
                self.detailed_print(f"Relaunching job {self.job_name}")
                if not claim_automl_child_relaunch(self.job_name):
                    if self._cancellation_requested():
                        self._finish_canceled_launch(backend_may_exist=True)
                    break
                wait_for_job_completion(self.job_name)
                nv_job_metadata = {}
                self.generate_nv_job_metadata(nv_job_metadata, workspace_metadata=self.workspace_metadata)
                if self._cancellation_requested():
                    self._finish_canceled_launch(backend_may_exist=True)
                    break
                try:
                    if BACKEND in (Backend.LOCAL_K8S, Backend.LOCAL_DOCKER):
                        self.create_microservice_action_job(
                            self.automl_brain_job_id, nv_job_metadata=nv_job_metadata
                        )
                    else:
                        ExecutionHandler.create_job_with_handler(
                            org_name=self.job_context.org_name,
                            job_name=self.job_name,
                            image=self.image,
                            command=run_command,
                            workspace_metadata=self.workspace_metadata,
                            num_gpu=self.num_gpu,
                            num_nodes=self.num_nodes,
                            accelerator=self.accelerator,
                            docker_env_vars=self.job_env_variables,
                            nv_job_metadata=nv_job_metadata,
                            automl_brain=False,
                            automl_exp_job=True,
                            backend_details=self.job_context.backend_details
                        )
                except Exception as err:
                    set_automl_child_launch_state(
                        self.job_name, "uncertain", error=err
                    )
                    raise
                if self._cancellation_requested():
                    self._finish_canceled_launch(backend_may_exist=True)
                    break
                set_automl_child_launch_state(self.job_name, "running")
            k8s_status = ExecutionHandler.get_job_status_with_handler(
                job_name=self.job_name,
                workspace_metadata=self.workspace_metadata,
                handler_id=self.handler_id,
                handler_kind=self.handler_kind,
                network=self.network,
                action=self.action,
                specs=None,
                docker_env_vars=self.job_env_variables,
                results_dir=results_dir,
                automl_exp_job=True,
                automl_experiment_id=str(self.rec_number),
                org_name=self.job_context.org_name
            )
        if k8s_status == "Error":
            self.recs_dict[self.rec_number]["status"] = "failure"
            save_automl_controller_info(self.automl_brain_job_id, self.recs_dict)
            update_automl_details_metadata(self.automl_brain_job_id, self.handler_id, self.handler_kind)

    def run(self):
        """Calls necessary setup functions and calls job creation"""
        launch_claimed = False
        backend_create_started = False
        try:
            launch_claimed = claim_automl_child_launch(self.job_name)
            if not launch_claimed:
                lifecycle = get_automl_child_lifecycle(self.job_name)
                if lifecycle["cancel_requested"] and lifecycle["launch_state"] not in (
                    "submitting", "running", "uncertain", "cancel_failed"
                ):
                    set_automl_child_launch_state(self.job_name, "quiescent")
                logger.info(
                    "AutoML child %s did not acquire launch claim; state=%s cancel_requested=%s",
                    self.job_name,
                    lifecycle["launch_state"],
                    lifecycle["cancel_requested"],
                )
                return False
            if self._cancellation_requested():
                self._finish_canceled_launch(backend_may_exist=False)
                return False

            recommended_values = self.recs_dict[self.rec_number].get("specs", {})
            self.cs_instance, _ = create_cs_instance(self.workspace_metadata)
            self.add_ptm_dependency(recommended_values)

            self.spec = self.generate_config(recommended_values)
            self.handle_ptm_anomalies()
            self.get_handler_cloud_details()
            self.save_recommendation_specs()
            run_command = self.generate_run_command()

            self.detailed_print(run_command)

            # Wait for existing AutoML jobs to complete
            wait_for_job_completion(self.job_name)

            delete_lingering_checkpoints(self.recs_dict[self.rec_number].get("best_epoch_number", ""), self.expt_root)
            docker_env_vars = self.handler_metadata.get("docker_env_vars", {})
            self.decrypt_docker_env_vars(docker_env_vars)
            self.job_env_variables = copy.deepcopy(docker_env_vars)
            self.generate_env_variables(
                automl_brain_job_id=self.automl_brain_job_id,
                experiment_number=str(self.rec_number),
                automl_exp_job_id=self.job_name
            )
            nv_job_metadata = {}
            self.generate_nv_job_metadata(nv_job_metadata, workspace_metadata=self.workspace_metadata)

            if self._cancellation_requested():
                self._finish_canceled_launch(backend_may_exist=False)
                return False

            backend_create_started = True
            if BACKEND in (Backend.LOCAL_K8S, Backend.LOCAL_DOCKER):
                self.create_microservice_action_job(self.automl_brain_job_id, nv_job_metadata=nv_job_metadata)
            else:
                ExecutionHandler.create_job_with_handler(
                    org_name=self.job_context.org_name,
                    job_name=self.job_name,
                    image=self.image,
                    command=run_command,
                    workspace_metadata=self.workspace_metadata,
                    num_gpu=self.num_gpu,
                    num_nodes=self.num_nodes,
                    accelerator=self.accelerator,
                    docker_env_vars=self.job_env_variables,
                    nv_job_metadata=nv_job_metadata,
                    automl_brain=False,
                    automl_exp_job=False,
                    backend_details=self.job_context.backend_details
                )

            if self._cancellation_requested():
                self._finish_canceled_launch(backend_may_exist=True)
                return False
            if not set_automl_child_launch_state(self.job_name, "running"):
                raise RuntimeError(
                    f"Unable to persist running launch state for AutoML child {self.job_name}"
                )
            self.detailed_print(
                f"AutoML recommendation with experiment id {self.rec_number} "
                f"and job id {self.job_name} submitted"
            )

            # Transition controller recommendation: pending → started.
            # This is the AutoML equivalent of the Pending → Started write
            # that normal jobs do in tao.jobs. It tells _reclaim_stale_gpus
            # that the container/pod now exists and can be checked.
            latest_recommendations = get_automl_controller_info(
                self.automl_brain_job_id
            ) or []
            if (
                latest_recommendations and
                self.rec_number is not None and
                self.rec_number < len(latest_recommendations) and
                not self._cancellation_requested()
            ):
                prev_status = latest_recommendations[self.rec_number].get("status", "")
                if prev_status in ("pending", ""):
                    latest_recommendations[self.rec_number]["status"] = "started"
                    save_automl_controller_info(
                        self.automl_brain_job_id, latest_recommendations
                    )
                    self.recs_dict = latest_recommendations
                    logger.info(
                        f"[LIFECYCLE] AutoML rec {self.rec_number} (job {self.job_name}): "
                        f"{prev_status or '(none)'} → started (container created)"
                    )

            # Start log monitoring for AutoML recommendation job
            logger.debug(
                f"[ACTIONS] Checking if log monitoring should start for AutoML rec job "
                f"{self.job_name}, BACKEND={BACKEND}"
            )
            if BACKEND in (Backend.LOCAL_K8S, Backend.LOCAL_DOCKER):
                try:
                    logger.debug(
                        f"[ACTIONS] Starting log monitoring setup for AutoML rec job "
                        f"{self.job_name} (experiment {self.rec_number})"
                    )
                    # Get callback URL if available
                    callback_url = self.job_env_variables.get("TAO_LOGGING_SERVER_URL")
                    logger.debug(f"[ACTIONS] AutoML callback URL from env: {callback_url}")
                    if callback_url:
                        callback_url = callback_url + ":log_update"
                        logger.debug(f"[ACTIONS] AutoML full callback URL: {callback_url}")

                    # Get namespace for K8s
                    namespace = None
                    if BACKEND == Backend.LOCAL_K8S:
                        namespace = os.getenv("NAMESPACE")
                        logger.debug(f"[ACTIONS] AutoML K8s namespace from env: {namespace}")
                        if not namespace:
                            try:
                                namespace_file = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
                                logger.debug(f"[ACTIONS] Reading namespace from {namespace_file}")
                                with open(namespace_file, 'r', encoding='utf-8') as f:
                                    namespace = f.read().strip()
                                logger.debug(f"[ACTIONS] Got namespace from service account: {namespace}")
                            except Exception as e:
                                namespace = "default"
                                logger.debug(f"[ACTIONS] Using default namespace (error: {e})")

                    # Start monitoring for this recommendation job
                    logger.info(
                        f"[ACTIONS] Starting log monitoring for AutoML recommendation job {self.job_name} "
                        f"(experiment {self.rec_number}), namespace={namespace}"
                    )
                    start_monitoring_job(
                        self.job_name,
                        callback_url=callback_url,
                        namespace=namespace,
                        metadata={
                            'handler_id': self.handler_id,
                            'handler_kind': 'experiment',
                            'action': self.action,
                            'network': self.network,
                            'automl_brain_job_id': self.automl_brain_job_id,
                            'experiment_number': str(self.rec_number)
                        }
                    )
                    logger.info(
                        f"[ACTIONS] Successfully started log monitoring for AutoML recommendation job {self.job_name} "
                        f"(experiment {self.rec_number})"
                    )
                except Exception as e:
                    logger.warning(
                        f"[ACTIONS] Failed to start log monitoring for AutoML job {self.job_name}: "
                        f"{type(e).__name__}: {e}"
                    )
                    logger.debug("[ACTIONS] Exception details:", exc_info=True)
            else:
                logger.debug(f"[ACTIONS] Skipping log monitoring for backend {BACKEND}")

            self.monitor_job(nv_job_metadata)

            return True

        except Exception as e:
            lifecycle = get_automl_child_lifecycle(self.job_name)
            if launch_claimed and lifecycle["launch_state"] == "submitting":
                set_automl_child_launch_state(
                    self.job_name,
                    "uncertain" if backend_create_started else "quiescent",
                    error=e if backend_create_started else "",
                )
            if self._cancellation_requested():
                self._finish_canceled_launch(
                    backend_may_exist=backend_create_started
                )
                return False
            self.detailed_print(
                f"AutoMLpipeline for network {self.network} failed due to "
                f"exception {traceback.format_exc()}"
            )
            result_dict = {
                self.job_name: {
                    "detailed_status": {
                        "message": "Error due to unmet dependencies",
                        "status": "FAILURE"
                    }
                }
            }
            if isinstance(e, ValueError):
                result_dict = {self.job_name: {"detailed_status": {"message": str(e), "status": "FAILURE"}}}
            update_job_metadata(
                self.handler_id,
                self.automl_brain_job_id,
                metadata_key="job_details",
                data=result_dict,
                kind=self.handler_kind
            )
            self.detailed_print(self.job_name)

            latest_recommendations = get_automl_controller_info(
                self.automl_brain_job_id
            ) or []
            if self.rec_number is not None and self.rec_number < len(latest_recommendations):
                latest_recommendations[self.rec_number]["status"] = "failure"
                save_automl_controller_info(
                    self.automl_brain_job_id, latest_recommendations
                )
                self.recs_dict = latest_recommendations
            update_automl_details_metadata(self.automl_brain_job_id, self.handler_id, self.handler_kind)

            update_job_status(self.handler_id, self.job_context.id, status="Error", kind=self.handler_kind)
            deletion_confirmed = ExecutionHandler.delete_with_handler(
                self.job_context.id, workspace_metadata=self.workspace_metadata
            )
            set_automl_child_launch_state(
                self.job_name,
                "quiescent" if deletion_confirmed is True else "uncertain",
                "" if deletion_confirmed is True else "backend quiescence was not confirmed",
            )
            return False
        finally:
            if launch_claimed:
                lifecycle = get_automl_child_lifecycle(self.job_name)
                if lifecycle["launch_state"] == "submitting":
                    set_automl_child_launch_state(
                        self.job_name,
                        "uncertain" if backend_create_started else "quiescent",
                        "launch exited without a terminal handshake state",
                    )


# Each Element can be called with a job_context and returns an ActionPipeline (or its derivative) object


ACTIONS_TO_FUNCTIONS = {"train": TrainVal,
                        "evaluate": TrainVal,
                        "prune": CLIPipeline,
                        "prune_with_spec": TrainVal,
                        "retrain": TrainVal,
                        "export": CLIPipeline,
                        "export_with_spec": TrainVal,
                        "inference": TrainVal,
                        "gen_trt_engine": TrainVal,
                        "trtexec": TrainVal,
                        "purpose_built_models_ds_convert": TrainVal,
                        "odconvert": TrainVal,
                        "pyt_odconvert": TrainVal,
                        "data_services": TrainVal,
                        }
