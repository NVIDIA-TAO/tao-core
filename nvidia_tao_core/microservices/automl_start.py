# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoML main handler"""
import argparse
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict

from nvidia_tao_core.microservices.automl.controller import Controller
from nvidia_tao_core.microservices.automl.bayesian import Bayesian
from nvidia_tao_core.microservices.automl.hyperband import HyperBand
from nvidia_tao_core.microservices.automl.bohb import BOHB
from nvidia_tao_core.microservices.automl.bfbo import BFBO
from nvidia_tao_core.microservices.automl.asha import ASHA
from nvidia_tao_core.microservices.automl.pbt import PBT
from nvidia_tao_core.microservices.automl.dehb import DEHB
from nvidia_tao_core.microservices.automl.hyperband_es import HyperBandES
from nvidia_tao_core.microservices.automl.params import generate_hyperparams_to_search
from nvidia_tao_core.microservices.utils.handler_utils import JobContext
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    update_job_status,
    update_job_metadata,
    update_job_message,
    get_job_specs
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
logger.info(f"Logging configured at level: {TAO_LOG_LEVEL}")


# Constants for algorithm names
class AlgorithmType:
    """Constants for AutoML algorithm types"""

    BAYESIAN = ("bayesian", "b")
    BFBO = ("bfbo",)
    HYPERBAND = ("hyperband", "h")
    BOHB = ("bohb",)
    ASHA = ("asha",)
    PBT = ("pbt",)
    DEHB = ("dehb",)
    HYPERBAND_ES = ("hyperband_es", "hes")


@dataclass
class AlgorithmParams:
    """Dataclass to hold algorithm-specific parameters with defaults"""

    automl_max_recommendations: int = 20
    automl_max_epochs: int = 27
    automl_reduction_factor: int = 3
    epoch_multiplier: int = 1
    automl_max_concurrent: int = 4
    automl_population_size: int = 10
    automl_max_generations: int = 20
    automl_eval_interval: int = 10
    automl_perturbation_factor: float = 1.2
    automl_mutation_factor: float = 0.5
    automl_crossover_prob: float = 0.5
    automl_early_stop_threshold: float = 0.1
    automl_min_early_stop_epochs: int = 3
    automl_kde_samples: int = 64
    automl_top_n_percent: float = 15.0
    automl_min_points_in_model: int = 10
    automl_max_trials: int = None  # ASHA: max configs to try (None = unlimited)
    automl_min_top_configs: int = 5  # ASHA: min configs that must reach final rung before stopping

    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> 'AlgorithmParams':
        """Create AlgorithmParams from dictionary with defaults"""
        return cls(
            automl_max_recommendations=params_dict.get("automl_max_recommendations", 20),
            automl_max_epochs=params_dict.get("automl_max_epochs", 27),
            automl_reduction_factor=params_dict.get("automl_reduction_factor", 3),
            epoch_multiplier=params_dict.get("epoch_multiplier", 1),
            automl_max_concurrent=params_dict.get("automl_max_concurrent", 4),
            automl_population_size=params_dict.get("automl_population_size", 10),
            automl_max_generations=params_dict.get("automl_max_generations", 20),
            automl_eval_interval=params_dict.get("automl_eval_interval", 10),
            automl_perturbation_factor=params_dict.get("automl_perturbation_factor", 1.2),
            automl_mutation_factor=params_dict.get("automl_mutation_factor", 0.5),
            automl_crossover_prob=params_dict.get("automl_crossover_prob", 0.5),
            automl_early_stop_threshold=params_dict.get("automl_early_stop_threshold", 0.1),
            automl_min_early_stop_epochs=params_dict.get("automl_min_early_stop_epochs", 3),
            automl_kde_samples=params_dict.get("automl_kde_samples", 64),
            automl_top_n_percent=params_dict.get("automl_top_n_percent", 15.0),
            automl_min_points_in_model=params_dict.get("automl_min_points_in_model", 10),
            automl_max_trials=params_dict.get("automl_max_trials", None),
            automl_min_top_configs=params_dict.get("automl_min_top_configs", 5)
        )


class BrainFactory:
    """Factory class for creating AutoML brain instances"""

    @staticmethod
    def create_brain(
        algorithm: str,
        jc: JobContext,
        root: str,
        network: str,
        parameters: Any,
        params: AlgorithmParams,
        metric: str = "loss",
        resume: bool = False
    ):
        """Create brain instance based on algorithm type

        Args:
            metric: Metric to optimize (e.g., 'loss', 'val_accuracy', 'mIoU')
        """
        algo_lower = algorithm.lower()

        if algo_lower in AlgorithmType.HYPERBAND:
            brain_class = HyperBand
            kwargs = {
                "job_context": jc,
                "root": root,
                "network": network,
                "parameters": parameters,
                "max_epochs": int(params.automl_max_epochs),
                "reduction_factor": int(params.automl_reduction_factor),
                "epoch_multiplier": int(params.epoch_multiplier),
                "metric": metric
            }
        elif algo_lower in AlgorithmType.BAYESIAN:
            brain_class = Bayesian
            kwargs = {
                "job_context": jc,
                "root": root,
                "network": network,
                "parameters": parameters
            }
        elif algo_lower in AlgorithmType.BOHB:
            brain_class = BOHB
            kwargs = {
                "job_context": jc,
                "root": root,
                "network": network,
                "parameters": parameters,
                "max_epochs": int(params.automl_max_epochs),
                "reduction_factor": int(params.automl_reduction_factor),
                "epoch_multiplier": int(params.epoch_multiplier),
                "kde_samples": int(params.automl_kde_samples),
                "top_n_percent": float(params.automl_top_n_percent),
                "min_points_in_model": int(params.automl_min_points_in_model),
                "metric": metric
            }
        elif algo_lower in AlgorithmType.BFBO:
            brain_class = BFBO
            kwargs = {
                "job_context": jc,
                "root": root,
                "network": network,
                "parameters": parameters
            }
        elif algo_lower in AlgorithmType.ASHA:
            brain_class = ASHA
            kwargs = {
                "job_context": jc,
                "root": root,
                "network": network,
                "parameters": parameters,
                "max_epochs": int(params.automl_max_epochs),
                "reduction_factor": int(params.automl_reduction_factor),
                "epoch_multiplier": int(params.epoch_multiplier),
                "max_concurrent": int(params.automl_max_concurrent),
                "max_trials": params.automl_max_trials if params.automl_max_trials else None,
                "min_top_configs": int(params.automl_min_top_configs),
                "metric": metric
            }
        elif algo_lower in AlgorithmType.PBT:
            brain_class = PBT
            kwargs = {
                "job_context": jc,
                "root": root,
                "network": network,
                "parameters": parameters,
                "population_size": int(params.automl_population_size),
                "max_generations": int(params.automl_max_generations),
                "eval_interval": int(params.automl_eval_interval),
                "perturbation_factor": float(params.automl_perturbation_factor),
                "metric": metric
            }
        elif algo_lower in AlgorithmType.DEHB:
            brain_class = DEHB
            kwargs = {
                "job_context": jc,
                "root": root,
                "network": network,
                "parameters": parameters,
                "max_epochs": int(params.automl_max_epochs),
                "reduction_factor": int(params.automl_reduction_factor),
                "epoch_multiplier": int(params.epoch_multiplier),
                "mutation_factor": float(params.automl_mutation_factor),
                "crossover_prob": float(params.automl_crossover_prob),
                "metric": metric
            }
        elif algo_lower in AlgorithmType.HYPERBAND_ES:
            brain_class = HyperBandES
            kwargs = {
                "job_context": jc,
                "root": root,
                "network": network,
                "parameters": parameters,
                "max_epochs": int(params.automl_max_epochs),
                "reduction_factor": int(params.automl_reduction_factor),
                "epoch_multiplier": int(params.epoch_multiplier),
                "early_stop_threshold": float(params.automl_early_stop_threshold),
                "min_early_stop_epochs": int(params.automl_min_early_stop_epochs)
            }
        else:
            raise ValueError(f"AutoML Algorithm {algorithm} is not valid")

        # Create brain instance (load_state for resume, new instance otherwise)
        if resume:
            return brain_class.load_state(**kwargs)
        return brain_class(**kwargs)


def automl_start(
    root: str,
    network: str,
    jc: JobContext,
    resume: bool,
    automl_algorithm: str,
    automl_delete_intermediate_ckpt: bool,
    metric: str,
    algorithm_specific_params: Dict[str, Any],
    automl_hyperparameters: Any,
    override_automl_disabled_params: bool,
    decrypted_workspace_metadata: Dict[str, Any]
) -> None:
    """Starts the AutoML controller with specified configuration.

    Args:
        root: Root directory for job files
        network: Network architecture name
        jc: Job context with metadata
        resume: Whether to resume from previous state
        automl_algorithm: AutoML algorithm to use
        automl_delete_intermediate_ckpt: Whether to delete intermediate checkpoints
        metric: Metric to optimize
        algorithm_specific_params: Algorithm-specific parameters dictionary
        automl_hyperparameters: Hyperparameters to search
        override_automl_disabled_params: Whether to override disabled parameters
        decrypted_workspace_metadata: Workspace metadata
    """
    # Generate hyperparameters to search
    parameters, parameter_names = generate_hyperparams_to_search(
        jc,
        automl_hyperparameters,
        "/".join(root.split("/")[0:-2]),
        override_automl_disabled_params
    )

    # Parse algorithm-specific parameters
    params = AlgorithmParams.from_dict(
        algorithm_specific_params if isinstance(algorithm_specific_params, dict) else {}
    )

    # Check if automl algorithm is valid for specific use-cases
    if network == "classification_pyt":
        if "model.head.type" in parameter_names and automl_algorithm == "hyperband":
            error_message = (
                "Hyperband not supported when non-epoch based models are chosen. "
                "Change algorithm to bayesian"
            )
            result = {"message": error_message}
            update_job_metadata(jc.handler_id, jc.id, metadata_key="job_details", data=result, kind="experiments")
            raise ValueError(error_message)

    # Create brain using factory pattern
    brain = BrainFactory.create_brain(
        algorithm=automl_algorithm,
        jc=jc,
        root=root,
        network=network,
        parameters=parameters,
        params=params,
        metric=metric,
        resume=resume
    )

    # Create and start controller
    if resume:
        controller = Controller.load_state(
            root,
            network,
            brain,
            jc,
            params,
            automl_delete_intermediate_ckpt,
            metric,
            automl_algorithm.lower(),
            decrypted_workspace_metadata,
            parameter_names
        )
    else:
        controller = Controller(
            root,
            network,
            brain,
            jc,
            params,
            automl_delete_intermediate_ckpt,
            metric,
            automl_algorithm.lower(),
            decrypted_workspace_metadata,
            parameter_names
        )

    controller.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='AutoML controller', description='Run AutoML.')
    parser.add_argument(
        '--user_id',
        type=str,
    )
    parser.add_argument(
        '--org_name',
        type=str,
    )
    parser.add_argument(
        '--name',
        type=str,
    )
    parser.add_argument(
        '--root',
        type=str,
    )
    parser.add_argument(
        '--automl_job_id',
        type=str,
    )
    parser.add_argument(
        '--network',
        type=str,
    )
    parser.add_argument(
        '--experiment_id',
        type=str,
    )
    parser.add_argument(
        '--resume',
        type=str,
    )
    parser.add_argument(
        '--automl_algorithm',
        type=str,
    )
    parser.add_argument(
        '--automl_delete_intermediate_ckpt',
        type=str,
    )
    parser.add_argument(
        '--metric',
        type=str,
    )
    parser.add_argument(
        '--algorithm_specific_params',
        type=str,
    )
    parser.add_argument(
        '--automl_hyperparameters',
        type=str,
    )
    parser.add_argument(
        '--override_automl_disabled_params',
        type=str,
    )
    parser.add_argument(
        '--retain_checkpoints_for_resume',
        type=str,
        default='False'
    )
    parser.add_argument(
        '--timeout_minutes',
        type=str,
        default='60'
    )
    parser.add_argument(
        '--decrypted_workspace_metadata',
        type=json.loads,
    )

    parser.add_argument(
        '--backend_details',
        type=str,
        help='Backend details as JSON string'
    )

    args = parser.parse_args()
    automl_job_id = args.automl_job_id
    handler_id = args.experiment_id
    network = args.network
    user_id = args.user_id
    org_name = args.org_name
    try:
        root = args.root
        name = args.name
        backend_details = None
        if args.backend_details:
            backend_details = json.loads(args.backend_details)
        specs = get_job_specs(automl_job_id)

        # Get retain_checkpoints_for_resume from CLI argument
        retain_checkpoints_for_resume = args.retain_checkpoints_for_resume.lower() in ("true", "1")
        timeout_minutes = int(args.timeout_minutes)

        from nvidia_tao_core.microservices.utils.handler_utils import get_num_gpus_from_spec, is_remote_backend
        num_gpu = get_num_gpus_from_spec(
            specs, "train", network=network, default=-1,
            skip_gpu_conditions_check=is_remote_backend(backend_details)
        )
        logger.debug(
            f"[AUTOML-START] AutoML brain job {automl_job_id}: num_gpu from spec = {num_gpu}, "
            f"NUM_GPU_PER_NODE={os.getenv('NUM_GPU_PER_NODE', '0')}"
        )

        jc = JobContext(
            automl_job_id,
            None,
            network,
            "train",
            handler_id,
            user_id,
            org_name,
            "experiment",
            name=name,
            num_gpu=num_gpu,
            backend_details=backend_details,
            specs=specs,
            retain_checkpoints_for_resume=retain_checkpoints_for_resume,
            timeout_minutes=timeout_minutes
        )
        resume = args.resume == "True"
        logger.debug(
            f"[AUTOML-START-ARGS] Parsed resume argument: args.resume='{args.resume}', "
            f"resume={resume}, job_id={automl_job_id}"
        )
        automl_algorithm = args.automl_algorithm
        automl_delete_intermediate_ckpt = args.automl_delete_intermediate_ckpt
        metric = args.metric
        # Parse algorithm_specific_params and automl_hyperparameters - normalized to JSON by handler
        algorithm_specific_params = json.loads(args.algorithm_specific_params)
        automl_hyperparameters = json.loads(args.automl_hyperparameters)
        override_automl_disabled_params = args.override_automl_disabled_params == "True"
        decrypted_workspace_metadata = args.decrypted_workspace_metadata
        automl_start(
            root=root,
            network=network,
            jc=jc,
            resume=resume,
            automl_algorithm=automl_algorithm,
            automl_delete_intermediate_ckpt=automl_delete_intermediate_ckpt,
            metric=metric,
            algorithm_specific_params=algorithm_specific_params,
            automl_hyperparameters=automl_hyperparameters,
            override_automl_disabled_params=override_automl_disabled_params,
            decrypted_workspace_metadata=decrypted_workspace_metadata)

    except Exception as e:
        logger.error("AutoML start for network %s failed due to exception %s", network, traceback.format_exc())
        update_job_status(handler_id, automl_job_id, status="Error", kind="experiments")
        error_message = {
            "date": datetime.now(tz=timezone.utc).strftime("%m/%d/%Y"),
            "time": datetime.now(tz=timezone.utc).strftime("%H:%M:%S"),
            "status": "Error",
            "message": str(e)
        }
        update_job_message(handler_id, automl_job_id, "experiments", error_message)
