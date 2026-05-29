# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ASHA (Asynchronous Successive Halving Algorithm) AutoML algorithm modules"""
import numpy as np
import math
import logging
from collections import defaultdict

from nvidia_tao_core.microservices.utils.automl_utils import (
    ResumeRecommendation, JobStates, get_valid_range, clamp_value
)
from nvidia_tao_core.microservices.automl.automl_algorithm_base import AutoMLAlgorithmBase, is_nan_value
from nvidia_tao_core.microservices.utils.handler_utils import get_flatten_specs
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    save_job_specs,
    get_job_specs,
    save_automl_brain_info,
    get_automl_brain_info
)
from nvidia_tao_core.microservices.automl import network_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ASHA(AutoMLAlgorithmBase):
    """ASHA (Asynchronous Successive Halving Algorithm) AutoML algorithm class

    ASHA improves upon Hyperband by allowing truly asynchronous parallel execution.
    Key differences from Hyperband:
    1. NO synchronization barriers - launches new configs immediately when slots free
    2. Progressive promotion quota: promotes when floor(m/reduction_factor) threshold is met
       where m = number of completions at a rung
    3. Configs at different rungs can run simultaneously (mixed-rung execution)
    4. Constant GPU utilization (always fills up to max_concurrent slots)

    This implementation follows the theoretical ASHA algorithm from:
    "A System for Massively Parallel Hyperparameter Tuning" (Li et al., 2020)
    """

    def __init__(
        self, job_context, root, network, parameters, max_epochs,
        reduction_factor, epoch_multiplier, max_concurrent=4, max_trials=None,
        min_top_configs=5, metric="loss"
    ):
        """Initialize the ASHA algorithm class

        Args:
            root: handler root
            network: model we are running AutoML on
            parameters: automl sweepable parameters
            max_epochs: the maximum amount of resource that can be allocated to a single configuration
            reduction_factor: reduction factor for successive halving (typically 3 or 4)
            epoch_multiplier: multiplying factor for epochs
            max_concurrent: maximum number of concurrent training jobs
            max_trials: maximum number of configurations to try (None = unlimited until min_top_configs reached)
            min_top_configs: minimum number of configs that must complete all rungs before stopping (default: 5)
            metric: metric to optimize (e.g., 'loss', 'val_accuracy', 'mIoU')
        """
        super().__init__(job_context, root, network, parameters)
        self.epoch_multiplier = int(epoch_multiplier)
        self.reduction_factor = int(reduction_factor)
        self.max_epochs = int(max_epochs)
        self.max_concurrent = int(max_concurrent)
        self.max_trials = max_trials
        self.min_top_configs = int(min_top_configs)
        self.metric = metric

        # Calculate rungs (resource levels) using paper's formula
        # K = floor(log_reduction_factor(max_epochs)),
        # rungs = [r0, r0*reduction_factor, r0*reduction_factor^2, ..., max_epochs]
        # Use geometric progression to avoid truncation drift
        K = int(math.floor(math.log(max_epochs) / math.log(reduction_factor)))
        r0 = max(1, int(math.floor(max_epochs / (reduction_factor ** K))))
        self.rungs = [(r0 * (reduction_factor ** i)) * self.epoch_multiplier for i in range(K + 1)]
        # Ensure last rung equals exactly max_epochs * epoch_multiplier
        self.rungs[-1] = max_epochs * self.epoch_multiplier
        logger.info(f"ASHA rungs (epochs): {self.rungs}")

        # State tracking
        self.rung_results = defaultdict(list)  # rung -> [(config_id, result)]
        self.config_to_rung = {}  # config_id -> current rung
        self.active_configs = set()  # Currently training config IDs
        self.pending_promotions = []  # Configs waiting to be promoted
        self.completed_configs = set()  # Configs that reached max rung
        self.config_specs = {}  # config_id -> specs
        self.next_config_id = 0
        self.total_configs_started = 0  # Total configs launched (for max_trials check)
        self.complete = False

        # Determine reverse_sort based on metric (same logic as controller)
        # Default: higher is better (accuracy, mIoU, etc.)
        self.reverse_sort = True
        # For loss metrics: lower is better
        if metric == "loss" or "loss" in metric.lower() or metric.lower() in ("evaluation_cost",):
            self.reverse_sort = False
        # Progressive promotion tracking
        # rung -> number of completions (including failures)
        self.rung_completions = defaultdict(int)
        self.rung_promotions = defaultdict(int)  # rung -> number of promotions so far
        # rung -> set of config_ids already promoted (prevents double-promotion)
        self.promoted_from_rung = defaultdict(set)
        # Current epoch target (used for early_stop_epoch by controller)
        self.epoch_number = self.rungs[0]  # Start with first rung

        # For ETA calculation compatibility with controller
        # ASHA doesn't have fixed brackets like Hyperband, but we provide a
        # simplified structure
        # Represents a single "virtual bracket" with ASHA's rung structure
        # Assume max_concurrent configs per rung
        self.ni = {"0": [max_concurrent] * len(self.rungs)}
        # Resource levels (in R units)
        self.ri = {"0": [r // self.epoch_multiplier for r in self.rungs]}
        self.bracket = "0"  # ASHA has single virtual bracket
        self.sh_iter = 0  # Current rung index (0-based)
        self.expt_iter = 0  # Number of completed experiments in current iteration (always 0 for async ASHA)

        # Override initial num epochs to MAX (training will be interrupted
        # at rungs via early_stop_epoch)
        self.override_num_epochs(self.rungs[-1])

        logger.info(
            f"ASHA initialized with max_epochs={max_epochs}, "
            f"reduction_factor={reduction_factor}, max_concurrent={max_concurrent}"
        )
        logger.info(f"Resource rungs: {self.rungs}")

    def override_num_epochs(self, num_epochs):
        """Override num epochs parameter in train spec file"""
        spec = get_job_specs(self.job_context.id)
        for key1 in spec:
            if key1 in ("training_config", "train_config", "train"):
                for key2 in spec[key1]:
                    if key2 in ("num_epochs", "epochs", "n_epochs", "max_iters", "epoch"):
                        spec[key1][key2] = num_epochs
                    elif key2 in ("train_config"):
                        for key3 in spec[key1][key2]:
                            if key3 == "runner":
                                for key4 in spec[key1][key2][key3]:
                                    if key4 == "max_epochs":
                                        spec[key1][key2][key3][key4] = num_epochs
            elif key1 in ("num_epochs"):
                spec[key1] = num_epochs
        save_job_specs(self.job_context.id, spec)

    def generate_automl_param_rec_value(self, parameter_config):
        """Generate a random value for the parameter passed"""
        parameter_name = parameter_config.get("parameter")

        # Apply custom overrides if provided
        if self.custom_ranges and parameter_name in self.custom_ranges:
            for override_key, override_value in self.custom_ranges[parameter_name].items():
                if override_value is not None:
                    parameter_config[override_key] = override_value

        tp = parameter_config.get("value_type")
        default_value = parameter_config.get("default_value", None)
        math_cond = parameter_config.get("math_cond", None)
        parent_param = parameter_config.get("parent_param", None)

        if tp == "float":
            v_min = parameter_config.get("valid_min", "")
            v_max = parameter_config.get("valid_max", "")

            # If no valid range, generate diverse values around default
            if v_min == "" or v_max == "":
                if default_value is not None and default_value != "":
                    default_val = float(default_value)
                    if default_val > 0:
                        v_min = default_val / 10.0
                        v_max = default_val * 10.0
                    elif default_val < 0:
                        v_min = default_val * 10.0
                        v_max = default_val / 10.0
                    else:
                        v_min = -1.0
                        v_max = 1.0
                    random_float = np.random.uniform(v_min, v_max)
                    logger.info(f"Generated random float for {parameter_name} (no range): {random_float}")
                    return random_float
                return np.random.uniform(0.0, 1.0)

            if is_nan_value(v_min) or is_nan_value(v_max):
                if default_value is not None:
                    default_val = float(default_value)
                    if default_val > 0:
                        return np.random.uniform(default_val / 10.0, default_val * 10.0)
                    return np.random.uniform(0.0, 1.0)
                return np.random.uniform(0.0, 1.0)

            # Handle list-based ranges (e.g., per-model-part learning rates)
            if isinstance(v_min, list) or isinstance(v_max, list):
                if isinstance(v_min, list) and isinstance(v_max, list):
                    base_min = float(v_min[0]) if v_min else 0.0
                    base_max = float(v_max[0]) if v_max else 1.0
                elif isinstance(v_min, list):
                    base_min = float(v_min[0]) if v_min else 0.0
                    base_max = float(v_max) if v_max not in (None, '', "") else base_min * 10
                else:
                    base_min = float(v_min) if v_min not in (None, '', "") else 0.0
                    base_max = float(v_max[0]) if v_max else 1.0

                if base_min > 0 and base_max > 0:
                    log_min = np.log10(base_min)
                    log_max = np.log10(base_max)
                    base_value = float(10 ** np.random.uniform(log_min, log_max))
                else:
                    base_value = float(np.random.uniform(base_min, base_max))

                # Check for disable_list option - if True, skip network-specific logic
                # and return pure float value for optimization
                disable_list = parameter_config.get("disable_list", False)
                if disable_list:
                    logger.info(
                        f"disable_list=True for {parameter_name}: "
                        f"returning pure float {base_value} (skipping network-specific logic)"
                    )
                    return base_value

                return network_utils.apply_network_specific_param_logic(
                    network=self.network,
                    data_type=tp,
                    parameter_name=parameter_name,
                    value=base_value,
                    v_max=v_max,
                    default_train_spec=self.default_train_spec,
                    parent_params=self.parent_params
                )

            v_min, v_max = get_valid_range(parameter_config, self.parent_params, self.custom_ranges)

            # Check for disable_list option early - log the parameter config for debugging
            disable_list = parameter_config.get("disable_list", False)
            logger.debug(
                f"[ASHA] Parameter {parameter_name}: v_min={v_min}, v_max={v_max}, "
                f"disable_list={disable_list}"
            )

            # Apply math condition if specified
            # Skip relational constraints (like "> depends_on") as they're handled in base class
            if math_cond and type(math_cond) is str and "depends_on" not in math_cond:
                parts = math_cond.split(" ")
                if len(parts) >= 2:
                    operator = parts[0]
                    factor = int(float(parts[1]))
                    if operator == "^":
                        fallback = np.random.uniform(low=v_min, high=v_max)
                        fallback = clamp_value(fallback, v_min, v_max)
                        random_float = float(self._apply_power_constraint_with_equal_priority(
                            v_min, v_max, factor, fallback))
                    else:
                        random_float = np.random.uniform(low=v_min, high=v_max)
                        random_float = clamp_value(random_float, v_min, v_max)
            else:
                random_float = np.random.uniform(low=v_min, high=v_max)
                random_float = clamp_value(random_float, v_min, v_max)

            if not (type(parent_param) is float and math.isnan(parent_param)):
                if ((type(parent_param) is str and parent_param != "nan" and parent_param == "TRUE") or
                        (type(parent_param) is bool and parent_param)):
                    self.parent_params[parameter_config.get("parameter")] = random_float

            # Check for disable_list option - if True, skip network-specific logic
            if disable_list:
                logger.info(
                    f"disable_list=True for {parameter_name}: "
                    f"returning pure float {random_float} (skipping network-specific logic)"
                )
                return random_float

            # Apply network-specific parameter logic
            return network_utils.apply_network_specific_param_logic(
                network=self.network,
                data_type=tp,
                parameter_name=parameter_name,
                value=random_float,
                v_max=v_max,
                default_train_spec=self.default_train_spec,
                parent_params=self.parent_params
            )

        return super().generate_automl_param_rec_value(parameter_config)

    def save_state(self):
        """Save the ASHA algorithm related variables to brain metadata"""
        state_dict = {}
        state_dict["next_config_id"] = self.next_config_id
        state_dict["total_configs_started"] = self.total_configs_started
        state_dict["complete"] = self.complete
        state_dict["epoch_multiplier"] = self.epoch_multiplier
        state_dict["rungs"] = self.rungs
        # Convert defaultdict to dict with string keys for MongoDB (requires string keys)
        state_dict["rung_results"] = {str(k): v for k, v in self.rung_results.items()}
        state_dict["rung_completions"] = {str(k): v for k, v in self.rung_completions.items()}
        state_dict["rung_promotions"] = {str(k): v for k, v in self.rung_promotions.items()}
        state_dict["promoted_from_rung"] = {str(k): list(v) for k, v in self.promoted_from_rung.items()}
        # Convert integer config IDs to strings for MongoDB
        state_dict["config_to_rung"] = {str(k): v for k, v in self.config_to_rung.items()}
        state_dict["active_configs"] = list(self.active_configs)
        state_dict["completed_configs"] = list(self.completed_configs)
        state_dict["config_specs"] = {str(k): v for k, v in self.config_specs.items()}
        state_dict["pending_promotions"] = self.pending_promotions
        state_dict["epoch_number"] = self.epoch_number
        state_dict["ni"] = self.ni
        state_dict["ri"] = self.ri
        state_dict["bracket"] = self.bracket
        state_dict["sh_iter"] = self.sh_iter
        state_dict["expt_iter"] = self.expt_iter
        state_dict["max_trials"] = self.max_trials
        state_dict["min_top_configs"] = self.min_top_configs
        state_dict["metric"] = self.metric

        save_automl_brain_info(self.job_context.id, state_dict)

    @staticmethod
    def load_state(
        job_context, root, network, parameters, max_epochs, reduction_factor,
        epoch_multiplier, max_concurrent=4, max_trials=None, min_top_configs=5,
        metric="loss"
    ):
        """Load the ASHA algorithm related variables from brain metadata"""
        json_loaded = get_automl_brain_info(job_context.id)
        if not json_loaded:
            return ASHA(
                job_context, root, network, parameters, max_epochs,
                reduction_factor, epoch_multiplier, max_concurrent, max_trials,
                min_top_configs, metric
            )

        brain = ASHA(
            job_context, root, network, parameters, max_epochs,
            reduction_factor, epoch_multiplier, max_concurrent, max_trials,
            min_top_configs, metric
        )
        # Load state
        brain.next_config_id = json_loaded["next_config_id"]
        brain.total_configs_started = json_loaded.get(
            "total_configs_started", brain.next_config_id
        )
        brain.complete = json_loaded["complete"]
        # Convert string keys back to integers (MongoDB requires string keys)
        brain.rung_results = defaultdict(
            list, {int(k): v for k, v in json_loaded["rung_results"].items()}
        )
        brain.rung_completions = defaultdict(
            int, {int(k): v for k, v in json_loaded.get("rung_completions", {}).items()}
        )
        brain.rung_promotions = defaultdict(
            int, {int(k): v for k, v in json_loaded.get("rung_promotions", {}).items()}
        )
        brain.promoted_from_rung = defaultdict(
            set, {int(k): set(v) for k, v in json_loaded.get("promoted_from_rung", {}).items()}
        )
        brain.config_to_rung = {int(k): v for k, v in json_loaded["config_to_rung"].items()}
        brain.active_configs = set(json_loaded["active_configs"])
        brain.completed_configs = set(json_loaded["completed_configs"])
        brain.config_specs = {int(k): v for k, v in json_loaded["config_specs"].items()}
        brain.pending_promotions = json_loaded.get("pending_promotions", [])
        brain.epoch_number = json_loaded.get("epoch_number", brain.rungs[0])
        brain.ni = json_loaded.get("ni", brain.ni)
        brain.ri = json_loaded.get("ri", brain.ri)
        brain.bracket = json_loaded.get("bracket", "0")
        brain.sh_iter = json_loaded.get("sh_iter", 0)
        brain.expt_iter = json_loaded.get("expt_iter", 0)
        # Load max_trials, min_top_configs, and metric from persisted state (with fallback to constructor values)
        brain.max_trials = json_loaded.get("max_trials", max_trials)
        brain.min_top_configs = json_loaded.get("min_top_configs", min_top_configs)
        brain.metric = json_loaded.get("metric", metric)

        # Re-determine reverse_sort based on loaded metric (in case it changed)
        brain.reverse_sort = True
        if brain.metric == "loss" or "loss" in brain.metric.lower() or brain.metric.lower() in ("evaluation_cost",):
            brain.reverse_sort = False

        return brain

    def _generate_random_parameters(self):
        """Generate random parameter values for a new configuration"""
        hyperparam_dict = {}
        for param in self.parameters:
            name = param["parameter"]
            rec = self.generate_automl_param_rec_value(param)
            logger.info(f"Generated random parameter in ASHA: {name} = {rec}")
            hyperparam_dict[name] = rec
        return hyperparam_dict

    def done(self):
        """Return if ASHA algorithm is complete or not"""
        return self.complete

    def generate_recommendations(self, history):
        """Generate recommendations asynchronously"""
        get_flatten_specs(self.default_train_spec, self.default_train_spec_flattened)

        if history == []:
            # Initial recommendations - fill all available worker slots
            recommendations = []
            for _ in range(self.max_concurrent):
                if self.max_trials is not None and self.total_configs_started >= self.max_trials:
                    break
                specs = self._generate_random_parameters()
                self.config_specs[self.next_config_id] = specs
                self.config_to_rung[self.next_config_id] = 0
                self.active_configs.add(self.next_config_id)
                self.total_configs_started += 1
                self.next_config_id += 1
                recommendations.append(specs)
            self.track_id = 0
            return recommendations if recommendations else [self._generate_random_parameters()]

        # Log current state for debugging
        active_by_rung = defaultdict(list)
        for rec in history:
            if rec.status in [JobStates.pending, JobStates.started, JobStates.running]:
                rung_idx = self.config_to_rung.get(rec.id, 0)
                active_by_rung[rung_idx].append(rec.id)

        if active_by_rung:
            rung_summary = ", ".join([
                f"r{idx}:[{','.join(map(str, configs))}]"
                for idx, configs in sorted(active_by_rung.items())
            ])
            logger.info(f"ASHA State: Active configs by rung: {rung_summary}")

        # Process completed configurations
        recommendations = []
        for rec in history:
            if rec.status in [JobStates.success, JobStates.failure] and rec.id in self.active_configs:
                # Configuration just completed
                self.active_configs.discard(rec.id)
                current_rung_idx = self.config_to_rung.get(rec.id, 0)
                rung_epochs = self.rungs[current_rung_idx]

                # Count ALL completions (success + failure) toward quota
                # This is critical for ASHA's floor(m/nu) promotion logic
                self.rung_completions[rung_epochs] += 1

                if rec.status == JobStates.success and rec.result is not None and math.isfinite(float(rec.result)):
                    # Record result at this rung (only successful configs can be promoted)
                    self.rung_results[rung_epochs].append((rec.id, rec.result))
                    logger.info(f"Config {rec.id} completed rung {current_rung_idx} with result {rec.result} "
                                f"(completion #{self.rung_completions[rung_epochs]} at this rung)")
                elif rec.status == JobStates.failure:
                    logger.info(f"Config {rec.id} failed at rung {current_rung_idx} "
                                f"(completion #{self.rung_completions[rung_epochs]} counted, not promotable)")

                # Check if quota increased and promote ALL eligible configs
                # ASHA paper: when m increases, check if floor(m/reduction_factor) > promotions_so_far
                # and promote the TOP unpromoted configs to fill the quota
                # This check happens AFTER any completion (success or failure) to handle quota increases
                if current_rung_idx < len(self.rungs) - 1:
                    next_rung_idx = current_rung_idx + 1
                    next_epochs = self.rungs[next_rung_idx]

                    # Calculate current quota
                    m = self.rung_completions[rung_epochs]
                    quota = int(m / self.reduction_factor)
                    promotions_so_far = self.rung_promotions[rung_epochs]

                    # If quota increased, promote best unpromoted configs
                    if quota > promotions_so_far:
                        # Get all successful results at this rung, sorted by performance
                        results_at_rung = list(self.rung_results[rung_epochs])
                        results_at_rung.sort(key=lambda x: x[1], reverse=self.reverse_sort)

                        # Find configs to promote (top quota configs that haven't been promoted)
                        for rank, (config_id, result) in enumerate(results_at_rung):
                            if rank >= quota:
                                break  # Beyond quota

                            if config_id in self.promoted_from_rung[rung_epochs]:
                                continue  # Already promoted

                            # Promote this config
                            self.promoted_from_rung[rung_epochs].add(config_id)
                            self.rung_promotions[rung_epochs] += 1
                            self.config_to_rung[config_id] = next_rung_idx
                            self.pending_promotions.append((config_id, next_epochs))
                            logger.info(
                                f"✓ QUEUED for promotion: Config {config_id} (rank {rank + 1}/"
                                f"{len(results_at_rung)}, result={result:.4f}) → rung "
                                f"{next_rung_idx} ({next_epochs} epochs). "
                                f"Quota: {quota}, promotions: {self.rung_promotions[rung_epochs]}"
                            )

                            if self.rung_promotions[rung_epochs] >= quota:
                                break  # Quota filled

                    # Log status for the config that just completed
                    if rec.id in self.promoted_from_rung[rung_epochs]:
                        logger.info(f"Config {rec.id} will be promoted")
                    else:
                        logger.info(f"Config {rec.id} eliminated at rung {current_rung_idx}")
                else:
                    # Already at max rung
                    self.completed_configs.add(rec.id)
                    logger.info(f"Config {rec.id} completed all rungs")

        # Check if we should stop
        # Stop when: (max_trials reached OR no max_trials set) AND enough configs completed at max rung
        max_trials_reached = self.max_trials is not None and self.total_configs_started >= self.max_trials
        enough_final_results = len(self.completed_configs) >= self.min_top_configs

        if enough_final_results and (max_trials_reached or self.max_trials is None):
            # If max_trials set: stop when we've tried all configs and have enough results
            # If max_trials None: stop when we have enough good results at max rung
            self.complete = True
            logger.info(f"ASHA: Stopping - {len(self.completed_configs)} configs reached max rung "
                        f"(min_top_configs={self.min_top_configs}), "
                        f"max_trials={'reached' if max_trials_reached else 'unlimited'}")
            return []

        # ASYNCHRONOUS: Generate recommendations immediately if we have free slots
        # No synchronization barrier - launch as soon as slots are available

        slots_available = self.max_concurrent - len(self.active_configs)
        logger.info(f"ASHA: Active={len(self.active_configs)}/{self.max_concurrent}, "
                    f"Slots available={slots_available}, Pending promotions={len(self.pending_promotions)}")

        # Generate new recommendations to fill available slots
        # ASHA keeps workers busy by launching promotions or new configs
        new_recommendations = []
        while len(self.active_configs) + len(new_recommendations) < self.max_concurrent:
            # Priority 1: Process pending promotions
            if self.pending_promotions:
                config_id, epochs = self.pending_promotions.pop(0)
                specs = self.config_specs[config_id]
                # Find the job_id for this config from history
                config_job_id = None
                for rec in reversed(history):
                    if rec.id == config_id:
                        config_job_id = rec.job_id
                        break
                self.active_configs.add(config_id)
                # Set epoch_number for controller to use as early_stop_epoch (interruption point)
                # Training is configured for max epochs, but will be interrupted at this rung
                self.epoch_number = epochs
                resume_rec = ResumeRecommendation(config_id, specs, config_job_id)
                self.track_id = config_id
                new_recommendations.append(resume_rec)
                logger.info(f"Launching promotion: Config {config_id} @ {epochs} epochs "
                            f"(active: {len(self.active_configs) + len(new_recommendations)})")

            # Priority 2: Start new configs if no pending promotions and budget allows
            elif self.max_trials is None or self.total_configs_started < self.max_trials:
                specs = self._generate_random_parameters()
                self.config_specs[self.next_config_id] = specs
                self.config_to_rung[self.next_config_id] = 0
                self.active_configs.add(self.next_config_id)
                self.total_configs_started += 1
                # Set epoch_number for controller to use as early_stop_epoch (interruption point)
                # Training is configured for max epochs, but will be interrupted at first rung
                self.epoch_number = self.rungs[0]
                self.track_id = self.next_config_id
                logger.info(
                    f"Launching new config: {self.next_config_id} @ {self.rungs[0]} epochs "
                    f"(total started: {self.total_configs_started}, "
                    f"active: {len(self.active_configs) + len(new_recommendations)})"
                )
                self.next_config_id += 1
                new_recommendations.append(specs)
            else:
                # No promotions pending and max_trials reached - no more work to do
                logger.debug(f"ASHA: max_trials={self.max_trials} reached, no new configs to start")
                break

        if new_recommendations:
            logger.info(f"ASHA: Returning {len(new_recommendations)} recommendation(s)")
        else:
            logger.info("ASHA: No slots available or no pending work - returning empty")

        return new_recommendations
