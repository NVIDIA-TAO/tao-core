# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hyperband AutoML algorithm modules"""
import numpy as np
import math
import logging
import os

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
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


class HyperBand(AutoMLAlgorithmBase):
    """Hyperband AutoML algorithm class"""

    def __init__(
        self, job_context, root, network, parameters, max_epochs,
        reduction_factor, epoch_multiplier, metric="loss"
    ):
        """Initialize the Hyperband algorithm class

        Args:
            root: handler root
            network: model we are running AutoML on
            parameters: automl sweepable parameters (epoch params will be filtered out)
            max_epochs: the maximum amount of resource that can be allocated to a single configuration
            reduction_factor: an input that controls the proportion of configurations
                discarded in each round of SuccessiveHalving
            epoch_multiplier: multiplying factor for epochs
            metric: metric to optimize (e.g., 'loss', 'val_accuracy', 'mIoU')
        """
        super().__init__(job_context, root, network, parameters)
        self.epoch_multiplier = int(epoch_multiplier)
        self.metric = metric
        self.ni = {}
        self.ri = {}
        self.brackets_and_sh_sequence(max_epochs, reduction_factor)
        self.epoch_number = 0
        # State variables
        self.bracket = "0"  # Bracket
        self.override_num_epochs(self.ri[self.bracket][-1] * self.epoch_multiplier)
        self.sh_iter = 0  # SH iteration
        self.experiments_considered = []
        self.expt_iter = 0  # Recommendations within the SH
        self.complete = False

        # Determine reverse_sort based on metric (same logic as controller)
        # Default: higher is better (accuracy, mIoU, etc.)
        self.reverse_sort = True
        # For loss metrics: lower is better
        if metric == "loss" or "loss" in metric.lower() or metric.lower() in ("evaluation_cost",):
            self.reverse_sort = False
        # Track how many configs were launched in current rung (for parallel execution)
        self.last_launched_count = 0
        logger.info(
            f"Hyperband initialized with max_epochs={max_epochs}, "
            f"reduction_factor={reduction_factor}, epoch_multiplier={self.epoch_multiplier}"
        )

    def brackets_and_sh_sequence(self, max_epochs, reduction_factor):
        """Generate ni,ri arrays based on max_epochs and reduction_factor values"""
        smax = int(np.log(max_epochs) / np.log(reduction_factor))
        logger.info(
            f"Hyperband bracket calculation: max_epochs={max_epochs}, "
            f"reduction_factor={reduction_factor}, smax={smax}"
        )
        for itr, s in enumerate(range(smax, 0, -1)):
            self.ni[str(itr)] = []
            self.ri[str(itr)] = []
            n = int(math.ceil(int((smax + 1) / (s + 1)) * (reduction_factor**s)))
            r = int(max_epochs / (reduction_factor**s))
            logger.info(f"  Bracket {itr} (s={s}): initial n={n}, r={r}")
            for s_idx in range(s + 1):
                ni = int(n * (reduction_factor**(-s_idx)))
                ri = int(r * (reduction_factor**s_idx))
                self.ni[str(itr)].append(ni)
                self.ri[str(itr)].append(ri)
            logger.info(f"  Bracket {itr} final: ni={self.ni[str(itr)]}, ri={self.ri[str(itr)]}")
        logger.info(f"All brackets: ni={self.ni}, ri={self.ri}")

    def _is_epoch_parameter(self, param_name):
        """Check if a parameter controls epoch count

        Args:
            param_name: Flattened parameter name (e.g., "train_config.num_epochs")

        Returns:
            bool: True if parameter controls epoch count
        """
        # Simple epoch parameter names at root level
        if param_name in ("num_epochs", "epochs", "n_epochs", "max_iters", "epoch"):
            return True

        # Split the parameter name to check nested paths
        parts = param_name.split(".")

        # Check if any part is in epoch-related config sections
        has_training_config = any(
            p in ("training_config", "train_config", "train") for p in parts
        )
        has_epoch_key = any(
            p in ("num_epochs", "epochs", "n_epochs", "max_iters", "epoch", "max_epochs")
            for p in parts
        )

        return has_training_config and has_epoch_key

    def override_num_epochs(self, num_epochs):
        """Override num epochs parameter in train spec file

        This ensures bracket-based epoch control by updating all epoch-related
        parameters in the spec, regardless of automl_params settings.
        """
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
                f"[HYPERBAND] Parameter {parameter_name}: v_min={v_min}, v_max={v_max}, "
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
                        # Use helper function for power constraints with equal priority
                        fallback = np.random.uniform(low=v_min, high=v_max)
                        fallback = clamp_value(fallback, v_min, v_max)
                        random_float = float(self._apply_power_constraint_with_equal_priority(
                            v_min, v_max, factor, fallback))
                    else:
                        # Regular sampling for non-power constraints
                        random_float = np.random.uniform(low=v_min, high=v_max)
                        random_float = clamp_value(random_float, v_min, v_max)
            else:
                # No math condition, regular sampling
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
        """Save the Hyperband algorithm related variables to brain metadata"""
        state_dict = {}
        state_dict["bracket"] = self.bracket
        state_dict["sh_iter"] = self.sh_iter
        state_dict["expt_iter"] = self.expt_iter
        state_dict["complete"] = self.complete
        state_dict["epoch_number"] = self.epoch_number
        state_dict["epoch_multiplier"] = self.epoch_multiplier
        state_dict["ni"] = self.ni
        state_dict["ri"] = self.ri
        state_dict["last_launched_count"] = self.last_launched_count
        state_dict["metric"] = self.metric

        save_automl_brain_info(self.job_context.id, state_dict)

    @staticmethod
    def load_state(
        job_context, root, network, parameters, max_epochs,
        reduction_factor, epoch_multiplier, metric="loss"
    ):
        """Load the Hyperband algorithm related variables to brain metadata"""
        json_loaded = get_automl_brain_info(job_context.id)
        if not json_loaded:
            return HyperBand(
                job_context, root, network, parameters, max_epochs,
                reduction_factor, epoch_multiplier, metric
            )

        # Load metric from state (with fallback to parameter)
        loaded_metric = json_loaded.get("metric", metric)
        brain = HyperBand(
            job_context, root, network, parameters, max_epochs,
            reduction_factor, epoch_multiplier, loaded_metric
        )
        # Load state (Remember everything)
        brain.bracket = json_loaded["bracket"]  # Bracket
        brain.sh_iter = json_loaded["sh_iter"]  # SH iteration
        brain.expt_iter = json_loaded["expt_iter"]  # Recommendations within the SH
        brain.complete = json_loaded["complete"]
        brain.epoch_number = json_loaded["epoch_number"]
        brain.last_launched_count = json_loaded.get("last_launched_count", 0)

        return brain

    def _generate_one_recommendation(self, history):
        """Updates the counter variables and performs successive halving"""
        if self.complete:
            return None

        num = self.ni[self.bracket][self.sh_iter]
        if self.expt_iter == num:
            self.expt_iter = 0
            self.sh_iter += 1
        if self.sh_iter == len(self.ni[self.bracket]):
            self.sh_iter = 0
            self.bracket = str(int(self.bracket) + 1)
            if self.bracket in self.ri.keys():
                self.override_num_epochs(self.ri[self.bracket][-1] * self.epoch_multiplier)
        if int(self.bracket) > int(max(list(self.ni.keys()), key=int)):
            logger.info(f"Hyperband: All brackets complete (bracket={self.bracket} > max), setting complete=True")
            self.complete = True
            return None

        if self.sh_iter == 0:
            specs = self._generate_random_parameters()
            self.epoch_number = self.ri[self.bracket][self.sh_iter] * self.epoch_multiplier
            final_epoch = self.ri[self.bracket][-1] * self.epoch_multiplier
            self.override_num_epochs(final_epoch)
            logger.info(
                f"Hyperband: New experiment in bracket {self.bracket}, SH iter {self.sh_iter}, "
                f"will_pause_at_epoch={self.epoch_number}, spec_total_epochs={final_epoch}"
            )
            to_return = specs
        else:
            # Do successive halving on the last bracket
            # Here, we are sloppy in defining the window, but we assume runs that run for more epochs will be better
            # We take history[-bracket_size:] and prune this at every SH step
            lower = -1 * self.ni.get(self.bracket, [0])[0]

            if self.expt_iter == 0:
                if self.sh_iter == 1:
                    self.experiments_considered = sorted(
                        history[lower:],
                        key=lambda rec: rec.result,
                        reverse=self.reverse_sort
                    )[0:self.ni[self.bracket][self.sh_iter]]
                else:
                    for experiment in self.experiments_considered:
                        experiment.result = history[experiment.id].result
                    self.experiments_considered = sorted(
                        self.experiments_considered,
                        key=lambda rec: rec.result,
                        reverse=self.reverse_sort
                    )[0:self.ni[self.bracket][self.sh_iter]]

            self.epoch_number = self.ri[self.bracket][self.sh_iter] * self.epoch_multiplier
            final_epoch = self.ri[self.bracket][-1] * self.epoch_multiplier
            # Calculate resume_from_epoch (previous rung's epochs)
            resume_from_epoch = (
                self.ri[self.bracket][self.sh_iter - 1] * self.epoch_multiplier
                if self.sh_iter > 0 else 0
            )
            self.override_num_epochs(final_epoch)
            logger.info(
                f"Hyperband: Resume experiment in bracket {self.bracket}, SH iter {self.sh_iter}, "
                f"will_run_to_epoch={self.epoch_number}, spec_total_epochs={final_epoch}, "
                f"resume_from={resume_from_epoch}"
            )
            resumerec = ResumeRecommendation(
                self.experiments_considered[self.expt_iter].id,
                self.experiments_considered[self.expt_iter].specs,
                self.experiments_considered[self.expt_iter].job_id
            )
            to_return = resumerec
        self.expt_iter += 1

        return to_return

    def done(self):
        """Return if Hyperband algorithm is complete or not.

        Returns True only if all recommendations have been issued AND all have completed.
        Checks last_launched_count to ensure running experiments finish before declaring done.
        """
        logger.info(
            f"Hyperband done() called: complete={self.complete}, last_launched_count={self.last_launched_count}"
        )

        if not self.complete:
            logger.info("Hyperband done() returning False: not complete yet")
            return False

        # If complete flag is set but we still have running experiments, not done yet
        if self.last_launched_count > 0:
            logger.warning(f"Hyperband done() returning False: {self.last_launched_count} experiments still running!")
            return False

        logger.info("Hyperband done() returning True: all recommendations issued and completed")
        return True

    @property
    def max_concurrent(self):
        """Maximum number of concurrent experiments for Hyperband.

        Returns the maximum ni value across all brackets to allow full parallelism.
        """
        max_ni = 1
        for bracket_ni in self.ni.values():
            if bracket_ni:
                max_ni = max([max_ni] + bracket_ni)
        return max_ni

    def _generate_random_parameters(self):
        """Generates random parameter values for a recommendation

        Note: Epoch-controlling parameters are skipped because Hyperband controls
        epochs via bracket configuration. The actual epoch count is set by
        override_num_epochs() based on ri[bracket][sh_iter] * epoch_multiplier.
        """
        hyperparam_dict = {}
        for param in self.parameters:
            name = param["parameter"]
            # Skip epoch parameters in Hyperband - epochs are controlled by brackets
            if self._is_epoch_parameter(name):
                logger.info(f"Skipping epoch parameter in Hyperband (bracket-controlled): {name}")
                continue
            rec = self.generate_automl_param_rec_value(param)
            logger.info(f"Generated random parameter in hyperband: {name} = {rec}")
            hyperparam_dict[name] = rec
        return hyperparam_dict

    def generate_recommendations(self, history):
        """Generates recommendations for the controller to run (supports parallel execution)"""
        get_flatten_specs(self.default_train_spec, self.default_train_spec_flattened)

        logger.info(
            f"Hyperband generate_recommendations: complete={self.complete}, "
            f"last_launched_count={self.last_launched_count}, history_len={len(history)}"
        )

        if self.complete:
            # Check if final experiments have completed
            # NOTE: Resumed experiments reuse the same ID, so we can't rely on history indices
            # Instead, check for ANY pending/running experiments
            if self.last_launched_count > 0 and history:
                logger.info(
                    f"Hyperband checking final batch completion: last_launched_count={self.last_launched_count}, "
                    f"history_len={len(history)}"
                )

                # Log status of ALL experiments
                for i, exp in enumerate(history):
                    logger.info(
                        f"  history[{i}]: id={exp.id}, status={exp.status}, "
                        f"job_id={exp.job_id}, result={exp.result}"
                    )

                # Check for ANY pending/started/running experiments (resumed experiments reuse IDs!)
                any_running = any(
                    exp.status in [JobStates.pending, JobStates.started, JobStates.running]
                    for exp in history
                )

                logger.info(f"Hyperband final batch check: any_running={any_running}")

                if not any_running:
                    logger.info(
                        f"Hyperband: All experiments complete, resetting last_launched_count "
                        f"from {self.last_launched_count} to 0"
                    )
                    self.last_launched_count = 0  # Reset counter after final batch completes
                else:
                    logger.info(
                        f"Hyperband: Experiments still running, keeping last_launched_count={self.last_launched_count}"
                    )
            return []

        # Initial case: launch all configs for first rung in parallel
        if history == []:
            num_configs_in_rung = self.ni[self.bracket][self.sh_iter]
            recommendations = []
            for _ in range(num_configs_in_rung):
                rec = self._generate_one_recommendation(history)
                if type(rec) is dict:
                    recommendations.append(rec)
            self.last_launched_count = len(recommendations)
            self.track_id = len(recommendations) - 1 if recommendations else 0
            logger.info(
                f"Hyperband: Launching {len(recommendations)} parallel configs for "
                f"bracket {self.bracket}, rung {self.sh_iter}"
            )
            return recommendations

        # Check if all experiments from last launch are complete
        # NOTE: Resumed experiments reuse IDs, so check for ANY running experiments
        if self.last_launched_count > 0:
            logger.info("Hyperband: Checking if last batch complete before generating new recommendations")

            # Check for ANY pending/started/running experiments (handles resumed experiments with reused IDs)
            any_running = any(
                exp.status in [JobStates.pending, JobStates.started, JobStates.running]
                for exp in history
            )

            logger.info(f"Hyperband: any_running={any_running} (last_launched_count={self.last_launched_count})")

            if any_running:
                logger.info("Hyperband: Experiments still running, waiting before generating new recommendations")
                return []  # Wait for current batch to complete

            logger.info("Hyperband: All experiments complete, proceeding to generate new recommendations")

        # All previous experiments are done, generate next batch
        num_configs_before = self.ni[self.bracket][self.sh_iter] if not self.complete else 0

        recommendations = []
        for _ in range(max(num_configs_before, 1)):  # At least try to generate one
            rec = self._generate_one_recommendation(history)
            if rec is None:
                break
            recommendations.append(rec)

            # After first recommendation, check if state changed (new rung/bracket)
            if len(recommendations) == 1:
                num_configs_in_new_rung = self.ni[self.bracket][self.sh_iter] if not self.complete else 0
                # If we moved to a new state, generate rest of configs for new rung
                if num_configs_in_new_rung != num_configs_before:
                    for _ in range(num_configs_in_new_rung - 1):
                        rec = self._generate_one_recommendation(history)
                        if rec:
                            recommendations.append(rec)
                        else:
                            break
                    break

        self.last_launched_count = len(recommendations)

        if recommendations:
            last_rec = recommendations[-1]
            if type(last_rec) is dict:
                self.track_id = len(history) + len(recommendations) - 1
            elif type(last_rec) is ResumeRecommendation:
                self.track_id = last_rec.id

            logger.info(
                f"Hyperband: Launching {len(recommendations)} recommendation(s) for "
                f"bracket {self.bracket}, rung {self.sh_iter}"
            )

        return recommendations
