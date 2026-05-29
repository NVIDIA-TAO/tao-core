# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BOHB (Bayesian Optimization and HyperBand) AutoML algorithm modules"""
import numpy as np
import math
import logging
from scipy.stats import gaussian_kde

from nvidia_tao_core.microservices.utils.automl_utils import (
    ResumeRecommendation, JobStates, get_valid_range, clamp_value,
    get_valid_options, get_option_weights, fix_input_dimension
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


class BOHB(AutoMLAlgorithmBase):
    """BOHB (Bayesian Optimization and HyperBand) AutoML algorithm class

    BOHB combines the resource allocation strategy of HyperBand with
    a Tree-structured Parzen Estimator (TPE) model for smart configuration sampling.
    """

    def __init__(self, job_context, root, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                 kde_samples=64, top_n_percent=15.0, min_points_in_model=10, metric="loss"):
        """Initialize the BOHB algorithm class

        Args:
            root: handler root
            network: model we are running AutoML on
            parameters: automl sweepable parameters
            max_epochs: the maximum amount of resource that can be allocated to a single configuration
            reduction_factor: an input that controls the proportion of configurations
                discarded in each round of SuccessiveHalving
            epoch_multiplier: multiplying factor for epochs
            kde_samples: number of samples to evaluate for each recommendation
            top_n_percent: percentage of top configurations to use for good KDE
            min_points_in_model: minimum number of observations needed to build KDE model
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
        self.bracket = "0"
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

        # TPE-specific variables for Bayesian optimization
        self.observations = []  # List of (config, result) tuples
        self.quantile = top_n_percent / 100.0  # Convert percentage to quantile
        self.min_bandwidth = 0.01  # Minimum bandwidth for KDE
        self.num_samples = int(kde_samples)  # Number of samples to evaluate for each recommendation
        self.min_points_in_model = int(min_points_in_model)  # Minimum points needed for KDE

        logger.info(
            f"BOHB initialized with max_epochs={max_epochs}, "
            f"reduction_factor={reduction_factor}, epoch_multiplier={self.epoch_multiplier}, "
            f"kde_samples={self.num_samples}, top_n_percent={top_n_percent}, "
            f"min_points_in_model={self.min_points_in_model}"
        )

    def brackets_and_sh_sequence(self, max_epochs, reduction_factor):
        """Generate ni,ri arrays based on max_epochs and reduction_factor values"""
        smax = int(np.log(max_epochs) / np.log(reduction_factor))
        for itr, s in enumerate(range(smax, 0, -1)):
            self.ni[str(itr)] = []
            self.ri[str(itr)] = []
            n = int(math.ceil(int((smax + 1) / (s + 1)) * (reduction_factor**s)))
            r = int(max_epochs / (reduction_factor**s))
            for s_idx in range(s + 1):
                ni = int(n * (reduction_factor**(-s_idx)))
                ri = int(r * (reduction_factor**s_idx))
                self.ni[str(itr)].append(ni)
                self.ri[str(itr)].append(ri)

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

    def _build_kde(self, data, bandwidth=None):
        """Build Kernel Density Estimator for TPE model

        Args:
            data: numpy array of shape (n_samples, n_dims)
            bandwidth: bandwidth for KDE (optional, uses Scott's rule if None)

        Returns:
            gaussian_kde object or None if insufficient data
        """
        if len(data) < max(2, self.min_points_in_model):
            return None

        try:
            if bandwidth is not None:
                kde = gaussian_kde(data.T, bw_method=bandwidth)
            else:
                kde = gaussian_kde(data.T)
            return kde
        except Exception as e:
            logger.warning(f"Failed to build KDE: {e}")
            return None

    def _sample_from_kde(self, kde, n_samples):
        """Sample configurations from KDE

        Args:
            kde: gaussian_kde object
            n_samples: number of samples to generate

        Returns:
            numpy array of shape (n_samples, n_dims)
        """
        if kde is None:
            return None

        try:
            samples = kde.resample(n_samples).T
            # Clip to [0, 1] range
            samples = np.clip(samples, 0.0, 1.0)
            return samples
        except Exception as e:
            logger.warning(f"Failed to sample from KDE: {e}")
            return None

    def _tpe_suggest(self):
        """Use Tree-structured Parzen Estimator to suggest next configuration

        Returns:
            numpy array of shape (n_dims,) representing suggested configuration in [0, 1]
        """
        if len(self.observations) < max(2, self.min_points_in_model):
            # Not enough data, return random sample
            min_required = max(2, self.min_points_in_model)
            logger.info(
                f"Insufficient observations for TPE ({len(self.observations)} < {min_required}), "
                "using random sampling"
            )
            return np.random.rand(len(self.parameters))

        # Sort observations by result
        sorted_obs = sorted(self.observations, key=lambda x: x[1], reverse=self.reverse_sort)

        # Split into good and bad observations
        n_good = max(1, int(self.quantile * len(sorted_obs)))
        good_obs = np.array([obs[0] for obs in sorted_obs[:n_good]])
        bad_obs = np.array([obs[0] for obs in sorted_obs[n_good:]])

        # Build KDE models
        good_kde = self._build_kde(good_obs)
        bad_kde = self._build_kde(bad_obs)

        if good_kde is None:
            logger.info("Failed to build good KDE, using random sampling")
            return np.random.rand(len(self.parameters))

        # Sample candidates from good KDE
        candidates = self._sample_from_kde(good_kde, self.num_samples)
        if candidates is None:
            logger.info("Failed to sample from good KDE, using random sampling")
            return np.random.rand(len(self.parameters))

        # Evaluate Expected Improvement for each candidate
        best_ei = -np.inf
        best_candidate = None

        for candidate in candidates:
            # Calculate likelihood ratio l(x) / g(x)
            good_prob = good_kde.pdf(candidate.reshape(-1, 1))[0]

            if bad_kde is not None:
                bad_prob = bad_kde.pdf(candidate.reshape(-1, 1))[0]
                # Avoid division by zero
                bad_prob = max(bad_prob, 1e-10)
                ei = good_prob / bad_prob
            else:
                ei = good_prob

            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate

        if best_candidate is None:
            logger.warning("No valid candidate found, using random sampling")
            return np.random.rand(len(self.parameters))

        logger.info(f"TPE suggested configuration with EI={best_ei}")
        return best_candidate

    def generate_automl_param_rec_value(self, parameter_config, suggestion=None):
        """Generate parameter value from TPE suggestion or randomly

        Args:
            parameter_config: parameter configuration dict
            suggestion: suggested value in [0, 1] (optional, generates randomly if None)

        Returns:
            Generated parameter value
        """
        if suggestion is None:
            # No suggestion provided, generate randomly
            return super().generate_automl_param_rec_value(parameter_config)

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

            # If no valid range, generate values around default using suggestion
            if v_min == "" or v_max == "":
                if default_value is not None and default_value != "":
                    default_val = float(default_value)
                    # Use suggestion to vary around default
                    if default_val > 0:
                        v_min = default_val / 10.0
                        v_max = default_val * 10.0
                    elif default_val < 0:
                        v_min = default_val * 10.0
                        v_max = default_val / 10.0
                    else:  # default is 0
                        v_min = -1.0
                        v_max = 1.0
                    # Use suggestion to pick value in range
                    random_float = suggestion * (v_max - v_min) + v_min
                    logger.info(
                        f"Generated float for {parameter_name} (no range): "
                        f"{random_float} from suggestion {suggestion}"
                    )
                    return random_float
                # No default, use suggestion in [0, 1]
                return float(suggestion)

            if is_nan_value(v_min) or is_nan_value(v_max):
                # NaN ranges, use default-based range
                if default_value is not None:
                    default_val = float(default_value)
                    if default_val > 0:
                        v_min = default_val / 10.0
                        v_max = default_val * 10.0
                    else:
                        v_min = 0.0
                        v_max = 1.0
                    random_float = suggestion * (v_max - v_min) + v_min
                    return random_float
                return float(suggestion)

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
                    base_value = float(10 ** (suggestion * (log_max - log_min) + log_min))
                else:
                    base_value = float(suggestion * (base_max - base_min) + base_min)

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
                f"[BOHB] Parameter {parameter_name}: v_min={v_min}, v_max={v_max}, "
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
                        normalized = suggestion * (v_max - v_min) + v_min
                        fallback = clamp_value(normalized, v_min, v_max)
                        random_float = float(self._apply_power_constraint_with_equal_priority(
                            v_min, v_max, factor, fallback))
                    else:
                        # Regular sampling for non-power constraints
                        normalized = suggestion * (v_max - v_min) + v_min
                        random_float = clamp_value(normalized, v_min, v_max)
            else:
                # No math condition, regular sampling
                normalized = suggestion * (v_max - v_min) + v_min
                random_float = clamp_value(normalized, v_min, v_max)

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

        if tp in ("int", "integer"):
            v_min = parameter_config.get("valid_min", "")
            v_max = parameter_config.get("valid_max", "")

            # If no valid range, generate values around default using suggestion
            if v_min == "" or v_max == "":
                if default_value is not None and default_value != "":
                    default_val = int(default_value)
                    # Use suggestion to vary around default
                    if default_val > 0:
                        v_min = max(1, default_val // 2)
                        v_max = default_val * 2
                    else:
                        v_min = 1
                        v_max = 100
                    # Use suggestion to pick value in range
                    continuous_value = suggestion * (v_max - v_min) + v_min
                    quantized_int = int(round(continuous_value))
                    logger.info(
                        f"Generated int for {parameter_name} (no range): "
                        f"{quantized_int} from suggestion {suggestion}"
                    )
                    return quantized_int
                # No default, use suggestion in [1, 100]
                return int(round(suggestion * 99 + 1))

            if is_nan_value(v_min) or is_nan_value(v_max):
                # NaN ranges, use default-based range
                if default_value is not None:
                    default_val = int(default_value)
                    v_min = max(1, default_val // 2)
                    v_max = default_val * 2
                    continuous_value = suggestion * (v_max - v_min) + v_min
                    return int(round(continuous_value))
                return int(round(suggestion * 99 + 1))

            v_min, v_max = get_valid_range(parameter_config, self.parent_params, self.custom_ranges)

            # Map continuous suggestion to discrete integer
            continuous_value = suggestion * (v_max - v_min) + v_min
            quantized_int = int(round(continuous_value))

            # Apply math condition if specified
            # Skip relational constraints (like "> depends_on") as they're handled in base class
            if math_cond and type(math_cond) is str and "depends_on" not in math_cond:
                parts = math_cond.split(" ")
                if len(parts) >= 2:
                    operator = parts[0]
                    factor = int(parts[1])
                    if operator == "^":
                        quantized_int = int(self._apply_power_constraint_with_equal_priority(
                            v_min, v_max, factor, quantized_int))
                    elif operator == "/":
                        quantized_int = fix_input_dimension(quantized_int, factor)

            if not (type(parent_param) is float and math.isnan(parent_param)):
                if ((type(parent_param) is str and parent_param != "nan" and parent_param == "TRUE") or
                        (type(parent_param) is bool and parent_param)):
                    self.parent_params[parameter_name] = quantized_int

            return network_utils.apply_network_specific_param_logic(
                network=self.network,
                data_type=tp,
                parameter_name=parameter_name,
                value=quantized_int,
                default_train_spec=self.default_train_spec,
                parent_params=self.parent_params
            )

        if tp in ("categorical", "ordered"):
            valid_options = get_valid_options(parameter_config, self.custom_ranges)
            if not valid_options or valid_options == "":
                return default_value

            # Map continuous suggestion [0,1] to discrete index
            idx = int(suggestion * len(valid_options))
            idx = min(idx, len(valid_options) - 1)  # Clamp to valid range

            # Get weights for weighted sampling if available
            weights = get_option_weights(parameter_config, self.custom_ranges)
            if weights and len(weights) == len(valid_options):
                # Use suggestion to pick from weighted distribution
                # Sort options by weight and use suggestion to pick
                sorted_pairs = sorted(zip(valid_options, weights), key=lambda x: x[1], reverse=True)
                cumulative = 0
                total_weight = sum(weights)
                for option, weight in sorted_pairs:
                    cumulative += weight / total_weight
                    if suggestion <= cumulative:
                        return option
                return sorted_pairs[0][0]  # Return highest weight if we get here

            return valid_options[idx]

        if tp == "ordered_int":
            valid_options = get_valid_options(parameter_config, self.custom_ranges)
            if not valid_options or valid_options == "":
                return int(default_value) if default_value else 0

            # Map continuous suggestion to discrete index
            idx = int(suggestion * len(valid_options))
            idx = min(idx, len(valid_options) - 1)

            # Get weights for weighted sampling if available
            weights = get_option_weights(parameter_config, self.custom_ranges)
            if weights and len(weights) == len(valid_options):
                sorted_pairs = sorted(zip(valid_options, weights), key=lambda x: x[1], reverse=True)
                cumulative = 0
                total_weight = sum(weights)
                for option, weight in sorted_pairs:
                    cumulative += weight / total_weight
                    if suggestion <= cumulative:
                        return int(option)
                return int(sorted_pairs[0][0])

            return int(valid_options[idx])

        return super().generate_automl_param_rec_value(parameter_config)

    def save_state(self):
        """Save the BOHB algorithm related variables to brain metadata"""
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
        # Save TPE observations
        state_dict["observations"] = [
            (obs[0].tolist(), obs[1]) for obs in self.observations
        ]

        save_automl_brain_info(self.job_context.id, state_dict)

    @staticmethod
    def load_state(
        job_context, root, network, parameters, max_epochs,
        reduction_factor, epoch_multiplier, metric="loss"
    ):
        """Load the BOHB algorithm related variables from brain metadata"""
        json_loaded = get_automl_brain_info(job_context.id)
        if not json_loaded:
            return BOHB(
                job_context, root, network, parameters, max_epochs,
                reduction_factor, epoch_multiplier, metric=metric
            )

        # Load metric from state (with fallback to parameter)
        loaded_metric = json_loaded.get("metric", metric)
        brain = BOHB(
            job_context, root, network, parameters, max_epochs,
            reduction_factor, epoch_multiplier, metric=loaded_metric
        )
        # Load state (Remember everything)
        brain.bracket = json_loaded["bracket"]
        brain.sh_iter = json_loaded["sh_iter"]
        brain.expt_iter = json_loaded["expt_iter"]
        brain.complete = json_loaded["complete"]
        brain.epoch_number = json_loaded["epoch_number"]
        brain.last_launched_count = json_loaded.get("last_launched_count", 0)

        # Load TPE observations if available
        if "observations" in json_loaded:
            brain.observations = [
                (np.array(obs[0]), obs[1]) for obs in json_loaded["observations"]
            ]

        return brain

    def _generate_one_recommendation(self, history):
        """Updates the counter variables and performs successive halving with TPE"""
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
            logger.info(f"BOHB: All brackets complete (bracket={self.bracket} > max), setting complete=True")
            self.complete = True
            return None

        if self.sh_iter == 0:
            # Use TPE to generate configuration instead of random sampling
            suggestions = self._tpe_suggest()
            specs = self._generate_parameters_from_suggestions(suggestions)
            self.epoch_number = self.ri[self.bracket][self.sh_iter] * self.epoch_multiplier
            to_return = specs
        else:
            # Do successive halving on the last bracket
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
            resumerec = ResumeRecommendation(
                self.experiments_considered[self.expt_iter].id,
                self.experiments_considered[self.expt_iter].specs,
                self.experiments_considered[self.expt_iter].job_id
            )
            to_return = resumerec
        self.expt_iter += 1

        return to_return

    def done(self):
        """Return if BOHB algorithm is complete or not.

        Returns True only if all recommendations have been issued AND all have completed.
        Checks last_launched_count to ensure running experiments finish before declaring done.
        """
        logger.info(f"BOHB done() called: complete={self.complete}, last_launched_count={self.last_launched_count}")

        if not self.complete:
            logger.info("BOHB done() returning False: not complete yet")
            return False

        # If complete flag is set but we still have running experiments, not done yet
        if self.last_launched_count > 0:
            logger.warning(f"BOHB done() returning False: {self.last_launched_count} experiments still running!")
            return False

        logger.info("BOHB done() returning True: all recommendations issued and completed")
        return True

    @property
    def max_concurrent(self):
        """Maximum number of concurrent experiments for BOHB.

        Returns the maximum ni value across all brackets to allow full parallelism.
        """
        max_ni = 1
        for bracket_ni in self.ni.values():
            if bracket_ni:
                max_ni = max([max_ni] + bracket_ni)
        return max_ni

    def _generate_parameters_from_suggestions(self, suggestions):
        """Generates parameter values from TPE suggestions

        Args:
            suggestions: numpy array of suggested values in [0, 1]

        Returns:
            dict of parameter names to generated values
        """
        hyperparam_dict = {}
        for param, suggestion in zip(self.parameters, suggestions):
            name = param["parameter"]
            rec = self.generate_automl_param_rec_value(param, suggestion)
            logger.info(f"Generated parameter in BOHB: {name} = {rec}")
            hyperparam_dict[name] = rec
        return hyperparam_dict

    def generate_recommendations(self, history):
        """Generates recommendations for the controller to run (supports parallel execution)"""
        get_flatten_specs(self.default_train_spec, self.default_train_spec_flattened)

        logger.info(
            f"BOHB generate_recommendations: complete={self.complete}, "
            f"last_launched_count={self.last_launched_count}, history_len={len(history)}"
        )

        if self.complete:
            # Check if final experiments have completed
            # NOTE: Resumed experiments reuse the same ID, so we can't rely on history indices
            # Instead, check for ANY pending/running experiments
            if self.last_launched_count > 0 and history:
                logger.info(
                    f"BOHB checking final batch completion: last_launched_count={self.last_launched_count}, "
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

                logger.info(f"BOHB final batch check: any_running={any_running}")

                if not any_running:
                    logger.info(
                        f"BOHB: All experiments complete, resetting last_launched_count "
                        f"from {self.last_launched_count} to 0"
                    )
                    self.last_launched_count = 0  # Reset counter after final batch completes
                else:
                    logger.info(
                        f"BOHB: Experiments still running, keeping last_launched_count={self.last_launched_count}"
                    )
            return []

        # Update observations with completed experiments
        for rec in history:
            if rec.status == JobStates.success and rec.result != 0.0:
                # Extract configuration as numpy array
                config = []
                for param in self.parameters:
                    param_name = param["parameter"]
                    value = rec.specs.get(param_name)
                    # Normalize to [0, 1] range if possible
                    if param["value_type"] == "float":
                        v_min, v_max = get_valid_range(param, self.parent_params, self.custom_ranges)
                        if v_max > v_min:
                            normalized = (value - v_min) / (v_max - v_min)
                            config.append(np.clip(normalized, 0.0, 1.0))
                        else:
                            config.append(0.5)
                    else:
                        # For non-float types, use a simple normalization
                        config.append(0.5)

                if len(config) == len(self.parameters):
                    # Only add if we have all parameters
                    config_array = np.array(config)
                    # Check if this configuration is already in observations
                    is_duplicate = any(
                        np.allclose(obs[0], config_array) for obs in self.observations
                    )
                    if not is_duplicate:
                        self.observations.append((config_array, rec.result))
                        logger.info(f"Added observation: config with result={rec.result}")

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
                f"BOHB: Launching {len(recommendations)} parallel configs for "
                f"bracket {self.bracket}, rung {self.sh_iter}"
            )
            return recommendations

        # Check if all experiments from last launch are complete
        # NOTE: Resumed experiments reuse IDs, so check for ANY running experiments
        if self.last_launched_count > 0:
            logger.info("BOHB: Checking if last batch complete before generating new recommendations")

            # Check for ANY pending/started/running experiments (handles resumed experiments with reused IDs)
            any_running = any(
                exp.status in [JobStates.pending, JobStates.started, JobStates.running]
                for exp in history
            )

            logger.info(f"BOHB: any_running={any_running} (last_launched_count={self.last_launched_count})")

            if any_running:
                logger.info("BOHB: Experiments still running, waiting before generating new recommendations")
                return []  # Wait for current batch to complete

            logger.info("BOHB: All experiments complete, proceeding to generate new recommendations")

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
                f"BOHB: Launching {len(recommendations)} recommendation(s) for "
                f"bracket {self.bracket}, rung {self.sh_iter}"
            )

        return recommendations
