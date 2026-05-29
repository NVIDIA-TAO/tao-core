# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BFBO (Bayesian First-Order Bayesian Optimization) AutoML algorithm modules"""
import numpy as np
import os
import math
import logging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from scipy.optimize import minimize
from scipy.stats import norm

from nvidia_tao_core.microservices.automl import network_utils
from nvidia_tao_core.microservices.utils.automl_utils import (
    JobStates, get_valid_range, clamp_value,
    get_valid_options, get_option_weights, fix_input_dimension
)
from nvidia_tao_core.microservices.automl.automl_algorithm_base import AutoMLAlgorithmBase, is_nan_value
from nvidia_tao_core.microservices.utils.handler_utils import get_total_epochs, get_flatten_specs
from nvidia_tao_core.microservices.utils.stateless_handler_utils import save_automl_brain_info, get_automl_brain_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BFBO(AutoMLAlgorithmBase):
    """BFBO (Bayesian First-Order Bayesian Optimization) AutoML algorithm class

    BFBO enhances traditional Bayesian Optimization by:
    1. Using Upper Confidence Bound (UCB) acquisition function with adaptive exploration
    2. Implementing local penalization to avoid querying similar points
    3. Using gradient-based optimization for better acquisition function optimization
    """

    def __init__(self, job_context, root, network, parameters):
        """Initialize the BFBO algorithm class

        Args:
            root: handler root
            network: model we are running AutoML on
            parameters: automl sweepable parameters
        """
        super().__init__(job_context, root, network, parameters)

        # Use RBF kernel for smoother gradients
        length_scale = [1.0] * len(self.parameters)
        kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=95051
        )

        # The following 2 need to be stored
        self.Xs = []
        self.ys = []

        # UCB parameters
        self.kappa = 2.0  # Exploration-exploitation trade-off (higher = more exploration)
        self.kappa_decay = 0.95  # Decay factor for kappa over iterations
        self.kappa_min = 0.5  # Minimum kappa value

        # Local penalization parameters
        self.local_penalization = True
        self.penalization_radius = 0.1  # Radius for local penalization

        # Optimization parameters
        self.num_restarts = 10  # Number of random restarts for acquisition optimization
        self.num_warmup = 5  # Number of warmup iterations before using gradients

        self.num_epochs_per_experiment = get_total_epochs(job_context, os.path.join(self.handler_root, "specs"))

        logger.info("BFBO initialized with UCB acquisition and local penalization")

    def generate_automl_param_rec_value(self, parameter_config, suggestion):
        """Convert 0 to 1 GP prediction into a possible value"""
        parameter_name = parameter_config.get("parameter")
        # Apply custom overrides if provided
        if self.custom_ranges and parameter_name in self.custom_ranges:
            for override_key, override_value in self.custom_ranges[parameter_name].items():
                if override_value is not None:
                    parameter_config[override_key] = override_value

        data_type = parameter_config.get("value_type")
        default_value = parameter_config.get("default_value", None)
        math_cond = parameter_config.get("math_cond", None)
        parent_param = parameter_config.get("parent_param", None)

        if data_type == "float":
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
                    quantized = suggestion * (v_max - v_min) + v_min
                    logger.info(
                        f"Generated float for {parameter_name} (no range): "
                        f"{quantized} from suggestion {suggestion}"
                    )
                    return quantized
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
                    quantized = suggestion * (v_max - v_min) + v_min
                    return quantized
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
                    data_type=data_type,
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
                f"[BFBO] Parameter {parameter_name}: v_min={v_min}, v_max={v_max}, "
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
                        quantized = float(self._apply_power_constraint_with_equal_priority(
                            v_min, v_max, factor, fallback))
                    else:
                        # Regular sampling for non-power constraints
                        normalized = suggestion * (v_max - v_min) + v_min
                        quantized = clamp_value(normalized, v_min, v_max)
            else:
                # No math condition, regular sampling
                normalized = suggestion * (v_max - v_min) + v_min
                quantized = clamp_value(normalized, v_min, v_max)

            if not (type(parent_param) is float and math.isnan(parent_param)):
                if (isinstance(parent_param, str) and parent_param != "nan" and parent_param == "TRUE") or (
                    isinstance(parent_param, bool) and parent_param
                ):
                    self.parent_params[parameter_name] = quantized

            # Check for disable_list option - if True, skip network-specific logic
            if disable_list:
                logger.info(
                    f"disable_list=True for {parameter_name}: "
                    f"returning pure float {quantized} (skipping network-specific logic)"
                )
                return quantized

            # Apply network-specific parameter logic
            return network_utils.apply_network_specific_param_logic(
                network=self.network,
                data_type=data_type,
                parameter_name=parameter_name,
                value=quantized,
                v_max=v_max,
                default_train_spec=self.default_train_spec,
                parent_params=self.parent_params
            )

        if data_type in ("int", "integer"):
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

            # Map GP suggestion to discrete integer
            continuous_value = suggestion * (v_max - v_min) + v_min
            quantized_int = int(round(continuous_value))

            # Apply math condition if specified
            # Skip relational constraints (like "> depends_on") as they're handled in base class
            if math_cond and type(math_cond) is str and "depends_on" not in math_cond:
                parts = math_cond.split(" ")
                if len(parts) >= 2:
                    operator = parts[0]
                    factor = int(float(parts[1]))
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
                data_type=data_type,
                parameter_name=parameter_name,
                value=quantized_int,
                default_train_spec=self.default_train_spec,
                parent_params=self.parent_params
            )

        if data_type in ("categorical", "ordered"):
            valid_options = get_valid_options(parameter_config, self.custom_ranges)
            if not valid_options or valid_options == "":
                return default_value

            # Map GP suggestion to discrete index
            idx = int(suggestion * len(valid_options))
            idx = min(idx, len(valid_options) - 1)

            # Handle weighted options
            weights = get_option_weights(parameter_config, self.custom_ranges)
            if weights and len(weights) == len(valid_options):
                sorted_pairs = sorted(zip(valid_options, weights), key=lambda x: x[1], reverse=True)
                cumulative = 0
                total_weight = sum(weights)
                for option, weight in sorted_pairs:
                    cumulative += weight / total_weight
                    if suggestion <= cumulative:
                        return option
                return sorted_pairs[0][0]

            return valid_options[idx]

        if data_type == "ordered_int":
            valid_options = get_valid_options(parameter_config, self.custom_ranges)
            if not valid_options or valid_options == "":
                return int(default_value) if default_value else 0

            idx = int(suggestion * len(valid_options))
            idx = min(idx, len(valid_options) - 1)

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
        """Save the BFBO algorithm related variables to brain metadata"""
        state_dict = {}
        state_dict["Xs"] = np.array(self.Xs).tolist()  # List of np arrays
        state_dict["ys"] = np.array(self.ys).tolist()  # List
        state_dict["kappa"] = self.kappa

        save_automl_brain_info(self.job_context.id, state_dict)

    @staticmethod
    def load_state(job_context, root, network, parameters):
        """Load the BFBO algorithm related variables from brain metadata"""
        json_loaded = get_automl_brain_info(job_context.id)
        if not json_loaded:
            return BFBO(job_context, root, network, parameters)

        Xs = []
        for x in json_loaded["Xs"]:
            Xs.append(np.array(x))
        ys = json_loaded["ys"]
        bfbo = BFBO(job_context, root, network, parameters)
        # Load state (Remember everything)
        bfbo.Xs = Xs
        bfbo.ys = ys
        bfbo.kappa = json_loaded.get("kappa", 2.0)

        len_y = len(ys)
        if Xs and ys:
            Xs_npy = np.array(Xs[:len_y])
            ys_npy = np.array(ys)

            # Validate data before fitting - check for inf/nan values
            if np.any(np.isinf(ys_npy)) or np.any(np.isnan(ys_npy)):
                logger.warning(
                    "Detected inf/nan values in loaded training data. "
                    "Replacing inf with large finite values and nan with 0."
                )
                ys_npy = np.nan_to_num(ys_npy, nan=0.0, posinf=1e7, neginf=-1e7)
                # Update the loaded ys with cleaned values
                bfbo.ys = ys_npy.tolist()

            bfbo.gp.fit(Xs_npy, ys_npy)

        return bfbo

    def generate_recommendations(self, history):
        """Generates parameter values and appends to recommendations"""
        get_flatten_specs(self.default_train_spec, self.default_train_spec_flattened)
        if history == []:
            # default recommendation => random points
            suggestions = np.random.rand(len(self.parameters))
            self.Xs.append(suggestions)
            recommendations = []
            for param_dict, suggestion in zip(self.parameters, suggestions):
                recommendation_value = self.generate_automl_param_rec_value(param_dict, suggestion)
                logger.info(f"Recommendation param: {param_dict['parameter']} value: {recommendation_value}")
                recommendations.append(recommendation_value)
            return [dict(zip([param["parameter"] for param in self.parameters], recommendations))]

        # This function will be called every 5 seconds or so.
        # If no change in history, dont give a recommendation
        # ie - wait for previous recommendation to finish
        if history[-1].status not in [JobStates.success, JobStates.failure]:
            return []

        # Update the GP based on results
        self.ys.append(history[-1].result)
        self.update_gp()

        # Decay kappa for adaptive exploration-exploitation
        self.kappa = max(self.kappa_min, self.kappa * self.kappa_decay)
        logger.info(f"Updated kappa to {self.kappa}")

        # Generate one recommendation
        # Generate "suggestions" which are in [0.0, 1.0] by optimizing UCB
        suggestions = self.optimize_ucb()  # length = len(self.parameters), np.array type
        self.Xs.append(suggestions)
        # Convert the suggestions to recommendations based on parameter type
        recommendations = []
        assert len(self.parameters) == len(suggestions), (
            f"Number of parameters ({len(self.parameters)}) does not match "
            f"number of suggestions ({len(suggestions)})"
        )
        for param_dict, suggestion in zip(self.parameters, suggestions):
            recommendation_value = self.generate_automl_param_rec_value(param_dict, suggestion)
            logger.info(f"Recommendation param: {param_dict['parameter']} value: {recommendation_value}")
            recommendations.append(recommendation_value)

        return [dict(zip([param["parameter"] for param in self.parameters], recommendations))]

    def update_gp(self):
        """Update gaussian regressor parameters"""
        Xs_npy = np.array(self.Xs)
        ys_npy = np.array(self.ys)

        # Validate data before fitting - check for inf/nan values
        if np.any(np.isinf(ys_npy)) or np.any(np.isnan(ys_npy)):
            logger.warning(
                f"Detected inf/nan values in training data. "
                f"ys_npy: {ys_npy}. "
                f"Replacing inf with large finite values and nan with 0."
            )
            # Replace inf with large finite value (1e7) and nan with 0
            ys_npy = np.nan_to_num(ys_npy, nan=0.0, posinf=1e7, neginf=-1e7)
            logger.info(f"Cleaned ys_npy: {ys_npy}")

        if len(Xs_npy) > 0 and len(ys_npy) > 0:
            self.gp.fit(Xs_npy, ys_npy)
        else:
            logger.warning("No valid training data available for Gaussian Process")

    def optimize_ucb(self):
        """Optimize Upper Confidence Bound acquisition function"""
        best_ucb = -np.inf
        best_x = None

        dim = len(self.Xs[0])
        bounds = [(0, 1)] * len(self.parameters)

        # Use more restarts for better optimization
        for i in range(self.num_restarts):
            # Use Latin Hypercube Sampling for better initial points
            if i == 0:
                # First restart: use best observed point
                if len(self.Xs) > 0:
                    best_idx = np.argmax(self.ys)
                    x0 = self.Xs[best_idx] + np.random.randn(dim) * 0.1
                    x0 = np.clip(x0, 0, 1)
                else:
                    x0 = np.random.rand(dim)
            else:
                x0 = np.random.rand(dim)

            res = minimize(
                self._upper_confidence_bound,
                x0=x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 100}
            )

            if -res.fun > best_ucb:
                best_ucb = -res.fun
                best_x = res.x

        if best_x is None:
            logger.warning("UCB optimization failed, using random point")
            best_x = np.random.rand(dim)

        logger.info(f"Optimized UCB value: {best_ucb}")
        return best_x.reshape(-1)

    def _upper_confidence_bound(self, X):
        """Calculate the Upper Confidence Bound at points X

        UCB(x) = μ(x) + κ * σ(x)

        Args:
            X: Points at which UCB shall be calculated (d,)

        Returns:
            float: Negative UCB value (for minimization)
        """
        X = X.reshape(1, -1)

        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)

        # Apply local penalization if enabled
        penalization = 1.0
        if self.local_penalization and len(self.Xs) > 0:
            # Calculate distance to all previous points
            distances = np.linalg.norm(np.array(self.Xs) - X, axis=1)
            # Penalize points close to previous evaluations
            penalization = np.prod(np.tanh(distances / self.penalization_radius))

        # UCB acquisition function
        ucb = mu + self.kappa * sigma * penalization

        return -1 * ucb[0, 0]

    def _probability_of_improvement(self, X, xi=0.01):
        """Calculate the Probability of Improvement at points X

        This can be used as an alternative acquisition function.

        Args:
            X: Points at which PI shall be calculated (m x d)
            xi: Exploitation-exploration trade-off parameter

        Returns:
            float: Negative PI value (for minimization)
        """
        X = X.reshape(1, -1)

        mu, sigma = self.gp.predict(X, return_std=True)
        mu_sample = self.gp.predict(np.array(self.Xs))

        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            pi = norm.cdf(Z)
            pi[sigma == 0.0] = 0.0

        return -1 * pi[0, 0]
