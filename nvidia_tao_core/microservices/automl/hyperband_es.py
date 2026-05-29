# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HyperBand with Early Stopping (Learning Curve Prediction) AutoML algorithm modules"""
import numpy as np
import logging
from scipy.optimize import curve_fit

from nvidia_tao_core.microservices.utils.automl_utils import JobStates
from nvidia_tao_core.microservices.automl.hyperband import HyperBand
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    get_automl_brain_info
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperBandES(HyperBand):
    """HyperBand with Early Stopping via Learning Curve Prediction

    Extends HyperBand with predictive early stopping using learning curve
    extrapolation. Stops configurations early if predicted final performance
    is unlikely to be competitive.
    """

    def __init__(self, job_context, root, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                 early_stop_threshold=0.8, min_early_stop_epochs=3):
        """Initialize the HyperBand ES algorithm class

        Args:
            root: handler root
            network: model we are running AutoML on
            parameters: automl sweepable parameters
            max_epochs: the maximum amount of resource that can be allocated to a single configuration
            reduction_factor: reduction factor for successive halving
            epoch_multiplier: multiplying factor for epochs
            early_stop_threshold: confidence threshold for early stopping (0-1)
            min_early_stop_epochs: minimum epochs before attempting prediction
        """
        super().__init__(job_context, root, network, parameters, max_epochs, reduction_factor, epoch_multiplier)

        self.min_epochs_for_prediction = int(min_early_stop_epochs)
        self.confidence_threshold = float(early_stop_threshold)

        # Track learning curves: config_id -> [(epoch, metric), ...]
        self.learning_curves = {}

        # Track early stopped configs
        self.early_stopped_configs = set()

        logger.info(
            f"HyperBandES initialized with early_stop_threshold={early_stop_threshold}, "
            f"min_early_stop_epochs={min_early_stop_epochs}"
        )

    @staticmethod
    def _power_law_model(x, a, b, c):
        """Power law learning curve model: y = a * x^b + c

        This model captures the typical behavior of neural network training
        where performance improves according to a power law.
        """
        return a * np.power(x, b) + c

    @staticmethod
    def _exponential_model(x, a, b, c):
        """Exponential learning curve model: y = a * exp(-b * x) + c

        Alternative model for learning curves that plateau exponentially.
        """
        return a * np.exp(-b * x) + c

    def _predict_final_performance(self, config_id, current_curve):
        """Predict final performance using learning curve extrapolation

        Args:
            config_id: configuration ID
            current_curve: list of (epoch, metric) tuples

        Returns:
            tuple: (predicted_final, confidence)
                - predicted_final: predicted metric at max epochs
                - confidence: confidence in prediction (0-1)
        """
        if len(current_curve) < self.min_epochs_for_prediction:
            return None, 0.0

        epochs = np.array([e for e, _ in current_curve])
        metrics = np.array([m for _, m in current_curve])

        # Try to fit power law model
        try:
            # Initial guess for parameters
            p0 = [metrics[0] - metrics[-1], -0.5, metrics[-1]]

            popt_power, _ = curve_fit(
                self._power_law_model,
                epochs,
                metrics,
                p0=p0,
                maxfev=1000
            )

            # Predict at max epochs
            max_epochs = self.ri[self.bracket][-1] * self.epoch_multiplier
            predicted_power = self._power_law_model(max_epochs, *popt_power)

            # Calculate confidence from fit quality
            residuals = metrics - self._power_law_model(epochs, *popt_power)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((metrics - np.mean(metrics)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            confidence = max(0, min(1, r_squared))

            logger.info(
                f"Config {config_id}: Predicted final performance = {predicted_power:.4f} "
                f"(confidence={confidence:.2f})"
            )

            return predicted_power, confidence

        except Exception as e:
            logger.warning(f"Failed to fit learning curve for config {config_id}: {e}")
            return None, 0.0

    def _should_early_stop(self, config_id, current_result, current_epoch):
        """Determine if a configuration should be stopped early

        Args:
            config_id: configuration ID
            current_result: current metric value
            current_epoch: current epoch number

        Returns:
            bool: True if should stop early
        """
        if config_id in self.early_stopped_configs:
            return False  # Already stopped

        # Get or initialize learning curve
        if config_id not in self.learning_curves:
            self.learning_curves[config_id] = []

        self.learning_curves[config_id].append((current_epoch, current_result))

        # Need enough data points to predict
        if len(self.learning_curves[config_id]) < self.min_epochs_for_prediction:
            return False

        # Predict final performance
        predicted_final, confidence = self._predict_final_performance(
            config_id,
            self.learning_curves[config_id]
        )

        if predicted_final is None or confidence < self.confidence_threshold:
            # Not confident enough to make decision
            return False

        # Get best performance seen so far across all configs
        all_results = []
        for rec_id, curve in self.learning_curves.items():
            if rec_id != config_id and curve:
                # Get latest result from each config
                all_results.append(curve[-1][1])

        if not all_results:
            return False  # No comparison baseline yet

        # Compare predicted final with current best
        if self.reverse_sort:
            # Higher is better
            current_best = max(all_results)
            # Stop if predicted to be worse than current best by margin
            margin = 0.05  # 5% margin
            should_stop = predicted_final < current_best * (1 - margin)
        else:
            # Lower is better
            current_best = min(all_results)
            margin = 0.05
            should_stop = predicted_final > current_best * (1 + margin)

        if should_stop:
            self.early_stopped_configs.add(config_id)
            logger.info(
                f"Early stopping config {config_id}: predicted={predicted_final:.4f}, "
                f"current_best={current_best:.4f}"
            )

        return should_stop

    def generate_recommendations(self, history):
        """Generates recommendations with predictive early stopping"""
        # Use parent class logic
        recommendations = super().generate_recommendations(history)

        # Check for early stopping on active configurations
        for rec in history:
            if rec.status == JobStates.running and rec.result != 0.0:
                # Check if this config should be stopped early
                current_epoch = self.epoch_number  # Approximate
                if self._should_early_stop(rec.id, rec.result, current_epoch):
                    logger.info(f"Triggering early stop for config {rec.id}")
                    # Mark as failed to trigger elimination in parent class logic
                    rec.update_status(JobStates.failure)

        return recommendations

    @staticmethod
    def load_state(job_context, root, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                   metric="loss", min_epochs_for_prediction=3, confidence_threshold=0.8):
        """Load the HyperBandES algorithm related variables from brain metadata"""
        json_loaded = get_automl_brain_info(job_context.id)
        if not json_loaded:
            return HyperBandES(
                job_context, root, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                min_epochs_for_prediction, confidence_threshold
            )

        brain = HyperBandES(
            job_context, root, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
            min_epochs_for_prediction, confidence_threshold
        )
        # Load base HyperBand state
        brain.bracket = json_loaded["bracket"]
        brain.sh_iter = json_loaded["sh_iter"]
        brain.expt_iter = json_loaded["expt_iter"]
        brain.complete = json_loaded["complete"]
        brain.epoch_number = json_loaded["epoch_number"]

        # Load ES-specific state if available
        if "learning_curves" in json_loaded:
            brain.learning_curves = json_loaded["learning_curves"]
        if "early_stopped_configs" in json_loaded:
            brain.early_stopped_configs = set(json_loaded["early_stopped_configs"])

        return brain

    def save_state(self):
        """Save the HyperBandES algorithm related variables to brain metadata"""
        # Call parent save_state
        super().save_state()

        # Add ES-specific state
        from nvidia_tao_core.microservices.utils.stateless_handler_utils import save_automl_brain_info

        state_dict = get_automl_brain_info(self.job_context.id)
        state_dict["learning_curves"] = self.learning_curves
        state_dict["early_stopped_configs"] = list(self.early_stopped_configs)

        save_automl_brain_info(self.job_context.id, state_dict)
