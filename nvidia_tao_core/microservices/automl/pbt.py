# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PBT (Population-Based Training) AutoML algorithm modules"""
import numpy as np
import logging
import copy

from nvidia_tao_core.microservices.utils.automl_utils import (
    ResumeRecommendation, JobStates, get_valid_range, clamp_value,
    get_valid_options, get_option_weights
)
from nvidia_tao_core.microservices.automl.automl_algorithm_base import AutoMLAlgorithmBase
from nvidia_tao_core.microservices.utils.handler_utils import get_flatten_specs
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    save_automl_brain_info,
    get_automl_brain_info,
    get_job_specs,
    save_job_specs
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PBT(AutoMLAlgorithmBase):
    """PBT (Population-Based Training) AutoML algorithm class

    PBT maintains a population of models training in parallel.
    Periodically, it evaluates all members and:
    1. Replaces poor performers by copying better ones
    2. Perturbs hyperparameters of copied configurations
    3. Continues training (exploit and explore)
    """

    def __init__(self, job_context, root, network, parameters,
                 population_size=10, max_generations=20, eval_interval=10, perturbation_factor=1.2, metric="loss"):
        """Initialize the PBT algorithm class

        Args:
            root: handler root
            network: model we are running AutoML on
            parameters: automl sweepable parameters
            population_size: number of models training in parallel
            max_generations: total number of generations to run
            eval_interval: epochs between evaluations/exploit-explore steps
            perturbation_factor: factor to perturb hyperparameters (1.2 = ±20%)
            metric: metric to optimize (e.g., 'loss', 'val_accuracy', 'mIoU')
        """
        super().__init__(job_context, root, network, parameters)

        self.population_size = int(population_size)
        self.max_generations = int(max_generations)
        self.eval_interval = int(eval_interval)
        self.perturbation_factor = float(perturbation_factor)
        self.metric = metric

        # Population state
        self.population = {}  # member_id -> {"specs": {}, "result": float, "epochs": int}
        self.generation = 0
        self.complete = False

        # Determine reverse_sort based on metric (same logic as controller)
        # Default: higher is better (accuracy, mIoU, etc.)
        self.reverse_sort = True
        # For loss metrics: lower is better
        if metric == "loss" or "loss" in metric.lower() or metric.lower() in ("evaluation_cost",):
            self.reverse_sort = False
        self.next_member_id = 0
        self.epoch_number = eval_interval  # Current epoch target (used for early_stop_epoch)

        # Track which parameters are perturbable
        self.perturbable_params = [
            p for p in parameters
            if p.get("value_type") in ("float", "int", "integer", "ordered_int")
        ]

        # Set training parameters once at initialization
        # num_epochs = total epochs the model will train (max_generations * eval_interval)
        # validation/checkpoint intervals = eval_interval (constant throughout)
        total_epochs = max_generations * eval_interval
        self.override_num_epochs(total_epochs, eval_interval)

        logger.info(
            f"PBT initialized with population_size={population_size}, "
            f"max_generations={max_generations}, eval_interval={eval_interval}, "
            f"perturbation_factor={perturbation_factor}"
        )
        logger.info(f"Perturbable parameters: {[p['parameter'] for p in self.perturbable_params]}")

    def override_num_epochs(self, total_epochs, interval):
        """Override training parameters in train spec file

        For PBT:
        - num_epochs = total epochs model will train (max_generations * eval_interval)
        - validation_interval = eval_interval (validate after each generation)
        - checkpoint_interval = eval_interval (checkpoint after each generation for resume)

        Args:
            total_epochs: Total number of epochs to train (max_generations * eval_interval)
            interval: Interval for validation and checkpointing (eval_interval)
        """
        spec = get_job_specs(self.job_context.id)
        for key1 in spec:
            if key1 in ("training_config", "train_config", "train"):
                for key2 in spec[key1]:
                    # Set main epoch parameter to total epochs
                    if key2 in ("num_epochs", "epochs", "n_epochs", "max_iters", "epoch"):
                        spec[key1][key2] = total_epochs
                        logger.info(f"PBT: Set {key1}.{key2} = {total_epochs}")
                    # Set validation interval to eval_interval
                    elif key2 in ("validation_interval", "val_interval"):
                        spec[key1][key2] = interval
                        logger.info(f"PBT: Set {key1}.{key2} = {interval}")
                    # Set checkpoint interval to eval_interval
                    elif key2 in ("checkpoint_interval", "ckpt_interval"):
                        spec[key1][key2] = interval
                        logger.info(f"PBT: Set {key1}.{key2} = {interval}")
                    elif key2 in ("train_config"):
                        for key3 in spec[key1][key2]:
                            if key3 in ("num_epochs", "epochs", "n_epochs", "max_iters", "epoch"):
                                spec[key1][key2][key3] = total_epochs
                                logger.info(f"PBT: Set {key1}.{key2}.{key3} = {total_epochs}")
                            elif key3 in ("validation_interval", "val_interval"):
                                spec[key1][key2][key3] = interval
                                logger.info(f"PBT: Set {key1}.{key2}.{key3} = {interval}")
                            elif key3 in ("checkpoint_interval", "ckpt_interval"):
                                spec[key1][key2][key3] = interval
                                logger.info(f"PBT: Set {key1}.{key2}.{key3} = {interval}")
        # Save the modified spec back to MongoDB
        save_job_specs(self.job_context.id, spec)

    def save_state(self):
        """Save the PBT algorithm related variables to brain metadata"""
        state_dict = {}
        # MongoDB requires string keys, so convert integer member IDs to strings
        state_dict["population"] = {str(k): v for k, v in self.population.items()}
        state_dict["generation"] = self.generation
        state_dict["complete"] = self.complete
        state_dict["next_member_id"] = self.next_member_id
        state_dict["population_size"] = self.population_size
        state_dict["max_generations"] = self.max_generations
        state_dict["eval_interval"] = self.eval_interval
        state_dict["epoch_number"] = self.epoch_number
        state_dict["metric"] = self.metric

        save_automl_brain_info(self.job_context.id, state_dict)

    @staticmethod
    def load_state(job_context, root, network, parameters,
                   population_size=10, max_generations=20, eval_interval=10, perturbation_factor=1.2, metric="loss"):
        """Load the PBT algorithm related variables from brain metadata"""
        json_loaded = get_automl_brain_info(job_context.id)
        if not json_loaded:
            return PBT(job_context, root, network, parameters,
                       population_size, max_generations, eval_interval, perturbation_factor, metric)

        # Load metric from state (with fallback to parameter)
        loaded_metric = json_loaded.get("metric", metric)
        brain = PBT(job_context, root, network, parameters,
                    population_size, max_generations, eval_interval, perturbation_factor, loaded_metric)
        # Load state
        # Convert string keys back to integers (MongoDB requires string keys)
        brain.population = {int(k): v for k, v in json_loaded["population"].items()}
        brain.generation = json_loaded["generation"]
        brain.complete = json_loaded["complete"]
        brain.next_member_id = json_loaded["next_member_id"]
        # Override with saved max_generations if available
        if "max_generations" in json_loaded:
            brain.max_generations = json_loaded["max_generations"]
        # Load epoch_number if available
        if "epoch_number" in json_loaded:
            brain.epoch_number = json_loaded["epoch_number"]

        return brain

    def _perturb_parameter(self, param_config, current_value):
        """Perturb a parameter value using resample or perturb strategy

        Args:
            param_config: parameter configuration dict
            current_value: current parameter value

        Returns:
            Perturbed parameter value
        """
        param_name = param_config.get("parameter")
        data_type = param_config.get("value_type")

        # 20% chance to resample completely, 80% chance to perturb
        if np.random.rand() < 0.2:
            # Resample
            logger.info(f"Resampling parameter {param_name}")
            return self.generate_automl_param_rec_value(param_config)

        # Perturb based on type
        if data_type == "float":
            v_min = param_config.get("valid_min", "")
            v_max = param_config.get("valid_max", "")
            if v_min == "" or v_max == "":
                return current_value

            v_min, v_max = get_valid_range(param_config, self.parent_params, self.custom_ranges)

            # Multiply or divide by perturbation_factor
            if np.random.rand() < 0.5:
                new_value = current_value * self.perturbation_factor
            else:
                new_value = current_value / self.perturbation_factor

            # Clamp to valid range
            new_value = clamp_value(new_value, v_min, v_max)

            logger.info(f"Perturbing {param_name}: {current_value:.4f} -> {new_value:.4f}")
            return new_value

        if data_type in ("int", "integer"):
            v_min = param_config.get("valid_min", "")
            v_max = param_config.get("valid_max", "")
            if v_min == "" or v_max == "":
                return current_value

            # Add or subtract a percentage
            delta = max(1, int(abs(current_value) * (self.perturbation_factor - 1.0)))
            if np.random.rand() < 0.5:
                new_value = current_value + delta
            else:
                new_value = current_value - delta

            new_value = max(int(v_min), min(int(v_max), new_value))
            logger.info(f"Perturbing {param_name}: {current_value} -> {new_value}")
            return new_value

        if data_type == "ordered_int":
            # Pick adjacent value in ordered list
            valid_options = get_valid_options(param_config, self.custom_ranges)
            if not valid_options or current_value not in valid_options:
                return current_value

            current_idx = valid_options.index(current_value)
            # Move up or down one step
            if np.random.rand() < 0.5 and current_idx < len(valid_options) - 1:
                new_value = valid_options[current_idx + 1]
            elif current_idx > 0:
                new_value = valid_options[current_idx - 1]
            else:
                new_value = current_value

            logger.info(f"Perturbing {param_name}: {current_value} -> {new_value}")
            return new_value

        if data_type == "bool":
            # 30% chance to flip boolean value
            if np.random.rand() < 0.3:
                new_value = not current_value
                logger.info(f"Perturbing {param_name}: {current_value} -> {new_value}")
                return new_value
            return current_value

        if data_type in ("categorical", "ordered"):
            # Pick a different random value from the options
            valid_options = get_valid_options(param_config, self.custom_ranges)
            if not valid_options or valid_options == "":
                return current_value

            # Filter out current value to get alternatives
            if isinstance(valid_options, (list, tuple)):
                alternative_options = [opt for opt in valid_options if opt != current_value]
            else:
                alternative_options = []

            if alternative_options:
                # 50% chance to change to a different option
                if np.random.rand() < 0.5:
                    # Get weights if available
                    weights = get_option_weights(param_config, self.custom_ranges)
                    if weights and len(weights) == len(valid_options):
                        # Build weights for alternatives only
                        alt_weights = []
                        for i, opt in enumerate(valid_options):
                            if opt != current_value:
                                alt_weights.append(weights[i])
                        # Normalize weights
                        total_weight = sum(alt_weights)
                        if total_weight > 0:
                            probabilities = [w / total_weight for w in alt_weights]
                            new_value = np.random.choice(alternative_options, p=probabilities)
                        else:
                            new_value = np.random.choice(alternative_options)
                    else:
                        new_value = np.random.choice(alternative_options)

                    logger.info(f"Perturbing {param_name}: {current_value} -> {new_value}")
                    return new_value

            return current_value

        # For other complex types (lists, dicts), keep unchanged
        return current_value

    def _exploit_and_explore(self, member_id, population_results, member_job_id=None):
        """Apply exploit (copy better member) and explore (perturb) to a member

        Args:
            member_id: ID of member to potentially replace
            population_results: sorted list of (member_id, result) tuples
            member_job_id: Job ID of the member (for logging)

        Returns:
            Tuple of (new_specs, source_member_id) for replaced members, or (None, None) if member survives
        """
        # Find member's rank in population
        member_result = self.population[member_id]["result"]
        member_rank = next(
            (i for i, (mid, _) in enumerate(population_results) if mid == member_id),
            len(population_results) - 1
        )

        # Bottom 20% get replaced
        threshold_rank = int(0.8 * len(population_results))
        if member_rank < threshold_rank:
            # This member is good enough, no change
            logger.info(
                f"Member {member_id} (rank {member_rank}, result={member_result:.4f}, job_id={member_job_id}) survives"
            )
            return None, None

        # Exploit: copy from top 20%
        top_rank = int(0.2 * len(population_results))
        top_members = population_results[:max(1, top_rank)]
        source_id, source_result = top_members[np.random.randint(len(top_members))]

        logger.info(
            f"Member {member_id} (rank {member_rank}, result={member_result:.4f}, job_id={member_job_id}) "
            f"replaced by member {source_id} (result={source_result:.4f})"
        )

        # Copy specs from source member
        new_specs = copy.deepcopy(self.population[source_id]["specs"])

        # Explore: perturb hyperparameters
        for param_config in self.perturbable_params:
            param_name = param_config["parameter"]
            if param_name in new_specs:
                current_value = new_specs[param_name]
                new_value = self._perturb_parameter(param_config, current_value)
                new_specs[param_name] = new_value

        return new_specs, source_id

    def _generate_random_parameters(self):
        """Generate random parameter values for a new population member"""
        hyperparam_dict = {}
        for param in self.parameters:
            name = param["parameter"]
            rec = self.generate_automl_param_rec_value(param)
            logger.info(f"Generated random parameter in PBT: {name} = {rec}")
            hyperparam_dict[name] = rec
        return hyperparam_dict

    @property
    def max_concurrent(self):
        """Return maximum number of concurrent experiments for PBT

        PBT trains entire population in parallel each generation,
        so max_concurrent = population_size
        """
        return self.population_size

    def done(self):
        """Return if PBT algorithm is complete or not"""
        return self.complete

    def generate_recommendations(self, history):
        """Generate recommendations using population-based training"""
        get_flatten_specs(self.default_train_spec, self.default_train_spec_flattened)

        if history == []:
            # Initialize population with random configurations
            recommendations = []
            for _ in range(self.population_size):
                specs = self._generate_random_parameters()
                member_id = self.next_member_id
                self.population[member_id] = {
                    "specs": specs,
                    "result": 0.0,
                    "epochs": 0
                }
                self.next_member_id += 1
                recommendations.append(specs)
            self.track_id = 0
            return recommendations

        # Check if generation is complete (all members trained for eval_interval epochs)
        all_complete = all(
            rec.status in [JobStates.success, JobStates.failure]
            for rec in history[-self.population_size:]
        )

        if not all_complete:
            return []  # Wait for all members to complete

        # Update population results from history
        for rec in history[-self.population_size:]:
            member_id = rec.id
            if member_id in self.population:
                self.population[member_id]["result"] = rec.result
                self.population[member_id]["epochs"] = self.population[member_id].get("epochs", 0) + self.eval_interval

        # Check if we've reached max generations
        self.generation += 1
        # Update epoch_number for next generation (used for early_stop_epoch)
        self.epoch_number = (self.generation + 1) * self.eval_interval
        logger.info(
            f"Completed generation {self.generation}/{self.max_generations}, "
            f"next epoch_number={self.epoch_number}"
        )

        if self.generation >= self.max_generations:
            self.complete = True
            return []

        # Sort population by performance
        population_results = sorted(
            [(mid, self.population[mid]["result"]) for mid in self.population.keys()],
            key=lambda x: x[1],
            reverse=self.reverse_sort
        )

        # Apply exploit and explore to each member
        recommendations = []
        for member_id in self.population.keys():
            # Find the current member's job_id from recent history
            member_job_id = None
            for rec in reversed(history):
                if rec.id == member_id:
                    member_job_id = rec.job_id
                    break

            new_specs, source_id = self._exploit_and_explore(member_id, population_results, member_job_id)

            if new_specs is not None:
                # Member was replaced, resume with new specs from source member
                # Find source member's job_id from recent history
                source_job_id = None
                for rec in reversed(history):
                    if rec.id == source_id:
                        source_job_id = rec.job_id
                        break

                if source_job_id:
                    logger.info(
                        f"PBT: Member {member_id} (replaced by {source_id}) "
                        f"will resume from checkpoint of source job {source_job_id}"
                    )
                else:
                    logger.warning(
                        f"PBT: Could not find source job_id for Member {source_id}, "
                        f"member {member_id} will start from scratch"
                    )

                self.population[member_id]["specs"] = new_specs
                self.population[member_id]["result"] = 0.0  # Reset result
                # Pass source_job_id directly to ResumeRecommendation
                resume_rec = ResumeRecommendation(member_id, new_specs, member_job_id, resume_from_job_id=source_job_id)
                recommendations.append(resume_rec)
            else:
                # Member continues with same specs (no replacement)
                specs = self.population[member_id]["specs"]
                resume_rec = ResumeRecommendation(member_id, specs, member_job_id, resume_from_job_id=None)
                recommendations.append(resume_rec)

        self.track_id = list(self.population.keys())[0]
        return recommendations
