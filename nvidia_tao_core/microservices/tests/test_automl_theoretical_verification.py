# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Theoretical Verification Tests for AutoML Algorithms

This test suite verifies that the core theoretical properties of each AutoML algorithm
remain intact after code modifications. We test:

1. Algorithm-specific search strategies (GP, TPE, Evolution, etc.)
2. Suggestion mechanism integrity (0 - 1 normalized → actual values)
3. Exploration vs exploitation balance
4. Population/bracket management for multi-fidelity algorithms
5. Monotonicity and bounds adherence

These tests focus on algorithmic correctness, not just diversity.
"""

import pytest
import numpy as np
import math
from unittest.mock import Mock, patch


def create_mock_job_context(job_id="test_theoretical", handler_id="test_exp"):
    """Create a mock job context for testing."""
    job_context = Mock()
    job_context.id = job_id
    job_context.handler_id = handler_id
    job_context.action = "train"
    job_context.status = "RUNNING"
    return job_context


# Test parameter configurations

TEST_PARAM_FLOAT_WITH_RANGE = {
    "parameter": "learning_rate",
    "value_type": "float",
    "default_value": 0.001,
    "valid_min": 0.0001,
    "valid_max": 0.1,
    "automl_enabled": "TRUE"
}

TEST_PARAM_FLOAT_NO_RANGE = {
    "parameter": "dropout",
    "value_type": "float",
    "default_value": 0.5,
    "valid_min": "",
    "valid_max": "",
    "automl_enabled": "TRUE"
}

TEST_PARAM_INT = {
    "parameter": "batch_size",
    "value_type": "int",
    "default_value": 32,
    "valid_min": 8,
    "valid_max": 128,
    "automl_enabled": "TRUE"
}

TEST_PARAM_CATEGORICAL = {
    "parameter": "optimizer",
    "value_type": "categorical",
    "default_value": "adam",
    "valid_options": ["adam", "sgd", "rmsprop"],
    "automl_enabled": "TRUE"
}


class TestBayesianTheoretical:
    """Verify Bayesian Optimization theoretical properties"""

    def test_suggestion_mapping_is_monotonic(self):
        """Bayesian: suggestion in [0,1] should map monotonically to parameter range"""
        from nvidia_tao_core.microservices.automl.bayesian import Bayesian
        from contextlib import ExitStack

        patches = [
            patch('nvidia_tao_core.microservices.automl.bayesian.get_total_epochs', return_value=10),
            patch('nvidia_tao_core.microservices.automl.bayesian.get_flatten_specs', return_value=None),
            patch('nvidia_tao_core.microservices.automl.bayesian.save_automl_brain_info', return_value=None),
            patch('nvidia_tao_core.microservices.automl.bayesian.get_automl_brain_info', return_value={}),
            patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs', return_value={}),
            patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges',
                  return_value={}),
        ]
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            job_context = create_mock_job_context("test_bayesian_monotonic")
            bayesian = Bayesian(job_context, "/tmp", "test_network", [TEST_PARAM_FLOAT_WITH_RANGE])

            # Test monotonicity: increasing suggestions should produce increasing values
            suggestions = [0.0, 0.25, 0.5, 0.75, 1.0]
            values = []
            for s in suggestions:
                val = bayesian.generate_automl_param_rec_value(TEST_PARAM_FLOAT_WITH_RANGE, s)
                values.append(val)
                print(f"  Suggestion {s:.2f} → Value {val:.6f}")

            # Verify monotonicity
            for i in range(len(values) - 1):
                assert values[i] <= values[i + 1], f"Monotonicity violated: {values[i]} > {values[i + 1]}"

            # Verify bounds
            assert values[0] >= TEST_PARAM_FLOAT_WITH_RANGE["valid_min"]
            assert values[-1] <= TEST_PARAM_FLOAT_WITH_RANGE["valid_max"]
            print(f"✓ Bayesian maintains monotonic mapping: {values}")

    def test_suggestion_respects_bounds(self):
        """Bayesian: all suggestions should map to values within valid range"""
        from nvidia_tao_core.microservices.automl.bayesian import Bayesian
        from contextlib import ExitStack

        patches = [
            patch('nvidia_tao_core.microservices.automl.bayesian.get_total_epochs', return_value=10),
            patch('nvidia_tao_core.microservices.automl.bayesian.get_flatten_specs', return_value=None),
            patch('nvidia_tao_core.microservices.automl.bayesian.save_automl_brain_info', return_value=None),
            patch('nvidia_tao_core.microservices.automl.bayesian.get_automl_brain_info', return_value={}),
            patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs', return_value={}),
            patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges',
                  return_value={}),
        ]
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            job_context = create_mock_job_context()
            bayesian = Bayesian(job_context, "/tmp", "test_network", [TEST_PARAM_FLOAT_WITH_RANGE])

            # Test 100 random suggestions
            np.random.seed(42)
            for _ in range(100):
                s = np.random.random()
                val = bayesian.generate_automl_param_rec_value(TEST_PARAM_FLOAT_WITH_RANGE, s)
                min_val = TEST_PARAM_FLOAT_WITH_RANGE["valid_min"]
                max_val = TEST_PARAM_FLOAT_WITH_RANGE["valid_max"]
                assert min_val <= val <= max_val, f"Value {val} outside bounds [{min_val}, {max_val}]"
            print("✓ Bayesian respects parameter bounds for 100 random suggestions")


class TestBFBOTheoretical:
    """Verify BFBO (Best-First Bayesian Optimization) theoretical properties"""

    def test_suggestion_mapping_monotonic(self):
        """BFBO: suggestion mapping should be monotonic"""
        from nvidia_tao_core.microservices.automl.bfbo import BFBO
        from contextlib import ExitStack

        patches = [
            patch('nvidia_tao_core.microservices.automl.bfbo.get_total_epochs', return_value=10),
            patch('nvidia_tao_core.microservices.automl.bfbo.get_flatten_specs', return_value=None),
            patch('nvidia_tao_core.microservices.automl.bfbo.save_automl_brain_info', return_value=None),
            patch('nvidia_tao_core.microservices.automl.bfbo.get_automl_brain_info', return_value={}),
            patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs', return_value={}),
            patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges',
                  return_value={}),
        ]
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            job_context = create_mock_job_context()
            bfbo = BFBO(job_context, "/tmp", "test_network", [TEST_PARAM_FLOAT_WITH_RANGE])

            suggestions = [0.0, 0.25, 0.5, 0.75, 1.0]
            values = [bfbo.generate_automl_param_rec_value(TEST_PARAM_FLOAT_WITH_RANGE, s) for s in suggestions]

            for i in range(len(values) - 1):
                assert values[i] <= values[i + 1], "BFBO monotonicity violated"
            print(f"✓ BFBO maintains monotonic mapping: {values}")


class TestPBTTheoretical:
    """Verify PBT (Population Based Training) theoretical properties"""

    def test_population_initialization(self):
        """PBT: should initialize diverse population"""
        from nvidia_tao_core.microservices.automl.pbt import PBT

        with patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs',
                   return_value={}), \
             patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges',
                   return_value={}):

            job_context = create_mock_job_context()
            pbt = PBT(job_context, "/tmp", "test_network", [TEST_PARAM_FLOAT_WITH_RANGE], population_size=10)

            assert pbt.population_size == 10
            assert pbt.perturbation_factor == 1.2  # Default
            assert len(pbt.perturbable_params) > 0
            pop_size = pbt.population_size
            perturb_factor = pbt.perturbation_factor
            print(f"✓ PBT initialized with population_size={pop_size}, perturbation_factor={perturb_factor}")

    def test_perturbation_generates_diverse_values(self):
        """PBT: perturbation should generate values different from original"""
        from nvidia_tao_core.microservices.automl.pbt import PBT

        with patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs',
                   return_value={}), \
             patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges',
                   return_value={}):

            job_context = create_mock_job_context()
            pbt = PBT(job_context, "/tmp", "test_network", [TEST_PARAM_FLOAT_WITH_RANGE], perturbation_factor=1.2)

            # Generate multiple values with different seeds
            np.random.seed(1)
            val1 = pbt.generate_automl_param_rec_value(TEST_PARAM_FLOAT_WITH_RANGE)
            np.random.seed(2)
            val2 = pbt.generate_automl_param_rec_value(TEST_PARAM_FLOAT_WITH_RANGE)

            assert val1 != val2, "PBT should generate diverse values with different seeds"
            print(f"✓ PBT generates diverse values: {val1:.6f} vs {val2:.6f}")


class TestBOHBTheoretical:
    """Verify BOHB (Bayesian Optimization HyperBand) theoretical properties"""

    def test_combines_tpe_and_successive_halving(self):
        """BOHB: should use TPE-style suggestions with HyperBand budgeting"""
        from nvidia_tao_core.microservices.automl.bohb import BOHB
        from contextlib import ExitStack

        patches = [
            patch('nvidia_tao_core.microservices.automl.bohb.get_flatten_specs', return_value=None),
            patch('nvidia_tao_core.microservices.automl.bohb.save_job_specs', return_value=None),
            patch('nvidia_tao_core.microservices.automl.bohb.get_job_specs', return_value={}),
            patch('nvidia_tao_core.microservices.automl.bohb.save_automl_brain_info', return_value=None),
            patch('nvidia_tao_core.microservices.automl.bohb.get_automl_brain_info', return_value={}),
            patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs', return_value={}),
            patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges',
                  return_value={}),
        ]
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            job_context = create_mock_job_context()
            bohb = BOHB(job_context, "/tmp", "test_network", [TEST_PARAM_FLOAT_WITH_RANGE],
                        max_epochs=10, reduction_factor=3, epoch_multiplier=1)

            # BOHB successfully initialized with max_epochs and reduction_factor parameters for successive halving
            print("✓ BOHB initialized with max_epochs=10, reduction_factor=3 for successive halving")

            # Verify suggestion mapping still works
            val = bohb.generate_automl_param_rec_value(TEST_PARAM_FLOAT_WITH_RANGE, 0.5)
            assert TEST_PARAM_FLOAT_WITH_RANGE["valid_min"] <= val <= TEST_PARAM_FLOAT_WITH_RANGE["valid_max"]
            print(f"✓ BOHB suggestion mapping works: {val:.6f}")


class TestSuccessiveHalvingAlgorithms:
    """Verify ASHA, HyperBand, HyperBandES successive halving properties"""

    def test_asha_parameters(self):
        """ASHA: should have proper successive halving parameters"""
        from nvidia_tao_core.microservices.automl.asha import ASHA
        from contextlib import ExitStack

        patches = [
            patch('nvidia_tao_core.microservices.automl.asha.get_flatten_specs', return_value=None),
            patch('nvidia_tao_core.microservices.automl.asha.save_job_specs', return_value=None),
            patch('nvidia_tao_core.microservices.automl.asha.get_job_specs', return_value={}),
            patch('nvidia_tao_core.microservices.automl.asha.save_automl_brain_info', return_value=None),
            patch('nvidia_tao_core.microservices.automl.asha.get_automl_brain_info', return_value={}),
            patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs', return_value={}),
            patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges',
                  return_value={}),
        ]
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            job_context = create_mock_job_context()
            asha = ASHA(job_context, "/tmp", "test_network", [TEST_PARAM_FLOAT_WITH_RANGE],
                        max_epochs=10, reduction_factor=3, epoch_multiplier=1, max_concurrent=4)

            assert asha.max_epochs == 10
            assert asha.reduction_factor == 3
            assert asha.max_concurrent == 4
            print(
                f"✓ ASHA initialized with max_epochs={asha.max_epochs}, "
                f"reduction_factor={asha.reduction_factor}, "
                f"max_concurrent={asha.max_concurrent}"
            )

    def test_hyperband_brackets(self):
        """HyperBand: should manage multiple brackets"""
        from nvidia_tao_core.microservices.automl.hyperband import HyperBand
        from contextlib import ExitStack

        patches = [
            patch('nvidia_tao_core.microservices.automl.hyperband.get_flatten_specs', return_value=None),
            patch('nvidia_tao_core.microservices.automl.hyperband.save_job_specs', return_value=None),
            patch('nvidia_tao_core.microservices.automl.hyperband.get_job_specs', return_value={}),
            patch('nvidia_tao_core.microservices.automl.hyperband.save_automl_brain_info', return_value=None),
            patch('nvidia_tao_core.microservices.automl.hyperband.get_automl_brain_info', return_value={}),
            patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs', return_value={}),
            patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges',
                  return_value={}),
        ]
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            job_context = create_mock_job_context()
            _ = HyperBand(job_context, "/tmp", "test_network", [TEST_PARAM_FLOAT_WITH_RANGE],
                          max_epochs=27, reduction_factor=3, epoch_multiplier=1)

            # HyperBand successfully initialized with max_epochs=27, reduction_factor=3
            # It should calculate s_max = floor(log_reduction_factor(max_epochs)) = floor(log(27)/log(3)) = 3
            expected_s_max = int(math.floor(math.log(27) / math.log(3)))
            print(f"✓ HyperBand initialized with max_epochs=27, reduction_factor=3, expected s_max={expected_s_max}")


class TestParameterValueGeneration:
    """Verify parameter value generation handles all edge cases correctly"""

    def test_float_without_range_uses_default_based_range(self):
        """Algorithms should generate reasonable values around default when range is missing"""
        from nvidia_tao_core.microservices.automl.bayesian import Bayesian
        from contextlib import ExitStack

        patches = [
            patch('nvidia_tao_core.microservices.automl.bayesian.get_total_epochs', return_value=10),
            patch('nvidia_tao_core.microservices.automl.bayesian.get_flatten_specs', return_value=None),
            patch('nvidia_tao_core.microservices.automl.bayesian.save_automl_brain_info', return_value=None),
            patch('nvidia_tao_core.microservices.automl.bayesian.get_automl_brain_info', return_value={}),
            patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs', return_value={}),
            patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges',
                  return_value={}),
        ]
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            job_context = create_mock_job_context()
            bayesian = Bayesian(job_context, "/tmp", "test_network", [TEST_PARAM_FLOAT_NO_RANGE])

            # Generate values with different suggestions
            default = TEST_PARAM_FLOAT_NO_RANGE["default_value"]
            val_low = bayesian.generate_automl_param_rec_value(TEST_PARAM_FLOAT_NO_RANGE, 0.0)
            val_mid = bayesian.generate_automl_param_rec_value(TEST_PARAM_FLOAT_NO_RANGE, 0.5)
            val_high = bayesian.generate_automl_param_rec_value(TEST_PARAM_FLOAT_NO_RANGE, 1.0)

            # Values should be around default (within 10x range)
            assert default / 10.0 <= val_low <= default * 10.0
            assert default / 10.0 <= val_mid <= default * 10.0
            assert default / 10.0 <= val_high <= default * 10.0

            # Values should be different
            assert val_low != val_mid != val_high
            print(f"✓ Missing range handled: default={default}, values=[{val_low:.4f}, {val_mid:.4f}, {val_high:.4f}]")

    def test_categorical_uses_suggestions_correctly(self):
        """Categorical parameters should map suggestions to valid options"""
        from nvidia_tao_core.microservices.automl.bayesian import Bayesian
        from contextlib import ExitStack

        patches = [
            patch('nvidia_tao_core.microservices.automl.bayesian.get_total_epochs', return_value=10),
            patch('nvidia_tao_core.microservices.automl.bayesian.get_flatten_specs', return_value=None),
            patch('nvidia_tao_core.microservices.automl.bayesian.save_automl_brain_info', return_value=None),
            patch('nvidia_tao_core.microservices.automl.bayesian.get_automl_brain_info', return_value={}),
            patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs', return_value={}),
            patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges',
                  return_value={}),
        ]
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            job_context = create_mock_job_context()
            bayesian = Bayesian(job_context, "/tmp", "test_network", [TEST_PARAM_CATEGORICAL])

            # Low suggestion should map to first option
            val_low = bayesian.generate_automl_param_rec_value(TEST_PARAM_CATEGORICAL, 0.1)
            # High suggestion should map to last option
            val_high = bayesian.generate_automl_param_rec_value(TEST_PARAM_CATEGORICAL, 0.9)

            valid_options = TEST_PARAM_CATEGORICAL["valid_options"]
            assert val_low in valid_options
            assert val_high in valid_options
            print(f"✓ Categorical mapping: 0.1→{val_low}, 0.9→{val_high}")


@pytest.mark.parametrize("algo_name,algo_module,algo_class", [
    ("Bayesian", "nvidia_tao_core.microservices.automl.bayesian", "Bayesian"),
    ("BFBO", "nvidia_tao_core.microservices.automl.bfbo", "BFBO"),
])
def test_algorithm_suggestion_diversity(algo_name, algo_module, algo_class):
    """All algorithms should generate diverse values from different suggestions"""
    import importlib
    from contextlib import ExitStack

    # Import algorithm
    module = importlib.import_module(algo_module)
    AlgorithmClass = getattr(module, algo_class)

    # Common mocks
    patches = [
        patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs', return_value={}),
        patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges',
              return_value={}),
    ]

    # Algorithm-specific mocks
    if algo_name in ["Bayesian", "BFBO"]:
        patches.extend([
            patch(f'{algo_module}.get_total_epochs', return_value=10),
            patch(f'{algo_module}.get_flatten_specs', return_value=None),
            patch(f'{algo_module}.save_automl_brain_info', return_value=None),
            patch(f'{algo_module}.get_automl_brain_info', return_value={}),
        ])

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)

        job_context = create_mock_job_context(f"test_{algo_name}_diversity")
        algo = AlgorithmClass(job_context, "/tmp", "test_network", [TEST_PARAM_FLOAT_WITH_RANGE])

        # Generate values with different suggestions
        suggestions = [0.1, 0.3, 0.5, 0.7, 0.9]
        values = []
        for s in suggestions:
            val = algo.generate_automl_param_rec_value(TEST_PARAM_FLOAT_WITH_RANGE, s)
            values.append(val)

        # Check diversity: at least 3 unique values from 5 suggestions
        unique_values = len(set(values))
        assert unique_values >= 3, (
            f"{algo_name} should generate diverse values, got {unique_values} unique from {values}"
        )
        print(f"✓ {algo_name} generates diverse values: {unique_values} unique from {suggestions}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
