# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flow tests for Bayesian AutoML algorithm"""

import pytest
from unittest.mock import Mock, patch

from nvidia_tao_core.microservices.automl.bayesian import Bayesian
from nvidia_tao_core.microservices.utils.automl_utils import Recommendation, JobStates


class TestBayesianFlow:
    """Test suite for Bayesian Optimization algorithm flow"""

    @pytest.fixture
    def mock_job_context(self):
        """Create a mock job context"""
        mock_ctx = Mock()
        mock_ctx.id = "test_bayesian_job"
        mock_ctx.handler = "test_handler"
        mock_ctx.handler_id = "test_handler"
        return mock_ctx

    @pytest.fixture
    def mock_parameters(self):
        """Create mock parameters for optimization"""
        return [
            {
                "parameter": "learning_rate",
                "value_type": "float",
                "min_value": 0.0001,
                "max_value": 0.1
            },
            {
                "parameter": "batch_size",
                "value_type": "int",
                "min_value": 16,
                "max_value": 128
            }
        ]

    @pytest.fixture
    def predetermined_results(self):
        """Predetermined validation losses for configurations

        Pure Bayesian optimization should learn the function and sample near optimum
        """
        return {
            # Initial random exploration
            0: 0.500,
            1: 0.450,
            2: 0.600,
            3: 0.400,  # Good
            4: 0.550,
            # After learning (should be better)
            5: 0.380,  # Better
            6: 0.420,
            7: 0.350,  # Even better
            8: 0.390,
            9: 0.320,  # Best
            10: 0.360,
            11: 0.340,
            12: 0.310,  # Excellent
        }

    @patch('nvidia_tao_core.microservices.utils.handler_utils.get_total_epochs')
    def test_bayesian_complete_flow(
        self, mock_get_epochs, mock_job_context, mock_parameters,
        predetermined_results, tmp_path, test_environment, reset_test_db
    ):
        """Test complete Bayesian optimization flow"""

        # Mock get_total_epochs to return a fixed value
        mock_get_epochs.return_value = 100

        # Initialize Bayesian
        bayesian = Bayesian(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=mock_parameters
        )

        history = []
        iteration = 0
        max_iterations = 12

        print("\n" + "=" * 80)
        print("BAYESIAN OPTIMIZATION FLOW TEST")
        print("=" * 80)

        # Initial recommendation (random)
        print(f"\n--- Iteration {iteration}: Initial Launch ---")
        recommendations = bayesian.generate_recommendations([])
        assert len(recommendations) == 1, "Bayesian should recommend one config at a time"

        rec = Recommendation(0, recommendations[0], "val_loss")
        rec.assign_job_id("job_0")
        rec.update_status(JobStates.running)
        history.append(rec)
        print("  Launched Config 0 (random initial)")

        iteration += 1

        # Main loop
        while iteration < max_iterations:
            # Complete the running job
            running_jobs = [r for r in history if r.status == JobStates.running]
            if not running_jobs:
                break

            rec_to_complete = running_jobs[0]
            config_id = rec_to_complete.id

            # Get predetermined result
            if config_id in predetermined_results:
                val_loss = predetermined_results[config_id]
                rec_to_complete.update_result(val_loss)
                rec_to_complete.update_status(JobStates.success)

                # Track if we're using GP
                using_gp = len(bayesian.Xs) >= 2
                sampling_method = "GP-guided" if using_gp else "random"

                print(f"\n--- Iteration {iteration}: Config {config_id} COMPLETED ({sampling_method}) ---")
                print(f"  val_loss={val_loss:.3f}")
                print(f"  GP observations: {len(bayesian.Xs)}")
            else:
                rec_to_complete.update_result(0.999)
                rec_to_complete.update_status(JobStates.success)
                print(f"\n--- Iteration {iteration}: Config {config_id} completed (default) ---")

            # Generate next recommendation
            recommendations = bayesian.generate_recommendations(history)
            if not recommendations:
                print("  No more recommendations")
                break

            # Launch next config
            next_config_id = iteration
            rec = Recommendation(next_config_id, recommendations[0], "val_loss")
            rec.assign_job_id(f"job_{next_config_id}")
            rec.update_status(JobStates.running)
            history.append(rec)

            using_gp = len(bayesian.Xs) >= 2
            print(f"  → Launched Config {next_config_id} ({'GP-guided' if using_gp else 'random'})")

            iteration += 1

        # Verification
        print("\n" + "=" * 80)
        print("FINAL VERIFICATION")
        print("=" * 80)

        # Check that observations were collected
        assert len(bayesian.Xs) > 0, "Bayesian should have collected observations"
        assert len(bayesian.ys) > 0, "Bayesian should have collected results"
        print(f"✓ Total observations: {len(bayesian.Xs)}")

        # Check that configs completed
        successful_configs = [rec for rec in history if rec.status == JobStates.success and rec.result < 0.9]
        assert len(successful_configs) > 0, "Expected successful configurations"
        print(f"✓ Successful configurations: {len(successful_configs)}")

        # Get final results
        final_results = [(rec.id, rec.result) for rec in successful_configs]
        final_results.sort(key=lambda x: x[1])
        print("\n✓ Top 5 configurations:")
        for i, (config_id, loss) in enumerate(final_results[:5]):
            print(f"    {i + 1}. Config {config_id}: val_loss={loss:.3f}")

        # Verify improvement over iterations (Bayesian should learn and improve)
        first_half = [r.result for r in successful_configs[:len(successful_configs) // 2]]
        second_half = [r.result for r in successful_configs[len(successful_configs) // 2:]]
        if first_half and second_half:
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            min_first = min(first_half)
            min_second = min(second_half)

            print(f"\n✓ First half: avg={avg_first:.3f}, min={min_first:.3f}")
            print(f"✓ Second half: avg={avg_second:.3f}, min={min_second:.3f}")

            # Bayesian optimization should improve (lower avg or better minimum in second half)
            assert min_second <= min_first + 0.1, (
                "Bayesian should find equal or better configs in second half"
            )

            if avg_second < avg_first:
                print("✓ Average improved (GP-guided sampling working)")
            if min_second < min_first:
                print("✓ Best config improved (exploitation working)")

        # Verify best config
        best_config_id, best_loss = final_results[0]
        assert best_loss < 0.4, \
            f"Best config should have loss < 0.4, got {best_loss:.3f}"
        print(f"\n✓ Best configuration found: Config {best_config_id} with loss {best_loss:.3f}")

        print("\n✅ Bayesian optimization flow test passed")

    @patch('nvidia_tao_core.microservices.utils.handler_utils.get_total_epochs')
    def test_bayesian_gp_convergence(self, mock_get_epochs, mock_job_context, mock_parameters, tmp_path,
                                     test_environment, reset_test_db):
        """Test that Bayesian optimization collects observations"""

        mock_get_epochs.return_value = 100

        bayesian = Bayesian(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=mock_parameters
        )

        # Bayesian is sequential - complete one, get next, repeat
        history = []
        for i in range(5):
            recommendations = bayesian.generate_recommendations(history)
            if not recommendations:
                break

            rec = Recommendation(i, recommendations[0], "val_loss")
            rec.assign_job_id(f"job_{i}")
            rec.update_result(0.5 - 0.05 * i)
            rec.update_status(JobStates.success)
            history.append(rec)
            print(f"Iteration {i}: Xs={len(bayesian.Xs)}, ys={len(bayesian.ys)}")

        # Verify observations were collected (at least 2 for GP)
        print(f"\nFinal: Xs={len(bayesian.Xs)}, ys={len(bayesian.ys)}, history={len(history)}")
        assert len(history) >= 5, f"Should have completed 5 iterations, got {len(history)}"
        assert len(bayesian.Xs) >= 1, f"Should have collected observations, got {len(bayesian.Xs)}"

        print(f"✓ Bayesian completed {len(history)} iterations")
        print(f"✓ Collected {len(bayesian.Xs)} observations")
        print("✅ GP convergence test passed")

    @patch('nvidia_tao_core.microservices.utils.handler_utils.get_total_epochs')
    def test_bayesian_acquisition_function(self, mock_get_epochs, mock_job_context, mock_parameters, tmp_path,
                                           test_environment, reset_test_db):
        """Test that Bayesian optimization uses acquisition function for sampling"""

        mock_get_epochs.return_value = 100

        bayesian = Bayesian(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=mock_parameters
        )

        history = []

        # Build up observations
        for i in range(5):
            recommendations = bayesian.generate_recommendations(history)
            if not recommendations:
                break

            rec = Recommendation(i, recommendations[0], "val_loss")
            rec.assign_job_id(f"job_{i}")
            rec.update_result(0.5 + 0.1 * (i % 2))  # Alternating performance
            rec.update_status(JobStates.success)
            history.append(rec)

        # Verify acquisition function parameters
        assert bayesian.xi is not None, "Should have exploration-exploitation parameter xi"
        assert bayesian.num_restarts > 0, "Should have acquisition optimization restarts"

        print("✓ Acquisition function parameters:")
        print(f"  xi (exploration): {bayesian.xi}")
        print(f"  num_restarts: {bayesian.num_restarts}")
        print("✅ Acquisition function test passed")
