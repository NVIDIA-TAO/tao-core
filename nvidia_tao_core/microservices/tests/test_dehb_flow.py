# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flow tests for DEHB AutoML algorithm"""

import pytest
from unittest.mock import Mock

from nvidia_tao_core.microservices.automl.dehb import DEHB
from nvidia_tao_core.microservices.utils.automl_utils import Recommendation, ResumeRecommendation, JobStates
from nvidia_tao_core.microservices.utils.stateless_handler_utils import save_job_specs


class TestDEHBFlow:
    """Test suite for DEHB (Differential Evolution HyperBand) algorithm flow"""

    @pytest.fixture
    def mock_job_context(self):
        """Create a mock job context"""
        mock_ctx = Mock()
        mock_ctx.id = "test_dehb_job"
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

        DEHB should evolve better configurations using differential evolution
        """
        return {
            # Initial population
            0: {1: 0.450, 3: 0.380, 9: 0.290},
            1: {1: 0.520, 3: 0.500, 9: None},
            2: {1: 0.480, 3: 0.410, 9: None},
            3: {1: 0.370, 3: 0.300, 9: 0.240},
            4: {1: 0.600, 3: None, 9: None},
            5: {1: 0.530, 3: 0.510, 9: None},
            # Evolved configs (should be better)
            6: {1: 0.390, 3: 0.320, 9: 0.260},
            7: {1: 0.380, 3: 0.310, 9: 0.250},
            8: {1: 0.360, 3: 0.290, 9: 0.230},  # Best evolved
            9: {1: 0.400, 3: 0.330, 9: 0.270},
            10: {1: 0.410, 3: 0.340, 9: 0.280},
            11: {1: 0.370, 3: 0.300, 9: 0.245},
            12: {1: 0.420, 3: 0.350, 9: 0.285},
        }

    def test_dehb_complete_flow(
        self, mock_job_context, mock_parameters, predetermined_results,
        tmp_path, test_environment, reset_test_db
    ):
        """Test DEHB flow through one complete bracket (synchronous like Hyperband)"""

        save_job_specs(mock_job_context.id, {"train_config": {"num_epochs": 1}})

        # Initialize DEHB
        dehb = DEHB(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=mock_parameters,
            max_epochs=9,
            reduction_factor=3,
            epoch_multiplier=1,
            mutation_factor=0.5,
            crossover_prob=0.5,
            metric="val_loss"  # Algorithm will automatically set reverse_sort=False
        )

        history = []

        print("\n" + "=" * 80)
        print("DEHB FLOW TEST - DIFFERENTIAL EVOLUTION + HYPERBAND")
        print("=" * 80)

        # Step 1: Launch rung 0 (9 configs at 1 epoch)
        print("\n--- Step 1: Launch Rung 0 (9 configs @ 1 epoch) ---")
        recommendations = dehb.generate_recommendations([])
        assert len(recommendations) == 9, f"Expected 9 configs, got {len(recommendations)}"

        for i, rec_specs in enumerate(recommendations):
            rec = Recommendation(i, rec_specs, "val_loss")
            rec.assign_job_id(f"job_{i}")
            rec.update_status(JobStates.running)
            history.append(rec)

        # Complete all rung 0 jobs (synchronous)
        print("--- Completing Rung 0 ---")
        for rec in history:
            if rec.status == JobStates.running:
                if rec.id in predetermined_results and 1 in predetermined_results[rec.id]:
                    rec.update_result(predetermined_results[rec.id][1])
                    rec.update_status(JobStates.success)
                    print(f"  Config {rec.id}: val_loss={predetermined_results[rec.id][1]:.3f}")
                else:
                    rec.update_result(0.5)
                    rec.update_status(JobStates.success)

        # Step 2: Get promotions to rung 1
        print("\n--- Step 2: Get Promotions to Rung 1 (top 3 @ 3 epochs) ---")
        recommendations = dehb.generate_recommendations(history)
        promotions = [r for r in recommendations if isinstance(r, ResumeRecommendation)]
        assert len(promotions) == 3, f"Expected 3 promotions, got {len(promotions)}"

        # Verify the TOP 3 configs are promoted (DE + Hyperband)
        rung0_results = [(rec.id, rec.result) for rec in history if rec.status == JobStates.success]
        rung0_results.sort(key=lambda x: x[1])  # Sort by loss ascending
        top3_ids = set([rung0_results[i][0] for i in range(3)])
        promoted_ids = set([r.id for r in promotions])

        assert promoted_ids == top3_ids, \
            f"DEHB should promote top 3 configs {top3_ids}, but got {promoted_ids}"
        print(f"  ✓ Correctly promoted top 3: {promoted_ids}")

        for rec_item in promotions:
            for rec in history:
                if rec.id == rec_item.id:
                    rec.update_status(JobStates.running)
                    print(f"  Promoted Config {rec.id} (loss={rec.result:.3f})")

        # Complete all rung 1 jobs
        print("--- Completing Rung 1 ---")
        for rec in history:
            if rec.status == JobStates.running:
                if rec.id in predetermined_results and 3 in predetermined_results[rec.id]:
                    rec.update_result(predetermined_results[rec.id][3])
                    rec.update_status(JobStates.success)
                    print(f"  Config {rec.id}: val_loss={predetermined_results[rec.id][3]:.3f}")
                else:
                    rec.update_result(0.5)
                    rec.update_status(JobStates.success)

        # Step 3: Get promotion to rung 2
        print("\n--- Step 3: Get Promotion to Rung 2 (top 1 @ 9 epochs) ---")
        recommendations = dehb.generate_recommendations(history)
        promotions = [r for r in recommendations if isinstance(r, ResumeRecommendation)]
        assert len(promotions) == 1, f"Expected 1 promotion, got {len(promotions)}"

        for rec_item in promotions:
            for rec in history:
                if rec.id == rec_item.id:
                    rec.update_status(JobStates.running)
                    print(f"  Promoted Config {rec.id}")

        # Complete rung 2 job
        print("--- Completing Rung 2 ---")
        for rec in history:
            if rec.status == JobStates.running:
                if rec.id in predetermined_results and 9 in predetermined_results[rec.id]:
                    rec.update_result(predetermined_results[rec.id][9])
                    rec.update_status(JobStates.success)
                    print(f"  Config {rec.id}: val_loss={predetermined_results[rec.id][9]:.3f}")
                else:
                    rec.update_result(0.5)
                    rec.update_status(JobStates.success)

        # Verification
        print("\n" + "=" * 80)
        print("FINAL VERIFICATION")
        print("=" * 80)

        # Check population was built
        assert len(dehb.population) > 0, "DEHB should have built a population"
        print(f"✓ DE population size: {len(dehb.population)}")

        successful_configs = [rec for rec in history if rec.status == JobStates.success and rec.result < 0.9]
        assert len(successful_configs) > 0, "Expected successful configurations"

        print(f"✓ Total configurations: {len(history)}")
        print(f"✓ Successful configurations: {len(successful_configs)}")

        # Get best config
        final_results = [(rec.id, rec.result) for rec in successful_configs]
        final_results.sort(key=lambda x: x[1])
        best_config_id, best_loss = final_results[0]

        print("\n✓ Best configuration:")
        print(
            f"    Config {best_config_id}: val_loss={best_loss:.3f}"
        )

        print("\n✅ DEHB flow test passed")

    def test_dehb_differential_evolution(
        self, mock_job_context, mock_parameters, tmp_path,
        test_environment, reset_test_db
    ):
        """Test that DEHB builds DE population"""

        save_job_specs(mock_job_context.id, {"train_config": {"num_epochs": 1}})

        dehb = DEHB(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=mock_parameters,
            max_epochs=9,
            reduction_factor=3,
            epoch_multiplier=1,
            mutation_factor=0.5,
            crossover_prob=0.5,
            metric="val_loss"
        )

        # Verify algorithm correctly set reverse_sort based on metric
        assert dehb.reverse_sort is False, "Algorithm should set reverse_sort=False for loss metrics"

        history = []

        # Launch first batch (9 configs at rung 0)
        recommendations = dehb.generate_recommendations([])
        assert len(recommendations) == 9, f"Expected 9 configs, got {len(recommendations)}"

        for i, rec_specs in enumerate(recommendations):
            rec = Recommendation(i, rec_specs, "val_loss")
            rec.assign_job_id(f"job_{i}")
            rec.update_status(JobStates.running)
            history.append(rec)

        # Complete all jobs (synchronous)
        for rec in history:
            rec.update_result(0.5 - 0.01 * rec.id)  # Decreasing loss
            rec.update_status(JobStates.success)

        # Trigger population building by generating next recommendations
        dehb.generate_recommendations(history)

        # Verify population was built
        print(f"DE population size: {len(dehb.population)}")
        assert len(dehb.population) > 0, f"Should have built DE population, got {len(dehb.population)}"

        print(f"✓ Built DE population with {len(dehb.population)} members")
        print("✅ Differential evolution test passed")
