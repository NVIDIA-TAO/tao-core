# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flow tests for BOHB AutoML algorithm"""

import pytest
from unittest.mock import Mock

from nvidia_tao_core.microservices.automl.bohb import BOHB
from nvidia_tao_core.microservices.utils.automl_utils import Recommendation, ResumeRecommendation, JobStates
from nvidia_tao_core.microservices.utils.stateless_handler_utils import save_job_specs


class TestBOHBFlow:
    """Test suite for BOHB (Bayesian Optimization HyperBand) algorithm flow"""

    @pytest.fixture
    def mock_job_context(self):
        """Create a mock job context"""
        mock_ctx = Mock()
        mock_ctx.id = "test_bohb_job"
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

        BOHB should learn from these and sample better configs over time
        """
        return {
            # Initial random configs
            0: {1: 0.450, 3: 0.380, 9: 0.290},
            1: {1: 0.520, 3: 0.500, 9: None},
            2: {1: 0.480, 3: 0.410, 9: None},
            3: {1: 0.370, 3: 0.300, 9: 0.240},   # Best early
            4: {1: 0.600, 3: None, 9: None},
            5: {1: 0.530, 3: 0.510, 9: None},
            6: {1: 0.440, 3: 0.370, 9: 0.280},
            7: {1: 0.500, 3: 0.450, 9: None},
            8: {1: 0.560, 3: None, 9: None},
            # Later configs (should be better due to Bayesian optimization)
            9: {1: 0.380, 3: 0.310, 9: 0.250},
            10: {1: 0.390, 3: 0.320, 9: 0.260},
            11: {1: 0.360, 3: 0.290, 9: 0.230},  # Best overall
            12: {1: 0.400, 3: 0.330, 9: 0.270},
        }

    def test_bohb_complete_flow(
        self, mock_job_context, mock_parameters, predetermined_results,
        tmp_path, test_environment, reset_test_db
    ):
        """Test BOHB flow through one complete bracket (synchronous like Hyperband)"""

        save_job_specs(mock_job_context.id, {"train_config": {"num_epochs": 1}})

        # Initialize BOHB
        bohb = BOHB(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=mock_parameters,
            max_epochs=9,
            reduction_factor=3,
            epoch_multiplier=1,
            kde_samples=64,
            top_n_percent=15.0,
            min_points_in_model=3,  # Lower threshold for testing
            metric="val_loss"  # Algorithm will automatically set reverse_sort=False
        )

        history = []

        print("\n" + "=" * 80)
        print("BOHB FLOW TEST - BAYESIAN OPTIMIZATION + HYPERBAND")
        print("=" * 80)

        # Step 1: Launch rung 0 (9 configs at 1 epoch)
        print("\n--- Step 1: Launch Rung 0 (9 configs @ 1 epoch) ---")
        recommendations = bohb.generate_recommendations([])
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
        recommendations = bohb.generate_recommendations(history)
        promotions = [r for r in recommendations if isinstance(r, ResumeRecommendation)]
        assert len(promotions) == 3, f"Expected 3 promotions, got {len(promotions)}"

        # Verify the TOP 3 configs are promoted (Bayesian optimization + Hyperband)
        rung0_results = [(rec.id, rec.result) for rec in history if rec.status == JobStates.success]
        rung0_results.sort(key=lambda x: x[1])  # Sort by loss ascending
        top3_ids = set([rung0_results[i][0] for i in range(3)])
        promoted_ids = set([r.id for r in promotions])

        assert promoted_ids == top3_ids, \
            f"BOHB should promote top 3 configs {top3_ids}, but got {promoted_ids}"
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
        recommendations = bohb.generate_recommendations(history)
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

        # Check observations were collected
        assert len(bohb.observations) > 0, "BOHB should have collected observations"
        print(f"✓ Total observations collected: {len(bohb.observations)}")

        successful_configs = [rec for rec in history if rec.status == JobStates.success and rec.result < 0.9]
        assert len(successful_configs) > 0, "Expected successful configurations"

        print(f"✓ Total configurations: {len(history)}")
        print(f"✓ Successful configurations: {len(successful_configs)}")

        # Get best config
        final_results = [(rec.id, rec.result) for rec in successful_configs]
        final_results.sort(key=lambda x: x[1])
        best_config_id, best_loss = final_results[0]

        print("\n✓ Best configuration:")
        print(f"    Config {best_config_id}: val_loss={best_loss:.3f}")

        print("\n✅ BOHB flow test passed")

    def test_bohb_bayesian_sampling(self, mock_job_context, mock_parameters, tmp_path, test_environment, reset_test_db):
        """Test that BOHB collects observations for Bayesian sampling"""

        save_job_specs(mock_job_context.id, {"train_config": {"num_epochs": 1}})

        bohb = BOHB(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=mock_parameters,
            max_epochs=9,
            reduction_factor=3,
            epoch_multiplier=1,
            min_points_in_model=3,
            metric="val_loss"
        )

        # Verify algorithm correctly set reverse_sort based on metric
        assert bohb.reverse_sort is False, "Algorithm should set reverse_sort=False for loss metrics"

        history = []

        # Launch first batch (9 configs at rung 0)
        recommendations = bohb.generate_recommendations([])
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

        # Trigger observation collection by generating next recommendations
        bohb.generate_recommendations(history)

        # Verify observations were collected
        # Note: BOHB filters duplicate configs, so may collect fewer than 9 unique observations
        print(f"Collected {len(bohb.observations)} observations (configs completed: {len(history)})")
        assert len(bohb.observations) > 0, \
            f"Should have collected at least 1 observation, got {len(bohb.observations)}"
        assert len(history) == 9, f"Should have completed 9 configs, got {len(history)}"

        print(f"✓ Completed {len(history)} configurations")
        print(f"✓ Collected {len(bohb.observations)} unique observations")
        print("✅ Bayesian sampling test passed")
