# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flow tests for Hyperband AutoML algorithm"""

import pytest
from unittest.mock import Mock

from nvidia_tao_core.microservices.automl.hyperband import HyperBand
from nvidia_tao_core.microservices.utils.automl_utils import Recommendation, ResumeRecommendation, JobStates
from nvidia_tao_core.microservices.utils.stateless_handler_utils import save_job_specs


class TestHyperBandFlow:
    """Test suite for HyperBand algorithm flow"""

    @pytest.fixture
    def mock_job_context(self):
        """Create a mock job context"""
        mock_ctx = Mock()
        mock_ctx.id = "test_hyperband_job"
        mock_ctx.handler = "test_handler"
        return mock_ctx

    @pytest.fixture
    def predetermined_results(self):
        """Predetermined validation losses for each config at each rung

        Format: {config_id: {rung_epochs: val_loss}}
        Lower loss is better
        """
        return {
            # Bracket 0 configurations
            0: {1: 0.450, 3: 0.380, 9: 0.290},   # Good
            1: {1: 0.520, 3: 0.500, 9: None},    # Poor
            2: {1: 0.480, 3: 0.410, 9: None},    # Mediocre
            3: {1: 0.370, 3: 0.300, 9: 0.240},   # Best
            4: {1: 0.600, 3: None, 9: None},     # Poor
            5: {1: 0.530, 3: 0.510, 9: None},    # Poor
            6: {1: 0.440, 3: 0.370, 9: 0.280},   # Good
            7: {1: 0.500, 3: 0.450, 9: None},    # Mediocre
            8: {1: 0.560, 3: None, 9: None},     # Poor
            # Additional configs
            9: {1: 0.420, 3: 0.350, 9: 0.270},   # Good
            10: {1: 0.490, 3: 0.430, 9: None},   # Mediocre
            11: {1: 0.410, 3: 0.340, 9: 0.260},  # Very good
            12: {1: 0.580, 3: None, 9: None},    # Poor
        }

    def test_hyperband_initialization(self, mock_job_context, tmp_path, test_environment, reset_test_db):
        """Test that Hyperband initializes correctly with proper brackets"""

        save_job_specs(mock_job_context.id, {"train_config": {"num_epochs": 1}})

        hyperband = HyperBand(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=[],
            max_epochs=9,
            reduction_factor=3,
            epoch_multiplier=1,
            metric="val_loss"
        )

        # Verify brackets were created
        assert len(hyperband.ni) > 0, "Should have created brackets"
        assert len(hyperband.ri) > 0, "Should have created resource schedules"

        # Verify bracket 0 structure (max epochs=9, reduction=3 => smax=2)
        assert "0" in hyperband.ni, "Should have bracket 0"
        assert hyperband.ni["0"] == [9, 3, 1], f"Expected ni=[9, 3, 1], got {hyperband.ni['0']}"
        assert hyperband.ri["0"] == [1, 3, 9], f"Expected ri=[1, 3, 9], got {hyperband.ri['0']}"

        print(f"✓ Brackets created: {list(hyperband.ni.keys())}")
        print(f"✓ Bracket 0: ni={hyperband.ni['0']}, ri={hyperband.ri['0']}")
        print("✅ Hyper band initialization test passed")

    def test_hyperband_complete_flow(
        self, mock_job_context, predetermined_results, tmp_path,
        test_environment, reset_test_db
    ):
        """Test Hyperband flow through one complete bracket"""

        save_job_specs(mock_job_context.id, {"train_config": {"num_epochs": 1}})

        hyperband = HyperBand(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=[],
            max_epochs=9,
            reduction_factor=3,
            epoch_multiplier=1,
            metric="val_loss"
        )

        # Verify algorithm correctly set reverse_sort based on metric
        assert hyperband.reverse_sort is False, "Algorithm should set reverse_sort=False for loss metrics"

        history = []

        print("\n" + "=" * 80)
        print("HYPERBAND COMPLETE BRACKET FLOW TEST")
        print("=" * 80)
        print(f"Bracket 0: ni={hyperband.ni['0']}, ri={hyperband.ri['0']}")

        # Step 1: Launch rung 0 (9 configs at 1 epoch)
        print("\n--- Step 1: Launch Rung 0 (9 configs @ 1 epoch) ---")
        recommendations = hyperband.generate_recommendations([])
        assert len(recommendations) == 9, f"Expected 9 configs, got {len(recommendations)}"

        for i, rec_specs in enumerate(recommendations):
            rec = Recommendation(i, rec_specs, "val_loss")
            rec.assign_job_id(f"job_{i}")
            rec.update_status(JobStates.running)
            history.append(rec)

        # Complete all rung 0 jobs
        print("--- Completing Rung 0 ---")
        for rec in history:
            if rec.status == JobStates.running:
                if rec.id in predetermined_results and 1 in predetermined_results[rec.id]:
                    rec.update_result(predetermined_results[rec.id][1])
                    rec.update_status(JobStates.success)
                    print(f"  Config {rec.id}: val_loss={predetermined_results[rec.id][1]:.3f}")
                else:
                    rec.update_result(0.999)
                    rec.update_status(JobStates.success)

        # Step 2: Get promotions to rung 1 (top 3 at 3 epochs)
        print("\n--- Step 2: Get Promotions to Rung 1 (top 3 @ 3 epochs) ---")
        recommendations = hyperband.generate_recommendations(history)
        promotions = [r for r in recommendations if isinstance(r, ResumeRecommendation)]
        assert len(promotions) == 3, f"Expected 3 promotions, got {len(promotions)}"

        # Verify the TOP 3 configs are promoted (lowest loss)
        rung0_results = [(rec.id, rec.result) for rec in history if rec.status == JobStates.success]
        rung0_results.sort(key=lambda x: x[1])  # Sort by loss ascending
        top3_ids = set([rung0_results[i][0] for i in range(3)])
        promoted_ids = set([r.id for r in promotions])

        assert promoted_ids == top3_ids, \
            f"Should promote top 3 configs {top3_ids}, but got {promoted_ids}"
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
                    rec.update_result(0.999)
                    rec.update_status(JobStates.success)

        # Step 3: Get promotion to rung 2 (top 1 at 9 epochs)
        print("\n--- Step 3: Get Promotion to Rung 2 (top 1 @ 9 epochs) ---")
        recommendations = hyperband.generate_recommendations(history)
        promotions = [r for r in recommendations if isinstance(r, ResumeRecommendation)]
        assert len(promotions) == 1, f"Expected 1 promotion, got {len(promotions)}"

        # Verify the BEST config from rung 1 is promoted
        rung1_results = [(rec.id, rec.result) for rec in history
                         if rec.status == JobStates.success and rec.id in promoted_ids]
        rung1_results.sort(key=lambda x: x[1])  # Sort by loss ascending
        best_id = rung1_results[0][0]
        final_promoted_id = promotions[0].id

        assert final_promoted_id == best_id, \
            f"Should promote best config {best_id}, but got {final_promoted_id}"
        print(f"  ✓ Correctly promoted best: {final_promoted_id}")

        for rec_item in promotions:
            for rec in history:
                if rec.id == rec_item.id:
                    rec.update_status(JobStates.running)
                    print(f"  Promoted Config {rec.id} (loss={rec.result:.3f})")

        # Complete rung 2 job
        print("--- Completing Rung 2 ---")
        for rec in history:
            if rec.status == JobStates.running:
                if rec.id in predetermined_results and 9 in predetermined_results[rec.id]:
                    rec.update_result(predetermined_results[rec.id][9])
                    rec.update_status(JobStates.success)
                    print(f"  Config {rec.id}: val_loss={predetermined_results[rec.id][9]:.3f}")
                else:
                    rec.update_result(0.999)
                    rec.update_status(JobStates.success)

        # Verification
        print("\n" + "=" * 80)
        print("FINAL VERIFICATION")
        print("=" * 80)

        successful_configs = [rec for rec in history if rec.status == JobStates.success and rec.result < 0.9]
        assert len(successful_configs) > 0, "Expected successful configurations"

        print(f"✓ Total configurations tried: {len(history)}")
        print(f"✓ Successful configurations: {len(successful_configs)}")

        # Get best config (should be the one that completed rung 2)
        final_results = [(rec.id, rec.result) for rec in successful_configs]
        final_results.sort(key=lambda x: x[1])
        best_config_id, best_loss = final_results[0]

        # Verify best config is the one promoted to final rung
        assert best_config_id == final_promoted_id, \
            f"Best config should be {final_promoted_id}, got {best_config_id}"

        # Verify it completed the final rung (9 epochs)
        assert best_config_id in predetermined_results, "Best config should be in test data"
        assert 9 in predetermined_results[best_config_id], "Best config should have 9 - epoch result"
        expected_loss = predetermined_results[best_config_id][9]
        assert best_loss == expected_loss, (
            f"Best config result should be from 9 epochs, "
            f"expected {expected_loss}, got {best_loss}"
        )

        print("\n✓ Best configuration (completed 9 epochs):")
        print(f"    Config {best_config_id}: val_loss={best_loss:.3f}")
        print("✓ Successive halving worked correctly: 9 → 3 → 1")

        print("\n✅ Hyperband complete bracket flow test passed")

    def test_hyperband_bracket_progression(self, mock_job_context, tmp_path, test_environment, reset_test_db):
        """Test that Hyperband has multiple brackets configured"""

        save_job_specs(mock_job_context.id, {"train_config": {"num_epochs": 1}})

        hyperband = HyperBand(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=[],
            max_epochs=27,  # Use larger max to get more brackets
            reduction_factor=3,
            epoch_multiplier=1,
            metric="val_loss"
        )

        # Verify multiple brackets exist
        assert len(hyperband.ni) > 1, f"Should have multiple brackets, got {len(hyperband.ni)}"
        assert len(hyperband.ri) > 1, f"Should have multiple resource schedules, got {len(hyperband.ri)}"

        # Verify bracket structure
        print(f"✓ Number of brackets: {len(hyperband.ni)}")
        for bracket_id in hyperband.ni:
            print(f"  Bracket {bracket_id}: ni={hyperband.ni[bracket_id]}, ri={hyperband.ri[bracket_id]}")

        # Verify algorithm starts with bracket 0
        assert hyperband.bracket == "0", f"Should start with bracket 0, got {hyperband.bracket}"

        print("✅ Bracket progression test passed")
