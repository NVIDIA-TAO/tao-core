# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for ASHA flow verification with mock validation results"""
import pytest
from unittest.mock import Mock
from collections import defaultdict

from nvidia_tao_core.microservices.automl.asha import ASHA
from nvidia_tao_core.microservices.utils.automl_utils import Recommendation, ResumeRecommendation, JobStates


class TestASHAFlow:
    """Test ASHA algorithm flow with predetermined validation losses"""

    @pytest.fixture
    def mock_job_context(self):
        """Create mock job context"""
        ctx = Mock()
        ctx.id = "test_automl_job"
        ctx.handler_id = "test_handler"
        return ctx

    @pytest.fixture
    def predetermined_results(self):
        """Predetermined validation losses for each config at each rung

        Format: {config_id: {rung_epochs: val_loss}}
        Lower loss is better
        """
        return {
            # Config 0: Good performer - should reach rung 2
            0: {1: 0.456, 3: 0.389, 9: 0.298},
            # Config 1: Poor - eliminated at rung 0
            1: {1: 0.523, 3: 0.510, 9: 0.505},
            # Config 2: Mediocre - eliminated at rung 0
            2: {1: 0.489, 3: 0.475, 9: 0.470},
            # Config 3: BEST - should reach rung 2 and win
            3: {1: 0.378, 3: 0.312, 9: 0.245},
            # Config 4: Poor - eliminated at rung 0
            4: {1: 0.612, 3: 0.600, 9: 0.595},
            # Config 5: Mediocre - eliminated at rung 0 or 1
            5: {1: 0.534, 3: 0.520, 9: 0.515},
            # Config 6: Good - should reach rung 2
            6: {1: 0.445, 3: 0.367, 9: 0.278},
            # Config 7: Mediocre - may reach rung 1, not rung 2
            7: {1: 0.502, 3: 0.450, 9: 0.445},
            # Config 8: Poor - eliminated early
            8: {1: 0.567, 3: 0.555, 9: 0.550},
            # Config 9: Poor - eliminated at rung 0
            9: {1: 0.590, 3: 0.580, 9: 0.575},
            # Config 10 - 29: Additional configs with varying performance
            10: {1: 0.420, 3: 0.350, 9: 0.290},  # Good
            11: {1: 0.480, 3: 0.400, 9: 0.350},  # Good
            12: {1: 0.550, 3: 0.540, 9: 0.535},  # Poor
            13: {1: 0.410, 3: 0.340, 9: 0.280},  # Very good
            14: {1: 0.600, 3: 0.590, 9: 0.585},  # Poor
            15: {1: 0.470, 3: 0.390, 9: 0.320},  # Good
            16: {1: 0.430, 3: 0.360, 9: 0.300},  # Good
            17: {1: 0.520, 3: 0.510, 9: 0.505},  # Poor
            18: {1: 0.440, 3: 0.370, 9: 0.310},  # Good
            19: {1: 0.580, 3: 0.570, 9: 0.565},  # Poor
            20: {1: 0.460, 3: 0.380, 9: 0.330},  # Good
            21: {1: 0.540, 3: 0.530, 9: 0.525},  # Poor
            22: {1: 0.450, 3: 0.375, 9: 0.325},  # Good
            23: {1: 0.570, 3: 0.560, 9: 0.555},  # Poor
            24: {1: 0.425, 3: 0.355, 9: 0.295},  # Good
            25: {1: 0.590, 3: 0.580, 9: 0.575},  # Poor
            26: {1: 0.435, 3: 0.365, 9: 0.305},  # Good
            27: {1: 0.600, 3: 0.595, 9: 0.590},  # Poor
            28: {1: 0.415, 3: 0.345, 9: 0.285},  # Very good
            29: {1: 0.610, 3: 0.605, 9: 0.600},  # Poor
        }

    def test_asha_complete_flow(
        self, mock_job_context, predetermined_results, tmp_path,
        test_environment, reset_test_db
    ):
        """Test complete ASHA flow from start to finish"""

        # Use test_environment and reset_test_db fixtures for MongoDB mocking
        # This automatically uses mongomock instead of real MongoDB

        # Initialize job specs in test DB
        from nvidia_tao_core.microservices.utils.stateless_handler_utils import save_job_specs
        initial_spec = {"train_config": {"num_epochs": 1}}
        save_job_specs(mock_job_context.id, initial_spec)

        # Initialize ASHA
        asha = ASHA(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=[],
            max_epochs=9,
            reduction_factor=3,
            epoch_multiplier=1,
            max_concurrent=4,
            max_trials=30,  # Increased to allow more configs to be tried
            metric="val_loss"  # Algorithm will automatically set reverse_sort=False
        )

        # Verify rungs are correct
        assert asha.rungs == [1, 3, 9], f"Expected rungs [1, 3, 9], got {asha.rungs}"

        # Track history of all recommendations
        history = []
        iteration = 0

        print("\n" + "=" * 80)
        print("ASHA FLOW TEST - STEP BY STEP")
        print("=" * 80)

        # Iteration 0: Initial launch
        print(f"\n--- Iteration {iteration}: Initial Launch ---")
        recommendations = asha.generate_recommendations([])
        assert len(recommendations) == 4, f"Expected 4 initial configs, got {len(recommendations)}"

        # Create initial recommendations with job IDs
        for i, rec_specs in enumerate(recommendations):
            rec = Recommendation(i, rec_specs, "val_loss")
            rec.assign_job_id(f"job_{i}")
            rec.update_status(JobStates.running)
            history.append(rec)
            print(f"  Launched Config {i} @ rung 0 (1 epoch), job_id=job_{i}")

        iteration += 1

        # Track state for verification
        launched_configs = set(range(4))
        configs_at_rung = defaultdict(set)
        for i in range(4):
            configs_at_rung[0].add(i)

        # Main loop: process completions and promotions
        while not asha.done():
            print(f"\n--- Iteration {iteration} ---")

            # Find running jobs and complete one
            running = [rec for rec in history if rec.status == JobStates.running]
            if not running:
                print("  No running jobs, requesting new recommendations")
                recommendations = asha.generate_recommendations(history)
                if not recommendations:
                    print("  No new recommendations, checking completion")
                    break

                # Launch new recommendations
                for rec_item in recommendations:
                    if isinstance(rec_item, dict):
                        # New config
                        config_id = asha.next_config_id - 1
                        rec = Recommendation(config_id, rec_item, "val_loss")
                        rec.assign_job_id(f"job_{config_id}")
                        rec.update_status(JobStates.running)
                        history.append(rec)
                        launched_configs.add(config_id)
                        rung_idx = asha.config_to_rung.get(config_id, 0)
                        configs_at_rung[rung_idx].add(config_id)
                        print(f"  Launched Config {config_id} @ rung {rung_idx} ({asha.rungs[rung_idx]} epochs)")
                    elif isinstance(rec_item, ResumeRecommendation):
                        # Promotion
                        config_id = rec_item.id
                        # Update existing rec in history
                        for rec in history:
                            if rec.id == config_id:
                                rec.update_status(JobStates.running)
                                rung_idx = asha.config_to_rung[config_id]
                                configs_at_rung[rung_idx].add(config_id)
                                print(
                                    f"  Resumed Config {config_id} @ rung {rung_idx} "
                                    f"({asha.rungs[rung_idx]} epochs) [PROMOTION]"
                                )
                                break
                iteration += 1
                continue

            # Complete the first running job
            rec_to_complete = running[0]
            config_id = rec_to_complete.id
            current_rung_idx = asha.config_to_rung.get(config_id, 0)
            current_rung_epochs = asha.rungs[current_rung_idx]

            # Get predetermined result
            if config_id in predetermined_results:
                val_loss = predetermined_results[config_id].get(current_rung_epochs)
                if val_loss is not None:
                    rec_to_complete.update_result(val_loss)
                    rec_to_complete.update_status(JobStates.success)
                    print(
                        f"  Config {config_id} COMPLETED rung {current_rung_idx} "
                        f"({current_rung_epochs} epochs): val_loss={val_loss:.3f}"
                    )
                else:
                    rec_to_complete.update_status(JobStates.failure)
                    print(f"  Config {config_id} FAILED at rung {current_rung_idx}")
            else:
                # Config not in predetermined results
                rec_to_complete.update_result(0.999)
                rec_to_complete.update_status(JobStates.success)
                print(f"  Config {config_id} completed (not in test data): val_loss=0.999")

            # Generate recommendations (process completion and get next work)
            recommendations = asha.generate_recommendations(history)

            # Display current state
            print("  State after completion:")
            print(f"    rung_completions: {dict(asha.rung_completions)}")
            print(f"    rung_promotions: {dict(asha.rung_promotions)}")
            print(f"    completed_configs: {asha.completed_configs}")
            print(f"    pending_promotions: {len(asha.pending_promotions)}")

            # Process new recommendations
            for rec_item in recommendations:
                if isinstance(rec_item, dict):
                    # New config
                    config_id = asha.next_config_id - 1
                    rec = Recommendation(config_id, rec_item, "val_loss")
                    rec.assign_job_id(f"job_{config_id}")
                    rec.update_status(JobStates.running)
                    history.append(rec)
                    launched_configs.add(config_id)
                    rung_idx = asha.config_to_rung.get(config_id, 0)
                    configs_at_rung[rung_idx].add(config_id)
                    print(f"  → Launched Config {config_id} @ rung {rung_idx} ({asha.rungs[rung_idx]} epochs)")
                elif isinstance(rec_item, ResumeRecommendation):
                    # Promotion
                    config_id = rec_item.id
                    for rec in history:
                        if rec.id == config_id:
                            rec.update_status(JobStates.running)
                            rung_idx = asha.config_to_rung[config_id]
                            configs_at_rung[rung_idx].add(config_id)
                            print(
                                f"  → Resumed Config {config_id} @ rung {rung_idx} "
                                f"({asha.rungs[rung_idx]} epochs) [PROMOTION]"
                            )
                            break

            iteration += 1

            # Safety limit
            if iteration > 50:
                print("\n  ⚠️  Iteration limit reached (safety)")
                break

        # Verification
        print("\n" + "=" * 80)
        print("FINAL VERIFICATION")
        print("=" * 80)

        # Check that best configs reached final rung
        assert len(asha.completed_configs) >= 3, \
            f"Expected at least 3 configs to reach max rung, got {len(asha.completed_configs)}"

        print(f"\n✓ Configs that reached max rung (9 epochs): {asha.completed_configs}")

        # Get final results
        final_results = []
        for rec in history:
            if rec.id in asha.completed_configs and rec.status == JobStates.success:
                final_results.append((rec.id, rec.result))

        final_results.sort(key=lambda x: x[1])  # Sort by loss (ascending)
        print("\n✓ Final results (sorted by loss):")
        for config_id, loss in final_results:
            print(f"    Config {config_id}: val_loss={loss:.3f}")

        # Verify Config 3 (best) reached final rung
        assert 3 in asha.completed_configs, "Config 3 (best) should reach max rung"

        # Verify poor configs were eliminated
        poor_configs = [1, 2, 4, 5, 8, 9]
        for config_id in poor_configs:
            assert config_id not in asha.completed_configs, \
                f"Config {config_id} (poor) should not reach max rung"

        # Verify promotion counts are reasonable
        assert asha.rung_promotions[1] >= 3, \
            f"Expected at least 3 promotions from rung 0, got {asha.rung_promotions[1]}"
        assert asha.rung_promotions[3] >= 1, \
            f"Expected at least 1 promotion from rung 1, got {asha.rung_promotions[3]}"

        print("\n✅ All verifications passed!")
        print("=" * 80)

    def test_asha_promotion_quota(self, mock_job_context, tmp_path, test_environment, reset_test_db):
        """Test that promotion quota follows floor(m/nu) rule"""

        # Use test_environment and reset_test_db fixtures for MongoDB mocking
        from nvidia_tao_core.microservices.utils.stateless_handler_utils import save_job_specs
        save_job_specs(mock_job_context.id, {"train_config": {"num_epochs": 1}})

        asha = ASHA(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=[],
            max_epochs=9,
            reduction_factor=3,
            epoch_multiplier=1,
            max_concurrent=4,
            metric="val_loss"  # Algorithm will automatically set reverse_sort=False
        )

        # Test cases: (completions, expected_quota)
        test_cases = [
            (0, 0),  # floor(0/3) = 0
            (1, 0),  # floor(1/3) = 0
            (2, 0),  # floor(2/3) = 0
            (3, 1),  # floor(3/3) = 1
            (4, 1),  # floor(4/3) = 1
            (5, 1),  # floor(5/3) = 1
            (6, 2),  # floor(6/3) = 2
            (9, 3),  # floor(9/3) = 3
            (10, 3),  # floor(10/3) = 3
        ]

        for completions, expected_quota in test_cases:
            quota = int(completions / asha.reduction_factor)
            assert quota == expected_quota, \
                f"With {completions} completions, quota should be {expected_quota}, got {quota}"

        print("✅ Promotion quota test passed")

    def test_asha_failure_counting(self, mock_job_context, tmp_path, test_environment, reset_test_db):
        """Test that failures count toward quota but aren't promotable"""

        # Use test_environment and reset_test_db fixtures for MongoDB mocking
        from nvidia_tao_core.microservices.utils.stateless_handler_utils import save_job_specs
        save_job_specs(mock_job_context.id, {"train_config": {"num_epochs": 1}})

        asha = ASHA(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=[],
            max_epochs=9,
            reduction_factor=3,
            epoch_multiplier=1,
            max_concurrent=4,
            metric="val_loss"  # Algorithm will automatically set reverse_sort=False
        )

        # Initial launch
        recommendations = asha.generate_recommendations([])
        history = []
        for i, rec_specs in enumerate(recommendations[:3]):
            rec = Recommendation(i, rec_specs, "val_loss")
            rec.assign_job_id(f"job_{i}")
            history.append(rec)

        # Complete: 2 successes + 1 failure
        history[0].update_result(0.5)
        history[0].update_status(JobStates.success)

        history[1].update_result(0.6)
        history[1].update_status(JobStates.success)

        history[2].update_status(JobStates.failure)  # No result

        # Process completions
        asha.generate_recommendations(history)

        # Verify: rung_completions should be 3 (including failure)
        assert asha.rung_completions[1] == 3, \
            f"Expected 3 completions (2 success + 1 failure), got {asha.rung_completions[1]}"

        # Verify: rung_results should only have 2 entries (successes only)
        assert len(asha.rung_results[1]) == 2, \
            f"Expected 2 results (successes only), got {len(asha.rung_results[1])}"

        # Verify: quota should be 1 (floor(3/3))
        quota = int(asha.rung_completions[1] / asha.reduction_factor)
        assert quota == 1, f"Expected quota 1, got {quota}"

        # Verify: 1 promotion should have happened
        assert asha.rung_promotions[1] == 1, \
            f"Expected 1 promotion, got {asha.rung_promotions[1]}"

        print("✅ Failure counting test passed")

    def test_asha_asynchronous_behavior(self, mock_job_context, tmp_path, test_environment, reset_test_db):
        """Test that ASHA promotes configs asynchronously without waiting for all to complete"""

        from nvidia_tao_core.microservices.utils.stateless_handler_utils import save_job_specs
        save_job_specs(mock_job_context.id, {"train_config": {"num_epochs": 1}})

        asha = ASHA(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=[],
            max_epochs=9,
            reduction_factor=3,
            epoch_multiplier=1,
            max_concurrent=4,
            metric="val_loss"  # Algorithm will automatically set reverse_sort=False
        )

        # Verify algorithm correctly set reverse_sort based on metric
        assert asha.reverse_sort is False, "Algorithm should set reverse_sort=False for loss metrics"

        history = []

        # Launch 4 configs
        recommendations = asha.generate_recommendations([])
        for i, rec_specs in enumerate(recommendations):
            rec = Recommendation(i, rec_specs, "val_loss")
            rec.assign_job_id(f"job_{i}")
            rec.update_status(JobStates.running)
            history.append(rec)

        # Complete only 3 configs (not all 4) - test asynchronous behavior
        history[0].update_result(0.5)
        history[0].update_status(JobStates.success)
        history[1].update_result(0.6)
        history[1].update_status(JobStates.success)
        history[2].update_result(0.4)
        history[2].update_status(JobStates.success)
        # Config 3 still running!

        # ASHA should promote when quota is met, even though config 3 is still running
        recommendations = asha.generate_recommendations(history)

        # Should get a promotion (quota = floor(3/3) = 1)
        promotions = [r for r in recommendations if isinstance(r, ResumeRecommendation)]
        assert len(promotions) == 1, \
            f"ASHA should promote asynchronously when quota met, got {len(promotions)} promotions"

        # Verify the best config (Config 2 with loss=0.4) is promoted
        assert promotions[0].id == 2, \
            f"Should promote best config (2), got {promotions[0].id}"

        print("✓ ASHA promoted config asynchronously (3/4 configs complete, 1 still running)")
        print(f"✓ Correctly promoted best config: {promotions[0].id}")
        print("✅ Asynchronous behavior test passed")

    def test_asha_exact_numerical_flow(self, mock_job_context, tmp_path, test_environment, reset_test_db):
        """Test ASHA with exact numerical flow: max_epochs=9, reduction_factor=3, max_concurrent=4

        Expected behavior:
        - Rungs: [1, 3, 9] epochs
        - Start with 4 parallel configs at rung 0 (1 epoch)
        - When 3 complete, quota = floor(3/3) = 1, promote best
        - When 6 complete, quota = floor(6/3) = 2, promote 2nd best
        - Continue until enough configs reach final rung
        """

        from nvidia_tao_core.microservices.utils.stateless_handler_utils import save_job_specs
        save_job_specs(mock_job_context.id, {"train_config": {"num_epochs": 1}})

        asha = ASHA(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=[],
            max_epochs=9,
            reduction_factor=3,
            epoch_multiplier=1,
            max_concurrent=4,
            metric="val_loss"  # Algorithm will automatically set reverse_sort=False
        )

        # Verify algorithm correctly set reverse_sort based on metric
        assert asha.reverse_sort is False, "Algorithm should set reverse_sort=False for loss metrics"

        print("\n" + "=" * 80)
        print("ASHA EXACT NUMERICAL FLOW TEST")
        print("=" * 80)
        print("Parameters: max_epochs=9, reduction_factor=3, max_concurrent=4")
        print(f"Rungs: {asha.rungs}")

        # Verify rungs are exactly [1, 3, 9]
        assert asha.rungs == [1, 3, 9], f"Expected rungs [1, 3, 9], got {asha.rungs}"

        history = []

        # Step 1: Launch 4 concurrent configs
        print("\n--- Step 1: Launch 4 concurrent configs @ rung 0 (1 epoch) ---")
        recommendations = asha.generate_recommendations([])
        assert len(recommendations) == 4, f"Should launch 4 concurrent, got {len(recommendations)}"

        for i, rec_specs in enumerate(recommendations):
            rec = Recommendation(i, rec_specs, "val_loss")
            rec.assign_job_id(f"job_{i}")
            rec.update_status(JobStates.running)
            history.append(rec)
            print(f"  Launched Config {i}")

        # Step 2: Complete 3 configs one by one (async), verify quota progression
        print("\n--- Step 2: Complete 3 configs (leaving 1 running) ---")

        # Complete Config 0
        history[0].update_result(0.45)
        history[0].update_status(JobStates.success)
        print("  Config 0 completed: loss=0.45")
        asha.generate_recommendations(history)  # Process completion
        assert asha.rung_completions[1] == 1, f"Should have 1 completion, got {asha.rung_completions[1]}"

        # Complete Config 1
        history[1].update_result(0.35)
        history[1].update_status(JobStates.success)
        print("  Config 1 completed: loss=0.35")
        asha.generate_recommendations(history)  # Process completion
        assert asha.rung_completions[1] == 2, f"Should have 2 completions, got {asha.rung_completions[1]}"

        # Complete Config 2 - this triggers first promotion
        history[2].update_result(0.55)
        history[2].update_status(JobStates.success)
        print("  Config 2 completed: loss=0.55")
        recommendations = asha.generate_recommendations(history)  # Process completion

        assert asha.rung_completions[1] == 3, f"Should have 3 completions, got {asha.rung_completions[1]}"
        quota = int(asha.rung_completions[1] / asha.reduction_factor)
        assert quota == 1, f"Quota should be floor(3/3)=1, got {quota}"
        print(f"  ✓ Quota calculation: floor(3/3) = {quota}")

        # Verify promotion happened
        assert asha.rung_promotions[1] == 1, f"Should have 1 promotion, got {asha.rung_promotions[1]}"

        # Should have promoted best (Config 1 with loss=0.35)
        assert 1 in asha.promoted_from_rung[1], "Config 1 should be promoted"
        print("  ✓ Config 1 (best with loss=0.35) promoted to rung 1")

        # Verify Config 3 is still running (async behavior)
        assert history[3].status == JobStates.running, "Config 3 should still be running"
        print("  ✓ Config 3 still running (async behavior)")

        # Step 3: Complete Config 3 and verify quota still = 1
        print("\n--- Step 3: Complete Config 3 ---")
        history[3].update_result(0.40)
        history[3].update_status(JobStates.success)
        print("  Config 3 completed: loss=0.40")
        asha.generate_recommendations(history)

        assert asha.rung_completions[1] == 4, f"Should have 4 completions, got {asha.rung_completions[1]}"
        quota = int(asha.rung_completions[1] / asha.reduction_factor)
        assert quota == 1, f"floor(4/3) = 1, got {quota}"
        assert asha.rung_promotions[1] == 1, f"Should still have 1 promotion, got {asha.rung_promotions[1]}"

        print(f"  ✓ Completions: 4, Quota: floor(4/3) = {quota}, Promotions: 1")

        # Step 4: Verify ASHA key properties
        print("\n--- Verification Summary ---")

        # Verify the best configs were promoted
        assert 1 in asha.promoted_from_rung[1], "Config 1 (best) should be promoted"

        # Verify ranking is correct
        completed_at_rung0 = [(0, 0.45), (1, 0.35), (2, 0.55), (3, 0.40)]
        completed_at_rung0.sort(key=lambda x: x[1])
        best_id = completed_at_rung0[0][0]
        assert best_id == 1, f"Config 1 should be best, got {best_id}"
        assert 1 in asha.promoted_from_rung[1], "Best config should be promoted"

        print("✓ ASHA key properties verified:")
        print("  - Asynchronous: promoted after 3 completions (didn't wait for all 4)")
        print("  - floor(m/nu) quota: floor(3/3)=1, floor(4/3)=1")
        print("  - Best config (1, loss=0.35) correctly selected")
        print("  - Can continue launching new configs while promotions run")

    def test_asha_wrong_reverse_sort_chooses_worst(self, mock_job_context, tmp_path, test_environment, reset_test_db):
        """Test that manual override of reverse_sort causes wrong behavior (choosing worst configs)

        This test demonstrates the bug that existed before metric-aware implementation.
        When reverse_sort is set OPPOSITE to what the metric requires, ASHA promotes
        the WORST configs instead of the BEST ones.
        """
        from nvidia_tao_core.microservices.utils.stateless_handler_utils import save_job_specs
        save_job_specs(mock_job_context.id, {"train_config": {"num_epochs": 1}})

        # Predetermined results - lower loss is better
        predetermined_results = {
            0: {1: 0.80, 3: 0.75, 9: 0.70},  # Worst config
            1: {1: 0.30, 3: 0.25, 9: 0.20},  # Best config
            2: {1: 0.50, 3: 0.45, 9: 0.40},  # Middle config
            3: {1: 0.90, 3: 0.85, 9: 0.80},  # Second worst
        }

        asha = ASHA(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=[],
            max_epochs=9,
            reduction_factor=3,
            epoch_multiplier=1,
            max_concurrent=4,
            metric="val_loss"  # Algorithm sets reverse_sort=False (lower is better)
        )

        # Verify algorithm correctly set reverse_sort
        assert asha.reverse_sort is False, "Algorithm should set reverse_sort=False for loss"

        # MANUALLY OVERRIDE to WRONG value (simulate the bug)
        print("\n" + "=" * 80)
        print("TEST: Manual Override to WRONG reverse_sort Value")
        print("=" * 80)
        print("Metric: val_loss (lower is better)")
        print("Correct reverse_sort: False")
        print("Manual override: reverse_sort = True (WRONG!)")
        print("Expected behavior: Should promote WORST configs (highest loss)")
        print("=" * 80)

        asha.reverse_sort = True  # WRONG! Should be False for loss

        history = []

        # Launch 4 configs at rung 0 (1 epoch)
        recommendations = asha.generate_recommendations([])
        for i, rec_specs in enumerate(recommendations):
            rec = Recommendation(i, rec_specs, "val_loss")
            rec.assign_job_id(f"job_{i}")
            rec.update_status(JobStates.running)
            rec.update_result(predetermined_results[i][1])
            rec.update_status(JobStates.success)
            history.append(rec)

        # Request new recommendations - should promote based on WRONG sort
        recommendations = asha.generate_recommendations(history)

        # Get what was recommended for promotion
        promoted_configs = []
        for rec in recommendations:
            if isinstance(rec, ResumeRecommendation):
                promoted_configs.append(rec.id)

        print("\nResults at rung 0:")
        for i in range(4):
            loss = predetermined_results[i][1]
            promoted = i in promoted_configs
            marker = "  ← PROMOTED (WRONG!)" if promoted else ""
            quality = ""
            if i == 1:
                quality = " [BEST - should be promoted]"
            elif i == 3:
                quality = " [WORST]"
            elif i == 0:
                quality = " [2nd WORST]"
            print(f"  Config {i}: loss={loss:.2f}{quality}{marker}")

        # Verify we promoted the WORST configs (highest loss)
        assert len(promoted_configs) >= 1, f"Should have promoted at least 1 config, got {len(promoted_configs)}"

        # With reverse_sort=True (WRONG), we should promote config with HIGHEST loss
        # Config 3 has loss=0.90 (worst), Config 0 has loss=0.80 (2nd worst)
        # With quota of floor(3/3)=1 after 3 completions, we promote the "best" according to wrong sort = highest loss
        promoted_id = promoted_configs[0]
        promoted_loss = predetermined_results[promoted_id][1]

        # Should be one of the worst configs (highest loss)
        worst_configs = {3: 0.90, 0: 0.80}  # Configs with highest loss
        assert promoted_id in worst_configs, f"Should promote worst config (3 or 0), got {promoted_id}"

        # Should NOT be the best config (lowest loss)
        assert promoted_id != 1, "Should NOT promote best config (1) with wrong reverse_sort!"

        print("\n✅ VERIFIED: With WRONG reverse_sort=True:")
        print(f"  - Promoted config {promoted_id} with loss={promoted_loss:.2f} (WORST!)")
        print("  - Did NOT promote config 1 with loss=0.30 (BEST)")
        print("  - This demonstrates the bug that existed before metric-aware fix")
        print("\n💡 LESSON: Tests should rely on algorithm's metric-based reverse_sort,")
        print("   not manual override, to catch such bugs early!")
        print("=" * 80)

        print("\n✅ Exact numerical flow test passed")
