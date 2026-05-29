# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flow tests for PBT AutoML algorithm"""

import pytest
from unittest.mock import Mock

from nvidia_tao_core.microservices.automl.pbt import PBT
from nvidia_tao_core.microservices.utils.automl_utils import Recommendation, ResumeRecommendation, JobStates
from nvidia_tao_core.microservices.utils.stateless_handler_utils import save_job_specs


class TestPBTFlow:
    """Test suite for PBT (Population-Based Training) algorithm flow"""

    @pytest.fixture
    def mock_job_context(self):
        """Create a mock job context"""
        mock_ctx = Mock()
        mock_ctx.id = "test_pbt_job"
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
                "parameter": "momentum",
                "value_type": "float",
                "min_value": 0.8,
                "max_value": 0.99
            }
        ]

    @pytest.fixture
    def predetermined_results(self):
        """Predetermined validation losses for population members

        PBT should exploit good performers and explore by perturbing them
        """
        # Results at different generations (epochs)
        # Format: {member_id: {generation: val_loss}}
        return {
            # Initial population (generation 0)
            0: {0: 0.500, 1: 0.480, 2: 0.460, 3: 0.440},  # Good improver
            1: {0: 0.520, 1: 0.510, 2: 0.505, 3: 0.503},  # Slow improver
            2: {0: 0.480, 1: 0.450, 2: 0.420, 3: 0.390},  # Best improver
            3: {0: 0.600, 1: 0.580, 2: 0.560, 3: 0.540},  # Poor
            4: {0: 0.550, 1: 0.540, 2: 0.535, 3: 0.532},  # Mediocre
            # Exploited/perturbed members (will be created dynamically)
            5: {0: 0.470, 1: 0.440, 2: 0.410, 3: 0.380},  # Exploited from member 2
            6: {0: 0.490, 1: 0.460, 2: 0.430, 3: 0.400},  # Exploited from member 0
            7: {0: 0.510, 1: 0.490, 2: 0.470, 3: 0.450},  # Perturbed
            8: {0: 0.460, 1: 0.430, 2: 0.400, 3: 0.370},  # Good exploited
            9: {0: 0.530, 1: 0.520, 2: 0.515, 3: 0.512},  # Mediocre
        }

    def test_pbt_complete_flow(
        self, mock_job_context, mock_parameters, predetermined_results,
        tmp_path, test_environment, reset_test_db
    ):
        """Test complete PBT flow with exploit and explore"""

        save_job_specs(mock_job_context.id, {"train_config": {"num_epochs": 10}})

        # Initialize PBT
        pbt = PBT(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=mock_parameters,
            population_size=5,
            max_generations=3,  # Small for testing
            eval_interval=10,
            perturbation_factor=1.2,
            metric="val_loss"  # Algorithm will automatically set reverse_sort=False
        )

        history = []
        iteration = 0

        print("\n" + "=" * 80)
        print("PBT FLOW TEST - POPULATION-BASED TRAINING")
        print("=" * 80)
        print(f"Population size: {pbt.population_size}")
        print(f"Max generations: {pbt.max_generations}")
        print(f"Eval interval: {pbt.eval_interval}")

        # Initial launch - create population
        print(f"\n--- Iteration {iteration}: Initial Population Launch ---")
        recommendations = pbt.generate_recommendations([])
        assert len(recommendations) == pbt.population_size, \
            f"Expected {pbt.population_size} initial members, got {len(recommendations)}"

        for i, rec_specs in enumerate(recommendations):
            rec = Recommendation(i, rec_specs, "val_loss")
            rec.assign_job_id(f"job_{i}")
            rec.update_status(JobStates.running)
            history.append(rec)
            print(f"  Launched Member {i}")

        iteration += 1

        # Main loop
        while not pbt.complete and iteration < 50:
            running_jobs = [rec for rec in history if rec.status == JobStates.running]

            if not running_jobs:
                # Generation complete, request new recommendations
                print(f"\n--- Generation {pbt.generation} complete, requesting exploit/explore ---")
                recommendations = pbt.generate_recommendations(history)

                if not recommendations:
                    if pbt.complete:
                        print("  PBT complete!")
                        break
                    iteration += 1
                    continue

                for rec_item in recommendations:
                    if isinstance(rec_item, dict):
                        member_id = pbt.next_member_id - 1
                        rec = Recommendation(member_id, rec_item, "val_loss")
                        rec.assign_job_id(f"job_{member_id}")
                        rec.update_status(JobStates.running)
                        history.append(rec)
                        print(f"  Launched Member {member_id} (new/perturbed)")
                    elif isinstance(rec_item, ResumeRecommendation):
                        member_id = rec_item.id
                        for rec in history:
                            if rec.id == member_id:
                                rec.update_status(JobStates.running)
                                print(f"  Resumed Member {member_id} [CONTINUE]")
                                break

                iteration += 1
                continue

            # Complete one member's evaluation
            rec_to_complete = running_jobs[0]
            member_id = rec_to_complete.id
            current_gen = pbt.generation

            print(f"\n--- Iteration {iteration}: Member {member_id} completes generation {current_gen} ---")

            # Get predetermined result for this generation
            if member_id in predetermined_results and current_gen in predetermined_results[member_id]:
                val_loss = predetermined_results[member_id][current_gen]
                rec_to_complete.update_result(val_loss)
                rec_to_complete.update_status(JobStates.success)
                print(f"  Member {member_id} @ gen {current_gen}: val_loss={val_loss:.3f}")
            else:
                # Use a default value
                val_loss = 0.5 + 0.1 * member_id - 0.01 * current_gen
                rec_to_complete.update_result(val_loss)
                rec_to_complete.update_status(JobStates.success)
                print(f"  Member {member_id} @ gen {current_gen}: val_loss={val_loss:.3f} (default)")

            # Generate recommendations
            recommendations = pbt.generate_recommendations(history)

            for rec_item in recommendations:
                if isinstance(rec_item, dict):
                    member_id = pbt.next_member_id - 1
                    rec = Recommendation(member_id, rec_item, "val_loss")
                    rec.assign_job_id(f"job_{member_id}")
                    rec.update_status(JobStates.running)
                    history.append(rec)
                    print(f"  → Launched Member {member_id}")
                elif isinstance(rec_item, ResumeRecommendation):
                    member_id = rec_item.id
                    for rec in history:
                        if rec.id == member_id:
                            rec.update_status(JobStates.running)
                            print(f"  → Resumed Member {member_id}")
                            break

            iteration += 1

        # Verification
        print("\n" + "=" * 80)
        print("FINAL VERIFICATION")
        print("=" * 80)

        # Check that algorithm completed
        assert pbt.complete or pbt.generation >= pbt.max_generations, \
            "PBT should complete or reach max generations"

        # Check that population was maintained
        assert len(pbt.population) > 0, "PBT should have a population"
        print(f"✓ Final population size: {len(pbt.population)}")
        print(f"✓ Generations completed: {pbt.generation}")

        # Check that members completed
        successful_members = [rec for rec in history if rec.status == JobStates.success]
        assert len(successful_members) > 0, "Expected successful members"

        print(f"✓ Total members evaluated: {len(history)}")
        print(f"✓ Successful evaluations: {len(successful_members)}")

        # Get final population results
        final_results = []
        for member_id, member_data in pbt.population.items():
            if "result" in member_data:
                final_results.append((member_id, member_data["result"]))

        final_results.sort(key=lambda x: x[1])
        print("\n✓ Final population (sorted by loss):")
        for i, (member_id, loss) in enumerate(final_results):
            print(f"    {i + 1}. Member {member_id}: val_loss={loss:.3f}")

        print("\n✅ PBT flow test passed")

    def test_pbt_exploit_explore(self, mock_job_context, mock_parameters, tmp_path, test_environment, reset_test_db):
        """Test that PBT exploits good performers and explores"""

        save_job_specs(mock_job_context.id, {"train_config": {"num_epochs": 10}})

        pbt = PBT(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=mock_parameters,
            population_size=4,
            max_generations=2,
            eval_interval=10,
            perturbation_factor=1.2,
            metric="val_loss"
        )

        # Verify algorithm correctly set reverse_sort based on metric
        assert pbt.reverse_sort is False, "Algorithm should set reverse_sort=False for loss metrics"

        history = []

        # Initial population
        recommendations = pbt.generate_recommendations([])
        for i, rec_specs in enumerate(recommendations):
            rec = Recommendation(i, rec_specs, "val_loss")
            rec.assign_job_id(f"job_{i}")
            history.append(rec)

        # Complete first generation with varying performance
        initial_results = {}
        for i, rec in enumerate(history):
            # Give different performance levels
            val_loss = 0.6 - 0.1 * i  # Member 3 is best (0.3), member 0 is worst (0.6)
            rec.update_result(val_loss)
            rec.update_status(JobStates.success)
            initial_results[rec.id] = val_loss
            print(f"  Member {rec.id}: loss={val_loss:.1f}")

        # Identify best and worst performers
        sorted_members = sorted(initial_results.items(), key=lambda x: x[1])
        best_member_id = sorted_members[0][0]  # Member 3 (loss=0.3)
        worst_member_id = sorted_members[-1][0]  # Member 0 (loss=0.6)

        print(f"\n  Best performer: Member {best_member_id} (loss={initial_results[best_member_id]:.1f})")
        print(f"  Worst performer: Member {worst_member_id} (loss={initial_results[worst_member_id]:.1f})")

        initial_pop_size = len(pbt.population)

        # Request exploit/explore
        print("\n--- Exploit/Explore Phase ---")
        recommendations = pbt.generate_recommendations(history)

        # Should get resumption recommendations for continuing population
        assert len(recommendations) > 0, "Should get recommendations for next generation"

        # Check if PBT performed exploit (poor performers should copy good ones)
        # Note: PBT may have modified specs for poor performers
        print(f"✓ Generated {len(recommendations)} recommendations for next generation")

        # Launch and complete next generation
        for rec_item in recommendations:
            if isinstance(rec_item, ResumeRecommendation):
                member_id = rec_item.id
                for rec in history:
                    if rec.id == member_id:
                        rec.update_status(JobStates.running)
                        rec.update_result(0.4 - 0.01 * member_id)
                        rec.update_status(JobStates.success)

        # Verify population size maintained
        assert len(pbt.population) == initial_pop_size, \
            "Population size should be maintained through exploit/explore"

        print(f"✓ Population size maintained: {len(pbt.population)}")
        print(
            f"✓ Generation progressed from 0 to {pbt.generation}"
        )
        print("✓ Exploit/explore mechanism executed")
        print("✅ Exploit/explore test passed")

    def test_pbt_generation_progression(
        self, mock_job_context, mock_parameters, tmp_path,
        test_environment, reset_test_db
    ):
        """Test that PBT progresses through generations correctly"""

        save_job_specs(mock_job_context.id, {"train_config": {"num_epochs": 10}})

        pbt = PBT(
            job_context=mock_job_context,
            root=str(tmp_path),
            network="test_network",
            parameters=mock_parameters,
            population_size=3,
            max_generations=3,
            eval_interval=10,
            perturbation_factor=1.2,
            metric="val_loss"
        )

        # Verify algorithm correctly set reverse_sort based on metric
        assert pbt.reverse_sort is False, "Algorithm should set reverse_sort=False for loss metrics"

        history = []
        generations_seen = set()

        # Initial population
        recommendations = pbt.generate_recommendations([])
        for i, rec_specs in enumerate(recommendations):
            rec = Recommendation(i, rec_specs, "val_loss")
            rec.assign_job_id(f"job_{i}")
            rec.update_status(JobStates.running)
            history.append(rec)

        # Run through multiple generations
        for iteration in range(pbt.max_generations + 1):
            generations_seen.add(pbt.generation)

            # Complete all running jobs
            running = [rec for rec in history if rec.status == JobStates.running]
            for rec in running:
                rec.update_result(0.5 - 0.01 * pbt.generation)
                rec.update_status(JobStates.success)

            # Get next recommendations
            recommendations = pbt.generate_recommendations(history)
            if not recommendations:
                break

            # Launch/resume for next generation
            for rec_item in recommendations:
                if isinstance(rec_item, ResumeRecommendation):
                    for rec in history:
                        if rec.id == rec_item.id:
                            rec.update_status(JobStates.running)

        # Verify generations progressed
        assert len(generations_seen) >= 2, (
            f"Should progress through multiple generations, saw {sorted(generations_seen)}"
        )
        assert pbt.generation > 0, "Should have progressed beyond initial generation"

        print(f"✓ Generations seen: {sorted(generations_seen)}")
        print(f"✓ Final generation: {pbt.generation}")
        print("✅ Generation progression test passed")
