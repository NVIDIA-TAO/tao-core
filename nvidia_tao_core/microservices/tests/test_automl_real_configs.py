# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AutoML algorithms using real model configurations.

This module tests AutoML algorithms with actual parameter sets extracted from
dataclasses in nvidia_tao_core/config/.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
import dataclasses
import importlib
from pathlib import Path


def extract_automl_params_from_dataclass(dataclass_obj, prefix=""):
    """Recursively extract AutoML-enabled parameters from a dataclass.

    Args:
        dataclass_obj: A dataclass instance or class
        prefix: Prefix for nested parameters (e.g., "train.")

    Returns:
        List of parameter dictionaries suitable for AutoML
    """
    if not dataclasses.is_dataclass(dataclass_obj):
        return []

    params = []

    for field in dataclasses.fields(dataclass_obj):
        # First check if it's a nested dataclass
        if dataclasses.is_dataclass(field.type):
            nested_params = extract_automl_params_from_dataclass(
                field.type,
                prefix=f"{prefix}{field.name}."
            )
            params.extend(nested_params)

        if not hasattr(field, 'metadata') or not field.metadata:
            continue

        metadata = field.metadata
        automl_enabled = metadata.get('automl_enabled', '').upper()

        if automl_enabled == 'TRUE':
            value_type = metadata.get('value_type', 'float')
            param_dict = {
                'parameter': f"{prefix}{field.name}",
                'value_type': value_type,
                'default_value': metadata.get(
                    'default_value', field.default if field.default != dataclasses.MISSING else None
                )
            }

            # Add type-specific fields
            if 'valid_min' in metadata:
                valid_min = metadata['valid_min']
                param_dict['valid_min'] = (
                    0 if valid_min == "-inf" else (float('inf') if valid_min == "inf" else valid_min)
                )
            if 'valid_max' in metadata:
                valid_max = metadata['valid_max']
                param_dict['valid_max'] = float('inf') if valid_max == "inf" else valid_max
            if 'valid_options' in metadata:
                options = metadata['valid_options']
                # Handle both list and comma-separated string
                if isinstance(options, str):
                    param_dict['valid_options'] = [opt.strip() for opt in options.split(',')]
                else:
                    param_dict['valid_options'] = options

            # Include all parameters with proper metadata
            # The algorithms should handle all types properly
            params.append(param_dict)

    return params


def discover_model_automl_params():
    """Discover AutoML parameters from actual model config dataclasses.

    Returns:
        Dict mapping model_name to list of AutoML parameters
    """
    config_dir = Path(__file__).parent.parent.parent / "config"
    discovered_params = {}

    # Scan all directories in nvidia_tao_core/config/
    if not config_dir.exists():
        print(f"Warning: Config directory not found: {config_dir}")
        return discovered_params

    model_dirs = [d for d in config_dir.iterdir() if d.is_dir() and not d.name.startswith('_')]
    print(f"Scanning {len(model_dirs)} model directories for AutoML parameters...")

    for model_dir in sorted(model_dirs):
        model_name = model_dir.name

        try:
            # Try to import default_config
            try:
                config_module = importlib.import_module(f"nvidia_tao_core.config.{model_name}.default_config")
            except (ModuleNotFoundError, ImportError):
                continue

            # Look for ExperimentConfig or similar
            all_params = []
            for attr_name in dir(config_module):
                if "config" in attr_name.lower():
                    config_class = getattr(config_module, attr_name)
                    if dataclasses.is_dataclass(config_class):
                        params = extract_automl_params_from_dataclass(config_class)
                        all_params.extend(params)

            # Limit to first 5 params per model for faster testing
            if all_params:
                discovered_params[model_name] = all_params[:5]

        except Exception:
            # Silently skip models that can't be loaded
            continue

    return discovered_params


# Discover actual parameters from dataclasses
print("Discovering AutoML parameters from model configs...")
MODEL_PARAM_SETS = discover_model_automl_params()
print(f"Discovered parameters for {len(MODEL_PARAM_SETS)} models")

# Fallback if discovery fails
if not MODEL_PARAM_SETS:
    print("Warning: Using fallback parameter sets")
    MODEL_PARAM_SETS = {
        "classification_pyt": [
            {"parameter": "optim.lr", "value_type": "float", "valid_min": 0, "valid_max": 1, "default_value": 0.00006},
        ],
    }


def create_mock_job_context(job_id="test_job_001", handler_id="test_exp_001"):
    """Create a mock job context for testing."""
    job_context = Mock()
    job_context.id = job_id
    job_context.handler_id = handler_id
    return job_context


@pytest.fixture
def mock_dependencies():
    """Mock common dependencies for all 8 AutoML algorithms."""
    from contextlib import ExitStack

    # Define all patches - only mock what each algorithm actually uses
    patches = [
        # Base class mocks
        patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_job_specs', return_value={}),
        patch('nvidia_tao_core.microservices.automl.automl_algorithm_base.get_automl_custom_param_ranges',
              return_value={}),
        # Bayesian mocks
        patch('nvidia_tao_core.microservices.automl.bayesian.get_total_epochs', return_value=10),
        patch('nvidia_tao_core.microservices.automl.bayesian.get_flatten_specs', return_value=None),
        patch('nvidia_tao_core.microservices.automl.bayesian.save_automl_brain_info', return_value=None),
        patch('nvidia_tao_core.microservices.automl.bayesian.get_automl_brain_info', return_value={}),
        # BFBO mocks
        patch('nvidia_tao_core.microservices.automl.bfbo.get_total_epochs', return_value=10),
        patch('nvidia_tao_core.microservices.automl.bfbo.get_flatten_specs', return_value=None),
        patch('nvidia_tao_core.microservices.automl.bfbo.save_automl_brain_info', return_value=None),
        patch('nvidia_tao_core.microservices.automl.bfbo.get_automl_brain_info', return_value={}),
        # DEHB mocks
        patch('nvidia_tao_core.microservices.automl.dehb.get_flatten_specs', return_value=None),
        patch('nvidia_tao_core.microservices.automl.dehb.save_job_specs', return_value=None),
        patch('nvidia_tao_core.microservices.automl.dehb.get_job_specs', return_value={}),
        patch('nvidia_tao_core.microservices.automl.dehb.save_automl_brain_info', return_value=None),
        patch('nvidia_tao_core.microservices.automl.dehb.get_automl_brain_info', return_value={}),
        # BOHB mocks
        patch('nvidia_tao_core.microservices.automl.bohb.get_flatten_specs', return_value=None),
        patch('nvidia_tao_core.microservices.automl.bohb.save_job_specs', return_value=None),
        patch('nvidia_tao_core.microservices.automl.bohb.get_job_specs', return_value={}),
        patch('nvidia_tao_core.microservices.automl.bohb.save_automl_brain_info', return_value=None),
        patch('nvidia_tao_core.microservices.automl.bohb.get_automl_brain_info', return_value={}),
        # ASHA mocks (only what it uses)
        patch('nvidia_tao_core.microservices.automl.asha.get_flatten_specs', return_value=None),
        patch('nvidia_tao_core.microservices.automl.asha.save_job_specs', return_value=None),
        patch('nvidia_tao_core.microservices.automl.asha.get_job_specs', return_value={}),
        patch('nvidia_tao_core.microservices.automl.asha.save_automl_brain_info', return_value=None),
        patch('nvidia_tao_core.microservices.automl.asha.get_automl_brain_info', return_value={}),
        # Hyperband mocks (only what it uses)
        patch('nvidia_tao_core.microservices.automl.hyperband.get_flatten_specs', return_value=None),
        patch('nvidia_tao_core.microservices.automl.hyperband.save_job_specs', return_value=None),
        patch('nvidia_tao_core.microservices.automl.hyperband.get_job_specs', return_value={}),
        patch('nvidia_tao_core.microservices.automl.hyperband.save_automl_brain_info', return_value=None),
        patch('nvidia_tao_core.microservices.automl.hyperband.get_automl_brain_info', return_value={}),
        # HyperbandES mocks (only what it uses - it inherits from Hyperband)
        patch('nvidia_tao_core.microservices.automl.hyperband_es.get_automl_brain_info', return_value={}),
    ]

    # Apply all patches using ExitStack
    with ExitStack() as stack:
        for patch_obj in patches:
            stack.enter_context(patch_obj)
        yield


@pytest.mark.timeout(30)  # 30 second timeout per test
@pytest.mark.parametrize("model_name,parameters", list(MODEL_PARAM_SETS.items()))
def test_bayesian_with_real_model_params(mock_dependencies, model_name, parameters):
    """Test Bayesian algorithm generates diverse recommendations for real model parameters."""
    print(f"\n✓ Testing Bayesian AutoML with {model_name} parameters")
    print(f"  Parameters: {[p['parameter'] for p in parameters]}")

    from nvidia_tao_core.microservices.automl.bayesian import Bayesian

    # Generate recommendations from multiple instances
    recommendations = []
    for i in range(3):
        job_context = create_mock_job_context(f"{model_name}_bayesian_job_{i:03d}")
        print(f"  Generating recommendation {i + 1}...")
        algorithm = Bayesian(
            job_context=job_context,
            root="/test/root/subdir",
            network=model_name,
            parameters=parameters
        )
        recs = algorithm.generate_recommendations([])
        if recs:
            recommendations.append(recs[0])
            # Print condensed version
            print(f"    → {list(recs[0].values())}")

    assert len(recommendations) == 3, "Should generate 3 recommendations"

    # Verify diversity - convert to hashable format
    def make_hashable(obj):
        """Convert obj to a hashable type, handling lists recursively."""
        if isinstance(obj, list):
            return tuple(make_hashable(item) for item in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        else:
            return obj

    values_list = [make_hashable(rec) for rec in recommendations]
    unique_recommendations = len(set(values_list))

    print(f"\n  Generated {len(recommendations)} recommendations, {unique_recommendations} unique")
    assert unique_recommendations > 1, f"{model_name}: All recommendations identical - algorithm not exploring"
    print(f"  ✓ {model_name}: Bayesian generates diverse recommendations")

    # Verify parameters are within valid ranges
    for rec in recommendations:
        for param in parameters:
            param_name = param['parameter']
            assert param_name in rec, f"Missing parameter {param_name} in recommendation"

            value = rec[param_name]
            value_type = param['value_type']

            # Skip validation for params without complete ranges
            if 'valid_min' not in param or 'valid_max' not in param:
                continue

            valid_min = param['valid_min']
            valid_max = param['valid_max']

            # Skip if bounds are not numeric
            if not isinstance(valid_min, (int, float)) or not isinstance(valid_max, (int, float)):
                continue
            if valid_min == float('inf') or valid_max == float('inf'):
                continue

            if value_type == 'float':
                assert valid_min <= value <= valid_max, \
                    f"{param_name}={value} out of range [{valid_min}, {valid_max}]"
                assert isinstance(value, (float, np.floating)), f"{param_name} should be float"
            elif value_type == 'int':
                assert valid_min <= value <= valid_max, \
                    f"{param_name}={value} out of range [{valid_min}, {valid_max}]"
                assert isinstance(value, (int, np.integer)), f"{param_name} should be int"

    print("  ✓ All parameters within valid ranges and correct types")


@pytest.mark.timeout(30)  # 30 second timeout per test
@pytest.mark.parametrize("algo_name,algo_module,algo_class,extra_params,model_name", [
    # Test each algorithm with each model - all 8 algorithms now supported!
    *[
        (name, module, cls, params, model)
        for name, module, cls, params in [
            ("BFBO", "nvidia_tao_core.microservices.automl.bfbo", "BFBO", {}),
            ("DEHB", "nvidia_tao_core.microservices.automl.dehb", "DEHB",
             {"max_epochs": 10, "reduction_factor": 3, "epoch_multiplier": 1}),
            ("PBT", "nvidia_tao_core.microservices.automl.pbt", "PBT", {}),
            ("BOHB", "nvidia_tao_core.microservices.automl.bohb", "BOHB",
             {"max_epochs": 10, "reduction_factor": 3, "epoch_multiplier": 1}),
            ("ASHA", "nvidia_tao_core.microservices.automl.asha", "ASHA",
             {"max_epochs": 10, "reduction_factor": 3, "epoch_multiplier": 1, "max_concurrent": 4}),
            ("HyperBand", "nvidia_tao_core.microservices.automl.hyperband", "HyperBand",
             {"max_epochs": 10, "reduction_factor": 3, "epoch_multiplier": 1}),
            ("HyperBandES", "nvidia_tao_core.microservices.automl.hyperband_es", "HyperBandES",
             {"max_epochs": 10, "reduction_factor": 3, "epoch_multiplier": 1}),
        ]
        for model in list(MODEL_PARAM_SETS.keys())
    ],
])
def test_algorithms_generate_diverse_recommendations(
    mock_dependencies, algo_name, algo_module, algo_class, extra_params, model_name
):
    """Test each AutoML algorithm generates diverse recommendations with all models."""
    parameters = MODEL_PARAM_SETS[model_name]

    print(f"\n✓ Testing {algo_name} AutoML algorithm with {model_name}")
    params_preview = [p['parameter'] for p in parameters[:3]]
    ellipsis = '...' if len(parameters) > 3 else ''
    print(f"  Parameters ({len(parameters)}): {params_preview}{ellipsis}")

    # Import the algorithm class
    import importlib
    module = importlib.import_module(algo_module)
    AlgorithmClass = getattr(module, algo_class)

    # Generate recommendations
    recommendations = []
    for i in range(3):
        job_context = create_mock_job_context(f"{algo_name}_{model_name}_job_{i:03d}")

        # Create algorithm instance with base params + any extra params
        init_params = {
            "job_context": job_context,
            "root": "/test/root/subdir",
            "network": model_name,
            "parameters": parameters
        }
        init_params.update(extra_params)

        algorithm = AlgorithmClass(**init_params)
        recs = algorithm.generate_recommendations([])
        if recs:
            recommendations.append(recs[0])
            # Print condensed version
            rec_values = list(recs[0].values())[:3]
            print(f"  Rec {i + 1}: {rec_values}{'...' if len(recs[0]) > 3 else ''}")

    assert len(recommendations) >= 2, f"{algo_name} + {model_name}: Could not generate enough recommendations"

    # Verify diversity - convert to hashable format
    def make_hashable(obj):
        """Convert obj to a hashable type, handling lists recursively."""
        if isinstance(obj, list):
            return tuple(make_hashable(item) for item in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        else:
            return obj

    values_list = [make_hashable(rec) for rec in recommendations]
    unique_recommendations = len(set(values_list))

    print(f"  → {unique_recommendations}/{len(recommendations)} unique")
    assert unique_recommendations > 1, f"{algo_name} + {model_name}: All recommendations identical"
    print(f"  ✓ {algo_name} + {model_name}: Diverse recommendations")


@pytest.mark.timeout(60)  # 60 second timeout for this test
def test_recommendation_diversity_across_multiple_runs(mock_dependencies):
    """Test that recommendations explore the parameter space effectively."""
    print("\n✓ Testing parameter space exploration with multiple runs")

    from nvidia_tao_core.microservices.automl.bayesian import Bayesian

    # Use classification_pyt which has float parameters
    model_name = "classification_pyt"
    parameters = MODEL_PARAM_SETS[model_name]

    # Get the first 3 parameters
    param_names = [p['parameter'] for p in parameters[:3]]
    param_values = {name: [] for name in param_names}

    # Generate many recommendations to see distribution (reduced from 20 to 10 for speed)
    for i in range(10):
        job_context = create_mock_job_context(f"exploration_job_{i:03d}")
        algorithm = Bayesian(
            job_context=job_context,
            root="/test/root/subdir",
            network=model_name,
            parameters=parameters
        )
        recs = algorithm.generate_recommendations([])
        if recs and len(recs) > 0:
            rec = recs[0]
            for param_name in param_names:
                if param_name in rec:
                    param_values[param_name].append(rec[param_name])

    # Print distributions for each parameter
    for param_name in param_names:
        values = param_values[param_name]
        if values:
            print(f"\n  {param_name} distribution:")
            if isinstance(values[0], (bool, np.bool_)):
                true_count = sum(values)
                false_count = len(values) - true_count
                print(f"    True: {true_count}/{len(values)}, False: {false_count}/{len(values)}")
            elif isinstance(values[0], (int, np.integer)):
                print(f"    Min: {min(values)}, Max: {max(values)}")
                print(f"    Unique values: {len(set(values))}")
            else:  # float
                print(f"    Min: {min(values):.6f}, Max: {max(values):.6f}")
                print(f"    Mean: {np.mean(values):.6f}, Std: {np.std(values):.6f}")

    # Verify we got some diversity
    for param_name, values in param_values.items():
        if len(values) > 1:
            unique_count = len(set(map(str, values)))  # Convert to str for hashability
            assert unique_count > 1, f"{param_name}: No diversity in recommendations"

    print("\n  ✓ Algorithm explores parameter space effectively")


@pytest.mark.timeout(30)  # 30 second timeout
def test_deterministic_with_same_seed(mock_dependencies):
    """Test that same job ID produces same recommendations (deterministic)."""
    print("\n✓ Testing deterministic behavior with same seed")

    from nvidia_tao_core.microservices.automl.bayesian import Bayesian

    model_name = "optical_inspection"
    parameters = MODEL_PARAM_SETS[model_name]

    # Generate with same job ID (same seed)
    job_context_1 = create_mock_job_context("same_seed_job_123")
    algorithm_1 = Bayesian(
        job_context=job_context_1,
        root="/test/root/subdir",
        network=model_name,
        parameters=parameters
    )
    recommendations_1 = algorithm_1.generate_recommendations([])

    job_context_2 = create_mock_job_context("same_seed_job_123")  # Same ID
    algorithm_2 = Bayesian(
        job_context=job_context_2,
        root="/test/root/subdir",
        network=model_name,
        parameters=parameters
    )
    recommendations_2 = algorithm_2.generate_recommendations([])

    print(f"  Recommendation 1: {recommendations_1[0]}")
    print(f"  Recommendation 2: {recommendations_2[0]}")

    assert recommendations_1 == recommendations_2, "Same seed should produce same recommendations"
    print("  ✓ Deterministic behavior confirmed")


def test_model_param_sets_summary():
    """Print summary of model parameter sets being tested."""
    separator = "=" * 70
    print(f"\n{separator}")
    print("AutoML Model Parameter Sets Summary")
    print(separator)
    print(f"\nTesting {len(MODEL_PARAM_SETS)} models:")

    for model_name, parameters in MODEL_PARAM_SETS.items():
        print(f"\n  {model_name}:")
        print(f"    Parameters: {len(parameters)}")
        for p in parameters:
            print(f"      - {p['parameter']}: {p['value_type']} range=[{p['valid_min']}, {p['valid_max']}]")

    print(f"\n{separator}")
    assert len(MODEL_PARAM_SETS) >= 3, "Should test at least 3 models"
