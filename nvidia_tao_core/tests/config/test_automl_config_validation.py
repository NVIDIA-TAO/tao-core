# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests to validate automl parameter constraints across all model configs."""

import importlib
import pytest
from dataclasses import fields, is_dataclass
from pathlib import Path


def get_all_dataclass_fields(dataclass_obj, parent_name=""):
    """Recursively extract all fields from a dataclass and its nested dataclasses.

    Args:
        dataclass_obj: The dataclass instance to extract fields from.
        parent_name: The parent field name for nested fields (used for reporting).

    Returns:
        list: A list of tuples containing (full_field_name, field_object).
    """
    if not is_dataclass(dataclass_obj):
        return []

    all_fields = []
    for field_obj in fields(dataclass_obj):
        field_name = f"{parent_name}.{field_obj.name}" if parent_name else field_obj.name

        # Add the current field
        all_fields.append((field_name, field_obj))

        # If the field has a default value that is a dataclass, recurse into it
        if hasattr(field_obj, 'default') and is_dataclass(field_obj.default):
            nested_fields = get_all_dataclass_fields(field_obj.default, field_name)
            all_fields.extend(nested_fields)
        # Handle default_factory case
        elif hasattr(field_obj, 'default_factory') and field_obj.default_factory:
            try:
                default_instance = field_obj.default_factory()
                if is_dataclass(default_instance):
                    nested_fields = get_all_dataclass_fields(default_instance, field_name)
                    all_fields.extend(nested_fields)
            except (TypeError, ValueError, AttributeError):
                # Skip if default_factory cannot be called without arguments
                pass

    return all_fields


def get_automl_enabled_numeric_fields(dataclass_obj, model_name=""):
    """Get all automl-enabled float and int fields from a dataclass.

    Args:
        dataclass_obj: The dataclass instance to analyze.
        model_name: The name of the model (for reporting).

    Returns:
        list: A list of tuples containing (model_name, field_name, field_object, metadata)
              for automl-enabled numeric fields.
    """
    all_fields = get_all_dataclass_fields(dataclass_obj)
    automl_numeric_fields = []

    for field_name, field_obj in all_fields:
        metadata = field_obj.metadata
        value_type = metadata.get('value_type', '')
        automl_enabled = metadata.get('automl_enabled', '').upper()

        # Check if field is automl-enabled and is a numeric type (float or int)
        if automl_enabled == 'TRUE' and value_type in ['float', 'int', 'ordered_int']:
            automl_numeric_fields.append((model_name, field_name, field_obj, metadata))

    return automl_numeric_fields


def discover_config_modules():
    """Discover all model config modules with default_config.py.

    Returns:
        list: A list of tuples containing (model_name, module_path).
    """
    config_base_path = Path(__file__).parent.parent.parent / "config"
    config_modules = []

    # Iterate through all subdirectories in the config directory
    for item in config_base_path.iterdir():
        if not item.is_dir():
            continue

        # Skip special directories
        if item.name in ['__pycache__', 'utils', 'common']:
            continue

        # Check if default_config.py exists
        default_config_path = item / "default_config.py"
        if default_config_path.exists():
            model_name = item.name
            module_path = f"nvidia_tao_core.config.{model_name}.default_config"
            config_modules.append((model_name, module_path))

    return config_modules


def load_experiment_config_from_module(module_path):
    """Load the ExperimentConfig from a module.

    Args:
        module_path: The module path to import.

    Returns:
        The ExperimentConfig class instance or None if not found.
    """
    try:
        module = importlib.import_module(module_path)

        # Try to get ExperimentConfig
        if hasattr(module, 'ExperimentConfig'):
            return module.ExperimentConfig()

        # Some modules might have different naming conventions
        # Try to find any config class that looks like a root config
        for attr_name in dir(module):
            if 'Config' in attr_name and not attr_name.startswith('_'):
                attr = getattr(module, attr_name)
                if is_dataclass(attr):
                    try:
                        return attr()
                    except (TypeError, ValueError):
                        continue

    except (ImportError, AttributeError, TypeError, ValueError):
        # Skip modules that can't be imported or instantiated
        return None

    return None


@pytest.mark.config
@pytest.mark.automl
def test_all_models_automl_fields_have_valid_min_max():
    """Test that all automl-enabled float and int fields have valid_min and valid_max across all models."""
    # Discover all config modules
    config_modules = discover_config_modules()

    assert len(config_modules) > 0, "No config modules found"

    all_issues = []
    models_checked = 0
    models_with_automl = 0

    for model_name, module_path in config_modules:
        # Load the experiment config
        experiment_config = load_experiment_config_from_module(module_path)

        if experiment_config is None:
            continue

        models_checked += 1

        # Get all automl-enabled numeric fields
        automl_numeric_fields = get_automl_enabled_numeric_fields(experiment_config, model_name)

        if automl_numeric_fields:
            models_with_automl += 1

        # Check each field for valid_min and valid_max
        for model, field_name, field_obj, metadata in automl_numeric_fields:
            valid_min = metadata.get('valid_min', '')
            valid_max = metadata.get('valid_max', '')

            issues = []
            if valid_min in ('', None):
                issues.append("missing 'valid_min'")
            if valid_max in ('', None):
                issues.append("missing 'valid_max'")

            if issues:
                issue_msg = f"[{model}] Field '{field_name}' is {' and '.join(issues)}"
                all_issues.append(issue_msg)

    # Report summary
    print("\n=== Summary ===")
    print(f"Total models checked: {models_checked}")
    print(f"Models with automl-enabled fields: {models_with_automl}")
    print(f"Total violations found: {len(all_issues)}")

    # Assert that all automl-enabled numeric fields have valid_min and valid_max
    assert not all_issues, (
        f"\nFound {len(all_issues)} automl-enabled numeric field(s) "
        f"without proper valid_min/valid_max:\n\n" +
        "\n".join(all_issues)
    )


@pytest.mark.config
@pytest.mark.automl
def test_generate_automl_fields_report():
    """Generate a report of all automl-enabled fields across all models."""
    config_modules = discover_config_modules()

    all_automl_fields = []
    models_by_automl_count = {}

    for model_name, module_path in config_modules:
        experiment_config = load_experiment_config_from_module(module_path)

        if experiment_config is None:
            continue

        automl_numeric_fields = get_automl_enabled_numeric_fields(experiment_config, model_name)

        if automl_numeric_fields:
            models_by_automl_count[model_name] = len(automl_numeric_fields)

            for model, field_name, field_obj, metadata in automl_numeric_fields:
                value_type = metadata.get('value_type', '')
                valid_min = metadata.get('valid_min', '')
                valid_max = metadata.get('valid_max', '')
                has_valid_range = valid_min not in ('', None) and valid_max not in ('', None)

                all_automl_fields.append({
                    'model': model,
                    'field': field_name,
                    'type': value_type,
                    'valid_min': valid_min,
                    'valid_max': valid_max,
                    'has_valid_range': has_valid_range
                })

    # Print summary report
    print(f"\n{'=' * 80}")
    print("AutoML Fields Report")
    print(f"{'=' * 80}")
    print(f"\nTotal models with automl fields: {len(models_by_automl_count)}")
    print(f"Total automl-enabled numeric fields: {len(all_automl_fields)}")

    # Count fields with/without valid ranges
    with_valid_range = sum(1 for f in all_automl_fields if f['has_valid_range'])
    without_valid_range = len(all_automl_fields) - with_valid_range

    print(f"\nFields with valid_min/valid_max: {with_valid_range}")
    print(f"Fields WITHOUT valid_min/valid_max: {without_valid_range}")

    # Print models sorted by number of automl fields
    print(f"\n{'=' * 80}")
    print("Models by AutoML Field Count:")
    print(f"{'=' * 80}")
    for model, count in sorted(models_by_automl_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model:30s}: {count:3d} automl fields")

    # Print fields without valid ranges
    if without_valid_range > 0:
        print(f"\n{'=' * 80}")
        print("Fields Missing valid_min/valid_max:")
        print(f"{'=' * 80}")
        for field_info in all_automl_fields:
            if not field_info['has_valid_range']:
                print(f"  [{field_info['model']}] {field_info['field']}")

    # This test always passes - it's just for reporting
    assert True


@pytest.mark.config
@pytest.mark.automl
def test_config_module_discovery():
    """Test that config module discovery works correctly."""
    config_modules = discover_config_modules()

    # Should find at least some config modules
    assert len(config_modules) > 0, "No config modules discovered"

    # Check that segformer is in the list (we know it exists)
    model_names = [name for name, _ in config_modules]
    assert 'segformer' in model_names, "segformer should be discovered"

    print(f"\nDiscovered {len(config_modules)} config modules:")
    for model_name, module_path in sorted(config_modules):
        print(f"  - {model_name:30s} ({module_path})")


@pytest.mark.config
@pytest.mark.automl
@pytest.mark.parametrize("model_name,module_path", discover_config_modules())
def test_individual_model_automl_validation(model_name, module_path):
    """Test each model individually for automl field validation."""
    experiment_config = load_experiment_config_from_module(module_path)

    if experiment_config is None:
        pytest.skip(f"Could not load config for {model_name}")

    automl_numeric_fields = get_automl_enabled_numeric_fields(experiment_config, model_name)

    if not automl_numeric_fields:
        pytest.skip(f"No automl-enabled numeric fields in {model_name}")

    issues = []
    for model, field_name, field_obj, metadata in automl_numeric_fields:
        valid_min = metadata.get('valid_min', '')
        valid_max = metadata.get('valid_max', '')

        if valid_min in ('', None):
            issues.append(f"Field '{field_name}' is missing 'valid_min'")
        if valid_max in ('', None):
            issues.append(f"Field '{field_name}' is missing 'valid_max'")

    assert not issues, (
        f"\n[{model_name}] Found {len(issues)} field(s) without proper valid_min/valid_max:\n" +
        "\n".join(f"  - {issue}" for issue in issues)
    )
