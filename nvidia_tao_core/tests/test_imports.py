#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest test to identify import errors in Python files

This test recursively scans all Python files and checks for import issues.

Usage:
    # Run the import test
    pytest test_imports.py -v

    # Run with print output
    pytest test_imports.py -v -s

    # Run all tests in the project
    pytest
"""

import sys
import os
import ast
import importlib.util
from pathlib import Path
from collections import defaultdict
import pytest

# Set environment variables needed for imports (mock values for local testing)
os.environ.setdefault('BACKEND', 'local-k8s')
os.environ.setdefault('TAO_ROOT', '/tmp/shared/orgs/')
os.environ.setdefault('NAMESPACE', 'default')

# MongoDB connection settings (mock for testing)
os.environ.setdefault('MONGO_HOST', 'mongodb://localhost:27017')
os.environ.setdefault('MONGO_USERNAME', 'test_user')
os.environ.setdefault('MONGO_PASSWORD', 'test_password')

# Suppress MongoDB connection attempts during import
# We need to patch pymongo BEFORE any imports happen
import unittest.mock  # noqa: E402

# Create a proper mock for pymongo and all its submodules
pymongo_mock = unittest.mock.MagicMock()
pymongo_mock.MongoClient = unittest.mock.MagicMock()
pymongo_mock.errors = unittest.mock.MagicMock()

# Patch all pymongo modules
sys.modules['pymongo'] = pymongo_mock
sys.modules['pymongo.errors'] = pymongo_mock.errors
sys.modules['pymongo.synchronous'] = unittest.mock.MagicMock()
sys.modules['pymongo.synchronous.mongo_client'] = unittest.mock.MagicMock()
sys.modules['pymongo.client_options'] = unittest.mock.MagicMock()
sys.modules['pymongo.auth_shared'] = unittest.mock.MagicMock()

# Root directory to scan - go up to the repository root
# From nvidia_tao_core/tests/test_imports.py -> tao-core/
ROOT_DIR = Path(__file__).parent.parent.parent

# Track results
files_with_import_errors = []
files_without_errors = []
import_error_details = defaultdict(list)


def find_python_files(root_dir):
    """Find all Python files in the project"""
    python_files = []
    exclude_dirs = {
        '__pycache__', '.git', '.pytest_cache', 'node_modules',
        'venv', 'env', '.venv', 'build', 'dist', '*.egg-info',
        'release', 'ci', 'telemetry_gateway', 'scripts'
    }
    exclude_files = {
        'test_imports.py', 'setup.py', 'conftest.py',
        'test_airgapped_loader.py',
    }

    for path in Path(root_dir).rglob('*.py'):
        # Skip excluded directories
        if any(excluded in path.parts for excluded in exclude_dirs):
            continue
        # Skip excluded files
        if path.name in exclude_files:
            continue
        python_files.append(path)

    return sorted(python_files)


def _get_try_except_line_ranges(tree):
    """Return a set of line numbers that fall inside try/except blocks.

    Imports guarded by try/except are intentionally optional and should not
    be flagged as errors when the module is missing.
    """
    guarded_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for child in node.body:
                for sub in ast.walk(child):
                    if hasattr(sub, 'lineno'):
                        guarded_lines.add(sub.lineno)
    return guarded_lines


def get_all_imports_in_file(file_path):
    """Extract all import statements from a Python file using AST"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))
        imports_list = []
        guarded_lines = _get_try_except_line_ranges(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if node.lineno in guarded_lines:
                    continue
                for alias in node.names:
                    imports_list.append({
                        'type': 'import',
                        'module': alias.name,
                        'line': node.lineno,
                        'level': 0
                    })
            elif isinstance(node, ast.ImportFrom):
                if node.lineno in guarded_lines:
                    continue
                # Construct full import path including relative level
                module_name = node.module or ''
                level = node.level  # Number of leading dots

                # Prepend dots for relative imports
                if level > 0:
                    module_name = '.' * level + module_name

                imported_names = [alias.name for alias in node.names]
                imports_list.append({
                    'type': 'from_import',
                    'module': module_name,
                    'names': imported_names,
                    'line': node.lineno,
                    'level': level
                })

        return imports_list, None

    except SyntaxError as e:
        return [], f"Syntax error: {e}"
    except Exception as e:
        return [], f"Error analyzing file: {e}"


def is_local_package_import(file_path, module_name):
    """Check if an import is from a local package (sibling/parent/ancestor directory)"""
    if not module_name or module_name.startswith('.'):
        return False

    # Get the directory containing the file
    file_dir = file_path.parent

    # Check if module exists as a sibling directory or file
    potential_module_dir = file_dir / module_name
    potential_module_file = file_dir / f"{module_name}.py"

    if potential_module_dir.exists() and potential_module_dir.is_dir():
        # Check if it's a package (has __init__.py)
        if (potential_module_dir / "__init__.py").exists():
            return True

    if potential_module_file.exists():
        return True

    # Check parent directory (for imports like 'utils' from handlers subdirectory)
    parent_dir = file_dir.parent
    potential_parent_module = parent_dir / module_name
    if potential_parent_module.exists() and potential_parent_module.is_dir():
        if (potential_parent_module / "__init__.py").exists():
            return True

    # Check grandparent directory (for imports like 'blueprints' from microservices subdirectories)
    grandparent_dir = parent_dir.parent
    potential_grandparent_module = grandparent_dir / module_name
    if potential_grandparent_module.exists() and potential_grandparent_module.is_dir():
        if (potential_grandparent_module / "__init__.py").exists():
            return True

    # Handle special cases for package-relative imports within nvidia_tao_core
    # e.g., "from nvidia_tao_core.microservices.job_utils import X"
    if 'nvidia_tao_core' in module_name:
        # This is an absolute import within the project - check if it resolves
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                return True
        except Exception:
            pass

    return False


def check_imports_exist(file_path, imports_list):
    """Check if imported modules/functions actually exist (fast, no execution)

    This function checks for structural import errors, focusing on:
    - Absolute imports from external packages
    - Cross-module imports that might be broken by refactoring

    It intentionally skips:
    - Relative imports (internal to packages, validated at runtime)
    - Local package imports (checked via filesystem)
    - Optional dependencies
    """
    errors = []

    # Optional public package dependencies (external packages not required for core functionality)
    optional_deps = [
        'hydra', 'clearml', 'wandb',
        'pytorch_lightning', 'tensorflow', 'mpi4py',
        'pycuda', 'pycuda.driver', 'torch',
        'diffusers', 'imageio', 'release', 'transformers'
    ]

    for imp in imports_list:
        if imp['type'] == 'import':
            # Simple import - check if module exists
            module_name = imp['module']

            # Skip optional dependencies
            if any(dep in module_name for dep in optional_deps):
                continue

            # Check if it's a local package import
            if is_local_package_import(file_path, module_name.split('.')[0]):
                continue

            try:
                # Just check if module can be found, don't execute
                spec = importlib.util.find_spec(module_name.split('.')[0])
                if spec is None:
                    errors.append({
                        'line': imp['line'],
                        'type': 'ModuleNotFound',
                        'message': f"Module '{module_name}' not found"
                    })
            except (ImportError, ModuleNotFoundError, ValueError):
                pass
            except Exception:
                # Ignore runtime errors like database connection failures
                pass

        elif imp['type'] == 'from_import':
            # from X import Y - check if Y exists in X
            module_name = imp['module']
            imported_names = imp['names']

            # Skip relative imports - they're internal to packages and validated at runtime
            if module_name and module_name.startswith('.'):
                continue

            # Skip optional dependencies
            if any(dep in module_name for dep in optional_deps):
                continue

            # Skip if module_name is None (can happen with certain import patterns)
            if not module_name:
                continue

            # Check if it's a local package import (like 'from utils import X')
            if is_local_package_import(file_path, module_name):
                continue

            try:
                # Try to find the module spec without executing
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    # Module doesn't exist
                    errors.append({
                        'line': imp['line'],
                        'type': 'ModuleNotFound',
                        'message': f"No module named '{module_name}'",
                        'importing': imported_names
                    })
            except (ImportError, ModuleNotFoundError, ValueError) as e:
                # Module doesn't exist or can't be found
                error_msg = str(e)
                if 'No module named' in error_msg:
                    errors.append({
                        'line': imp['line'],
                        'type': 'ImportError',
                        'message': error_msg,
                        'importing': imported_names
                    })
            except Exception:
                # Ignore runtime errors like database connection failures during find_spec
                pass

    return errors


def scan_and_test_files():
    """Scan all Python files and test imports (fast - no execution)"""
    print("=" * 100)
    print("TESTING IMPORT STRUCTURE IN ALL PYTHON FILES (Fast Mode)")
    print("=" * 100)

    python_files = find_python_files(ROOT_DIR)
    print(f"\nFound {len(python_files)} Python files to analyze\n")
    print("Checking import statements without executing code...\n")

    files_with_errors = []
    files_ok = []
    error_count = 0

    for idx, file_path in enumerate(python_files, 1):
        relative_path = file_path.relative_to(ROOT_DIR)

        # Get all imports from the file (fast - just AST parsing)
        imports_list, parse_error = get_all_imports_in_file(file_path)

        if parse_error:
            # Syntax error in the file
            error_count += 1
            print(f"❌ [{error_count}] {relative_path}")
            print(f"   Syntax Error: {parse_error}")
            files_with_errors.append({
                'path': relative_path,
                'error_type': 'SyntaxError',
                'error_msg': parse_error,
                'imports': []
            })
            print()
            continue

        # Check if imported modules exist (without executing them)
        import_errors = check_imports_exist(file_path, imports_list)

        if import_errors:
            error_count += 1
            print(f"❌ [{error_count}] {relative_path}")
            for err in import_errors:
                print(f"   Line {err['line']}: {err['message']}")
                if 'importing' in err:
                    print(f"      Trying to import: {', '.join(err['importing'])}")

            files_with_errors.append({
                'path': relative_path,
                'error_type': 'ImportError',
                'error_msg': '; '.join([e['message'] for e in import_errors]),
                'imports': imports_list,
                'errors': import_errors
            })
            print()
        else:
            files_ok.append(str(relative_path))

    print(f"\nProcessed {len(python_files)} files:")
    print(f"  ✓ {len(files_ok)} files have valid imports")
    print(f"  ❌ {len(files_with_errors)} files have import errors")

    return files_with_errors, files_ok


def test_all_imports():
    """Pytest: Test that all Python files have valid imports

    This test scans all Python files in the project and verifies that:
    - All absolute imports resolve to existing modules
    - No imports from non-existent modules
    - Relative imports are properly structured

    Excludes:
    - Optional external dependencies (tensorflow, pycuda, etc.)
    - Relative imports (validated at runtime)
    """
    print("\n" + "=" * 100)
    print("TESTING IMPORT STRUCTURE IN ALL PYTHON FILES")
    print("=" * 100)

    # Scan all files and test imports
    files_with_errors, files_ok = scan_and_test_files()

    # Build detailed error message if there are issues
    if files_with_errors:
        error_details = [f"\n❌ Found {len(files_with_errors)} file(s) with import errors:\n"]

        for idx, file_info in enumerate(files_with_errors, 1):
            error_details.append(f"\n{idx}. {file_info['path']}")
            error_details.append(f"   Error Type: {file_info['error_type']}")
            error_details.append(f"   Error: {file_info['error_msg'][:200]}")
            if len(file_info['error_msg']) > 200:
                error_details.append("          ...")

        error_details.append(f"\n\nPlease fix the import errors in the {len(files_with_errors)} file(s) listed above.")

        # Fail the test with detailed information
        pytest.fail("\n".join(error_details))

    print(f"\n✓ All {len(files_ok)} Python files have valid imports")
