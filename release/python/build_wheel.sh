#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

echo "Installing required packages"
pip install --upgrade pip setuptools
pip install pyinstaller pybind11

echo "Clearing build and dists"
python setup.py clean --all
echo "Clearing pycache and pycs"
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

echo "Building bdist wheel"
python setup.py bdist_wheel || exit $?
