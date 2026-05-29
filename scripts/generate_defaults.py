# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
from omegaconf import OmegaConf

from nvidia_tao_core.api_utils.dataclass2json_converter import import_module_from_path


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_ROOT = os.path.join(REPO_ROOT, "nvidia_tao_core/config")

SUPPORTED_MODULES = [
    item for item in os.listdir(CONFIG_ROOT) if item not in ["utils"] and os.path.isdir(os.path.join(CONFIG_ROOT, item))
]


def parse_command_line(args=sys.argv[1:]):
    """Parse command line args"""
    parser = argparse.ArgumentParser(
        prog="generate_default",
        description="Generate default spec from dataclasses."
    )
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        choices=SUPPORTED_MODULES,
        help="Name of the module to be checked.")
    parser.add_argument(
        "--output",
        type=str,
        default=os.getcwd(),
        help="Path to store the yaml file."
    )
    return parser.parse_args(args)


def dataclass_to_yaml(dataclass_obj, yaml_file_path):
    """
    Converts a dataclass object to a YAML file using omegaconf.

    Parameters:
        dataclass_obj (object): The dataclass object to convert.
        yaml_file_path (str): The path to the output YAML file.

    Returns:
        None
    """
    if not hasattr(dataclass_obj, "__dataclass_fields__"):
        raise ValueError("Provided object is not a dataclass instance.")

    # Convert dataclass to OmegaConf structured object
    conf = OmegaConf.structured(dataclass_obj)

    # Save as YAML
    if not os.path.exists(os.path.dirname(yaml_file_path)):
        os.makedirs(os.path.dirname(yaml_file_path), exist_ok=True)
    with open(yaml_file_path, 'w') as yaml_file:
        yaml_file.write(OmegaConf.to_yaml(conf))
        print(f"Dataclass object has been saved to {yaml_file_path}")


def main(cl_args=sys.argv[1:]):
    """Main function."""
    args = parse_command_line(cl_args)
    module = args.module
    output = args.output
    module_path = f"nvidia_tao_core.config.{module}.default_config"
    imported_module = import_module_from_path(module_path)
    dataclass_to_yaml(imported_module.ExperimentConfig, output)


if __name__=="__main__":
    main()