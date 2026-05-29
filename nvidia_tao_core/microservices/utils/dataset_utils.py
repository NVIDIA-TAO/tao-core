# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset upload modules"""
import tarfile
import os
import glob
import logging

from .cloud_utils import create_cs_instance
from .stateless_handler_utils import get_handler_metadata
from .core_utils import read_network_config

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


# Simple helper class for ease of code migration


class SimpleHandler:
    """Helper class holding dataset information"""

    def __init__(self, org_name, handler_metadata, temp_dir="", workspace_metadata=None):
        """Initialize the Handler helper class"""
        self.root = temp_dir
        self.type = handler_metadata.get("type")
        self.format = handler_metadata.get("format")
        self.intent = handler_metadata.get("use_for", [])
        assert type(self.intent) is list, "Intent must be a list"
        self.cloud_instance = None
        self.cloud_file_path = handler_metadata.get("cloud_file_path", "")
        if workspace_metadata:
            self.cloud_instance, _ = create_cs_instance(workspace_metadata)

    def check_for_file_existence(self, path, file_type="file", file_extension=""):
        """Check for existence of file or extracted folder (for tar files)"""
        if self.cloud_instance:
            # Use cloud_file_path for cloud operations, not local temp path
            # For SLURM, keep absolute paths; for other clouds, strip leading /
            base_path = self.cloud_file_path
            if self.cloud_instance.cloud_type != "slurm":
                base_path = base_path.strip("/")

            if path in [".", ""]:
                cloud_path = base_path
            else:
                # Normalize the joining - remove trailing / from base, leading / from path
                cloud_path = f"{base_path.rstrip('/')}/{path.lstrip('/')}"

            if file_type == "file":
                file_exists = self.cloud_instance.is_file(cloud_path)
                # If tar file not found, check for extracted folder
                if not file_exists and (cloud_path.endswith('.tar.gz') or cloud_path.endswith('.tar')):
                    extracted_dir = cloud_path.replace('.tar.gz', '').replace('.tar', '')
                    folder_exists = self.cloud_instance.is_folder(extracted_dir)
                    if folder_exists:
                        logger.debug(f"Tar file {cloud_path} not found, but extracted folder exists: {extracted_dir}")
                        return True
                return file_exists
            if file_type == "folder":
                return self.cloud_instance.is_folder(cloud_path)
            if file_type == "regex":
                # file_extension contains the full regex pattern, not just extension
                if path in [".", ""]:
                    pattern = f"{base_path.rstrip('/')}/{file_extension}"
                else:
                    pattern = f"{cloud_path}/{file_extension}"
                return any(self.cloud_instance.glob_files(pattern))
        else:
            if file_type == "file":
                file_exists = os.path.isfile(path)
                # If tar file not found, check for extracted folder
                if not file_exists and (path.endswith('.tar.gz') or path.endswith('.tar')):
                    extracted_dir = path.replace('.tar.gz', '').replace('.tar', '')
                    folder_exists = os.path.isdir(extracted_dir)
                    if folder_exists:
                        logger.debug(f"Tar file {path} not found, but extracted folder exists: {extracted_dir}")
                        return True
                return file_exists
            if file_type == "folder":
                return os.path.isdir(path)
            if file_type == "regex":
                # file_extension contains the full regex pattern, not just extension
                if path in [".", ""]:
                    pattern = f"{self.cloud_file_path.strip('/')}/{file_extension}"
                else:
                    pattern = os.path.join(path, file_extension)
                return bool(glob.glob(pattern))
        return False


def _untar_file(tar_path, dest, strip_components=0):
    """Function to untar a file"""
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            # Remove leading directory components using strip_components
            components = member.name.split(os.sep)
            if len(components) > strip_components:
                member.name = os.path.join(*components[strip_components:])
            if member.isdir():
                # Make subdirs ahead because tarfile extracts them with user permissions only
                os.makedirs(os.path.join(dest, member.name), exist_ok=True)
            tar.extract(member, path=dest, set_attrs=False)


def _extract_images(tar_path, dest):
    """Function to extract images, other directories on same level as images to root of dataset"""
    # Infer how many components to strip to get images,labels to top of dataset directory
    # Assumes: images, other necessary directories are in the same level
    with tarfile.open(tar_path) as tar:
        strip_components = 0
        names = [tinfo.name for tinfo in tar.getmembers()]
        for name in names:
            if "/images/" in name:
                strip_components = name.split("/").index("images")
                break
    # Build shell command for untarring
    logger.info("Untarring data started")
    _untar_file(tar_path, dest, strip_components)
    logger.info("Untarring data complete")

    # Remove .tar.gz file
    logger.info("Removing data tar file")
    os.remove(tar_path)
    logger.info("Deleted data tar file")


def _get_actual_files_in_dataset(handler):
    """Get list of actual files present in the dataset"""
    files = []
    try:
        if handler.cloud_instance:
            # For cloud storage (including SLURM via SSH), list files in the cloud_file_path
            # For SLURM, keep absolute paths (starts with /); for other clouds, strip leading /
            folder_path = handler.cloud_file_path
            if folder_path:
                # For non-SLURM cloud types, strip leading slash as paths are relative to bucket
                if handler.cloud_instance.cloud_type != "slurm":
                    folder_path = folder_path.strip("/")

                cloud_files, _ = handler.cloud_instance.list_files_in_folder(folder_path)
                files = cloud_files
            else:
                files = []
            logger.debug(f"Listed {len(files)} files from {handler.cloud_instance.cloud_type} storage at {folder_path}")
        else:
            # For local storage only, use glob to find files
            if handler.root and os.path.exists(handler.root):
                files = [
                    os.path.relpath(f, handler.root)
                    for f in glob.glob(os.path.join(handler.root, "**", "*"), recursive=True)
                    if os.path.isfile(f)
                ]
                logger.debug(f"Listed {len(files)} files from local storage")
            else:
                logger.warning(f"Local root path does not exist: {handler.root}")
    except Exception as e:
        logger.warning("Could not list files in dataset: %s", str(e))
        files = []

    return sorted(files)


def write_dir_contents(directory, file):
    """Write contents of a directory to a file"""
    with open(file, "w", encoding='utf-8') as f:
        for dir_files in sorted(glob.glob(directory + "/*")):
            f.write(dir_files + "\n")


def _get_format_requirements(validation_config, dataset_format):
    """Get format-specific requirements with support for multiple formats and wildcards

    Args:
        validation_config (dict): The dataset validation configuration
        dataset_format (str): The dataset format (can be single format, comma-separated, or wildcard)
    Returns:
        list: List of validation requirements
    """
    required_files_config = validation_config.get("required_files", {})
    # Handle wildcard case - return requirements from "*" key or collect from all formats
    if dataset_format == "*":
        # First try to get requirements from the "*" key directly
        wildcard_reqs = required_files_config.get("*", [])
        if wildcard_reqs:
            return wildcard_reqs
        # If no "*" key, collect requirements from all other format keys
        all_requirements = []
        for format_key, requirements in required_files_config.items():
            if format_key not in ["default", "*"]:
                all_requirements.extend(requirements)
        # If no specific format requirements found, use default
        if not all_requirements:
            all_requirements = required_files_config.get("default", [])
        return all_requirements
    # Handle comma-separated formats
    if "," in dataset_format:
        formats = [fmt.strip() for fmt in dataset_format.split(",")]
        combined_requirements = []
        for fmt in formats:
            # Try to get requirements for this specific format
            fmt_reqs = required_files_config.get(fmt, [])
            if fmt_reqs:
                combined_requirements.extend(fmt_reqs)
            else:
                # If no specific requirements for this format, use default or wildcard
                fallback_reqs = required_files_config.get("default", required_files_config.get("*", []))
                combined_requirements.extend(fallback_reqs)
        return combined_requirements
    # Handle single format (original behavior)
    return required_files_config.get(
        dataset_format,
        required_files_config.get("default", required_files_config.get("*", []))
    )


def validate_dataset(org_name, handler_metadata, temp_dir="", workspace_metadata=None):
    """Generic dataset validator using config

    Args:
        org_name (str): Organization name
        handler_metadata (dict): Dataset metadata containing type, format, cloud_file_path, etc.
        temp_dir (str): Local temp directory path (empty for cloud-only validation)
        workspace_metadata (dict): Workspace metadata for cloud storage access

    Returns:
        tuple: (is_valid, validation_result)
            - is_valid (bool): True if validation passes, False otherwise
            - validation_result (dict): Detailed validation information including:
                - success (bool): Same as is_valid
                - expected_structure (dict): What the service expects
                - actual_structure (list): What files were found
                - missing_files (list): Files that are required but missing
                - error_details (str): Human-readable error description
    """
    # For cloud-based datasets, workspace_metadata may need to be resolved from handler_metadata
    if not workspace_metadata and handler_metadata.get("workspace"):
        workspace_metadata = get_handler_metadata(handler_metadata.get("workspace"), kind="workspace")

    handler = SimpleHandler(org_name, handler_metadata, temp_dir=temp_dir, workspace_metadata=workspace_metadata)

    # Initialize detailed validation result
    validation_result = {
        "success": False,
        "expected_structure": {},
        "actual_structure": [],
        "missing_files": [],
        "error_details": "",
        "network_type": handler.type,
        "dataset_format": handler.format,
        "dataset_intent": handler.intent
    }

    try:
        # Load network config
        logger.debug("handler.type: %s", handler.type)
        network_config = read_network_config(handler.type)
        logger.debug("network_config: %s", network_config)
        validation_config = network_config.get("dataset_validation", {})

        # Get format-specific requirements with support for multiple formats and wildcards
        format_reqs = _get_format_requirements(validation_config, handler.format)

        # Store expected structure
        validation_result["expected_structure"] = {
            "format": handler.format,
            "requirements": format_reqs
        }

        # Validate each requirement and collect detailed errors
        validation_errors = []
        missing_files = []

        for req in format_reqs:
            if "path" in req:
                # For cloud validation, use the relative path directly instead of joining with temp_dir
                if handler.cloud_instance:
                    path = req["path"]
                else:
                    path = os.path.join(handler.root, req["path"])
                file_type = req.get("type", "file")
                file_extension = req.get("regex", "") if file_type == "regex" else ""

                if not handler.check_for_file_existence(
                    path, file_type=file_type, file_extension=file_extension
                ):
                    missing_files.append({
                        "path": req["path"],
                        "type": file_type,
                        "regex": file_extension if file_type == "regex" else None
                    })
                    validation_errors.append(f"Required file not found: {req['path']} (type: {file_type})")
            elif "all_of" in req:
                # Check if all requirements are met
                all_of_missing = []
                for subreq in req["all_of"]:
                    # For cloud validation, use the relative path directly instead of joining with temp_dir
                    if handler.cloud_instance:
                        path = subreq["path"]
                    else:
                        path = os.path.join(handler.root, subreq["path"])
                    file_type = subreq.get("type", "file")
                    file_extension = subreq.get("regex", "") if file_type == "regex" else ""

                    if not handler.check_for_file_existence(
                        path, file_type=file_type, file_extension=file_extension
                    ):
                        all_of_missing.append({
                            "path": subreq["path"],
                            "type": file_type,
                            "regex": file_extension if file_type == "regex" else None
                        })

                if all_of_missing:
                    missing_files.extend(all_of_missing)
                    paths_str = ", ".join([f["path"] for f in all_of_missing])
                    validation_errors.append(f"Missing required files (all required): {paths_str}")
            elif "any_of" in req:
                # Check if any of the requirements are met
                any_valid = False
                any_of_options = []

                for subreq in req["any_of"]:
                    if "path" in subreq:
                        # For cloud validation, use the relative path directly instead of joining with temp_dir
                        if handler.cloud_instance:
                            path = subreq["path"]
                        else:
                            path = os.path.join(handler.root, subreq["path"])
                        file_type = subreq.get("type", "file")
                        file_extension = subreq.get("regex", "") if file_type == "regex" else ""
                        any_of_options.append(subreq["path"])
                        if handler.check_for_file_existence(
                            path, file_type=file_type, file_extension=file_extension
                        ):
                            any_valid = True
                            break
                    elif "all_of" in subreq:
                        # Check if all sub-requirements are met
                        all_valid = True
                        subreq_paths = []
                        # First collect all paths for error message
                        for subsubreq in subreq["all_of"]:
                            subreq_paths.append(subsubreq["path"])
                        # Then check if all files exist
                        for subsubreq in subreq["all_of"]:
                            # For cloud validation, use the relative path directly instead of joining with temp_dir
                            if handler.cloud_instance:
                                path = subsubreq["path"]
                            else:
                                path = os.path.join(handler.root, subsubreq["path"])
                            file_type = subsubreq.get("type", "file")
                            file_extension = subsubreq.get("regex", "") if file_type == "regex" else ""
                            if not handler.check_for_file_existence(
                                path, file_type=file_type, file_extension=file_extension
                            ):
                                all_valid = False
                                break
                        any_of_options.append(f"({', '.join(subreq_paths)})")
                        if all_valid:
                            any_valid = True
                            break

                if not any_valid:
                    # For any_of validation, show the complete requirement options, not just missing files
                    options_str = " OR ".join(any_of_options)
                    validation_errors.append(f"Dataset must contain one of the following combinations: {options_str}")
            elif "intent_based_path" in req:
                # Check if intent exists
                if not handler.intent:
                    validation_errors.append("Intent is required for this dataset")
                    continue
                if len(handler.intent) != 1:
                    validation_errors.append("Only one intent is allowed")
                    continue

                intent = handler.intent[0]
                # Get path requirement for this intent
                intent_req = req["intent_based_path"].get(intent)
                if not intent_req:
                    validation_errors.append(f"No path requirement found for intent: {intent}")
                    continue

                # For cloud validation, use the relative path directly instead of joining with temp_dir
                if handler.cloud_instance:
                    path = intent_req["path"]
                else:
                    path = os.path.join(handler.root, intent_req["path"])
                file_type = intent_req.get("type", "file")
                file_extension = intent_req.get("regex", "") if file_type == "regex" else ""

                if not handler.check_for_file_existence(
                    path, file_type=file_type, file_extension=file_extension
                ):
                    missing_files.append({
                        "path": intent_req["path"],
                        "type": file_type,
                        "regex": file_extension if file_type == "regex" else None,
                        "intent": intent
                    })
                    validation_errors.append(f"Required file for intent '{intent}' not found: {intent_req['path']}")
            if "intent_restriction" in req:
                if handler.intent:
                    if handler.intent != req["intent_restriction"]:
                        validation_errors.append(
                            f"Intent mismatch: handler intent {handler.intent} does not match "
                            f"required intent {req['intent_restriction']}"
                        )

        # Compile validation results
        validation_result["missing_files"] = missing_files
        if validation_errors:
            validation_result["actual_structure"] = _get_actual_files_in_dataset(handler)
            validation_result["error_details"] = "; ".join(validation_errors)
            validation_result["success"] = False
            logger.error("Dataset validation failed: %s", validation_result["error_details"])
            return False, validation_result
        validation_result["success"] = True
        validation_result["error_details"] = "Dataset structure is valid"
        return True, validation_result

    except Exception as e:
        logger.error("Error occurred during validation: %s", str(e))
        validation_result["error_details"] = f"Validation error: {str(e)}"
        validation_result["success"] = False
        return False, validation_result
