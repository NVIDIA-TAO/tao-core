# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime dataset validation for job creation"""
import logging
import uuid
from nvidia_tao_core.microservices.utils.dataset_utils import validate_dataset
from nvidia_tao_core.microservices.utils.core_utils import read_network_config
from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_handler_metadata

logger = logging.getLogger(__name__)


def validate_dataset_uri_structure(
    dataset_uri: str,
    network_arch: str,
    dataset_format: str,
    dataset_type: str = None,
    dataset_intent: list = None,
    workspace_id: str = None,
    skip_validation: bool = False
) -> tuple:
    """Validate dataset structure at job creation time.

    This function reuses the existing dataset validation logic that was previously
    used during dataset object creation. It checks if required files exist based
    on network config validation rules.

    Args:
        dataset_uri: Full dataset URI (e.g., "aws://bucket/data", "lustre:///data/train")
        network_arch: Network architecture name (e.g., "resnet", "cosmos-rl")
        dataset_format: Dataset format (e.g., "coco", "kitti", "llava")
        dataset_type: Dataset type (e.g., "object_detection", "vlm")
        dataset_intent: List of intents (e.g., ["training"], ["evaluation"])
        workspace_id: Workspace ID for cloud credentials (required for cloud paths)
        skip_validation: If True, skip validation and return success

    Returns:
        tuple: (is_valid: bool, validation_details: dict)

    Example validation_details on failure:
        {
            "error": "Dataset validation failed",
            "error_details": "Missing required files",
            "expected_structure": {...},
            "actual_structure": [...],
            "missing_files": ["annotations.json"],
            "network_arch": "resnet",
            "dataset_format": "coco",
            "dataset_intent": ["training"]
        }
    """
    if skip_validation:
        logger.info(f"Skipping dataset validation for {dataset_uri} (skip_validation=True)")
        return True, {"message": "Validation skipped"}

    logger.info(
        f"Starting dataset validation for path: {dataset_uri}, "
        f"network: {network_arch}, format: {dataset_format}"
    )

    # Detect if path is cloud or local (do this early as it's needed in multiple places)
    # Cloud paths have protocols: aws://, azure://, lepton://, lustre://, slurm://
    is_cloud_path = any(dataset_uri.startswith(proto) for proto in
                        ["aws://", "azure://", "lepton://", "lustre://", "slurm://", "seaweedfs://"])

    # Extract the path without protocol prefix for cloud_file_path
    # For AWS/Azure, workspace contains bucket config, so cloud_file_path should be path within bucket
    # Format: aws://bucket-name/path/to/data -> /path/to/data
    # For SLURM/Lustre, keep the full path
    cloud_file_path_clean = dataset_uri
    if is_cloud_path:
        for proto in ["aws://", "azure://", "lepton://", "lustre://", "slurm://", "seaweedfs://"]:
            if dataset_uri.startswith(proto):
                path_after_proto = dataset_uri[len(proto):]

                if proto in ["aws://", "azure://", "lepton://", "seaweedfs://"]:
                    # Split into bucket and path: "bucket-name/path/to/data" -> "/path/to/data"
                    parts = path_after_proto.split('/', 1)
                    if len(parts) > 1:
                        cloud_file_path_clean = '/' + parts[1]  # Path within bucket
                    else:
                        cloud_file_path_clean = '/'  # Root of bucket
                else:
                    # For SLURM/Lustre, keep full path
                    cloud_file_path_clean = path_after_proto
                    if not cloud_file_path_clean.startswith('/'):
                        cloud_file_path_clean = '/' + cloud_file_path_clean
                break
        logger.info(f"Parsed cloud path: {dataset_uri} -> cloud_file_path={cloud_file_path_clean}")

    # Get workspace metadata for cloud credentials
    workspace_metadata = None
    if is_cloud_path:
        if not workspace_id:
            return False, {
                "error": "Workspace ID is required for cloud dataset URIs",
                "error_details": f"Dataset URI '{dataset_uri}' is a cloud path but no workspace_id was provided",
                "dataset_uri": dataset_uri
            }
        workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
        if not workspace_metadata:
            return False, {
                "error": "Workspace not found or access denied",
                "error_details": f"Could not retrieve workspace metadata for workspace_id '{workspace_id}'",
                "workspace_id": workspace_id,
                "dataset_uri": dataset_uri
            }

    # First, load network config to infer dataset_type and dataset_format if not provided
    network_config = read_network_config(network_arch)
    if not network_config:
        logger.error(f"Network config not found for '{network_arch}'")
        return False, {
            "error": f"Network config not found for '{network_arch}'",
            "network_arch": network_arch
        }

    # Infer dataset_type and dataset_format from network config if not provided
    api_params = network_config.get("api_params", {})
    if not dataset_type:
        dataset_type = api_params.get("dataset_type")
        logger.info(f"Inferred dataset_type={dataset_type} from network_arch={network_arch}")
    if not dataset_format:
        formats = api_params.get("formats", [])
        dataset_format = formats[0] if formats else None
        logger.info(f"Inferred dataset_format={dataset_format} from network_arch={network_arch}")

    # NOW load the validation config based on dataset_type (not network_arch!)
    # This matches how dataset creation works - validation rules come from dataset type config
    if dataset_type and dataset_type != network_arch:
        logger.info(f"Loading validation rules from dataset_type config: {dataset_type}")
        validation_network_config = read_network_config(dataset_type)
        if validation_network_config:
            network_config = validation_network_config
        else:
            logger.warning(f"Could not load config for dataset_type '{dataset_type}', using network_arch config")

    # Get dataset validation rules from config
    dataset_validation_config = network_config.get("dataset_validation")
    if not dataset_validation_config:
        # No format-specific validation rules defined
        # But still verify the path is accessible (catch non-existent paths)
        logger.warning(f"No format-specific validation rules defined for {network_arch}")
        logger.info(f"Performing basic accessibility check for {dataset_uri}")

        if is_cloud_path:
            # For cloud paths, try to access the storage to verify path is accessible
            # This replicates the dataset creation behavior where accessing non-existent
            # paths throws an exception that results in "invalid_pull" status
            from nvidia_tao_core.microservices.utils.dataset_utils import SimpleHandler
            try:
                handler_metadata_basic = {
                    "cloud_file_path": cloud_file_path_clean,
                    "type": dataset_type,
                    "format": dataset_format,
                    "use_for": dataset_intent or ["training"],
                    "workspace": workspace_id
                }

                handler = SimpleHandler(
                    org_name="",
                    handler_metadata=handler_metadata_basic,
                    temp_dir="",
                    workspace_metadata=workspace_metadata
                )

                if not handler.cloud_instance:
                    return False, {
                        "error": "Cannot access cloud storage",
                        "error_details": f"Failed to create cloud instance for path '{dataset_uri}'",
                        "dataset_uri": dataset_uri
                    }

                # For paths without validation rules, we can't reliably check if they're valid
                # without knowing the expected file structure. Cloud storage (S3) doesn't have
                # real folders, so checking if a "folder exists" is unreliable.
                #
                # The best we can do is verify the cloud connection works (already done above).
                # If the path truly doesn't exist or is invalid, the job will fail during execution
                # when it tries to read the actual files.
                #
                # This matches the dataset creation behavior where paths without validation rules
                # are accepted if credentials are valid, and only fail later during pull/execution.
                logger.info("No format-specific validation rules - skipping detailed file checks")
                logger.info("Cloud storage accessible. Path will be validated during job execution.")

                return True, {
                    "message": "No format-specific validation rules defined. Path is accessible.",
                    "network_arch": network_arch,
                    "dataset_format": dataset_format
                }

            except Exception as e:
                logger.error(f"Error validating cloud path: {str(e)}")
                return False, {
                    "error": "Failed to validate dataset URI",
                    "error_details": f"Error accessing path '{dataset_uri}': {str(e)}",
                    "dataset_uri": dataset_uri
                }
        else:
            # For local paths, check if directory exists
            import os
            if not os.path.isdir(dataset_uri):
                return False, {
                    "error": "Dataset URI does not exist",
                    "error_details": f"Local directory '{dataset_uri}' does not exist",
                    "dataset_uri": dataset_uri
                }

            return True, {
                "message": "No format-specific validation rules defined. Local path exists.",
                "network_arch": network_arch,
                "dataset_format": dataset_format
            }

    # If we reach here, there ARE validation rules - proceed with format-specific validation
    # Build dataset metadata dict (matching what dataset creation used)
    # The validate_dataset function expects handler_metadata, not a handler object
    handler_metadata = {
        "cloud_file_path": cloud_file_path_clean if is_cloud_path else "",
        "type": dataset_type or network_config.get("api_params", {}).get("dataset_type"),
        "format": dataset_format,
        "use_for": dataset_intent or ["training"],  # Note: field is "use_for" not "intent"
        "workspace": workspace_id
    }

    # Call the existing validation function with correct signature
    # validate_dataset(org_name, handler_metadata, temp_dir="",
    #                  workspace_metadata=None)
    # For cloud: pass dataset_uri in cloud_file_path, temp_dir=""
    # For local: pass dataset_uri in temp_dir, cloud_file_path=""
    logger.info(
        f"Calling validate_dataset with: cloud_path={is_cloud_path}, "
        f"has_workspace={workspace_metadata is not None}"
    )

    is_valid, validation_result = validate_dataset(
        org_name="",  # org_name not needed for validation
        handler_metadata=handler_metadata,
        temp_dir=dataset_uri if not is_cloud_path else "",  # Local path goes in temp_dir
        workspace_metadata=workspace_metadata if is_cloud_path else None
    )

    logger.info(f"Validation result for {dataset_uri}: is_valid={is_valid}, details={validation_result}")

    if not is_valid:
        # Add context to error message
        validation_result["network_arch"] = network_arch
        validation_result["dataset_uri"] = dataset_uri
        validation_result["workspace_id"] = workspace_id
        logger.error(f"Dataset validation failed for {dataset_uri}: {validation_result.get('error_details')}")
    else:
        logger.info(f"Dataset validation passed for {dataset_uri}")

    return is_valid, validation_result


def validate_all_dataset_uris_structure(
    experiment_metadata: dict,
    network_arch: str,
    skip_validation: bool = False
) -> tuple:
    """Validate all dataset URIs in experiment metadata.

    Args:
        experiment_metadata: Experiment metadata containing:
            - train_dataset_uris: List of training dataset URIs
            - eval_dataset_uri: Evaluation dataset URI
            - inference_dataset_uri: Inference dataset URI
            - calibration_dataset_uri: Calibration dataset URI
            - dataset_format: Dataset format (optional, will infer from network config)
            - dataset_type: Dataset type (optional, will infer from network config)
            - workspace: Workspace ID for cloud credentials
        network_arch: Network architecture name
        skip_validation: If True, skip validation

    Returns:
        tuple: (is_valid: bool, error_message: str, validation_details: dict)
    """
    if skip_validation:
        return True, "", {"message": "Validation skipped"}

    # Load network config to get defaults
    network_config = read_network_config(network_arch)
    if not network_config:
        return False, f"Network config not found for '{network_arch}'", {}

    api_params = network_config.get("api_params", {})

    # Get dataset format and type from experiment metadata or network config
    dataset_format = experiment_metadata.get("dataset_format")
    if not dataset_format:
        formats = api_params.get("formats", [])
        dataset_format = formats[0] if formats else None

    dataset_type = experiment_metadata.get("dataset_type") or api_params.get("dataset_type")
    workspace_id = experiment_metadata.get("workspace")

    # Collect all paths to validate
    paths_to_validate = []

    train_paths = experiment_metadata.get("train_dataset_uris", [])
    if train_paths:
        for path in train_paths:
            paths_to_validate.append((path, "train_dataset_uris", ["training"]))

    eval_path = experiment_metadata.get("eval_dataset_uri")
    if eval_path:
        paths_to_validate.append((eval_path, "eval_dataset_uri", ["evaluation"]))

    inference_path = experiment_metadata.get("inference_dataset_uri")
    if inference_path:
        paths_to_validate.append((inference_path, "inference_dataset_uri", ["testing"]))

    calibration_path = experiment_metadata.get("calibration_dataset_uri")
    if calibration_path:
        paths_to_validate.append((calibration_path, "calibration_dataset_uri", ["calibration"]))

    # If no paths to validate, return success
    if not paths_to_validate:
        return True, "", {"message": "No dataset URIs to validate"}

    # Get all supported formats for fallback validation
    all_formats = api_params.get("formats", [])

    # Validate each path
    for path, field_name, intents in paths_to_validate:
        # Skip if it's a UUID (legacy dataset reference - already validated)
        if is_uuid(path):
            logger.info(f"Skipping validation for dataset UUID: {path}")
            continue

        is_valid, validation_details = validate_dataset_uri_structure(
            dataset_uri=path,
            network_arch=network_arch,
            dataset_format=dataset_format,
            dataset_type=dataset_type,
            dataset_intent=intents,
            workspace_id=workspace_id,
            skip_validation=False
        )

        if not is_valid and all_formats:
            # Train and eval datasets may use different formats (e.g. odvg for
            # train, coco for eval).  Try remaining supported formats before
            # reporting a failure.
            for alt_format in all_formats:
                if alt_format == dataset_format:
                    continue
                logger.info(
                    f"Retrying validation for {field_name} with format '{alt_format}'"
                )
                is_valid, validation_details = validate_dataset_uri_structure(
                    dataset_uri=path,
                    network_arch=network_arch,
                    dataset_format=alt_format,
                    dataset_type=dataset_type,
                    dataset_intent=intents,
                    workspace_id=workspace_id,
                    skip_validation=False
                )
                if is_valid:
                    logger.info(
                        f"Validation passed for {field_name} with alt format '{alt_format}'"
                    )
                    break

        if not is_valid:
            error_msg = (
                f"Dataset validation failed for {field_name}: "
                f"{validation_details.get('error_details', 'Unknown error')}"
            )
            return False, error_msg, validation_details

    return True, "", {"message": "All dataset URIs validated successfully"}


def is_uuid(value: str) -> bool:
    """Check if a string is a valid UUID"""
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError, TypeError):
        return False
