# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functions to infer data sources"""
import os
import re
import logging

from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    get_handler_metadata,
    get_workspace_string_identifier,
    get_handler_job_metadata
)
from nvidia_tao_core.microservices.utils.core_utils import read_network_config

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


def contains_results_uuid(data_path):
    """Check if data path is from another TAO API job"""
    # Define the regex pattern to match "/results/{uuid}"
    pattern = r"/results/[0-9a-fA-F-]{36}"
    # Search the pattern in the input string
    match = re.search(pattern, data_path)
    # Return True if the pattern is found, otherwise False
    return bool(match)


def is_direct_path(value):
    """Check if value is a direct path with protocol prefix (e.g., lustre://, aws://, s3://)

    Args:
        value: Value to check (can be string, list, or other)

    Returns:
        bool: True if value contains protocol prefix, False otherwise
    """
    if not isinstance(value, str):
        return False
    return "://" in value


def parse_direct_path(value):
    """Parse a direct path string to extract protocol and path.

    Args:
        value (str): Direct path string (e.g., "lustre://path/to/data" or "aws://bucket/path")

    Returns:
        tuple: (protocol, path) where protocol is the storage type and path is the file path

    Example:
        >>> parse_direct_path("lustre://lustre/fsw/data")
        ('lustre', '/lustre/fsw/data')
        >>> parse_direct_path("aws://bucket/key/file.txt")
        ('aws', 'bucket/key/file.txt')
    """
    if not is_direct_path(value):
        return None, value

    protocol, path = value.split("://", 1)
    protocol = protocol.lower()

    # Normalize s3:// to aws://
    if protocol == "s3":
        protocol = "aws"

    # For local filesystem protocols, ensure path starts with /
    if protocol in ['lustre', 'file', 'local'] and not path.startswith('/'):
        path = '/' + path

    return protocol, path


def create_storage_handler_for_protocol(protocol, path, workspace_metadata=None):
    """Create appropriate storage handler for the given protocol.

    Args:
        protocol (str): Storage protocol (aws, azure, lustre, file, local, etc.)
        path (str): Path within the storage system
        workspace_metadata (dict): Workspace metadata for cloud credentials (optional)

    Returns:
        tuple: (storage_handler, formatted_path) or (None, original_path) if local
    """
    # Local filesystem protocols - no handler needed, return path as-is
    if protocol in ['lustre', 'file', 'local']:
        # For local paths, ensure they start with /
        formatted_path = path if path.startswith('/') else '/' + path
        return None, formatted_path

    # Cloud protocols - create CloudStorage instance
    if protocol in ['aws', 'azure', 'lepton']:
        # If workspace_metadata is provided, use existing credentials
        if workspace_metadata and workspace_metadata.get('cloud_type') == protocol:
            from nvidia_tao_core.microservices.utils.cloud_utils import create_cs_instance
            try:
                storage_handler, _ = create_cs_instance(workspace_metadata)
                return storage_handler, path
            except Exception as e:
                logger.warning(f"Failed to create cloud storage handler: {e}")
                return None, path

        # Otherwise, extract bucket from path and use it
        # Format: bucket/key/to/file
        logger.warning(
            f"Direct {protocol} path specified without workspace credentials. "
            f"Path will be used as-is: {path}"
        )
        return None, path

    # Unknown protocol - log warning and return path as-is
    logger.warning(f"Unknown protocol '{protocol}' in direct path. Path will be used as-is: {path}")
    return None, path


def resolve_dataset_reference(value, workspace_metadata=None, backend_type=None):
    """Resolves dataset reference - can be dataset_id (UUID) or direct path.

    This function provides backward compatibility by supporting both the legacy
    dataset_id approach and the new direct path approach.

    Args:
        value: Either UUID string (dataset_id) or path string (aws://bucket/path, lustre://path, etc.)
        workspace_metadata: Workspace metadata for cloud credentials (optional)
        backend_type: Backend type for validation (optional)

    Returns:
        str: Resolved path

    Raises:
        ValueError: If validation fails or dataset not found
    """
    # Check if it's a direct path (has protocol prefix)
    if is_direct_path(value):
        # Validate path against backend
        from nvidia_tao_core.microservices.utils.dataset_uri_validator import validate_dataset_uri
        is_valid, error = validate_dataset_uri(value, backend_type)
        if not is_valid:
            raise ValueError(error)

        # Parse and return formatted path
        protocol, path = parse_direct_path(value)
        _, formatted_path = create_storage_handler_for_protocol(
            protocol, path, workspace_metadata
        )
        logger.info(f"Resolved direct path: {value} -> {formatted_path}")
        return formatted_path

    # Legacy path: Treat as dataset_id (UUID)
    # This maintains backward compatibility with existing experiments
    try:
        dataset_metadata = get_handler_metadata(value, "datasets")
        if not dataset_metadata:
            raise ValueError(f"Dataset {value} not found")

        # Use existing logic to resolve dataset path
        workspace_id = dataset_metadata.get('workspace')
        workspace_identifier = get_workspace_string_identifier(workspace_id, workspace_cache={})
        source_root = get_source_root(dataset_metadata, workspace_identifier)
        logger.info(f"Resolved dataset_id: {value} -> {source_root}")
        return source_root
    except Exception as e:
        logger.error(f"Failed to resolve dataset reference '{value}': {str(e)}")
        raise ValueError(f"Failed to resolve dataset reference '{value}': {str(e)}") from e


def get_datasets_from_metadata(metadata, source_key):
    """Gets a list of datasets from metadata based on source key.

    Args:
        metadata (dict): Handler metadata containing dataset information
        source_key (str): Key to lookup in metadata (e.g. 'train_datasets')

    Returns:
        list: List of dataset IDs, or empty list if not found
    """
    dataset = metadata.get(source_key)
    return dataset if isinstance(dataset, list) else [dataset] if dataset else []


def get_nested_config_value(config, path):
    """Gets a value from nested config using dot notation path."""
    parts = path.replace("]", "").replace("[", ".").split(".")
    current = config

    for part in parts:
        if part.isdigit():
            part = int(part)
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list):
            if part >= len(current):
                return None
            current = current[part]
        else:
            return None

    return current


def remove_nested_config_value(config, path):
    """Removes a value from nested config using dot notation path."""
    parts = path.replace("]", "").replace("[", ".").split(".")
    current = config

    for part in parts[:-1]:
        if part.isdigit():
            part = int(part)
        if part not in current:
            return
        current = current[part]

    if parts[-1] in current:
        del current[parts[-1]]


def set_nested_config_value(config, path, value):
    """Sets a value in nested config using dot notation path.

    Args:
        config (dict): Config dictionary to modify
        path (str): Dot notation path (e.g. "dataset.train_data_sources")
        value: Value to set at the path

    Example:
        >>> config = {}
        >>> set_nested_config_value(config, "a.b.c", 123)
        >>> config
        {'a': {'b': {'c': 123}}}
    """
    parts = path.replace("]", "").replace("[", ".").split(".")
    current = config

    for part in parts[:-1]:
        if part.isdigit():
            part = int(part)
        if part not in current:
            # Create list if next part is numeric, dict otherwise
            next_part = parts[parts.index(part) + 1]
            current[part] = [] if next_part.isdigit() else {}
        current = current[part]

    last_part = parts[-1]
    if last_part.isdigit():
        last_part = int(last_part)

    # If both are dictionaries, merge them instead of replacing
    if isinstance(current.get(last_part), dict) and isinstance(value, dict):
        current[last_part].update(value)
    else:
        current[last_part] = value


def get_job_id_of_action(dataset_id, kind, action):
    """Gets job ID for a specific action on a dataset."""
    # Implementation of getting job ID from dataset and action
    dataset_metadata = get_handler_metadata(dataset_id, kind)
    job_id = None
    for job in dataset_metadata.get("jobs", []):
        job_metadata = get_handler_job_metadata(job)
        if job_metadata.get("action") == action and job_metadata.get("status") == "Done":
            job_id = job_metadata.get("id")
    return job_id


def _find_convert_job_by_uri_match(source_uri, dataset_convert_action):
    """Find the most recent successful dataset_convert job whose dataset URI matches source_uri.

    When using virtual datasets (direct paths), each create-job call creates a new
    dataset so the normal per-dataset job lookup fails.  This function queries MongoDB
    for any dataset whose URI fields match *source_uri* and that has a completed
    dataset_convert job, returning the most recently created one.
    """
    from nvidia_tao_core.microservices.handlers.mongo_handler import MongoHandler

    if not source_uri:
        return None

    mongo_ds = MongoHandler("tao", "datasets")
    query = {"$or": [
        {"train_dataset_uris": source_uri},
        {"eval_dataset_uri": source_uri},
        {"inference_dataset_uri": source_uri},
        {"calibration_dataset_uri": source_uri},
    ]}
    matching_datasets = list(mongo_ds.find(query))
    best_job_id = None
    best_created = ""
    for ds in matching_datasets:
        ds_id = ds.get("id")
        if not ds_id:
            continue
        job_id = get_job_id_of_action(ds_id, "datasets", dataset_convert_action)
        if job_id:
            job_meta = get_handler_job_metadata(job_id)
            created = job_meta.get("created_on", "") if job_meta else ""
            if not best_job_id or created > best_created:
                best_job_id = job_id
                best_created = created
    if best_job_id:
        logger.info(
            "Found dataset_convert job %s via URI match (uri=%s)",
            best_job_id, source_uri
        )
    return best_job_id


def get_dataset_convert_downloaded_locally(network_config):
    """Get if dataset convert is downloaded locally"""
    dataset_convert_downloaded_locally = False
    if network_config and "cloud_upload" in network_config:
        cloud_upload = network_config["cloud_upload"]
        upload_strategy = cloud_upload.get("upload_strategy", {})
        dataset_convert_strategy = upload_strategy.get("dataset_convert")
        # If dataset_convert has tarball strategy, it will be downloaded locally
        if (dataset_convert_strategy == "tarball_after_completion" or
                (isinstance(dataset_convert_strategy, dict) and "selective_tarball" in dataset_convert_strategy)):
            dataset_convert_downloaded_locally = True
    return dataset_convert_downloaded_locally


def apply_transforms(
    value,
    transforms,
    source_root=None,
    source_ds=None,
    dataset_convert_action=None,
    workspace_identifier=None,
    dataset_convert_downloaded_locally=None,
    parent_job_id=None
):
    """Apply a list of transforms to a value.

    Args:
        value: The value to transform
        transforms: List of transforms to apply
        source_root: Root path of the source dataset
        source_ds: Source dataset ID
        dataset_convert_action: Action for dataset conversion
        workspace_identifier: Workspace identifier for dataset convert job paths
        parent_job_id: Parent job ID for fallback dataset_convert lookup
    """
    if isinstance(transforms, str):
        transforms = [transforms]

    for transform in transforms:
        if transform == "handle_tar_path":
            value = os.path.join(source_root, "images") if contains_results_uuid(source_root) else value
        elif transform == "wrap_in_list":
            value = [value]
        elif transform == "use_dataset_convert_job":
            dataset_convert_job_id = get_job_id_of_action(
                source_ds, kind="datasets", action=dataset_convert_action
            ) or ""
            if not dataset_convert_job_id and source_root:
                dataset_convert_job_id = _find_convert_job_by_uri_match(
                    source_root, dataset_convert_action
                ) or ""
            if "{dataset_convert_job_id}" in value and not dataset_convert_job_id:
                logger.warning(
                    "Unable to resolve dataset-convert job for dataset %s; skipping transform.",
                    source_ds,
                )
                return value

            value = value.replace("{dataset_convert_job_id}", dataset_convert_job_id)
            if dataset_convert_downloaded_locally:
                return value

            if value.startswith("/results/"):
                value = f"{workspace_identifier}{value}"
            else:
                corrected_value = value.replace(source_root, "")
                value = f"{workspace_identifier}{corrected_value.lstrip('/')}"

    return value


def get_source_root(source_ds_metadata, workspace_identifier):
    """Get the source root path for a dataset.

    For SLURM, if cloud_file_path is already absolute, prepend only the protocol prefix.
    For cloud storage, concatenate workspace_identifier with relative cloud_file_path.
    For datasets created with train_dataset_uris (no cloud_file_path), derive path from the URI.
    """
    cloud_file_path = source_ds_metadata.get('cloud_file_path', '')

    # If cloud_file_path is empty, derive from whichever dataset URI is populated
    if not cloud_file_path:
        uri = None
        train_uris = source_ds_metadata.get('train_dataset_uris', [])
        if train_uris and isinstance(train_uris, list) and train_uris[0]:
            uri = train_uris[0]
        elif source_ds_metadata.get('eval_dataset_uri'):
            uri = source_ds_metadata['eval_dataset_uri']
        elif source_ds_metadata.get('inference_dataset_uri'):
            uri = source_ds_metadata['inference_dataset_uri']
        elif source_ds_metadata.get('calibration_dataset_uri'):
            uri = source_ds_metadata['calibration_dataset_uri']

        if uri and workspace_identifier and uri.startswith(workspace_identifier):
            cloud_file_path = uri[len(workspace_identifier):]
            source_ds_metadata['cloud_file_path'] = cloud_file_path
            logger.info(f"Derived cloud_file_path from dataset URI: {cloud_file_path}")

    # For SLURM with absolute paths, only add protocol prefix (not the full base path)
    if workspace_identifier.startswith('slurm://') and cloud_file_path.startswith('/'):
        return f"slurm://{cloud_file_path}"

    # For cloud storage or relative paths, concatenate normally
    return f"{workspace_identifier}{cloud_file_path}"


def get_dataset_metadata_and_paths(source_ds, workspace_cache, kind="datasets", handler_metadata=None):
    """Helper function to get common dataset metadata and paths.

    Args:
        source_ds: Either a dataset UUID or a direct path (aws://, azure://, lustre://, etc.)
        workspace_cache: Cache for workspace metadata
        kind: Handler kind (default: "datasets")
        handler_metadata: Optional experiment/job metadata (for getting workspace when using direct paths)

    Returns:
        tuple: (source_ds_metadata, workspace_identifier, source_root)
    """
    # Check if source_ds is a direct path (new approach)
    if is_direct_path(source_ds):
        logger.info(f"Processing direct path: {source_ds}")
        protocol, path = parse_direct_path(source_ds)

        # For cloud storage (aws, azure, lepton), we need workspace for credentials
        workspace_id = None
        workspace_identifier = ""
        workspace_metadata = None

        if protocol in ['aws', 's3', 'azure', 'lepton'] and handler_metadata:
            workspace_id = handler_metadata.get("workspace")
            if workspace_id:
                workspace_identifier = get_workspace_string_identifier(workspace_id, workspace_cache)
                # Get full workspace metadata for cloud credentials
                workspace_metadata = get_handler_metadata(workspace_id, "workspaces")

        # Format the path based on protocol
        if protocol in ['lustre', 'file', 'local']:
            # Local filesystems - just use the path directly (without protocol)
            formatted_path = path
        else:
            # Cloud storage - get storage handler and format path
            _, path_without_protocol = create_storage_handler_for_protocol(
                protocol, path, workspace_metadata or {}
            )

            # IMPORTANT: For Lepton backend, use lepton:// protocol
            # Detect Lepton from workspace cloud_type (same pattern as results_dir)
            cloud_type = workspace_metadata.get('cloud_type', '') if workspace_metadata else ''
            if cloud_type.lower() == 'lepton':
                # Format: lepton://bucket/path (bucket is already in path_without_protocol)
                formatted_path = f"lepton://{path_without_protocol}"
                logger.info(f"Formatted Lepton path (cloud_type={cloud_type}): {formatted_path}")
            else:
                # For other backends, use original protocol
                formatted_path = f"{protocol}://{path_without_protocol}"
                logger.info(f"Formatted cloud path (cloud_type={cloud_type}): {formatted_path}")

        # Return minimal metadata for direct paths
        # Get dataset type and format from network config and user input
        dataset_type = None
        dataset_format = None

        if handler_metadata:
            # Get network config to extract dataset_type and default formats
            network_arch = handler_metadata.get("network_arch")
            if network_arch:
                network_config = read_network_config(network_arch)
                if network_config and "api_params" in network_config:
                    api_params = network_config["api_params"]
                    # Get dataset type from network config
                    dataset_type = api_params.get("dataset_type")

                    # Get format: user-specified takes priority, then fall back to network config
                    # Check if user specified dataset_format in request
                    user_dataset_format = handler_metadata.get("dataset_format")
                    if user_dataset_format:
                        dataset_format = user_dataset_format
                    else:
                        # Fall back to first format in network config
                        formats = api_params.get("formats", [])
                        if formats:
                            dataset_format = formats[0] if isinstance(formats, list) else formats

        # For cloud storage paths, extract cloud_file_path for check_file_exists_in_cloud()
        # Path format: "bucket-name/path/to/data" -> bucket="bucket-name", cloud_file_path="path/to/data"
        cloud_file_path = ""
        if protocol in ['aws', 's3', 'azure', 'lepton']:
            # Split path into bucket and file path
            # path_without_protocol is in format "bucket-name/path/to/data"
            parts = path_without_protocol.split('/', 1)
            if len(parts) > 1:
                cloud_file_path = parts[1]  # Everything after bucket name
                logger.info(f"Extracted cloud_file_path for file existence checks: {cloud_file_path}")

        source_ds_metadata = {
            "id": source_ds,  # Use the path as ID
            "workspace": workspace_id,
            "type": dataset_type,
            "format": dataset_format,
            "cloud_file_path": cloud_file_path  # Needed by check_file_exists_in_cloud()
        }

        return source_ds_metadata, workspace_identifier, formatted_path

    # Legacy path: source_ds is a dataset UUID
    source_ds_metadata = get_handler_metadata(source_ds, kind=kind)
    workspace_identifier = get_workspace_string_identifier(
        source_ds_metadata.get('workspace'),
        workspace_cache
    )
    source_root = get_source_root(source_ds_metadata, workspace_identifier)
    return source_ds_metadata, workspace_identifier, source_root


def get_source_datasets_from_config(config_source, handler_metadata):
    """Helper function to get source datasets from config.

    Args:
        config_source (str): Source key to lookup in handler metadata (old field names like 'train_datasets')
        handler_metadata (dict): Handler metadata containing dataset information

    Returns:
        list: List of dataset IDs or direct paths
    """
    if config_source == "id":
        return [handler_metadata.get("id")]

    # Try old field name first (for backward compatibility)
    datasets = get_datasets_from_metadata(handler_metadata, config_source)

    # If old field is empty, try new field name (direct paths)
    if not datasets:
        # Map old field names to new field names
        field_mapping = {
            "train_datasets": "train_dataset_uris",
            "eval_dataset": "eval_dataset_uri",
            "inference_dataset": "inference_dataset_uri",
            "calibration_dataset": "calibration_dataset_uri"
        }
        new_field = field_mapping.get(config_source)
        if new_field:
            datasets = get_datasets_from_metadata(handler_metadata, new_field)
            logger.info(f"Using new field '{new_field}' instead of '{config_source}': {datasets}")

    # Return all datasets (both UUIDs and direct paths)
    # Direct paths will be handled appropriately by downstream code
    return datasets if datasets else []


def process_convert_job_spec_path(spec_config, source_ds, dataset_convert_action):
    """Helper function to process path from convert job spec."""
    spec_path = spec_config.get("spec_path")
    mapping = spec_config.get("mapping", {})

    # Get the dataset convert job for this dataset
    dataset_convert_job_id = get_job_id_of_action(
        source_ds, kind="datasets", action=dataset_convert_action
    )
    # For direct-path (URI) datasets, fall back to URI-based lookup
    if not dataset_convert_job_id and is_direct_path(source_ds):
        dataset_convert_job_id = _find_convert_job_by_uri_match(
            source_ds, dataset_convert_action
        )

    if not dataset_convert_job_id:
        return None, None

    # Get the job metadata to access specs
    convert_job_metadata = get_handler_job_metadata(dataset_convert_job_id)
    if not convert_job_metadata:
        return None, None

    # Get the spec value from the convert job
    job_specs = convert_job_metadata.get("specs", {})
    spec_value = get_nested_config_value(job_specs, spec_path)

    # Use the mapping to determine the path
    path_template = (mapping.get(spec_value, mapping.get("*", ""))
                     if spec_value else mapping.get("*", ""))

    return path_template, dataset_convert_job_id


def process_mapping_path_and_transforms(mapping, source_root, source_ds, dataset_convert_action,
                                        workspace_identifier, dataset_convert_downloaded_locally,
                                        parent_job_id=None):
    """Helper function to process mapping path and apply transforms."""
    # Skip optional paths that don't exist
    if mapping.get("optional") and not check_file_exists(source_root, mapping["path"]):
        return None

    # Get the path value
    if mapping.get("path"):
        path = mapping["path"]
        # Resolve tar file to folder if tar doesn't exist but folder does
        source_ds_metadata = get_handler_metadata(source_ds, kind="datasets")
        resolved_path = resolve_tar_or_folder_path(source_ds_metadata, source_root, path)
        value = os.path.join(source_root, resolved_path)
    else:
        value = source_root

    # Apply any transforms
    if "transform" in mapping:
        value = apply_transforms(
            value, mapping["transform"],
            source_root, source_ds, dataset_convert_action,
            workspace_identifier, dataset_convert_downloaded_locally,
            parent_job_id=parent_job_id)

    return value


def apply_workspace_identifier_to_results_path(path, workspace_identifier):
    """Helper function to prepend workspace identifier to results paths."""
    if path and path.startswith("/results/"):
        return workspace_identifier + path
    return path


def replace_placeholder_and_apply_workspace_id(value, placeholder, replacement, workspace_identifier):
    """Helper function to replace placeholders and apply workspace identifier."""
    if isinstance(value, str) and placeholder in value:
        value = value.replace(placeholder, replacement)
        # If this is a path to results, prepend workspace identifier
        if "results/" in value:
            value = workspace_identifier + value
    return value


def process_mapping_entry(mapping, source_root, source_ds, dataset_convert_action,
                          workspace_identifier, dataset_convert_downloaded_locally=False,
                          parent_job_id=None, source_ds_metadata=None):
    """Process a single mapping entry.

    Handles three types of mappings:
    1. Simple mappings with direct path and transform (like ann_file)
    2. Nested mappings with sub-mappings (like data_prefix with pts and img)
    3. String mappings that reference dataset metadata fields (like dataset_format)
    """
    # Handle string mappings that reference dataset metadata fields
    if isinstance(mapping, str):
        # Use provided metadata if available (e.g. for direct path URIs where source_ds is not a UUID)
        if source_ds_metadata is None:
            source_ds_metadata = get_handler_metadata(source_ds, kind="datasets")
        if mapping == "dataset_format":
            return source_ds_metadata.get("format")
        if mapping == "dataset_type":
            return source_ds_metadata.get("type")
        if mapping == "dataset_intent":
            use_for = source_ds_metadata.get("use_for", [])
            return use_for[0] if use_for else None
        # For any other string, try to get it directly from metadata
        return source_ds_metadata.get(mapping)
    # Check if this is a nested mapping (like data_prefix with pts and img)
    if any(isinstance(v, dict) and "path" in v for k, v in mapping.items()):
        # This is a nested mapping (e.g., data_prefix with pts and img)
        result = {}
        for key, sub_mapping in mapping.items():
            if isinstance(sub_mapping, dict) and "path" in sub_mapping:
                value = process_mapping_path_and_transforms(
                    sub_mapping, source_root, source_ds, dataset_convert_action,
                    workspace_identifier, dataset_convert_downloaded_locally,
                    parent_job_id=parent_job_id)
                if value is not None:
                    result[key] = value
        return result if result else None

    # This is a simple mapping (e.g., ann_file with direct path and transform)
    if "path" in mapping:
        return process_mapping_path_and_transforms(
            mapping, source_root, source_ds, dataset_convert_action,
            workspace_identifier, dataset_convert_downloaded_locally,
            parent_job_id=parent_job_id)

    return None


def get_metadata_value(metadata, path_type):
    """Helper function to get metadata values safely"""
    if path_type == "intent":
        use_for = metadata.get("use_for", [])
        if use_for:
            return use_for[0]
        # Infer intent from dataset URI fields when use_for is not set
        if metadata.get("train_dataset_uris"):
            return "training"
        if metadata.get("eval_dataset_uri"):
            return "evaluation"
        if metadata.get("inference_dataset_uri"):
            return "inference"
        return None
    return metadata.get(path_type) if path_type in ["type", "format"] else None


def process_additional_downloads(
    network_config, job_context, handler_metadata, workspace_cache, dataset_convert_action, endpoint_action
):
    """Process additional downloads configuration from network config"""
    additional_downloads = []

    # Auto-generate additional downloads based on upload strategy for dataset_convert
    if job_context.action in ["train", "evaluate", "inference", "retrain", "prune", "export"]:
        cloud_upload = network_config.get("cloud_upload", {})
        upload_strategy = cloud_upload.get("upload_strategy", {})
        dataset_convert_strategy = upload_strategy.get("dataset_convert")

        if dataset_convert_strategy:
            # Get datasets that might have dataset_convert results
            # Check both legacy (UUID) and new (URI) field names
            source_datasets = (get_datasets_from_metadata(handler_metadata, "train_datasets") or
                               get_datasets_from_metadata(handler_metadata, "train_dataset_uris") or
                               get_datasets_from_metadata(handler_metadata, "eval_dataset") or
                               get_datasets_from_metadata(handler_metadata, "eval_dataset_uri") or
                               get_datasets_from_metadata(handler_metadata, "inference_dataset") or
                               get_datasets_from_metadata(handler_metadata, "inference_dataset_uri"))
            if source_datasets:
                source_ds_ref = source_datasets[0]
                dataset_convert_job_id = get_job_id_of_action(
                    source_ds_ref, kind="datasets", action=dataset_convert_action
                )
                # For direct-path (URI) datasets, fall back to URI-based lookup
                if not dataset_convert_job_id and is_direct_path(source_ds_ref):
                    dataset_convert_job_id = _find_convert_job_by_uri_match(
                        source_ds_ref, dataset_convert_action
                    )

                if dataset_convert_job_id:
                    # Get workspace identifier
                    if is_direct_path(source_ds_ref):
                        workspace_identifier = get_workspace_string_identifier(
                            handler_metadata.get('workspace'),
                            workspace_cache
                        )
                    else:
                        source_ds_metadata = get_handler_metadata(source_ds_ref, kind="datasets")
                        workspace_identifier = get_workspace_string_identifier(
                            source_ds_metadata.get('workspace'),
                            workspace_cache
                        )

                    # Generate download path based on strategy
                    ws_prefix = workspace_identifier.rstrip('/')
                    if dataset_convert_strategy == "tarball_after_completion":
                        download_path = (f"{ws_prefix}/results/{dataset_convert_job_id}/"
                                         f"{endpoint_action}_results.tar.gz")
                        additional_downloads.append(download_path)
                    elif isinstance(dataset_convert_strategy, dict) and "selective_tarball" in dataset_convert_strategy:
                        download_path = (f"{ws_prefix}/results/{dataset_convert_job_id}/"
                                         f"{endpoint_action}_selective.tar.gz")
                        additional_downloads.append(download_path)

    # Get additional downloads for the current action
    downloads_config = network_config.get("additional_download", {}).get(job_context.action, [])

    if not downloads_config:
        return additional_downloads

    for download_config in downloads_config:
        # Get source datasets
        source_datasets = get_source_datasets_from_config(download_config["source"], handler_metadata)
        if not source_datasets:
            continue

        # Process each source dataset
        for source_ds in source_datasets:
            (source_ds_metadata,
             workspace_identifier,
             _) = get_dataset_metadata_and_paths(source_ds, workspace_cache, handler_metadata=handler_metadata)

            # Handle path from convert job spec
            if "path_from_convert_job_spec" in download_config:
                path_template, dataset_convert_job_id = process_convert_job_spec_path(
                    download_config["path_from_convert_job_spec"], source_ds, dataset_convert_action
                )

                if path_template:
                    # Replace {dataset_convert_job_id} with actual job ID
                    download_path = path_template.replace("{dataset_convert_job_id}", dataset_convert_job_id)

                    # Replace {dataset_path} with dataset-specific path
                    if "{dataset_path}" in download_path:
                        # Use dataset ID or a default path component
                        dataset_path = source_ds_metadata.get("cloud_file_path", source_ds)
                        download_path = download_path.replace("{dataset_path}", dataset_path)

                    # Prepend workspace identifier if this is a results path
                    download_path = apply_workspace_identifier_to_results_path(download_path, workspace_identifier)
                    additional_downloads.append(download_path)

            # Handle direct path (fallback)
            elif "path" in download_config:
                path = download_config["path"]
                if path:
                    # Replace {dataset_convert_job_id} if present
                    dataset_convert_job_id = get_job_id_of_action(
                        source_ds, kind="datasets", action=dataset_convert_action
                    )
                    if dataset_convert_job_id and "{dataset_convert_job_id}" in path:
                        path = path.replace("{dataset_convert_job_id}", dataset_convert_job_id)

                    # Replace {dataset_path} with dataset-specific path
                    if "{dataset_path}" in path:
                        # Use dataset ID or a default path component
                        dataset_path = source_ds_metadata.get("cloud_file_path", source_ds)
                        path = path.replace("{dataset_path}", dataset_path)
                        path = workspace_identifier + path

                    # Prepend workspace identifier if this is a results path
                    path = apply_workspace_identifier_to_results_path(path, workspace_identifier)
                    additional_downloads.append(path)

    return list(set(additional_downloads))


def apply_data_source_config(config, job_context, handler_metadata):
    """Generic data source configuration using config file"""
    workspace_cache = {}
    dataset_convert_action = "dataset_convert"

    job_network, job_action = (("image", "validate") if job_context.action == "validate_images"
                               else (job_context.network, job_context.action))
    network_config = read_network_config(job_network)
    dataset_convert_downloaded_locally = get_dataset_convert_downloaded_locally(network_config)

    # Keep track of paths that have already been set by special handlers
    already_configured_paths = set()

    # Handle dynamic config adjustments first
    if "dynamic_config" in network_config:
        dynamic_config = network_config["dynamic_config"]
        model_type_key = dynamic_config.get("model_type_key")
        model_type = get_nested_config_value(config, model_type_key) if model_type_key else None

        # Check parent action rules
        if "parent_action_rules" in dynamic_config and job_context.parent_id:
            parent_job_metadata = get_handler_job_metadata(job_context.parent_id)
            parent_action = parent_job_metadata.get("action") if parent_job_metadata else None

            if parent_action and parent_action in dynamic_config["parent_action_rules"]:
                rules = dynamic_config["parent_action_rules"][parent_action]

                # Handle direct set_value rules
                if "set_value" in rules:
                    # Check if there's an action_restriction
                    action_restriction = rules.get("action_restriction", None)

                    # Check conditional if present
                    conditional_pass = True
                    if "conditional" in rules:
                        cond = rules["conditional"]
                        if "metadata_key" in cond:
                            meta_value = handler_metadata.get(cond["metadata_key"], None)
                            conditional_pass = not (("equals" in cond and meta_value != cond["equals"]) or
                                                    ("not_equals" in cond and meta_value == cond["not_equals"]))
                        elif "config_path" in cond:
                            config_value = get_nested_config_value(config, cond["config_path"])
                            conditional_pass = not (("equals" in cond and config_value != cond["equals"]) or
                                                    ("not_equals" in cond and config_value == cond["not_equals"]))

                    if conditional_pass:
                        for config_path, value in rules["set_value"].items():
                            # Check if this config_path is restricted to specific actions
                            if action_restriction:
                                is_list_restriction = (isinstance(action_restriction, list) and
                                                       job_action not in action_restriction)
                                is_dict_restriction = (isinstance(action_restriction, dict) and
                                                       config_path in action_restriction and
                                                       job_action not in action_restriction[config_path])
                                if is_list_restriction or is_dict_restriction:
                                    continue

                            # Replace {parent_id} with actual parent ID if present
                            workspace_identifier = get_workspace_string_identifier(
                                handler_metadata.get('workspace'),
                                workspace_cache
                            )
                            value = replace_placeholder_and_apply_workspace_id(
                                value, "{parent_id}", job_context.parent_id, workspace_identifier
                            )

                            set_nested_config_value(config, config_path, value)
                            already_configured_paths.add(config_path)

                # Handle check_parent_specs rules
                if "check_parent_specs" in rules:
                    # Get action_restriction if present
                    action_restriction = rules.get("action_restriction", None)

                    for spec_path, expected_value in rules["check_parent_specs"].items():
                        if spec_path == "if_match":
                            continue

                        # Get the actual value from parent specs
                        parent_specs = parent_job_metadata.get("specs", {})
                        actual_value = get_nested_config_value(parent_specs, spec_path)

                        # If the value matches, apply the rules in if_match
                        if actual_value == expected_value and "if_match" in rules["check_parent_specs"]:
                            match_rules = rules["check_parent_specs"]["if_match"]

                            if "set_value" in match_rules:
                                for config_path, value in match_rules["set_value"].items():
                                    # Check if this config_path is restricted to specific actions
                                    if action_restriction:
                                        if isinstance(action_restriction, list):
                                            # Simple list of allowed actions
                                            if job_action not in action_restriction:
                                                continue
                                        elif isinstance(action_restriction, dict):
                                            # Dict mapping config paths to allowed actions
                                            if config_path in action_restriction:
                                                if job_action not in action_restriction[config_path]:
                                                    continue
                                            else:
                                                # If config_path not in action_restriction dict, apply to all actions
                                                pass

                                    # Replace {parent_id} with actual parent ID if present
                                    workspace_identifier = get_workspace_string_identifier(
                                        handler_metadata.get('workspace'),
                                        workspace_cache
                                    )
                                    value = replace_placeholder_and_apply_workspace_id(
                                        value, "{parent_id}", job_context.parent_id, workspace_identifier
                                    )

                                    set_nested_config_value(config, config_path, value)
                                    already_configured_paths.add(config_path)

        # Check action rules
        if "action_rules" in dynamic_config:
            if job_action in dynamic_config["action_rules"]:
                rules = dynamic_config["action_rules"][job_action]
                if "set_value" in rules:
                    for config_path, value in rules["set_value"].items():
                        set_nested_config_value(config, config_path, value)
                if "remove" in rules:
                    for path in rules["remove"]:
                        remove_nested_config_value(config, path)

        # Handle model type specific rules
        if model_type and "rules" in dynamic_config:
            rules = dynamic_config["rules"].get(model_type, {})

            # Handle removals
            for path in rules.get("remove", []):
                remove_nested_config_value(config, path)

            # Handle action-specific removals
            for path in rules.get("remove_if_action", {}).get(job_action, []):
                remove_nested_config_value(config, path)

            # Handle joint model path splitting
            if (rules.get("transform") == "split_pretrained_paths" and
                    "rgb_pretrained_model_path" in config.get("model", {})):
                ptm_paths = config["model"]["rgb_pretrained_model_path"].split(",")
                rgb_path = next((p for p in ptm_paths if "_rgb_" in p), ptm_paths[0])
                of_path = next((p for p in ptm_paths if "_of_" in p), ptm_paths[1])
                config["model"]["rgb_pretrained_model_path"] = rgb_path
                config["model"]["of_pretrained_model_path"] = of_path

        # Handle defaults
        for path, value in dynamic_config.get("defaults", {}).items():
            if not get_nested_config_value(config, path):
                set_nested_config_value(config, path, value)

    # Apply data source mappings
    data_sources = network_config.get("data_sources", {}).get(job_action, {})

    for config_path, source_config in data_sources.items():
        # Get source datasets
        if config_path in already_configured_paths:
            continue

        # Check if user provided direct path in specs (takes precedence)
        existing_value = get_nested_config_value(config, config_path)
        if existing_value:
            # Handle direct paths (e.g., lustre://path, aws://bucket/path)
            if isinstance(existing_value, str) and is_direct_path(existing_value):
                protocol, path = parse_direct_path(existing_value)
                logger.info(f"Direct path detected for {config_path}: {protocol}://{path}")

                # For local filesystems, strip protocol and use absolute path
                if protocol in ['lustre', 'file', 'local']:
                    set_nested_config_value(config, config_path, path)
                    logger.info(f"Stripped local filesystem protocol, using path: {path}")
                # For cloud paths on Lepton backend, convert to lepton:// protocol
                elif protocol in ['aws', 's3', 'azure']:
                    # Check if running on Lepton backend from workspace cloud_type
                    workspace_id = handler_metadata.get("workspace")
                    cloud_type = ""
                    if workspace_id:
                        workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
                        if workspace_metadata:
                            cloud_type = workspace_metadata.get('cloud_type', '').lower()

                    if cloud_type == 'lepton':
                        # Convert aws:// or azure:// to lepton://
                        lepton_path = f"lepton://{path}"
                        set_nested_config_value(config, config_path, lepton_path)
                        logger.info(f"Converted to Lepton path (cloud_type={cloud_type}): {lepton_path}")

                # User specified direct path, skip inference
                already_configured_paths.add(config_path)
                continue
            # Handle absolute paths without protocol (e.g., /lustre/fsw/...)
            # These are user-provided paths that should be preserved
            if isinstance(existing_value, str) and existing_value.startswith('/'):
                logger.info(
                    f"Absolute path detected for {config_path}: {existing_value}, preserving user-provided value"
                )
                already_configured_paths.add(config_path)
                continue
            # Handle lists with potential direct paths
            if isinstance(existing_value, list):
                has_direct_paths = any(isinstance(v, str) and is_direct_path(v) for v in existing_value)
                if has_direct_paths:
                    # Check if running on Lepton backend from workspace cloud_type
                    workspace_id = handler_metadata.get("workspace")
                    cloud_type = ""
                    if workspace_id:
                        workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
                        if workspace_metadata:
                            cloud_type = workspace_metadata.get('cloud_type', '').lower()

                    # Process each item in list
                    processed_list = []
                    for item in existing_value:
                        if isinstance(item, str) and is_direct_path(item):
                            protocol, path = parse_direct_path(item)
                            if protocol in ['lustre', 'file', 'local']:
                                processed_list.append(path)
                                logger.info(f"Stripped protocol from list item: {protocol}://{path} -> {path}")
                            elif protocol in ['aws', 's3', 'azure'] and cloud_type == 'lepton':
                                # Convert cloud paths to lepton:// for Lepton backend
                                lepton_path = f"lepton://{path}"
                                processed_list.append(lepton_path)
                                logger.info(
                                    f"Converted list item to Lepton path "
                                    f"(cloud_type={cloud_type}): {lepton_path}"
                                )
                            else:
                                processed_list.append(item)  # Keep cloud paths as-is for other backends
                        else:
                            processed_list.append(item)
                    set_nested_config_value(config, config_path, processed_list)
                    logger.info(f"Processed direct paths in list for {config_path}")
                    already_configured_paths.add(config_path)
                    continue

        # Handle special source: parent_job_specs
        if source_config["source"] == "parent_job_specs":
            if job_context.parent_id and "check_key_exists" in source_config:
                parent_job_metadata = get_handler_job_metadata(job_context.parent_id)
                if parent_job_metadata:
                    parent_specs = parent_job_metadata.get("specs", {})
                    key_path = source_config["check_key_exists"]

                    # Check if the key exists in parent specs
                    key_exists = get_nested_config_value(parent_specs, key_path) is not None
                    set_nested_config_value(config, config_path, key_exists)
                    already_configured_paths.add(config_path)
                    logger.info(f"Set {config_path} = {key_exists} based on parent job specs key '{key_path}'")
            continue

        source_datasets = get_source_datasets_from_config(
            source_config["source"], handler_metadata
        )
        if not source_datasets:
            continue
        (source_ds_metadata,
         workspace_identifier,
         source_root) = get_dataset_metadata_and_paths(
            source_datasets[0], workspace_cache, handler_metadata=handler_metadata
        )

        # Handle value from metadata
        if "value_from_metadata" in source_config:
            meta_config = source_config["value_from_metadata"]
            meta_value = source_ds_metadata.get(meta_config["key"], None)
            value = meta_config["mapping"].get(meta_value, meta_config["mapping"].get("*", meta_config["default"]))
            set_nested_config_value(config, config_path, value)
            already_configured_paths.add(config_path)  # Mark as configured
            continue

        # Handle path from convert job spec
        if "path_from_convert_job_spec" in source_config:
            path, dataset_convert_job_id = process_convert_job_spec_path(
                source_config["path_from_convert_job_spec"], source_datasets[0], dataset_convert_action
            )

            if path:
                # Check if we're using the dataset convert job transform
                if "use_dataset_convert_job" in source_config.get("transform", []):
                    # Build the results path template for the transform
                    value = f"/results/{dataset_convert_job_id}/{path}"
                else:
                    # Resolve tar file to folder if tar doesn't exist but folder does
                    resolved_path = resolve_tar_or_folder_path(source_ds_metadata, source_root, path)
                    value = os.path.join(source_root, resolved_path)
                value = apply_transforms(
                    value, source_config.get("transform", []),
                    source_root, source_datasets[0], dataset_convert_action,
                    workspace_identifier, dataset_convert_downloaded_locally,
                    parent_job_id=job_context.parent_id)
                set_nested_config_value(config, config_path, value)
                already_configured_paths.add(config_path)  # Mark as configured
                continue

        # Handle path from type/intent/source/model_type cases
        for path_type in ["type", "format", "intent", "source", "model_type"]:
            key = f"path_from_{path_type}"
            if key in source_config:
                meta_value = {
                    "model_type": lambda: get_nested_config_value(config, model_type_key) if model_type_key else None,
                    "source": lambda x=source_config: x["source"],
                    "type": lambda x=source_ds_metadata: get_metadata_value(x, "type"),
                    "format": lambda x=source_ds_metadata: get_metadata_value(x, "format"),
                    "intent": lambda x=source_ds_metadata: get_metadata_value(x, "intent")
                }[path_type]()

                if path_type == "model_type" and meta_value == "openpose":
                    meta_value = "kinetics"

                if meta_value in source_config[key]:
                    path_config = source_config[key][meta_value]
                    # Handle array of paths (fallback mechanism)
                    if isinstance(path_config, list):
                        path = next((p for p in path_config
                                    if check_file_exists_in_cloud(source_ds_metadata, source_root, p)),
                                    path_config[0])
                    else:
                        path = path_config
                elif path_type != "intent":  # intent doesn't use fallback
                    path = source_config[key].get("*")
                else:
                    continue

                # Resolve tar file to folder if tar doesn't exist but folder does
                resolved_path = resolve_tar_or_folder_path(source_ds_metadata, source_root, path)
                value = os.path.join(source_root, resolved_path)
                if path_type == "source":
                    value = apply_transforms(
                        value, source_config.get("transform", []),
                        source_root, source_datasets[0], dataset_convert_action,
                        workspace_identifier, dataset_convert_downloaded_locally,
                        parent_job_id=job_context.parent_id)
                set_nested_config_value(config, config_path, value)
                already_configured_paths.add(config_path)  # Mark as configured
                continue

        # Skip rest of processing if already configured by a special handler
        if config_path in already_configured_paths:
            continue

        # Check conditional
        if "conditional" in source_config:
            cond = source_config["conditional"]
            meta_value = source_ds_metadata.get(cond["metadata_key"], None)
            if (("equals" in cond and meta_value != cond["equals"]) or
                    ("not_equals" in cond and meta_value == cond["not_equals"])):
                continue

        if source_config.get("multiple_sources", False):
            # Handle multiple sources
            result_list = []
            for source_ds in source_datasets:
                (source_ds_metadata,
                 workspace_identifier,
                 source_root) = get_dataset_metadata_and_paths(
                    source_ds, workspace_cache, handler_metadata=handler_metadata
                )

                if "mapping" in source_config:
                    entry = {}
                    for key, mapping in source_config["mapping"].items():
                        value = process_mapping_entry(
                            mapping, source_root, source_ds,
                            dataset_convert_action, workspace_identifier, network_config,
                            parent_job_id=job_context.parent_id,
                            source_ds_metadata=source_ds_metadata)
                        if value is not None:
                            entry[key] = value
                    if entry:
                        result_list.append(entry)
                else:
                    path = source_config.get("path", "")
                    if path:
                        # Resolve tar file to folder if tar doesn't exist but folder does
                        resolved_path = resolve_tar_or_folder_path(source_ds_metadata, source_root, path)
                        value = os.path.join(source_root, resolved_path)
                    else:
                        value = source_root
                    result_list.append(value)

            if result_list:
                set_nested_config_value(config, config_path, result_list)

        else:
            # Handle single source
            source_ds = source_datasets[0]
            if source_config.get("value") is not None:
                # Replace {job_id} with actual job ID if present
                value = source_config["value"]
                if isinstance(value, str) and "{job_id}" in value:
                    value = value.replace("{job_id}", job_context.id)
                set_nested_config_value(config, config_path, value)
                continue

            (source_ds_metadata,
             workspace_identifier,
             source_root) = get_dataset_metadata_and_paths(
                source_ds, workspace_cache, handler_metadata=handler_metadata
            )

            if "mapping" in source_config:
                result = {}
                for key, mapping in source_config["mapping"].items():
                    value = process_mapping_entry(
                        mapping, source_root, source_ds,
                        dataset_convert_action, workspace_identifier, network_config,
                        parent_job_id=job_context.parent_id,
                        source_ds_metadata=source_ds_metadata)
                    if value is not None:
                        result[key] = value

                if result:
                    set_nested_config_value(config, config_path, result)
            else:
                path = source_config.get("path", "")
                # Skip optional paths that don't exist (user can still provide via specs)
                if source_config.get("optional") and path:
                    if not check_file_exists_in_cloud(source_ds_metadata, source_root, path):
                        continue
                if path and not dataset_convert_downloaded_locally:
                    # Resolve tar file to folder if tar doesn't exist but folder does
                    resolved_path = resolve_tar_or_folder_path(source_ds_metadata, source_root, path)
                    value = os.path.join(source_root, resolved_path)
                elif path:
                    value = path
                else:
                    value = source_root

                value = apply_transforms(
                    value, source_config.get("transform", []),
                    source_root, source_ds, dataset_convert_action,
                    workspace_identifier, dataset_convert_downloaded_locally,
                    parent_job_id=job_context.parent_id)
                set_nested_config_value(config, config_path, value)

    # Add preserve_source_path_params from network config if specified
    preserve_source_path_params = network_config.get("preserve_source_path_params")
    if preserve_source_path_params:
        config["preserve_source_path_params"] = preserve_source_path_params
        logger.info("Added preserve_source_path_params to config: %s", preserve_source_path_params)

    # Process additional downloads
    endpoint_action = (
        network_config.get("actions_mapping", {})
        .get(dataset_convert_action, {})
        .get("action", dataset_convert_action)
    )
    additional_downloads = process_additional_downloads(
        network_config, job_context, handler_metadata, workspace_cache, dataset_convert_action, endpoint_action
    )

    # Handle custom data loader script for cosmos-rl and other networks
    if "custom_script" in config:
        custom_script_path = config.get("custom_script")
        if custom_script_path:
            logger.info("Custom data loader script found in config: %s", custom_script_path)
            additional_downloads.append(custom_script_path)

    if additional_downloads:
        config["additional_downloads"] = additional_downloads

    # For networks that export models larger than 2 GB, the ONNX exporter splits output into
    # model.onnx + model.bin (and possibly other artefacts).  When gen_trt_engine runs, the
    # ONNX runtime requires all sibling files to be present in the same directory as the .onnx.
    #
    # We achieve this by adding a top-level "_companion_onnx_folder" key to the spec whose
    # value is the cloud path of the directory containing the ONNX file.  The container handler
    # already calls download_files_from_spec on the full spec dict before launching the job, so
    # this triggers a folder download (download_folder) that places every export artefact next
    # to the ONNX file.  The key is popped by the container handler before the YAML spec is
    # written, so it never reaches the CLI tool.
    if (job_action == "gen_trt_engine" and
            network_config.get("companion_onnx_bin_download", False)):
        onnx_file_path = get_nested_config_value(config, "gen_trt_engine.onnx_file")
        if onnx_file_path and isinstance(onnx_file_path, str) and onnx_file_path.endswith(".onnx"):
            onnx_folder_path = os.path.dirname(onnx_file_path)
            if onnx_folder_path:
                config["_companion_onnx_folder"] = onnx_folder_path
                logger.info(
                    "Added parent export folder for companion download: %s", onnx_folder_path
                )
            else:
                logger.warning(
                    "Could not derive parent export folder from onnx_file path '%s'; "
                    "companion artefacts (e.g. model.bin) will not be pre-downloaded.",
                    onnx_file_path
                )

    # For trt_inference, the deploy script requires a *_config.yaml alongside model.engine
    # (produced during ONNX export).  Only model.engine is in the spec, so the config yaml is
    # never downloaded.  We reuse _companion_onnx_folder to download the full parent TRT folder
    # (excluding status.json) before job launch.  The .endswith(".engine") check distinguishes
    # TRT inference (parent action == gen_trt_engine) from TAO inference (parent is a checkpoint).
    elif (job_action == "inference" and
            network_config.get("companion_onnx_bin_download", False)):
        trt_engine_path = get_nested_config_value(config, "inference.trt_engine")
        if trt_engine_path and isinstance(trt_engine_path, str) and trt_engine_path.endswith(".engine"):
            trt_folder_path = os.path.dirname(trt_engine_path)
            if trt_folder_path:
                config["_companion_onnx_folder"] = trt_folder_path
                logger.info(
                    "Added parent TRT engine folder for companion download: %s", trt_folder_path
                )
            else:
                logger.warning(
                    "Could not derive parent TRT folder from trt_engine path '%s'; "
                    "companion artefacts (e.g. *_config.yaml) will not be pre-downloaded.",
                    trt_engine_path
                )

    return config


def check_file_exists(root_path, file_path):
    """Check if a file exists in the given root path."""
    full_path = os.path.join(root_path, file_path)
    return os.path.exists(full_path)


def resolve_tar_or_folder_path(source_ds_metadata, source_root, file_path):
    """Resolve tar file path to folder path if tar doesn't exist but folder does.

    Args:
        source_ds_metadata (dict): Dataset metadata
        source_root (str): Root path of the dataset
        file_path (str): Relative file path (e.g., "images.tar.gz")

    Returns:
        str: Original file_path or resolved folder path if tar doesn't exist
    """
    # Check if this is a tar file path
    if not (file_path.endswith('.tar.gz') or file_path.endswith('.tar')):
        return file_path

    extracted_dir = file_path.replace('.tar.gz', '').replace('.tar', '')

    # Try cloud storage check first (includes SLURM via SSH)
    if source_ds_metadata.get('workspace'):
        try:
            from nvidia_tao_core.microservices.utils.cloud_utils import create_cs_instance
            workspace_metadata = get_handler_metadata(source_ds_metadata.get('workspace'), kind="workspaces")
            if workspace_metadata:
                cloud_type = workspace_metadata.get('cloud_type', '')
                cloud_instance, _ = create_cs_instance(workspace_metadata)
                if cloud_instance:
                    cloud_file_path = source_ds_metadata.get('cloud_file_path', '')
                    # For SLURM, preserve absolute paths; for cloud storage, strip leading slash
                    if cloud_type == 'slurm':
                        cloud_path = f"{cloud_file_path}/{file_path}" if cloud_file_path else file_path
                        extracted_path = f"{cloud_file_path}/{extracted_dir}" if cloud_file_path else extracted_dir
                    else:
                        cloud_path = f"{cloud_file_path.strip('/')}/{file_path}" if cloud_file_path else file_path
                        stripped = cloud_file_path.strip('/') if cloud_file_path else ''
                        extracted_path = f"{stripped}/{extracted_dir}" if stripped else extracted_dir

                    # Check if tar file exists
                    if cloud_instance.is_file(cloud_path):
                        return file_path

                    # If tar doesn't exist, check for extracted folder
                    if cloud_instance.is_folder(extracted_path):
                        logger.info(f"Using extracted folder {extracted_dir} instead of tar file {file_path}")
                        return extracted_dir

                    # Neither exists, return original path
                    return file_path
        except Exception as e:
            logger.warning(f"Cloud storage check failed during path resolution: {e}")
            return file_path

    # Fallback to local check
    cleaned_source_root = source_root.replace('slurm://', '').replace('aws://', '').replace('azure://', '')
    full_tar_path = os.path.join(cleaned_source_root, file_path)
    full_dir_path = full_tar_path.replace('.tar.gz', '').replace('.tar', '')

    if os.path.isfile(full_tar_path):
        return file_path
    if os.path.isdir(full_dir_path):
        logger.info(f"Using extracted folder {extracted_dir} instead of tar file {file_path}")
        return extracted_dir

    # Neither exists, return original path
    return file_path


def check_file_exists_in_cloud(source_ds_metadata, source_root, file_path):
    """Check if a file exists, supporting cloud, SLURM, and local storage.

    For tar files (e.g., videos.tar.gz), checks for both the tar file AND the extracted directory (videos/).
    """
    file_exists = False

    # Try cloud storage check first (includes SLURM via SSH)
    if source_ds_metadata.get('workspace'):
        try:
            from nvidia_tao_core.microservices.utils.cloud_utils import create_cs_instance
            workspace_metadata = get_handler_metadata(source_ds_metadata.get('workspace'), kind="workspaces")
            if workspace_metadata:
                cloud_type = workspace_metadata.get('cloud_type', '')
                cloud_instance, _ = create_cs_instance(workspace_metadata)
                if cloud_instance:
                    cloud_file_path = source_ds_metadata.get('cloud_file_path', '')
                    # For SLURM, preserve absolute paths; for cloud storage, strip leading slash
                    if cloud_type == 'slurm':
                        # SLURM uses absolute paths - don't strip leading slash
                        cloud_path = f"{cloud_file_path}/{file_path}" if cloud_file_path else file_path
                    else:
                        # Cloud storage (AWS/Azure) needs relative paths
                        cloud_path = f"{cloud_file_path.strip('/')}/{file_path}" if cloud_file_path else file_path

                    # Check if the tar file exists
                    file_exists = cloud_instance.is_file(cloud_path)
                    logger.info(f"{cloud_type.upper()} check for {cloud_path}: {file_exists}")

                    # If tar file not found, check for extracted directory (e.g., videos.tar.gz -> videos/)
                    if not file_exists and (file_path.endswith('.tar.gz') or file_path.endswith('.tar')):
                        extracted_dir = file_path.replace('.tar.gz', '').replace('.tar', '')
                        extracted_path = f"{cloud_file_path}/{extracted_dir}" if cloud_file_path else extracted_dir
                        if not cloud_type == 'slurm':
                            stripped = cloud_file_path.strip('/') if cloud_file_path else ''
                            extracted_path = f"{stripped}/{extracted_dir}" if stripped else extracted_dir

                        dir_exists = cloud_instance.is_folder(extracted_path)
                        logger.info(
                            f"{cloud_type.upper()} check for extracted directory "
                            f"{extracted_path}: {dir_exists}"
                        )
                        if dir_exists:
                            logger.info(f"Found extracted directory instead of tar file: {extracted_path}")
                            return True

                    return file_exists
        except Exception as e:
            logger.warning(f"Cloud storage check failed: {e}")
            # For SLURM, don't fallback to local check as paths are on remote cluster
            if workspace_metadata and workspace_metadata.get('cloud_type') == 'slurm':
                logger.error(f"SLURM path check failed for {file_path}, cannot fallback to local")
                return False

    # Fallback to local check only if not a SLURM workspace
    file_exists = check_file_exists(source_root, file_path)
    logger.info(f"Local check for {file_path}: {file_exists}")
    return file_exists
