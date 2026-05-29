# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions"""
import os
import json
import time
import ruamel.yaml
import base64
import orjson
import shutil
import hashlib
import requests
import functools
import numpy as np
import logging
from filelock import FileLock
from kubernetes import client, config
from enum import Enum

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)

NUM_OF_RETRY = 5
base_exp_uuid = "00000000-0000-0000-0000-000000000000"


def sha256_checksum(file_path):
    """Return sh256 checksum of a file"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def remove_key_by_flattened_string(d, key_string, sep='.'):
    """Removes the flattened key from the dictionary"""
    keys = key_string.split(sep)
    current_dict = d
    for key in keys[:-1]:
        current_dict = current_dict.get(key, {})
    if current_dict:
        current_dict.pop(keys[-1], None)


def create_folder_with_permissions(folder_name):
    """Create folder with write permissions"""
    os.makedirs(folder_name, exist_ok=True)
    os.chmod(folder_name, 0o777)


def is_pvc_space_free(threshold_bytes):
    """Check if pvc has required free space"""
    _, _, free_space = shutil.disk_usage('/')
    return free_space > threshold_bytes, free_space


def read_network_config(network):
    """Reads the network handler json config file"""
    # CLONE EXISTS AT pretrained_models.py
    _dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    logger.debug("_dir_path: %s", _dir_path)
    # If dataset_format is user_custom, return empty dict. TODO: separate dataset format vs. network config reading
    if network == "user_custom":
        return {}
    config_json_path = os.path.join(_dir_path, "handlers", "network_configs", f"{network}.config.json")
    if not os.path.exists(config_json_path):
        logger.warning("Network config doesn't exist at %s", config_json_path)
        return {}
    cli_config = {}
    with open(config_json_path, mode='r', encoding='utf-8-sig') as f:
        cli_config = json.load(f)
    return cli_config


def get_spec_file_extension(backend):
    """Get file extension for the given spec backend"""
    backend_to_extension = {
        "yaml": "yaml",
        "protobuf": "txt",
        "toml": "toml",
        "json": "json"
    }
    return backend_to_extension.get(backend, "yaml")  # default to yaml if unknown


def get_spec_backend_info(network):
    """Get spec backend and file extension from network config"""
    config = read_network_config(network)
    spec_backend = config.get("api_params", {}).get("spec_backend", "yaml")
    file_extension = get_spec_file_extension(spec_backend)
    return spec_backend, file_extension


def get_microservices_network_and_action(network, action):
    """Maps a network and action to the appropriate microservices network and action.

    Args:
        network (str): The original network name.
        action (str): The original action name.

    Returns:
        tuple: (microservices_network, microservices_action) - The mapped network and action names.
    """
    # Start with defaults (no change)
    microservices_network = network
    microservices_action = action

    if action == "validate_images":
        return "image", "validate"

    # Try to get the mapping from the network config
    try:
        network_config = read_network_config(network)
        action_mapping = network_config.get("actions_mapping", {})

        # If this action has a mapping defined
        if action in action_mapping or "*" in action_mapping:
            mapping = action_mapping[action] if action in action_mapping else action_mapping["*"]
            if "network" in mapping:
                microservices_network = mapping["network"]
            if "action" in mapping:
                microservices_action = mapping["action"]
    except Exception:
        # Fallback to the original values if any error occurs
        pass

    return microservices_network, microservices_action


def get_orchestration_network_from_microservices(microservices_network):
    """Reverse mapping: finds the orchestration network that maps to a given microservices network.

    This function scans all network config files to find which orchestration network
    has an actions_mapping that maps to the provided microservices network.

    Args:
        microservices_network (str): The microservices network name (e.g., "auto_label", "augmentation").

    Returns:
        str: The orchestration network name (e.g., "object_detection"), or the original
             microservices_network if no mapping is found.

    Example:
        >>> get_orchestration_network_from_microservices("auto_label")
        "object_detection"
        >>> get_orchestration_network_from_microservices("augmentation")
        "object_detection"
    """
    import pathlib

    # Start with default (no change)
    orchestration_network = microservices_network

    # Scan all network config files
    _dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config_dir = pathlib.Path(_dir_path) / "handlers" / "network_configs"

    if not config_dir.exists():
        return orchestration_network

    try:
        for config_file in config_dir.glob("*.config.json"):
            network_name = config_file.stem.replace(".config", "")

            # Skip non-orchestration networks (dataset types are not orchestration networks)
            if network_name in ["image_classification", "object_detection", "segmentation", "image"]:
                # These might be dataset types, but check if they have actions_mapping that reference our network
                pass

            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    actions_mapping = config.get("actions_mapping", {})

                    # Check if any action maps to our microservices network
                    for _, mapping in actions_mapping.items():
                        if isinstance(mapping, dict) and mapping.get("network") == microservices_network:
                            # Found a mapping to this microservices network
                            return network_name
            except (json.JSONDecodeError, IOError):
                continue
    except Exception as e:
        logger.warning(f"Error scanning network configs for reverse mapping: {e}")

    # If no mapping found, return the original network
    return orchestration_network


def get_monitoring_metric(network):
    """Get the monitoring metric for a specific network.

    Args:
        network (str): Name of the network

    Returns:
        str or list: The monitoring metric(s) for the network.
            - If config has a single string, returns that string.
            - If config has a list, returns the list (first available will be used).
        None: If network not found or has no monitoring metric defined
    """
    _dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config_json_path = os.path.join(_dir_path, "handlers", "network_configs", f"{network}.config.json")

    try:
        with open(config_json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            if "metrics" in config and "monitoring_metric" in config["metrics"]:
                return config["metrics"]["monitoring_metric"]
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Error reading config file for network {network}: {e}")

    return None


def find_first_available_metric(metric_config, available_metrics):
    """Find the first available metric from a metric configuration.

    Args:
        metric_config: Either a string (single metric) or list of metrics (priority order)
        available_metrics: Set or list of metrics that are actually available in the data

    Returns:
        str: The first metric from metric_config that exists in available_metrics
        None: If no matching metric found
    """
    if metric_config is None:
        return None

    # Handle single string metric
    if isinstance(metric_config, str):
        return metric_config if metric_config in available_metrics else None

    # Handle list of metrics - return first one that exists
    if isinstance(metric_config, list):
        for metric in metric_config:
            if metric in available_metrics:
                return metric
        # If none found but list is non-empty, return first as fallback
        return metric_config[0] if metric_config else None

    return None


def normalize_metric_config(metric_config):
    """Normalize metric config to always return a list for consistent handling.

    Args:
        metric_config: Either a string or list of metrics

    Returns:
        list: List of metric names
    """
    if metric_config is None:
        return []
    if isinstance(metric_config, str):
        return [metric_config]
    if isinstance(metric_config, list):
        return metric_config
    return []


def find_closest_number(x, arr):
    """Find the closest number to x in arr"""
    return arr[min(range(len(arr)), key=lambda i: abs(arr[i] - x))]


def override_dicts(dict1, dict2):
    """Recursively override the values of dict1 with the values from dict2 for keys present in dict1.

    Args:
        dict1 (dict): The original dictionary.
        dict2 (dict): The dictionary with overriding values.

    Returns:
        dict: The updated dictionary with values from dict2 for keys that exist in dict1.

    Example:
        dict1 = {'a':1, 'b':{'x':2, 'y':3}, 'c':{'a':1}}
        dict2 = {'a':3, 'b':{'x':4,'z':5}, 'c':3, 'd': 4}
        dict1 = override_dicts(dict1, dict2)
        {'a': 3, 'b': {'x': 4, 'y': 3}, 'c': {'a': 1}}
    """
    for key, value in dict1.items():
        if key in dict2:
            # If both dict1 and dict2 have a nested dictionary at the same key, recurse
            if isinstance(value, dict) and isinstance(dict2[key], dict):
                override_dicts(value, dict2[key])
            else:
                # Override the value in dict1 with the value from dict2
                dict1[key] = dict2[key]
    return dict1


def merge_nested_dicts(dict1, dict2):
    """Merge two nested dictionaries. Overwrite values of dict1 where keys are the same and add new keys.

    Args:
        dict1 (dict): The first nested dictionary.
        dict2 (dict): The second nested dictionary.

    Returns:
        dict: The merged nested dictionary.

    Example:
        dict1 = {'a':1,'b':2, 'c':{'a':1}}
        dict2 = {'a':3,'b':{'c':1,'d':'2'}, 'c':3}
        dict1 = merge_nested_dicts(dict1, dict2)
        {'a': 3, 'b': {'c': 1, 'd': '2'}, 'c': 3}
    """
    merged_dict = dict1.copy()

    for key, value in dict2.items():
        if key in merged_dict and isinstance(value, dict) and isinstance(merged_dict[key], dict):
            # Recursively merge nested dictionaries
            merged_dict[key] = merge_nested_dicts(merged_dict[key], value)
        else:
            # Overwrite values or add new keys
            merged_dict[key] = value

    return merged_dict


def _get_key_from_secrets_file():
    """Try to read NGC API key from the mounted secrets.json file."""
    secrets_path = os.getenv("VAULT_SECRET_FILE", "/var/secrets/secrets.json")
    try:
        if os.path.isfile(secrets_path):
            with open(secrets_path, "r", encoding="utf-8") as f:
                secrets = json.load(f)
            key = secrets.get("ngc_api_key", "")
            if key:
                logger.info("Obtained NGC API key from %s", secrets_path)
                return key
    except Exception as e:
        logger.warning("Failed to read NGC key from %s: %s", secrets_path, e)
    return ""


def get_admin_key(legacy_key=False):
    """Get admin api key from k8s secret, secrets file, or environment."""
    try:
        try:
            if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                key = os.environ.get('NGC_KEY')
                if key:
                    return key
                config.load_kube_config()
            else:
                config.load_incluster_config()
                secret = client.CoreV1Api().read_namespaced_secret("adminclustersecret", "default")
        except client.exceptions.ApiException as e:
            if e.status == 404:
                logger.warning("Secret 'adminclustersecret' not found in 'default' namespace.")
                logger.info("Falling back to bcpclustersecret")
                secret = get_bcp_key()
                if not secret:
                    return _get_key_from_secrets_file()
                return secret
            logger.error("Failed to obtain secret from k8s: %s", e)
            return _get_key_from_secrets_file()

        encoded_key = base64.b64decode(next(iter(secret.data.values())))
        key = json.loads(encoded_key)["auths"]["nvcr.io"]["password"]

        return key
    except Exception as e:
        logger.error("Failed to obtain api key from k8s: %s", e)
        return _get_key_from_secrets_file()


def get_bcp_key():
    """Get bcp api key from k8s secret"""
    try:
        config.load_incluster_config()
        # TODO: Use a better way to get the secret for various deployments
        try:
            secret = client.CoreV1Api().read_namespaced_secret("bcpclustersecret", "default")
        except client.exceptions.ApiException as e:
            if e.status == 404:
                logger.info(
                    "Secret 'bcpclustersecret' not found in 'default' namespace. "
                    "Falling back to imagepullsecret"
                )
                secret = client.CoreV1Api().read_namespaced_secret(
                    os.getenv('IMAGEPULLSECRET', default='imagepullsecret'),
                    "default"
                )
            else:
                logger.error(f"Failed to obtain secret from k8s: {e}")
                return ""

        encoded_key = base64.b64decode(next(iter(secret.data.values())))
        key = json.loads(encoded_key)["auths"]["nvcr.io"]["password"]

        return key
    except Exception as e:
        logger.error(f"Failed to obtain api key from k8s: {e}")
        return ""


def find_differences(dict1, dict2):
    """Finds differences in values for keys that exist in both nested dictionaries.

    Args:
        dict1 (dict): The first dictionary to compare.
        dict2 (dict): The second dictionary to compare.

    Returns:
        dict: A dictionary containing the keys that have differing values in `dict1` and `dict2`.
              - If the differing values are nested dictionaries, the returned dictionary will
                include the nested differences.
              - For non-dictionary values, the differing values will be returned as a tuple (value1, value2).

    Example:
        dict1 = {
            "a": {"x": 1, "y": 2, "z": 3},
            "b": {"p": 4, "q": 5},
            "c": 10,
        }

        dict2 = {
            "a": {"x": 1, "y": 20, "z": 3},
            "b": {"p": 40, "q": 5},
            "c": 10,
            "d": {"extra": 99},
        }

        result = find_differences(dict1, dict2)
        # Output: {"a": {"y": 2)}, "b": {"p": 4}}
    """
    differences = {}
    for key, value1 in dict1.items():
        if key in dict2:  # Check if key exists in both dictionaries
            value2 = dict2[key]
            # Check if the values are both dictionaries (nested)
            if isinstance(value1, dict) and isinstance(value2, dict):
                # Recursive call for nested dictionaries
                nested_diff = find_differences(value1, value2)
                if nested_diff:  # Only add if there are differences
                    differences[key] = nested_diff
            # If not dictionaries, compare values directly
            elif value2 and value1 != value2:
                differences[key] = value1
    return differences


def check_and_convert(user_spec, schema_spec):
    """Convert data type of user_spec value to match with data type of schema_spec value

    Check if nested keys in user_spec are present in schema_spec. If present, ensure that the type and value
    of each key in user_spec matches the corresponding key in schema_spec. If the type mismatch is found,
    attempt to convert the value to the correct type based on the schema_spec.

    Args:
        user_spec (dict): The user-specified dictionary to be validated and converted.
        schema_spec (dict): The schema specification dictionary against which user_spec will be validated.
    """
    for key, value in schema_spec.items():
        if key in user_spec:
            if isinstance(value, dict):
                if not isinstance(user_spec[key], dict):
                    # Convert to dictionary if necessary
                    try:
                        user_spec[key] = dict(value)
                    except ValueError:
                        pass  # Unable to convert, leave unchanged
                else:
                    # Recursively check nested dictionaries
                    check_and_convert(user_spec[key], value)
            elif isinstance(value, list):
                if not isinstance(user_spec[key], list):
                    # Convert to list if necessary
                    try:
                        user_spec[key] = list(value)
                    except ValueError:
                        pass  # Unable to convert, leave unchanged
                else:
                    # Check each element of the list
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            # Recursively check nested dictionaries
                            if i < len(user_spec[key]):
                                check_and_convert(user_spec[key][i], item)
                        elif i < len(user_spec[key]) and not isinstance(user_spec[key][i], type(item)):
                            # Convert type if necessary
                            try:
                                user_spec[key][i] = type(item)(user_spec[key][i])
                            except ValueError:
                                pass  # Unable to convert, leave unchanged
            else:
                # Convert type if necessary
                if not isinstance(user_spec[key], type(value)):
                    try:
                        user_spec[key] = type(value)(user_spec[key])
                    except ValueError:
                        pass  # Unable to convert, leave unchanged


def get_ngc_artifact_base_url(ngc_path):
    """Construct NGC artifact base url from ngc_path provided"""
    ngc_configs = ngc_path.split('/')
    org = ngc_configs[0]
    model, version = ngc_configs[-1].split(':')
    team = ""
    if len(ngc_configs) == 3:
        team = ngc_configs[1]
    base_url = "https://api.ngc.nvidia.com"
    url_substring = ""
    if team and team != "no-team":
        url_substring = f"team/{team}"
    endpoint = base_url + f"/v2/org/{org}/{url_substring}/models/{model}/{version}".replace("//", "/")
    return endpoint, model, version


def send_get_request_with_retry(endpoint, headers, retry=0):
    """Send admin GET request with retries"""
    try:
        r = requests.get(endpoint, headers=headers, timeout=120)
    except Exception as e:
        logger.error("Exception caught during sending get request in utils: %s", e)
        raise e
    if not r.ok:
        if retry < NUM_OF_RETRY:
            logger.info("Retrying %d time(s) to GET %s.", retry, endpoint)
            return send_get_request_with_retry(endpoint, headers, retry + 1)
        logger.error("Request to GET %s failed after %d retries.", endpoint, retry)
    return r


def send_delete_request_with_retry(endpoint, headers, retry=0):
    """Send DELETE request with retries"""
    try:
        r = requests.delete(endpoint, headers=headers, timeout=120)
    except Exception as e:
        logger.error("Exception caught during sending delete request in retry: %s", e)
        raise e
    if not r.ok:
        if retry < NUM_OF_RETRY:
            logger.info("Retrying %d time(s) to DELETE %s.", retry, endpoint)
            return send_delete_request_with_retry(endpoint, headers, retry + 1)
        logger.error("Request to DELETE %s failed after %d retries.", endpoint, retry)
    return r


class ErrorResponse:
    """Custom error response object"""

    def __init__(self, status_code, message="An error occurred"):
        """Initialize the ErrorResponse object.

        Args:
            status_code (int): The HTTP status code to represent the error.
            message (str, optional): A message providing additional context about the error.
                                     Defaults to "An error occurred".
        """
        self.status_code = status_code
        self.ok = False
        self.message = message

    def json(self):
        """Return a JSON-like dictionary for the error response.

        Returns:
            dict: A dictionary containing the status code and error message.
        """
        return {
            "status_code": self.status_code,
            "message": self.message
        }

    def __repr__(self):
        """Return a string representation of the ErrorResponse object.

        Returns:
            str: A string that shows the status code and error message.
        """
        return f"<ErrorResponse(status_code={self.status_code}, message={self.message})>"


def retry_method(response=False):
    """Retry Cloud storage methods for NUM_RETRY times"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(NUM_OF_RETRY):
                try:
                    # Call the actual function
                    result = func(*args, **kwargs)
                    # If response handling is enabled, check for `ok` status
                    if response:
                        if result.ok:
                            return result
                        logger.error("Response not OK (attempt %d): %s", attempt + 1, result.status_code)
                        time.sleep(30)  # Wait between retries
                    else:
                        # If no response-based retry, just return the result
                        return result
                except Exception as e:
                    # Log or handle the exception
                    logger.error("Exception in %s on attempt %d: %s", func.__name__, attempt + 1, e)
                    time.sleep(30)  # Wait between retries

            # After retries, return error response or raise an error based on decorator parameter
            if response:
                return ErrorResponse(
                    status_code=404,
                    message=f"Failed to execute {func.__name__} after {NUM_OF_RETRY} retries"
                )
            raise ValueError(f"Failed to execute {func.__name__} after {NUM_OF_RETRY} retries")
        return wrapper
    return decorator


def get_default_lock_file_path(filepath):
    """Returns the default lock file path"""
    return os.path.splitext(filepath)[0] + "_lock.lock"


def create_lock(filepath, existing_lock=None):
    """Creates a lock file"""
    if existing_lock:
        return existing_lock
    lock_file = get_default_lock_file_path(filepath)
    if not os.path.exists(lock_file):
        with open(lock_file, "w", encoding="utf-8") as _:
            pass
    return FileLock(lock_file)


def safe_get_file_modified_time(filepath):
    """Returns the modified time of the file"""
    lock_file = get_default_lock_file_path(filepath)

    if not os.path.exists(lock_file):
        with open(lock_file, "w", encoding="utf-8") as _:
            pass

    with FileLock(lock_file):
        return os.path.getmtime(filepath)


def __convert_keys_to_str(data):
    if isinstance(data, dict):
        return {
            str(key) if isinstance(key, int) else key: __convert_keys_to_str(value)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [__convert_keys_to_str(item) for item in data]
    if isinstance(data, np.float64):
        return float(data)
    if isinstance(data, Enum):
        return data.value
    return data


def load_file(filepath, attempts=3, file_type="json"):
    """Unsynchronized file load"""
    assert file_type in ("json", "yaml"), f"Unsupported file type '{file_type}'. Only json and yaml are supported."
    if attempts == 0:
        return {}

    if not os.path.exists(filepath):
        logger.warning("File trying to read doesn't exists: %s", filepath)
        return {}

    try:
        data = {}
        if file_type == "json":
            with open(filepath, "r", encoding="utf-8") as f:
                json_data = f.read()
                data = orjson.loads(json_data)
        elif file_type == "yaml":
            with open(filepath, "r", encoding="utf-8") as f:
                yaml = ruamel.yaml.YAML(typ='safe')
                data = yaml.load(f)
        return data
    except Exception as e:
        logger.error("Exception thrown in load_file: %s", str(e))
        data = {}
        logger.warning("Data not in %s loadable format: %s", file_type, filepath)
        with open(filepath, "r", encoding='utf-8') as f:
            file_lines = f.readlines()
            logger.warning("Data: \n%s", file_lines)
        return load_file(filepath, attempts - 1, file_type=file_type)


def safe_load_file(filepath, existing_lock=None, attempts=3, file_type="json"):
    """Loads the json file with synchronization"""
    assert file_type in ("json", "yaml"), f"Unsupported file type '{file_type}'. Only json and yaml are supported."
    if attempts == 0:
        return {}

    if not os.path.exists(filepath):
        logger.warning("File trying to read doesn't exists: %s", filepath)
        return {}

    lock = create_lock(filepath, existing_lock=existing_lock)
    try:
        with lock:
            data = {}
            if file_type == "json":
                with open(filepath, "r", encoding="utf-8") as f:
                    json_data = f.read()
                    data = orjson.loads(json_data)
            elif file_type == "yaml":
                with open(filepath, "r", encoding="utf-8") as f:
                    yaml = ruamel.yaml.YAML(typ='safe')
                    data = yaml.load(f)
            return data
    except Exception as e:
        logger.error("Exception thrown in safe_load_file: %s", str(e))
        data = {}
        logger.warning("Data not in %s loadable format: %s", file_type, filepath)
        if not os.path.exists(filepath):
            logger.warning("File trying to read doesn't exists: %s", filepath)
            return {}
        with open(filepath, "r", encoding='utf-8') as f:
            file_lines = f.readlines()
            logger.warning("Data: \n%s", file_lines)
        return safe_load_file(filepath, lock, attempts - 1, file_type=file_type)


def safe_dump_file(filepath, data, existing_lock=None, file_type="json"):
    """Dumps the json file"""
    assert file_type in ("json", "yaml", "protobuf"), (
        f"Unsupported file type '{file_type}'. Only json, yaml, and protobuf are supported."
    )
    parent_folder = os.path.dirname(filepath)
    if not os.path.exists(parent_folder):
        logger.warning("Parent folder %s doesn't exists yet", parent_folder)
        return

    lock = create_lock(filepath, existing_lock=existing_lock)

    with lock:
        tmp_file_path = filepath.replace(f".{file_type}", f"_tmp.{file_type}")
        if file_type == "json":
            json_data = orjson.dumps(__convert_keys_to_str(data))
            with open(tmp_file_path, "w", encoding="utf-8") as f:
                f.write(json_data.decode('utf-8'))
        elif file_type == "yaml":
            with open(tmp_file_path, "w", encoding="utf-8") as f:
                yaml = ruamel.yaml.YAML()
                yaml.dump(data, f)
        elif file_type == "protobuf":
            with open(tmp_file_path, "w", encoding='utf-8') as f:
                f.write(data)
        if os.path.exists(tmp_file_path):
            os.rename(tmp_file_path, filepath)


class DataMonitorLogTypeEnum(str, Enum):
    """Class defining data monitor log type."""

    api = "API"
    tao_job = "TAO_JOB"
    tao_experiment = "TAO_EXPERIMENT"
    tao_dataset = "TAO_DATASET"


def log_monitor(log_type, log_content):
    """Log format information for data monitor servers like Kibana.

    Print the log in a fixed format so that it would be easier for the log monitor
    to analyse or visualize the log like how many times the specific user calls
    the specific API.
    """
    monitor_type = os.getenv("SERVER_MONITOR_TYPE", "DATA_COLLECTION")
    print_string = f"[{monitor_type}][{log_type}] {log_content}"
    logger.info(print_string)


def log_api_error(user_id, org_name, schema_dict, log_type, action, from_ui=False):
    """Log the api call error."""
    error_desc = schema_dict.get("error_desc", None)
    error_code = schema_dict.get("error_code", None)
    log_content = (
        f"user_id:{user_id}, org_name:{org_name}, from_ui:{from_ui}, "
        f"action:{action}, error_code:{error_code}, error_desc:{error_desc}"
    )
    log_monitor(log_type=log_type, log_content=log_content)


def print_start_script_path():
    """Print the path to the start script."""
    print(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app_start.sh'))


def print_nginx_conf_path():
    """Print the path to the nginx.conf file."""
    print(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nginx.conf'))
