#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create metadata info for all base experiments supported"""
import argparse
import ast
import csv
import datetime
import json
import operator
import os
import uuid
import random
import requests
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from packaging import version
from enum import Enum
import logging

if os.getenv("AIRGAPPED_MODE", "False").lower() == "false":
    from nvidia_tao_core.microservices.utils.mongo_utils import MongoHandler
else:
    MongoHandler = None  # type: ignore
from nvidia_tao_core.microservices.utils.ngc_utils import get_ngc_token_from_api_key
from nvidia_tao_core.microservices.utils.core_utils import (
    read_network_config,
    get_admin_key,
    safe_load_file,
    get_orchestration_network_from_microservices
)
from nvidia_tao_core.microservices.handlers.cloud_handlers.huggingface import (
    download_huggingface_model as download_hf_model
)
from nvidia_tao_core.microservices.constants import TAO_NETWORKS
from nvidia_tao_core.microservices.enum_constants import (
    BaseExperimentTask,
    BaseExperimentDomain,
    BaseExperimentBackboneClass,
    BaseExperimentBackboneType,
    BaseExperimentLicense,
    ExperimentNetworkArch
)

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)

base_exp_uuid = "00000000-0000-0000-0000-000000000000"

PTM_PULL = os.getenv("PTMPULL", "True").lower()
TIMEOUT = 120

ngc_api_base_url = "https://api.ngc.nvidia.com/v2"


class BaseExperimentMetadata:
    """Base Experiment Metadata class"""

    def __init__(
        self,
        shared_folder_path: str,
        org_teams: str,
        ngc_key: str = None,
        override: bool = False,
        dry_run: bool = False,
        use_csv: bool = False,
        use_both: bool = False,
        model_names: str = None,
        csv_path: str = None,
    ):
        """Initialize Base Experiment Metadata class

        Args:
            shared_folder_path (str): Root path for base experiments
            org_teams (str): Organization and team names. Each pair of org/team separated by a comma.
            ngc_key (str, optional): NGC Personal Key. Defaults to None.
            override (bool, optional): Override existing base experiments. Defaults to False.
            dry_run (bool, optional): Dry run mode. Defaults to False.
            use_csv (bool, optional): Use predefined CSV file for models instead of NGC discovery. Defaults to False.
            use_both (bool, optional): Use both CSV file and NGC auto-discovery. Defaults to False.
            model_names (str, optional): Comma-separated list of model names/entrypoints to download. Defaults to None.
            csv_path (str, optional): Path to CSV file. Defaults to pretrained_models.csv next to this module.
        """
        self.shared_folder_path = shared_folder_path
        self.ngc_key = (os.getenv("NGC_API_KEY") or os.getenv("NGC_KEY") or ngc_key or "").strip() or None
        if self.ngc_key is None and not (use_csv and not use_both):
            self.ngc_key = get_admin_key()
        self.ngc_key = self.ngc_key or ""
        self.airgapped = os.getenv("AIRGAPPED_MODE", "False").lower() == "true"
        self.use_csv = use_csv
        self.use_both = use_both
        self.csv_path = csv_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models.csv")
        self.model_names = model_names.split(",") if model_names else None
        self.org_team_list = self.prepare_org_team(org_teams)
        self.override = override
        self.metadata: dict = {}
        self.dry_run = dry_run
        self._cached_tao_version: str | None = None
        self._ngc_clients_cache: dict = {}  # Cache for NGC clients per org/team
        self._network_config_cache: dict = {}  # Cache for network configs
        self._max_workers = int(os.getenv("PTM_MAX_WORKERS", "10"))  # Configurable parallelism

        if self.override and self.dry_run:
            raise ValueError("Cannot use both `--override` and `--dry-run` flags together!")

        if self.use_csv and self.use_both:
            raise ValueError(
                "Cannot use both `--use-csv` and `--use-both` flags together! Use `--use-both` for CSV + NGC discovery."
            )

        # set default uuids
        self.base_exp_uuid = uuid.UUID(base_exp_uuid)
        self.ptm_uuid = uuid.UUID(base_exp_uuid)

        # create rootdir and metadata file path
        self.rootdir = os.path.abspath(
            os.path.join(self.shared_folder_path, "orgs", str(self.base_exp_uuid), "experiments", str(self.ptm_uuid))
        )
        if self.airgapped:
            self.rootdir = self.shared_folder_path
        self.metadata_file = os.path.join(self.rootdir, "ptm_metadatas.json")

        # create rootdir if it doesn't exist
        os.makedirs(self.rootdir, exist_ok=True)

        # set tao version and comparison operators
        self.tao_version = None  # type: version.Version
        self.comparison_operators = {
            "<=": operator.le,
            "<": operator.lt,
            ">": operator.gt,
            ">=": operator.ge,
            "==": operator.eq,
            "!=": operator.ne,
        }

    def get_tao_version(self) -> str:
        """Return current version of Nvidia TAO.

        Priority:
          1. installed package metadata (reflects actual pip-installed version)
          2. nvidia_tao_core.version.__version__ (source module, if available)
          3. $TAO_TOOLKIT_VERSION (image env var, may be stale)
          4. hard-coded fallback
        """
        if self._cached_tao_version:
            return self._cached_tao_version

        try:
            from importlib.metadata import version as pkg_version
            self._cached_tao_version = pkg_version("nvidia-tao-core")
            return self._cached_tao_version
        except Exception:
            pass

        try:
            from nvidia_tao_core.version import __version__ as src_version
            self._cached_tao_version = src_version
            return self._cached_tao_version
        except Exception:
            pass

        env_ver = os.getenv("TAO_TOOLKIT_VERSION")
        if env_ver:
            self._cached_tao_version = env_ver
            return env_ver

        self._cached_tao_version = "6.0.0"
        return self._cached_tao_version

    def get_ngc_client(self, org: str, team: str, ngc_key: str):
        """Get or create a cached NGC client for the given org/team

        Returns:
            Client: Configured NGC client, unconfigured client, or None for failed auth
            None: If authentication failed (also cached to prevent retry)
        """
        client_key = (org, team, ngc_key)

        if client_key not in self._ngc_clients_cache:
            from ngcsdk import Client  # pylint: disable=C0415
            client = Client()
            try:
                client.configure(api_key=ngc_key, org_name=org, team_name=team)
                self._ngc_clients_cache[client_key] = client
                logger.info(f"Created and cached NGC client for org: {org}, team: {team}")
            except Exception as e:
                if "Invalid org" in str(e) or "Invalid team" in str(e):
                    logger.warning(
                        f"Can't validate the passed NGC KEY for Org {org}, team {team}, "
                        "going to try download without configuring credentials"
                    )
                    self._ngc_clients_cache[client_key] = client
                elif ngc_key and ngc_key.startswith("nvapi"):
                    # nvapi: try configure with api_key only so SDK derives org from key
                    try:
                        client.configure(api_key=ngc_key)
                        self._ngc_clients_cache[client_key] = client
                        logger.info("Created and cached NGC client for nvapi key (org/team from key)")
                    except Exception as e2:
                        logger.warning(
                            "NGC client configure failed for nvapi key; attempting with unconfigured client: %s",
                            e2,
                        )
                        self._ngc_clients_cache[client_key] = client
                else:
                    logger.error(f"Can't configure the passed NGC KEY for Org {org}, team {team}")
                    self._ngc_clients_cache[client_key] = None
                    return None

        return self._ngc_clients_cache[client_key]

    def get_network_config_cached(self, network_arch: str):
        """Get network config with caching to avoid repeated file reads

        Tries to load the config in the following order:
        1. Direct network config for network_arch
        2. Orchestration network config (if network_arch is a microservices network)
        3. Default fallback config

        Args:
            network_arch: Network architecture name

        Returns:
            dict: Network configuration
        """
        if network_arch not in self._network_config_cache:
            # Try loading direct network config first
            try:
                self._network_config_cache[network_arch] = read_network_config(network_arch)
                logger.debug(f"Cached network config for {network_arch} (direct)")
                if not self._network_config_cache[network_arch]:
                    logger.debug(f"Could not load direct network config for {network_arch}")
                    # Try to find orchestration network and load its config
                    orchestration_network = get_orchestration_network_from_microservices(network_arch)
                    if orchestration_network != network_arch:
                        self._network_config_cache[network_arch] = read_network_config(orchestration_network)
                        logger.info(
                            f"Cached network config for {network_arch} using "
                            f"orchestration network: {orchestration_network}"
                        )
                    else:
                        # No orchestration mapping found, use default
                        raise Exception("No orchestration network mapping found")
            except Exception as e:
                logger.warning(
                    f"Could not load network config for {network_arch} or its orchestration network: {e}"
                )
                # Cache default config to avoid repeated failures
                self._network_config_cache[network_arch] = {
                    "api_params": {
                        "dataset_type": "object_detection",
                        "actions": ["train", "evaluate", "inference"],
                        "accepted_ds_intents": ["training", "evaluation"],
                        "formats": ["coco"]
                    }
                }
        return self._network_config_cache[network_arch]

    def check_version_compatibility(self, version_list: list):
        """Check if the current TAO version is compatible with the provided version list"""
        if self.tao_version is None:
            try:
                self.tao_version = version.Version(self.get_tao_version())
            except FileNotFoundError as e:
                logger.warning("Skipping TAO Version Check!!! Failed to get current NVTL version! >>> %s", e)
                return True
        version_ok = True
        for version_str in version_list:
            ver = version_str.strip(r"<>=! ")
            op = version_str[: -len(ver)].strip()
            if op not in self.comparison_operators:
                raise ValueError(f"Invalid version comparison operator: {op}")
            version_ok = self.comparison_operators[op](self.tao_version, version.Version(ver))
            if not version_ok:
                break
        return version_ok

    def validate_network_endpoint(self, network_arch: str):
        """Validate that network architecture is a valid endpoint (not orchestration-only).

        Args:
            network_arch: Network architecture name to validate

        Raises:
            ValueError: If network is orchestration-only without a dedicated config file

        A valid endpoint must either:
        1. Have its own config file (e.g., visual_changenet_classify, visual_changenet_segment)
        2. Be a microservices network that maps to a different orchestration network
           (e.g., auto_label -> object_detection)

        Invalid endpoints are orchestration-only networks without config files that map to
        themselves (e.g., visual_changenet)
        """
        direct_config = read_network_config(network_arch)
        if not direct_config:
            # Check if this is a microservices network that maps to a different orchestration network
            orchestration_network = get_orchestration_network_from_microservices(network_arch)
            if orchestration_network == network_arch:
                # This network doesn't have a config and doesn't map to a different orchestration network
                # It's an orchestration-only network that shouldn't be used as an endpoint
                raise ValueError(
                    f"Entrypoint `{network_arch}` is not supported. "
                    f"This is an orchestration-only network without a dedicated config file."
                )

    def get_ngc_token(self, org: str = "", team: str = ""):
        """Authenticate to NGC"""
        # Get the NGC login token: PTM_API_KEY env, then --ngc-key, then secrets file (k8s/infra only)
        ngc_api_key = os.getenv("PTM_API_KEY")
        ngc_token = ""
        if not ngc_api_key and self.ngc_key:
            ngc_api_key = self.ngc_key
        if not ngc_api_key:
            secrets_path = os.getenv("VAULT_SECRET_FILE", "/var/secrets/secrets.json")
            if os.path.isfile(secrets_path):
                with open(secrets_path, "r", encoding="utf-8") as f:
                    secrets = json.load(f)
                ngc_api_key = secrets.get("ptm_api_key") or secrets.get("ngc_api_key")

        if ngc_api_key:
            # nvapi-* keys are used directly as Bearer; legacy keys are exchanged for a JWT
            if ngc_api_key.startswith("nvapi"):
                return ngc_api_key, ngc_api_key
            ngc_token = get_ngc_token_from_api_key(ngc_api_key, org, team)
            return ngc_api_key, ngc_token
        if self.ngc_key and self.ngc_key.startswith("nvapi"):
            return self.ngc_key, self.ngc_key
        raise ValueError(
            'Credentials error: Invalid NGC_PERSONAL_KEY, NGC_API_KEYs are no longer valid, '
            'generate a personal key with Cloud Functions, NGC Catalog and Private registry services '
            'https://org.ngc.nvidia.com/setup/personal-keys'
        )

    def prepare_org_team(self, org_teams: str):
        """Prepare org team list"""
        if self.use_csv and not self.use_both:
            # In CSV-only mode, we don't need org/team validation since we specify exact models
            logger.info("> CSV mode: will process specific models from CSV file.")
            org_team_list = []
        elif self.use_both:
            # In both mode, we need org/team for NGC discovery but also use CSV
            logger.info("> Both mode: will process models from CSV file AND NGC discovery.")
            if org_teams:
                org_team_list = []
                for org_team in org_teams.split(","):
                    org_team = org_team.replace("/no-team", "")
                    org = org_team
                    team = ""
                    if "/" in org_team:
                        org, team = org_team.split("/")
                    org_team_list.append((org, team))
            else:
                org_team_list = self.get_org_teams()
        elif org_teams:
            org_team_list = []
            for org_team in org_teams.split(","):
                org_team = org_team.replace("/no-team", "")
                org = org_team
                team = ""
                if "/" in org_team:
                    org, team = org_team.split("/")
                org_team_list.append((org, team))
        else:
            org_team_list = self.get_org_teams()
        return org_team_list

    def get_org_teams(self):
        """Get all orgs and teams for the user"""
        logger.info("--------------------------------------------------------")
        logger.info("Getting accessible org/team for the provided NGC Personal key")
        logger.info("--------------------------------------------------------")
        _, ngc_token = self.get_ngc_token()
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {ngc_token}",
            "Accept-Encoding": "identity"
        }
        url = f"{ngc_api_base_url}/orgs"
        try:
            response = requests.get(url, headers=headers, params={"page-size": 1000}, timeout=TIMEOUT)
        except Exception as e:
            logger.error("Exception caught during getting orgs info: %s", e)
            raise e
        if response.status_code != 200:
            raise ValueError(response.json())
        orgs = [org["name"] for org in response.json()["organizations"]]
        org_teams = []
        # get team for each org
        for org in orgs:
            url = f"{ngc_api_base_url}/org/{org}/teams"
            try:
                response = requests.get(url, headers=headers, params={"page-size": 1000}, timeout=TIMEOUT)
            except Exception as e:
                logger.error("Exception caught during getting teams info: %s", e)
                raise e
            if response.status_code != 200:
                logger.error(response.json())
                continue
            teams = [team["name"] for team in response.json()["teams"]]
            logger.info(f"{org}: {teams}")
            org_teams.extend([(org, team) for team in teams])
            org_teams.append((org, ""))
        logger.info("nvidia: ['tao']")
        org_teams.append(("nvidia", "tao"))

        logger.info(f"Created {len(org_teams)} org/team pairs for the provided NGC Personal key ")
        logger.info("--------------------------------------------------------")
        return org_teams

    @staticmethod
    def split_ngc_path(ngc_path):
        """Split ngc path into org, team and model name, model version"""
        path_split = ngc_path.replace("/no-team", "").split("/")
        if len(path_split) == 3:
            org, team, model_name = path_split
        elif len(path_split) == 2:
            org, model_name = path_split
            team = ""
        else:
            raise ValueError(f"Invalid ngc_path: {ngc_path}")
        if ":" in model_name:
            model_name, model_version = model_name.split(":")
        else:
            model_version = ""
        return org, team, model_name, model_version

    @staticmethod
    def get_model_info_from_ngc(
        ngc_token: str, org: str, team: str, model_name: str, model_version: str, file: str = ""
    ):
        """Get model info from NGC"""
        url = ngc_api_base_url
        if team:
            url += f"/org/{org}/team/{team}/models/{model_name}/versions/{model_version}"
        else:
            url += f"/org/{org}/models/{model_name}/versions/{model_version}"
        if file:
            url += f"/files/{file}"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {ngc_token}",
            "Accept-Encoding": "identity"
        }
        try:
            response = requests.get(url, headers=headers, params={"page-size": 1000}, timeout=TIMEOUT)
        except Exception as e:
            logger.error("Exception caught during getting model info: %s", e)
            raise e
        if response.status_code != 200:
            raise ValueError(
                f"Failed to get model info for {model_name}:{model_version} ({response.status_code} {response.reason})"
            )
        return response.json()

    def get_model_info_batch(self, ngc_paths_with_tokens: list):
        """Fetch model info from NGC in parallel for multiple models

        Args:
            ngc_paths_with_tokens: List of tuples (ngc_path, ngc_token)

        Returns:
            dict: Mapping of ngc_path to model_info
        """
        model_info_batch = {}

        def fetch_single_model(ngc_path, ngc_token):
            """Helper to fetch a single model's info"""
            try:
                org, team, model_name, model_version = self.split_ngc_path(ngc_path)
                info = self.get_model_info_from_ngc(ngc_token, org, team, model_name, model_version)
                return ngc_path, info, None
            except Exception as e:
                return ngc_path, None, e

        # Process in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(fetch_single_model, ngc_path, token): ngc_path
                for ngc_path, token in ngc_paths_with_tokens
            }

            for future in as_completed(futures):
                ngc_path, info, error = future.result()
                if error:
                    logger.error(f"Failed to fetch model info for {ngc_path}: {error}")
                else:
                    model_info_batch[ngc_path] = info
                    logger.debug(f"Successfully fetched model info for {ngc_path}")

        return model_info_batch

    def load_base_experiments_from_csv(self) -> dict:
        """Get base experiments from CSV file

        CSV format: display_name, model_path, network_arch[, is_backbone]
        The is_backbone column is optional for backwards compatibility.

        Supports both NGC and Hugging Face models.

        Args:
            display_name: Human-readable name for the model
            model_path: NGC path or HF model path (with ngc:// or hf_model:// prefix)
            network_arch: Network architecture identifier
            is_backbone: Optional boolean (True/False/empty) indicating if model is a backbone
        """
        base_experiments: dict[str, dict] = {}
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row_num, row in enumerate(reader, 2):  # Start from 2 since we skip header
                try:
                    # Handle both 3-column and 4-column CSV formats for backwards compatibility
                    if len(row) == 3:
                        display_name, model_path, network_arch = row
                        is_backbone = None  # Use default value
                    elif len(row) == 4:
                        display_name, model_path, network_arch, is_backbone_str = row
                        # Parse is_backbone string to boolean, handling empty values
                        if is_backbone_str.strip().lower() in ['true', '1', 'yes']:
                            is_backbone = True
                        elif is_backbone_str.strip().lower() in ['false', '0', 'no']:
                            is_backbone = False
                        else:
                            is_backbone = None  # Use default value for empty or invalid strings
                    else:
                        logger.error(f"CSV row {row_num}: Invalid number of columns. Expected 3 or 4, got {len(row)}")
                        continue

                    if self.model_names and network_arch not in self.model_names:
                        logger.info(f"Skipping {model_path} - not in requested model names: {self.model_names}")
                        continue

                    # Validate that the network architecture is a valid endpoint
                    try:
                        self.validate_network_endpoint(network_arch)
                    except ValueError as e:
                        logger.error(f"CSV row {row_num}: {e}")
                        continue

                    # Parse model path to determine source type
                    source_type, cleaned_path = self.parse_model_path(model_path)

                    if source_type == "ngc":
                        # Handle NGC models (existing logic)
                        org, team, _, _ = self.split_ngc_path(cleaned_path)
                        ngc_key, _ = self.get_ngc_token(org, team)
                        self.add_experiment(
                            base_experiments, display_name, cleaned_path, network_arch,
                            ngc_key, source_type, is_backbone
                        )

                    elif source_type == "huggingface":
                        # Handle Hugging Face models
                        exp_id = str(uuid.uuid5(self.base_exp_uuid, f"hf:{cleaned_path}"))

                        # Validate network architecture for Hugging Face models
                        if network_arch not in ExperimentNetworkArch.__members__:
                            logger.warning(
                                f"CSV row {row_num}: Network architecture '{network_arch}' "
                                f"is not supported for Hugging Face model. "
                                f"Supported architectures: {', '.join(ExperimentNetworkArch.__members__)}"
                            )
                            logger.info(
                                f"Proceeding with custom network architecture '{network_arch}' for Hugging Face model"
                            )

                        # Download model in airgapped mode
                        spec_data = {}
                        if self.airgapped:
                            download_success = self.download_huggingface_model(cleaned_path, exp_id)
                            if not download_success:
                                logger.warning(f"Failed to download Hugging Face model: {cleaned_path}")
                                continue

                        # Create basic experiment info first
                        basic_experiment = {
                            "id": exp_id,
                            "name": display_name,
                            "ngc_path": cleaned_path,  # Store original path
                            "network_arch": network_arch,
                            "source_type": source_type,
                            "is_backbone": is_backbone,  # Store is_backbone value from CSV
                            "base_experiment_metadata": {
                                "spec_file_present": bool(spec_data),
                                "specs": spec_data
                            }
                        }

                        # Extract full metadata using network config
                        base_experiments[exp_id] = self.extract_huggingface_metadata(basic_experiment)

                    logger.info(
                        f"CSV: Processed model: {cleaned_path} with network_arch: {network_arch} "
                        f"(source: {source_type})"
                    )

                except Exception as e:
                    logger.error(f"Failed to process CSV row {row_num} '{row}': {e}")
                    continue

        # Download specs in parallel for all NGC experiments
        self.download_specs_parallel(base_experiments)

        return base_experiments

    def add_experiment(self, base_experiments, display_name, ngc_path, network_arch,
                       ngc_key, source_type="ngc", is_backbone=None):
        """Add experiment to the base experiments list with unique id (without downloading specs)"""
        hash_str = f"{ngc_path}:{network_arch}"
        exp_id = str(uuid.uuid5(self.base_exp_uuid, hash_str))

        # Just add the experiment metadata without downloading specs yet
        # Specs will be downloaded in parallel later
        base_experiments[exp_id] = {
            "id": exp_id,
            "name": display_name,
            "ngc_path": ngc_path,
            "network_arch": network_arch,
            "source_type": source_type,
            "is_backbone": is_backbone,
            "ngc_key": ngc_key,  # Store for later use
            "base_experiment_metadata": {
                "spec_file_present": False,  # Will be updated after parallel download
                "specs": {}
            }
        }

    def load_base_experiments_from_ngc(self, page_size: int = 1000) -> dict:
        """Get base experiments from NGC with parallel page fetching"""
        base_experiments: dict[str, dict] = {}

        def fetch_page(org, team, page_number, ngc_token, query, page_size):
            """Fetch a single page of NGC models"""
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {ngc_token}",
                "Accept-Encoding": "identity"
            }
            url = f"{ngc_api_base_url}/search/resources/MODEL"
            params = {
                "q": json.dumps(
                    {
                        "fields": ["resourceId", "name", "displayName", "orgName", "teamName"],
                        "page": page_number,
                        "pageSize": page_size,
                        "query": query,
                    }
                )
            }
            try:
                response = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
                return response.json()["results"]
            except Exception as e:
                logger.error(f"Exception caught during model search in page {page_number}: %s", e)
                return []

        for org, team in self.org_team_list:
            logger.info(f"Querying base experiments from '{org}{'/' + team if team else ''}'")
            ngc_key, ngc_token = self.get_ngc_token(org, team)
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {ngc_token}",
                "Accept-Encoding": "identity"
            }
            url = f"{ngc_api_base_url}/search/resources/MODEL"

            # Create the query to filter models and the required return fields
            query = f"resourceId:{org}/{team + '/' if team else ''}*"

            # Get the number of pages
            params = {"q": json.dumps({"pageSize": page_size, "query": query})}
            try:
                response = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
            except Exception as e:
                logger.error("Exception caught during model search: %s", e)
                raise e

            n_pages = response.json()["resultPageTotal"]
            logger.info(f"Found {n_pages} pages of models for {org}{'/' + team if team else ''}")

            # Fetch all pages in parallel
            all_results = []
            if n_pages > 0:
                with ThreadPoolExecutor(max_workers=min(self._max_workers, n_pages)) as executor:
                    futures = [
                        executor.submit(fetch_page, org, team, page_num, ngc_token, query, page_size)
                        for page_num in range(n_pages)
                    ]
                    for future in as_completed(futures):
                        results = future.result()
                        all_results.extend(results)

            # Process all models from all pages
            for model_list in all_results:
                for model in model_list["resources"]:
                    try:
                        model_meta = self.get_model_info_from_ngc(
                            ngc_token, model["orgName"], model.get("teamName", ""), model["name"], ""
                        )
                    except ValueError as e:
                        logger.error(e)
                        continue
                    if "modelVersions" in model_meta:
                        for model_version in model_meta["modelVersions"]:
                            if "customMetrics" in model_version:
                                ngc_path = f'{model["resourceId"]}:{model_version["versionId"]}'
                                for customMetrics in model_version["customMetrics"]:
                                    endpoints = []
                                    for key_value in customMetrics.get("attributes", []):
                                        if key_value["key"] == "endpoints":
                                            try:
                                                endpoints = ast.literal_eval(key_value["value"])
                                            except (SyntaxError, ValueError):
                                                logger.error(f"{key_value} not loadable by `ast.literal_eval`.")
                                    for network_arch in endpoints:
                                        # Filter by model names if specified
                                        if self.model_names and network_arch not in self.model_names:
                                            logger.debug(
                                                f"Skipping {network_arch} - not in requested model names: "
                                                f"{self.model_names}"
                                            )
                                            continue

                                        self.add_experiment(
                                            base_experiments,
                                            model.get("displayName", network_arch),
                                            ngc_path,
                                            network_arch,
                                            ngc_key,
                                            "ngc",
                                            None  # NGC discovery doesn't have CSV is_backbone value
                                        )

        # Download specs in parallel for all NGC experiments
        self.download_specs_parallel(base_experiments)

        return base_experiments

    def get_base_spec(self, ngc_path, exp_id, ngc_key):
        """Retrieves base experiment specs if present"""
        org, team, model, version = self.split_ngc_path(ngc_path)

        # Get cached NGC client
        clt = self.get_ngc_client(org, team, ngc_key)
        if clt is None:
            return {}
        # Check and download experiment.yaml file
        try:
            model_files = list(clt.registry.model.list_files(ngc_path))
            file_paths = list(map(lambda x: x.path, model_files))
            spec_file = "experiment.yaml"
            if spec_file in file_paths:
                dest_path = f"{self.rootdir}/{exp_id}/"
                os.makedirs(dest_path, exist_ok=True)
                clt.registry.model.download_version(ngc_path, destination=dest_path, file_patterns=[spec_file])
                spec_data = safe_load_file(dest_path + f"{model}_v{version}/experiment.yaml", file_type="yaml")
                if spec_data:
                    logger.info("Successfully got spec data for %s", ngc_path)
                else:
                    logger.error("Unable to get spec data for %s", ngc_path)
                return spec_data
        except Exception as e:
            logger.error("Unable to get spec data for %s", ngc_path)
            logger.error(e)
        return {}

    def download_specs_parallel(self, base_experiments: dict):
        """Download experiment specs in parallel for NGC models

        Args:
            base_experiments: Dictionary of experiments to download specs for
        """
        ngc_experiments = [
            (exp_id, exp_data)
            for exp_id, exp_data in base_experiments.items()
            if exp_data.get("source_type") == "ngc"
        ]

        if not ngc_experiments:
            return

        logger.info(f"Downloading specs for {len(ngc_experiments)} NGC experiments in parallel...")

        def download_single_spec(exp_id, exp_data):
            """Helper to download a single experiment spec"""
            ngc_path = exp_data["ngc_path"]
            ngc_key = exp_data["ngc_key"]

            try:
                if self.airgapped:
                    # In airgapped mode, download complete model
                    spec_data = {}
                    download_success = self.download_complete_model(ngc_path, exp_id, ngc_key)
                    if download_success:
                        # Try to get spec data if experiment.yaml exists
                        org, team, model, version = self.split_ngc_path(ngc_path)
                        spec_file_path = (
                            f"{self.rootdir}/{org}/{team}/{model}/{version}/"
                            f"{model}_v{version}/experiment.yaml"
                        )
                        if os.path.isfile(spec_file_path):
                            spec_data = safe_load_file(spec_file_path, file_type="yaml") or {}
                else:
                    # Regular mode - only download experiment specs
                    spec_data = self.get_base_spec(ngc_path, exp_id, ngc_key)

                return exp_id, spec_data, None
            except Exception as e:
                return exp_id, {}, e

        # Download specs in parallel
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(download_single_spec, exp_id, exp_data): exp_id
                for exp_id, exp_data in ngc_experiments
            }

            for future in as_completed(futures):
                exp_id, spec_data, error = future.result()
                if error:
                    logger.error(f"Failed to download spec for {exp_id}: {error}")

                # Update the experiment with spec data
                base_experiments[exp_id]["base_experiment_metadata"]["spec_file_present"] = bool(spec_data)
                base_experiments[exp_id]["base_experiment_metadata"]["specs"] = spec_data

        logger.info("Completed parallel spec downloads")

    def download_complete_model(self, ngc_path, exp_id, ngc_key):
        """Download complete model for airgapped deployment"""
        logger.info(f"Downloading complete model: {ngc_path}")

        org, team, model, version = self.split_ngc_path(ngc_path)

        # Get cached NGC client
        clt = self.get_ngc_client(org, team, ngc_key)
        if clt is None:
            logger.error(f"Failed to get NGC client for {ngc_path}")
            return False

        # Download complete model
        try:
            model_dir = f"{self.rootdir}/{org}/{team}/{model}/{version}"
            os.makedirs(model_dir, exist_ok=True)

            # Download all files for the model version
            logger.info(f"Downloading complete model: {ngc_path}")
            clt.registry.model.download_version(ngc_path, destination=model_dir)

            logger.info(f"Successfully downloaded complete model for {ngc_path}")
            return True
        except Exception as e:
            logger.error(f"Unable to download complete model for {ngc_path}")
            logger.error(e)
            return False

    def parse_model_path(self, model_path: str):
        """Parse model path and return source type and cleaned path"""
        if model_path.startswith("ngc://"):
            return "ngc", model_path[6:]  # Remove ngc:// prefix
        if model_path.startswith("hf_model://"):
            return "huggingface", model_path[11:]  # Remove hf_model:// prefix
        return "ngc", model_path

    def download_huggingface_model(self, hf_path: str, exp_id: str):
        """Download complete model from Hugging Face"""
        try:
            # Create directory structure
            model_dir = f"{self.rootdir}/huggingface/{hf_path.replace('/', '_')}"
            os.makedirs(model_dir, exist_ok=True)

            # Download model using utility function
            logger.info(f"Downloading Hugging Face model {hf_path}")
            download_hf_model(
                download_url=hf_path,
                destination_folder=model_dir,
                token=os.getenv("HF_TOKEN", "")
            )

            logger.info(f"Successfully downloaded Hugging Face model: {hf_path}")
            return True
        except Exception as e:
            logger.error(f"Unable to download Hugging Face model {hf_path}: {e}")
            return False

    def convert_str_to_enum(self, string_value: str, enum_type: Enum):
        """Convert string to enum based on value."""
        if not string_value:
            return None
        try:
            enum_type = enum_type(
                string_value.replace('-', ' ').replace('_', ' ').lower() if string_value is not None else None
            )
            if enum_type:
                return enum_type.value
            return None
        except ValueError:
            return None

    def extract_common_metadata(self, experiment_info, api_params, model_specific_overrides=None):
        """Create common metadata structure for both NGC and Hugging Face models"""
        network_arch = experiment_info["network_arch"]
        dataset_type = api_params.get("dataset_type", None)
        if dataset_type is None:
            dataset_type = get_orchestration_network_from_microservices(network_arch)

        # Common network architecture processing
        accepted_ds_intents = api_params.get("accepted_ds_intents", ["training", "evaluation"])

        base_experiment_pull_complete = "starting"
        if network_arch in TAO_NETWORKS:
            base_experiment_pull_complete = "pull_complete"

        # Base metadata structure
        metadata = {
            "id": experiment_info["id"],
            "kind": "experiment",  # Add kind field for job schema compatibility
            "user_id": "00000000-0000-0000-0000-000000000000",  # System user for pretrained models
            "shared": False,  # Not shared (public instead)
            "public": True,
            "read_only": True,  # Default, can be overridden
            "base_experiment_ids": [],
            "train_datasets": [],
            "eval_dataset": None,
            "calibration_dataset": None,
            "inference_dataset": None,
            "checkpoint_choose_method": "best_model",
            "checkpoint_epoch_number": {"id": 0},
            "encryption_key": "tlt_encode",  # Default encryption key
            "logo": "https://www.nvidia.com",  # Default, can be overridden
            "network_arch": network_arch,
            "dataset_type": dataset_type,
            "dataset_formats": api_params.get("formats", ["coco"]),
            "accepted_dataset_intents": accepted_ds_intents,
            "actions": api_params["actions"],
            "name": experiment_info["name"],
            "description": f"Base Experiment for {network_arch}",  # Default, can be overridden
            "model_description": f"Base Experiment for {network_arch}",  # Default, can be overridden
            "version": "unknown",  # Default, can be overridden
            "created_on": datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z'),
            "last_modified": datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z'),
            "ngc_path": experiment_info["ngc_path"],
            "sha256_digest": {},
            "base_experiment_metadata": {
                "task": None,
                "backbone_type": None,
                "backbone_class": None,
                "domain": None,
                "license": None,
                "is_backbone": experiment_info.get("is_backbone", True),  # Use CSV value or default to True
                "is_trainable": False,  # Default, can be overridden
                "num_parameters": "Unknown",
                "accuracy": "Unknown",
                "model_card_link": "https://www.nvidia.com",  # Default, can be overridden
                "spec_file_present": experiment_info["base_experiment_metadata"]["spec_file_present"],
                "specs": experiment_info["base_experiment_metadata"]["specs"]
            },
            "base_experiment_pull_complete": base_experiment_pull_complete,
        }

        # Apply model-specific overrides
        if model_specific_overrides:
            # Deep merge overrides into metadata
            for key, value in model_specific_overrides.items():
                if key == "base_experiment_metadata" and isinstance(value, dict):
                    metadata["base_experiment_metadata"].update(value)
                else:
                    metadata[key] = value

        # Preserve source_type from experiment_info if it exists
        if "source_type" in experiment_info:
            metadata["source_type"] = experiment_info["source_type"]

        return metadata

    def extract_huggingface_metadata(self, experiment_info):
        """Create metadata for Hugging Face models using network config"""
        network_arch = experiment_info["network_arch"]

        # Validate that the network architecture is a valid endpoint
        self.validate_network_endpoint(network_arch)

        # Load network configuration (cached with automatic fallback)
        network_config = self.get_network_config_cached(network_arch)
        api_params = network_config["api_params"]

        # Extract model name for links
        model_path = experiment_info["ngc_path"]  # This contains the HF path

        # Hugging Face specific overrides
        hf_overrides = {
            "read_only": False,  # HF models are typically trainable/fine-tunable
            "logo": "https://huggingface.co/front/assets/huggingface_logo-noborder.svg",
            "description": f"Hugging Face model {model_path} for {network_arch}",
            "model_description": f"Hugging Face model: {model_path}",
            "version": "latest",
            "base_experiment_pull_complete": "pull_complete",  # HF models are always considered available
            "base_experiment_metadata": {
                "is_trainable": True,  # HF models are typically trainable
                "model_card_link": f"https://huggingface.co/{model_path}",
            }
        }

        return self.extract_common_metadata(experiment_info, api_params, hf_overrides)

    def extract_metadata(self, model_info, experiment_info, model_name):
        """Create metadata for NGC models"""
        if model_info is None:
            raise ValueError(f"Failed to get model info for {experiment_info['ngc_path']}")
        if model_info["modelVersion"]["status"] != "UPLOAD_COMPLETE":
            raise ValueError(f"Model {experiment_info['ngc_path']} is not in UPLOAD_COMPLETE status!")
        network_arch = experiment_info["network_arch"]
        if network_arch not in ExperimentNetworkArch.__members__:
            raise ValueError(f"Network architecture of `{network_arch}` is not supported by API!")

        # Validate that the network architecture is a valid endpoint
        self.validate_network_endpoint(network_arch)

        # Extract NGC-specific attributes from customMetrics
        attr = {}
        if model_info["modelVersion"].get("customMetrics"):
            for customMetrics in model_info["modelVersion"]["customMetrics"]:
                for key_value in customMetrics.get("attributes", []):
                    attr_key = key_value.get("key")
                    if attr_key:
                        try:
                            attr[attr_key] = ast.literal_eval(key_value.get("value"))
                        except (SyntaxError, ValueError):
                            attr[attr_key] = key_value.get("value")

            # NGC-specific validations
            if attr.get("tao_version"):
                attr["tao_version_check"] = self.check_version_compatibility(attr["tao_version"])
                if not attr["tao_version_check"]:
                    raise ValueError(
                        f"Model {experiment_info['ngc_path']} requires API version of {attr['tao_version']} "
                        f"but the current API version is {self.tao_version}!"
                    )
            if not attr.get("trainable"):
                raise ValueError(f"Model {experiment_info['ngc_path']} is not trainable!")
            for endpoint in attr.get("endpoints", []):
                if endpoint not in ExperimentNetworkArch.__members__:
                    logger.warning(
                        f"Skipping the 'endpoint' metadata for {experiment_info['ngc_path']}. "
                        f"'endpoint' metadata [{endpoint}] is not supported by API!"
                        "This may prevent base experiment creation in the future releases."
                    )

        # Load network configuration (cached with automatic fallback)
        network_config = self.get_network_config_cached(network_arch)
        logger.debug(f"network_arch: {network_arch}")
        logger.debug(f"network_config: {network_config}")
        api_params = network_config["api_params"]
        dataset_type = api_params.get("dataset_type", None)
        if dataset_type is None:
            dataset_type = get_orchestration_network_from_microservices(network_arch)

        # NGC-specific overrides
        ngc_overrides = {
            "read_only": model_info["model"].get("isReadOnly", True),
            "description": (model_info["modelVersion"].get("description", "") or
                            model_info["model"].get("shortDescription", f"Base Experiment for {network_arch}")),
            "model_description": model_info["model"].get("shortDescription", f"Base Experiment for {network_arch}"),
            "version": model_info["modelVersion"].get("versionId", ""),
            "created_on": model_info["modelVersion"].get("createdDate", datetime.datetime.now().isoformat()),
            "last_modified": model_info["model"].get("updatedDate", datetime.datetime.now().isoformat()),
            "sha256_digest": attr.get("sha256_digest", {}),
            "dataset_formats": api_params.get(
                "formats",
                self.get_network_config_cached(dataset_type).get("api_params", {}).get("formats", None)
            ),
            "base_experiment_metadata": {
                "task": self.convert_str_to_enum(attr.get("task", None), BaseExperimentTask),
                "backbone_type": self.convert_str_to_enum(attr.get("backbone_type", None), BaseExperimentBackboneType),
                "backbone_class": self.convert_str_to_enum(
                    attr.get("backbone_class", None),
                    BaseExperimentBackboneClass
                ),
                "domain": self.convert_str_to_enum(attr.get("domain", None), BaseExperimentDomain),
                "license": self.convert_str_to_enum(attr.get("license", None), BaseExperimentLicense),
                "is_backbone": (
                    experiment_info.get("is_backbone")
                    if experiment_info.get("is_backbone") is not None
                    else attr.get("is_backbone", True)
                ),
                "is_trainable": attr.get("trainable", False),
                "num_parameters": (
                    f"{round(random.uniform(1, 150))}M"
                    if attr.get("num_parameters", None) is None or not attr.get("num_parameters").endswith("M")
                    else attr.get("num_parameters")
                ),  # TODO: @bingjiez reverse after ngc models are updated
                "accuracy": (
                    f"{round(random.uniform(60, 100), 2)}%"
                    if attr.get("accuracy", None) is None
                    else attr.get("accuracy")
                ),  # TODO: @bingjiez reverse after ngc models are updated
                "model_card_link": f"https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/{model_name}",
            }
        }

        # Get common metadata with NGC-specific overrides
        metadata = self.extract_common_metadata(experiment_info, api_params, ngc_overrides)

        return metadata

    def get_ngc_hosted_base_experiments(self):
        """Get base experiments hosted on NGC"""
        model_info = {}
        valid_base_experiments = {}
        ngc_base_experiments = {}

        # Load from NGC if not CSV-only mode
        if not (self.use_csv and not self.use_both):
            ngc_experiments = self.load_base_experiments_from_ngc()
            logger.info("Loaded base experiments from NGC discovery: %s", len(ngc_experiments))
            ngc_base_experiments.update(ngc_experiments)

        # Load from CSV if CSV or both mode
        if self.use_csv or self.use_both:
            csv_experiments = self.load_base_experiments_from_csv()
            logger.info("Loaded base experiments from CSV file: %s", len(csv_experiments))
            # CSV takes precedence over NGC
            ngc_base_experiments.update(csv_experiments)

        logger.info("Total experiments loaded: %s", len(ngc_base_experiments))
        if self.model_names:
            logger.info("Applied model name filtering for: %s", ', '.join(self.model_names))
        logger.info("--------------------------------------------------------")

        # Separate NGC and non-NGC experiments
        ngc_experiments_to_fetch = []
        non_ngc_experiments = {}

        for exp_id, base_experiment in ngc_base_experiments.items():
            ngc_path = base_experiment["ngc_path"]
            source_type = base_experiment.get("source_type", "ngc")

            if source_type == "ngc":
                try:
                    org, team, model_name, *_ = self.split_ngc_path(ngc_path)
                    # In CSV or both mode, process all experiments; in auto-discovery mode, check org/team membership
                    if self.use_csv or self.use_both or (org, team) in self.org_team_list:
                        _, ngc_token = self.get_ngc_token(org, team)
                        ngc_experiments_to_fetch.append((exp_id, ngc_path, ngc_token, base_experiment, model_name))
                except Exception as e:
                    logger.error(f"Failed to parse NGC path {ngc_path}: {e}")
                    continue
            else:
                non_ngc_experiments[exp_id] = base_experiment

        # Batch fetch NGC model info in parallel
        if ngc_experiments_to_fetch:
            logger.info(f"Fetching model info for {len(ngc_experiments_to_fetch)} NGC models in parallel...")
            ngc_paths_with_tokens = [(ngc_path, token) for _, ngc_path, token, _, _ in ngc_experiments_to_fetch]
            model_info = self.get_model_info_batch(ngc_paths_with_tokens)
            logger.info(f"Successfully fetched {len(model_info)} NGC model infos")

            # Process NGC experiments with fetched model info
            for exp_id, ngc_path, _, base_experiment, model_name in ngc_experiments_to_fetch:
                if ngc_path in model_info:
                    try:
                        # Remove ngc_key before extracting metadata (shouldn't be in final output)
                        base_experiment.pop("ngc_key", None)
                        valid_base_experiments[exp_id] = self.extract_metadata(
                            model_info[ngc_path], base_experiment, model_name
                        )
                        logger.info(
                            f"Successfully created a base experiment for {ngc_path},"
                            f"{base_experiment['network_arch']}"
                        )
                    except ValueError as e:
                        logger.error(f"Failed to create a base experiment for {ngc_path} >>> {e}")
                        continue
                else:
                    logger.error(f"Model info not available for {ngc_path}")

        # Process non-NGC models (e.g., Hugging Face)
        for exp_id, base_experiment in non_ngc_experiments.items():
            ngc_path = base_experiment["ngc_path"]
            source_type = base_experiment.get("source_type", "unknown")
            try:
                # Remove ngc_key before extracting metadata (shouldn't be in final output)
                base_experiment.pop("ngc_key", None)
                valid_base_experiments[exp_id] = self.extract_huggingface_metadata(base_experiment)
                logger.info(
                    f"Successfully created a base experiment for {ngc_path},"
                    f"{base_experiment['network_arch']} (source: {source_type})"
                )
            except Exception as e:
                logger.error(f"Failed to create a base experiment for {ngc_path} >>> {e}")
                logger.error(traceback.format_exc())
                continue

        return valid_base_experiments

    def get_existing_base_experiments(self):
        """Get existing base experiments"""
        if os.path.isfile(self.metadata_file):
            existing_models = safe_load_file(self.metadata_file)
            return existing_models
        return {}

    def sync(self):
        """Sync metadata info for ngc hosted base experiments"""
        existing_base_experiments = self.get_existing_base_experiments()
        ngc_hosted_base_experiments = self.get_ngc_hosted_base_experiments()
        self.metadata = {**existing_base_experiments, **ngc_hosted_base_experiments}

        if self.airgapped and self.metadata_file:
            # Write to JSON file for airgapped deployment
            os.makedirs(os.path.dirname(os.path.abspath(self.metadata_file)), exist_ok=True)
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            logger.info(f"Base experiments metadata written to JSON file: {self.metadata_file}")
        elif not self.dry_run and not (self.use_csv and not self.use_both):
            # Regular mode - write to MongoDB (skip when local CSV-only, no backend)
            if MongoHandler is not None:
                mongo_jobs = MongoHandler("tao", "jobs")
                for base_exp_id in self.metadata:
                    base_exp_metadata = self.metadata[base_exp_id]
                    mongo_jobs.upsert({'id': base_exp_id}, base_exp_metadata)
                logger.info("Base experiments metadata written to jobs database")
            else:
                logger.info("Skipping MongoDB write (no backend / local CSV-only mode)")
        elif self.use_csv and not self.use_both:
            logger.info("Skipping metadata write (local CSV-only mode, no MongoDB)")
        else:
            logger.info("Skipping metadata write in dry run mode!")

        logger.info("--------------------------------------------------------")
        logger.info("Existing base experiments: %s", len(existing_base_experiments))
        logger.info("New base experiments: %s", len(ngc_hosted_base_experiments))
        logger.info("Total base experiments: %s", len(self.metadata))


if __name__ == "__main__":
    if PTM_PULL == "true":
        parser = argparse.ArgumentParser(description="Generate base experiment metadata file")
        parser.add_argument("--shared-folder-path", help="Root path for base experiments", default="ptms")
        parser.add_argument(
            "--org-teams",
            help="Organization and team names. Each pair of org/team separated by a comma."
        )
        parser.add_argument("--ngc-key", help="NGC Key (or set NGC_API_KEY/NGC_KEY env)", default=None)
        parser.add_argument("--dry-run", help="Dry run mode", default=False, action="store_true")
        parser.add_argument("--override", help="Override existing base experiments", action="store_true")
        parser.add_argument("--use-csv", help="Use predefined CSV file for model selection instead of NGC discovery",
                            default=False, action="store_true")
        parser.add_argument("--use-both", help="Use both CSV file and NGC auto-discovery for model selection",
                            default=False, action="store_true")
        parser.add_argument("--csv-path", help="Path to CSV file (default: pretrained_models.csv next to this module)")
        parser.add_argument("--model-names",
                            help="Comma-separated list of model names/entrypoints to download "
                                 "(e.g., 'classification_pyt,dino')")
        args = parser.parse_args()

        bem = BaseExperimentMetadata(
            args.shared_folder_path,
            args.org_teams,
            args.ngc_key,
            args.override,
            args.dry_run,
            args.use_csv,
            args.use_both,
            args.model_names,
            args.csv_path,
        )
        bem.sync()
