# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Defining enums for dataset and model formats and types"""
import enum
import json
import os
import pathlib
from typing import Set, Tuple
import logging

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


def _scan_config_files() -> Tuple[Set[str], Set[str]]:
    """Scan all .config.json files to collect dataset types and formats.

    Returns:
        Tuple[Set[str], Set[str]]: Set of dataset types and formats
    """
    config_dir = pathlib.Path(__file__).parent / "handlers" / "network_configs"
    dataset_types = set()
    dataset_formats = set()

    if config_dir.exists():
        for config_file in config_dir.glob("*.config.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    api_params = config.get("api_params", {})

                    # Get dataset type
                    if "dataset_type" in api_params:
                        dataset_types.add(api_params["dataset_type"])

                    # Get formats
                    if "formats" in api_params:
                        dataset_formats.update(api_params["formats"])
            except (json.JSONDecodeError, IOError):
                continue

    return dataset_types, dataset_formats


def _get_all_dataset_types():
    """Get all valid dataset types including defaults."""
    dataset_types, _ = _scan_config_files()
    # Add not_restricted and user_custom as they might be special cases
    dataset_types.update({"not_restricted", "user_custom"})
    result = {dtype.upper(): dtype for dtype in dataset_types}
    logger.debug("Found dataset types: %s", result)  # Debug print
    return result


def _get_valid_actions():
    """Get all valid actions from config files."""
    config_dir = pathlib.Path(__file__).parent / "handlers" / "network_configs"
    actions = set()

    if config_dir.exists():
        for config_file in config_dir.glob("*.config.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if "actions" in config.get("api_params", {}):
                        actions.update(config["api_params"]["actions"])
                    actions_mapping = config.get("actions_mapping", {})
                    for _, mapping in actions_mapping.items():
                        if "action" in mapping:
                            actions.add(mapping["action"])
            except (json.JSONDecodeError, IOError):
                continue

    # Add default actions if needed
    actions.update({"train", "evaluate", "export", "inference"})
    return actions


def _get_valid_params(config_file: str, network_name: str, param: str):
    """Get all valid enum values for a given parameter from config files."""
    actions = set()
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            if param in config.get("api_params", {}):
                if isinstance(config["api_params"][param], list):
                    actions.update(config["api_params"][param])
                else:
                    actions.add(config["api_params"][param])
        if param == "actions":
            if "dataset_type" in config["api_params"]:
                dataset_config_file = (
                    pathlib.Path(__file__).parent / "handlers" / "network_configs" /
                    f"{config['api_params']['dataset_type']}.config.json"
                )
                with open(dataset_config_file, 'r', encoding='utf-8') as f:
                    dataset_config = json.load(f)
                    actions_mapping = dataset_config.get("actions_mapping", {})
                    for api_action_name, mapping in actions_mapping.items():
                        if "network" in mapping and mapping["network"] == network_name:
                            actions.add(api_action_name)
    except (json.JSONDecodeError, IOError):
        pass
    return actions


def _get_valid_config_json_param_for_network(network_name: str, param: str):
    """Choose the correct config file and get all valid enum values for a given parameter."""
    config_file = pathlib.Path(__file__).parent / "handlers" / "network_configs" / f"{network_name}.config.json"
    actions = set()

    if config_file.exists():
        actions = _get_valid_params(config_file, network_name, param)
    else:
        for dataset_type in ["object_detection", "image_classification", "segmentation"]:
            config_file = pathlib.Path(__file__).parent / "handlers" / "network_configs" / f"{dataset_type}.config.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    actions_mapping = config.get("actions_mapping", {})
                    for _, mapping in actions_mapping.items():
                        if "network" in mapping and mapping["network"] == network_name:
                            actions = _get_valid_params(config_file, network_name, param)
                            if param == "dataset_type":
                                actions.add(dataset_type)
                            break

    return actions


def _get_mapped_network_architectures(config: dict, architectures: Set[str], arch_name: str = None) -> None:
    """Get mapped network architectures from config files."""
    actions_mapping = config.get("actions_mapping", {})
    for _, mapping in actions_mapping.items():
        if "network" in mapping:
            architectures.add(mapping["network"])
            if arch_name and arch_name in architectures and arch_name != mapping["network"]:
                architectures.remove(arch_name)


def _get_network_architectures(get_mapped: bool = False) -> Set[str]:
    """Scan config directory for .config.json files to determine valid network architectures.

    Returns:
        Set[str]: Set of valid network architecture names
    """
    config_dir = pathlib.Path(__file__).parent / "handlers" / "network_configs"

    architectures = set()

    if config_dir.exists():
        for config_file in config_dir.glob("*.config.json"):
            arch_name = config_file.stem.replace(".config", "")
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                    # Add the main architecture
                    if arch_name not in ["image_classification",
                                         "object_detection",
                                         "segmentation",
                                         "character_recognition",
                                         "vlm"]:
                        architectures.add(arch_name)
                    else:
                        # Add networks from action mappings
                        _get_mapped_network_architectures(config, architectures)
                    if get_mapped:
                        _get_mapped_network_architectures(config, architectures, arch_name)

            except (json.JSONDecodeError, IOError):
                continue

    return architectures


def _get_all_metrics() -> Set[str]:
    """Scan all config files to collect available metrics.

    Returns:
        Set[str]: Set of all available metrics
    """
    config_dir = pathlib.Path(__file__).parent / "handlers" / "network_configs"
    all_metrics = set()

    if config_dir.exists():
        for config_file in config_dir.glob("*.config.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    metrics = config.get("metrics", {}).get("available_metrics", [])
                    all_metrics.update(metrics)

            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Error reading metrics from %s: %s", config_file, e)
                continue
    # Add all BaseMetrics values
    all_metrics.update(m.value for m in BaseMetrics)
    return all_metrics


def _get_dynamic_metric_patterns() -> Set[str]:
    """Get dynamic metric patterns from config files.

    Returns:
        Set[str]: Set of regex patterns for dynamic metrics
    """
    config_dir = pathlib.Path(__file__).parent / "handlers" / "network_configs"
    patterns = set()

    if config_dir.exists():
        for config_file in config_dir.glob("*.config.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    metric_patterns = config.get("metrics", {}).get("dynamic_metric_patterns", [])
                    patterns.update(metric_patterns)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Error reading metric patterns from %s: %s", config_file, e)
                continue
    return patterns


class BaseMetrics(str, enum.Enum):
    """Class defining metric types in enum"""

    three_d_mAP = '3d mAP'
    AP = 'AP'
    AP11 = 'AP11'
    AP40 = 'AP40'
    AP50 = 'AP50'
    AP75 = 'AP75'
    APl = 'APl'
    APm = 'APm'
    APs = 'APs'
    ARl = 'ARl'
    ARm = 'ARm'
    ARmax1 = 'ARmax1'
    ARmax10 = 'ARmax10'
    ARmax100 = 'ARmax100'
    ARs = 'ARs'
    bbox_val_mAP = 'bbox_val_mAP'
    bbox_val_mAP50 = 'bbox_val_mAP50'
    bbox_test_mAP = 'bbox_test_mAP'
    bbox_test_mAP50 = 'bbox_test_mAP50'
    Hmean = 'Hmean'
    Mean_IOU = 'Mean IOU'
    Precision = 'Precision'
    ema_precision = 'ema_precision'
    Recall = 'Recall'
    ema_recall = 'ema_recall'
    Thresh = 'Thresh'
    ACC_all = 'ACC_all'
    accuracy = 'accuracy'
    m_accuracy = 'm_accuracy'
    avg_accuracy = 'avg_accuracy'
    accuracy_top_1 = 'accuracy_top-1'
    bev_mAP = 'bev mAP'
    cmc_rank_1 = 'cmc_rank_1'
    cmc_rank_10 = 'cmc_rank_10'
    cmc_rank_5 = 'cmc_rank_5'
    defect_acc = 'defect_acc'
    embedder_base_lr = 'embedder_base_lr'
    hmean = 'hmean'
    ema_hmean = 'ema_hmean'
    learning_rate = 'learning_rate'
    loss = 'loss'
    lr = 'lr'
    matched_ious = 'matched_ious'
    mAP = 'mAP'
    mAcc = 'mAcc'
    mIoU = 'mIoU'
    mIoU_large = 'mIoU_large'
    mIoU_medium = 'mIoU_medium'
    mIoU_small = 'mIoU_small'
    param_count = 'param_count'
    precision = 'precision'
    pruning_ratio = 'pruning_ratio'
    recall = 'recall'
    recall_rcnn_0_3 = 'recall/rcnn_0.3'
    recall_rcnn_0_5 = 'recall/rcnn_0.5'
    recall_rcnn_0_7 = 'recall/rcnn_0.7'
    recall_roi_0_3 = 'recall/roi_0.3'
    recall_roi_0_5 = 'recall/roi_0.5'
    recall_roi_0_7 = 'recall/roi_0.7'
    size = 'size'
    segm_val_mAP = 'segm_val_mAP'
    segm_val_mAP50 = 'segm_val_mAP50'
    segm_test_mAP = 'segm_test_mAP'
    segm_test_mAP50 = 'segm_test_mAP50'
    test_Mean_Average_Precision = 'test Mean Average Precision'
    test_Mean_Reciprocal_Rank = 'test Mean Reciprocal Rank'
    test_Precision_at_Rank_1 = 'test Precision at Rank 1'
    test_r_Precision = 'test r-Precision'
    test_AMI = 'test_AMI'
    test_NMI = 'test_NMI'
    test_acc = 'test_acc'
    test_fnr = 'test_fnr'
    test_fpr = 'test_fpr'
    test_mAP = 'test_mAP'
    test_mAP50 = 'test_mAP50'
    test_mf1 = 'test_mf1'
    test_miou = 'test_miou'
    test_mprecision = 'test_mprecision'
    test_mrecall = 'test_mrecall'
    top_k = 'top_k'
    train_acc = 'train_acc'
    train_accuracy = 'train_accuracy'
    train_fpr = 'train_fpr'
    train_loss = 'train_loss'
    trunk_base_lr = 'trunk_base_lr'
    val_Mean_Average_Precision = 'val Mean Average Precision'
    val_Mean_Reciprocal_Rank = 'val Mean Reciprocal Rank'
    val_Precision_at_Rank_1 = 'val Precision at Rank 1'
    val_r_Precision = 'val r-Precision'
    val_2DMPE = 'val_2DMPE'
    val_3DIoU = 'val_3DIoU'
    test_2DMPE = 'test_2DMPE'
    test_3DIoU = 'test_3DIoU'
    val_AMI = 'val_AMI'
    val_NMI = 'val_NMI'
    val_acc = 'val_acc'
    val_accuracy = 'val_accuracy'
    val_fpr = 'val_fpr'
    val_loss = 'val_loss'
    val_mAP = 'val_mAP'
    val_mAP50 = 'val_mAP50'
    val_mf1 = 'val_mf1'
    val_miou = 'val_miou'
    val_mprecision = 'val_mprecision'
    val_mrecall = 'val_mrecall'

    # Mask Grounding DINO metrics
    val_gIoU = 'val_gIoU'
    val_cIoU = 'val_cIoU'
    val_T_acc = 'val_T_acc'
    val_N_acc = 'val_N_acc'
    val_Pr_0_7 = 'val_Pr@0.7'
    val_Pr_0_8 = 'val_Pr@0.8'
    val_Pr_0_9 = 'val_Pr@0.9'

    # Mask Grounding DINO test metrics
    test_gIoU = 'test_gIoU'
    test_cIoU = 'test_cIoU'
    test_T_acc = 'test_T_acc'
    test_N_acc = 'test_N_acc'
    test_Pr_0_7 = 'test_Pr@0.7'
    test_Pr_0_8 = 'test_Pr@0.8'
    test_Pr_0_9 = 'test_Pr@0.9'

    # Data Service Analytics KPI metrics
    num_objects = 'num_objects'
    object_count_index = 'object_count_index'
    object_count_num = 'object_count_num'
    object_count_percent = 'object_count_percent'
    bbox_area_type = 'bbox_area_type'
    bbox_area_mean = 'bbox_area_mean'


class BaseExperimentTask(enum.Enum):
    """Class defining base experiment metadata task field"""

    unknown = None
    object_detection = "object detection"
    image_classification = "image classification"
    segmentation = "segmentation"
    re_identification = "re identification"
    pose_classification = "pose classification"
    action_recognition = "action recognition"
    optical_character_recognition = "optical character recognition"
    visual_changenet_segmentation = "visual changenet segmentation"
    visual_changenet_classification = "visual changenet classification"


class BaseExperimentDomain(enum.Enum):
    """Class defining base experiment metadata license field"""

    unknown = None
    general = "general"
    purpose_built = "purpose built"


class BaseExperimentBackboneType(enum.Enum):
    """Class defining base experiment metadata license field"""

    unknown = None
    cnn = "cnn"
    transformer = "transformer"


class BaseExperimentBackboneClass(enum.Enum):
    """Class defining base experiment metadata license field"""

    unknown = None
    swin = "swin"
    fan = "fan"
    vit = "vit"
    gcvit = "gcvit"
    fastervit = "fastervit"
    efficientnet = "efficientnet"
    resnet = "resnet"
    stgcn = "st gcn"
    convnext = "convnext"
    densenet = "densenet"


class BaseExperimentLicense(enum.Enum):
    """Class defining base experiment metadata license field"""

    unknown = None
    nvaie_eula = "nvaie eula"
    nvidia_model_eula = "nvidia model eula"
    cc_by_nc_sa_4 = "cc by nc sa 4.0"


class Backend(enum.Enum):
    """Class defining execution backend enum"""

    LOCAL_K8S = "local-k8s"
    LOCAL_DOCKER = "local-docker"
    SLURM = "slurm"
    LEPTON = "lepton"

    @classmethod
    def _missing_(cls, value):
        """Enable lookup by value"""
        for member in cls:
            if member.value == value:
                return member
        return None

# Create Enums


dataset_types, _ = _scan_config_files()
DatasetType = enum.Enum('DatasetType', {name: name for name in dataset_types}, type=str)

_, dataset_formats = _scan_config_files()
DatasetFormat = enum.Enum('DatasetFormat', {name: name for name in dataset_formats}, type=str)

actions = _get_valid_actions()
ActionEnum = enum.Enum('ActionEnum', {name: name for name in actions}, type=str)

network_architectures = _get_network_architectures()
ExperimentNetworkArch = enum.Enum('ExperimentNetworkArch', {name: name for name in network_architectures}, type=str)
container_network_architectures = _get_network_architectures(get_mapped=True)
ContainerNetworkArch = enum.Enum(
    'ContainerNetworkArch', {name: name for name in container_network_architectures}, type=str
)

metrics = _get_all_metrics()
Metrics = enum.Enum('Metrics', {name: name for name in metrics}, type=str)
