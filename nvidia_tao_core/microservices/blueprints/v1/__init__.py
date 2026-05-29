# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API v1 blueprints for organizing endpoints by version."""

from .auth import auth_bp_v1
from .automl import automl_params_bp_v1
from .workspaces import workspaces_bp_v1
from .datasets import datasets_bp_v1
from .experiments import experiments_bp_v1
from .health import health_bp_v1
from .internal import internal_bp_v1
from .admin import admin_bp_v1

# Import all schemas from v1 schemas module
from .schemas import (
    AWSCloudPull,
    AutoMLResults,
    AutoMLParameterDetailsRsp,
    AutoMLUpdateParameterRangesReq,
    AutoML,
    AzureCloudPull,
    BaseExperimentMetadata,
    BulkOpsRsp,
    BulkOps,
    Category,
    CategoryWise,
    ContainerJob,
    ContainerJobStatus,
    DatasetJobList,
    DatasetJob,
    DatasetListRsp,
    DatasetActions,
    DatasetUriLst,
    DatasetReq,
    DatasetRsp,
    DetailedStatus,
    ErrorRsp,
    ExperimentActions,
    ExperimentDownload,
    ExperimentJobList,
    ExperimentJob,
    ExperimentListRsp,
    ExperimentReq,
    ExperimentRsp,
    ExperimentTagList,
    GpuDetails,
    Graph,
    HuggingFaceCloudPull,
    JobResult,
    JobResume,
    JobSubset,
    KPI,
    LoadAirgappedExperimentsReq,
    LoadAirgappedExperimentsRsp,
    LoginReq,
    LoginRsp,
    LstInt,
    LstStr,
    MessageOnly,
    MissingFile,
    PaginationInfo,
    PublishModel,
    Stats,
    ValidationDetails,
    WorkspaceBackupReq,
    WorkspaceListRsp,
    WorkspaceReq,
    WorkspaceRsp
)

# V1 Blueprint Schemas - List of tuples with (name, schema_class)
V1_SCHEMAS = [
    ("AWSCloudPull", AWSCloudPull),
    ("AutoMLParameterDetailsRsp", AutoMLParameterDetailsRsp),
    ("AutoMLUpdateParameterRangesReq", AutoMLUpdateParameterRangesReq),
    ("AutoMLResults", AutoMLResults),
    ("AutoML", AutoML),
    ("AzureCloudPull", AzureCloudPull),
    ("BaseExperimentMetadata", BaseExperimentMetadata),
    ("BulkOpsRsp", BulkOpsRsp),
    ("BulkOps", BulkOps),
    ("Category", Category),
    ("CategoryWise", CategoryWise),
    ("ContainerJob", ContainerJob),
    ("ContainerJobStatus", ContainerJobStatus),
    ("DatasetActions", DatasetActions),
    ("DatasetJobList", DatasetJobList),
    ("DatasetJob", DatasetJob),
    ("DatasetListRsp", DatasetListRsp),
    ("DatasetUriLst", DatasetUriLst),
    ("DatasetReq", DatasetReq),
    ("DatasetRsp", DatasetRsp),
    ("DetailedStatus", DetailedStatus),
    ("ErrorRsp", ErrorRsp),
    ("ExperimentActions", ExperimentActions),
    ("ExperimentDownload", ExperimentDownload),
    ("ExperimentJobList", ExperimentJobList),
    ("ExperimentJob", ExperimentJob),
    ("ExperimentListRsp", ExperimentListRsp),
    ("ExperimentReq", ExperimentReq),
    ("ExperimentRsp", ExperimentRsp),
    ("ExperimentTagList", ExperimentTagList),
    ("GpuDetails", GpuDetails),
    ("Graph", Graph),
    ("HuggingFaceCloudPull", HuggingFaceCloudPull),
    ("JobResult", JobResult),
    ("JobResume", JobResume),
    ("JobSubset", JobSubset),
    ("KPI", KPI),
    ("LoadAirgappedExperimentsReq", LoadAirgappedExperimentsReq),
    ("LoadAirgappedExperimentsRsp", LoadAirgappedExperimentsRsp),
    ("LoginReq", LoginReq),
    ("LoginRsp", LoginRsp),
    ("LstInt", LstInt),
    ("LstStr", LstStr),
    ("MessageOnly", MessageOnly),
    ("MissingFile", MissingFile),
    ("PaginationInfo", PaginationInfo),
    ("PublishModel", PublishModel),
    ("Stats", Stats),
    ("ValidationDetails", ValidationDetails),
    ("WorkspaceBackupReq", WorkspaceBackupReq),
    ("WorkspaceListRsp", WorkspaceListRsp),
    ("WorkspaceReq", WorkspaceReq),
    ("WorkspaceRsp", WorkspaceRsp)
]

__all__ = [
    'auth_bp_v1',
    'automl_params_bp_v1',
    'workspaces_bp_v1',
    'datasets_bp_v1',
    'experiments_bp_v1',
    'health_bp_v1',
    'internal_bp_v1',
    'admin_bp_v1',
    'V1_SCHEMAS'
]
