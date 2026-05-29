# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Marshmallow schemas for API."""

import re
import sys
import math
from datetime import datetime
from marshmallow import Schema, fields, EXCLUDE, RAISE, validates_schema, ValidationError, validate
from marshmallow_enum import EnumField, Enum
from marshmallow_oneofschema import OneOfSchema

from nvidia_tao_core.microservices.enum_constants import (
    ActionEnum,
    DatasetFormat,
    DatasetType,
    ExperimentNetworkArch,
    Metrics,
    BaseExperimentTask,
    BaseExperimentDomain,
    BaseExperimentBackboneType,
    BaseExperimentBackboneClass,
    BaseExperimentLicense,
    _get_dynamic_metric_patterns
)


def sys_int_format():
    """Get integer format based on system."""
    if sys.maxsize > 2**31 - 1:
        return "int64"
    return "int32"


def validate_endpoint_url(url):
    """Custom URL validator that accepts internal hostnames and services.

    This validator is more lenient than marshmallow's default URL validator,
    specifically allowing internal hostnames like 'seaweedfs-s3', 'localhost',
    IP addresses, and service names common in containerized environments.
    """
    if not url:
        return True  # allow_none=True is handled by the field

    # Basic URL structure validation using regex
    # This pattern allows for:
    # - http/https protocols
    # - hostnames with hyphens, underscores, alphanumeric characters
    # - IP addresses
    # - ports
    # - paths, query strings, fragments
    url_pattern = re.compile(
        r'^https?://'  # http or https protocol
        r'(?:'
        r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-_]*[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-_]*[a-zA-Z0-9])?'  # hostname
        r'|'
        r'[a-zA-Z0-9](?:[a-zA-Z0-9-_]*[a-zA-Z0-9])?'  # simple hostname (like 'seaweedfs-s3')
        r'|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'  # IPv4
        r'|'
        r'\[[0-9a-fA-F:]+\]'  # IPv6
        r')'
        r'(?::\d+)?'  # optional port
        r'(?:/[^\s]*)?$',  # optional path
        re.IGNORECASE
    )

    if not url_pattern.match(url):
        raise ValidationError('Invalid URL format.')

    return True


class DateTimeField(fields.DateTime):
    """Field for handling datetime objects.

    This field is used to handle datetime objects in the API.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, datetime):
            return value
        return super()._deserialize(value, attr, data, **kwargs)


class EnumFieldPrefix(fields.Field):
    """Enum field override for Metrics"""

    def __init__(self, enum, *args, **kwargs):
        """Init function of class"""
        self.enum = enum
        super().__init__(*args, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        if value in self.enum._value2member_map_:
            return value
        # Accept any best_* metric (e.g. best_train_loss, best_None from AutoML)
        if value.startswith('best_'):
            return value

        # Check against dynamic metric patterns for networks like sparse4d
        if self._validate_dynamic_metric(value):
            return value

        raise ValidationError(f"Invalid value '{value}' for enum '{self.enum.__name__}'")

    def _validate_dynamic_metric(self, value: str) -> bool:
        """Validate value against dynamic metric patterns."""
        patterns = _get_dynamic_metric_patterns()
        for pattern in patterns:
            try:
                if re.match(pattern, value):
                    return True
            except re.error:
                # Skip invalid regex patterns
                continue
        return False

    def _serialize(self, value, attr, obj, **kwargs):
        return value

# ============================================================================
# ENUMS
# ============================================================================


class JobKindEnum(Enum):
    """Class defining job kind enum"""

    dataset = 'dataset'
    experiment = 'experiment'


class JobStatusEnum(Enum):
    """Class defining job status enum"""

    Done = 'Done'
    Started = 'Started'
    Running = 'Running'
    Error = 'Error'
    Pending = 'Pending'
    Canceled = 'Canceled'
    Canceling = 'Canceling'
    Pausing = 'Pausing'
    Paused = 'Paused'
    Resuming = 'Resuming'


class PullStatus(Enum):
    """Class defining artifact upload/download status"""

    starting = "starting"
    in_progress = "in_progress"
    pull_complete = "pull_complete"
    invalid_pull = "invalid_pull"


class BulkOpsStatus(Enum):
    """Class defining bulk operation status enum"""

    success = "success"
    failed = "failed"


class AllowedDockerEnvVariables(Enum):
    """Allowed docker environment variables while launching DNN containers"""

    HF_TOKEN = "HF_TOKEN"

    WANDB_API_KEY = "WANDB_API_KEY"
    WANDB_BASE_URL = "WANDB_BASE_URL"
    WANDB_USERNAME = "WANDB_USERNAME"
    WANDB_ENTITY = "WANDB_ENTITY"
    WANDB_PROJECT = "WANDB_PROJECT"
    WANDB_INSECURE_LOGGING = "WANDB_INSECURE_LOGGING"

    CLEARML_WEB_HOST = "CLEARML_WEB_HOST"
    CLEARML_API_HOST = "CLEARML_API_HOST"
    CLEARML_FILES_HOST = "CLEARML_FILES_HOST"
    CLEARML_API_ACCESS_KEY = "CLEARML_API_ACCESS_KEY"
    CLEARML_API_SECRET_KEY = "CLEARML_API_SECRET_KEY"

    CLOUD_BASED = "CLOUD_BASED"
    TELEMETRY_OPT_OUT = "TELEMETRY_OPT_OUT"
    TAO_API_KEY = "TAO_API_KEY"
    TAO_USER_KEY = "TAO_USER_KEY"
    TAO_ADMIN_KEY = "TAO_ADMIN_KEY"
    TAO_API_SERVER = "TAO_API_SERVER"
    TAO_LOGGING_SERVER_URL = "TAO_LOGGING_SERVER_URL"
    RECURSIVE_DATASET_FILE_DOWNLOAD = "RECURSIVE_DATASET_FILE_DOWNLOAD"
    ORCHESTRATION_API_NETWORK = "ORCHESTRATION_API_NETWORK"
    ORCHESTRATION_API_ACTION = "ORCHESTRATION_API_ACTION"
    TAO_EXECUTION_BACKEND = "TAO_EXECUTION_BACKEND"
    AUTOML_EXPERIMENT_NUMBER = "AUTOML_EXPERIMENT_NUMBER"
    JOB_ID = "JOB_ID"
    TAO_API_RESULTS_DIR = "TAO_API_RESULTS_DIR"
    TAO_API_JOB_ID = "TAO_API_JOB_ID"  # Automl brain job id
    RETAIN_CHECKPOINTS_FOR_RESUME = "RETAIN_CHECKPOINTS_FOR_RESUME"
    EARLY_STOP_EPOCH = "EARLY_STOP_EPOCH"

    DEBUG_ENABLED = "DEBUG_ENABLED"

    TAO_TELEMETRY_SERVER = "TAO_TELEMETRY_SERVER"
    TAO_CLIENT_TYPE = "TAO_CLIENT_TYPE"  # Client type: container, api, cli, sdk, ui, etc.
    TAO_AUTOML_TRIGGERED = "TAO_AUTOML_TRIGGERED"  # Whether job is triggered by AutoML
    TAO_LOG_LEVEL = "TAO_LOG_LEVEL"  # Log level passed from brain to train jobs (e.g. INFO, DEBUG)

    CUDA_OVERRIDE_VERSION = "CUDA_OVERRIDE_VERSION"

    LEPTON_SHARED_MEMORY_SIZE = "LEPTON_SHARED_MEMORY_SIZE"


class CloudPullTypesEnum(Enum):
    """Class defining cloud pull types enum"""

    aws = 'aws'
    azure = 'azure'
    seaweedfs = 'seaweedfs'
    huggingface = 'huggingface'
    self_hosted = 'self_hosted'
    lepton = 'lepton'
    slurm = 'slurm'


class CloudFileType(Enum):
    """Class defining cloud file types enum"""

    file = "file"
    folder = "folder"


class DatasetIntentEnum(Enum):
    """Class defining dataset intent enum"""

    training = 'training'
    evaluation = 'evaluation'
    testing = 'testing'
    calibration = 'calibration'


class CheckpointChooseMethodEnum(Enum):
    """Class defining enum for methods of picking a trained checkpoint"""

    latest_model = 'latest_model'
    best_model = 'best_model'
    from_epoch_number = 'from_epoch_number'


class AutoMLAlgorithm(Enum):
    """Class defining automl algorithm enum"""

    bayesian = "bayesian"
    hyperband = "hyperband"
    bohb = "bohb"
    bfbo = "bfbo"
    asha = "asha"
    pbt = "pbt"
    dehb = "dehb"
    hyperband_es = "hyperband_es"


class SourceType(Enum):
    """Class defining source type enum for base experiments"""

    ngc = "ngc"
    huggingface = "huggingface"


# ============================================================================
# SCHEMAS
# ============================================================================


class MessageOnly(Schema):
    """Class defining dataset upload schema"""

    message = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))


class JobEventsRsp(Schema):
    """Class defining job events response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    events = fields.List(fields.Dict(), validate=validate.Length(max=sys.maxsize))


class MissingFile(Schema):
    """Schema for individual missing file entries"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    path = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    type = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=50))
    regex = fields.Str(format="regex", regex=r'.*', allow_none=True)


class ValidationDetails(Schema):
    """Class defining dataset validation details schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    error_details = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    expected_structure = fields.Dict(
        keys=fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100)),
        values=fields.Raw(),
        validate=validate.Length(max=sys.maxsize)
    )
    actual_structure = fields.List(fields.Str(format="regex", regex=r'.*'))
    missing_files = fields.List(fields.Nested(MissingFile))
    network_type = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    dataset_format = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    dataset_intent = fields.List(fields.Str(format="regex", regex=r'.*'))


class ErrorRsp(Schema):
    """Class defining error response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    error_desc = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    error_code = fields.Int(
        validate=fields.validate.Range(min=-sys.maxsize - 1, max=sys.maxsize),
        format=sys_int_format()
    )
    validation_details = fields.Dict(
        allow_none=True,
        metadata={
            "description": "Detailed validation information including expected structure, "
                           "actual structure, and missing files"
        }
    )


class PaginationInfo(Schema):
    """Class defining pagination info schema"""

    total_records = fields.Int(validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())
    total_pages = fields.Int(validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())
    page_size = fields.Int(validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())
    page_index = fields.Int(validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())


class BulkOps(Schema):
    """Class defining bulk operation schema"""

    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    status = EnumField(BulkOpsStatus)
    error_desc = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    error_code = fields.Int(
        validate=fields.validate.Range(min=-sys.maxsize - 1, max=sys.maxsize),
        format=sys_int_format(),
        allow_none=True
    )


class BulkOpsRsp(Schema):
    """Class defining bulk operation response schema"""

    results = fields.List(
        fields.Nested(BulkOps, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )


class DetailedStatus(Schema):
    """Class defining Status schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    date = fields.Str(format="mm/dd/yyyy", validate=fields.validate.Length(max=26))
    time = fields.Str(format="hh:mm:ss", validate=fields.validate.Length(max=26))
    message = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=6400))
    status = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))


class Graph(Schema):
    """Class defining Graph schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    metric = EnumFieldPrefix(Metrics)
    x_min = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=-sys.maxsize - 1, max=sys.maxsize),
        format=sys_int_format()
    )
    x_max = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=-sys.maxsize - 1, max=sys.maxsize),
        format=sys_int_format()
    )
    y_min = fields.Float(allow_none=True)
    y_max = fields.Float(allow_none=True)
    values = fields.Dict(keys=fields.Str(allow_none=True), values=fields.Float(allow_none=True))
    units = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=100))


class CategoryWise(Schema):
    """Class defining CategoryWise schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    category = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    value = fields.Float(allow_none=True)


class Category(Schema):
    """Class defining Category schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    metric = EnumFieldPrefix(Metrics)
    category_wise_values = fields.List(
        fields.Nested(CategoryWise, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )


class KPI(Schema):
    """Class defining KPI schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    metric = EnumFieldPrefix(Metrics)
    values = fields.Dict(allow_none=True)


class CustomFloatField(fields.Float):
    """Class defining custom Float field allown NaN and Inf values in Marshmallow"""

    def _deserialize(self, value, attr, data, **kwargs):
        if value == "nan" or (isinstance(value, float) and math.isnan(value)):
            return float("nan")
        if value == "inf" or (isinstance(value, float) and math.isinf(value)):
            return float("inf")
        if value == "-inf" or (isinstance(value, float) and math.isinf(value)):
            return float("-inf")
        if value is None:
            return None
        return super()._deserialize(value, attr, data)


class AutoMLResults(Schema):
    """Class defining AutoML results schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    metric = EnumFieldPrefix(Metrics)
    value = CustomFloatField(allow_none=True)


class Stats(Schema):
    """Class defining results stats schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    metric = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    value = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))


class JobSubset(Schema):
    """Class defining dataset job result total schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    status = EnumField(JobStatusEnum)
    action = EnumField(ActionEnum)
    eta = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))
    epoch = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        format=sys_int_format(),
        error="Epoch must be non-negative."
    )
    max_epoch = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        format=sys_int_format(),
        error="Max epoch should be non negative."
    )
    detailed_status_message = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=6400),
        allow_none=True
    )


class JobResult(Schema):
    """Class defining job results schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    # Metrics
    graphical = fields.List(
        fields.Nested(Graph, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )
    categorical = fields.List(
        fields.Nested(Category, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )
    kpi = fields.List(fields.Nested(KPI, allow_none=True), validate=fields.validate.Length(max=sys.maxsize))
    # AutoML
    epoch = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        format=sys_int_format(),
        error="Epoch must be non-negative."
    )
    max_epoch = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        format=sys_int_format(),
        error="Max epoch should be non negative."
    )
    automl_brain_info = fields.List(
        fields.Nested(Stats, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )
    automl_result = fields.List(
        fields.Nested(AutoMLResults, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )
    # Timing
    time_per_epoch = fields.Str(
        allow_none=True,
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=sys.maxsize)
    )
    time_per_iter = fields.Str(
        allow_none=True,
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=sys.maxsize)
    )
    cur_iter = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        format=sys_int_format()
    )
    eta = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))
    # General
    detailed_status = fields.Nested(DetailedStatus, allow_none=True)
    key_metric = fields.Float(allow_none=True)
    message = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))
    # Specs (only populated for AutoML experiments)
    specs = fields.Raw(allow_none=True)


class LoginReq(Schema):
    """Class defining login request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    ngc_key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    ngc_org_name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    enable_telemetry = fields.Bool(default=False, allow_none=True)  # NVAIE requires disable telemetry by default


class LoginRsp(Schema):
    """Class defining login response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    token = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))
    user_name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    user_email = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)


class GpuDetails(Schema):
    """Class defining telemetry request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    cluster = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    node = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    gpu_type = fields.Str(validate=validate.Length(max=2048))
    instance_type = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    gpu_count = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    cpu_cores = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    system_memory = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    gpu_memory = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    regions = fields.List(fields.Str(validate=validate.Length(max=2048)), allow_none=True)
    storage = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    driver_version = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    max_limit = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    current_used = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    current_available = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    node_type = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    backend_type = fields.Str(validate=validate.Length(max=2048), allow_none=True)


class TelemetryReq(Schema):
    """Class defining telemetry request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    version = fields.Str()
    network = fields.Str()
    action = fields.Str()
    success = fields.Bool()
    gpu = fields.List(fields.Str())
    time_lapsed = fields.Int(allow_none=True)
    user_error = fields.Bool(allow_none=True)
    client_type = fields.Str(allow_none=True)  # Client type: container, api, cli, sdk, ui, etc.
    automl_triggered = fields.Bool(allow_none=True)  # Whether job is triggered by AutoML


# Shared field definitions for cloud storage credentials
# These classes contain reusable field definitions that are mixed into concrete schemas
class AWSCredentialsFields:
    """Reusable field definitions for AWS-compatible storage credentials"""

    access_key = fields.Str(required=True, validate=validate.Length(min=1, max=2048))
    secret_key = fields.Str(required=True, validate=validate.Length(min=1, max=2048))
    cloud_region = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    endpoint_url = fields.Str(validate=[validate_endpoint_url, validate.Length(max=2048)], allow_none=True)
    cloud_bucket_name = fields.Str(validate=validate.Length(min=1, max=2048), allow_none=True)
    cloud_type = fields.Constant(CloudPullTypesEnum.aws.value)


class AzureCredentialsFields:
    """Reusable field definitions for Azure storage credentials"""

    account_name = fields.Str(required=True, validate=validate.Length(min=1, max=2048))
    access_key = fields.Str(required=True, validate=validate.Length(min=1, max=2048))
    cloud_region = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    endpoint_url = fields.Str(validate=[validate_endpoint_url, validate.Length(max=2048)], allow_none=True)
    cloud_bucket_name = fields.Str(validate=validate.Length(min=1, max=2048), allow_none=True)
    cloud_type = fields.Constant(CloudPullTypesEnum.azure.value)


class HuggingFaceCredentialsFields:
    """Reusable field definitions for Hugging Face credentials"""

    token = fields.Str(validate=validate.Length(max=2048))
    cloud_type = fields.Constant(CloudPullTypesEnum.huggingface.value)


class AWSCloudPull(AWSCredentialsFields, Schema):
    """Class defining AWS Cloud pull schema"""

    cloud_type = fields.Constant(CloudPullTypesEnum.aws.value)


class SeaweedfsCloudPull(AWSCredentialsFields, Schema):
    """Class defining Seaweed Cloud pull schema"""

    cloud_type = fields.Constant(CloudPullTypesEnum.seaweedfs.value)


class AzureCloudPull(AzureCredentialsFields, Schema):
    """Class defining Azure Cloud pull schema"""

    cloud_type = fields.Constant(CloudPullTypesEnum.azure.value)


class HuggingFaceCloudPull(HuggingFaceCredentialsFields, Schema):
    """Class defining Hugging Face Cloud pull schema"""

    cloud_type = fields.Constant(CloudPullTypesEnum.huggingface.value)


class StorageBackendAWS(AWSCredentialsFields, Schema):
    """AWS storage backend for Lepton - uses same credentials as AWSCloudPull"""

    storage_type = fields.Constant('aws')


class StorageBackendSeaweedfs(AWSCredentialsFields, Schema):
    """Seaweedfs storage backend for Lepton - uses same credentials as SeaweedfsCloudPull"""

    storage_type = fields.Constant('seaweedfs')


class StorageBackendAzure(AzureCredentialsFields, Schema):
    """Azure storage backend for Lepton - uses same credentials as AzureCloudPull"""

    storage_type = fields.Constant('azure')


class StorageBackendHuggingface(HuggingFaceCredentialsFields, Schema):
    """Huggingface storage backend for Lepton - uses same credentials as HuggingFaceCloudPull"""

    storage_type = fields.Constant('huggingface')


class LeptonStorageBackend(OneOfSchema):
    """Polymorphic storage backend for Lepton"""

    type_schemas = {
        "aws": StorageBackendAWS,
        "azure": StorageBackendAzure,
        "seaweedfs": StorageBackendSeaweedfs,
        "huggingface": StorageBackendHuggingface,
    }
    type_field = "storage_type"

    def get_obj_type(self, obj):
        """Determine the schema to use based on storage_type"""
        storage_type = obj.get("storage_type")
        if storage_type in self.type_schemas:
            return storage_type
        raise ValidationError(f"Invalid storage type: {storage_type}")


class SlurmCloudPull(Schema):
    """Class defining Slurm Cloud pull schema

    slurm_hostname must be a list of hostname strings for multi-host failover support.
    """

    slurm_user = fields.Str(validate=validate.Length(max=2048), required=True)
    slurm_hostname = fields.List(fields.Str(validate=validate.Length(max=2048)),
                                 required=True, validate=validate.Length(min=1))
    base_results_dir = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    cloud_type = fields.Constant(CloudPullTypesEnum.slurm.value)


class LocalBackendDetails(Schema):
    """Backend details for local execution - no additional parameters"""

    backend_type = fields.Constant("local")


class SlurmBackendDetails(Schema):
    """Backend details for Slurm execution"""

    backend_type = fields.Constant("slurm")
    partition = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    cluster_name = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    slurm_metadata = fields.Dict(allow_none=True)  # For storing slurm_job_id and other runtime metadata


class LeptonBackendDetails(Schema):
    """Backend details for Lepton execution"""

    backend_type = fields.Constant("lepton")
    platform_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)


class BackendDetails(OneOfSchema):
    """Class defining polymorphic backend execution details schema"""

    type_schemas = {
        "local": LocalBackendDetails,
        "slurm": SlurmBackendDetails,
        "lepton": LeptonBackendDetails,
    }
    type_field = "backend_type"

    def get_obj_type(self, obj):
        """Determine the schema to use based on the properties of the Python object"""
        backend_type = obj.get("backend_type")
        if backend_type in self.type_schemas:
            return backend_type
        raise fields.ValidationError(f"Invalid backend type: {backend_type}")


class LeptonCloudPull(Schema):
    """Class defining Lepton Cloud pull schema

    Lepton workspaces can use AWS S3 or Azure Blob storage.
    Provide either AWS credentials (access_key, secret_key) or Azure credentials (account_name, access_key).
    """

    # Lepton-specific fields
    lepton_workspace_id = fields.Str(validate=validate.Length(max=2048), required=True)
    lepton_auth_token = fields.Str(validate=validate.Length(max=2048), required=True)
    cloud_type = fields.Constant(CloudPullTypesEnum.lepton.value)

    # AWS fields (optional - required if using AWS storage)
    access_key = fields.Str(validate=validate.Length(min=1, max=2048), allow_none=True)
    secret_key = fields.Str(validate=validate.Length(min=1, max=2048), allow_none=True)

    # Azure fields (optional - required if using Azure storage)
    account_name = fields.Str(validate=validate.Length(min=1, max=2048), allow_none=True)

    # Shared fields
    cloud_region = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    endpoint_url = fields.Str(validate=[validate_endpoint_url, validate.Length(max=2048)], allow_none=True)
    cloud_bucket_name = fields.Str(validate=validate.Length(min=1, max=2048), allow_none=True)

    @validates_schema
    def validate_storage_credentials(self, data, **kwargs):
        """Ensure either AWS or Azure credentials are provided, but not both"""
        has_aws = data.get('access_key') and data.get('secret_key')
        has_azure = data.get('account_name') and data.get('access_key') and not data.get('secret_key')
        has_lepton = data.get('lepton_workspace_id') and data.get('lepton_auth_token')

        if not (has_aws or has_azure) or not has_lepton:
            raise ValidationError(
                'Must provide either AWS credentials (access_key, secret_key) '
                'or Azure credentials (account_name, access_key) and Lepton workspace ID and auth token'
            )


class CloudSpecificDetails(OneOfSchema):
    """Class defining a polymorphic cloud specific details schema"""

    type_schemas = {
        "aws": AWSCloudPull,
        "azure": AzureCloudPull,
        "huggingface": HuggingFaceCloudPull,
        "seaweedfs": SeaweedfsCloudPull,
        "lepton": LeptonCloudPull,
        "slurm": SlurmCloudPull,
    }
    type_field = "cloud_type"

    def get_obj_type(self, obj):
        """Determine the schema to use based on the properties of the Python object"""
        cloud_type = obj.get("cloud_type")
        if cloud_type in [e.value for e in CloudPullTypesEnum]:
            return cloud_type
        raise fields.ValidationError(f"Invalid cloud type: {cloud_type}")


class WorkspaceReq(Schema):
    """Class defining Cloud Workspace request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    version = fields.Str(format="regex", regex=r'^\d+\.\d+\.\d+$', validate=fields.validate.Length(max=10))
    cloud_type = EnumField(CloudPullTypesEnum, allow_none=False)
    cloud_specific_details = fields.Nested(CloudSpecificDetails, allow_none=False)
    force_create = fields.Bool(allow_none=True)

    @validates_schema
    def validate_cloud_specific_details(self, data, **kwargs):
        """Return schema based on cloud_type and validate credentials"""
        cloud_type = data.get('cloud_type')

        if cloud_type:
            # First, validate the schema structure
            if cloud_type == CloudPullTypesEnum.aws:
                schema = AWSCloudPull()
            elif cloud_type == CloudPullTypesEnum.azure:
                schema = AzureCloudPull()
            elif cloud_type == CloudPullTypesEnum.seaweedfs:
                schema = AWSCloudPull()
            elif cloud_type == CloudPullTypesEnum.huggingface:
                schema = HuggingFaceCloudPull()
            elif cloud_type == CloudPullTypesEnum.lepton:
                schema = LeptonCloudPull()
            elif cloud_type == CloudPullTypesEnum.slurm:
                schema = SlurmCloudPull()
            else:
                schema = Schema()

            try:
                # Validate schema structure
                schema.load(data.get('cloud_specific_details', {}), unknown=EXCLUDE)
            except ValidationError:
                # Re-raise ValidationError as-is
                raise
            except Exception as e:
                raise fields.ValidationError(str(e))


class WorkspaceBackupReq(Schema):
    """Class defining workspace backup schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    backup_file_name = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    workspace_metadata = fields.Nested(WorkspaceReq, allow_none=False)


class WorkspaceRsp(Schema):
    """Class defining Cloud pull schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    created_on = DateTimeField(metadata={"maxLength": 24})
    last_modified = DateTimeField(metadata={"maxLength": 24})
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    version = fields.Str(format="regex", regex=r'^\d+\.\d+\.\d+$', validate=fields.validate.Length(max=10))
    cloud_type = EnumField(CloudPullTypesEnum, allow_none=False)


class WorkspaceListRsp(Schema):
    """Class defining workspace list response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    workspaces = fields.List(fields.Nested(WorkspaceRsp), validate=validate.Length(max=sys.maxsize))
    pagination_info = fields.Nested(PaginationInfo, allow_none=True)


class DatasetUriLst(Schema):
    """Class defining dataset actions schema"""

    dataset_uris = fields.List(
        fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=500),
            allow_none=True
        ),
        validate=validate.Length(max=sys.maxsize)
    )


class DatasetActions(Schema):
    """Class defining dataset actions schema"""

    parent_job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    action = EnumField(ActionEnum)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    specs = fields.Raw()
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    backend_details = fields.Nested(BackendDetails, allow_none=True)


class LstStr(Schema):
    """Class defining dataset actions schema"""

    dataset_formats = fields.List(
        fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=500),
            allow_none=True
        ),
        validate=validate.Length(max=sys.maxsize)
    )
    accepted_dataset_intents = fields.List(
        EnumField(DatasetIntentEnum),
        allow_none=True,
        validate=validate.Length(max=4)
    )


class DatasetReq(Schema):
    """Class defining dataset request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    docker_env_vars = fields.Dict(
        keys=EnumField(AllowedDockerEnvVariables),
        values=fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=500),
            allow_none=True
        )
    )
    version = fields.Str(format="regex", regex=r'^\d+\.\d+\.\d+$', validate=fields.validate.Length(max=10))
    logo = fields.URL(validate=fields.validate.Length(max=2048))
    type = EnumField(DatasetType)
    format = EnumField(DatasetFormat)
    workspace = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    url = fields.URL(validate=fields.validate.Length(max=2048))  # For HuggingFace and Self_hosted
    cloud_file_path = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=2048),
        allow_none=True
    )
    client_url = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_id = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_secret = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    filters = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    status = EnumField(PullStatus)
    use_for = fields.List(EnumField(DatasetIntentEnum), allow_none=True, validate=validate.Length(max=4))
    base_experiment_pull_complete = EnumField(PullStatus)
    base_experiment_ids = fields.List(
        fields.Str(format="uuid", validate=fields.validate.Length(max=36)),
        validate=validate.Length(max=2)
    )
    skip_validation = fields.Bool(allow_none=True)
    authorized_party_nca_id = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=2048),
        allow_none=True
    )
    tags = fields.List(
        fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=36)
        ),
        validate=validate.Length(max=16)
    )
    force_create = fields.Bool(allow_none=True)


class DatasetJob(Schema):
    """Class defining dataset job result total schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    parent_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    created_on = DateTimeField(metadata={"maxLength": 24})
    last_modified = DateTimeField(metadata={"maxLength": 24})
    action = EnumField(ActionEnum)
    status = EnumField(JobStatusEnum)
    job_details = fields.Dict(
        keys=fields.Str(format="uuid", validate=fields.validate.Length(max=36)),
        values=fields.Nested(JobResult),
        validate=validate.Length(max=sys.maxsize)
    )
    specs = fields.Raw(allow_none=True)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    backend_details = fields.Nested(BackendDetails, allow_none=True)
    dataset_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    epoch_numbers = fields.List(
        fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize)),
        validate=validate.Length(max=sys.maxsize),
        allow_none=True
    )


class DatasetRsp(Schema):
    """Class defining dataset response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        load_only = ("user_id", "docker_env_vars", "client_id", "client_secret", "filters")
        unknown = EXCLUDE

    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    created_on = DateTimeField(metadata={"maxLength": 24})
    last_modified = DateTimeField(metadata={"maxLength": 24})
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    docker_env_vars = fields.Dict(
        keys=EnumField(AllowedDockerEnvVariables),
        values=fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=500),
            allow_none=True
        )
    )
    version = fields.Str(format="regex", regex=r'^\d+\.\d+\.\d+$', validate=fields.validate.Length(max=10))
    logo = fields.URL(validate=fields.validate.Length(max=2048), allow_none=True)
    type = EnumField(DatasetType)
    format = EnumField(DatasetFormat)
    workspace = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    url = fields.URL(validate=fields.validate.Length(max=2048), allow_none=True)  # For HuggingFace and Self_hosted
    cloud_file_path = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=2048),
        allow_none=True
    )
    actions = fields.List(EnumField(ActionEnum), allow_none=True, validate=validate.Length(max=sys.maxsize))
    jobs = fields.Dict(
        keys=fields.Str(format="uuid", validate=fields.validate.Length(max=36)),
        values=fields.Nested(JobSubset),
        validate=validate.Length(max=sys.maxsize)
    )
    client_url = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_id = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_secret = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    filters = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    status = EnumField(PullStatus)
    use_for = fields.List(EnumField(DatasetIntentEnum), allow_none=True, validate=validate.Length(max=4))
    base_experiment_pull_complete = EnumField(PullStatus)
    base_experiment_ids = fields.List(
        fields.Str(format="uuid", validate=fields.validate.Length(max=36)),
        validate=validate.Length(max=2)
    )
    skip_validation = fields.Bool(allow_none=True)
    authorized_party_nca_id = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=2048),
        allow_none=True
    )
    validation_details = fields.Nested(ValidationDetails, allow_none=True)
    tags = fields.List(
        fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=36)
        ),
        validate=validate.Length(max=16)
    )


class DatasetListRsp(Schema):
    """Class defining dataset list response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    datasets = fields.List(fields.Nested(DatasetRsp), validate=validate.Length(max=sys.maxsize))
    pagination_info = fields.Nested(PaginationInfo, allow_none=True)


class ExperimentActions(Schema):
    """Class defining job actions schema"""

    parent_job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    action = EnumField(ActionEnum)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    specs = fields.Raw()
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    backend_details = fields.Nested(BackendDetails, allow_none=True)


class PublishModel(Schema):
    """Class defining Publish model schema"""

    display_name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    team_name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    # format, framework, precision - to be determined by backend


class JobResume(Schema):
    """Class defining job resume request schema"""

    parent_job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    specs = fields.Raw(allow_none=True)
    backend_details = fields.Nested(BackendDetails, allow_none=True)


class ParameterRange(Schema):
    """Schema for parameter attributes (used for both default and custom)"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    parameter = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=500),
        required=False  # Not required when used as nested schema
    )
    default_value = fields.Raw(allow_none=True)  # Only used for default section
    valid_min = fields.Raw(allow_none=True)  # Can be float or list of floats
    valid_max = fields.Raw(allow_none=True)  # Can be float or list of floats
    valid_options = fields.List(fields.Raw(), allow_none=True)
    option_weights = fields.List(fields.Float(), allow_none=True)  # Weights for valid_options
    math_cond = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), allow_none=True)
    depends_on = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    parent_param = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    # When True, skip network-specific logic and treat as pure float for optimization
    disable_list = fields.Bool(allow_none=True)


class AutoMLParameterDetail(Schema):
    """Class defining individual parameter detail schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    parameter = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    value_type = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    default = fields.Nested(ParameterRange)
    custom = fields.Nested(ParameterRange, allow_none=True)


class AutoMLParameterDetailsRsp(Schema):
    """Class defining response schema for getting parameter details"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    parameter_details = fields.List(fields.Nested(AutoMLParameterDetail), validate=validate.Length(max=sys.maxsize))


class AutoMLUpdateParameterRangesReq(Schema):
    """Class defining request schema for updating parameter ranges"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), required=True)
    network_arch = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), required=True)
    parameter_ranges = fields.List(
        fields.Nested(ParameterRange),
        validate=validate.Length(min=1, max=sys.maxsize),
        required=True
    )


# Algorithm-specific parameter schemas (nested structure)
class AutoMLBayesianParams(Schema):
    """Schema for Bayesian and BFBO algorithm parameters"""

    class Meta:
        """Marshmallow schema configuration"""

        ordered = True
        unknown = EXCLUDE

    automl_max_recommendations = fields.Int(
        format="int64", validate=validate.Range(min=1, max=sys.maxsize), required=True
    )


class AutoMLHyperbandParams(Schema):
    """Schema for Hyperband algorithm parameters"""

    class Meta:
        """Marshmallow schema configuration"""

        ordered = True
        unknown = EXCLUDE

    automl_max_epochs = fields.Int(format="int64", validate=validate.Range(min=2, max=sys.maxsize), required=True)
    automl_reduction_factor = fields.Int(format="int64", validate=validate.Range(min=2, max=sys.maxsize), required=True)
    epoch_multiplier = fields.Int(format="int64", validate=validate.Range(min=1, max=sys.maxsize), required=True)


class AutoMLBOHBParams(AutoMLHyperbandParams):
    """Schema for BOHB algorithm parameters"""

    automl_kde_samples = fields.Int(format="int64", validate=validate.Range(min=1, max=sys.maxsize), allow_none=True)
    automl_top_n_percent = fields.Float(validate=validate.Range(min=0.0, max=100.0), allow_none=True)
    automl_min_points_in_model = fields.Int(
        format="int64", validate=validate.Range(min=1, max=sys.maxsize), allow_none=True
    )


class AutoMLASHAParams(AutoMLHyperbandParams):
    """Schema for ASHA algorithm parameters"""

    automl_max_concurrent = fields.Int(format="int64", validate=validate.Range(min=1, max=sys.maxsize), required=True)
    automl_max_trials = fields.Int(format="int64", validate=validate.Range(min=1, max=sys.maxsize), allow_none=True)


class AutoMLDEHBParams(AutoMLHyperbandParams):
    """Schema for DEHB algorithm parameters"""

    automl_mutation_factor = fields.Float(validate=validate.Range(min=0.0, max=2.0), allow_none=True)
    automl_crossover_prob = fields.Float(validate=validate.Range(min=0.0, max=1.0), allow_none=True)


class AutoMLHyperBandESParams(AutoMLHyperbandParams):
    """Schema for HyperBand with Early Stopping algorithm parameters"""

    automl_early_stop_threshold = fields.Float(validate=validate.Range(min=0.0, max=1.0), allow_none=True)
    automl_min_early_stop_epochs = fields.Int(
        format="int64", validate=validate.Range(min=1, max=sys.maxsize), allow_none=True
    )


class AutoMLPBTParams(Schema):
    """Schema for Population-Based Training algorithm parameters"""

    class Meta:
        """Marshmallow schema configuration"""

        ordered = True
        unknown = EXCLUDE

    automl_population_size = fields.Int(format="int64", validate=validate.Range(min=1, max=sys.maxsize), required=True)
    automl_eval_interval = fields.Int(format="int64", validate=validate.Range(min=1, max=sys.maxsize), required=True)
    automl_perturbation_factor = fields.Float(validate=validate.Range(min=1.0, max=10.0), allow_none=True)


class AutoML(Schema):
    """AutoML schema with nested algorithm-specific parameters"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = RAISE

    automl_enabled = fields.Bool(allow_none=True)
    automl_algorithm = EnumField(AutoMLAlgorithm, allow_none=True)
    automl_delete_intermediate_ckpt = fields.Bool(allow_none=True)
    override_automl_disabled_params = fields.Bool(allow_none=True)
    automl_hyperparameters = fields.Str(
        format="regex", regex=r'\[.*\]', validate=fields.validate.Length(max=5000), allow_none=True
    )
    automl_range_override = fields.List(
        fields.Nested(ParameterRange),
        validate=validate.Length(max=sys.maxsize),
        allow_none=True
    )
    # Nested algorithm-specific parameters
    algorithm_specific_params = fields.Field(allow_none=True)
    metric = fields.Str(allow_none=True)

    @validates_schema
    def validate_algorithm_specific_params(self, data, **kwargs):
        """Validate algorithm-specific parameters based on algorithm type"""
        if not data.get('automl_enabled', False):
            return

        algorithm = data.get('automl_algorithm')
        if not algorithm:
            raise ValidationError('automl_algorithm is required when automl_enabled is True')

        # Convert enum to string if needed
        algo_str = algorithm.value if hasattr(algorithm, 'value') else str(algorithm)

        # Select appropriate schema based on algorithm
        if algo_str in ('bayesian', 'b', 'bfbo'):
            schema = AutoMLBayesianParams()
        elif algo_str in ('hyperband', 'h'):
            schema = AutoMLHyperbandParams()
        elif algo_str == 'bohb':
            schema = AutoMLBOHBParams()
        elif algo_str == 'asha':
            schema = AutoMLASHAParams()
        elif algo_str == 'dehb':
            schema = AutoMLDEHBParams()
        elif algo_str in ('hyperband_es', 'hes'):
            schema = AutoMLHyperBandESParams()
        elif algo_str == 'pbt':
            schema = AutoMLPBTParams()
        else:
            raise ValidationError(f'Unknown automl_algorithm: {algo_str}')

        # Always validate algorithm-specific parameters (required fields will error if missing)
        params = data.get('algorithm_specific_params') or {}
        try:
            schema.load(params, unknown=EXCLUDE)
        except ValidationError:
            raise
        except Exception as e:
            raise fields.ValidationError(str(e))


class BaseExperimentMetadata(Schema):
    """Class defining base experiment metadata schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    task = EnumField(BaseExperimentTask, by_value=True, allow_none=True)
    domain = EnumField(BaseExperimentDomain, by_value=True, allow_none=True)
    backbone_type = EnumField(BaseExperimentBackboneType, by_value=True, allow_none=True)
    backbone_class = EnumField(BaseExperimentBackboneClass, by_value=True, allow_none=True)
    num_parameters = fields.Str(
        format="regex",
        regex=r'^\d+(\.\d+)?M$',
        validate=fields.validate.Length(max=10),
        allow_none=True
    )
    accuracy = fields.Str(
        format="regex",
        regex=r'^\d{1,3}(\.\d+)?%$',
        validate=fields.validate.Length(max=10),
        allow_none=True
    )
    license = EnumField(BaseExperimentLicense, by_value=True, allow_none=True)
    model_card_link = fields.URL(validate=fields.validate.Length(max=2048), allow_none=True)
    is_backbone = fields.Bool()
    is_trainable = fields.Bool()
    spec_file_present = fields.Bool()
    specs = fields.Raw(allow_none=True)


class InferenceMicroserviceReq(Schema):
    """Class defining inference microservice request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    parent_job_id = fields.Str(
        format="uuid",
        validate=fields.validate.Length(max=36),
        description="Parent job ID",
        example="12345678-1234-1234-1234-123456789012",
        allow_none=True,
        required=False
    )
    kind = EnumField(
        JobKindEnum,
        description="Job kind",
        example="experiment",
        allow_none=True,
        required=False
    )
    model_path = fields.Str(
        validate=fields.validate.Length(max=2048),
        description="Path to the model",
        example="/workspace/model",
        allow_none=True,
        required=False
    )
    hf_model = fields.Str(
        validate=fields.validate.Length(max=2048),
        description="HuggingFace model name (e.g., meta-llama/Llama-3-8B-Instruct, Qwen/Qwen-VL-Chat)",
        example="meta-llama/Llama-3-8B-Instruct",
        allow_none=True,
        required=False
    )
    enable_lora = fields.Bool(
        description="Enable LoRA for inference",
        default=False
    )
    base_model_path = fields.Str(
        description="Base model path (e.g., hf_model://nvidia/Cosmos-Reason1-7B)",
        required=False,
        allow_none=True
    )
    torch_dtype = fields.Str(
        validate=fields.validate.Length(max=50),
        description="PyTorch data type for HuggingFace models (auto, float16, bfloat16, float32)",
        example="auto",
        allow_none=True,
        required=False
    )
    device_map = fields.Str(
        validate=fields.validate.Length(max=50),
        description="Device mapping strategy for HuggingFace models (auto, cuda, cpu)",
        example="auto",
        allow_none=True,
        required=False
    )
    docker_image = fields.Str(
        validate=fields.validate.Length(max=2048),
        description="Docker image for inference",
        example="nvcr.io/nvidia/vila-inference:latest"
    )
    gpu_type = fields.Str(
        validate=fields.validate.Length(max=2048),
        description="GPU type",
        example="h100"
    )
    num_gpus = fields.Int(
        format="int64",
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        description="Number of GPUs required",
        example=1
    )
    docker_env_vars = fields.Dict(
        keys=fields.Str(validate=validate.OneOf([e.value for e in AllowedDockerEnvVariables])),
        values=fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=500),
            allow_none=True
        )
    )
    checkpoint_choose_method = EnumField(CheckpointChooseMethodEnum)
    checkpoint_epoch_number = fields.Dict(
        keys=fields.Str(
            format="regex",
            regex=(
                r'(from_epoch_number|latest_model|best_model)_[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}$'
            ),
            validate=fields.validate.Length(max=100),
            allow_none=True
        ),
        values=fields.Int(
            format="int64",
            validate=validate.Range(min=0, max=sys.maxsize),
            allow_none=True
        )
    )
    network_arch = EnumField(ExperimentNetworkArch, allow_none=True)
    custom_pipeline_loader = fields.Str(
        validate=fields.validate.Length(max=10000),
        description=(
            "Python code string defining a load_pipeline(model_name, **kwargs) "
            "function for custom model loading"
        ),
        example=(
            "def load_pipeline(model_name, **kwargs):\n"
            "    from diffusers import WanPipeline\n"
            "    return WanPipeline.from_pretrained(model_name), 'diffusion'"
        ),
        allow_none=True,
        required=False
    )
    custom_inference_fn = fields.Str(
        validate=fields.validate.Length(max=10000),
        description="Python code string defining a run_inference(pipeline, **kwargs) function for custom inference",
        example=(
            "def run_inference(pipeline, **kwargs):\n"
            "    output = pipeline(kwargs['prompt'])\n"
            "    return {'response': 'done', 'frames': output.frames}"
        ),
        allow_none=True,
        required=False
    )


class InferenceMicroserviceRsp(Schema):
    """Class defining inference microservice response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    job_id = fields.Str(
        format="uuid",
        validate=fields.validate.Length(max=36),
        allow_none=True,
        description="Unique job ID for this Inference Microservice"
    )
    status = fields.Str(
        validate=fields.validate.Length(max=2048),
        description="Service status"
    )
    message = fields.Str(
        validate=fields.validate.Length(max=2048),
        description="Success message"
    )


class InferenceReq(Schema):
    """Class defining inference request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    input = fields.List(
        fields.Str(
            description="Base64-encoded images/videos with data URI format (data:image/jpeg;base64,...)",
            required=False
        ),
        allow_none=True
    )
    media = fields.Str(
        description="Cloud path to media file (e.g., aws://bucket/path/to/video.mp4)",
        required=False,
        allow_none=True
    )
    images = fields.List(
        fields.Str(
            description="Image paths/URLs for VLM inference",
            required=False
        ),
        allow_none=True
    )
    model = fields.Str(
        description="Model identifier (e.g. nvidia/nvdino-v2)",
        required=False,
        allow_none=True
    )
    prompt = fields.Str(
        description="Text prompt for LLM/VLM inference",
        required=False,
        allow_none=True,
        default=""
    )
    system_prompt = fields.Str(
        description="System prompt for chat models",
        required=False,
        allow_none=True
    )
    max_new_tokens = fields.Int(
        format="int64",
        validate=validate.Range(min=1, max=32768),
        description="Maximum number of new tokens to generate",
        default=512,
        allow_none=True
    )
    temperature = fields.Float(
        validate=validate.Range(min=0.0, max=2.0),
        description="Sampling temperature (0.0 = deterministic, higher = more random)",
        default=0.7,
        allow_none=True
    )
    top_p = fields.Float(
        validate=validate.Range(min=0.0, max=1.0),
        description="Nucleus sampling parameter",
        default=0.9,
        allow_none=True
    )
    top_k = fields.Int(
        format="int64",
        validate=validate.Range(min=1, max=1000),
        description="Top-k sampling parameter",
        default=50,
        allow_none=True
    )
    enable_lora = fields.Bool(
        description="Enable LoRA for inference",
        required=False,
        allow_none=True
    )
    base_model_path = fields.Str(
        description="Base model path (e.g., hf_model://nvidia/Cosmos-Reason1-7B)",
        required=False,
        allow_none=True
    )
    # Diffusion model parameters (for Cosmos-Predict2, Stable Diffusion, etc.)
    negative_prompt = fields.Str(
        description="Negative prompt for diffusion models (what to avoid in generation)",
        required=False,
        allow_none=True
    )
    num_inference_steps = fields.Int(
        format="int64",
        validate=validate.Range(min=1, max=1000),
        description="Number of denoising steps for diffusion models (default: 50)",
        default=50,
        allow_none=True
    )
    guidance_scale = fields.Float(
        validate=validate.Range(min=0.0, max=50.0),
        description="Classifier-free guidance scale for diffusion models (default: 7.5)",
        default=7.5,
        allow_none=True
    )
    width = fields.Int(
        format="int64",
        validate=validate.Range(min=64, max=4096),
        description="Output image width for diffusion models",
        required=False,
        allow_none=True
    )
    height = fields.Int(
        format="int64",
        validate=validate.Range(min=64, max=4096),
        description="Output image height for diffusion models",
        required=False,
        allow_none=True
    )
    seed = fields.Int(
        format="int64",
        description="Random seed for reproducible diffusion generation",
        required=False,
        allow_none=True
    )
    num_images = fields.Int(
        format="int64",
        validate=validate.Range(min=1, max=16),
        description="Number of images to generate (default: 1)",
        default=1,
        allow_none=True
    )


class LoadAirgappedExperimentsReq(Schema):
    """Class defining load airgapped models request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    workspace_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    models_base_dir = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=2048),
        allow_none=True
    )


class LoadAirgappedExperimentsRsp(Schema):
    """Class defining load airgapped models response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    success = fields.Bool()
    message = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=2048)
    )
    experiments_loaded = fields.Int(
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        format=sys_int_format()
    )
    experiments_failed = fields.Int(
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        format=sys_int_format()
    )


class DatasetJobReq(Schema):
    """Class defining an dataset job request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    dataset_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True, load_default=None)
    parent_job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    action = EnumField(ActionEnum)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    specs = fields.Raw()
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    backend_details = fields.Nested(BackendDetails, allow_none=True)
    epoch_numbers = fields.List(
        fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize)),
        validate=validate.Length(max=sys.maxsize),
        allow_none=True
    )
    kind = fields.Constant(JobKindEnum.dataset.value)
    base_experiment_pull_complete = EnumField(PullStatus)
    base_experiment_ids = fields.List(
        fields.Str(format="uuid", validate=fields.validate.Length(max=36)),
        validate=validate.Length(max=2)
    )
    # New fields for direct dataset paths (alternative to UUID-based dataset_id)
    train_dataset_uris = fields.List(
        fields.Str(validate=fields.validate.Length(max=1000)),
        validate=validate.Length(max=sys.maxsize),
        allow_none=True,
        metadata={"description": "List of dataset URIs (aws://, azure://, lustre://, file://, or local)"}
    )
    eval_dataset_uri = fields.Str(
        validate=fields.validate.Length(max=1000),
        allow_none=True,
        metadata={"description": "Evaluation dataset URI"}
    )
    inference_dataset_uri = fields.Str(
        validate=fields.validate.Length(max=1000),
        allow_none=True,
        metadata={"description": "Inference dataset URI"}
    )
    calibration_dataset_uri = fields.Str(
        validate=fields.validate.Length(max=1000),
        allow_none=True,
        metadata={"description": "Calibration dataset URI"}
    )
    dataset_format = fields.Str(
        validate=fields.validate.Length(max=100),
        allow_none=True,
        metadata={"description": "Dataset format (e.g., 'kitti', 'coco', 'custom')"}
    )
    dataset_type = fields.Str(
        validate=fields.validate.Length(max=100),
        allow_none=True,
        metadata={
            "description": "Dataset type (e.g., 'object_detection', 'classification'). "
                           "Required when using direct dataset paths."
        }
    )
    skip_dataset_validation = fields.Bool(
        allow_none=True,
        metadata={"description": "Skip dataset structure validation. Default: False."}
    )
    workspace = fields.Str(
        format="uuid",
        validate=fields.validate.Length(max=36),
        allow_none=True,
        metadata={"description": "Workspace ID. Used when creating jobs with direct dataset paths."}
    )
    force_create = fields.Bool(allow_none=True)


class ExperimentJobReq(Schema):
    """Class defining an experiment job request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    description = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=1000),
        allow_none=True
    )  # Model version description - not changing variable name for backward compatibility
    model_description = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=1000),
        allow_none=True
    )  # Description common to all versions of models
    version = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    logo = fields.URL(validate=fields.validate.Length(max=2048), allow_none=True)
    ngc_path = fields.Str(
        format="regex",
        regex=r'^\w+(/[\w-]+)?/[\w-]+:[\w.-]+$',
        validate=fields.validate.Length(max=250)
    )
    workspace = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    sha256_digest = fields.Dict(allow_none=True)
    base_experiment_pull_complete = EnumField(PullStatus)
    additional_id_info = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=100),
        allow_none=True
    )
    docker_env_vars = fields.Dict(
        keys=EnumField(AllowedDockerEnvVariables),
        values=fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=500),
            allow_none=True
        )
    )
    checkpoint_choose_method = EnumField(CheckpointChooseMethodEnum)
    checkpoint_epoch_number = fields.Dict(
        keys=fields.Str(
            format="regex",
            regex=(
                r'(from_epoch_number|latest_model|best_model)_[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}$'
            ),
            validate=fields.validate.Length(max=100),
            allow_none=True
        ),
        values=fields.Int(
            format="int64",
            validate=validate.Range(min=0, max=sys.maxsize),
            allow_none=True
        )
    )
    encryption_key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    network_arch = EnumField(ExperimentNetworkArch)
    base_experiment_ids = fields.List(
        fields.Str(
            format="uuid",
            validate=fields.validate.Length(max=36)
        ),
        validate=validate.Length(max=2)
    )
    eval_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    inference_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    calibration_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    train_datasets = fields.List(
        fields.Str(
            format="uuid",
            validate=fields.validate.Length(max=36)
        ),
        validate=validate.Length(max=sys.maxsize)
    )
    # New fields for direct dataset paths (alternative to UUID-based dataset references)
    train_dataset_uris = fields.List(
        fields.Str(validate=fields.validate.Length(max=1000)),
        validate=validate.Length(max=sys.maxsize),
        allow_none=True,
        metadata={"description": "List of dataset URIs (aws://, azure://, lustre://, file://, or local)"}
    )
    eval_dataset_uri = fields.Str(
        validate=fields.validate.Length(max=1000),
        allow_none=True,
        metadata={"description": "Evaluation dataset URI"}
    )
    inference_dataset_uri = fields.Str(
        validate=fields.validate.Length(max=1000),
        allow_none=True,
        metadata={"description": "Inference dataset URI"}
    )
    calibration_dataset_uri = fields.Str(
        validate=fields.validate.Length(max=1000),
        allow_none=True,
        metadata={"description": "Calibration dataset URI"}
    )
    dataset_format = fields.Str(
        validate=fields.validate.Length(max=100),
        allow_none=True,
        metadata={
            "description": "Dataset format (e.g., 'llava', 'coco', 'kitti'). "
                           "If not specified, uses network's default format"
        }
    )
    read_only = fields.Bool()
    public = fields.Bool()
    automl_settings = fields.Nested(AutoML, allow_none=True)
    metric = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), allow_none=True)
    model_params = fields.Dict(allow_none=True)
    experiment_actions = fields.List(
        fields.Nested(ExperimentActions, allow_none=True),
        validate=validate.Length(max=sys.maxsize)
    )
    tensorboard_enabled = fields.Bool(allow_none=True)
    tags = fields.List(
        fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=36)
        ),
        validate=validate.Length(max=16)
    )
    retry_experiment_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    authorized_party_nca_id = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=2048),
        allow_none=True
    )
    parent_job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    action = EnumField(ActionEnum)
    specs = fields.Raw()
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    backend_details = fields.Nested(BackendDetails, allow_none=True)
    epoch_numbers = fields.List(
        fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize)),
        validate=validate.Length(max=sys.maxsize),
        allow_none=True
    )
    skip_dataset_validation = fields.Bool(
        allow_none=True,
        metadata={
            "description": "Skip dataset structure validation at job creation. "
                           "Default: False. Set to True to bypass validation checks."
        }
    )
    kind = fields.Constant(JobKindEnum.experiment.value)
    force_create = fields.Bool(allow_none=True)


class JobReq(OneOfSchema):
    """Class defining a polymorphic job request schema"""

    type_schemas = {
        "dataset": DatasetJobReq,
        "experiment": ExperimentJobReq
    }
    type_field = "kind"

    def get_obj_type(self, obj):
        """Determine the schema to use based on the properties of the Python object"""
        kind = obj.get("kind")
        if kind in JobKindEnum:
            return kind
        raise fields.ValidationError(f"Invalid job kind: {kind}")


class DatasetJobRsp(Schema):
    """Class defining dataset job response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    parent_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    created_on = DateTimeField(metadata={"maxLength": 24})
    last_modified = DateTimeField(metadata={"maxLength": 24})
    action = EnumField(ActionEnum)
    status = EnumField(JobStatusEnum)
    job_details = fields.Dict(
        keys=fields.Str(format="uuid", validate=fields.validate.Length(max=36)),
        values=fields.Nested(JobResult),
        validate=validate.Length(max=sys.maxsize)
    )
    specs = fields.Raw(allow_none=True)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    backend_details = fields.Nested(BackendDetails, allow_none=True)
    dataset_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    epoch_numbers = fields.List(
        fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize)),
        validate=validate.Length(max=sys.maxsize),
        allow_none=True
    )
    kind = fields.Constant(JobKindEnum.dataset.value)
    base_experiment_pull_complete = EnumField(PullStatus)
    base_experiment_ids = fields.List(
        fields.Str(format="uuid", validate=fields.validate.Length(max=36)),
        validate=validate.Length(max=2)
    )


class ExperimentJobRsp(Schema):
    """Class defining experiment job response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        load_only = ("user_id", "docker_env_vars")
        unknown = EXCLUDE

    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    created_on = DateTimeField(metadata={"maxLength": 24})
    last_modified = DateTimeField(metadata={"maxLength": 24})
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    description = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=1000),
        allow_none=True
    )  # Model version description - not changing variable name for backward compatibility
    model_description = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=1000),
        allow_none=True
    )  # Description common to all versions of models
    version = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    logo = fields.URL(validate=fields.validate.Length(max=2048), allow_none=True)
    ngc_path = fields.Str(
        format="regex",
        regex=r'^\w+(/[\w-]+)?/[\w-]+:[\w.-]+$',
        validate=fields.validate.Length(max=250)
    )
    workspace = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    sha256_digest = fields.Dict(allow_none=True)
    base_experiment_pull_complete = EnumField(PullStatus)
    additional_id_info = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=100),
        allow_none=True
    )
    docker_env_vars = fields.Dict(
        keys=EnumField(AllowedDockerEnvVariables),
        values=fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=500),
            allow_none=True
        )
    )
    checkpoint_choose_method = EnumField(CheckpointChooseMethodEnum)
    checkpoint_epoch_number = fields.Dict(
        keys=fields.Str(
            format="regex",
            regex=(
                r'(from_epoch_number|latest_model|best_model)_[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}$'
            ),
            validate=fields.validate.Length(max=100),
            allow_none=True
        ),
        values=fields.Int(
            format="int64",
            validate=validate.Range(min=0, max=sys.maxsize),
            allow_none=True
        )
    )
    encryption_key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    network_arch = EnumField(ExperimentNetworkArch)
    base_experiment_ids = fields.List(
        fields.Str(
            format="uuid",
            validate=fields.validate.Length(max=36)
        ),
        validate=validate.Length(max=2)
    )
    dataset_type = EnumField(DatasetType)
    dataset_formats = fields.List(EnumField(DatasetFormat), allow_none=True, validate=validate.Length(max=sys.maxsize))
    accepted_dataset_intents = fields.List(
        EnumField(DatasetIntentEnum, allow_none=True),
        validate=validate.Length(max=sys.maxsize)
    )
    eval_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    inference_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    calibration_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    train_datasets = fields.List(
        fields.Str(
            format="uuid",
            validate=fields.validate.Length(max=36)
        ),
        validate=validate.Length(max=sys.maxsize)
    )
    # New fields for direct dataset paths (alternative to UUID-based dataset references)
    train_dataset_uris = fields.List(
        fields.Str(validate=fields.validate.Length(max=1000)),
        validate=validate.Length(max=sys.maxsize),
        allow_none=True,
        metadata={"description": "List of dataset URIs (aws://, azure://, lustre://, file://, or local)"}
    )
    eval_dataset_uri = fields.Str(
        validate=fields.validate.Length(max=1000),
        allow_none=True,
        metadata={"description": "Evaluation dataset URI"}
    )
    inference_dataset_uri = fields.Str(
        validate=fields.validate.Length(max=1000),
        allow_none=True,
        metadata={"description": "Inference dataset URI"}
    )
    calibration_dataset_uri = fields.Str(
        validate=fields.validate.Length(max=1000),
        allow_none=True,
        metadata={"description": "Calibration dataset URI"}
    )
    dataset_format = fields.Str(
        validate=fields.validate.Length(max=100),
        allow_none=True,
        metadata={
            "description": "Dataset format (e.g., 'llava', 'coco', 'kitti'). "
                           "If not specified, uses network's default format"
        }
    )
    read_only = fields.Bool()
    public = fields.Bool()
    actions = fields.List(EnumField(ActionEnum), allow_none=True, validate=validate.Length(max=sys.maxsize))
    jobs = fields.Dict(
        keys=fields.Str(
            format="uuid",
            validate=fields.validate.Length(max=36)
        ),
        values=fields.Nested(JobSubset),
        validate=validate.Length(max=sys.maxsize)
    )
    status = EnumField(JobStatusEnum)
    all_jobs_cancel_status = EnumField(JobStatusEnum, allow_none=True)
    automl_settings = fields.Nested(AutoML)
    metric = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), allow_none=True)
    model_params = fields.Dict(allow_none=True)
    base_experiment_metadata = fields.Nested(BaseExperimentMetadata, allow_none=True)
    source_type = EnumField(SourceType, allow_none=True)
    experiment_actions = fields.List(
        fields.Nested(ExperimentActions, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )
    tensorboard_enabled = fields.Bool(default=False)
    tags = fields.List(
        fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=36)
        ),
        validate=validate.Length(max=16)
    )
    authorized_party_nca_id = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=2048),
        allow_none=True
    )
    parent_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    action = EnumField(ActionEnum)
    status = EnumField(JobStatusEnum)
    job_details = fields.Dict(
        keys=fields.Str(
            format="uuid",
            validate=fields.validate.Length(max=36)
        ),
        values=fields.Nested(JobResult),
        validate=validate.Length(max=sys.maxsize)
    )
    sync = fields.Bool()
    specs = fields.Raw(allow_none=True)
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    backend_details = fields.Nested(BackendDetails, allow_none=True)
    experiment_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    epoch_numbers = fields.List(
        fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize)),
        validate=validate.Length(max=sys.maxsize),
        allow_none=True
    )
    kind = fields.Constant(JobKindEnum.experiment.value)


class JobRsp(OneOfSchema):
    """Class defining a polymorphic job response schema"""

    type_schemas = {
        "dataset": DatasetJobRsp,
        "experiment": ExperimentJobRsp
    }
    type_field = "kind"

    def get_obj_type(self, obj):
        """Determine the schema to use based on the properties of the Python object"""
        kind = obj.get("kind")
        if kind in JobKindEnum:
            return kind
        raise fields.ValidationError(f"Invalid job kind: {kind}")


class JobListRsp(Schema):
    """Class defining job list response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    jobs = fields.List(
        fields.Nested(JobRsp),
        validate=validate.Length(max=sys.maxsize)
    )
    pagination_info = fields.Nested(PaginationInfo, allow_none=True)
