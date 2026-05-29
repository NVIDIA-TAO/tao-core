# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration module for Flask application."""

import os
import sys
import pkg_resources
from apispec import APISpec
from apispec_webframeworks.flask import FlaskPlugin
from apispec_oneofschema import MarshmallowPlugin

# Configuration constants
TIMEOUT = 240
NAMESPACE = os.getenv("NAMESPACE", "default")


def inline_resolver(schema_class):
    """Inline schema name resolver for marshmallow plugin."""
    return None


# API Spec configuration
flask_plugin = FlaskPlugin()
marshmallow_plugin = MarshmallowPlugin(schema_name_resolver=inline_resolver)


def sys_int_format():
    """Get integer format based on system."""
    if sys.maxsize > 2**31 - 1:
        return "int64"
    return "int32"


def get_tao_version():
    """Get TAO version."""
    try:
        return pkg_resources.get_distribution('nvidia_tao_core').version
    except Exception:
        return os.getenv('TAO_VERSION', '6.0.0')


def _configure_common_spec_components(spec):
    """Configure common components for API specs."""
    # Configure security schemes
    api_key_scheme = {"type": "apiKey", "in": "header", "name": "ngc_key"}
    jwt_scheme = {"type": "http", "scheme": "bearer", "bearerFormat": "JWT", "description": "RFC8725 Compliant JWT"}

    spec.components.security_scheme("api-key", api_key_scheme)
    spec.components.security_scheme("bearer-token", jwt_scheme)

    # Configure headers
    spec.components.header("X-RateLimit-Limit", {
        "description": "The number of allowed requests in the current period",
        "schema": {
            "type": "integer",
            "format": sys_int_format(),
            "minimum": -sys.maxsize - 1,
            "maximum": sys.maxsize,
        }
    })
    spec.components.header("Access-Control-Allow-Origin", {
        "description": "Origins that are allowed to share response",
        "schema": {
            "type": "string",
            "format": "regex",
            "maxLength": sys.maxsize,
        }
    })

    # Enum stuff for APISpecs
    def enum_to_properties(self, field, **kwargs):
        """Add an OpenAPI extension for marshmallow_enum.EnumField instances"""
        from marshmallow_enum import EnumField
        if isinstance(field, EnumField):
            return {'type': 'string', 'enum': [m.name for m in field.enum]}
        return {}

    marshmallow_plugin.converter.add_attribute_function(enum_to_properties)


def create_api_spec_v1():
    """Create and configure APISpec instance for v1 API."""
    tao_version = get_tao_version()

    spec = APISpec(
        title='NVIDIA TAO API v1',
        version=tao_version,
        openapi_version='3.0.3',
        info={"description": 'NVIDIA TAO (Train, Adapt, Optimize) API v1 document'},
        tags=[
            {"name": 'AUTHENTICATION', "description": 'Endpoints related to User Authentication'},
            {"name": 'DATASET', "description": 'Endpoints related to Datasets'},
            {"name": 'EXPERIMENT', "description": 'Endpoints related to Experiments'},
            {"name": 'WORKSPACE', "description": 'Endpoints related to Workspaces'},
            {"name": 'HEALTH', "description": 'Health check endpoints'},
            {"name": "nSpectId",
             "description": "NSPECT-1T59-RTYH",
             "externalDocs": {
                 "url": "https://nspect.nvidia.com/review?id=NSPECT-1T59-RTYH"
             }}
        ],
        plugins=[flask_plugin, marshmallow_plugin],
        security=[{"bearer-token": []}],
    )

    _configure_common_spec_components(spec)
    return spec


def create_api_spec_v2():
    """Create and configure APISpec instance for v2 API."""
    tao_version = get_tao_version()

    spec = APISpec(
        title='NVIDIA TAO API v2',
        version=tao_version,
        openapi_version='3.0.3',
        info={"description": 'NVIDIA TAO (Train, Adapt, Optimize) API v2 document'},
        tags=[
            {"name": 'AUTHENTICATION', "description": 'Endpoints related to User Authentication'},
            {"name": 'WORKSPACE', "description": 'Endpoints related to Workspaces'},
            {"name": 'DATASET', "description": 'Endpoints related to Datasets'},
            {"name": 'JOB', "description": 'Endpoints related to Jobs'},
            {"name": 'INFERENCE_MICROSERVICE', "description": 'Endpoints related to Inference Microservices'},
            {"name": "nSpectId",
             "description": "NSPECT-1T59-RTYH",
             "externalDocs": {
                 "url": "https://nspect.nvidia.com/review?id=NSPECT-1T59-RTYH"
             }}
        ],
        plugins=[flask_plugin, marshmallow_plugin],
        security=[{"bearer-token": []}],
    )

    _configure_common_spec_components(spec)
    return spec


def create_api_spec():
    """Create and configure APISpec instance (legacy function for backward compatibility)."""
    return create_api_spec_v1()

# Flask app configuration


FLASK_CONFIG = {
    'WTF_CSRF_ENABLED': False,
    'TRAP_HTTP_EXCEPTIONS': True,
}

# Rate limiter configuration
RATE_LIMITER_CONFIG = {
    'default_limits': ["10000/hour"],
    'headers_enabled': True,
    'storage_uri': "memory://",
}
