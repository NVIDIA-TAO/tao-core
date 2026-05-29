# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API version management and blueprint registration system."""

import logging
from typing import Dict, List, Tuple
from flask import Flask

from .config import create_api_spec_v1, create_api_spec_v2

logger = logging.getLogger(__name__)


class APIVersionManager:
    """Manages registration of versioned API blueprints."""

    def __init__(self):
        """Initialize the API version manager with empty version registry."""
        self.versions = {}
        self.available_versions = []

    def register_version(self, version: str, blueprints: List[Tuple], description: str = ""):
        """Register a new API version with its blueprints.

        Args:
            version: Version string (e.g., 'v1', 'v2')
            blueprints: List of tuples (blueprint, url_prefix)
            description: Optional description of the version
        """
        self.versions[version] = {
            'blueprints': blueprints,
            'description': description,
            'enabled': True
        }
        if version not in self.available_versions:
            self.available_versions.append(version)

        logger.info(f"Registered API version {version} with {len(blueprints)} blueprints")

    def register_blueprints_with_app(self, app: Flask):
        """Register all enabled version blueprints with the Flask app."""
        for version, config in self.versions.items():
            if not config['enabled']:
                continue

            logger.info(f"Registering blueprints for API {version}")
            for blueprint, url_prefix in config['blueprints']:
                full_prefix = f"/api/{version}{url_prefix}" if url_prefix else f"/api/{version}"
                app.register_blueprint(blueprint, url_prefix=full_prefix)
                logger.debug(f"Registered {blueprint.name} with prefix {full_prefix}")

    def get_available_versions(self) -> List[str]:
        """Get list of available API versions."""
        return sorted(self.available_versions)

    def get_version_info(self, version: str) -> Dict:
        """Get information about a specific version."""
        return self.versions.get(version, {})

    def enable_version(self, version: str):
        """Enable a specific API version."""
        if version in self.versions:
            self.versions[version]['enabled'] = True
            logger.info(f"Enabled API version {version}")

    def disable_version(self, version: str):
        """Disable a specific API version."""
        if version in self.versions:
            self.versions[version]['enabled'] = False
            logger.info(f"Disabled API version {version}")


# Global instance for easy access


api_version_manager = APIVersionManager()


def setup_v1_api():
    """Setup and register v1 API blueprints."""
    try:
        from .blueprints.v1 import (
            admin_bp_v1,
            auth_bp_v1,
            automl_params_bp_v1,
            workspaces_bp_v1,
            datasets_bp_v1,
            experiments_bp_v1,
            health_bp_v1,
            internal_bp_v1
        )

        # Define v1 blueprints with their URL prefixes
        v1_blueprints = [
            (admin_bp_v1, ''),          # /api/v1/version, /api/v1/swagger, /api/v1/metrics
            (auth_bp_v1, ''),           # /api/v1/login, /api/v1/auth
            (health_bp_v1, ''),         # /api/v1/health/*
            (internal_bp_v1, ''),       # /api/v1/internal/*, /api/v1/orgs/<org>:gpu_types
            (workspaces_bp_v1, ''),     # /api/v1/orgs/<org>/workspaces/*
            # /api/v1/orgs/<org>/experiments/*:get_automl_param_details,
            # /api/v1/orgs/<org>/experiments/*:update_automl_param_ranges
            (automl_params_bp_v1, ''),
            (datasets_bp_v1, ''),       # /api/v1/orgs/<org>/datasets/*
            (experiments_bp_v1, ''),    # /api/v1/orgs/<org>/experiments/*
        ]

        api_version_manager.register_version(
            'v1',
            v1_blueprints,
            'Initial API version with full TAO functionality'
        )
        logger.info("Successfully set up v1 API blueprints")
        return True

    except ImportError as e:
        logger.error(f"Failed to import v1 blueprints: {e}")
        return False


def setup_v2_api():
    """Setup and register v1 API blueprints."""
    try:
        from .blueprints.v2 import (
            admin_bp_v2,
            auth_bp_v2,
            automl_params_bp_v2,
            workspaces_bp_v2,
            datasets_bp_v2,
            health_bp_v2,
            jobs_bp_v2,
            inference_microservices_bp_v2,
        )

        v2_blueprints = [
            (admin_bp_v2, ''),          # /api/v2/version, /api/v1/swagger, /api/v1/metrics
            (auth_bp_v2, ''),           # /api/v2/login, /api/v1/auth
            (health_bp_v2, ''),         # /api/v2/health/*
            (workspaces_bp_v2, ''),     # /api/v2/orgs/<org>/workspaces/*
            (automl_params_bp_v2, ''),  # /api/v2/orgs/<org>/automl:get_param_details
            (datasets_bp_v2, ''),       # /api/v2/orgs/<org>/datasets/*
            (jobs_bp_v2, ''),           # /api/v2/orgs/<org>/jobs/*
            (inference_microservices_bp_v2, ''),  # /api/v2/orgs/<org>/inference_microservices/*
        ]

        api_version_manager.register_version(
            'v2',
            v2_blueprints,
            'Enhanced API version with improved functionality'
        )
        return True
    except ImportError as e:
        logger.error(f"v2 blueprints not available yet: {e}")
        return False


def register_all_api_versions(app: Flask):
    """Register all available API versions with the Flask app."""
    logger.info("Setting up API versioning...")

    # Setup available versions
    setup_v1_api()
    setup_v2_api()

    # Register all blueprints with the app
    api_version_manager.register_blueprints_with_app(app)

    # Add version info endpoints
    @app.route('/api/', methods=['GET'])
    def api_root():
        """Get available API versions."""
        from flask import jsonify
        return jsonify(api_version_manager.get_available_versions())

    logger.info(f"API versioning setup complete. Available versions: {api_version_manager.get_available_versions()}")


def initialize_all_apispec_version(app: Flask):
    """Initialize all APISpec versions"""
    logger.info("Initialize APISpec versions")

    # Initialize separate OpenAPI specs for v1 and v2
    logger.info("Initialize versioned OpenAPI specs")
    spec_v1 = create_api_spec_v1()
    spec_v2 = create_api_spec_v2()

    # Store specs in app for access by documentation endpoints
    app.spec_v1 = spec_v1
    app.spec_v2 = spec_v2

    logger.info("Versioned OpenAPI specs complete.")


def register_all_schemas_with_apispec(app: Flask):
    """Register all schemas with APISpec"""
    logger.info("Setting up schemas in APISpec...")

    spec_v1 = app.spec_v1
    spec_v2 = app.spec_v2

    from .blueprints.v1 import V1_SCHEMAS
    for name, schema in sorted(V1_SCHEMAS):
        # logger.info(f"Adding schema {name} in APISpec v1 titled: {spec_v1.title}")
        spec_v1.components.schema(name, schema=schema)

    from .blueprints.v2 import V2_SCHEMAS
    for name, schema in sorted(V2_SCHEMAS):
        # logger.info(f"Adding schema {name} in APISpec v2 titled: {spec_v2.title}")
        spec_v2.components.schema(name, schema=schema)

    logger.info("Schemas in APISpec complete.")


def register_all_routes_with_apispec(app: Flask):
    """Register all routes with APISpec"""
    logger.info("Setting up routes in APISpec...")

    spec_v1 = app.spec_v1
    spec_v2 = app.spec_v2

    with app.app_context():

        # Register routes with appropriate OpenAPI specs
        logger.info("Register routes with versioned OpenAPI specs")
        for endpoint, view_func in app.view_functions.items():
            logger.debug(f"Processing endpoint: {endpoint}")
            original_view_func = view_func

            # Determine which spec to use based on endpoint path
            is_v1_endpoint = '_v1.' in endpoint
            is_v2_endpoint = '_v2.' in endpoint

            # Extract the original view function if it's decorated
            while hasattr(view_func, '__wrapped__'):
                view_func = view_func.__wrapped__
            # Add endpoint to OpenAPI specs only if the view function has docstrings present
            if hasattr(view_func, '__doc__') and view_func.__doc__:
                try:
                    if is_v1_endpoint:
                        # logger.info(f"Registering v1 endpoint: {endpoint}")
                        spec_v1.path(view=original_view_func)
                    elif is_v2_endpoint:
                        # logger.info(f"Registering v2 endpoint: {endpoint}")
                        spec_v2.path(view=original_view_func)
                    else:
                        # For non-versioned endpoints, register with both specs
                        # logger.info(f"Registering non-versioned endpoint with both specs: {endpoint}")
                        spec_v1.path(view=original_view_func)
                        spec_v2.path(view=original_view_func)
                except Exception as e:
                    logger.warning(f"Failed to register {endpoint} with OpenAPI spec: {e}")

    logger.info("Routes in APISpec complete.")
