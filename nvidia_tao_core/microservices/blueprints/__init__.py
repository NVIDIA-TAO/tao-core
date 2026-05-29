# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flask blueprints for organizing API endpoints.

This package uses versioned APIs with the following structure:

- v1/: Contains all v1 API blueprints including admin (current production)
- v2/: Contains all v2 API blueprints including admin (enhanced features)

All admin functionality is now versioned and integrated into v1 and v2.
Use the api_versions module for automatic registration of all versions.
"""

# Admin blueprints are now versioned and imported via v1/ and v2/ directories

# Versioned imports
try:
    from .v1 import *  # Import all v1 blueprints  # noqa: F403, F401
except ImportError:
    pass

try:
    from .v2 import *  # Import all v2 blueprints (when available)  # noqa: F403, F401
except ImportError:
    pass

__all__ = []

# Add versioned blueprints to __all__ if they exist
try:
    from .v1 import __all__ as v1_all
    __all__.extend(v1_all)
except ImportError:
    pass

try:
    from .v2 import __all__ as v2_all
    __all__.extend(v2_all)
except ImportError:
    pass
