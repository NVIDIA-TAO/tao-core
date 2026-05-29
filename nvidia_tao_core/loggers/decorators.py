# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common decorators used in TAO Toolkit."""

from functools import wraps
import logging
import os
from typing import Optional, Callable

from nvidia_tao_core.loggers.logging import logger as tao_logger
from nvidia_tao_core.loggers.logging import (
    set_status_logger,
    get_status_logger,
    StatusLogger,
    Status,
    Verbosity
)


def monitor_status(name: str = 'TAO-Toolkit',
                   mode: str = 'train',
                   logger: logging.Logger = tao_logger,
                   results_dir: Optional[str] = None,
                   verbosity: int = None):
    """Status monitoring decorator for TAO-Toolkit functions.

    This decorator provides comprehensive status logging for training and other
    operations, similar to the TAO Deploy and TAO PyTorch implementations.

    Args:
        name: Name of the operation being monitored
        mode: Mode of operation (e.g., 'train', 'rollout', 'evaluate')
        logger: Logger to use for logging (default: tao_logger)
        results_dir: Directory to save status logs (auto-detects from config/TAO_API_JOB_ID)
        verbosity: Logging verbosity level
    """
    def inner(runner: Callable) -> Callable:
        @wraps(runner)
        def _func(*args, **kwargs):
            # Setup results directory - use job results dir if available
            is_master = int(os.environ.get("NODE_RANK", 0)) == 0
            if results_dir is None:
                # Try to get results_dir from config if available
                config = None
                if args and hasattr(args[0], 'config'):
                    config = args[0].config
                elif 'config' in kwargs:
                    config = kwargs['config']
                elif 'cfg' in kwargs:
                    config = kwargs.get('cfg')
                if config and hasattr(config, 'results_dir'):
                    default_results_dir = config.results_dir
                else:
                    # Use TAO job environment if available
                    job_id = os.getenv('TAO_API_JOB_ID')
                    if job_id:
                        # Use TAO_API_RESULTS_DIR for SLURM compatibility, fallback to /results
                        results_base = os.getenv('TAO_API_RESULTS_DIR', '/results')
                        default_results_dir = os.path.join(results_base, job_id)
                        logger.info(f"Using TAO job results dir: {default_results_dir}")
                    else:
                        default_results_dir = './results'
                        logger.warning(f"TAO_API_JOB_ID not found, using default: {default_results_dir}")
            else:
                default_results_dir = results_dir

            # Create results directory
            os.makedirs(default_results_dir, exist_ok=True)

            # Setup status logging - single consolidated status.json file
            status_file = os.path.join(default_results_dir, "status.json")
            log_verbosity = verbosity if verbosity is not None else Verbosity.INFO
            status_logger = StatusLogger(
                filename=status_file,
                is_master=is_master,
                verbosity=log_verbosity,
                append=True  # Append to consolidated status.json file
            )
            set_status_logger(status_logger)
            s_logger = get_status_logger()

            # Extract config if it's the first argument
            config = None
            if args and hasattr(args[0], 'logging'):
                config = args[0]
            elif 'config' in kwargs:
                config = kwargs['config']
            elif 'cfg' in kwargs:
                config = kwargs.get('cfg')

            try:
                s_logger.write(
                    status_level=Status.STARTED,
                    message=f"Starting {name} {mode}"
                )
                logger.info(f"Starting {name} {mode} with status logging to {status_file}")

                # Execute the wrapped function
                result = runner(*args, **kwargs)

                # Log successful completion
                s_logger.write(
                    status_level=Status.RUNNING,
                    message=f"{name} {mode} completed successfully"
                )
                logger.info(f"{name} {mode} completed successfully")

                # Check for cloud upload
                if os.getenv("CLOUD_BASED") == "True":
                    s_logger.write(
                        status_level=Status.RUNNING,
                        message="Job artifacts are being uploaded to the cloud"
                    )

                return result

            except (KeyboardInterrupt, SystemError) as e:
                s_logger.write(
                    message=f"{name} {mode} was interrupted: {str(e)}",
                    verbosity_level=Verbosity.WARNING,
                    status_level=Status.FAILURE
                )
                logger.warning(f"{name} {mode} was interrupted")
                raise

            except Exception as e:
                s_logger.write(
                    message=f"{name} {mode} failed: {str(e)}",
                    verbosity_level=Verbosity.ERROR,
                    status_level=Status.FAILURE
                )
                logger.error(f"{name} {mode} failed: {str(e)}")
                raise

        return _func
    return inner
