# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base Inference Microservice Server - Abstract base class for persistent model servers in StatefulSet containers

Provides common functionality for loading models, serving inference requests, and managing server lifecycle
"""

import base64
import os
import json
import logging
import re
import tempfile
import threading
import time
import requests
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Tuple
from flask import Flask, request, jsonify

from nvidia_tao_core.microservices.utils.core_utils import safe_load_file

TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)

# MIME types accepted on the /v1/chat/completions endpoint for inline base64 media.
_EXT_MAP = {
    'video/mp4': '.mp4', 'video/avi': '.avi',
    'video/webm': '.webm', 'video/mkv': '.mkv',
    'video/x-matroska': '.mkv', 'video/mov': '.mov',
    'image/jpeg': '.jpg', 'image/png': '.png',
    'image/webp': '.webp', 'image/gif': '.gif',
    'image/bmp': '.bmp', 'image/tiff': '.tiff',
}
_MAX_MEDIA_BYTES = 512 * 1024 * 1024  # mirrors OpenAI's chat-completions 512 MB payload cap

# vLLM/OpenAI sampling parameters forwarded from the /v1/chat/completions
# request body to ``run_inference(**kwargs)``. The key is the request-body
# field; the value is the kwarg name passed downstream. The only rename is
# OpenAI's ``max_tokens`` → HuggingFace's ``max_new_tokens``. Everything
# else is forwarded as-is; subclasses (or model.generate) decide which
# keys they actually consume. Mirrors the documented surface at
# https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html.
_SAMPLING_PARAM_MAP = {
    "max_tokens": "max_new_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repetition_penalty",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "seed": "seed",
    "stop": "stop",
    "n": "n",
    "best_of": "best_of",
    "length_penalty": "length_penalty",
    "ignore_eos": "ignore_eos",
    "min_tokens": "min_tokens",
}
_SAMPLING_PARAM_DEFAULTS = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
}


class BaseInferenceMicroserviceServer(ABC):
    """Abstract base class for TAO model servers in StatefulSet containers"""

    def __init__(self, job_id: str, port: int = 8080, **model_params):
        """Initialize base model server

        Args:
            job_id: Unique job identifier
            port: Server port (default 8080)
            **model_params: Model-specific parameters (e.g., model_path, etc.)
        """
        self.job_id = job_id
        self.port = port
        self.model = None
        self.model_loaded = False
        self.model_loading = False
        self.model_load_error = None
        self.server_initializing = True
        self.initialization_error = None
        self.model_state_dir = "/tmp/tao_models"
        self.model_params = model_params  # Store all model-specific params

        # Progress tracking for initialization and downloads
        self.progress = {
            "stage": "initializing",  # initializing, loading_model, ready, error
            "message": "Server initialization starting",
            "remaining_steps": [
                "Load model into memory",
                "Ready for inference"
            ],
            "details": {}
        }

        # Health check and auto-deletion configuration
        self.last_request_time = datetime.now()
        self.idle_timeout_minutes = 30  # Default 30 minutes idle timeout
        self.auto_deletion_enabled = True
        self._health_monitor_thread = None
        self._shutdown_flag = threading.Event()

    def update_last_request_time(self):
        """Update the last request timestamp - called on each inference request"""
        self.last_request_time = datetime.now()
        logger.debug(f"Updated last request time: {self.last_request_time}")

    def update_progress(
        self, stage: str, message: str, remaining_steps: List[str] = None,
        details: Dict[str, Any] = None
    ):
        """Update progress information

        Args:
            stage: Current stage (initializing, downloading_model, loading_model, ready, error)
            message: Human-readable progress message
            remaining_steps: List of remaining steps until ready (optional)
            details: Optional dictionary with additional details
        """
        self.progress = {
            "stage": stage,
            "message": message,
            "remaining_steps": remaining_steps or [],
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        steps_info = f" (Remaining: {len(remaining_steps)} steps)" if remaining_steps else ""
        logger.info(f"Progress update: {stage} - {message}{steps_info}")
        # Save progress to state file
        self.save_model_state(
            loaded=self.model_loaded,
            loading=self.model_loading
        )

    def get_idle_time_minutes(self) -> float:
        """Get the current idle time in minutes

        Returns:
            Float representing minutes since last request, or 0 if model is not ready yet.
        """
        # Return 0 idle time if model is not loaded yet (still initializing/loading)
        if not self.model_loaded:
            return 0.0

        idle_time = datetime.now() - self.last_request_time
        return idle_time.total_seconds() / 60.0

    def is_idle_timeout_exceeded(self) -> bool:
        """Check if the idle timeout has been exceeded

        Returns:
            True if server has been idle longer than timeout, False otherwise
        """
        return self.get_idle_time_minutes() > self.idle_timeout_minutes

    def _start_health_monitor(self):
        """Start the health monitoring thread for auto-deletion"""
        if self._health_monitor_thread is None and self.auto_deletion_enabled:
            self._health_monitor_thread = threading.Thread(
                target=self._health_monitor_loop,
                daemon=True
            )
            self._health_monitor_thread.start()
            logger.info(f"Started health monitor with {self.idle_timeout_minutes} minute timeout")

    def _health_monitor_loop(self):
        """Background thread that monitors server health and triggers auto-deletion request"""
        while not self._shutdown_flag.is_set():
            try:
                if self.is_idle_timeout_exceeded():
                    idle_minutes = self.get_idle_time_minutes()
                    logger.warning(
                        f"Server has been idle for {idle_minutes:.1f} minutes "
                        f"(timeout: {self.idle_timeout_minutes}). Requesting auto-deletion."
                    )

                    # Request auto-deletion by updating status of job.
                    # The workflow service will monitor and handle actual deletion
                    try:
                        logger.info("Requesting auto-deletion via status callback")
                        self.request_auto_deletion()
                        logger.info(
                            "Auto-deletion request sent. Workflow service will handle deletion."
                        )
                    except Exception as e:
                        logger.error(f"Failed to save auto-deletion request: {e}")

                    # Stop monitoring after requesting deletion
                    break

                # Check every 5 minutes
                self._shutdown_flag.wait(3)

            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                self._shutdown_flag.wait(60)  # Wait 1 minute before retrying

    def shutdown_health_monitor(self):
        """Gracefully shutdown the health monitoring thread"""
        if self._health_monitor_thread:
            logger.info("Shutting down health monitor")
            self._shutdown_flag.set()
            self._health_monitor_thread.join(timeout=10)
            if self._health_monitor_thread.is_alive():
                logger.warning("Health monitor thread did not shutdown gracefully")
            self._health_monitor_thread = None

    def _initialize_background(self, job_data: Dict[str, Any], docker_env_vars: Dict[str, Any]):
        """Initialize server configuration in background thread.

        The inference service does not depend on cloud storage: model weights are
        loaded directly from the HuggingFace Hub or a local filesystem path, and
        request inputs must be sent as inline data or http(s) URLs.
        """
        try:
            logger.info("Starting background initialization...")
            self.server_initializing = True
            self.initialization_error = None

            # Update progress: Starting initialization
            self.update_progress(
                stage="initializing",
                message="Starting server initialization",
                remaining_steps=[
                    "Load model into memory",
                    "Ready for inference"
                ],
                details={"phase": "starting"}
            )

            specs = job_data.get("specs", {}) or {}
            self.model_params.update(specs)

            self.server_initializing = False
            logger.info("Background initialization completed - starting model loading")

            # Update progress: Initialization complete, starting model load
            self.update_progress(
                stage="loading_model",
                message="Loading model into memory",
                remaining_steps=[
                    "Complete model loading",
                    "Ready for inference"
                ],
                details={"phase": "model_loading_starting"}
            )

            # Now start model loading
            self.load_model()

        except Exception as e:
            error_msg = f"Failed to initialize server: {e}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            self.server_initializing = False
            self.initialization_error = str(e)
            self.update_progress(
                stage="error",
                message=f"Initialization failed: {str(e)}",
                remaining_steps=[],
                details={"error": str(e), "phase": "initialization"}
            )
            self.save_model_state(loaded=False, loading=False, error=str(e))

            # Fatal error during data preparation — request auto-deletion so
            # the container doesn't stay running forever in an error state.
            logger.info("Requesting auto-deletion due to initialization failure")
            self.request_auto_deletion(
                reason="initialization_failed",
                error=str(e)
            )

    def save_model_state(self, loaded: bool = False, loading: bool = False, load_time: float = None, error: str = None):
        """Save model loading state to file

        Args:
            loaded: Whether model is loaded successfully
            loading: Whether model is currently loading
            load_time: Time taken to load model
            error: Error message if loading failed
        """
        model_state = {
            "job_id": self.job_id,
            "model_params": self.model_params,
            "loaded": loaded,
            "loading": loading,
            "initializing": self.server_initializing,
            "timestamp": datetime.now().isoformat(),
            "server_port": self.port,
            "model_type": self.__class__.__name__,
            "progress": self.progress
        }

        if load_time is not None:
            model_state["load_time"] = load_time
        if error:
            model_state["error"] = error
        if hasattr(self, 'initialization_error') and self.initialization_error:
            model_state["initialization_error"] = self.initialization_error

        os.makedirs(self.model_state_dir, exist_ok=True)
        state_file = f"{self.model_state_dir}/{self.job_id}_server.json"

        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(model_state, f, indent=2)

    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state from file

        Returns:
            Model state dictionary or empty dict if file doesn't exist
        """
        state_file = f"{self.model_state_dir}/{self.job_id}_server.json"
        if os.path.exists(state_file):
            return safe_load_file(state_file)
        return {}

    def request_auto_deletion(self, reason: str = "idle_timeout_exceeded", error: str = None):
        """Request auto-deletion by POSTing status to TAO logging URL.

        Called on idle timeout or a fatal init / model-load failure. This is a
        plain HTTP call — the inference service has no cloud-storage dependency.

        Args:
            reason: Reason for requesting deletion (idle_timeout_exceeded, initialization_failed, model_loading_failed)
            error: Optional error message for failure-triggered deletions
        """
        try:
            logger.info("Requesting auto-deletion via status callback (reason: %s)", reason)

            status_data = {
                "message": "AUTO_DELETION_REQUESTED",
                "status": "AUTO_DELETION_REQUESTED",
                "idle_time_minutes": self.get_idle_time_minutes(),
                "idle_timeout_minutes": self.idle_timeout_minutes,
                "reason": reason,
                "last_request_time": self.last_request_time.isoformat(),
                "timestamp": datetime.now().isoformat(),
            }
            if error:
                status_data["error"] = error

            if os.getenv("DEBUG_ENABLED", "").lower() == "true":
                logger.info("DEBUG_ENABLED=True: Skipping auto-deletion status callback")
                return

            logging_url = os.getenv("TAO_LOGGING_SERVER_URL", "")
            admin_key = os.getenv("TAO_ADMIN_KEY")
            if not logging_url or not admin_key:
                logger.warning(
                    "TAO_LOGGING_SERVER_URL or TAO_ADMIN_KEY not set; "
                    "skipping auto-deletion callback for job %s", self.job_id
                )
                return

            payload = {
                "experiment_number": os.getenv("AUTOML_EXPERIMENT_NUMBER", "0"),
                "status": json.dumps(status_data),
            }
            response = requests.post(
                logging_url + ":status_update",
                json=payload,
                headers={"Authorization": f"Bearer {admin_key}"},
                timeout=180,
            )
            if response.ok:
                logger.info(f"Auto-deletion status callback sent for job {self.job_id}")
            else:
                logger.error(
                    "Auto-deletion status callback failed: %s %s",
                    response.status_code, response.text[:500]
                )

        except Exception as e:
            logger.error(f"Failed to send auto-deletion status callback: {e}")
            import traceback
            logger.error(traceback.format_exc())

    @abstractmethod
    def load_model_into_memory(self, **kwargs) -> bool:
        """Load the specific model implementation

        Args:
            **kwargs: Model-specific configuration parameters

        Returns:
            True if model loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    def run_model_inference(self, **kwargs) -> Dict[str, Any]:
        """Run model-specific inference

        Args:
            **kwargs: All inference parameters (model-specific naming)

        Returns:
            Inference results dictionary
        """
        pass

    @abstractmethod
    def get_supported_file_extensions(self) -> Tuple[List[str], List[str]]:
        """Get supported file extensions for this model

        Returns:
            Tuple of (image_extensions, video_extensions)
        """
        pass

    def _load_model_background(self, **kwargs):
        """Load model in background thread"""
        try:
            print(f"Loading {self.__class__.__name__} model in background")
            self.model_loading = True
            self.model_load_error = None
            start_time = time.time()

            # Update progress: Model loading started
            self.update_progress(
                stage="loading_model",
                message="Loading model into memory",
                remaining_steps=[
                    "Complete model loading",
                    "Ready for inference"
                ],
                details={"phase": "loading", "model_type": self.__class__.__name__}
            )

            # Merge model_params with provided kwargs
            all_params = {**self.model_params, **kwargs}

            # Call model-specific loading implementation
            success = self.load_model_into_memory(**all_params)

            if success:
                load_time = time.time() - start_time
                self.model_loaded = True
                self.model_loading = False
                print(f"{self.__class__.__name__} model loaded successfully in {load_time:.2f} seconds")

                # Update progress: Model loaded successfully
                self.update_progress(
                    stage="ready",
                    message="Model loaded successfully, ready for inference",
                    remaining_steps=[],
                    details={"load_time": load_time, "model_type": self.__class__.__name__}
                )

                logger.info(f"Model loaded successfully in {load_time:.2f}s - ready for inference")

                # Start health monitoring for idle timeout now that server is ready for inference
                self.update_last_request_time()  # Reset timer to start from when model is ready
                self._start_health_monitor()
            else:
                self.model_loading = False
                self.model_load_error = "Model loading failed"
                self.update_progress(
                    stage="error",
                    message="Model loading failed",
                    remaining_steps=[],
                    details={"error": "Model loading returned False", "phase": "model_loading"}
                )
                self.save_model_state(loaded=False, loading=False, error="Model loading failed")
                logger.info("Requesting auto-deletion due to model loading failure")
                self.request_auto_deletion(
                    reason="model_loading_failed",
                    error="Model loading returned False"
                )

        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            logger.error(error_msg)
            self.model_loaded = False
            self.model_loading = False
            self.model_load_error = str(e)
            self.update_progress(
                stage="error",
                message=f"Model loading failed: {str(e)}",
                remaining_steps=[],
                details={"error": str(e), "phase": "model_loading"}
            )
            self.save_model_state(loaded=False, loading=False, error=str(e))
            logger.info("Requesting auto-deletion due to model loading exception")
            self.request_auto_deletion(
                reason="model_loading_failed",
                error=str(e)
            )

    def load_model(self, **kwargs) -> bool:
        """Start model loading in background thread

        Args:
            **kwargs: Model-specific configuration parameters

        Returns:
            True (loading started), False if already loading
        """
        if self.model_loading or self.model_loaded:
            return False

        # Start loading in background thread
        load_thread = threading.Thread(target=self._load_model_background, kwargs=kwargs)
        load_thread.daemon = True
        load_thread.start()
        return True

    @staticmethod
    def _reject_cloud_uri(input_file: str) -> str:
        """Reject cloud-storage URIs on the inference path.

        The inference service does not authenticate to any cloud. Callers must
        send inline data (data: URIs / base64) or http(s):// URLs that the
        service can fetch anonymously.
        """
        cloud_prefixes = ('s3://', 'gs://', 'az://', 'cs://', 'aws://')
        if isinstance(input_file, str) and input_file.startswith(cloud_prefixes):
            raise ValueError(
                f"Cloud-storage URIs are not supported by the inference service: "
                f"{input_file}. Send the media inline (data: URI) or via an "
                f"http(s):// URL."
            )
        return input_file

    def run_inference(self, **kwargs) -> Dict[str, Any]:
        """Run inference with error handling and timing

        Args:
            **kwargs: All inference parameters (model-specific)

        Returns:
            Inference results dictionary
        """
        if not self.model_loaded:
            if self.model_loading:
                raise RuntimeError("Model is still loading, please wait")
            if self.model_load_error:
                raise RuntimeError(f"Model failed to load: {self.model_load_error}")
            raise RuntimeError("Model not loaded")

        try:
            start_time = time.time()

            # Call model-specific inference implementation
            result = self.run_model_inference(**kwargs)

            inference_time = time.time() - start_time

            # Add common metadata to result
            if isinstance(result, dict):
                result.update({
                    "inference_time": inference_time,
                    "timestamp": datetime.now().isoformat(),
                    "model_type": self.__class__.__name__
                })

            logger.info(f"Inference completed in {inference_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def create_flask_app(self):
        """Create Flask app with common endpoints"""
        app = Flask(__name__)

        @app.route('/api/v1/health/liveness', methods=['GET'])
        def health():
            """Health check endpoint"""
            idle_minutes = self.get_idle_time_minutes()
            return jsonify({
                "status": "healthy",
                "model_loaded": self.model_loaded,
                "model_loading": self.model_loading,
                "server_initializing": self.server_initializing,
                "job_id": self.job_id,
                "model_type": self.__class__.__name__,
                "last_request_time": self.last_request_time.isoformat(),
                "idle_time_minutes": round(idle_minutes, 2),
                "idle_timeout_minutes": self.idle_timeout_minutes,
                "auto_deletion_enabled": self.auto_deletion_enabled,
                "progress": self.progress
            })

        @app.route('/api/v1/health/readiness', methods=['GET'])
        def readiness():
            """Readiness check endpoint - returns success only when fully ready to serve requests"""
            # Check if server is still initializing
            if self.server_initializing:
                return jsonify({
                    "status": "not_ready",
                    "reason": "server_initializing",
                    "message": "Server is still initializing",
                    "job_id": self.job_id,
                    "timestamp": datetime.now().isoformat()
                }), 503

            # Check for initialization errors
            if self.initialization_error:
                return jsonify({
                    "status": "not_ready",
                    "reason": "initialization_failed",
                    "message": f"Server initialization failed: {self.initialization_error}",
                    "job_id": self.job_id,
                    "timestamp": datetime.now().isoformat()
                }), 503

            # Check if model is still loading
            if self.model_loading:
                return jsonify({
                    "status": "not_ready",
                    "reason": "model_loading",
                    "message": "Model is currently loading",
                    "job_id": self.job_id,
                    "timestamp": datetime.now().isoformat()
                }), 503

            # Check for model loading errors
            if self.model_load_error:
                return jsonify({
                    "status": "not_ready",
                    "reason": "model_load_failed",
                    "message": f"Model failed to load: {self.model_load_error}",
                    "job_id": self.job_id,
                    "timestamp": datetime.now().isoformat()
                }), 503

            # Check if model is loaded successfully
            if not self.model_loaded:
                return jsonify({
                    "status": "not_ready",
                    "reason": "model_not_loaded",
                    "message": "Model not loaded yet",
                    "job_id": self.job_id,
                    "timestamp": datetime.now().isoformat()
                }), 503

            # All checks passed - server is ready
            return jsonify({
                "status": "ready",
                "message": "Server is ready to accept inference requests",
                "job_id": self.job_id,
                "model_type": self.__class__.__name__,
                "timestamp": datetime.now().isoformat()
            }), 200

        @app.route('/api/v1/status', methods=['GET'])
        def status():
            """Detailed status endpoint"""
            model_state = self.get_model_state()
            idle_minutes = self.get_idle_time_minutes()
            return jsonify({
                "job_id": self.job_id,
                "model_loaded": self.model_loaded,
                "model_loading": self.model_loading,
                "server_initializing": self.server_initializing,
                "initialization_error": self.initialization_error,
                "model_load_error": self.model_load_error,
                "model_state": model_state,
                "server_port": self.port,
                "last_request_time": self.last_request_time.isoformat(),
                "idle_time_minutes": round(idle_minutes, 2),
                "idle_timeout_minutes": self.idle_timeout_minutes,
                "auto_deletion_enabled": self.auto_deletion_enabled,
                "health_monitor_active": (
                    self._health_monitor_thread is not None and
                    self._health_monitor_thread.is_alive()
                ),
                "progress": self.progress
            })

        @app.route('/api/v1/inference', methods=['POST'])
        def inference():
            """Inference endpoint"""
            try:
                # Update request timestamp for health monitoring
                self.update_last_request_time()

                # Check initialization status first
                if self.server_initializing:
                    response_data = {
                        "job_id": self.job_id,
                        "status": "initializing",
                        "message": "Server is initializing (downloading files, setting up), please wait",
                        "timestamp": datetime.now().isoformat()
                    }
                    return jsonify(response_data), 202  # 202 Accepted - processing

                if self.initialization_error:
                    response_data = {
                        "job_id": self.job_id,
                        "status": "error",
                        "error": f"Server initialization failed: {self.initialization_error}",
                        "timestamp": datetime.now().isoformat()
                    }
                    return jsonify(response_data), 503  # 503 Service Unavailable

                # Check model status
                if self.model_loading:
                    response_data = {
                        "job_id": self.job_id,
                        "status": "loading",
                        "message": "Model is currently loading, please wait and try again",
                        "timestamp": datetime.now().isoformat()
                    }
                    return jsonify(response_data), 202  # 202 Accepted - processing

                if self.model_load_error:
                    response_data = {
                        "job_id": self.job_id,
                        "status": "error",
                        "error": f"Model failed to load: {self.model_load_error}",
                        "timestamp": datetime.now().isoformat()
                    }
                    return jsonify(response_data), 503  # 503 Service Unavailable

                if not self.model_loaded:
                    response_data = {
                        "job_id": self.job_id,
                        "status": "not_ready",
                        "message": "Model not loaded yet, please try again later",
                        "timestamp": datetime.now().isoformat()
                    }
                    return jsonify(response_data), 503  # 503 Service Unavailable

                # Parse request data - pass all parameters to model implementation
                data = request.json
                if not data:
                    return jsonify({
                        "error": "Request body is required",
                        "job_id": self.job_id,
                        "timestamp": datetime.now().isoformat()
                    }), 400

                # Run inference with all parameters
                results = self.run_inference(**data)

                response_data = {
                    "status": "completed",
                    "results": results,
                    "job_id": self.job_id,
                    "message": f"{self.__class__.__name__} inference completed"
                }

                return jsonify(response_data)

            except Exception as e:
                logger.error(f"Inference request failed: {e}")
                return jsonify({
                    "status": "error",
                    "error": str(e),
                    "job_id": self.job_id,
                    "timestamp": datetime.now().isoformat()
                }), 500

        @app.route('/v1/chat/completions', methods=['POST'])
        def chat_completions():
            """vLLM/OpenAI-compatible chat completions endpoint.

            Request body (subset of OpenAI Chat Completions):
                messages: list of {role, content}. `content` may be a string,
                    or a list of items with `type` in {"text", "image_url",
                    "video_url"}. `image_url`/`video_url` items carry a
                    nested {"url": "<http url or data: URI>"}. Inline base64
                    payloads must declare a MIME type present in `_EXT_MAP`
                    and stay under `_MAX_MEDIA_BYTES`.
                Sampling params: any key in ``_SAMPLING_PARAM_MAP`` is
                    forwarded to ``run_inference`` (``max_tokens`` →
                    ``max_new_tokens``, plus ``temperature``, ``top_p``,
                    ``top_k``, ``min_p``, ``repetition_penalty``,
                    ``frequency_penalty``, ``presence_penalty``, ``seed``,
                    ``stop``, ``n``, ``best_of``, ``length_penalty``,
                    ``ignore_eos``, ``min_tokens``). Defaults:
                    ``max_new_tokens=1024``, ``temperature=0.7``. Subclasses
                    consume whichever keys their model supports.
                model (str, optional): accepted for client compatibility but
                    ignored — the response always reports the actual loaded
                    model identifier (the configured ``hf_model`` /
                    ``model_path``, falling back to the server class name).

            Subclasses' `run_model_inference` receives kwargs `media`
            (list of local paths or URLs), `prompt`, `system_prompt`, plus
            whatever sampling params the request included.

            Response: an OpenAI Chat Completions object with `id`, `object`,
            `created`, `model`, and a single `choices[0]` entry holding the
            assistant message.
            """
            try:
                self.update_last_request_time()

                if self.server_initializing:
                    return jsonify({
                        "error": {"message": "Server is initializing, please wait", "type": "server_initializing"}
                    }), 202
                if self.initialization_error:
                    return jsonify({
                        "error": {
                            "message": f"Server initialization failed: {self.initialization_error}",
                            "type": "initialization_error"
                        }
                    }), 503
                if self.model_loading:
                    return jsonify({
                        "error": {"message": "Model is currently loading, please wait", "type": "model_loading"}
                    }), 202
                if self.model_load_error:
                    return jsonify({
                        "error": {
                            "message": f"Model failed to load: {self.model_load_error}",
                            "type": "model_load_error"
                        }
                    }), 503
                if not self.model_loaded:
                    return jsonify({
                        "error": {"message": "Model not loaded yet", "type": "model_not_ready"}
                    }), 503

                data = request.json
                if not data:
                    return jsonify({
                        "error": {"message": "Request body is required", "type": "invalid_request"}
                    }), 400

                messages = data.get("messages", [])

                system_prompt = next(
                    (msg.get("content", "") for msg in messages if msg.get("role") == "system"),
                    ""
                )

                media_uris = []
                prompt = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            for item in content:
                                item_type = item.get("type")
                                if item_type == "video_url":
                                    video_url = item.get("video_url", {}).get("url")
                                    if video_url:
                                        media_uris.append(video_url)
                                elif item_type == "image_url":
                                    image_url = item.get("image_url", {}).get("url")
                                    if image_url:
                                        media_uris.append(image_url)
                                elif item_type == "text":
                                    prompt = item.get("text", "")
                        elif isinstance(content, str):
                            prompt = content
                        break

                sampling_kwargs = dict(_SAMPLING_PARAM_DEFAULTS)
                for req_key, kwarg_key in _SAMPLING_PARAM_MAP.items():
                    if req_key in data:
                        sampling_kwargs[kwarg_key] = data[req_key]

                temp_files = []
                media = []
                try:
                    for uri in media_uris:
                        m = re.match(r'data:([^;]+);base64,(.+)', uri, re.DOTALL)
                        if m:
                            media_type = m.group(1)
                            ext = _EXT_MAP.get(media_type)
                            if ext is None:
                                raise ValueError(f"Unsupported media type: {media_type}")
                            b64_payload = m.group(2)
                            # Approximate decoded size from base64 length without decoding first.
                            if (len(b64_payload) * 3) // 4 > _MAX_MEDIA_BYTES:
                                raise ValueError(
                                    f"Media payload exceeds {_MAX_MEDIA_BYTES} byte limit"
                                )
                            raw = base64.b64decode(b64_payload)
                            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                                tmp.write(raw)
                            temp_files.append(tmp.name)
                            media.append(tmp.name)
                        else:
                            media.append(uri)

                    result = self.run_inference(
                        media=media,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        **sampling_kwargs,
                    )
                finally:
                    for f in temp_files:
                        try:
                            os.unlink(f)
                        except OSError:
                            pass

                response_text = result.get("response", "") if isinstance(result, dict) else str(result)

                # Always report the actual loaded model — don't echo the
                # client-supplied "model" field, which could disagree with
                # what's running on this container.
                served_model = (
                    self.model_params.get("hf_model") or
                    self.model_params.get("model_path") or
                    self.__class__.__name__
                )
                chat_id = "chatcmpl-" + datetime.now().strftime("%Y%m%d%H%M%S%f")
                return jsonify({
                    "id": chat_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": served_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": response_text},
                            "finish_reason": "stop"
                        }
                    ]
                })

            except ValueError as e:
                logger.warning(f"Chat completions invalid request: {e}")
                return jsonify({
                    "error": {"message": str(e), "type": "invalid_request"}
                }), 400
            except Exception as e:
                logger.error(f"Chat completions request failed: {e}")
                return jsonify({
                    "error": {"message": str(e), "type": "internal_error"}
                }), 500

        return app

    def start_server_immediate(self):
        """Start the server immediately and initialize in background

        Uses stored job_data and docker_env_vars from factory method
        """
        try:
            # Start initialization in background if data is available
            if hasattr(self, '_job_data') and hasattr(self, '_docker_env_vars'):
                init_thread = threading.Thread(
                    target=self._initialize_background,
                    args=(self._job_data, self._docker_env_vars)
                )
                init_thread.daemon = True
                init_thread.start()

            # Start server immediately (don't wait for initialization or model loading)
            app = self.create_flask_app()
            logger.info(f"Starting {self.__class__.__name__} Server on port {self.port}")
            logger.info("Server starting immediately - initialization and model loading in background")
            logger.info(
                f"Health monitor will start after model loads (idle timeout: "
                f"{self.idle_timeout_minutes} minutes)"
            )
            app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
            return True

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
        finally:
            # Cleanup on server shutdown
            self.shutdown_health_monitor()

    def start_server(self, load_model_params: Dict[str, Any] = None):
        """Start the persistent model server immediately and load model in background

        Args:
            load_model_params: Parameters for model loading
        """
        try:
            # Start model loading in background
            if load_model_params is None:
                load_model_params = {}
            self.load_model(**load_model_params)

            # Start server immediately (don't wait for model to load)
            app = self.create_flask_app()
            logger.info(f"Starting {self.__class__.__name__} Server on port {self.port}")
            logger.info("Server starting immediately - model will load in background")
            logger.info(
                f"Health monitor will start after model loads (idle timeout: "
                f"{self.idle_timeout_minutes} minutes)"
            )
            app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
            return True

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
        finally:
            # Cleanup on server shutdown
            self.shutdown_health_monitor()

    def shutdown_server(self):
        """Gracefully shutdown the server and cleanup resources"""
        logger.info("Shutting down inference microservice server")
        self.shutdown_health_monitor()
        # Additional cleanup can be added here if needed

    @classmethod
    def create_from_tao_job(cls, job_data: Dict[str, Any], docker_env_vars: Dict[str, Any],
                            port: int = 8080):
        """Factory method to create model server from TAO job data

        Args:
            job_data: TAO job metadata
            docker_env_vars: Docker environment variables
            port: Server port

        Returns:
            Configured model server instance (ready to start)
        """
        # Create server instance immediately with minimal data
        server_instance = cls(
            job_id=job_data["job_id"],
            port=port,
            **{}  # Empty model params initially
        )

        # Store initialization data for later use
        server_instance._job_data = job_data
        server_instance._docker_env_vars = docker_env_vars

        return server_instance
