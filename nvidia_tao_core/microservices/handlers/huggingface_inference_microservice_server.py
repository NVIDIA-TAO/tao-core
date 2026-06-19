# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Inference Microservice Server - Supports LLM, VLM, and Diffusion models

This module provides a generic inference server for HuggingFace models, supporting:
- Text-only LLMs (e.g., meta-llama/Llama-3-8B, mistralai/Mistral-7B-v0.1)
- Vision-Language Models (e.g., Qwen/Qwen-VL-Chat, llava-hf/llava-1.5-7b-hf, nvidia/Cosmos-Reason1-7B)
- Diffusion Models for image/video generation (e.g., nvidia/Cosmos-Predict2, stable-diffusion)

Usage:
    Any container using tao-core can use this server by specifying a HuggingFace model name.
"""

import os
import ast
import json
import base64
import logging
import argparse
import requests
import tempfile
import traceback
from PIL import Image
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

from .base_inference_microservice_server import BaseInferenceMicroserviceServer

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


def _apply_qwen3vl_cudnn_workaround() -> None:
    # cuDNN 9.20 on sm_80 (A100) has no Conv3d engine for kernel spatial size
    # >= 16 in bf16/fp16 with the default NCDHW layout — Qwen3-VL's PatchEmbed
    # (patch_size=16, temporal_patch_size=2) hits this exactly and raises
    # "GET was unable to find an engine to execute this computation". cuDNN's
    # NDHWC engines cover the same shape, so running the conv in
    # channels_last_3d memory format avoids the gap without touching dtype.
    try:
        import torch
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed
    except ImportError:
        return

    if getattr(Qwen3VLVisionPatchEmbed.forward, "_tao_channels_last_3d", False):
        return

    def forward(self, hidden_states):
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size,
        ).to(dtype=target_dtype, memory_format=torch.channels_last_3d)
        if not self.proj.weight.is_contiguous(memory_format=torch.channels_last_3d):
            self.proj.weight.data = self.proj.weight.data.to(memory_format=torch.channels_last_3d)
        return self.proj(hidden_states).reshape(-1, self.embed_dim)

    forward._tao_channels_last_3d = True
    Qwen3VLVisionPatchEmbed.forward = forward
    logger.info("Applied channels_last_3d workaround for Qwen3-VL PatchEmbed (cuDNN 9.20 conv3d gap)")


_apply_qwen3vl_cudnn_workaround()


# Model type detection based on architecture names
VLM_ARCHITECTURES = {
    # Qwen VL models
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "QWenLMHeadModel",  # Old Qwen-VL
    # LLaVA models
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "LlavaMistralForCausalLM",
    "LlavaLlamaForCausalLM",
    # InternVL models
    "InternVLChatModel",
    # BLIP models
    "Blip2ForConditionalGeneration",
    "BlipForConditionalGeneration",
    # Idefics models
    "IdeficsForVisionText2Text",
    "Idefics2ForConditionalGeneration",
    # PaliGemma
    "PaliGemmaForConditionalGeneration",
    # Florence
    "Florence2ForConditionalGeneration",
    # Phi-3 Vision
    "Phi3VForCausalLM",
    # Molmo
    "MolmoForCausalLM",
    # Pixtral
    "PixtralForConditionalGeneration",
}

# Known VLM model name patterns (fallback detection)
VLM_MODEL_PATTERNS = [
    "qwen-vl", "qwen2-vl", "qwen2.5-vl",
    "llava", "llava-next", "llava-1.5", "llava-1.6",
    "internvl", "intern-vl",
    "blip", "blip2",
    "idefics", "idefics2",
    "paligemma",
    "florence",
    "phi-3-vision", "phi3-vision",
    "molmo",
    "pixtral",
    "cogvlm",
    "minicpm-v",
    "cosmos-reason",  # Cosmos-Reason1 is a VLM
]

# Known Diffusion model name patterns
DIFFUSION_MODEL_PATTERNS = [
    # NVIDIA Cosmos Predict2 models
    "cosmos-predict", "cosmos-predict2",
    # Stable Diffusion family
    "stable-diffusion", "stabilityai/stable",
    "sdxl", "sd-turbo", "sd3",
    # FLUX models
    "flux",
    # Other diffusion models
    "kandinsky", "deepfloyd", "dall-e",
    "pixart", "playground",
    "latent-consistency",
]

# Diffusion model task types based on model name patterns
DIFFUSION_TASK_PATTERNS = {
    "text2image": ["text2image", "t2i", "txt2img"],
    "image2video": ["image2video", "i2v", "img2vid"],
    "video2world": ["video2world", "v2w"],
    "text2video": ["text2video", "t2v", "txt2vid"],
    "image2image": ["image2image", "i2i", "img2img"],
    "inpainting": ["inpaint"],
}


class HuggingFaceInferenceMicroserviceServer(BaseInferenceMicroserviceServer):
    """HuggingFace-based inference microservice supporting LLM, VLM, and Diffusion models

    This server automatically detects model type and handles:
    - Text generation for LLMs
    - Text + image/video understanding for VLMs
    - Image/video generation for Diffusion models (Cosmos-Predict2, Stable Diffusion, etc.)

    Attributes:
        model_type: Detected model type ('llm', 'vlm', or 'diffusion')
        diffusion_task: For diffusion models, the task type ('text2image', 'image2video', etc.)
        processor: HuggingFace processor/tokenizer
        pipeline: Diffusion pipeline (for diffusion models)
        is_vlm: Whether the loaded model supports vision inputs
        is_diffusion: Whether the loaded model is a diffusion model
    """

    def __init__(self, job_id: str, port: int = 8080, **model_params):
        """Initialize HuggingFace inference server

        Args:
            job_id: Unique job identifier
            port: Server port (default 8080)
            **model_params: Model-specific parameters
        """
        super().__init__(job_id, port, **model_params)
        self.model_state_dir = "/tmp/hf_models"
        self.processor = None
        self.tokenizer = None
        self.model_type = None  # 'llm', 'vlm', 'diffusion', or 'custom'
        self.diffusion_task = None  # 'text2image', 'image2video', 'video2world', etc.
        self.is_vlm = False
        self.is_diffusion = False
        self.pipeline = None  # Diffusion pipeline
        self.model_config = None
        self._generation_config = None
        # Custom pipeline loader/inference functions (Python code strings)
        self.custom_pipeline_loader = None
        self.custom_inference_fn = None

    def get_supported_file_extensions(self) -> Tuple[List[str], List[str]]:
        """Get supported file extensions for this model

        Returns:
            Tuple of (image_extensions, video_extensions)
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']
        video_extensions = ['.mp4', '.mkv', '.webm', '.avi', '.mov']
        return image_extensions, video_extensions

    def _detect_model_type(self, model_name: str, model_config: Any = None) -> str:
        """Detect if model is LLM, VLM, or Diffusion based on architecture/name

        Args:
            model_name: HuggingFace model name or path
            model_config: Model configuration object (optional for diffusion models)

        Returns:
            'diffusion' for diffusion models, 'vlm' for vision-language models, 'llm' for text-only
        """
        model_name_lower = model_name.lower()

        # First check for diffusion model patterns (highest priority)
        for pattern in DIFFUSION_MODEL_PATTERNS:
            if pattern in model_name_lower:
                logger.info(f"Detected Diffusion model from name pattern: {pattern}")
                # Detect diffusion task type
                self.diffusion_task = self._detect_diffusion_task(model_name_lower)
                return 'diffusion'

        # If no config available, cannot check architectures
        if model_config is None:
            logger.info("No config available, defaulting to LLM")
            return 'llm'

        # Check architecture from config for VLM
        architectures = getattr(model_config, 'architectures', []) or []
        for arch in architectures:
            if arch in VLM_ARCHITECTURES:
                logger.info(f"Detected VLM model from architecture: {arch}")
                return 'vlm'

        # Check model name patterns for VLM
        for pattern in VLM_MODEL_PATTERNS:
            if pattern in model_name_lower:
                logger.info(f"Detected VLM model from name pattern: {pattern}")
                return 'vlm'

        # Check if config has vision-related attributes
        vision_attrs = ['vision_config', 'visual', 'image_size', 'vision_tower']
        for attr in vision_attrs:
            if hasattr(model_config, attr) and getattr(model_config, attr) is not None:
                logger.info(f"Detected VLM model from config attribute: {attr}")
                return 'vlm'

        logger.info("Model detected as LLM (text-only)")
        return 'llm'

    def _detect_diffusion_task(self, model_name_lower: str) -> str:
        """Detect the diffusion task type from model name

        Args:
            model_name_lower: Lowercase model name

        Returns:
            Task type string ('text2image', 'image2video', 'video2world', etc.)
        """
        for task, patterns in DIFFUSION_TASK_PATTERNS.items():
            for pattern in patterns:
                if pattern in model_name_lower:
                    logger.info(f"Detected diffusion task: {task}")
                    return task

        # Default to text2image if no specific task detected
        logger.info("No specific diffusion task detected, defaulting to text2image")
        return 'text2image'

    def _get_torch_dtype(self, dtype_str: str):
        """Convert string dtype to torch dtype

        Args:
            dtype_str: String representation of dtype

        Returns:
            torch dtype object
        """
        import torch
        dtype_map = {
            'auto': 'auto',
            'float16': torch.float16,
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'fp16': torch.float16,
            'fp32': torch.float32,
            'bf16': torch.bfloat16,
        }
        return dtype_map.get(dtype_str.lower(), 'auto')

    def _load_with_custom_loader(self, custom_loader_code: str, kwargs: Dict) -> bool:
        """Load model using a custom Python function

        The custom_loader_code should define a function called `load_pipeline` that takes
        model_name and **kwargs and returns a tuple of (pipeline/model, model_type).

        Example custom_loader_code:
        '''
        def load_pipeline(model_name, **kwargs):
            from diffusers import WanPipeline
            import torch
            pipe = WanPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16
            ).to("cuda")
            return pipe, "diffusion"
        '''

        Args:
            custom_loader_code: Python code string defining load_pipeline function
            kwargs: Model loading parameters

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            import torch

            logger.info("Loading model with custom pipeline loader")

            # Create execution namespace with common imports
            exec_namespace = {
                'torch': torch,
                '__builtins__': __builtins__,
            }

            # Execute the custom code to define functions
            exec(custom_loader_code, exec_namespace)  # pylint: disable=exec-used

            # Check if load_pipeline function was defined
            if 'load_pipeline' not in exec_namespace:
                raise ValueError(
                    "custom_pipeline_loader must define a 'load_pipeline(model_name, **kwargs)' function"
                )

            load_pipeline_fn = exec_namespace['load_pipeline']

            # Get model name
            model_name = kwargs.get('model_path') or kwargs.get('model_name')

            # Call the custom loader
            result = load_pipeline_fn(model_name, **kwargs)

            # Handle return value - can be (pipeline, model_type) or just pipeline
            if isinstance(result, tuple) and len(result) >= 2:
                pipeline_or_model, model_type = result[0], result[1]
            else:
                pipeline_or_model = result
                model_type = 'custom'

            # Store the loaded model/pipeline
            self.model_type = model_type
            if model_type == 'diffusion':
                self.pipeline = pipeline_or_model
                self.model = pipeline_or_model
                self.is_diffusion = True
                # Try to detect diffusion task from model name
                self.diffusion_task = self._detect_diffusion_task(model_name.lower() if model_name else '')
            else:
                self.model = pipeline_or_model
                self.is_vlm = model_type == 'vlm'

            # Store custom inference function if also provided in the code
            if 'run_inference' in exec_namespace:
                self.custom_inference_fn = custom_loader_code
                logger.info("Custom run_inference function also detected")

            # Also check for separately-provided custom_inference_fn in kwargs
            custom_inference_fn = kwargs.get('custom_inference_fn')
            if custom_inference_fn and not self.custom_inference_fn:
                if kwargs.get('custom_inference_fn_encoded'):
                    custom_inference_fn = base64.b64decode(custom_inference_fn).decode('utf-8')
                    logger.debug("Decoded base64 custom_inference_fn in _load_with_custom_loader")
                self.custom_inference_fn = custom_inference_fn

            logger.info(f"Successfully loaded model with custom loader (type: {model_type})")
            return True

        except Exception as e:
            logger.error(f"Failed to load with custom loader: {e}")
            logger.error(traceback.format_exc())
            return False

    def load_model_into_memory(self, **kwargs) -> bool:
        """Load HuggingFace model into memory

        Automatically detects model type (LLM, VLM, or Diffusion) and loads appropriate
        model class and processor/tokenizer/pipeline.

        Args:
            **kwargs: Model configuration parameters
                - model_path: HuggingFace model name or local path (required)
                - torch_dtype: Data type (auto, float16, bfloat16, float32)
                - device_map: Device mapping strategy (auto, cuda, cpu)
                - trust_remote_code: Whether to trust remote code (default: True)
                - use_flash_attention_2: Whether to use Flash Attention 2
                - max_memory: Memory limits per device
                - hf_token: HuggingFace token for private models
                - custom_pipeline_loader: Python code string defining a load_pipeline() function
                - custom_inference_fn: Python code string defining a run_inference() function

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Check for custom pipeline loader first
            custom_loader = kwargs.get('custom_pipeline_loader')
            if custom_loader:
                # Decode base64 if it was encoded (to avoid shell escaping issues)
                if kwargs.get('custom_pipeline_loader_encoded'):
                    custom_loader = base64.b64decode(custom_loader).decode('utf-8')
                    logger.debug("Decoded base64 custom_pipeline_loader")
                return self._load_with_custom_loader(custom_loader, kwargs)

            # Store custom inference function if provided
            custom_inference = kwargs.get('custom_inference_fn')
            if custom_inference:
                # Decode base64 if it was encoded
                if kwargs.get('custom_inference_fn_encoded'):
                    custom_inference = base64.b64decode(custom_inference).decode('utf-8')
                    logger.debug("Decoded base64 custom_inference_fn")
                self.custom_inference_fn = custom_inference

            # Extract parameters
            model_name = kwargs.get('model_path') or kwargs.get('model_name')
            if not model_name:
                raise ValueError("model_path or model_name is required")

            torch_dtype = self._get_torch_dtype(kwargs.get('torch_dtype', 'auto'))
            device_map = kwargs.get('device_map', 'auto')
            trust_remote_code = kwargs.get('trust_remote_code', True)
            use_flash_attention = kwargs.get('use_flash_attention_2', False)
            hf_token = kwargs.get('hf_token') or os.getenv('HF_TOKEN')
            max_memory = kwargs.get('max_memory')

            logger.info(f"Loading HuggingFace model: {model_name}")

            # First, detect model type from name (for diffusion models, config may not exist)
            preliminary_type = self._detect_model_type(model_name, None)

            if preliminary_type == 'diffusion':
                # Load diffusion model using diffusers
                self.model_type = 'diffusion'
                self.is_diffusion = True
                return self._load_diffusion_model(model_name, torch_dtype, device_map,
                                                  trust_remote_code, hf_token)

            # For LLM/VLM, load config first
            from transformers import AutoConfig
            config_kwargs = {'trust_remote_code': trust_remote_code}
            if hf_token:
                config_kwargs['token'] = hf_token

            self.model_config = AutoConfig.from_pretrained(model_name, **config_kwargs)
            self.model_type = self._detect_model_type(model_name, self.model_config)
            self.is_vlm = self.model_type == 'vlm'
            self.is_diffusion = self.model_type == 'diffusion'

            # If detected as diffusion after config check, load diffusion model
            if self.is_diffusion:
                return self._load_diffusion_model(model_name, torch_dtype, device_map,
                                                  trust_remote_code, hf_token)

            # Prepare model loading kwargs for LLM/VLM
            model_kwargs = {
                'torch_dtype': torch_dtype,
                'device_map': device_map,
                'trust_remote_code': trust_remote_code,
            }

            if hf_token:
                model_kwargs['token'] = hf_token
            if max_memory:
                model_kwargs['max_memory'] = max_memory
            if use_flash_attention:
                model_kwargs['attn_implementation'] = 'flash_attention_2'

            # Load model and processor based on type
            if self.is_vlm:
                self._load_vlm_model(model_name, model_kwargs, config_kwargs)
            else:
                self._load_llm_model(model_name, model_kwargs, config_kwargs)

            # Store generation config if available
            if hasattr(self.model, 'generation_config'):
                self._generation_config = self.model.generation_config

            logger.info(f"Successfully loaded {self.model_type.upper()} model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            logger.error(traceback.format_exc())
            return False

    def _load_diffusion_model(self, model_name: str, torch_dtype, device_map: str,
                              trust_remote_code: bool, hf_token: str = None) -> bool:
        """Load diffusion model using diffusers library

        Args:
            model_name: HuggingFace model name or path
            torch_dtype: Torch data type
            device_map: Device mapping strategy
            trust_remote_code: Whether to trust remote code
            hf_token: HuggingFace token for gated models

        Returns:
            True if loaded successfully
        """
        try:
            import torch
            from diffusers import DiffusionPipeline, AutoPipelineForText2Image, AutoPipelineForImage2Image

            logger.info(f"Loading Diffusion model: {model_name} (task: {self.diffusion_task})")

            # Prepare pipeline kwargs
            pipeline_kwargs = {
                'torch_dtype': torch_dtype if torch_dtype != 'auto' else torch.bfloat16,
                'trust_remote_code': trust_remote_code,
            }

            if hf_token:
                pipeline_kwargs['token'] = hf_token

            # Try to load with specific pipeline based on task
            model_name_lower = model_name.lower()

            # Check for model-specific pipelines
            if 'cosmos-predict2' in model_name_lower or 'cosmos-predict' in model_name_lower:
                self.pipeline = self._load_cosmos_predict_pipeline(model_name, pipeline_kwargs)
            elif self.diffusion_task == 'text2image':
                try:
                    self.pipeline = AutoPipelineForText2Image.from_pretrained(
                        model_name, **pipeline_kwargs
                    )
                except Exception:
                    logger.info("AutoPipelineForText2Image failed, trying DiffusionPipeline")
                    self.pipeline = DiffusionPipeline.from_pretrained(
                        model_name, **pipeline_kwargs
                    )
            elif self.diffusion_task == 'image2image':
                try:
                    self.pipeline = AutoPipelineForImage2Image.from_pretrained(
                        model_name, **pipeline_kwargs
                    )
                except Exception:
                    self.pipeline = DiffusionPipeline.from_pretrained(
                        model_name, **pipeline_kwargs
                    )
            else:
                # Generic diffusion pipeline
                self.pipeline = DiffusionPipeline.from_pretrained(
                    model_name, **pipeline_kwargs
                )

            # Move to GPU if available
            if torch.cuda.is_available() and device_map != 'cpu':
                if hasattr(self.pipeline, 'to'):
                    self.pipeline = self.pipeline.to('cuda')
                # Enable memory optimizations if available
                if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                    try:
                        # Only use CPU offload for very large models or limited VRAM
                        pass  # self.pipeline.enable_model_cpu_offload()
                    except Exception:
                        pass

            # Store reference in self.model for compatibility
            self.model = self.pipeline

            logger.info(f"Successfully loaded Diffusion model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load Diffusion model: {e}")
            logger.error(traceback.format_exc())
            return False

    def _load_cosmos_predict_pipeline(self, model_name: str, pipeline_kwargs: Dict) -> Any:
        """Load NVIDIA Cosmos-Predict2 specific pipeline

        Args:
            model_name: Model name (e.g., nvidia/Cosmos-Predict2-2B-Text2Image)
            pipeline_kwargs: Pipeline loading kwargs

        Returns:
            Loaded pipeline
        """
        from diffusers import DiffusionPipeline

        model_name_lower = model_name.lower()

        # Try Cosmos-specific pipeline classes first
        try:
            if 'text2image' in model_name_lower:
                # Try CosmosTextToImagePipeline
                try:
                    from diffusers import CosmosTextToImagePipeline
                    logger.info("Loading with CosmosTextToImagePipeline")
                    return CosmosTextToImagePipeline.from_pretrained(model_name, **pipeline_kwargs)
                except ImportError:
                    logger.info("CosmosTextToImagePipeline not available, using DiffusionPipeline")

            elif 'video2world' in model_name_lower or 'image2video' in model_name_lower:
                # Try CosmosVideoToWorldPipeline
                try:
                    from diffusers import CosmosVideoToWorldPipeline
                    logger.info("Loading with CosmosVideoToWorldPipeline")
                    return CosmosVideoToWorldPipeline.from_pretrained(model_name, **pipeline_kwargs)
                except ImportError:
                    logger.info("CosmosVideoToWorldPipeline not available, using DiffusionPipeline")

        except Exception as e:
            logger.warning(f"Cosmos-specific pipeline failed: {e}, falling back to DiffusionPipeline")

        # Fallback to generic DiffusionPipeline
        return DiffusionPipeline.from_pretrained(model_name, **pipeline_kwargs)

    def _load_llm_model(self, model_name: str, model_kwargs: Dict, config_kwargs: Dict):
        """Load text-only LLM model

        Args:
            model_name: HuggingFace model name or path
            model_kwargs: Model loading kwargs
            config_kwargs: Config loading kwargs
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading LLM model with AutoModelForCausalLM")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, **config_kwargs
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.processor = self.tokenizer  # For unified interface

    def _load_vlm_model(self, model_name: str, model_kwargs: Dict, config_kwargs: Dict):
        """Load vision-language model

        Args:
            model_name: HuggingFace model name or path
            model_kwargs: Model loading kwargs
            config_kwargs: Config loading kwargs
        """
        from transformers import AutoModelForVision2Seq, AutoProcessor

        logger.info("Loading VLM model with AutoModelForVision2Seq")

        # Try AutoModelForVision2Seq first, fall back to AutoModel
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name, **model_kwargs
            )
        except Exception:
            logger.info("AutoModelForVision2Seq failed, trying AutoModel")
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                model_name, **model_kwargs
            )

        # Load processor (handles both text and vision)
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name, **config_kwargs
            )
        except Exception as e:
            logger.warning(f"AutoProcessor failed, trying AutoTokenizer: {e}")
            from transformers import AutoTokenizer
            self.processor = AutoTokenizer.from_pretrained(
                model_name, **config_kwargs
            )

        # Also store tokenizer reference for convenience
        if hasattr(self.processor, 'tokenizer'):
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor

    def _prepare_llm_inputs(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Prepare inputs for LLM inference

        Args:
            prompt: Text prompt
            **kwargs: Additional parameters (system_prompt, chat_format, etc.)

        Returns:
            Tokenized inputs ready for model
        """
        system_prompt = kwargs.get('system_prompt', '')
        use_chat_template = kwargs.get('use_chat_template', True)

        # Try to use chat template if available
        if use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Chat template failed, using raw prompt: {e}")
                text = prompt
        else:
            text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True
        )

        return inputs.to(self.model.device)

    def _prepare_vlm_inputs(
        self, prompt: str, images: List[Any] = None, videos: List[Any] = None, **kwargs
    ) -> Dict[str, Any]:
        """Prepare inputs for VLM inference

        Args:
            prompt: Text prompt
            images: List of images (PIL.Image, file paths, or URLs)
            videos: List of video paths
            **kwargs: Additional parameters

        Returns:
            Processed inputs ready for model
        """
        system_prompt = kwargs.get('system_prompt', '')

        # Build conversation format for VLM
        content = []

        # Add images
        if images:
            for img in images:
                if isinstance(img, str):
                    # Load image from path or URL
                    img = self._load_image(img)
                content.append({"type": "image", "image": img})

        # Add videos (if supported by model)
        if videos:
            for video_path in videos:
                content.append({"type": "video", "video": video_path})

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        # Create conversation
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        # Process with processor
        try:
            if hasattr(self.processor, 'apply_chat_template'):
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                # Extract images for processor
                pil_images = [c.get('image') for c in content if c.get('type') == 'image' and 'image' in c]
                inputs = self.processor(
                    text=[text],
                    images=pil_images if pil_images else None,
                    return_tensors="pt",
                    padding=True
                )
            else:
                # Fallback for processors without chat template
                pil_images = [c.get('image') for c in content if c.get('type') == 'image' and 'image' in c]
                inputs = self.processor(
                    text=prompt,
                    images=pil_images[0] if pil_images else None,
                    return_tensors="pt"
                )
        except Exception as e:
            logger.warning(f"VLM processing with images failed, trying text-only: {e}")
            inputs = self.processor(
                text=prompt,
                return_tensors="pt"
            )

        return inputs.to(self.model.device)

    def _load_image(self, image_input: Any) -> Any:
        """Load image from various input formats

        Args:
            image_input: Image path, URL, base64 string, or PIL Image

        Returns:
            PIL.Image object
        """
        if isinstance(image_input, Image.Image):
            return image_input

        if isinstance(image_input, str):
            # Check if base64 encoded
            if image_input.startswith('data:image'):
                # Data URL format
                _, encoded = image_input.split(',', 1)
                image_data = base64.b64decode(encoded)
                return Image.open(BytesIO(image_data))

            # Check if URL
            if image_input.startswith(('http://', 'https://')):
                response = requests.get(image_input, timeout=30)
                return Image.open(BytesIO(response.content))

            # Reject cloud-storage URIs — the inference service does not
            # authenticate to any cloud. Callers must send inline data or
            # plain http(s) URLs.
            self._reject_cloud_uri(image_input)

            # Local file path
            return Image.open(image_input)

        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def run_model_inference(self, **kwargs) -> Dict[str, Any]:
        """Run inference on the loaded model

        Args:
            **kwargs: Inference parameters
                For LLM/VLM:
                - prompt: Text prompt (required)
                - images: List of images for VLM (optional)
                - videos: List of video paths for VLM (optional)
                - media: Alternative to images/videos (list of media files)
                - max_new_tokens: Maximum tokens to generate (default: 512)
                - temperature: Sampling temperature (default: 0.7)
                - top_p: Nucleus sampling parameter (default: 0.9)
                - top_k: Top-k sampling parameter (default: 50)
                - system_prompt: System prompt for chat models

                For Diffusion models:
                - prompt: Text prompt for generation (required)
                - negative_prompt: Negative prompt for guidance (optional)
                - num_inference_steps: Number of denoising steps (default: 50)
                - guidance_scale: Classifier-free guidance scale (default: 7.5)
                - width: Output image width (optional)
                - height: Output image height (optional)
                - seed: Random seed for reproducibility (optional)
                - num_images: Number of images to generate (default: 1)

        Returns:
            Inference results dictionary
        """
        # Check for custom inference function first
        if self.custom_inference_fn:
            return self._run_custom_inference(**kwargs)

        # Route to appropriate inference method based on model type
        if self.is_diffusion:
            return self._run_diffusion_inference(**kwargs)
        return self._run_llm_vlm_inference(**kwargs)

    def _run_custom_inference(self, **kwargs) -> Dict[str, Any]:
        """Run inference using custom inference function

        The custom_inference_fn code should define a function called `run_inference`
        that takes (pipeline, **kwargs) and returns a dict with results.

        Example:
        '''
        def run_inference(pipeline, **kwargs):
            prompt = kwargs.get('prompt', '')
            output = pipeline(
                prompt=prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
            )
            # Save video frames
            frames = output.frames[0]
            # ... save to file ...
            return {
                "response": "Generated video",
                "generated_files": ["/path/to/video.gif"],
            }
        '''

        Args:
            **kwargs: Inference parameters

        Returns:
            Inference results dictionary
        """
        try:
            import torch

            logger.info("Running inference with custom function")

            # Create execution namespace. The inference service has no cloud
            # storage; custom code must return generated artifacts via the
            # response payload (or write them to local paths).
            exec_namespace = {
                'torch': torch,
                'pipeline': self.pipeline or self.model,
                'model': self.model,
                'processor': self.processor,
                'tokenizer': self.tokenizer,
                'tempfile': tempfile,
                'os': os,
                'logger': logger,
                '__builtins__': __builtins__,
            }

            # Execute the custom code
            exec(self.custom_inference_fn, exec_namespace)  # pylint: disable=exec-used

            # Check if run_inference function was defined
            if 'run_inference' not in exec_namespace:
                raise ValueError(
                    "custom_inference_fn must define a 'run_inference(pipeline, **kwargs)' function"
                )

            run_inference_fn = exec_namespace['run_inference']

            # Call the custom inference function
            result = run_inference_fn(self.pipeline or self.model, **kwargs)

            # Ensure result is a dict
            if not isinstance(result, dict):
                result = {"response": str(result), "model_type": "custom"}

            if "model_type" not in result:
                result["model_type"] = "custom"

            return result

        except Exception as e:
            logger.error(f"Custom inference failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def _run_llm_vlm_inference(self, **kwargs) -> Dict[str, Any]:
        """Run inference for LLM/VLM models

        Args:
            **kwargs: Inference parameters

        Returns:
            Inference results dictionary
        """
        try:
            # Extract parameters
            prompt = kwargs.get('prompt', kwargs.get('text', ''))
            if not prompt:
                raise ValueError("'prompt' or 'text' parameter is required")

            # Get media inputs
            images = kwargs.get('images', [])
            videos = kwargs.get('videos', [])
            media = kwargs.get('media', [])

            # Ensure media is a list
            if media and isinstance(media, str):
                media = [media]

            # Process media parameter (can contain both images and videos)
            if media:
                images, videos = self._categorize_media(media)

            # Generation parameters
            max_new_tokens = kwargs.get('max_new_tokens', 512)
            temperature = kwargs.get('temperature', 0.7)
            top_p = kwargs.get('top_p', 0.9)
            top_k = kwargs.get('top_k', 50)
            do_sample = kwargs.get('do_sample', temperature > 0)
            repetition_penalty = kwargs.get('repetition_penalty', 1.0)

            # Remove keys that are passed explicitly to avoid duplicate argument error
            excluded_keys = (
                'prompt', 'text', 'images', 'videos', 'media',
                'max_new_tokens', 'temperature', 'top_p', 'top_k',
                'do_sample', 'repetition_penalty'
            )
            prepare_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_keys}

            # Prepare inputs based on model type
            if self.is_vlm and (images or videos):
                inputs = self._prepare_vlm_inputs(prompt, images=images, videos=videos, **prepare_kwargs)
            else:
                inputs = self._prepare_llm_inputs(prompt, **prepare_kwargs)

            # Generation config
            gen_kwargs = {
                'max_new_tokens': max_new_tokens,
                'do_sample': do_sample,
                'pad_token_id': self.tokenizer.pad_token_id if self.tokenizer else None,
            }

            if do_sample:
                gen_kwargs.update({
                    'temperature': temperature,
                    'top_p': top_p,
                    'top_k': top_k,
                })

            if repetition_penalty != 1.0:
                gen_kwargs['repetition_penalty'] = repetition_penalty

            # Run generation
            import torch
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_kwargs)

            # Decode output
            input_len = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0

            # Trim input tokens from output
            if input_len > 0:
                generated_ids_trimmed = [
                    out_ids[input_len:]
                    for out_ids in generated_ids
                ]
            else:
                generated_ids_trimmed = generated_ids

            # Use tokenizer or processor for decoding
            decoder = self.tokenizer if self.tokenizer else self.processor
            output_text = decoder.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            response = output_text[0] if output_text else ""

            return {
                "response": response,
                "model_type": self.model_type,
                "is_vlm": self.is_vlm,
                "input_tokens": input_len,
                "output_tokens": len(generated_ids_trimmed[0]) if generated_ids_trimmed else 0,
                "total_images": len(images) if images else 0,
                "total_videos": len(videos) if videos else 0,
            }

        except Exception as e:
            logger.error(f"LLM/VLM inference failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def _run_diffusion_inference(self, **kwargs) -> Dict[str, Any]:
        """Run inference for Diffusion models (image/video generation)

        Args:
            **kwargs: Diffusion-specific parameters
                - prompt: Text prompt for generation
                - negative_prompt: Negative prompt for guidance
                - num_inference_steps: Number of denoising steps
                - guidance_scale: Classifier-free guidance scale
                - width: Output width
                - height: Output height
                - seed: Random seed
                - num_images: Number of images to generate
                - image: Input image for image2image/video2world tasks
                - video: Input video for video2world tasks

        Returns:
            Inference results with generated image/video paths
        """
        try:
            import torch

            prompt = kwargs.get('prompt', kwargs.get('text', ''))
            if not prompt:
                raise ValueError("'prompt' parameter is required for diffusion models")

            # Diffusion-specific parameters
            negative_prompt = kwargs.get('negative_prompt', '')
            num_inference_steps = kwargs.get('num_inference_steps', 50)
            guidance_scale = kwargs.get('guidance_scale', 7.5)
            width = kwargs.get('width')
            height = kwargs.get('height')
            seed = kwargs.get('seed')
            num_images = kwargs.get('num_images', 1)

            # Set random seed if provided
            generator = None
            if seed is not None:
                generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
                generator.manual_seed(seed)

            # Build pipeline call kwargs
            pipe_kwargs = {
                'prompt': prompt,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
            }

            if negative_prompt:
                pipe_kwargs['negative_prompt'] = negative_prompt
            if generator:
                pipe_kwargs['generator'] = generator
            if width:
                pipe_kwargs['width'] = width
            if height:
                pipe_kwargs['height'] = height

            # Handle input images/videos for image2image or video2world tasks
            if self.diffusion_task in ['image2image', 'video2world', 'image2video']:
                input_image = kwargs.get('image') or kwargs.get('media')
                if input_image:
                    if isinstance(input_image, str):
                        input_image = self._load_image(input_image)
                    pipe_kwargs['image'] = input_image

            # Run the diffusion pipeline
            logger.info(f"Running diffusion inference with task: {self.diffusion_task}")
            output = self.pipeline(**pipe_kwargs)

            # Process outputs based on type. Generated artifacts stay on the
            # local filesystem of the inference container; their paths are
            # returned in the response so callers can fetch the bytes (e.g.
            # via the /v1/chat/completions video data-URI flow) without the
            # service touching cloud storage.
            output_paths = []

            # Handle image outputs
            if hasattr(output, 'images') and output.images:
                for i, img in enumerate(output.images[:num_images]):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"generated_{timestamp}_{i}.png"
                    local_path = os.path.join(tempfile.gettempdir(), filename)
                    img.save(local_path)
                    output_paths.append(local_path)

            # Handle video outputs (for video generation models)
            if hasattr(output, 'frames') and output.frames is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_{timestamp}.gif"
                local_path = os.path.join(tempfile.gettempdir(), filename)
                local_path = self._save_video_frames(output.frames, local_path)
                output_paths.append(local_path)

            return {
                "response": (
                    f"Generated {len(output_paths)} "
                    f"{'image' if self.diffusion_task == 'text2image' else 'media'}(s)"
                ),
                "model_type": "diffusion",
                "diffusion_task": self.diffusion_task,
                "generated_files": output_paths,
                "num_generated": len(output_paths),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
            }

        except Exception as e:
            logger.error(f"Diffusion inference failed: {e}")
            logger.error(traceback.format_exc())
            raise

    @staticmethod
    def _is_frame_like(value: Any) -> bool:
        """Return whether value looks like one image frame."""
        if isinstance(value, Image.Image):
            return True
        shape = getattr(value, "shape", None)
        if shape is None:
            return False
        return len(shape) in (2, 3)

    @classmethod
    def _normalize_video_frames(cls, frames: Any) -> List[Any]:
        """Flatten common diffusion output frame containers to one frame list."""
        if frames is None:
            return []

        shape = getattr(frames, "shape", None)
        if shape is not None and len(shape) >= 4:
            return [frames[i] for i in range(len(frames))]

        if isinstance(frames, (str, bytes, Image.Image)):
            return [frames]

        try:
            frame_count = len(frames)
        except TypeError:
            return [frames]

        if frame_count == 0:
            return []

        first = frames[0]
        if (
            not cls._is_frame_like(first) and
            not isinstance(first, (str, bytes)) and
            hasattr(first, "__len__")
        ):
            return cls._normalize_video_frames(first)

        return list(frames)

    @staticmethod
    def _frame_to_pil(frame: Any) -> Image.Image:
        """Convert a diffusion frame object to an RGB Pillow image."""
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")

        import numpy as np

        if hasattr(frame, "detach"):
            frame = frame.detach()
        if hasattr(frame, "cpu"):
            frame = frame.cpu()
        if hasattr(frame, "numpy"):
            frame = frame.numpy()

        array = np.asarray(frame)
        if array.ndim == 0:
            raise ValueError("Video frame must be an image-like array")

        if (
            array.ndim == 3 and
            array.shape[0] in (1, 3, 4) and
            array.shape[-1] not in (1, 3, 4)
        ):
            array = np.moveaxis(array, 0, -1)

        if np.issubdtype(array.dtype, np.floating):
            array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
            if array.size and array.min() >= 0.0 and array.max() <= 1.0:
                array = array * 255.0
            elif array.size and array.min() >= -1.0 and array.max() <= 1.0:
                array = (array + 1.0) * 127.5
            array = np.clip(array, 0, 255).astype(np.uint8)
        elif array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        if array.ndim == 2:
            return Image.fromarray(array).convert("RGB")
        if array.ndim == 3 and array.shape[-1] == 1:
            return Image.fromarray(array[:, :, 0]).convert("RGB")
        if array.ndim == 3 and array.shape[-1] in (3, 4):
            return Image.fromarray(array).convert("RGB")

        raise ValueError(f"Unsupported video frame shape: {array.shape}")

    def _save_video_frames(self, frames, output_path: str, fps: int = 16) -> str:
        """Save generated frames as an animated image with Pillow.

        Args:
            frames: Frames from diffusion pipeline (can be list, tensor, or nested structure)
            output_path: Desired output path. The suffix is normalized to .gif.
            fps: Frames per second

        Returns:
            Path to the animated GIF that was written.
        """
        try:
            if fps <= 0:
                raise ValueError("fps must be greater than 0")

            video_frames = self._normalize_video_frames(frames)
            pil_frames = [self._frame_to_pil(frame) for frame in video_frames]
            if not pil_frames:
                raise ValueError("No video frames were provided")

            gif_path = str(Path(output_path).with_suffix(".gif"))
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=max(1, int(1000 / fps)),
                loop=0,
            )
            logger.info(f"Saved generated animation to {gif_path} using Pillow")
            return gif_path
        except Exception as e:
            logger.error(f"Failed to save generated animation: {e}")
            logger.error(traceback.format_exc())
            raise

    def _categorize_media(self, media: List[str]) -> Tuple[List[str], List[str]]:
        """Categorize media files into images and videos

        Args:
            media: List of media file paths

        Returns:
            Tuple of (image_paths, video_paths)
        """
        image_exts, video_exts = self.get_supported_file_extensions()
        images = []
        videos = []

        for file_path in media:
            ext = Path(file_path).suffix.lower()
            if ext in image_exts:
                images.append(file_path)
            elif ext in video_exts:
                videos.append(file_path)
            else:
                logger.warning(f"Unknown media type, treating as image: {file_path}")
                images.append(file_path)

        return images, videos


def parse_job_arg(value):
    """Parse job argument - supports both JSON and Python dict literal formats"""
    # Try JSON first (from shell-escaped command)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    # Fall back to ast.literal_eval for backwards compatibility
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse job argument: {e}") from e


def main():
    """Main function for HuggingFace Inference Microservice Server"""
    parser = argparse.ArgumentParser(description="HuggingFace Inference Microservice Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--job", type=parse_job_arg)
    parser.add_argument("--docker_env_vars", type=parse_job_arg)
    parser.add_argument("--idle_timeout_minutes", type=int, default=30,
                        help="Minutes of inactivity before auto-deletion (default: 30)")
    parser.add_argument("--disable_auto_deletion", action="store_true",
                        help="Disable automatic deletion on idle timeout")

    args = parser.parse_args()
    logger.info(f"Job data: {args.job}")
    logger.info(f"Docker env vars: {args.docker_env_vars}")

    # Create HuggingFace model server using factory method
    server = HuggingFaceInferenceMicroserviceServer.create_from_tao_job(
        job_data=args.job,
        docker_env_vars=args.docker_env_vars,
        port=args.port
    )

    # Configure auto-deletion settings
    server.idle_timeout_minutes = args.idle_timeout_minutes
    server.auto_deletion_enabled = not args.disable_auto_deletion

    if server.auto_deletion_enabled:
        logger.info(f"Auto-deletion enabled with {server.idle_timeout_minutes} minute timeout")

    logger.info("HuggingFace inference microservice server created")

    # Start server immediately - initialization and model loading will happen in background
    server.start_server_immediate()


if __name__ == "__main__":
    main()
