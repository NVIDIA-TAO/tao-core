# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Defines a dictionary mapping docker image to an internal tag"""
import os

DOCKER_IMAGE_MAPPER = {
    "TAO_PYTORCH": os.getenv('IMAGE_TAO_PYTORCH', default='nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt'),
    "TAO_DEPLOY": os.getenv('IMAGE_TAO_DEPLOY', default='nvcr.io/nvidia/tao/tao-toolkit:6.0.0-deploy'),
    "": os.getenv('IMAGE_DEFAULT', default='nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt'),  # Default
    "API": os.getenv('IMAGE_API', default='nvcr.io/nvidia/tao/tao-toolkit:6.0.0-api'),
    "TAO_DS": os.getenv('IMAGE_TAO_DS', default='nvcr.io/nvidia/tao/tao-toolkit:6.0.0-data-services'),
    "VILA": os.getenv('IMAGE_VILA', default='nvcr.io/nvidia/tao/tao-toolkit:6.0.0-vila'),
    "COSMOS_RL": os.getenv('IMAGE_COSMOS_RL', default='nvcr.io/ea-tlt/tao_ea/cosmos_rl:latest'),
    "HUGGINGFACE": os.getenv(
        'IMAGE_HUGGINGFACE',
        os.getenv('IMAGE_TAO_PYTORCH', 'nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt')
    ),
    "tensorboard": os.getenv('IMAGE_TAO_PYTORCH', default='nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt')
}


DOCKER_IMAGE_VERSION = {  # (Release tao version for DNN framework, Overriden version for this model)
    "action_recognition": ("6.25.7", "6.0.0"),
    "depth_net_mono": ("6.25.10", "6.25.10"),
    "depth_net_stereo": ("6.25.10", "6.25.10"),
    "bevfusion": ("6.25.7", "6.0.0"),
    "centerpose": ("6.25.7", "6.0.0"),
    "classification_pyt": ("6.25.7", "6.0.0"),
    "deformable_detr": ("6.25.7", "6.0.0"),
    "dino": ("6.25.7", "6.0.0"),
    "image": ("6.25.7", "6.0.0"),
    "mae": ("6.25.7", "6.0.0"),
    "mal": ("6.25.7", "6.0.0"),
    "nvdinov2": ("6.25.7", "6.0.0"),
    "ml_recog": ("6.25.7", "6.0.0"),
    "object_detection": ("6.25.7", "6.0.0"),
    "ocdnet": ("6.25.7", "6.0.0"),
    "ocrnet": ("6.25.7", "6.0.0"),
    "optical_inspection": ("6.25.7", "6.0.0"),
    "pointpillars": ("6.25.7", "6.0.0"),
    "pose_classification": ("6.25.7", "6.0.0"),
    "re_identification": ("6.25.7", "6.0.0"),
    "rtdetr": ("6.25.7", "6.0.0"),
    "segformer": ("6.25.7", "6.0.0"),
    "sparse4d": ("6.25.7", "6.25.7"),
    "stylegan_xl": ("6.25.7", "6.0.0"),
    "visual_changenet_classify": ("6.25.7", "6.0.0"),
    "visual_changenet_segment": ("6.25.7", "6.0.0"),
}
