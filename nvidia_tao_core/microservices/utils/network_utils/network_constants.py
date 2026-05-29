# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants unifying individual networks."""

epochs_mapper = {
    "action_recognition": "train.num_epochs",
    "analytics": "",
    "annotations": "",
    "augmentation": "",
    "auto_label": "",
    "classification_pyt": "train.num_epochs",
    "deformable_detr": "train.num_epochs",
    "dino": "train.num_epochs",
    "mal": "train.num_epochs",
    "ml_recog": "train.num_epochs",
    "ocdnet": "train.num_epochs",
    "ocrnet": "train.num_epochs",
    "optical_inspection": "train.num_epochs",
    "pointpillars": "train.num_epochs",
    "pose_classification": "train.num_epochs",
    "re_identification": "train.num_epochs",
    "segformer": "train.num_epochs",
}

backbone_mapper = {
    "action_recognition": "model.backbone",
    "analytics": "",
    "annotations": "",
    "augmentation": "",
    "auto_label": "",
    "classification_pyt": "model.backbone.type",
    "deformable_detr": "model.backbone",
    "dino": "model.backbone",
    "mal": "model.arch",
    "ml_recog": "model.backbone",
    "ocdnet": "model.backbone",
    "ocrnet": "model.backbone",
    "optical_inspection": "model.backbone",
    "pointpillars": "model.backbone_2d.name",
    "pose_classification": "",
    "re_identification": "model.backbone",
    "segformer": "model.backbone.type",
}

image_size_mapper = {
    "action_recognition": "model.input_height,model.input_width",
    "analytics": "",
    "annotations": "",
    "augmentation": "",
    "auto_label": "",
    "classification_pyt": "",
    "deformable_detr": "",
    "dino": "",
    "mal": "",
    "ml_recog": "model.input_height,model.input_width",
    "ocdnet": "",
    "ocrnet": "model.input_height,model.input_width",
    "optical_inspection": "dataset.image_height,dataset.image_width",
    "pointpillars": "",
    "pose_classification": "",
    "re_identification": "model.input_height,model.input_width",
    "segformer": "dataset.segment.img_size",
}

node_mapper = {
    # cosmos-rl nodes are computed dynamically via _get_cosmos_rl_num_nodes
    # based on total GPUs (policy + rollout) / gpus_per_node
}

gpu_mapper = {
    "cosmos-rl": "policy.parallelism.dp_shard_size",
    "action_recognition": "",
    "analytics": "",
    "annotations": "",
    "augmentation": "",
    "auto_label": "",
    "classification_pyt": "",
    "deformable_detr": "",
    "dino": "",
    "mal": "",
    "ml_recog": "",
    "ocdnet": "",
    "ocrnet": "",
    "optical_inspection": "",
    "pointpillars": "",
    "pose_classification": "",
    "re_identification": "",
    "segformer": "",
}

# Include your network if it has spec fields to load full network as PTM and loading backbone portion alone
ptm_mapper = {
    "backbone": {
        "classification_pyt": "model.backbone.pretrained_backbone_path",
        "segformer": "model.backbone.pretrained_backbone_path",
        "visual_changenet_classify": "model.backbone.pretrained_backbone_path",
        "visual_changenet_segment": "model.backbone.pretrained_backbone_path",
        "dino": "model.pretrained_backbone_path",
        "grounding_dino": "model.pretrained_backbone_path",
        "mask_grounding_dino": "model.pretrained_backbone_path",
    },
    "end_to_end": {
        "classification_pyt": "train.pretrained_model_path",
        "visual_changenet_classify": "train.pretrained_model_path",
        "visual_changenet_segment": "train.pretrained_model_path",
        "segformer": "train.pretrained_model_path",
        "dino": "train.pretrained_model_path",
    },
    "default": {
        "mask2former": "model.backbone.pretrained_weights",
    }

}
