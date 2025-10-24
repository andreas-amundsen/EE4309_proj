# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from .backbones.vit import ViTBackboneConfig, build_vit_fpn_backbone
from .detection_config import (
    DEFAULT_ANCHOR_SIZES,
    DEFAULT_ASPECT_RATIOS,
    DEFAULT_BOX_DETECTIONS_PER_IMG,
    DEFAULT_BOX_NMS_THRESH,
    DEFAULT_BOX_SCORE_THRESH,
    DEFAULT_RPN_NMS_THRESH,
    DEFAULT_RPN_POST_NMS_TOP_N_TEST,
    DEFAULT_RPN_POST_NMS_TOP_N_TRAIN,
    DEFAULT_RPN_PRE_NMS_TOP_N_TEST,
    DEFAULT_RPN_PRE_NMS_TOP_N_TRAIN,
    DEFAULT_RPN_SCORE_THRESH,
)
from .faster_rcnn import BackboneBundle, DetectorConfig, build_faster_rcnn, make_standard_rpn_head


__all__ = ["get_vit_fasterrcnn_model", "ViTBackboneConfig"]


def get_vit_fasterrcnn_model(
    num_classes: int,
    *,
    backbone_config: Optional[ViTBackboneConfig] = None,
    box_score_thresh: float = DEFAULT_BOX_SCORE_THRESH,
    box_nms_thresh: float = DEFAULT_BOX_NMS_THRESH,
    detections_per_img: int = DEFAULT_BOX_DETECTIONS_PER_IMG,
    rpn_pre_nms_top_n_train: int = DEFAULT_RPN_PRE_NMS_TOP_N_TRAIN,
    rpn_pre_nms_top_n_test: int = DEFAULT_RPN_PRE_NMS_TOP_N_TEST,
    rpn_post_nms_top_n_train: int = DEFAULT_RPN_POST_NMS_TOP_N_TRAIN,
    rpn_post_nms_top_n_test: int = DEFAULT_RPN_POST_NMS_TOP_N_TEST,
    rpn_nms_thresh: float = DEFAULT_RPN_NMS_THRESH,
    rpn_score_thresh: float = DEFAULT_RPN_SCORE_THRESH,
):
    """Construct a ViT-based Faster R-CNN detector."""
    # ===== STUDENT TODO: Implement ViT Faster R-CNN construction =====
    # Hint: Build complete ViT-based detector:
    # 1. Build ViT+FPN backbone using build_vit_fpn_backbone
    # 2. Wrap backbone in BackboneBundle with feature names and channels
    # 3. Create anchor generator, RPN head factory, and ROI pooler
    # 4. Configure detection parameters in DetectorConfig
    # 5. Assemble final detector using build_faster_rcnn
    # This combines ViT features with Faster R-CNN detection framework

    backbone_config = backbone_config or ViTBackboneConfig()

    backbone_module = build_vit_fpn_backbone(config=backbone_config)
    
    backbone = BackboneBundle(
        body=backbone_module,
        featmap_names=backbone_module._out_features,
        out_channels=backbone_module.out_channels,
    )

    anchor_generator = AnchorGenerator(sizes=DEFAULT_ANCHOR_SIZES, aspect_ratios=DEFAULT_ASPECT_RATIOS)
    rpn_head_factory = make_standard_rpn_head(in_channels=backbone.out_channels)
    roi_pool = MultiScaleRoIAlign(featmap_names=backbone.featmap_names, output_size=7, sampling_ratio=2)

    detector_config = DetectorConfig(
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        detections_per_img=detections_per_img,
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
        rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        rpn_nms_thresh=rpn_nms_thresh,
        rpn_score_thresh=rpn_score_thresh,
    )

    return build_faster_rcnn(
        backbone=backbone,
        anchor_generator=anchor_generator, 
        rpn_head_factory=rpn_head_factory, 
        roi_pool=roi_pool, 
        num_classes=num_classes,
        config=detector_config,
        
    )
    # ================================================================
