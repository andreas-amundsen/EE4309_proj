# -*- coding: utf-8 -*-
"""Faster R-CNN detector assembly and ResNet-50 preset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.ops import MultiScaleRoIAlign

from .backbones import RESNET_FPN_FEATMAP_NAMES, ResNetBackboneConfig, build_resnet50_fpn_backbone
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

__all__ = [
    "BackboneBundle",
    "DetectorConfig",
    "build_faster_rcnn",
    "make_standard_rpn_head",
    "make_two_conv_rpn_head",
    "get_resnet50_fasterrcnn_model",
    "ResNetBackboneConfig",
]


@dataclass
class BackboneBundle:
    """Container describing a backbone and its feature metadata."""

    body: nn.Module
    featmap_names: Sequence[str]
    out_channels: int


@dataclass
class DetectorConfig:
    """Configuration knobs for Faster R-CNN assembly."""

    box_score_thresh: float = DEFAULT_BOX_SCORE_THRESH
    box_nms_thresh: float = DEFAULT_BOX_NMS_THRESH
    detections_per_img: int = DEFAULT_BOX_DETECTIONS_PER_IMG
    rpn_pre_nms_top_n_train: int = DEFAULT_RPN_PRE_NMS_TOP_N_TRAIN
    rpn_pre_nms_top_n_test: int = DEFAULT_RPN_PRE_NMS_TOP_N_TEST
    rpn_post_nms_top_n_train: int = DEFAULT_RPN_POST_NMS_TOP_N_TRAIN
    rpn_post_nms_top_n_test: int = DEFAULT_RPN_POST_NMS_TOP_N_TEST
    rpn_nms_thresh: float = DEFAULT_RPN_NMS_THRESH
    rpn_score_thresh: float = DEFAULT_RPN_SCORE_THRESH

    def to_kwargs(self) -> dict:
        return {
            "box_score_thresh": self.box_score_thresh,
            "box_nms_thresh": self.box_nms_thresh,
            "box_detections_per_img": self.detections_per_img,
            "rpn_pre_nms_top_n_train": self.rpn_pre_nms_top_n_train,
            "rpn_pre_nms_top_n_test": self.rpn_pre_nms_top_n_test,
            "rpn_post_nms_top_n_train": self.rpn_post_nms_top_n_train,
            "rpn_post_nms_top_n_test": self.rpn_post_nms_top_n_test,
            "rpn_nms_thresh": self.rpn_nms_thresh,
            "rpn_score_thresh": self.rpn_score_thresh,
        }


def build_faster_rcnn(
    *,
    backbone: BackboneBundle,
    anchor_generator: AnchorGenerator,
    rpn_head_factory: Callable[[int], nn.Module],
    roi_pool: MultiScaleRoIAlign,
    num_classes: int,
    config: Optional[DetectorConfig] = None,
) -> FasterRCNN:
    """Assemble a Faster R-CNN detector from modular components."""

    # ===== STUDENT TODO: Implement Faster R-CNN assembly =====
    # Hint: Build the complete Faster R-CNN detector:
    # 1. Get configuration (use default if none provided)
    # 2. Create RPN head using the factory with correct number of anchors
    # 3. Assemble FasterRCNN with all components:
    #    - backbone.body as the feature extractor
    #    - anchor_generator for RPN proposals
    #    - rpn_head for region proposal network
    #    - roi_pool for feature extraction from proposals
    #    - config parameters for detection thresholds
    # 4. NB! not done explicitly: Replace the box predictor head for the correct number of classes
    # 5. Return the assembled model

    cfg = config or DetectorConfig()

    # override thresholds to guarantee that bmodel predicts something
    cfg.box_score_thresh=0.0
    cfg.box_nms_thresh=0.9
    cfg.detections_per_img=300
    cfg.rpn_nms_thresh=0.9
    cfg.rpn_score_thresh=0.0
    #space

    # RPN head
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    rpn_head = rpn_head_factory(num_anchors)

    # Build the detector
    model = FasterRCNN(
        backbone=backbone.body,
        num_classes=num_classes,  # we'll set the box predictor below
        rpn_anchor_generator=anchor_generator,
        rpn_head=rpn_head,
        box_roi_pool=roi_pool,
        # RPN config
        rpn_pre_nms_top_n_train=cfg.rpn_pre_nms_top_n_train,
        rpn_pre_nms_top_n_test=cfg.rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_train=cfg.rpn_post_nms_top_n_train,
        rpn_post_nms_top_n_test=cfg.rpn_post_nms_top_n_test,
        rpn_nms_thresh=cfg.rpn_nms_thresh,
        rpn_score_thresh=cfg.rpn_score_thresh,
        # ROI heads / box config
        box_score_thresh=cfg.box_score_thresh,
        box_nms_thresh=cfg.box_nms_thresh,
        box_detections_per_img=cfg.detections_per_img,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
    raise NotImplementedError("build_faster_rcnn() not implemented")
    # =========================================================


def make_standard_rpn_head(in_channels: int) -> Callable[[int], nn.Module]:
    """Factory for the torchvision standard RPN head."""

    def factory(num_anchors: int) -> nn.Module:
        return RPNHead(in_channels, num_anchors)

    return factory


class TwoConvRPNHead(nn.Module):
    """RPN head with stacked conv layers, used for ResNet baselines."""

    def __init__(self, in_channels: int, num_anchors: int, conv_depth: int = 2) -> None:
        super().__init__()
        layers = []
        for _ in range(conv_depth):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, features):
        logits = []
        bbox_reg = []
        for feature in features:
            t = self.conv(feature)
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def make_two_conv_rpn_head(in_channels: int, conv_depth: int = 2) -> Callable[[int], nn.Module]:
    """Factory for the custom two-convolution RPN head."""

    def factory(num_anchors: int) -> nn.Module:
        return TwoConvRPNHead(in_channels, num_anchors, conv_depth=conv_depth)

    return factory


def get_resnet50_fasterrcnn_model(
    num_classes: int,
    *,
    backbone_config: Optional[ResNetBackboneConfig] = None,
    box_score_thresh: float = DEFAULT_BOX_SCORE_THRESH,
    box_nms_thresh: float = DEFAULT_BOX_NMS_THRESH,
    detections_per_img: int = DEFAULT_BOX_DETECTIONS_PER_IMG,
    rpn_pre_nms_top_n_train: int = DEFAULT_RPN_PRE_NMS_TOP_N_TRAIN,
    rpn_pre_nms_top_n_test: int = DEFAULT_RPN_PRE_NMS_TOP_N_TEST,
    rpn_post_nms_top_n_train: int = DEFAULT_RPN_POST_NMS_TOP_N_TRAIN,
    rpn_post_nms_top_n_test: int = DEFAULT_RPN_POST_NMS_TOP_N_TEST,
    rpn_nms_thresh: float = DEFAULT_RPN_NMS_THRESH,
    rpn_score_thresh: float = DEFAULT_RPN_SCORE_THRESH,
) -> FasterRCNN:
    """Construct a ResNet-50 + FPN Faster R-CNN detector."""

    backbone_module = build_resnet50_fpn_backbone(backbone_config)
    backbone = BackboneBundle(
        body=backbone_module,
        featmap_names=RESNET_FPN_FEATMAP_NAMES,
        out_channels=backbone_module.out_channels,
    )

    anchor_generator = AnchorGenerator(sizes=DEFAULT_ANCHOR_SIZES, aspect_ratios=DEFAULT_ASPECT_RATIOS)
    rpn_head_factory = make_two_conv_rpn_head(backbone.out_channels, conv_depth=2)
    roi_pool = MultiScaleRoIAlign(featmap_names=backbone.featmap_names, output_size=7, sampling_ratio=2)

    detector_cfg = DetectorConfig(
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
        config=detector_cfg,
    )
