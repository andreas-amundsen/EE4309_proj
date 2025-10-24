# -*- coding: utf-8 -*-
"""Shared configuration constants for Faster R-CNN style detectors."""

from __future__ import annotations

# Anchor configuration reused by both CNN and ViT backbones.
DEFAULT_ANCHOR_SIZES = ((32,), (64,), (128,), (256,), (512,))
DEFAULT_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(DEFAULT_ANCHOR_SIZES)

# Region Proposal Network defaults matching the reference notebook.
DEFAULT_RPN_PRE_NMS_TOP_N_TRAIN = 2000
DEFAULT_RPN_PRE_NMS_TOP_N_TEST = 1000
DEFAULT_RPN_POST_NMS_TOP_N_TRAIN = 2000
DEFAULT_RPN_POST_NMS_TOP_N_TEST = 1000
DEFAULT_RPN_NMS_THRESH = 0.7
DEFAULT_RPN_SCORE_THRESH = 0.0

# Box head defaults. Detectors can override these per-experiment.
DEFAULT_BOX_SCORE_THRESH = 0.1
DEFAULT_BOX_NMS_THRESH = 0.4
DEFAULT_BOX_DETECTIONS_PER_IMG = 300


__all__ = [
    "DEFAULT_ANCHOR_SIZES",
    "DEFAULT_ASPECT_RATIOS",
    "DEFAULT_RPN_PRE_NMS_TOP_N_TRAIN",
    "DEFAULT_RPN_PRE_NMS_TOP_N_TEST",
    "DEFAULT_RPN_POST_NMS_TOP_N_TRAIN",
    "DEFAULT_RPN_POST_NMS_TOP_N_TEST",
    "DEFAULT_RPN_NMS_THRESH",
    "DEFAULT_RPN_SCORE_THRESH",
    "DEFAULT_BOX_SCORE_THRESH",
    "DEFAULT_BOX_NMS_THRESH",
    "DEFAULT_BOX_DETECTIONS_PER_IMG",
]
