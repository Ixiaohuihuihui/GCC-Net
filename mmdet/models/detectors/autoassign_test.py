# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from .single_stage_enhance import SingleStageTwoBranchDetector


@DETECTORS.register_module()
class AutoAssign_fusion(SingleStageTwoBranchDetector):
    """Implementation of `AutoAssign: Differentiable Label Assignment for Dense
    Object Detection <https://arxiv.org/abs/2007.03496>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(AutoAssign_fusion, self).__init__(backbone, neck,
                                                bbox_head, train_cfg,
                                                test_cfg, pretrained)
