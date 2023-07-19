# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from .single_stage_enhance import SingleStageTwoBranchDetector


@DETECTORS.register_module()
class AutoAssign(SingleStageDetector):
    """Implementation of `AutoAssign: Differentiable Label Assignment for Dense
    Object Detection <https://arxiv.org/abs/2007.03496>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 # cross_attention=False,
                 # self_attention=False,
                 # channel_gated=False,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(AutoAssign, self).__init__(backbone, neck,
                                         # cross_attention, self_attention,
                                         # channel_gated,
                                         bbox_head, train_cfg,
                                         test_cfg, pretrained)
