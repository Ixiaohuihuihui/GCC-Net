# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .detr import DETR

from .single_stage_enhance import SingleStageTwoBranchDetector

@DETECTORS.register_module()
class DeformableDETRFusion(DETR):

    def __init__(self, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)
