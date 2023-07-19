# -*- coding: utf-8 -*-
# @Time    : 2022/11/24 09:08
# @Author  : Linhui Dai
# @FileName: backup_single.py
# @Software: PyCharm
# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

from ..utils.crisscross_attention import CrissCrossAttention
from ..utils.cmat import CMAT
import torch.nn as nn
from ..utils.enhance_fusion import EnhanceFusion, edge_attention

@DETECTORS.register_module()
class SingleStageTwoBranchDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 cross_attention=False,
                 self_attention=False,
                 channel_gated=False,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageTwoBranchDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        channels = [256, 512, 1024, 2048]
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.cross_attention = cross_attention
        self.self_attention = self_attention
        self.channel_gated = channel_gated
        # if cross_attention:
        #     self.criss = CrissCrossAttention(256)
        self.fusion = EnhanceFusion


        if self_attention:
            pass

        if channel_gated:
            pass

    def extract_feat(self, img, img_retinex):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        y = self.backbone(img_retinex)
        if self.with_neck:
            x = self.neck(x)
            y = self.neck(y)
        return x, y

    def forward_dummy(self, img, img_retinex=None):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img, img_retinex)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      img_retinex,
                      gt_bboxes,
                      gt_labels,
                      coffient,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageTwoBranchDetector, self).forward_train(img, img_metas)
        x, y = self.extract_feat(img, img_retinex)

        if self.cross_attention:
            out1 = self.criss(x[0], y[0], coffient)
            # out1_t = self.criss(y[0], x[0])
            # out1 = self.beta * out1_t + out1
            out2 = self.criss(x[1], y[1], coffient)
            # out2_t = self.criss(y[1], x[1])
            # out2 = self.beta * out2_t + out2
            out3 = self.criss(x[2], y[2], coffient)
            # out3_t = self.criss(y[2], x[2])
            # out3 = self.beta * out3_t + out3
            out4 = self.criss(x[3], y[3], coffient)
            # out4_t = self.criss(y[3], x[3])
            # out4 = self.beta * out4_t + out4
            out5 = self.criss(x[4], y[4], coffient)
            # out5_t = self.criss(y[4], x[4])
            # out5 = self.beta * out5_t + out5

            # out1 = self.criss(x[0])
            # out2 = self.criss(x[1])
            # out3 = self.criss(x[2])
            # out4 = self.criss(x[3])
            # out5 = self.criss(x[4])

            tmp = tuple((out1, out2, out3, out4, out5))
            # tmp = tuple((x[0], x[1], x[2], out4, out5))
            losses = self.bbox_head.forward_train(tmp, img_metas, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore)
        else:
            losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, img_retinex, coffient, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat_x, feat_y = self.extract_feat(img, img_retinex)
        if self.cross_attention:
            out1 = self.criss(feat_x[0], feat_y[0], coffient)
            out2 = self.criss(feat_x[1], feat_y[1], coffient)
            out3 = self.criss(feat_x[2], feat_y[2], coffient)
            out4 = self.criss(feat_x[3], feat_y[3], coffient)
            out5 = self.criss(feat_x[4], feat_y[4], coffient)
            #
            # out1 = self.criss(feat_x[0])
            # out2 = self.criss(feat_x[1])
            # out3 = self.criss(feat_x[2])
            # out4 = self.criss(feat_x[3])
            # out5 = self.criss(feat_x[4])

            tmp = tuple((out1, out2, out3, out4, out5))
            results_list = self.bbox_head.simple_test(
                tmp, img_metas, rescale=rescale)
        else:
            results_list = self.bbox_head.simple_test(feat_x, img_metas, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def forward_test(self, imgs, img_metas, img_retinex, coffient, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], img_retinex[0], coffient[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, img_retinex, **kwargs)

    def aug_test(self, imgs, img_metas, img_retinex, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels
