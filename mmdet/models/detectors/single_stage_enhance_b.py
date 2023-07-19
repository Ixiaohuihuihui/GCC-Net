# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import math
import numpy as np

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import cv2
import time
from ..utils.color_correction_msr import MultiRetinex
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
# from .utils import MultiRetinex

class SA(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(SA, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = norm_layer(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True) #256
        out2 = self.conv2(out1)
        w, b = out2[:, :256, :, :], out2[:, 256:, :, :]

        return F.relu(w * out1 + b, inplace=True)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv0 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv0(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

""" fusion two level features """
class FAM(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, norm_layer=nn.BatchNorm2d):
        super(FAM, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256*2, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(256)

    def forward(self, left, down):
        down_mask = self.conv_d1(down)
        left_mask = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(left_mask * down_, inplace=True)
        else:
            z1 = F.relu(left_mask * down, inplace=True)

        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_mask * left, inplace=True)

        out = torch.cat((z1, z2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)

class Fusion(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(Fusion, self).__init__()
        self.conv0 = nn.Conv2d(in_channel*2, in_channel, 3, 1, 1)
        self.bn0 = norm_layer(in_channel)

    # def forward(self, x1, x2, alpha, beta):
    #     out1 = alpha * x1 + beta*(1.0 - alpha) * x2
    #     out2 = x1 * x2
    #     out = torch.cat((out1, out2), dim=1)
    #     out = F.relu(self.bn0(self.conv0(out)), inplace=True)

    def forward(self, x1, x2, alpha, beta):
        # out1 = alpha * x1 + beta * (1.0 - alpha) * x2
        out1 = x1 + x2
        out2 = x1 * x2
        out = torch.cat((out1, out2), dim=1)
        out = F.relu(self.bn0(self.conv0(out)), inplace=True)

        return out

class CrossAttention(nn.Module):
    def __init__(self, in_channel, ratio=8):
        super(CrossAttention, self).__init__()
        self.conv_query = nn.Conv2d(in_channel, in_channel//ratio, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channel, in_channel//ratio, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channel, in_channel, kernel_size=1)

    def forward(self, rgb, depth):
        bz, c, h, w = rgb.shape
        depth_q = self.conv_query(depth).view(bz, -1, h*w).permute(0, 2, 1)
        depth_k = self.conv_key(depth).view(bz, -1, h*w)
        mask = torch.bmm(depth_q, depth_k) #bz, hw, hw
        mask = torch.softmax(mask, dim=-1)
        rgb_v = self.conv_value(rgb).view(bz, c, -1)
        feat = torch.bmm(rgb_v, mask.permute(0, 2, 1)) # bz, c, hw
        feat = feat.view(bz, c, h, w)

        return feat

class CMAT(nn.Module):
    def __init__(self, in_channel, CA=True, ratio=8):
        super(CMAT, self).__init__()
        self.CA = CA

        # self.sa1 = SA(in_channel)
        # self.sa2 = SA(in_channel)
        self.sa = SpatialAttention()
        if self.CA:
            self.att1 = CrossAttention(in_channel, ratio=ratio)
            self.att2 = CrossAttention(in_channel, ratio=ratio)
        else:
            print("Warning: not use CrossAttention!")
            # self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)
            # self.conv3 = nn.Conv2d(256, 256, 3, 1, 1)
            self.conv2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
            self.conv3 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)

    def forward(self, rgb, depth, beta, gamma, gate):

        rgb = self.sa(rgb) * rgb
        depth = self.sa(depth) * rgb
        if self.CA:
            feat_1 = self.att1(rgb, depth)
            feat_2 = self.att2(depth, rgb)
        else:
            w1 = self.conv2(rgb)
            w2 = self.conv3(depth)
            feat_1 = F.relu(w2*rgb, inplace=True)
            feat_2 = F.relu(w1*depth, inplace=True)

        # out1 = rgb + gate * beta * feat_1
        # out2 = depth + (1.0-gate) * gamma * feat_2
        out1 = rgb + beta * feat_1
        out2 = depth + gamma * feat_2

        return out1, out2

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        m_batchsize, _, height, width = x.size()

        proj_query_x = self.query_conv(x)
        proj_query_H = proj_query_x.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query_x.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key_x = self.key_conv(x)
        proj_key_H = proj_key_x.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key_x.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value_y = self.value_conv(y)
        proj_value_H = proj_value_y.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value_y.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x

@DETECTORS.register_module()
class SingleStageTwoBranchDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 cross_attention=False,
                 self_attention=False,
                 channel_gated=False,
                 init_cfg=None):
        super(SingleStageTwoBranchDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        channels = [256, 512, 1024, 2048]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # self.multi_retinex = MultiRetinex(256)
        # self.gap1 = nn.AdaptiveAvgPool2d(1)
        # self.gap2 = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Sequential(
        #            nn.Linear(channels[-1]*2, 512),
        #         ##nn.Dropout(p=0.3),
        #            nn.ReLU(True),
        #            nn.Linear(512, 256+1),
        #            nn.Sigmoid(),
        #            )
        #
        # # fusion modules
        # self.cmat5 = CMAT(channels[3], True, ratio=8)
        # self.cmat4 = CMAT(channels[2], True, ratio=8)
        # self.cmat3 = CMAT(channels[1], False, ratio=8)
        # self.cmat2 = CMAT(channels[0], False, ratio=8)
        # self.fusion2 = Fusion(channels[0])
        # self.fusion3 = Fusion(channels[1])
        # self.fusion4 = Fusion(channels[2])
        # self.fusion5 = Fusion(channels[3])


        # self.conv = convlayer(self.base_channels, self.channel, init_cfg=None)

    def extract_feat(self, img, img_retinex):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        y = self.backbone(img_retinex)

        # -------------Detailed Image----------
        # detailed_img = []
        # for i in range(len(img)):
        #     tmp = img[i].detach().cpu()
        #     # gaussian filter to generate detail image
        #     (B, G, R) = torch.split(tmp, 1)
        #     b_gaussian = cv2.GaussianBlur(np.array(B.squeeze()), (3, 5), cv2.BORDER_DEFAULT)
        #     g_gaussian = cv2.GaussianBlur(np.array(G.squeeze()), (3, 5), cv2.BORDER_DEFAULT)
        #     r_gaussian = cv2.GaussianBlur(np.array(R.squeeze()), (3, 5), cv2.BORDER_DEFAULT)
        #     detailed_b = B - 0.5 * b_gaussian
        #     detailed_g = G - 0.5 * g_gaussian
        #     detailed_r = R - 0.5 * r_gaussian
        #
        #     tmp_detail = torch.cat((detailed_b, detailed_g, detailed_r), 0)
        #     detailed_img.append(tmp_detail)
        #     # attenuation map
        #     # a_b = 255 - B**1.2
        #     # a_g = 255 - G**1.2
        #     # a_r = 255 - R**1.2
        #     # tmp_atten = torch.cat((a_b, a_g, a_r), 0)
        #
        #
        #     # restored_image_color_enhance
        # detailed_img = torch.stack(detailed_img, 0).to(img.device)
        # # z = self.backbone(detailed_img)
        # -------------Detailed Image----------

        # -------------Self Attention----------
        # x[0] = self.ca(x[0])*x[0]


        # -------------Self Attention----------

        # -------------Cross Attention----------
        # bz = x[0].shape[0]
        # rgb_gap = self.gap1(x[3])
        # rgb_gap = rgb_gap.view(bz, -1)
        #
        # retinex_gap = self.gap2(y[3])
        # retinex_gap = retinex_gap.view(bz, -1)
        #
        # feat = torch.cat((rgb_gap, retinex_gap), dim=1)
        # feat = self.fc(feat)
        # gate = feat[:, -1].view(bz, 1, 1, 1)
        #
        # # alpha = feat[:, :256]
        # # alpha = alpha.view(bz, 256, 1, 1)
        # #----
        # alpha = 0.5
        # #
        # x5, y5 = self.cmat5(x[3], y[3], 1, 1, gate)
        # x4, y4 = self.cmat4(x[2], y[2], 1, 1, gate)
        # x3, y3 = self.cmat3(x[1], y[1], 1, 1, gate)
        # x2, y2 = self.cmat2(x[0], y[0], 1, 1, gate)
        #
        # out2 = self.fusion2(x2, y2, alpha, gate)
        # out3 = self.fusion3(x3, y3, alpha, gate)
        # out4 = self.fusion4(x4, y4, alpha, gate)
        # out5 = self.fusion5(x5, y5, alpha, gate)
        # -------------Cross Attention----------

        # salience map
        # salience map
        # red_channel_concat

        # dark_channel_concat

        # -------------Feature Fusion----------
        # tmp = tuple(map(sum, zip(x, y, z)))
        # tmp = tuple((out2, out3, out4, out5))
        # -------------Feature Fusion----------

        # tmp = tuple(map(sum, zip(x, y)))
        # out2 = self.criss2(y[0], x[0])
        # out3 = self.criss3(y[1], x[1])
        # out4 = self.criss4(y[2], x[2])
        # out5 = self.criss5(y[3], x[3])
        # tmp = tuple((out2, out3, out4, out5))

        if self.with_neck:
            x = self.neck(x)
            y = self.neck(y)
        return x, y

    def forward_dummy(self, img, img_retinex):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`d
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
        x = self.extract_feat(img, img_retinex)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def forward_test(self, imgs, img_metas, img_retinex, **kwargs):
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
            return self.simple_test(imgs[0], img_metas[0], img_retinex[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, img_retinex, **kwargs)

    def simple_test(self, img, img_metas, img_retinex, rescale=False):
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
        feat = self.extract_feat(img, img_retinex)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
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

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
