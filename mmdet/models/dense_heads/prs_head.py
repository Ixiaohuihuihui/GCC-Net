import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv.runner import force_fp32

from mmdet.core import (build_prior_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from ..builder import HEADS, build_loss
from ..losses import smooth_l1_loss
from .anchor_head import AnchorHead
from mmcv.ops import DeformConv2d
from mmcv.runner import BaseModule
from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin

def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets

class AdaptiveConv(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=3,
                 groups=1,
                 deform_groups=1,
                 bias=False,
                 type='offset',
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv'))):
        super(AdaptiveConv, self).__init__(init_cfg)
        assert type in ['offset', 'dilation']
        self.adapt_type = type

        assert kernel_size == 3, 'Adaptive conv only supports kernels 3'
        if self.adapt_type == 'offset':
            assert stride == 1 and padding == 1, \
                'Adaptive conv offset mode only supports padding: {1}, ' \
                f'stride: {1}, groups: {1}'
            self.conv = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                groups=groups,
                deform_groups=deform_groups,
                bias=bias)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=dilation,
                dilation=dilation)

    def forward(self, x, offset, num_anchors):
        """Forward function."""
        if self.adapt_type == 'offset':
            N, _, H, W = x.shape
            assert offset is not None
            # reshape [N, NA, 18] to (N, 18, H, W)
            offset = offset.reshape(N, H, W, -1)
            offset = offset.permute(0, 3, 1, 2)
            offset = offset.contiguous()
            x = self.conv(x, offset)
        else:
            assert offset is None
            x = self.conv(x)
        return x

# TODO: add loss evaluator for SSD
@HEADS.register_module()
class PRSHead(AnchorHead):

    def __init__(self,
                 num_classes=80,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 anchor_generator=dict(
                     type='SSDAnchorGenerator',
                     scale_major=False,
                     input_size=300,
                     strides=[8, 16, 32, 64, 100, 300],
                     ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                     basesize_ratio_range=(0.1, 0.9)),
                 background_label=None,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[.0, .0, .0, .0],
                     target_stds=[1.0, 1.0, 1.0, 1.0],
                 ),
                 reg_decoded_bbox=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='cls_convs_refine',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        super(AnchorHead, self).__init__(init_cfg, **kwargs)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes + 1  # add background class
        self.prior_generator = build_prior_generator(anchor_generator)
        # self.num_anchors = self.prior_generator.num_base_anchors
        self.num_base_priors = self.prior_generator.num_base_priors
        num_anchors = self.num_base_priors
        self.anchor_strides = anchor_generator['strides']
        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
            cls_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i],
                    kernel_size=3,
                    padding=1))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)

        self.background_label = (
            num_classes if background_label is None else background_label)
        # background_label should be either 0 or num_classes
        assert (self.background_label == 0
                or self.background_label == num_classes)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # set sampling=False for archor_target
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

        # dcn
        dcn = []
        reg_convs_refine = []
        cls_convs_refine = []
        for i in range(len(in_channels)):
            dcn.append(AdaptiveConv(num_anchors[i]*in_channels[i], in_channels[i],deform_groups=num_anchors[i]))
            reg_convs_refine.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
            cls_convs_refine.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * self.cls_out_channels,
                    kernel_size=3,
                    padding=1))
        self.dcn = nn.ModuleList(dcn)
        self.reg_convs_refine = nn.ModuleList(reg_convs_refine)
        self.cls_convs_refine = nn.ModuleList(cls_convs_refine)
        self.relu = nn.ReLU(inplace=True)
        # self.BCE = build_loss(loss_cls_pre)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        fg_scores, bbox_preds = self(x)

        featmap_sizes = [featmap.size()[-2:] for featmap in fg_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = fg_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        if gt_labels is None:
            loss_inputs = (anchor_list, valid_flag_list, fg_scores, bbox_preds, gt_bboxes, img_metas)
        else:
            loss_inputs = (anchor_list, valid_flag_list, fg_scores, bbox_preds, gt_bboxes, gt_labels, img_metas)
        losses = self.loss_pre(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        new_feats = []

        # pre-processing
        for i in range(len(fg_scores)):
            score = fg_scores[i]
            s,_ = torch.max(score, dim = 1)
            s = s.unsqueeze(1)
            s = torch.sigmoid(s)
            new_feats.append(s*x[i]+x[i])
        # new_feats = x

        anchor_list_refine = self.refine_bboxes(anchor_list, bbox_preds, img_metas)
        offset_list = self.anchor_offset(anchor_list_refine, self.anchor_strides, featmap_sizes)

        cls_scores, bbox_preds_refine = self.forward_post(new_feats, offset_list)
        if gt_labels is None:
            loss_inputs = (anchor_list_refine, valid_flag_list, cls_scores, bbox_preds_refine, gt_bboxes, img_metas)
        else:
            loss_inputs = (anchor_list_refine, valid_flag_list, cls_scores, bbox_preds_refine, gt_bboxes, gt_labels, img_metas)
        losses_post = self.loss_post(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        losses.update(losses_post)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(anchor_list_refine, cls_scores, bbox_preds_refine, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []

        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))

        return cls_scores, bbox_preds

    def forward_post(self, feats, offset_list):
        cls_scores = []
        bbox_preds = []
        for i in range(len(feats)):
            x = feats[i]
            shape = list(x.shape)
            x = x.unsqueeze(1).expand((shape[0], self.num_base_priors[i],shape[1],shape[2],shape[3]))
            shape[1] = shape[1] * self.num_base_priors[i]
            x = x.reshape(shape)
            offset = offset_list[i]
            feat = self.relu(self.dcn[i](x, offset, self.num_base_priors[i]))
            cls_score = self.cls_convs_refine[i](feat)
            bbox_pred = self.reg_convs_refine[i](feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return cls_scores, bbox_preds

    def loss_single_post(self, cls_score, bbox_pred, anchor, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) &
                    (labels < self.background_label)).nonzero().reshape(-1)
        neg_inds = (labels == self.background_label).nonzero().view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        if self.reg_decoded_bbox:
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights*0.5,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples)

        return loss_cls[None], loss_bbox

    def loss_single_pre(self, fg_scores, bbox_pred, anchor, fg_labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        fg_loss = (F.binary_cross_entropy(
            torch.sigmoid(fg_scores), fg_labels, reduction='none') * label_weights).mean()


        if self.reg_decoded_bbox:
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights*0.5,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples)


        return fg_loss, loss_bbox

    def loss_pre(self,
             anchor_list,
             valid_flag_list,
             fg_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in fg_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        # device = fg_scores[0].device

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))

        # check NaN and Inf
        assert torch.isfinite(all_bbox_preds).all().item(), \
            'bbox predications become infinite or NaN!'

        fg_labels = all_labels.clone().float()
        fg_labels[fg_labels != self.num_classes] = 1
        fg_labels[fg_labels == self.num_classes] = 0

        all_fg_scores = torch.cat([
            f.permute(0, 2, 3, 1).reshape(
                num_images, -1) for f in fg_scores
        ], 1)


        fg_losses, losses_bbox = multi_apply(
            self.loss_single_pre,
            all_fg_scores,
            all_bbox_preds,
            all_anchors,
            fg_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos)
        return dict(losses_fg=fg_losses, loss_bbox=losses_bbox)

    def loss_post(self,
             anchor_list,
             valid_flag_list,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        # device = cls_scores[0].device

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))

        # check NaN and Inf
        assert torch.isfinite(all_cls_scores).all().item(), \
            'classification scores become infinite or NaN!'
        assert torch.isfinite(all_bbox_preds).all().item(), \
            'bbox predications become infinite or NaN!'

        fg_labels = all_labels.clone().float()
        fg_labels[fg_labels != self.num_classes] = 1
        fg_labels[fg_labels == self.num_classes] = 0


        losses_cls, losses_bbox_ref = multi_apply(
            self.loss_single_post,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos)
        return dict(loss_cls=losses_cls, losses_bbox_ref=losses_bbox_ref)

    def anchor_offset(self, anchor_list, anchor_strides, featmap_sizes):
        def _shape_offset(anchors, stride, ks=3, dilation=1):
            # currently support kernel_size=3 and dilation=1
            assert ks == 3 and dilation == 1
            pad = (ks - 1) // 2
            idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
            yy, xx = torch.meshgrid(idx, idx)  # return order matters
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)
            w = (anchors[:, 2] - anchors[:, 0]) / stride
            h = (anchors[:, 3] - anchors[:, 1]) / stride
            w = w / (ks - 1) - dilation
            h = h / (ks - 1) - dilation
            offset_x = w[:, None] * xx  # (NA, ks**2)
            offset_y = h[:, None] * yy  # (NA, ks**2)
            return offset_x, offset_y

        def _ctr_offset(anchors, stride, featmap_size, num_anchors):
            feat_h, feat_w = featmap_size

            x = (anchors[:, 0] + anchors[:, 2]) * 0.5
            y = (anchors[:, 1] + anchors[:, 3]) * 0.5
            # compute centers on feature map
            x = x / stride
            y = y / stride
            # compute predefine centers
            xx = torch.arange(0, feat_w, device=anchors.device)
            yy = torch.arange(0, feat_h, device=anchors.device)
            yy, xx = torch.meshgrid(yy, xx)
            xx = xx.reshape(-1).type_as(x)
            yy = yy.reshape(-1).type_as(y)

            xx = xx.unsqueeze(1).expand(xx.shape+(num_anchors,)).reshape(-1)
            yy = yy.unsqueeze(1).expand(yy.shape+(num_anchors,)).reshape(-1)

            offset_x = x - xx  # (NA, )
            offset_y = y - yy  # (NA, )
            return offset_x, offset_y

        num_imgs = len(anchor_list)
        num_lvls = len(anchor_list[0])
        dtype = anchor_list[0][0].dtype
        device = anchor_list[0][0].device
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        offset_list = []
        for i in range(num_imgs):
            mlvl_offset = []
            for lvl in range(num_lvls):
                c_offset_x, c_offset_y = _ctr_offset(anchor_list[i][lvl],
                                                     anchor_strides[lvl],
                                                     featmap_sizes[lvl],
                                                     self.num_base_priors[lvl])
                s_offset_x, s_offset_y = _shape_offset(anchor_list[i][lvl],
                                                       anchor_strides[lvl])

                # offset = ctr_offset + shape_offset
                offset_x = s_offset_x + c_offset_x[:, None]
                offset_y = s_offset_y + c_offset_y[:, None]

                # offset order (y0, x0, y1, x2, .., y8, x8, y9, x9)
                offset = torch.stack([offset_y, offset_x], dim=-1)
                offset = offset.reshape(offset.size(0), -1)  # [NA, 2*ks**2]
                mlvl_offset.append(offset)
            offset_list.append(torch.cat(mlvl_offset))  # [totalNA, 2*ks**2]
        offset_list = images_to_levels(offset_list, num_level_anchors)
        return offset_list

    def refine_bboxes(self, anchor_list, bbox_preds, img_metas):
        """Refine bboxes through stages."""
        num_levels = len(bbox_preds)
        new_anchor_list = []
        for img_id in range(len(img_metas)):
            mlvl_anchors = []
            for i in range(num_levels):
                bbox_pred = bbox_preds[i][img_id].detach()
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
                img_shape = img_metas[img_id]['img_shape']
                bboxes = self.bbox_coder.decode(anchor_list[img_id][i],
                                                bbox_pred, img_shape)
                mlvl_anchors.append(bboxes)
            new_anchor_list.append(mlvl_anchors)
        return new_anchor_list

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        fg_scores, bbox_preds = self(feats)

        featmap_sizes = [featmap.size()[-2:] for featmap in fg_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = fg_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        new_feats = []

        # pre-processing
        # for i in range(len(cls_scores)):
        #     score = fg_scores[i]
        #     s,_ = torch.max(score, dim = 1)
        #     s = s.unsqueeze(1)
        #     s = F.sigmoid(s)
        #     new_feats.append(s*feats[i]+feats[i])
        new_feats = feats

        anchor_list_refine = self.refine_bboxes(anchor_list, bbox_preds, img_metas)
        offset_list = self.anchor_offset(anchor_list_refine, self.anchor_strides, featmap_sizes)
        cls_scores, bbox_preds_refine = self.forward_post(new_feats, offset_list)
        results_list = self.get_bboxes(anchor_list_refine[0], cls_scores, bbox_preds_refine, img_metas, rescale=rescale)
        return results_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   anchor_list,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        # device = cls_scores[0].device
        # featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        # mlvl_anchors2 = self.anchor_generator.grid_anchors(
        #     featmap_sizes, device=device)
        # anchor_list = images_to_levels(anchor_list, num_levels)
        mlvl_anchors = [anchor_list[i].detach() for i in range(num_levels)]
        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]

        if with_nms:
            # some heads don't support with_nms argument
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
                                           mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale)
        else:
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
                                           mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale,
                                           with_nms)
        return result_list

    def _get_bboxes(self,
                    mlvl_cls_scores,
                    mlvl_bbox_preds,
                    mlvl_anchors,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a batch item into bbox predictions.

            Args:
                mlvl_cls_scores (list[Tensor]): Each element in the list is
                    the scores of bboxes of single level in the feature pyramid,
                    has shape (N, num_anchors * num_classes, H, W).
                mlvl_bbox_preds (list[Tensor]):  Each element in the list is the
                    bboxes predictions of single level in the feature pyramid,
                    has shape (N, num_anchors * 4, H, W).
                mlvl_anchors (list[Tensor]): Each element in the list is
                    the anchors of single level in feature pyramid, has shape
                    (num_anchors, 4).
                img_shapes (list[tuple[int]]): Each tuple in the list represent
                    the shape(height, width, 3) of single image in the batch.
                scale_factors (list[ndarray]): Scale factor of the batch
                    image arange as list[(w_scale, h_scale, w_scale, h_scale)].
                cfg (mmcv.Config): Test / postprocessing configuration,
                    if None, test_cfg would be used.
                rescale (bool): If True, return boxes in original image space.
                    Default: False.
                with_nms (bool): If True, do nms before return boxes.
                    Default: True.

            Returns:
                list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                    The first item is an (n, 5) tensor, where 5 represent
                    (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                    The shape of the second tensor in the tuple is (n,), and
                    each element represents the class label of the corresponding
                    box.
            """

        cfg = self.test_cfg if cfg is None else cfg
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(
            mlvl_anchors)
        batch_size = mlvl_cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1),
            device=mlvl_cls_scores[0].device,
            dtype=torch.long)

        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(mlvl_cls_scores,
                                                 mlvl_bbox_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(0, 2, 3,
                                          1).reshape(batch_size, -1,
                                                     self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            anchors = anchors.expand_as(bbox_pred)
            # Always keep topk op for dynamic input in onnx
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(-1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[..., :-1].max(-1)

                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds)
                anchors = anchors[batch_inds, topk_inds, :]
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]

            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            from mmdet.core.export import add_dummy_nms_for_onnx
            # ignore background class
            if not self.use_sigmoid_cls:
                num_classes = batch_mlvl_scores.shape[2] - 1
                batch_mlvl_scores = batch_mlvl_scores[..., :num_classes]
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = batch_mlvl_scores.new_zeros(batch_size,
                                                  batch_mlvl_scores.shape[1],
                                                  1)
            batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes,
                                                  batch_mlvl_scores):
                det_bbox, det_label = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                     cfg.score_thr, cfg.nms,
                                                     cfg.max_per_img)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
            ]
        return det_results