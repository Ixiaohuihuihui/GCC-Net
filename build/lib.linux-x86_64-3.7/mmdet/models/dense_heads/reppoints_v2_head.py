import mmcv
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.core import (PointGenerator, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from mmcv.ops import DeformConv2d
from mmcv.ops import CornerPool
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from .corner_head import BiCornerPool
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from ..utils import TLPool, BRPool


@HEADS.register_module()
class RepPointsV2Head(AnchorFreeHead):
    """RepPoint head.

    Args:
        point_feat_channels (int): Number of channels of points features.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        loss_cls (dict): Config of classification loss.
        loss_bbox_init (dict): Config of initial points loss.
        loss_bbox_refine (dict): Config of points loss in refinement.
        use_grid_points (bool): If we use bounding box representation, the
        reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
        transform_method (str): The methods to transform RepPoints to bbox.
    """  # noqa: W605

    def __init__(
            self,
            num_classes,
            in_channels,
            point_feat_channels=256,
            shared_stacked_convs=1,
            first_kernel_size=3,
            kernel_size=1,
            corner_dim=64,
            num_points=9,
            gradient_mul=0.1,
            point_strides=[8, 16, 32, 64, 128],
            point_base_scale=4,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
        loss_bbox_init=dict(
                type='SmoothL1Loss',
                beta=1.0 / 9.0,
                loss_weight=0.5),
            loss_bbox_refine=dict(
                type='SmoothL1Loss',
                beta=1.0 / 9.0,
                loss_weight=1.0),
            loss_heatmap=dict(
                type='GaussianFocalLoss',
                alpha=2.0,
                gamma=4.0,
                loss_weight=0.25),
            loss_offset=dict(
                type='SmoothL1Loss',
                beta=1.0 / 9.0,
                loss_weight=1.0),
            loss_uie=dict(
                type='SmoothL1Loss',
                beta=1.0 / 9.0,
                loss_weight=1.0),
            use_grid_points=False,
            center_init=True,
            transform_method='moment',
            moment_mul=0.01,
            init_cfg=dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=dict(
                    type='Normal',
                    name='reppoints_cls_out',
                    std=0.01,
                    bias_prob=0.01)),
            **kwargs):
        self.num_points = num_points
        self.point_feat_channels = point_feat_channels
        self.shared_stacked_convs = shared_stacked_convs
        self.use_grid_points = use_grid_points
        self.center_init = center_init

        self.first_kernel_size = first_kernel_size
        self.kernel_size = kernel_size
        self.corner_dim = corner_dim

        # we use deformable conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            init_cfg=init_cfg,
            **kwargs)
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.point_generators = [PointGenerator() for _ in self.point_strides]

        self.sampling = loss_cls['type'] not in ['FocalLoss']
        if self.train_cfg:
            self.init_assigner = build_assigner(self.train_cfg.init.assigner)
            self.refine_assigner = build_assigner(
                self.train_cfg.refine.assigner)
            self.hm_assigner = build_assigner(self.train_cfg.heatmap.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.transform_method = transform_method
        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(
                data=torch.zeros(2), requires_grad=True)
            self.moment_mul = moment_mul

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1

        self.cls_out_channels = self.num_classes
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.loss_heatmap = build_loss(loss_heatmap)
        self.loss_offset = build_loss(loss_offset)
        self.loss_uie = build_loss(loss_uie)

    def _make_layers(self, out_channels, in_channels=256, feat_channels=256):
        """Initialize conv sequential for CornerHead."""
        return nn.Sequential(
            ConvModule(in_channels, feat_channels, 3, padding=1),
            ConvModule(
                feat_channels, out_channels, 1, norm_cfg=None, act_cfg=None))

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.shared_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        for i in range(self.shared_stacked_convs):
            self.shared_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.hem_tl = TLPool(
            self.feat_channels,
            self.conv_cfg,
            self.norm_cfg,
            first_kernel_size=self.first_kernel_size,
            kernel_size=self.kernel_size,
            corner_dim=self.corner_dim)
        self.hem_br = BRPool(
            self.feat_channels,
            self.conv_cfg,
            self.norm_cfg,
            first_kernel_size=self.first_kernel_size,
            kernel_size=self.kernel_size,
            corner_dim=self.corner_dim)

        pts_out_dim = 4 if self.use_grid_points else 2 * self.num_points

        cls_in_channels = self.feat_channels
        self.reppoints_cls_conv = DeformConv2d(
            cls_in_channels,
            self.point_feat_channels,
            self.dcn_kernel,
            1,
            self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)

        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        pts_in_channels = self.feat_channels
        self.reppoints_pts_refine_conv = DeformConv2d(pts_in_channels,
                                                      self.point_feat_channels,
                                                      self.dcn_kernel, 1,
                                                      self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)

        self.reppoints_hem_tl_score_out = nn.Conv2d(
            self.feat_channels, 1, 3, 1, 1)
        self.reppoints_hem_br_score_out = nn.Conv2d(
            self.feat_channels, 1, 3, 1, 1)
        self.reppoints_hem_tl_offset_out = nn.Conv2d(
            self.feat_channels, 2, 3, 1, 1)
        self.reppoints_hem_br_offset_out = nn.Conv2d(
            self.feat_channels, 2, 3, 1, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.shared_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.reppoints_cls_conv, std=0.01)
        normal_init(self.reppoints_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)
        normal_init(self.reppoints_pts_refine_conv, std=0.01)
        normal_init(self.reppoints_pts_refine_out, std=0.01)
        normal_init(self.reppoints_hem_tl_score_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_hem_tl_offset_out, std=0.01)
        normal_init(self.reppoints_hem_br_score_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_hem_br_offset_out, std=0.01)

    def points2bbox(self, pts, y_first=True):
        """Converting the points set into bounding box.

        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_fisrt=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :return: each points set is converting to a bbox [x1, y1, x2, y2].
        """
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                                      ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                                      ...]
        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'partial_minmax':
            pts_y = pts_y[:, :4, ...]
            pts_x = pts_x[:, :4, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'moment':
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ],
                dim=1)
        elif self.transform_method == "exact_minmax":
            pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
            pts_reshape = pts_reshape[:, :2, ...]
            pts_y = pts_reshape[:, :,
                                0, ...] if y_first else pts_reshape[:, :, 1, ...]
            pts_x = pts_reshape[:, :,
                                1, ...] if y_first else pts_reshape[:, :, 0, ...]
            bbox_left = pts_x[:, 0:1, ...]
            bbox_right = pts_x[:, 1:2, ...]
            bbox_up = pts_y[:, 0:1, ...]
            bbox_bottom = pts_y[:, 1:2, ...]
            bbox = torch.cat(
                [bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)
        else:
            raise NotImplementedError
        return bbox

    def gen_grid_from_reg(self, reg, previous_boxes):
        """Base on the previous bboxes and regression values, we compute the
        regressed bboxes and generate the grids on the bboxes.

        :param reg: the regression value to previous bboxes.
        :param previous_boxes: previous bboxes.
        :return: generate grids on the regressed bboxes.
        """
        b, _, h, w = reg.shape
        bxy = (previous_boxes[:, :2, ...] + previous_boxes[:, 2:, ...]) / 2.
        bwh = (previous_boxes[:, 2:, ...] -
               previous_boxes[:, :2, ...]).clamp(min=1e-6)
        grid_topleft = bxy + bwh * reg[:, :2, ...] - 0.5 * bwh * torch.exp(
            reg[:, 2:, ...])
        grid_wh = bwh * torch.exp(reg[:, 2:, ...])
        grid_left = grid_topleft[:, [0], ...]
        grid_top = grid_topleft[:, [1], ...]
        grid_width = grid_wh[:, [0], ...]
        grid_height = grid_wh[:, [1], ...]
        intervel = torch.linspace(0., 1., self.dcn_kernel).view(
            1, self.dcn_kernel, 1, 1).type_as(reg)
        grid_x = grid_left + grid_width * intervel
        grid_x = grid_x.unsqueeze(1).repeat(1, self.dcn_kernel, 1, 1, 1)
        grid_x = grid_x.view(b, -1, h, w)
        grid_y = grid_top + grid_height * intervel
        grid_y = grid_y.unsqueeze(2).repeat(1, 1, self.dcn_kernel, 1, 1)
        grid_y = grid_y.view(b, -1, h, w)
        grid_yx = torch.stack([grid_y, grid_x], dim=2)
        grid_yx = grid_yx.view(b, -1, h, w)
        regressed_bbox = torch.cat([
            grid_left, grid_top, grid_left + grid_width, grid_top + grid_height
        ], 1)
        return grid_yx, regressed_bbox

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_train(self,
                      x,
                      x_or,
                      raw_img,
                      enhanced_img,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (x, x_or, gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (x, x_or, gt_bboxes, gt_labels, img_metas)

        # LOSS FOR UOD
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def forward_single(self, x):
        """Forward feature map of a single FPN level."""
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        # If we use center_init, the initial reppoints is from center points.
        # If we use bounding bbox representation, the initial reppoints is
        #   from regular grid placed on a pre-defined bbox.
        if self.use_grid_points or not self.center_init:
            scale = self.point_base_scale / 2
            points_init = dcn_base_offset / dcn_base_offset.max() * scale
            bbox_init = x.new_tensor([-scale, -scale, scale,
                                      scale]).view(1, 4, 1, 1)
        else:
            points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)

        # shared_feat = pts_feat
        # for shared_conv in self.shared_convs:
        #     shared_feat = shared_conv(shared_feat)

        hem_feat = pts_feat
        # generate heatmap and offset
        hem_tl_feat = self.hem_tl(hem_feat)
        hem_br_feat = self.hem_br(hem_feat)

        hem_tl_score_out = self.reppoints_hem_tl_score_out(hem_tl_feat)
        hem_tl_offset_out = self.reppoints_hem_tl_offset_out(hem_tl_feat)
        hem_br_score_out = self.reppoints_hem_br_score_out(hem_br_feat)
        hem_br_offset_out = self.reppoints_hem_br_offset_out(hem_br_feat)

        hem_score_out = torch.cat([hem_tl_score_out, hem_br_score_out], dim=1)
        hem_offset_out = torch.cat(
            [hem_tl_offset_out, hem_br_offset_out], dim=1)

        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        if self.use_grid_points:
            pts_out_init, bbox_out_init = self.gen_grid_from_reg(
                pts_out_init, bbox_init.detach())
        else:
            pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(
        ) + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        cls_out = self.reppoints_cls_out(
            self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))
        pts_out_refine = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
        if self.use_grid_points:
            pts_out_refine, bbox_out_refine = self.gen_grid_from_reg(
                pts_out_refine, bbox_out_init.detach())
        else:
            pts_out_refine = pts_out_refine + pts_out_init.detach()
        return cls_out, pts_out_init, pts_out_refine, hem_score_out, hem_offset_out

    def get_points(self, featmap_sizes, img_metas, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = self.prior_generator.grid_priors(
            featmap_sizes, device=device, with_stride=True)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'])
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def centers_to_bboxes(self, point_list):
        """Get bboxes according to center points.

        Only used in :class:`MaxIoUAssigner`.
        """
        bbox_list = []
        for i_img, point in enumerate(point_list):
            bbox = []
            for i_lvl in range(len(self.point_strides)):
                scale = self.point_base_scale * self.point_strides[i_lvl] * 0.5
                bbox_shift = torch.Tensor([-scale, -scale, scale,
                                           scale]).view(1, 4).type_as(point[0])
                bbox_center = torch.cat(
                    [point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift)
            bbox_list.append(bbox)
        return bbox_list

    def offset_to_pts(self, center_list, pred_list):
        """Change from point offset to point coordinate."""
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                    -1, 2 * self.num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def _point_target_single(self,
                             flat_proposals,
                             valid_flags,
                             num_level_proposals,
                             gt_bboxes,
                             gt_bboxes_ignore,
                             gt_labels,
                             stage='init',
                             unmap_outputs=True):
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]

        num_level_proposals_inside = self.get_num_level_proposals_inside(
            num_level_proposals, inside_flags)
        if stage == 'init':
            assigner = self.init_assigner
            assigner_type = self.train_cfg.init.assigner.type
            pos_weight = self.train_cfg.init.pos_weight
        else:
            assigner = self.refine_assigner
            assigner_type = self.train_cfg.refine.assigner.type
            pos_weight = self.train_cfg.refine.pos_weight
        if assigner_type != "ATSSAssigner":
            assign_result = assigner.assign(
                proposals, gt_bboxes, gt_bboxes_ignore, gt_labels)
        else:
            assign_result = assigner.assign(
                proposals,
                num_level_proposals_inside,
                gt_bboxes,
                gt_bboxes_ignore,
                gt_labels)
        sampling_result = self.sampler.sample(assign_result, proposals,
                                              gt_bboxes)

        num_valid_proposals = proposals.shape[0]
        bbox_gt = proposals.new_zeros([num_valid_proposals, 4])
        pos_proposals = torch.zeros_like(proposals)
        proposals_weights = proposals.new_zeros([num_valid_proposals, 4])
        labels = proposals.new_full((num_valid_proposals, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = proposals.new_zeros(
            num_valid_proposals, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            bbox_gt[pos_inds, :] = pos_gt_bboxes
            pos_proposals[pos_inds, :] = proposals[pos_inds, :]
            proposals_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(labels, num_total_proposals, inside_flags)
            label_weights = unmap(label_weights, num_total_proposals,
                                  inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
            pos_proposals = unmap(pos_proposals, num_total_proposals,
                                  inside_flags)
            proposals_weights = unmap(proposals_weights, num_total_proposals,
                                      inside_flags)

        return (labels, label_weights, bbox_gt, pos_proposals,
                proposals_weights, pos_inds, neg_inds)

    def get_targets(self,
                    proposals_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    stage='init',
                    label_channels=1,
                    unmap_outputs=True):
        """Compute corresponding GT box and classification targets for
        proposals.

        Args:
            proposals_list (list[list]): Multi level points/bboxes of each
                image.
            valid_flag_list (list[list]): Multi level valid flags of each
                image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_bboxes_list (list[Tensor]): Ground truth labels of each box.
            stage (str): `init` or `refine`. Generate target for init stage or
                refine stage
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each level.  # noqa: E501
                - bbox_gt_list (list[Tensor]): Ground truth bbox of each level.
                - proposal_list (list[Tensor]): Proposals(points/bboxes) of each level.  # noqa: E501
                - proposal_weights_list (list[Tensor]): Proposal weights of each level.  # noqa: E501
                - num_total_pos (int): Number of positive samples in all images.  # noqa: E501
                - num_total_neg (int): Number of negative samples in all images.  # noqa: E501
        """
        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]
        num_level_proposals_list = [num_level_proposals] * num_imgs

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_labels, all_label_weights, all_bbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._point_target_single,
             proposals_list,
             valid_flag_list,
             num_level_proposals_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             stage=stage,
             unmap_outputs=unmap_outputs)
        # no valid points
        if any([labels is None for labels in all_labels]):
            return None
        # sampled points of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_proposals)
        bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
        proposals_list = images_to_levels(all_proposals, num_level_proposals)
        proposal_weights_list = images_to_levels(all_proposal_weights,
                                                 num_level_proposals)
        return (labels_list, label_weights_list, bbox_gt_list, proposals_list,
                proposal_weights_list, num_total_pos, num_total_neg)

    def _hm_target_single(self,
                          flat_points,
                          inside_flags,
                          gt_bboxes,
                          gt_labels,
                          unmap_outputs=True):
        # assign gt and sample points
        if not inside_flags.any():
            return (None, ) * 12
        points = flat_points[inside_flags, :]

        assigner = self.hm_assigner
        gt_hm_tl, gt_offset_tl, pos_inds_tl, neg_inds_tl, \
            gt_hm_br, gt_offset_br, pos_inds_br, neg_inds_br = \
            assigner.assign(points, gt_bboxes, gt_labels)

        num_valid_points = points.shape[0]
        hm_tl_weights = points.new_zeros(num_valid_points, dtype=torch.float)
        hm_br_weights = points.new_zeros(num_valid_points, dtype=torch.float)
        offset_tl_weights = points.new_zeros(
            [num_valid_points, 2], dtype=torch.float)
        offset_br_weights = points.new_zeros(
            [num_valid_points, 2], dtype=torch.float)

        hm_tl_weights[pos_inds_tl] = 1.0
        hm_tl_weights[neg_inds_tl] = 1.0
        offset_tl_weights[pos_inds_tl, :] = 1.0

        hm_br_weights[pos_inds_br] = 1.0
        hm_br_weights[neg_inds_br] = 1.0
        offset_br_weights[pos_inds_br, :] = 1.0

        # map up to original set of grids
        if unmap_outputs:
            num_total_points = flat_points.shape[0]
            gt_hm_tl = unmap(gt_hm_tl, num_total_points, inside_flags)
            gt_offset_tl = unmap(gt_offset_tl, num_total_points, inside_flags)
            hm_tl_weights = unmap(
                hm_tl_weights,
                num_total_points,
                inside_flags)
            offset_tl_weights = unmap(
                offset_tl_weights, num_total_points, inside_flags)

            gt_hm_br = unmap(gt_hm_br, num_total_points, inside_flags)
            gt_offset_br = unmap(gt_offset_br, num_total_points, inside_flags)
            hm_br_weights = unmap(
                hm_br_weights,
                num_total_points,
                inside_flags)
            offset_br_weights = unmap(
                offset_br_weights, num_total_points, inside_flags)

        return (
            gt_hm_tl,
            gt_offset_tl,
            hm_tl_weights,
            offset_tl_weights,
            pos_inds_tl,
            neg_inds_tl,
            gt_hm_br,
            gt_offset_br,
            hm_br_weights,
            offset_br_weights,
            pos_inds_br,
            neg_inds_br)

    def get_hm_targets(self,
                       proposals_list,
                       valid_flag_list,
                       gt_bboxes_list,
                       img_metas,
                       gt_labels_list=None,
                       unmap_outputs=True):
        """Compute refinement and classification targets for points.

        Args:
            points_list (list[list]): Multi level points of each image.
            valid_flag_list (list[list]): Multi level valid flags of each image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            cfg (dict): train sample configs.

        Returns:
            tuple
        """
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]

        # concat all level points and flags to a single tensor
        for i in range(len(proposals_list)):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_gt_hm_tl,
         all_gt_offset_tl,
         all_hm_tl_weights,
         all_offset_tl_weights,
         pos_inds_tl_list,
         neg_inds_tl_list,
         all_gt_hm_br,
         all_gt_offset_br,
         all_hm_br_weights,
         all_offset_br_weights,
         pos_inds_br_list,
         neg_inds_br_list) = multi_apply(self._hm_target_single,
                                         proposals_list,
                                         valid_flag_list,
                                         gt_bboxes_list,
                                         gt_labels_list,
                                         unmap_outputs=unmap_outputs)
        # no valid points
        if any([gt_hm_tl is None for gt_hm_tl in all_gt_hm_tl]):
            return None
        # sampled points of all images
        num_total_pos_tl = sum([max(inds.numel(), 1)
                               for inds in pos_inds_tl_list])
        num_total_neg_tl = sum([max(inds.numel(), 1)
                               for inds in neg_inds_tl_list])
        num_total_pos_br = sum([max(inds.numel(), 1)
                               for inds in pos_inds_br_list])
        num_total_neg_br = sum([max(inds.numel(), 1)
                               for inds in neg_inds_br_list])

        gt_hm_tl_list = images_to_levels(all_gt_hm_tl, num_level_proposals)
        gt_offset_tl_list = images_to_levels(
            all_gt_offset_tl, num_level_proposals)
        hm_tl_weight_list = images_to_levels(
            all_hm_tl_weights, num_level_proposals)
        offset_tl_weight_list = images_to_levels(
            all_offset_tl_weights, num_level_proposals)

        gt_hm_br_list = images_to_levels(all_gt_hm_br, num_level_proposals)
        gt_offset_br_list = images_to_levels(
            all_gt_offset_br, num_level_proposals)
        hm_br_weight_list = images_to_levels(
            all_hm_br_weights, num_level_proposals)
        offset_br_weight_list = images_to_levels(
            all_offset_br_weights, num_level_proposals)

        return (
            gt_hm_tl_list,
            gt_offset_tl_list,
            hm_tl_weight_list,
            offset_tl_weight_list,
            gt_hm_br_list,
            gt_offset_br_list,
            hm_br_weight_list,
            offset_br_weight_list,
            num_total_pos_tl,
            num_total_neg_tl,
            num_total_pos_br,
            num_total_neg_br)

    def loss_single(
            self,
            cls_score,
            pts_pred_init,
            pts_pred_refine,
            hm_score,
            hm_offset,
            labels,
            label_weights,
            bbox_gt_init,
            bbox_weights_init,
            bbox_gt_refine,
            bbox_weights_refine,
            gt_hm_tl,
            gt_offset_tl,
            gt_hm_tl_weight,
            gt_offset_tl_weight,
            gt_hm_br,
            gt_offset_br,
            gt_hm_br_weight,
            gt_offset_br_weight,
            stride,
            num_total_samples_init,
            num_total_samples_refine,
            num_total_samples_tl,
            num_total_samples_br):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        cls_score = cls_score.contiguous()
        loss_cls = self.loss_cls(
            cls_score,
            labels,
            label_weights,
            avg_factor=num_total_samples_refine)

        # points loss
        bbox_gt_init = bbox_gt_init.reshape(-1, 4)
        bbox_weights_init = bbox_weights_init.reshape(-1, 4)
        bbox_pred_init = self.points2bbox(
            pts_pred_init.reshape(-1, 2 * self.num_points), y_first=False)
        bbox_gt_refine = bbox_gt_refine.reshape(-1, 4)
        bbox_weights_refine = bbox_weights_refine.reshape(-1, 4)
        bbox_pred_refine = self.points2bbox(
            pts_pred_refine.reshape(-1, 2 * self.num_points), y_first=False)
        normalize_term = self.point_base_scale * stride
        loss_pts_init = self.loss_bbox_init(
            bbox_pred_init / normalize_term,
            bbox_gt_init / normalize_term,
            bbox_weights_init,
            avg_factor=num_total_samples_init)
        loss_pts_refine = self.loss_bbox_refine(
            bbox_pred_refine / normalize_term,
            bbox_gt_refine / normalize_term,
            bbox_weights_refine,
            avg_factor=num_total_samples_refine)

        # heatmap cls loss
        hm_score = hm_score.permute(0, 2, 3, 1).reshape(-1, 2)
        hm_score_tl, hm_score_br = torch.chunk(hm_score, 2, dim=-1)
        hm_score_tl = hm_score_tl.squeeze(1).sigmoid()
        hm_score_br = hm_score_br.squeeze(1).sigmoid()

        gt_hm_tl = gt_hm_tl.reshape(-1)
        gt_hm_tl_weight = gt_hm_tl_weight.reshape(-1)
        gt_hm_br = gt_hm_br.reshape(-1)
        gt_hm_br_weight = gt_hm_br_weight.reshape(-1)

        loss_heatmap = 0
        loss_heatmap += self.loss_heatmap(hm_score_tl,
                                          gt_hm_tl,
                                          gt_hm_tl_weight,
                                          avg_factor=num_total_samples_tl)
        loss_heatmap += self.loss_heatmap(hm_score_br,
                                          gt_hm_br,
                                          gt_hm_br_weight,
                                          avg_factor=num_total_samples_br)
        loss_heatmap /= 2.0

        # heatmap offset loss
        hm_offset = hm_offset.permute(0, 2, 3, 1).reshape(-1, 4)
        hm_offset_tl, hm_offset_br = torch.chunk(hm_offset, 2, dim=-1)

        gt_offset_tl = gt_offset_tl.reshape(-1, 2)
        gt_offset_tl_weight = gt_offset_tl_weight.reshape(-1, 2)
        gt_offset_br = gt_offset_br.reshape(-1, 2)
        gt_offset_br_weight = gt_offset_br_weight.reshape(-1, 2)

        loss_offset = 0
        loss_offset += self.loss_offset(
            hm_offset_tl, gt_offset_tl, gt_offset_tl_weight,
            avg_factor=num_total_samples_tl
        )
        loss_offset += self.loss_offset(
            hm_offset_br, gt_offset_br, gt_offset_br_weight,
            avg_factor=num_total_samples_br
        )
        loss_offset /= 2.0

        return loss_cls, loss_pts_init, loss_pts_refine, loss_heatmap, loss_offset

    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             hm_scores,
             hm_offsets,
             x,
             x_or,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        # 传进参数

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        # target for initial stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas, device)
        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)

        if self.train_cfg.init.assigner['type'] != 'MaxIoUAssigner':
            # Assign target for center list
            candidate_list = center_list
        else:
            # transform center list to bbox list and
            #   assign target for bbox list
            bbox_list = self.centers_to_bboxes(center_list)
            candidate_list = bbox_list
        cls_reg_targets_init = self.get_targets(
            candidate_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            stage='init',
            label_channels=label_channels)
        (*_, bbox_gt_list_init, candidate_list_init, bbox_weights_list_init,
         num_total_pos_init, num_total_neg_init) = cls_reg_targets_init
        num_total_samples_init = (
            num_total_pos_init +
            num_total_neg_init if self.sampling else num_total_pos_init)

        # target for heatmap in initial stage
        proposal_list, valid_flag_list = self.get_points(
            featmap_sizes, img_metas, device)
        heatmap_targets = self.get_hm_targets(
            proposal_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_labels)
        (gt_hm_tl_list,
         gt_offset_tl_list,
         gt_hm_tl_weight_list,
         gt_offset_tl_weight_list,
         gt_hm_br_list,
         gt_offset_br_list,
         gt_hm_br_weight_list,
         gt_offset_br_weight_list,
         num_total_pos_tl,
         num_total_neg_tl,
         num_total_pos_br,
         num_total_neg_br) = heatmap_targets

        # target for refinement stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas, device)
        pts_coordinate_preds_refine = self.offset_to_pts(
            center_list, pts_preds_refine)
        bbox_list = []
        for i_img, center in enumerate(center_list):
            bbox = []
            for i_lvl in range(len(pts_preds_refine)):
                bbox_preds_init = self.points2bbox(
                    pts_preds_init[i_lvl].detach())
                bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
                bbox_center = torch.cat(
                    [center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center +
                            bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))
            bbox_list.append(bbox)
        cls_reg_targets_refine = self.get_targets(
            bbox_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            stage='refine',
            label_channels=label_channels)
        (labels_list, label_weights_list, bbox_gt_list_refine,
         candidate_list_refine, bbox_weights_list_refine, num_total_pos_refine,
         num_total_neg_refine) = cls_reg_targets_refine
        num_total_samples_refine = (
            num_total_pos_refine +
            num_total_neg_refine if self.sampling else num_total_pos_refine)

        # compute loss
        losses_cls, losses_pts_init, losses_pts_refine, losses_heatmap, losses_offset = multi_apply(
            self.loss_single,
            cls_scores,
            pts_coordinate_preds_init,
            pts_coordinate_preds_refine,
            hm_scores,
            hm_offsets,
            labels_list,
            label_weights_list,
            bbox_gt_list_init,
            bbox_weights_list_init,
            bbox_gt_list_refine,
            bbox_weights_list_refine,
            gt_hm_tl_list,
            gt_offset_tl_list,
            gt_hm_tl_weight_list,
            gt_offset_tl_weight_list,
            gt_hm_br_list,
            gt_offset_br_list,
            gt_hm_br_weight_list,
            gt_offset_br_weight_list,
            self.point_strides,
            num_total_samples_init=num_total_samples_init,
            num_total_samples_refine=num_total_samples_refine,
            num_total_samples_tl=num_total_pos_tl,
            num_total_samples_br=num_total_pos_br)

        # loss for FUNIE_GAN
        loss_cycle = self.loss_cycle(
            x, x_or, gt_bboxes, img_metas)

        loss_dict_all = {'loss_cls': losses_cls,
                         'loss_pts_init': losses_pts_init,
                         'loss_pts_refine': losses_pts_refine,
                         'loss_heatmap': losses_heatmap,
                         'loss_offset': losses_offset,
                         'loss_uie': loss_cycle
                         }
        return loss_dict_all

    def loss_cycle(self, x, x_ori, gt_bboxes, img_metas):
        # 传进参数
        # object-aware lossd

        a = []
        gt_bbox_list = torch.cat(gt_bboxes, dim=0)
        featmap_strides = [8, 16, 32, 64, 128]
        loss_cycle = []
        for j in range(len(featmap_strides)):
            location = gt_bbox_list / featmap_strides[j]
            loss_object = 0
            for z in range(len(location)):
                object_feat_enhanced = x[j][:, :, int(location[z][0]):math.ceil(
                    location[z][2]), int(location[z][1]):math.ceil(location[z][3])]
                object_feat_original = x_ori[j][:, :, int(location[z][0]):math.ceil(
                    location[z][2]), int(location[z][1]):math.ceil(location[z][3])]

                if object_feat_original.data is None or object_feat_enhanced.data is None:
                    print("WARNING WORNING")

                loss_object += self.loss_uie(
                    object_feat_enhanced, object_feat_original)
            loss_cycle.append(loss_object)
                    # 高层特征图
                    # 全局一致性loss
        # image content losssss
        return loss_cycle

    def get_bboxes(self,
                   cls_scores,
                   pts_preds_init,
                   pts_preds_refine,
                   hm_scores,
                   hm_offsets,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        assert len(cls_scores) == len(pts_preds_refine)
        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)
        bbox_preds_refine = [self.points2bbox(
            pts_pred_refine) for pts_pred_refine in pts_preds_refine]
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []
        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds_refine, img_id)
            hm_scores_list = select_single_mlvl(hm_scores, img_id)
            hm_offsets_list = select_single_mlvl(hm_offsets, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]
            proposals = self._get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                score_factor_list,
                hm_scores_list,
                hm_offsets_list,
                mlvl_priors,
                img_meta,
                cfg,
                rescale,
                with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           hm_scores,
                           hm_offsets,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True):
        def select(score_map, x, y, ks=2, i=0):
            H, W = score_map.shape[-2], score_map.shape[-1]
            score_map = score_map.sigmoid()
            score_map_original = score_map.clone()

            score_map, indices = F.max_pool2d_with_indices(
                score_map.unsqueeze(0), kernel_size=ks, stride=1, padding=(ks - 1) // 2)

            indices = indices.squeeze(0).squeeze(0)

            if ks % 2 == 0:
                round_func = torch.floor
            else:
                round_func = torch.round

            x_round = round_func(
                (x / self.point_strides[i]).clamp(min=0, max=score_map.shape[-1] - 1))
            y_round = round_func(
                (y / self.point_strides[i]).clamp(min=0, max=score_map.shape[-2] - 1))

            select_indices = indices[y_round.to(
                torch.long), x_round.to(torch.long)]
            new_x = select_indices % W
            new_y = select_indices // W

            score_map_squeeze = score_map_original.squeeze(0)
            score = score_map_squeeze[new_y, new_x]

            new_x, new_y = new_x.to(torch.float), new_y.to(torch.float)

            return new_x, new_y, score

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, priors) in enumerate(
                zip(cls_score_list, bbox_pred_list, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                priors = priors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bbox_pos_center = torch.cat([priors[:, :2], priors[:, :2]], dim=1)
            bboxes = bbox_pred * \
                self.point_strides[level_idx] + bbox_pos_center
            x1 = bboxes[:, 0].clamp(min=0, max=img_shape[1])
            y1 = bboxes[:, 1].clamp(min=0, max=img_shape[0])
            x2 = bboxes[:, 2].clamp(min=0, max=img_shape[1])
            y2 = bboxes[:, 3].clamp(min=0, max=img_shape[0])

            if level_idx > 0:
                i = 0 if level_idx in (1, 2) else 1

                x1_new, y1_new, score1_new = select(
                    hm_scores[i][0, ...], x1, y1, 2, i)
                x2_new, y2_new, score2_new = select(
                    hm_scores[i][1, ...], x2, y2, 2, i)

                hm_offset = hm_offsets[i].permute(1, 2, 0)
                point_stride = self.point_strides[i]

                x1 = ((x1_new +
                       hm_offset[y1_new.to(torch.long), x1_new.to(torch.long), 0]) *
                      point_stride).clamp(min=0, max=img_shape[1])
                y1 = ((y1_new +
                       hm_offset[y1_new.to(torch.long), x1_new.to(torch.long), 1]) *
                      point_stride).clamp(min=0, max=img_shape[0])
                x2 = ((x2_new +
                       hm_offset[y2_new.to(torch.long), x2_new.to(torch.long), 2]) *
                      point_stride).clamp(min=0, max=img_shape[1])
                y2 = ((y2_new +
                       hm_offset[y2_new.to(torch.long), x2_new.to(torch.long), 3]) *
                      point_stride).clamp(min=0, max=img_shape[0])
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(img_meta['scale_factor'])
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        if with_nms:
            det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def get_num_level_proposals_inside(
            self, num_level_proposals, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_proposals)
        num_level_proposals_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_proposals_inside
