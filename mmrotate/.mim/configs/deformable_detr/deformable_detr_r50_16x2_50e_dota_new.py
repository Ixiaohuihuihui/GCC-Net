# -*- coding: utf-8 -*-
# @Time    : 09/10/2021 16:19
# @Author  : Linhui Dai
# @FileName: deformable_detr_r50_16x2_50e_dota.py.py
# @Software: PyCharm
_base_ = [
    '../_base_/default_runtime.py'
]
model = dict(
    type='DeformableDETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=5),
    # neck=dict(
    #     type='FPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=256,
    #     start_level=1,
    #     add_extra_convs='on_input',
    #     num_outs=5),
    bbox_head=dict(
        type='DeformableDETRHead',
        num_query=300,
        num_classes=15,
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=True,
        num_feature_levels=5,

        transformer=dict(
            type='DeformableDetrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256,
                        num_levels=5),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1
                            # type='MultiScaleDeformableAttention',
                            # embed_dims=256,
                            # num_levels=5
                            ),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=5)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        # loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_bbox=dict( type='RotatedIoULoss', loss_weight=5.0)
        # loss_iou=dict(type='GIoULoss', loss_weight=2.0)
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            # reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            reg_cost=dict(type='RBBoxL1Cost', weight=5.0, box_format='xywha'),
            # iou_cost=dict(type='RotatedIoUCost', iou_mode='iou', weight=2.0)
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)
        )),
    test_cfg=dict(max_per_img=100))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# dataset settings
dataset_type = 'DOTADataset'
data_root = '/data2/dailh/dota_coco/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval1024/DOTA_trainval1024.json',
        img_prefix=data_root + 'trainval1024/images/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=False,
        with_label=True,
        rotate_aug=dict(
            border_value=0,
            small_filter=6,
            rotate_mode='value',
            rotate_ratio=0.5,
            rotate_values=[30, 60, 90, 120, 150],
        )
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval1024/DOTA_trainval1024.json',
        img_prefix=data_root + 'trainval1024/images/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        # flip=False,
        with_mask=True,
        with_crowd=False,
        with_label=True,
        rotate_test_aug=None,
    ),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'trainval1024/DOTA_trainval1024.json',
        # img_prefix=data_root + 'trainval1024/images/',
        ann_file=data_root + 'test1024/DOTA_test1024.json',
        img_prefix=data_root + 'test1024/images/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        # flip=False,
        with_mask=False,
        with_label=False,
        rotate_test_aug=None,
        test_mode=True))
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

work_dir = 'work_dirs/proposal_level/'
evaluation = dict(work_dir= work_dir, gt_dir='/data2/dailh/dota/trainval/labelTxt/',
                  imagesetfile='work_dirs/proposal_level/test.txt')
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=50)