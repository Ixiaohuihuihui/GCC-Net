_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/utdac_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    bbox_head=dict(
        _delete_=True,
        type='SABLRetinaHead',
        num_classes=4,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        approx_anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        square_anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[4],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='BucketingBBoxCoder', num_buckets=14, scale_factor=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.5),
        loss_bbox_reg=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.5)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='ApproxMaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0.0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))
# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
# img_norm_cfg = dict(
#     mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Retinex', model='MSR', sigma=[30, 150, 300],
#          restore_factor=2.0, color_gain=6.0, gain=128.0, offset=128.0),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'img_retinex', 'gt_bboxes', 'gt_labels'])
# ]
#
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Retinex', model='MSR', sigma=[30, 150, 300], restore_factor=2.0, color_gain=6.0, gain=128.0,
#                  offset=128.0),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img', 'img_retinex']),
#             dict(type='Collect', keys=['img', 'img_retinex'])
#         ])
# ]
#
# data = dict(
#     train=dict(pipeline=train_pipeline),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))