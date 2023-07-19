_base_ = './reppoints_v2_r50_fpn_1x_utdac.py'
model = dict(
    bbox_head=dict(
        loss_bbox_init=dict(_delete_=True, type='GIoULoss', loss_weight=1.0),
        loss_bbox_refine=dict(_delete_=True, type='GIoULoss', loss_weight=2.0))
)
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Retinex', model='MSR', sigma=[30, 150, 300],
         restore_factor=2.0, color_gain=6.0, gain=128.0, offset=128.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img_retinex', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Retinex', model='MSR', sigma=[30, 150, 300], restore_factor=2.0, color_gain=6.0, gain=128.0,
                 offset=128.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'img_retinex']),
            dict(type='Collect', keys=['img', 'img_retinex'])
        ])
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))