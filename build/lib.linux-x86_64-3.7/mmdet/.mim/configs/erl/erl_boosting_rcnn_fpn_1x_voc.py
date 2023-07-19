_base_ = [
     '../_base_/datasets/voc0712.py',
    '../_base_/default_runtime.py','../_base_/schedules/schedule_1x.py'
]
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='SiEdgenet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    rpn_head=dict(
        type='ATSSRPNHead',
        in_channels=256,
        feat_channels=256,
        stacked_convs=4,
        reg_decoded_bbox=True,
        gamma=2,
        atss=False,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        # strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=2.0),
        aug_reg_loss=dict(type='MSELoss', loss_weight=2.0),
        # loss_bbox=dict(type='CIoULoss', loss_weight=1.0),
    ),
    # loss_bbox=dict(type='L1Loss', loss_weight=2.0)),
    roi_head=dict(
        type='ProbRoIHead',
        boost=True,
        gamma=0.5,
        reg_norm='num_bbox',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            # featmap_strides=[4, 8, 16, 32, 64, 128]),
            featmap_strides=[8, 16, 32, 64, 128]),
        bbox_head=dict(
            type='ProbConvFCBBoxHead',
            num_shared_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=20,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
            # loss_bbox=dict(type='IoULoss', loss_weight=10.0))
            loss_bbox=dict(type='L1Loss', loss_weight=2.0))
    ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(_delete_=True, type='PseudoSampler'),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=4000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=256,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            # nms=dict(type='soft_nms', iou_threshold=0.7, min_score=0.000),
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    )

)

data = dict(
    samples_per_gpu = 1,
    workers_per_gpu = 1,
)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

