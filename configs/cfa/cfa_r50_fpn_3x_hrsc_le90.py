_base_ = ['../rotated_reppoints/rotated_reppoints_r50_fpn_3x_hrsc_le90.py']
model = dict(
    bbox_head=dict(use_reassign=True),
    train_cfg=dict(
        refine=dict(assigner=dict(pos_iou_thr=0.1, neg_iou_thr=0.1))))
