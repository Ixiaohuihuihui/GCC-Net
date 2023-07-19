_base_ = ['./cfa_r50_fpn_3x_hrsc_le90.py']
# evaluation
evaluation = dict(interval=12, metric='mAP')
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[24, 32, 38])
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(interval=10)
