# -*- coding: utf-8 -*-
# @Time    : 07/06/2022 21:17
# @Author  : Linhui Dai
# @FileName: cfa_r50_fpn_1x_utdac.py.py
# @Software: PyCharm
_base_ = ['../reppoints/reppoints_moment_r50_fpn_1x_utdac.py']
model = dict(
    bbox_head=dict(use_reassign=True),
    train_cfg=dict(
        refine=dict(assigner=dict(pos_iou_thr=0.1, neg_iou_thr=0.1))))


