# -*- coding: utf-8 -*-
# @Time    : 2022/11/24 09:42
# @Author  : Linhui Dai
# @FileName: enhance_fusion.py
# @Software: PyCharm

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.cnn import (Conv2d, ConvModule)
import torch.nn.functional as F

class EnhanceFusion(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_planes, out_planes, stride=1):
        super(EnhanceFusion, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.edge = edge_attention(64, 512)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x, y, edge, coffient=None):
        fusion = self.conv(y) + x
        if edge is None:
            fusion_attn = fusion
        else:
            fusion_attn = self.gamma * edge + fusion
        return fusion_attn

    # def forward(self, x, y, edge=None, coffient=None):
    #     fusion = self.conv(y) * x
    #     return fusion

def CAM(feature):
    weights = torch.mean(feature, dim=(2, 3))
    cam = torch.matmul(feature, weights.expand(feature.shape))
    cam = torch.mean(cam,dim=1,keepdim=True)
    cam = F.relu(cam)
    return cam

def DeepSobel_tmp(im):
    # im = im.squeeze()
    # im = np.swapaxes(im, 0, 2)
    # im = np.swapaxes(im, 1, 2)
    Gx = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).cuda()
    # Build sobel x, y gradient filters
    Gy = Gx.permute(0, 1)
    Gx = Gx.expand(1, 1, 3, 3)
    Gy = Gy.expand(1, 1, 3, 3)
    # Gy = np.swapaxes(Gx, 0, 1)
    ndim = im.shape[1]
    # TotGrad = np.zeros(im[1, :, :].shape)
    TotGrad = torch.zeros(im.shape).cuda()
    # sobel = torch.conv2d(im[None], Gx, stride=1, padding=1)

    for ii in range(ndim):
        # gradx = signal.convolve2d(im[ii, :, :], Gx, boundary='symm', mode='same')
        gradx = torch.conv2d(im[:, ii, :, :][None], Gx, stride=1, padding=1)
        grady = torch.conv2d(im[:, ii, :, :][None], Gy, stride=1, padding=1)
        # grady = signal.convolve2d(im[ii, :, :], Gy, boundary='symm', mode='same')
        grad = torch.sqrt(torch.pow(gradx, 2) + torch.pow(grady, 2))
        # grad = np.sqrt(np.power(gradx, 2) + np.power(grady, 2))
        TotGrad += grad
    TotGrad /= ndim
    return TotGrad

def DeepSobel(im):
    num_channel = im.shape[1]
    Gx = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).cuda()
    # Build sobel x, y gradient filters
    # Gy = Gx.permute(0, 1)
    Gy = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).cuda()
    Gx = Gx.expand(num_channel, 1, 3, 3)
    Gy = Gy.expand(num_channel, 1, 3, 3)

    gradx = torch.conv2d(im, Gx, stride=1, padding=1, groups=num_channel)
    grady = torch.conv2d(im, Gy, stride=1, padding=1, groups=num_channel)
    grad = torch.sqrt(torch.pow(gradx, 2) + torch.pow(grady, 2))
    return grad

class edge_attention(BaseModule):
    def __init__(self, in_channels, out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(edge_attention, self).__init__(init_cfg=init_cfg)
        self.relu = nn.ReLU(False)
        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    def forward(self, x):
        x0 = self.conv(x)
        x = self.relu(x0) + x
        return x