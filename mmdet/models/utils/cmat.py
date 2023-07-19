# -*- coding: utf-8 -*-
# @Time    : 2022/11/16 22:31
# @Author  : Linhui Dai
# @FileName: cmat.py
# @Software: PyCharm
import torch.nn as nn
import torch
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv0 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv0(x)
        return self.sigmoid(x)

class CrossAttention(nn.Module):
    def __init__(self, in_channel, ratio=8):
        super(CrossAttention, self).__init__()
        self.conv_query = nn.Conv2d(in_channel, in_channel//ratio, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channel, in_channel//ratio, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channel, in_channel, kernel_size=1)

    def forward(self, rgb, depth):
        bz, c, h, w = rgb.shape
        # [bz, h*w, channel]
        depth_q = self.conv_query(depth).view(bz, -1, h*w).permute(0, 2, 1)
        # [bz, channel, h*w]
        depth_k = self.conv_key(depth).view(bz, -1, h*w)
        # [bz, h*w, h*w]
        mask = torch.bmm(depth_q, depth_k) #bz, hw, hw
        mask = torch.softmax(mask, dim=-1)
        rgb_v = self.conv_value(rgb).view(bz, c, -1)
        feat = torch.bmm(rgb_v, mask.permute(0, 2, 1)) # bz, c, hw
        feat = feat.view(bz, c, h, w)

        return feat

class CMAT(nn.Module):
    def __init__(self, in_channel, CA=True, ratio=8):
        super(CMAT, self).__init__()
        self.CA = CA

        # self.sa1 = SA(in_channel)
        # self.sa2 = SA(in_channel)
        self.sa = SpatialAttention()
        if self.CA:
            self.att1 = CrossAttention(in_channel, ratio=ratio)
            self.att2 = CrossAttention(in_channel, ratio=ratio)
        else:
            print("Warning: not use CrossAttention!")
            # self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)
            # self.conv3 = nn.Conv2d(256, 256, 3, 1, 1)
            self.conv2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
            self.conv3 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)

    def forward(self, rgb, depth, beta, gamma, gate):

        rgb = self.sa(rgb) * rgb
        depth = self.sa(depth) * rgb
        if self.CA:
            feat_1 = self.att1(rgb, depth)
            feat_2 = self.att2(depth, rgb)
        else:
            w1 = self.conv2(rgb)
            w2 = self.conv3(depth)
            feat_1 = F.relu(w2*rgb, inplace=True)
            feat_2 = F.relu(w1*depth, inplace=True)

        # out1 = rgb + gate * beta * feat_1
        # out2 = depth + (1.0-gate) * gamma * feat_2
        out1 = rgb + beta * feat_1
        out2 = depth + gamma * feat_2

        return out1, out2
