# -*- coding: utf-8 -*-
# @Time    : 2022/11/9 10:13
# @Author  : Linhui Dai
# @FileName: crisscross_attention.py
# @Software: PyCharm

import torch
from torch.nn import Softmax
import torch.nn as nn

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        m_batchsize, channel, height, width = x.size()
        # x.shape [4, 256, 96, 168]
        # proj_query_x.shape = [4, 32, 96, 168]
        proj_query_x = self.query_conv(x)
        # proj_query_H.shape = [672, 96, 32] [bz*width, height, channel]
        proj_query_H = proj_query_x.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        # proj_query_W.shape = [384, 168, 32]
        proj_query_W = proj_query_x.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        # proj_key_x.shape = [4, 32, 96, 168]
        proj_key_x = self.key_conv(y)
        # proj_key_H.shape = [672, 32, 96]
        proj_key_H = proj_key_x.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        # proj_key_W.shape = [384, 32, 168]
        proj_key_W = proj_key_x.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value_y = self.value_conv(y)
        proj_value_H = proj_value_y.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value_y.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x
        # return self.gamma * (out_H + out_W) + x + self.beta * y
        # return x + z * y
        # return x + self.beta * y

        # # enhance vk ori q
        # ori_q = self.query_conv(x).view(m_batchsize, -1, height*width).permute(0, 2, 1)
        # enhance_k = self.key_conv(y).view(m_batchsize, -1, height*width)
        # mask = torch.bmm(ori_q, enhance_k)
        # mask = torch.softmax(mask, dim=-1)
        # enhance_v = self.value_conv(y).view(m_batchsize, channel, -1)
        # feat = torch.bmm(enhance_v, mask.permute(0, 2, 1))
        # feat = feat.view(m_batchsize, channel, height, width)
        #
        # return self.beta * y + x

