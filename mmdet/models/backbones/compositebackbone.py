import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import VGG, constant_init, kaiming_init, normal_init, xavier_init
from mmcv.runner import load_checkpoint

from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from .resnet import ResNet
from mmcv.runner import BaseModule
import warnings

@BACKBONES.register_module()
class CCB(VGG, BaseModule):
    """VGG Backbone network for single-shot-detection.

    Args:
        input_size (int): width and height of input, from {300, 512}.
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        out_indices (Sequence[int]): Output from which stages.

    Example:
        >>> self = CCB(input_size=300, depth=11)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 300, 300)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 1024, 19, 19)
        (1, 512, 10, 10)
        (1, 256, 5, 5)
        (1, 256, 3, 3)
        (1, 256, 1, 1)
    """
    extra_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256),
        512: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128),
    }

    def __init__(self,
                 input_size,
                 depth,
                 with_last_pool=False,
                 ceil_mode=True,
                 out_indices=(3, 4),
                 out_feature_indices=(22, 34),
                 pretrained=None,
                 init_cfg=None,
                 l2_norm_scale=20.):
        # TODO: in_channels for mmcv.VGG
        super(CCB, self).__init__(
            depth,
            with_last_pool=with_last_pool,
            ceil_mode=ceil_mode,
            out_indices=out_indices)
        assert input_size in (300, 512)
        self.input_size = input_size

        self.features.add_module(
            str(len(self.features)),
            RFAM(512, 0.1) )
        self.features.add_module(
            str(len(self.features)),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        self.features.add_module(
            str(len(self.features)), nn.Conv2d(1024, 1024, kernel_size=1))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        self.out_feature_indices = out_feature_indices

        self.inplanes = 1024
        self.extra = self._make_extra_layers(self.extra_setting[input_size])
        self.l2_norm = L2Norm(
            self.features[out_feature_indices[0] - 1].out_channels,
            l2_norm_scale)


        self.resnet50 = ResNet(depth=50, init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))

        if init_cfg is not None:
            self.init_cfg = init_cfg
        elif isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(type='Constant', val=1, layer='BatchNorm2d'),
                dict(type='Normal', std=0.01, layer='Linear'),
            ]
        else:
            raise TypeError('pretrained must be a str or None')

        self.chaAdj = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.Conv2d(384, 128, kernel_size=1), nn.Conv2d(768, 256, kernel_size=1))
        self.RFAM_PRO = RFAM_PRO(512, 0.1)
        self.RFAMs = nn.Sequential( RFAM(512, 0.1), RFAM(1024, 0.1), RFAM(512, 0.1))


    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

        elif pretrained is None:
            for m in self.features.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        constant_init(self.l2_norm, self.l2_norm.scale)
        # self.resnet50.init_weights('torchvision://resnet50')


    def forward(self, x):
        """Forward function."""
        # self.resnet5
        outs = []
        resnet_feat = []
        feat = self.resnet50.conv1(x)
        feat = self.resnet50.norm1(feat)
        feat = self.resnet50.relu(feat)
        resnet_feat.append(feat)
        feat = self.resnet50.maxpool(feat)
        feat = self.resnet50.layer1(feat)
        resnet_feat.append(feat)
        feat = self.resnet50.layer2(feat)
        resnet_feat.append(feat)
        count = 0
        rfam_count = 0
        for i, layer in enumerate(self.features):
            x = layer(x)
            if type(layer) == nn.MaxPool2d and count < 3:
                x = torch.cat((x,resnet_feat[count]), dim=1)
                x = self.chaAdj[count](x)
                count += 1
            if i in self.out_feature_indices:
                if i == 22:
                    outs.append(self.RFAM_PRO(x))
                else:
                    outs.append(x)
                x = self.RFAMs[rfam_count](x)
                rfam_count += 1
        for i, layer in enumerate(self.extra):
            x = F.relu(layer(x), inplace=True)
            if i % 2 == 1:
                outs.append(x)
                if rfam_count < 3:
                    x = self.RFAMs[rfam_count](x)
                    rfam_count += 1
        outs[0] = self.l2_norm(outs[0])
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def _make_extra_layers(self, outplanes):
        layers = []
        kernel_sizes = (1, 3)
        num_layers = 0
        outplane = None
        for i in range(len(outplanes)):
            if self.inplanes == 'S':
                self.inplanes = outplane
                continue
            k = kernel_sizes[num_layers % 2]
            if outplanes[i] == 'S':
                outplane = outplanes[i + 1]
                conv = nn.Conv2d(
                    self.inplanes, outplane, k, stride=2, padding=1)
            else:
                outplane = outplanes[i]
                conv = nn.Conv2d(
                    self.inplanes, outplane, k, stride=1, padding=0)
            layers.append(conv)
            self.inplanes = outplanes[i]
            num_layers += 1
        if self.input_size == 512:
            layers.append(nn.Conv2d(self.inplanes, 256, 4, padding=1))

        return nn.Sequential(*layers)



class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20., eps=1e-10):
        """L2 normalization layer.

        Args:
            n_dims (int): Number of dimensions to be normalized
            scale (float, optional): Defaults to 20..
            eps (float, optional): Used to avoid division by zero.
                Defaults to 1e-10.
        """
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        """Forward function."""
        # normalization layer convert to FP32 in FP16 training
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) *
                x_float / norm).type_as(x)

# class RFAM(nn.Module):
#     def __init__(self, indim, scale):
#         super(RFAM, self).__init__()
#         embdim = indim//4
#         self.branch1 = nn.Sequential(nn.Conv2d(indim, embdim, kernel_size=1), nn.Conv2d(embdim, embdim, kernel_size=3, padding=1), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
#         self.branch2 = nn.Sequential(nn.Conv2d(indim, embdim, kernel_size=1), nn.Conv2d(embdim, embdim, kernel_size=3, stride=2, padding=1), nn.Conv2d(embdim, embdim, kernel_size=5, dilation=3, padding=6))
#         self.branch3 = nn.Sequential(nn.Conv2d(indim, embdim, kernel_size=1),
#                                      nn.Conv2d(embdim, embdim, kernel_size=5, stride=2, padding=2),
#                                      nn.Conv2d(embdim, embdim, kernel_size=3, dilation=5, padding=5))
#         self.conv = nn.Conv2d(3*embdim, indim, kernel_size = 1)
#         self.relu = nn.ReLU()
#         self.scale = scale
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#     def forward(self, x):
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#         x3 = self.branch3(x)
#         x = self.maxpool(x)
#         res = torch.cat((x1,x2,x3), dim = 1)
#         res = self.conv(res)
#         x = x + (res * self.scale)
#         x = self.relu(x)
#         return x

class RFAM(nn.Module):
    def __init__(self, indim, scale):
        super(RFAM, self).__init__()
        embdim = indim//4
        self.branch1 = nn.Sequential(nn.Conv2d(indim, embdim, kernel_size=1), nn.Conv2d(embdim, embdim, kernel_size=3, padding=1))
        self.branch2 = nn.Sequential(nn.Conv2d(indim, embdim, kernel_size=1), nn.Conv2d(embdim, embdim, kernel_size=3, padding=1), nn.Conv2d(embdim, embdim, kernel_size=5, dilation=3, padding=6))
        self.branch3 = nn.Sequential(nn.Conv2d(indim, embdim, kernel_size=1),
                                     nn.Conv2d(embdim, embdim, kernel_size=5, padding=3),
                                     nn.Conv2d(embdim, embdim, kernel_size=3, dilation=5, padding=4))
        self.conv = nn.Conv2d(3*embdim, indim, kernel_size = 1)
        self.relu = nn.ReLU()
        self.scale = scale

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        res = torch.cat((x1,x2,x3), dim = 1)
        res = self.conv(res)
        x = x + (res * self.scale)
        x = self.relu(x)
        return x

class RFAM_PRO(nn.Module):
    def __init__(self, indim, scale):
        super(RFAM_PRO, self).__init__()
        embdim = indim // 4
        self.branch1 = nn.Sequential(nn.Conv2d(indim, embdim, kernel_size=1),
                                     nn.Conv2d(embdim, embdim, kernel_size=3, padding=1))
        self.branch2 = nn.Sequential(nn.Conv2d(indim, embdim, kernel_size=1),
                                     nn.Conv2d(embdim, embdim, kernel_size=(3,1), padding=(1,0)),
                                     nn.Conv2d(embdim, embdim, kernel_size=3, dilation=3, padding=3))
        self.branch3 = nn.Sequential(nn.Conv2d(indim, embdim, kernel_size=1),
                                     nn.Conv2d(embdim, embdim, kernel_size=(1,3), padding=(0,1)),
                                     nn.Conv2d(embdim, embdim, kernel_size=3, dilation=3, padding=3))
        self.branch4 = nn.Sequential(nn.Conv2d(indim, embdim, kernel_size=1),
                                     nn.Conv2d(embdim, embdim, kernel_size=(1,3), padding=(0,1)),
                                     nn.Conv2d(embdim, embdim, kernel_size=(3, 1), padding=(1, 0)),
                                     nn.Conv2d(embdim, embdim, kernel_size=3, dilation=5, padding=5))
        self.conv = nn.Conv2d(4 * embdim, indim, kernel_size=1)
        self.relu = nn.ReLU()
        self.scale = scale

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4  = self.branch4(x)
        res = torch.cat((x1, x2, x3, x4), dim=1)
        res = self.conv(res)
        x = x + (res * self.scale)
        x = self.relu(x)
        return x

if __name__ == '__main__':
    x = torch.rand(8,512,32,32)
    conv = RFAM(512,0.1)
    x = conv(x)