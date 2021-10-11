# ---------------------------------------------------------------
# scrn_saliency_detector.py
# Set-up time: 2021/5/14 23:08
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import torch
import numpy as np
import torch.nn as nn

from .. import builder
from ..registry import SALIENCY_DETECTORS
from .sal_base import BaseSalineyDetector
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class conv_upsample(nn.Module):
    def __init__(self, channel):
        super(conv_upsample, self).__init__()
        self.conv = BasicConv2d(channel, channel, 1)

    def forward(self, x, target):
        if x.size()[2:] != target.size()[2:]:
            x = self.conv(F.upsample(x, size=target.size()[2:], mode='bilinear', align_corners=True))
        return x


class DenseFusion(nn.Module):
    # Cross Refinement Unit
    def __init__(self, channel):
        super(DenseFusion, self).__init__()
        self.conv1 = conv_upsample(channel)
        self.conv2 = conv_upsample(channel)
        self.conv3 = conv_upsample(channel)
        self.conv4 = conv_upsample(channel)
        self.conv5 = conv_upsample(channel)
        self.conv6 = conv_upsample(channel)
        self.conv7 = conv_upsample(channel)
        self.conv8 = conv_upsample(channel)
        self.conv9 = conv_upsample(channel)
        self.conv10 = conv_upsample(channel)
        self.conv11 = conv_upsample(channel)
        self.conv12 = conv_upsample(channel)

        self.conv_f1 = nn.Sequential(
            BasicConv2d(5 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f2 = nn.Sequential(
            BasicConv2d(4 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f3 = nn.Sequential(
            BasicConv2d(3 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f4 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

        self.conv_f5 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f6 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f7 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f8 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

    def forward(self, x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4):
        x_sf1 = x_s1 + self.conv_f1(torch.cat((x_s1, x_e1,
                                               self.conv1(x_e2, x_s1),
                                               self.conv2(x_e3, x_s1),
                                               self.conv3(x_e4, x_s1)), 1))
        x_sf2 = x_s2 + self.conv_f2(torch.cat((x_s2, x_e2,
                                               self.conv4(x_e3, x_s2),
                                               self.conv5(x_e4, x_s2)), 1))
        x_sf3 = x_s3 + self.conv_f3(torch.cat((x_s3, x_e3,
                                               self.conv6(x_e4, x_s3)), 1))
        x_sf4 = x_s4 + self.conv_f4(torch.cat((x_s4, x_e4), 1))

        x_ef1 = x_e1 + self.conv_f5(x_e1 * x_s1 *
                                    self.conv7(x_s2, x_e1) *
                                    self.conv8(x_s3, x_e1) *
                                    self.conv9(x_s4, x_e1))
        x_ef2 = x_e2 + self.conv_f6(x_e2 * x_s2 *
                                    self.conv10(x_s3, x_e2) *
                                    self.conv11(x_s4, x_e2))
        x_ef3 = x_e3 + self.conv_f7(x_e3 * x_s3 *
                                    self.conv12(x_s4, x_e3))
        x_ef4 = x_e4 + self.conv_f8(x_e4 * x_s4)

        return x_sf1, x_sf2, x_sf3, x_sf4, x_ef1, x_ef2, x_ef3, x_ef4


class ConcatOutput(nn.Module):
    def __init__(self, channel):
        super(ConcatOutput, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.output = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

    def forward(self, x1, x2, x3, x4):
        x3 = torch.cat((x3, self.conv_upsample1(self.upsample(x4))), 1)
        x3 = self.conv_cat1(x3)

        x2 = torch.cat((x2, self.conv_upsample2(self.upsample(x3))), 1)
        x2 = self.conv_cat2(x2)

        x1 = torch.cat((x1, self.conv_upsample3(self.upsample(x2))), 1)
        x1 = self.conv_cat3(x1)

        x = self.output(x1)
        return x

@SALIENCY_DETECTORS.register_module
class SCRNSaliencyDetector(BaseSalineyDetector):
    """
    The R3Net saliency detector, please refer to:
    R3Net: Recurrent Residual Refinement Network for Saliency Detection, IJCAI 2018.
    """
    def __init__(self, backbone, channel=32, pretrained=None, eval_mode=False):
        super(SCRNSaliencyDetector, self).__init__(eval_mode)
        self.backbone = builder.build_backbone(backbone)

        if not self.eval_mode:
            self.loss = nn.BCEWithLogitsLoss()
            
        self.reduce_s1 = Reduction(256, channel)
        self.reduce_s2 = Reduction(512, channel)
        self.reduce_s3 = Reduction(1024, channel)
        self.reduce_s4 = Reduction(2048, channel)

        self.reduce_e1 = Reduction(256, channel)
        self.reduce_e2 = Reduction(512, channel)
        self.reduce_e3 = Reduction(1024, channel)
        self.reduce_e4 = Reduction(2048, channel)

        self.df1 = DenseFusion(channel)
        self.df2 = DenseFusion(channel)
        self.df3 = DenseFusion(channel)
        self.df4 = DenseFusion(channel)

        self.output_s = ConcatOutput(channel)
        self.output_e = ConcatOutput(channel)

        self.init_weights(pretrained=pretrained)


    def init_weights(self, pretrained=None):
        if not self.eval_mode:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(std=0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            self.backbone.init_weights(pretrained=pretrained)
        else:
            super(SCRNSaliencyDetector, self).init_weights(pretrained=pretrained)

    def forward_core(self, x):
        size = x.size()[2:]
        x = self.backbone.conv1(x)
        x = self.backbone.norm1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        # feature abstraction
        x_s1 = self.reduce_s1(x1)
        x_s2 = self.reduce_s2(x2)
        x_s3 = self.reduce_s3(x3)
        x_s4 = self.reduce_s4(x4)

        x_e1 = self.reduce_e1(x1)
        x_e2 = self.reduce_e2(x2)
        x_e3 = self.reduce_e3(x3)
        x_e4 = self.reduce_e4(x4)

        # four cross refinement units
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df1(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df2(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df3(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df4(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)

        # feature aggregation using u-net
        pred_s = self.output_s(x_s1, x_s2, x_s3, x_s4)
        pred_e = self.output_e(x_e1, x_e2, x_e3, x_e4)

        pred_s = F.upsample(pred_s, size=size, mode='bilinear', align_corners=True)
        pred_e = F.upsample(pred_e, size=size, mode='bilinear', align_corners=True)

        return pred_s, pred_e

    def forward_train(self, x, img_meta, gt_maps):
        pred_s, pred_e = self.forward_core(x)
        losses = dict()

        edge_gt_maps = gt_maps.gt(0.5).float()
        edge_gt_maps = F.pad(edge_gt_maps, [1, 1, 1, 1], mode='replicate')
        fx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().to(x.device).view(1, 1, 3, 3)
        fy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().to(x.device).view(1, 1, 3, 3)
        contour_th = 1.5
        label_fx = F.conv2d(edge_gt_maps, fx)
        label_fy = F.conv2d(edge_gt_maps, fy)
        label_grad = torch.sqrt(torch.mul(label_fx, label_fx) + torch.mul(label_fy, label_fy))
        label_grad = torch.gt(label_grad, contour_th).float()

        losses['loss1'] = self.loss(pred_s, gt_maps)
        losses['loss2'] = self.loss(pred_e, label_grad)
        return losses

    def forward_test(self, x, img_meta):
        pred_s, pred_e = self.forward_core(x)
        return F.sigmoid(pred_s)