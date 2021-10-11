# ---------------------------------------------------------------
# r3net_saliency_detector.py
# Set-up time: 2021/5/12 11:10
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


@SALIENCY_DETECTORS.register_module
class R3NetSaliencyDetector(BaseSalineyDetector):
    """
    The R3Net saliency detector, please refer to:
    R3Net: Recurrent Residual Refinement Network for Saliency Detection, IJCAI 2018.
    """
    def __init__(self, backbone, pretrained=None, eval_mode=False):
        super(R3NetSaliencyDetector, self).__init__(eval_mode)
        self.backbone = builder.build_backbone(backbone)

        if not self.eval_mode:
            self.loss = nn.BCEWithLogitsLoss()

        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

        self.reduce_low = nn.Sequential(
            nn.Conv2d(64 + 256 + 512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce_high = nn.Sequential(
            nn.Conv2d(1024 + 2048, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            _ASPP(256)
        )

        self.predict0 = nn.Conv2d(256, 1, kernel_size=1)
        self.predict1 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict2 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict3 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict4 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict5 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict6 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if not self.eval_mode:
            self.backbone.init_weights(pretrained=pretrained)
        else:
            super(R3NetSaliencyDetector, self).init_weights(pretrained)

    def forward_core(self, x):
        layer0 = self.backbone.conv1(x)
        layer0 = self.backbone.norm1(layer0)
        layer0 = self.backbone.relu(layer0)
        layer0 = self.backbone.maxpool(layer0)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        l0_size = layer0.size()[2:]
        reduce_low = self.reduce_low(torch.cat((
            layer0,
            F.upsample(layer1, size=l0_size, mode='bilinear', align_corners=True),
            F.upsample(layer2, size=l0_size, mode='bilinear', align_corners=True)), 1))
        reduce_high = self.reduce_high(torch.cat((
            layer3,
            F.upsample(layer4, size=layer3.size()[2:], mode='bilinear', align_corners=True)), 1))
        reduce_high = F.upsample(reduce_high, size=l0_size, mode='bilinear', align_corners=True)

        predict0 = self.predict0(reduce_high)
        predict1 = self.predict1(torch.cat((predict0, reduce_low), 1)) + predict0
        predict2 = self.predict2(torch.cat((predict1, reduce_high), 1)) + predict1
        predict3 = self.predict3(torch.cat((predict2, reduce_low), 1)) + predict2
        predict4 = self.predict4(torch.cat((predict3, reduce_high), 1)) + predict3
        predict5 = self.predict5(torch.cat((predict4, reduce_low), 1)) + predict4
        predict6 = self.predict6(torch.cat((predict5, reduce_high), 1)) + predict5

        predict0 = F.upsample(predict0, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict4 = F.upsample(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict5 = F.upsample(predict5, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict6 = F.upsample(predict6, size=x.size()[2:], mode='bilinear', align_corners=True)

        return predict0, predict1, predict2, predict3, predict4, predict5, predict6

    def forward_train(self, x, img_meta, gt_maps):
        predict0, predict1, predict2, predict3, predict4, predict5, predict6 = self.forward_core(x)
        losses = dict()
        gt_maps = gt_maps.float()
        losses['loss0'] = self.loss(predict0, gt_maps)
        losses['loss1'] = self.loss(predict1, gt_maps)
        losses['loss2'] = self.loss(predict2, gt_maps)
        losses['loss3'] = self.loss(predict3, gt_maps)
        losses['loss4'] = self.loss(predict4, gt_maps)
        losses['loss5'] = self.loss(predict5, gt_maps)
        losses['loss6'] = self.loss(predict6, gt_maps)
        return losses

    def forward_test(self, x, img_meta):
        _, _, _, _, _, _, predict6 = self.forward_core(x)
        return F.sigmoid(predict6)

class _ASPP(nn.Module):
    #  this module is proposed in deeplabv3 and we use it in all of our baselines
    def __init__(self, in_dim):
        super(_ASPP, self).__init__()
        down_dim = in_dim // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear',
                           align_corners=True)
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))
