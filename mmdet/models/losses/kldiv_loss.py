# ---------------------------------------------------------------
# kldiv_loss.py
# Set-up time: 2021/5/31 15:51
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weighted_loss

def kldiv_loss(
        preds,
        targets,
):
    if isinstance(preds, (list, tuple)):
        sum = 0
        bs = 0
        for pred, target in zip(preds, targets):
            sum = sum + F.kl_div(F.log_softmax(pred, dim=-1), target, reduction='none').sum(-1)
            bs += pred.size(0)
        sum = sum / bs
        return sum
    else:
        loss = F.kl_div(F.log_softmax(preds, dim=-1), targets, reduction='batchmean')
        return loss



@LOSSES.register_module
class KLDivLoss(nn.Module):
    """KL Div Loss

    """

    def __init__(self,
                 reduction='none',
                 loss_weight=1.0):
        super(KLDivLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        loss = self.loss_weight * kldiv_loss(
            pred,
            target)
        return loss