import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weighted_loss


@LOSSES.register_module
class SigmoidDRLoss(nn.Module):
    """SigmoidDR Loss

    Qian_DR_Loss_Improving_Object_Detection_by_Distributional_Ranking_CVPR_2020_paper.pdf (CVPR 2020)
    """

    def __init__(self,
                 pos_lambda=1,
                 neg_lambda=0.1 / np.log(3.5),
                 L=6.,
                 tau=4.,
                 reduction='mean',
                 loss_weight=1.0):
        super(SigmoidDRLoss, self).__init__()
        self.margin = 0.5
        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda
        self.L = L
        self.tau = tau

        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                preds,
                targets,
                weight=None,
                **kwargs):
        assert isinstance(preds, (list, tuple))
        assert isinstance(targets, (tuple, list))
        bs = len(preds)
        loss = 0
        for pred, target in zip(preds, targets):
            pos_ind = (target == 1)
            neg_ind = (target == 0)
            if len(neg_ind) == 0:
                neg_ind = (target == -1)
            pos_prob = pred[pos_ind].sigmoid()
            neg_prob = pred[neg_ind].sigmoid()
            neg_q = F.softmax(neg_prob/self.neg_lambda, dim=0)
            neg_dist = torch.sum(neg_q * neg_prob)
            if pos_prob.numel() > 0:
                pos_q = F.softmax(-pos_prob/self.pos_lambda, dim=0)
                pos_dist = torch.sum(pos_q * pos_prob)
                minus_term = pos_dist
            else:
                minus_term = 1.
            loss = loss + self.tau * torch.log(1. + torch.exp(self.L * (neg_dist - minus_term + self.margin))) / self.L
        loss = loss / bs
        return loss
