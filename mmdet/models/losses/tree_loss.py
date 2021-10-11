# ---------------------------------------------------------------
# tree_loss.py
# Set-up time: 2020/10/19 16:52
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss
import anytree

def tree_cross_entropy(preds, labels, hierarchy, vocab, weight=None, info=None):
    loss = 0
    if info is not None:
        for pred, label in zip(preds, labels):
            label_int = label.item()
            levels = info[label_int]
            K = len(levels)
            loss_i = 0
            for level in levels:
                collect_pred = pred[level['same_level']][None]
                level_label = torch.Tensor([level['local_label']]).to(label)
                loss_i_level_k = F.cross_entropy(collect_pred, level_label, reduction='none')
                if weight is not None:
                    loss_i_level_k = weight[label_int] * loss_i_level_k
                loss_i += loss_i_level_k
            loss_i /= K
            loss += loss_i
        loss /= preds.shape[0]
    else:
        for pred, label in zip(preds, labels):
            label_int = label.item()
            K = 0
            node = anytree.search.find(hierarchy, lambda x: x.id == vocab[label_int])
            #node = hierarchy.children[0]
            loss_i = 0
            while node.parent is not None:
                all_children = node.parent.children
                c_fake_label = 0 #-1
                collect_pred = []
                target_weight = None
                for cidx, c in enumerate(all_children):
                    if c.id == node.id:
                        c_fake_label = cidx
                        if weight is not None:
                            target_weight = weight[vocab.index(c.id)].item()
                    collect_pred.append(pred[vocab.index(c.id)][None])
                collect_pred = torch.cat(collect_pred)[None]
                fake_label = torch.Tensor([c_fake_label]).to(labels)
                loss_i_level_k = F.cross_entropy(collect_pred, fake_label, reduction='none')
                if target_weight is not None:
                    loss_i_level_k = target_weight * loss_i_level_k
                loss_i += loss_i_level_k
                node = node.parent
                K += 1
            loss_i /= K
            loss += loss_i
        loss /= preds.shape[0]
    return loss


@LOSSES.register_module
class TreeLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 info=None):
        super(TreeLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.info = info

    def forward(self,
                cls_score,
                label,
                hierarchy,
                vocab,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * tree_cross_entropy(
            cls_score,
            label,
            hierarchy,
            vocab,
            weight=weight,
            info=self.info)
        return loss_cls