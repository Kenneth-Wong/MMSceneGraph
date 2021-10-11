# ---------------------------------------------------------------
# vtranse.py
# Set-up time: 2020/6/4 下午10:02
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from mmcv.cnn import xavier_init, kaiming_init
from torch.nn import functional as F
from .motif_util import obj_edge_vectors, encode_box_info, to_onehot


class VTransEContext(nn.Module):
    def __init__(self, config, obj_classes, rel_classes):
        super(VTransEContext, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        in_channels = self.cfg.roi_dim

        self.use_gt_box = self.cfg.use_gt_box
        self.use_gt_label = self.cfg.use_gt_label

        # mode
        if self.cfg.use_gt_box:
            if self.cfg.use_gt_label:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.embed_dim = self.cfg.embed_dim
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.glove_dir, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.dropout_rate
        self.hidden_dim = self.cfg.hidden_dim

        self.pred_layer = nn.Linear(self.obj_dim + self.embed_dim + 128, self.num_obj_classes)
        self.fc_layer = nn.Linear(self.obj_dim + self.embed_dim + 128, self.hidden_dim)


    def init_weights(self):
        for m in self.pos_embed:
            if isinstance(m, nn.Linear):
                kaiming_init(m, distribution='uniform', a=1)
        kaiming_init(self.pred_layer, a=1, distribution='uniform')
        kaiming_init(self.fc_layer, a=1, distribution='uniform')

    def forward(self, x, det_result, all_average=False, ctx_average=False):
        num_objs = [len(b) for b in det_result.bboxes]
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.use_gt_box:
            obj_labels = torch.cat(det_result.labels)
        else:
            obj_labels = None

        if self.cfg.use_gt_label:
            obj_embed = self.obj_embed1(obj_labels.long())
        else:
            obj_dists = torch.cat(det_result.dists, dim=0).detach()
            obj_embed = obj_dists @ self.obj_embed1.weight

        pos_embed = self.pos_embed(encode_box_info(det_result))

        batch_size = x.shape[0]

        obj_pre_rep = torch.cat((x, obj_embed, pos_embed), -1)

        # object level contextual feature
        if self.mode != 'predcls':
            obj_scores = self.pred_layer(obj_pre_rep)
            obj_dists = F.softmax(obj_scores, dim=1)
            obj_preds = obj_dists[:, 1:].max(1)[1] + 1
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_scores = to_onehot(obj_preds, self.num_obj_classes)

        # edge level contextual feature
        obj_embed2 = self.obj_embed2(obj_preds.long())
        obj_rel_rep = torch.cat((x, pos_embed, obj_embed2), -1)

        edge_ctx = F.relu(self.fc_layer(obj_rel_rep))

        return obj_scores, obj_preds, edge_ctx, None
