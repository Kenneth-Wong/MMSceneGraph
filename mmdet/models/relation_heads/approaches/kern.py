# ---------------------------------------------------------------
# kern.py
# Set-up time: 2021/4/7 9:58
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from mmcv.cnn import kaiming_init
from .motif_util import (obj_edge_vectors, center_x, sort_by_score, to_onehot,
                         get_dropout_mask, encode_box_info, block_orthogonal)
import numpy as np


class GGNNObj(nn.Module):
    def __init__(self, num_obj_classes, time_step, hidden_dim, output_dim, prior_matrix):
        super(GGNNObj, self).__init__()
        self.num_obj_classes = num_obj_classes
        self.time_step = time_step
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.register_buffer('matrix', prior_matrix)

        self.fc_eq3_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq3_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq4_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq5_u = nn.Linear(hidden_dim, hidden_dim)

        self.fc_output = nn.Linear(2 * hidden_dim, output_dim)
        self.ReLU = nn.ReLU(True)
        self.fc_obj_cls = nn.Linear(self.num_obj_classes * output_dim, self.num_obj_classes)

    def forward(self, input_ggnn):
        # propogation process
        num_object = input_ggnn.size()[0]
        hidden = input_ggnn.repeat(1, self.num_obj_classes).view(num_object, self.num_obj_classes, -1)
        for t in range(self.time_step):
            # eq(2)
            # here we use some matrix operation skills
            hidden_sum = torch.sum(hidden, 0)
            av = torch.cat(
                [torch.cat([self.matrix.transpose(0, 1) @ (hidden_sum - hidden_i) for hidden_i in hidden], 0),
                 torch.cat([self.matrix @ (hidden_sum - hidden_i) for hidden_i in hidden], 0)], 1)

            # eq(3)
            hidden = hidden.view(num_object * self.num_obj_classes, -1)
            zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(hidden))

            # eq(4)
            rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq4_u(hidden))

            # eq(5)
            hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * hidden))

            hidden = (1 - zv) * hidden + zv * hv
            hidden = hidden.view(num_object, self.num_obj_classes, -1)

        output = torch.cat((hidden.view(num_object * self.num_obj_classes, -1),
                            input_ggnn.repeat(1, self.num_obj_classes).view(num_object * self.num_obj_classes, -1)), 1)
        output = self.fc_output(output)
        output = self.ReLU(output)
        obj_dists = self.fc_obj_cls(output.view(-1, self.num_obj_classes * self.output_dim))
        return obj_dists


class GGNNRel(nn.Module):
    def __init__(self, num_rel_classes, time_step, hidden_dim, output_dim, prior_matrix):
        super(GGNNRel, self).__init__()
        self.num_rel_classes = num_rel_classes
        self.time_step = time_step
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.register_buffer('matrix', prior_matrix)

        self.fc_eq3_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq3_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq4_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq5_u = nn.Linear(hidden_dim, hidden_dim)

        self.fc_output = nn.Linear(2 * hidden_dim, output_dim)
        self.ReLU = nn.ReLU(True)
        self.fc_rel_cls = nn.Linear((self.num_rel_classes + 2) * output_dim, self.num_rel_classes)

    def forward(self, rel_pair_idxes, sub_obj_preds, input_ggnn, num_objs):
        (input_rel_num, node_num, _) = input_ggnn.size()
        batch_in_matrix_sub = torch.zeros((input_rel_num, 2, self.num_rel_classes)).float().to(input_ggnn.device)

        rel_inds = []
        acc_obj = 0
        for i, rel_idx in enumerate(rel_pair_idxes):
            rel_inds.append(rel_idx + acc_obj)
            acc_obj += num_objs[i]
        rel_inds = torch.cat(rel_inds, 0)

        # construct adjacency matrix depending on the predicted labels of subject and object.
        for index, rel in enumerate(rel_inds):
            batch_in_matrix_sub[index][0] = \
                self.matrix[sub_obj_preds[index, 0], sub_obj_preds[index, 1]]
            batch_in_matrix_sub[index][1] = batch_in_matrix_sub[index][0]

        hidden = input_ggnn
        for t in range(self.time_step):
            # eq(2)
            # becase in this case, A^(out) == A^(in), so we use function "repeat"
            # What is A^(out) and A^(in)? Please refer to paper "Gated graph sequence neural networks"
            av = torch.cat((torch.bmm(batch_in_matrix_sub, hidden[:, 2:]),
                            torch.bmm(batch_in_matrix_sub.transpose(1, 2), hidden[:, :2])), 1).repeat(1, 1, 2)
            av = av.view(input_rel_num * node_num, -1)
            flatten_hidden = hidden.view(input_rel_num * node_num, -1)
            # eq(3)
            zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(flatten_hidden))
            # eq(4)
            rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq4_u(flatten_hidden))
            # eq(5)
            hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * flatten_hidden))
            flatten_hidden = (1 - zv) * flatten_hidden + zv * hv
            hidden = flatten_hidden.view(input_rel_num, node_num, -1)

        output = torch.cat((flatten_hidden, input_ggnn.view(input_rel_num * node_num, -1)), 1)
        output = self.fc_output(output)
        output = self.ReLU(output)

        rel_dists = self.fc_rel_cls(output.view(input_rel_num, -1))
        return rel_dists


class GGNNObjReason(nn.Module):
    def __init__(self, config, obj_classes, obj_knowledge):
        super(GGNNObjReason, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.num_obj_classes = len(obj_classes)
        self.register_buffer('matrix', obj_knowledge)
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

        self.time_step = self.cfg.time_step
        self.obj_dim = self.cfg.roi_dim
        self.hidden_dim = self.cfg.hidden_dim

        self.obj_proj = nn.Linear(self.obj_dim, self.hidden_dim)
        self.ggnn_obj = GGNNObj(self.num_obj_classes, self.time_step, self.hidden_dim, self.hidden_dim, self.matrix)

    def forward(self, roi_feats, num_rois, obj_labels=None):
        if self.mode != 'predcls':
            input_ggnn = self.obj_proj(roi_feats)
            obj_dists = []
            split_roi_feats = input_ggnn.split(num_rois)
            for roi_feat in split_roi_feats:
                obj_dists.append(self.ggnn_obj(roi_feat))
            obj_dists = torch.cat(obj_dists, 0)
            obj_preds = obj_dists[:, 1:].max(1)[1] + 1
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        return obj_dists, obj_preds


class GGNNRelReason(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, rel_knowledge):
        super(GGNNRelReason, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.num_obj_classes = len(obj_classes)
        self.rel_classes = rel_classes
        self.num_rel_classes = len(rel_classes)
        self.register_buffer('matrix', rel_knowledge)
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

        self.time_step = self.cfg.time_step
        self.obj_dim = self.cfg.roi_dim
        self.rel_dim = self.cfg.roi_dim
        self.hidden_dim = self.cfg.hidden_dim

        self.obj_proj = nn.Linear(self.obj_dim, self.hidden_dim)
        self.rel_proj = nn.Linear(self.rel_dim, self.hidden_dim)
        self.ggnn_rel = GGNNRel(self.num_rel_classes, self.time_step, self.hidden_dim, self.hidden_dim, self.matrix)

    def forward(self, roi_feats, union_feats, num_rois, num_rels, obj_dists, obj_preds, rel_pair_idxes):
        roi_feats = self.obj_proj(roi_feats)
        union_feats = self.rel_proj(union_feats)

        pair_reps = []
        pair_preds = []
        split_roi_feats = roi_feats.split(num_rois)
        split_union_feats = union_feats.split(num_rels)
        obj_preds = obj_preds.split(num_rois)
        for pair_idx, obj_pred, roi_feat, union_feat in zip(rel_pair_idxes, obj_preds, split_roi_feats, split_union_feats):
            pair_reps.append(torch.cat((roi_feat[pair_idx[:, 0]],
                                        roi_feat[pair_idx[:, 1]],
                                        union_feat.repeat(1, self.num_rel_classes)
                                        ), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        pair_preds = torch.cat(pair_preds, dim=0)
        pair_reps = torch.cat(pair_reps, dim=0)
        pair_reps = torch.stack(pair_reps.view(-1, self.hidden_dim).split(2 + self.num_rel_classes))

        rel_dists = self.ggnn_rel(rel_pair_idxes, pair_preds, pair_reps, num_rois)

        return rel_dists

class KERNContext(nn.Module):
    """
    Modified from KERN (Chen et al. CVPR'19)
    """

    def __init__(self, config, obj_classes, rel_classes, obj_knowledge, rel_knowledge):
        super(KERNContext, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

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

        self.ggnn_obj_reason = GGNNObjReason(config, obj_classes, obj_knowledge)
        self.ggnn_rel_reason = GGNNRelReason(config, obj_classes, rel_classes, rel_knowledge)


    def init_weights(self):
        pass

    def forward(self, x, union_feats, det_result, all_average=False, ctx_average=False):
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.use_gt_box:  # predcls or sgcls or training, just put obj_labels here
            obj_labels = torch.cat(det_result.labels)
        else:
            obj_labels = None

        # object level contextual feature
        num_objs = [len(b) for b in det_result.bboxes]
        obj_dists, obj_preds = self.ggnn_obj_reason(x, num_objs, obj_labels)

        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        rel_dists = self.ggnn_rel_reason(x, union_feats, num_objs, num_rels, obj_dists, obj_preds,
                                         det_result.rel_pair_idxes)

        return obj_dists, obj_preds, rel_dists
