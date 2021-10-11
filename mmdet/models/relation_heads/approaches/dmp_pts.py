# ---------------------------------------------------------------
# dmp.py
# Set-up time: 2020/10/7 22:23
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
from .motif_util import obj_edge_vectors, encode_box_info, to_onehot

def matmul(tensor3d, mat):
    out = []
    for i in range(tensor3d.size(-1)):
        out.append(torch.mm(tensor3d[:, :, i], mat))
    return torch.cat(out, -1)

class DirectionAwareMessagePassingPTS(nn.Module):
    """Adapted from the [CVPR 2020] GPS-Net: Graph Property Scensing Network for Scene Graph Generation]
    This version is for point message passing, which does not consider object semantic information, and
    does not predict the object categories.
    """
    def __init__(self, config, obj_classes):
        super(DirectionAwareMessagePassingPTS, self).__init__()
        self.cfg = config
        in_channels = self.cfg.roi_dim

        self.obj_dim = in_channels
        self.obj_input_dim = self.obj_dim

        # set the direction-aware attention mapping
        self.ws = nn.Linear(self.obj_dim, self.obj_dim)
        self.wo = nn.Linear(self.obj_dim, self.obj_dim)
        self.wu = nn.Linear(self.obj_dim, self.obj_dim)
        self.w = nn.Linear(self.obj_dim, 1)

        # now begin to set the DMP
        self.project_input = nn.Sequential(*[nn.Linear(self.obj_input_dim, self.obj_dim), nn.ReLU(inplace=True)])
        self.trans = nn.Sequential(*[nn.Linear(self.obj_dim, self.obj_dim // 4), nn.LayerNorm(self.obj_dim // 4),
                                     nn.ReLU(inplace=True), nn.Linear(self.obj_dim // 4, self.obj_dim)])
        self.W_t3 = nn.Sequential(*[nn.Linear(self.obj_dim, self.obj_dim // 2), nn.ReLU(inplace=True)])

    def get_attention(self, obj_feat, union_feat, rel_pair_idx):
        num_obj = obj_feat.shape[0]
        atten_coeff = self.w(self.ws(obj_feat[rel_pair_idx[:, 0]]) * self.wo(obj_feat[rel_pair_idx[:, 1]]) *
                             self.wu(union_feat))
        atten_tensor = torch.zeros(num_obj, num_obj, 1).to(atten_coeff)
        atten_tensor[rel_pair_idx[:, 0], rel_pair_idx[:, 1]] += atten_coeff
        atten_tensor = F.sigmoid(atten_tensor)
        atten_tensor = atten_tensor * (1 - torch.eye(num_obj).unsqueeze(-1).to(atten_tensor))
        return atten_tensor / torch.sum(atten_tensor, dim=1, keepdim=True)

    def forward(self, obj_feats, union_feats, det_result):

        obj_rep = obj_feats  # N x (1024 )

        rel_pair_idxes = det_result.rel_pair_idxes
        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]
        neighbour_feats = []
        split_obj_rep = obj_rep.split(num_objs)
        split_union_rep = union_feats.split(num_rels)
        for obj_feat, union_feat, rel_pair_idx in zip(split_obj_rep, split_union_rep, rel_pair_idxes):
            atten_tensor = self.get_attention(obj_feat, union_feat, rel_pair_idx)  # N x N x 1
            atten_tensor_t = torch.transpose(atten_tensor, 1, 0)
            atten_tensor = torch.cat((atten_tensor, atten_tensor_t), dim=-1)   # N x N x 2
            context_feats = matmul(atten_tensor, self.W_t3(obj_feat))
            neighbour_feats.append(self.trans(context_feats))

        obj_context_rep = F.relu(obj_rep + torch.cat(neighbour_feats, 0), inplace=True)

        return obj_context_rep

