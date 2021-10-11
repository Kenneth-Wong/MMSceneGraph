# ---------------------------------------------------------------
# transformer.py
# Set-up time: 2021/3/26 11:21
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
import copy
import math
from .motif_util import obj_edge_vectors, encode_box_info, to_onehot

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, num_objs):
        "Pass the input (and mask) through each layer in turn."
        split_x = x.split(num_objs, 0)

        max_num = max(num_objs)
        num_atts = torch.LongTensor(num_objs).to(x.device)
        bs = len(num_objs)
        mask = torch.arange(0, max_num, device=x.device).long().unsqueeze(0).expand(bs, max_num).lt(
            num_atts.unsqueeze(1)).long().unsqueeze(-2)   # B * 1 * MAXN
        padded_x = nn.utils.rnn.pad_sequence(split_x, batch_first=True)   # B * MAXN * D
        for layer in self.layers:
            padded_x = layer(padded_x, mask)
        padded_x = self.norm(padded_x)

        # restore:
        return torch.cat([padded_x[i, :n] for i, n in enumerate(num_objs)], 0)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TransformerContext(nn.Module):
    """
    Modified from neural-motifs to encode contexts for each objects
    """

    def __init__(self, config, obj_classes, rel_classes):
        super(TransformerContext, self).__init__()
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
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.glove_dir, wv_dim=self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1)])

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.dropout_rate
        self.hidden_dim = self.cfg.hidden_dim
        self.nl_obj = self.cfg.context_object_layer
        self.nl_edge = self.cfg.context_edge_layer
        assert self.nl_obj > 0 and self.nl_edge > 0
        # transformer
        self.num_head = self.cfg.num_head
        self.inner_dim = self.cfg.inner_dim
        self.k_dim = self.cfg.k_dim
        self.v_dim = self.cfg.v_dim

        self.lin_obj = nn.Linear(self.obj_dim + self.embed_dim + 128, self.hidden_dim)
        self.lin_edge = nn.Linear(self.obj_dim + self.embed_dim + self.hidden_dim, self.hidden_dim)
        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_classes)

        # make model
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_head, self.hidden_dim, self.dropout_rate)
        ff = PositionwiseFeedForward(self.hidden_dim, self.inner_dim, self.dropout_rate)
        self.context_obj = Encoder(EncoderLayer(self.hidden_dim, c(attn), c(ff), self.dropout_rate), self.nl_obj)
        self.context_edge = Encoder(EncoderLayer(self.hidden_dim, c(attn), c(ff), self.dropout_rate), self.nl_edge)

    def init_weights(self):
        for m in self.pos_embed:
            if isinstance(m, nn.Linear):
                kaiming_init(m, distribution='uniform', a=1)
        kaiming_init(self.lin_obj, distribution='uniform', a=1)
        kaiming_init(self.lin_edge, distribution='uniform', a=1)

    def forward(self, x, det_result):
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.use_gt_box:  # predcls or sgcls or training, just put obj_labels here
            obj_labels = torch.cat(det_result.labels)
        else:
            obj_labels = None

        if self.use_gt_label:  # predcls
            obj_embed = self.obj_embed1(obj_labels.long())
        else:
            obj_dists = torch.cat(det_result.dists, dim=0).detach()
            obj_embed = obj_dists @ self.obj_embed1.weight

        pos_embed = self.pos_embed(encode_box_info(det_result))  # N x 128

        batch_size = x.shape[0]
        num_objs = [len(b) for b in det_result.bboxes]
        obj_pre_rep = torch.cat((x, obj_embed, pos_embed), -1)  # N x (1024 + 200 + 128)
        obj_pre_rep = self.lin_obj(obj_pre_rep)  # N x hidden_dim
        obj_feats = self.context_obj(obj_pre_rep, num_objs)

        if self.mode != 'predcls':
            obj_dists = self.out_obj(obj_feats)
            obj_preds = obj_dists[:, 1:].max(1)[1] + 1
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        obj_embed2 = self.obj_embed2(obj_preds.long())
        edge_pre_rep = torch.cat((obj_embed2, x, obj_feats), -1)
        edge_pre_rep = self.lin_edge(edge_pre_rep)
        edge_ctx = self.context_edge(edge_pre_rep, num_objs)

        return obj_dists, obj_preds, edge_ctx
