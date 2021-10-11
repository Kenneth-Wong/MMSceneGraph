# ---------------------------------------------------------------
# layers.py
# Set-up time: 2021/1/3 15:14
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from .misc import activation
import math

class Attention(nn.Module):
    def __init__(self, head_config, attention_config):
        super(Attention, self).__init__()
        self.head_config = head_config
        self.attention_config = attention_config
        self.Wah = nn.Linear(self.head_config.rnn_size, self.attention_config.att_hidden_size, bias=False)
        self.alpha = nn.Linear(self.attention_config.att_hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(self.attention_config.att_hidden_drop) if self.attention_config.att_hidden_drop > 0 else None

        self.act = activation(self.attention_config.att_act)

        self.need_attn = self.attention_config.need_attn

    # h -- batch_size * cfg.MODEL.RNN_SIZE
    # att_feats -- batch_size * att_num * att_feats_dim
    # p_att_feats -- batch_size * att_num * cfg.ATT_HIDDEN_SIZE
    # att_mask -- batch_size * att_num
    def forward(self, h, att_feats, att_mask, p_att_feats):
        Wah = self.Wah(h).unsqueeze(1)
        alpha = self.act(Wah + p_att_feats)
        if self.dropout is not None:
            alpha = self.dropout(alpha)
        alpha = self.alpha(alpha).squeeze(-1)  # batch_size * att_num
        if att_mask is not None:
            alpha = alpha.masked_fill(att_mask == 0, -1e9)  # put a large negative value, so that the exp() is almost 0
        alpha = F.softmax(alpha, dim=-1)
        att = torch.bmm(alpha.unsqueeze(1), att_feats).squeeze(1)
        if self.need_attn:
            return att, alpha
        else:
            return att


class BasicAtt(nn.Module):
    def __init__(self, mid_dims, mid_dropout):
        super(BasicAtt, self).__init__()

        sequential = []
        for i in range(1, len(mid_dims) - 1):
            sequential.append(nn.Linear(mid_dims[i - 1], mid_dims[i]))
            sequential.append(nn.ReLU())
            if mid_dropout > 0:
                sequential.append(nn.Dropout(mid_dropout))
        self.attention_basic = nn.Sequential(*sequential) if len(sequential) > 0 else None
        self.attention_last = nn.Linear(mid_dims[-2], mid_dims[-1])

    def forward(self, att_map, att_mask, value1, value2):
        if self.attention_basic is not None:
            att_map = self.attention_basic(att_map)
        attn_weights = self.attention_last(att_map)
        attn_weights = attn_weights.squeeze(-1)
        if att_mask is not None:
            attn_weights = attn_weights.masked_fill(att_mask.unsqueeze(1) == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn = torch.matmul(attn_weights.unsqueeze(-2), value2).squeeze(-2)
        return attn


class SCAtt(BasicAtt):
    def __init__(self, mid_dims, mid_dropout):
        super(SCAtt, self).__init__(mid_dims, mid_dropout)
        self.attention_last = nn.Linear(mid_dims[-2], 1)
        self.attention_last2 = nn.Linear(mid_dims[-2], mid_dims[-1])

    def forward(self, att_map, att_mask, value1, value2):
        if self.attention_basic is not None:
            att_map = self.attention_basic(att_map)

        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)
            att_mask_ext = att_mask.unsqueeze(-1)
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, -2)
        else:
            att_map_pool = att_map.mean(-2)

        alpha_spatial = self.attention_last(att_map)
        alpha_channel = self.attention_last2(att_map_pool)
        alpha_channel = torch.sigmoid(alpha_channel)

        alpha_spatial = alpha_spatial.squeeze(-1)
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask == 0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1)

        if len(alpha_spatial.shape) == 4: # batch_size * head_num * seq_num * seq_num (for xtransformer)
            value2 = torch.matmul(alpha_spatial, value2)
        else:
            value2 = torch.matmul(alpha_spatial.unsqueeze(-2), value2).squeeze(-2)

        attn = value1 * value2 * alpha_channel
        return attn

class LowRank(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, act):
        super(LowRank, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        output_dim = 2 * embed_dim if act == 'GLU' else embed_dim

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = activation(act)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_q = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = activation(act)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_k = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = activation(act)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v1 = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = activation(act)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v2 = nn.Sequential(*sequential)

        att_layer = {'scatt': SCAtt, 'basicatt': BasicAtt}
        self.attn_net = att_layer[att_type](att_mid_dim, att_mid_drop)
        self.clear_buffer()

    def apply_to_states(self, fn):
        self.buffer_keys = fn(self.buffer_keys)
        self.buffer_value2 = fn(self.buffer_value2)

    def init_buffer(self, batch_size):
        self.buffer_keys = torch.zeros((batch_size, self.num_heads, 0, self.head_dim)).cuda()
        self.buffer_value2 = torch.zeros((batch_size, self.num_heads, 0, self.head_dim)).cuda()

    def clear_buffer(self):
        self.buffer_keys = None
        self.buffer_value2 = None

    # query -- batch_size * qdim
    # value -- batch_size * att_num * vdim
    def forward(self, query, key, mask, value1, value2, precompute=False):
        batch_size = query.size()[0]
        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)

        q = q.view(batch_size, self.num_heads, self.head_dim)
        v1 = v1.view(batch_size, self.num_heads, self.head_dim)

        if precompute == False:
            key = key.view(-1, key.size()[-1])
            value2 = value2.view(-1, value2.size()[-1])
            k = self.in_proj_k(key)
            v2 = self.in_proj_v2(value2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k = key
            v2 = value2

        attn_map = q.unsqueeze(-2) * k
        attn = self.attn_net(attn_map, mask, v1, v2)
        attn = attn.view(batch_size, self.num_heads * self.head_dim)
        return attn

    # query -- batch_size * seq_num * qdim
    # value -- batch_size * att_num * vdim
    def forward2(self, query, key, mask, value1, value2, precompute=False):
        batch_size = query.size()[0]
        query = query.view(-1, query.size()[-1])
        value1 = value1.view(-1, value1.size()[-1])

        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = v1.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if precompute == False:
            key = key.view(-1, key.size()[-1])
            value2 = value2.view(-1, value2.size()[-1])
            k = self.in_proj_k(key)
            v2 = self.in_proj_v2(value2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            if self.buffer_keys is not None and self.buffer_value2 is not None:
                self.buffer_keys = torch.cat([self.buffer_keys, k], dim=2)
                self.buffer_value2 = torch.cat([self.buffer_value2, v2], dim=2)
                k = self.buffer_keys
                v2 = self.buffer_value2
        else:
            k = key
            v2 = value2

        attn_map = q.unsqueeze(-2) * k.unsqueeze(-3)
        attn = self.attn_net.forward(attn_map, mask, v1, v2).transpose(1, 2).contiguous()
        attn = attn.view(batch_size, -1, self.num_heads * self.head_dim)
        return attn

    def precompute(self, key, value2):
        batch_size = value2.size()[0]
        key = key.view(-1, key.size()[-1])
        value2 = value2.view(-1, value2.size()[-1])

        k = self.in_proj_k(key)
        v2 = self.in_proj_v2(value2)

        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        return k, v2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(max_len * 2.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x_size):
        return self.pe[:, :x_size]


