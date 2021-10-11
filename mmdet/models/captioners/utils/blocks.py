# ---------------------------------------------------------------
# block.py
# Set-up time: 2021/1/3 15:12
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import LowRank
from .misc import activation

class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, relu_dropout, dropout):
        super(FeedForwardBlock, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.layer_norms = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.layer_norms(x)
        return x


class LowRankBilinearLayer(nn.Module):
    def __init__(
            self,
            embed_dim,
            att_type,
            att_heads,
            att_mid_dim,
            att_mid_drop,
            dropout,
            act):
        super(LowRankBilinearLayer, self).__init__()
        self.encoder_attn = LowRank(
            embed_dim=embed_dim,
            att_type=att_type,
            att_heads=att_heads,
            att_mid_dim=att_mid_dim,
            att_mid_drop=att_mid_drop,
            act=act
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
            self,
            x,
            key=None,
            mask=None,
            value1=None,
            value2=None,
            precompute=False
    ):
        x = self.encoder_attn(
            query=x,
            key=key if key is not None else x,
            mask=mask,
            value1=value1 if value1 is not None else x,
            value2=value2 if value2 is not None else x,
            precompute=precompute
        )
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def precompute(self, key, value2):
        return self.encoder_attn.precompute(key, value2)


class LowRankBilinearEncBlock(nn.Module):
    def __init__(self, head_config):
        super(LowRankBilinearEncBlock, self).__init__()
        self.head_config = head_config
        self.layers = nn.ModuleList([])
        self.bifeat_emb = nn.ModuleList([])
        self.layer_norms = nn.ModuleList([])
        for _ in range(self.head_config.encode_layers):
            sublayer = LowRankBilinearLayer(
                embed_dim=self.head_config.bilinear_dim,
                att_type=self.head_config.atttype,
                att_heads=self.head_config.head,
                att_mid_dim=self.head_config.encode_att_mid_dim,
                att_mid_drop=self.head_config.encode_att_mid_dropout,
                dropout=self.head_config.encode_dropout,
                act=self.head_config.act
            )
            self.layers.append(sublayer)

            self.bifeat_emb.append(nn.Sequential(
                nn.Linear(2 * self.head_config.bilinear_dim, self.head_config.bilinear_dim),
                activation(self.head_config.bifeat_emb_act),
                nn.Dropout(self.head_config.encode_bifeat_emb_dropout)
            ))

            self.layer_norms.append(torch.nn.LayerNorm(self.head_config.bilinear_dim))

        self.proj = nn.Linear(self.head_config.bilinear_dim * (self.head_config.encode_layers + 1), self.head_config.bilinear_dim)
        self.layer_norm = torch.nn.LayerNorm(self.head_config.bilinear_dim)

    def forward(self, gv_feat, att_feats, att_mask, p_att_feats=None):
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)

        feat_arr = [gv_feat]
        for i, layer in enumerate(self.layers):
            gv_feat = layer(gv_feat, att_feats, att_mask, gv_feat, att_feats)
            att_feats_cat = torch.cat([gv_feat.unsqueeze(1).expand_as(att_feats), att_feats], dim=-1)

            att_feats = self.bifeat_emb[i](att_feats_cat) + att_feats
            att_feats = self.layer_norms[i](att_feats)
            feat_arr.append(gv_feat)

        gv_feat = torch.cat(feat_arr, dim=-1)
        gv_feat = self.proj(gv_feat)
        gv_feat = self.layer_norm(gv_feat)
        return gv_feat, att_feats


class LowRankBilinearDecBlock(nn.Module):
    def __init__(self, head_config):
        super(LowRankBilinearDecBlock, self).__init__()
        self.head_config = head_config
        self.layers = nn.ModuleList([])
        for _ in range(self.head_config.decode_layers):
            sublayer = LowRankBilinearLayer(
                embed_dim=self.head_config.bilinear_dim,
                att_type=self.head_config.atttype,
                att_heads=self.head_config.head,
                att_mid_dim=self.head_config.decode_att_mid_dim,
                att_mid_drop=self.head_config.decode_att_mid_dropout,
                dropout=self.head_config.decode_dropout,
                act=self.head_config.act
            )
            self.layers.append(sublayer)

        self.proj = nn.Linear(self.head_config.bilinear_dim * (self.head_config.decode_layers + 1), self.head_config.bilinear_dim)
        self.layer_norm = torch.nn.LayerNorm(self.head_config.bilinear_dim)

    def precompute(self, key, value2):
        keys = []
        value2s = []
        for layer in self.layers:
            k, v = layer.precompute(key, value2)
            keys.append(k)
            value2s.append(v)
        return torch.cat(keys, dim=-1), torch.cat(value2s, dim=-1)

    def forward(self, gv_feat, att_feats, att_mask, p_att_feats=None, precompute=False):
        if precompute == True:
            dim = p_att_feats.size()[-1]
            keys = p_att_feats.narrow(-1, 0, dim // 2)
            value2s = p_att_feats.narrow(-1, dim // 2, dim // 2)
            dim = keys.size()[-1] // len(self.layers)

        if gv_feat.shape[-1] == 1:  # empty gv_feat
            if att_mask is not None:
                gv_feat = (torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1))
            else:
                gv_feat = torch.mean(att_feats, 1)

        feat_arr = [gv_feat]
        for i, layer in enumerate(self.layers):
            key = keys.narrow(-1, i * dim, dim) if precompute else att_feats
            value2 = value2s.narrow(-1, i * dim, dim) if precompute else att_feats

            gv_feat = layer(gv_feat, key, att_mask, gv_feat, value2, precompute)
            feat_arr.append(gv_feat)

        gv_feat = torch.cat(feat_arr, dim=-1)
        gv_feat = self.proj(gv_feat)
        gv_feat = self.layer_norm(gv_feat)
        return gv_feat, att_feats