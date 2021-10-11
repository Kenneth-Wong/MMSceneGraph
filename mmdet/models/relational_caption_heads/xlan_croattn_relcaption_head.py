# ---------------------------------------------------------------
# xlan_croattn_relcaption_head.py
# Set-up time: 2021/2/21 上午10:10
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

from ..registry import HEADS
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import builder
from ..losses import accuracy
from mmdet.datasets import build_dataset
import os
from mmdet.models.relation_heads.approaches import (FrequencyBias, RelationSampler, PostProcessor)
from mmdet.core import force_fp32
from mmdet.core import get_classes, get_predicates, get_attributes, get_tokens
import numpy as np
import mmcv
from mmdet.core import bbox2roi
import itertools
import copy
from mmcv.cnn import xavier_init, normal_init, kaiming_init

from .relational_caption_head import RelationalCaptionHead
from .att_base_relcaption_head import AttBaseRelationalCaptionHead
from mmdet.models.captioners.utils import LowRankBilinearEncBlock, LowRankBilinearDecBlock, FeedForwardBlock
from mmdet.models.relation_heads.approaches.motif_util import block_orthogonal
from mmdet.models.captioners.utils import activation, expand_tensor


@HEADS.register_module
class XlanCrossAttnRelationalCaptionHead(AttBaseRelationalCaptionHead):
    def __init__(self, **kwargs):
        super(XlanCrossAttnRelationalCaptionHead, self).__init__(**kwargs)
        self.num_layers = 2

        # First LSTM layer
        rnn_input_size = self.captioner_config.rnn_size + self.captioner_config.bilinear_dim
        self.att_lstm = nn.LSTMCell(rnn_input_size, self.captioner_config.rnn_size)
        self.ctx_drop = nn.Dropout(self.captioner_config.dropout_lm)

        block_cls = {
            'FeedForward': FeedForwardBlock,
            'LowRankBilinearEnc': LowRankBilinearEncBlock,
            'LowRankBilinearDec': LowRankBilinearDecBlock,
        }
        self.attention = block_cls[self.captioner_config.decode_block](self.captioner_config)
        self.att2ctx = nn.Sequential(
            nn.Linear(self.captioner_config.bilinear_dim + self.captioner_config.rnn_size, 2 * self.captioner_config.rnn_size),
            nn.GLU()
        )

    def caption_forward(self, gv_feat, att_feats, att_mask, p_att_feats, state, wt):
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            if att_mask is not None:
                gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
            else:
                gv_feat = torch.mean(att_feats, 1)
        xt = self.word_embed(wt)

        h_att, c_att = self.att_lstm(torch.cat([xt, gv_feat + self.ctx_drop(state[0][1])], 1),
                                     (state[0][0], state[1][0]))
        att, _ = self.attention(h_att, att_feats, att_mask, p_att_feats, precompute=True)
        ctx_input = torch.cat([att, h_att], 1)

        output = self.att2ctx(ctx_input)
        state = [torch.stack((h_att, output)), torch.stack((c_att, state[1][1]))]

        #TODO: hook the weights
        att_weights = 0

        return output, state, att_weights

    def get_caption_logprobs_state(self, state, wt, gv_feat, att_feats, att_mask, p_att_feats):
        output, state, _ = self.caption_forward(gv_feat, att_feats, att_mask, p_att_feats, state, wt)
        logprobs = F.log_softmax(self.logit(output), dim=1)
        return logprobs, state


