# ---------------------------------------------------------------
# updown_croattn_relcaption_head.py
# Set-up time: 2021/2/25 10:03
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
from mmdet.models.captioners.utils import Attention


@HEADS.register_module
class UpDownCrossAttnRelationalCaptionHead(AttBaseRelationalCaptionHead):
    def __init__(self, **kwargs):
        super(UpDownCrossAttnRelationalCaptionHead, self).__init__(**kwargs)
        self.num_layers = 2

        # First LSTM layer
        rnn_input_size = self.captioner_config.rnn_size + self.word_embed_config.word_embed_dim + self.att_dim
        self.lstm1 = nn.LSTMCell(rnn_input_size, self.captioner_config.rnn_size)
        # Second LSTM Layer
        self.lstm2 = nn.LSTMCell(self.captioner_config.rnn_size + self.att_dim, self.captioner_config.rnn_size)
        self.att = Attention(self.captioner_config, self.attention_feat_config)

        if self.captioner_config.dropout_first_input > 0:
            self.dropout1 = nn.Dropout(self.captioner_config.dropout_first_input)
        else:
            self.dropout1 = None

        if self.captioner_config.dropout_sec_input > 0:
            self.dropout2 = nn.Dropout(self.captioner_config.dropout_sec_input)
        else:
            self.dropout2 = None

    def caption_forward(self, gv_feat, att_feats, att_mask, p_att_feats, state, wt):
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            if att_mask is not None:
                gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
            else:
                gv_feat = torch.mean(att_feats, 1)
        xt = self.word_embed(wt)

        # lstm1
        h2_tm1 = state[0][-1]
        input1 = torch.cat([h2_tm1, gv_feat, xt], 1)
        if self.dropout1 is not None:
            input1 = self.dropout1(input1)
        h1_t, c1_t = self.lstm1(input1, (state[0][0], state[1][0]))
        att, att_alpha = self.att(h1_t, att_feats, att_mask, p_att_feats)

        # lstm2
        input2 = torch.cat([att, h1_t], 1)
        if self.dropout2 is not None:
            input2 = self.dropout2(input2)
        h2_t, c2_t = self.lstm2(input2, (state[0][1], state[1][1]))

        state = [torch.stack([h1_t, h2_t]), torch.stack([c1_t, c2_t])]

        return h2_t, state, att_alpha

    def get_caption_logprobs_state(self, state, wt, gv_feat, att_feats, att_mask, p_att_feats):
        output, state, att_alpha = self.caption_forward(gv_feat, att_feats, att_mask, p_att_feats, state, wt)
        logprobs = F.log_softmax(self.logit(output), dim=1)
        return logprobs, state, att_alpha