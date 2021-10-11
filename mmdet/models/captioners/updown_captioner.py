# ---------------------------------------------------------------
# Updown_caption_head.py
# Set-up time: 2021/1/3 16:10
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
from ..registry import CAPTIONERS
import torch
import torch.nn as nn
import torch.nn.functional as F
from .att_base_captioner import AttBaseCaptioner
from mmdet.models.captioners.utils import Attention

@CAPTIONERS.register_module
class UpDownCaptioner(AttBaseCaptioner):
    def __init__(self, **kwargs):
        super(UpDownCaptioner, self).__init__(**kwargs)
        self.num_layers = 2

        # First LSTM layer
        rnn_input_size = self.head_config.rnn_size + self.word_embed_config.word_embed_dim + self.att_dim
        self.lstm1 = nn.LSTMCell(rnn_input_size, self.head_config.rnn_size)
        # Second LSTM Layer
        self.lstm2 = nn.LSTMCell(self.head_config.rnn_size + self.att_dim, self.head_config.rnn_size)
        self.att = Attention(self.head_config, self.attention_feat_config)

        if self.head_config.dropout_first_input > 0:
            self.dropout1 = nn.Dropout(self.head_config.dropout_first_input)
        else:
            self.dropout1 = None

        if self.head_config.dropout_sec_input > 0:
            self.dropout2 = nn.Dropout(self.head_config.dropout_sec_input)
        else:
            self.dropout2 = None

    # state[0] -- h, state[1] -- c
    def Forward(self, gv_feat, att_feats, att_mask, p_att_feats, state, wt):
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            gv_feat = torch.mean(att_feats, 1)
        xt = self.word_embed(wt)

        # lstm1
        h2_tm1 = state[0][-1]
        input1 = torch.cat([h2_tm1, gv_feat, xt], 1)
        if self.dropout1 is not None:
            input1 = self.dropout1(input1)
        h1_t, c1_t = self.lstm1(input1, (state[0][0], state[1][0]))
        att = self.att(h1_t, att_feats, att_mask, p_att_feats)

        # lstm2
        input2 = torch.cat([att, h1_t], 1)
        if self.dropout2 is not None:
            input2 = self.dropout2(input2)
        h2_t, c2_t = self.lstm2(input2, (state[0][1], state[1][1]))

        state = [torch.stack([h1_t, h2_t]), torch.stack([c1_t, c2_t])]
        return h2_t, state