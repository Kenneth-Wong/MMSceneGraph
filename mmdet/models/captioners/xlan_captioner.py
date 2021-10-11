# ---------------------------------------------------------------
# xlan_caption_head.py
# Set-up time: 2021/1/3 16:01
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
from ..registry import CAPTIONERS
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.captioners.utils import LowRankBilinearEncBlock, LowRankBilinearDecBlock, FeedForwardBlock
from mmdet.models.captioners.utils import activation, expand_tensor

from .att_base_captioner import AttBaseCaptioner


@CAPTIONERS.register_module
class XlanCaptioner(AttBaseCaptioner):
    def __init__(self, **kwargs):
        super(XlanCaptioner, self).__init__(**kwargs)
        self.num_layers = 2

        # First LSTM layer
        rnn_input_size = self.head_config.rnn_size + self.head_config.bilinear_dim
        self.att_lstm = nn.LSTMCell(rnn_input_size, self.head_config.rnn_size)
        self.ctx_drop = nn.Dropout(self.head_config.dropout_lm)

        block_cls = {
            'FeedForward': FeedForwardBlock,
            'LowRankBilinearEnc': LowRankBilinearEncBlock,
            'LowRankBilinearDec': LowRankBilinearDecBlock,
        }
        self.attention = block_cls[self.head_config.decode_block](self.head_config)
        self.att2ctx = nn.Sequential(
            nn.Linear(self.head_config.bilinear_dim + self.head_config.rnn_size, 2 * self.head_config.rnn_size),
            nn.GLU()
        )

    # state[0] -- h, state[1] -- c
    def Forward(self, gv_feat, att_feats, att_mask, p_att_feats, state, wt):
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

        return output, state