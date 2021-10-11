# ---------------------------------------------------------------
# rnn.py
# Set-up time: 2020/10/21 9:35
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

class GRUWriter(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUWriter, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gate_size = 3 * self.hidden_size

        self.ih = nn.Linear(self.input_size, self.gate_size, bias=True)
        self.hh = nn.Linear(self.hidden_size, self.gate_size, bias=True)

    def forward(self, x, h):
        input_trans = self.ih(x)
        hidden_trans = self.hh(x)
        reset_gate = torch.sigmoid(input_trans[:, 0:self.hidden_size] + hidden_trans[:, 0:self.hidden_size])
        update_gate = torch.sigmoid(input_trans[:, self.hidden_size:2*self.hidden_size] +
                                    hidden_trans[:, self.hidden_size:2*self.hidden_size])
        update = torch.tanh(input_trans[:, 2*self.hidden_size:3*self.hidden_size] +
                            reset_gate * hidden_trans[:, 2*self.hidden_size:3*self.hidden_size])
        out = (1 - update_gate) * update + update_gate * h
        return out