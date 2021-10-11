# ---------------------------------------------------------------
# het.py
# Set-up time: 2020/9/4 11:18
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import xavier_init
from .motif_util import (obj_edge_vectors, to_onehot, get_dropout_mask, encode_box_info)

from .treelstm_util import TreeLSTM_IO
from .hybridlstm_util import MultiLayer_HybridLSTM, TreeLSTM_Forward, TreeLSTM_Backward


class DecoderTreeLSTM(torch.nn.Module):
    def __init__(self, cfg, classes, embed_dim, inputs_dim, hidden_dim, direction='backward', dropout=0.2,
                 pass_root=False):
        super(DecoderTreeLSTM, self).__init__()
        """
        Initializes the RNN
        :param embed_dim: Dimension of the embeddings
        :param encoder_hidden_dim: Hidden dim of the encoder, for attention purposes
        :param hidden_dim: Hidden dim of the decoder
        :param vocab_size: Number of words in the vocab
        :param bos_token: To use during decoding (non teacher forcing mode))
        :param bos: beginning of sentence token
        :param unk: unknown token (not used)
        direction = forward | backward
        """
        self.cfg = cfg
        self.classes = classes
        self.hidden_size = hidden_dim
        self.inputs_dim = inputs_dim
        self.nms_thresh = 0.5
        self.dropout = dropout
        self.pass_root = pass_root
        # generate embed layer
        embed_vecs = obj_edge_vectors(['start'] + self.classes, wv_dir=self.cfg.glove_dir, wv_dim=embed_dim)
        self.obj_embed = nn.Embedding(len(self.classes) + 1, embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(embed_vecs, non_blocking=True)
        # generate out layer
        self.out = nn.Linear(self.hidden_size, len(self.classes))

        if direction == 'backward':  # a root to all children
            self.input_size = inputs_dim
            self.decoderTreeLSTM = TreeLSTM_Backward(self.input_size, self.hidden_size, self.pass_root,
                                                     is_pass_embed=True, embed_layer=self.obj_embed,
                                                     embed_out_layer=self.out)
        elif direction == 'forward':  # multi children to a root
            self.input_size = inputs_dim
            self.decoderTreeLSTM = TreeLSTM_Forward(self.input_size, self.hidden_size, self.pass_root,
                                                    is_pass_embed=True, embed_layer=self.obj_embed,
                                                    embed_out_layer=self.out)
        else:
            print('Error Decoder LSTM Direction')

        # self.decoderChainLSTM = ChainLSTM(direction, self.input_size, self.hidden_size / 2, is_pass_embed=True,
        #                                  embed_layer=self.obj_embed, embed_out_layer=self.out)

    def forward(self, forest, features, num_obj, labels=None, boxes_for_nms=None, batch_size=0):
        # generate dropout
        if self.dropout > 0.0:
            tree_dropout_mask = get_dropout_mask(self.dropout, self.hidden_size, device=features.device)
        else:
            tree_dropout_mask = None

        # generate tree lstm input/output class
        tree_out_h = None
        tree_out_dists = None
        tree_out_commitments = None
        tree_h_order = torch.LongTensor(num_obj).zero_().to(features.device)
        tree_order_idx = 0
        tree_lstm_io = TreeLSTM_IO(tree_out_h, tree_h_order, tree_order_idx, tree_out_dists,
                                   tree_out_commitments, tree_dropout_mask)

        # chain_out_h = None
        # chain_out_dists = None
        # chain_out_commitments = None
        # chain_h_order = Variable(torch.LongTensor(num_obj).zero_().cuda())
        # chain_order_idx = 0
        # chain_lstm_io = tree_utils.TreeLSTM_IO(chain_out_h, chain_h_order, chain_order_idx, chain_out_dists,
        #                                       chain_out_commitments, chain_dropout_mask)
        for idx in range(len(forest)):
            self.decoderTreeLSTM(forest[idx], features, tree_lstm_io)
            # self.decoderChainLSTM(forest[idx], features, chain_lstm_io)

        out_tree_h = torch.index_select(tree_lstm_io.hidden, 0, tree_lstm_io.order.long())
        out_dists = torch.index_select(tree_lstm_io.dists, 0, tree_lstm_io.order.long())[:-batch_size]
        out_commitments = torch.index_select(tree_lstm_io.commitments, 0, tree_lstm_io.order.long())[:-batch_size]

        return out_dists, out_commitments


class HybridLSTMContext(nn.Module):
    """
    Modified from neural-motifs to encode contexts for each objects
    """

    def __init__(self, config, obj_classes, rel_classes, statistics):
        super(HybridLSTMContext, self).__init__()
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
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.glove_dir, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.dropout_rate
        self.hidden_dim = self.cfg.hidden_dim
        self.nl_obj = self.cfg.context_object_layer
        self.nl_edge = self.cfg.context_edge_layer
        self.with_chain = getattr(self.cfg, 'with_chain', True)
        self.chain_style = getattr(self.cfg, 'chain_style', 'LSTM')
        self.attn_style = getattr(self.cfg, 'attn_style', 'cat')
        self.num_head = getattr(self.cfg, 'num_head', 1)
        assert self.nl_obj > 0 and self.nl_edge > 0

        self.obj_ctx_rnn = MultiLayer_HybridLSTM(
            in_dim=self.obj_dim + self.embed_dim + 128,
            out_dim=self.hidden_dim,
            num_layer=self.nl_obj,
            dropout=self.dropout_rate if self.nl_obj > 1 else 0,
            with_chain=self.with_chain,
            chain_style=self.chain_style,
            num_head=self.num_head,
            attn_style=self.attn_style)
        self.decoder_rnn = DecoderTreeLSTM(self.cfg, self.obj_classes, embed_dim=self.embed_dim,
                                           inputs_dim=self.hidden_dim + self.obj_dim + self.embed_dim + 128,
                                           hidden_dim=self.hidden_dim,
                                           dropout=self.dropout_rate)
        self.edge_ctx_rnn = MultiLayer_HybridLSTM(
            in_dim=self.embed_dim + self.hidden_dim + self.obj_dim,
            out_dim=self.hidden_dim,
            num_layer=self.nl_edge,
            dropout=self.dropout_rate if self.nl_edge > 1 else 0,
            with_chain=self.with_chain,
            chain_style=self.chain_style,
            num_head=self.num_head,
            attn_style=self.attn_style)

    def init_weights(self):
        for m in self.pos_embed:
            if isinstance(m, nn.Linear):
                xavier_init(m)

    def obj_ctx(self, obj_feats, obj_labels=None, forest=None):
        """
        Object context and object classification.
        :param num_objs:
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param det_result:
        :param vc_forest:
        :param: ctx_average:
        :param obj_labels: [num_obj] the GT labels of the image
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        # generate the virtual embeddings for the roots
        virtual_feats = torch.rand(len(forest), obj_feats.size(1)).to(obj_feats.device)
        obj_feats = torch.cat((obj_feats, virtual_feats), 0)
        num_objs = obj_feats.shape[0]

        encoder_rep = self.obj_ctx_rnn(forest, obj_feats, num_objs)
        # Decode in order
        if self.mode != 'predcls':
            decoder_inp = torch.cat((obj_feats, encoder_rep), 1)
            obj_dist, obj_pred = self.decoder_rnn(forest, decoder_inp, num_objs, batch_size=len(forest))
        else:
            assert obj_labels is not None
            obj_pred = obj_labels
            obj_dist = to_onehot(obj_pred, self.num_obj_classes)
        encoder_rep = encoder_rep[:-len(forest)]
        return encoder_rep, obj_pred, obj_dist

    def edge_ctx(self, obj_feats, forest):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :return: edge_ctx: [num_obj, #feats] For later!
        """
        virtual_feats = torch.rand(len(forest), obj_feats.size(1)).to(obj_feats.device)
        obj_feats = torch.cat((obj_feats, virtual_feats), 0)
        edge_rep = self.edge_ctx_rnn(forest, obj_feats, obj_feats.shape[0])
        edge_rep = edge_rep[:-len(forest)]
        return edge_rep

    def forward(self, x, forest, det_result):
        num_objs = [len(b) for b in det_result.bboxes]
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.use_gt_box:
            obj_labels = torch.cat(det_result.labels)
        else:
            obj_labels = None

        if self.cfg.use_gt_label:
            obj_embed = self.obj_embed1(obj_labels.long())
        else:
            obj_dists = torch.cat(det_result.dists, dim=0).detach()
            obj_embed = obj_dists @ self.obj_embed1.weight

        box_info = encode_box_info(det_result)
        pos_embed = self.pos_embed(box_info)  # N x 128

        obj_pre_rep = torch.cat((x, obj_embed, pos_embed), -1)

        # object level contextual feature
        obj_ctxs, obj_preds, obj_dists = self.obj_ctx(obj_pre_rep, obj_labels, forest)

        # edge level contextual feature
        obj_embed2 = self.obj_embed2(obj_preds.long())

        obj_rel_rep = torch.cat((obj_embed2, x, obj_ctxs), -1)

        edge_ctx = self.edge_ctx(obj_rel_rep, forest)

        return obj_dists, obj_preds, edge_ctx
