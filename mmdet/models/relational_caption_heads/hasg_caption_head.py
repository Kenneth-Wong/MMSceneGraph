# ---------------------------------------------------------------
# hasg_caption_head.py
# Set-up time: 2021/6/13 21:36
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
from mmdet.models.captioners.utils import activation, expand_tensor, decode_sequence
from mmdet.models.captioners.utils import Attention
from mmdet.models.relation_heads.approaches.motif_util import obj_edge_vectors


@HEADS.register_module
class HASGCaptionHead(nn.Module):
    def __init__(self,
                 seq_len=17,
                 seq_per_img=5,
                 vocab_size=11437,
                 word_embed_config=None,
                 global_feat_config=None,
                 attention_feat_config=None,
                 attention_rel_feat_config=None,
                 head_config=None,
                 param_config=None,
                 loss_caption=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ):
        super(HASGCaptionHead, self).__init__()
        self.seq_len = seq_len
        self.seq_per_img = seq_per_img
        self.vocab_size = vocab_size + 1
        self.word_embed_config = word_embed_config
        self.global_feat_config = global_feat_config
        self.attention_feat_config = attention_feat_config
        self.attention_rel_feat_config = attention_rel_feat_config
        self.head_config = head_config
        self.param_config = param_config
        self.vocab = get_tokens('visualgenomegn')
        self.vocab.insert(0, '<EOF>')
        self.loss_caption = builder.build_loss(loss_caption)

        self.num_layers = 1

        # word embed
        sequential = [nn.Embedding(self.vocab_size, self.word_embed_config.word_embed_dim)]
        sequential.append(activation(self.word_embed_config.word_embed_act, elu_alpha=self.head_config.elu_alpha))
        if self.word_embed_config.word_embed_norm:
            sequential.append(nn.LayerNorm(self.word_embed_config.word_embed_dim))
        if self.word_embed_config.dropout_word_embed > 0:
            sequential.append(nn.Dropout(self.word_embed_config.dropout_word_embed))
        self.word_embed = nn.Sequential(*sequential)

        # attention feats embed
        self.att_dim = self.attention_feat_config.att_feats_embed_dim \
            if self.attention_feat_config.att_feats_embed_dim > 0 else self.attention_feat_config.att_feats_dim
        sequential = []
        if self.attention_feat_config.att_feats_embed_dim > 0:
            sequential.append(nn.Linear(self.attention_feat_config.att_feats_dim,
                                        self.attention_feat_config.att_feats_embed_dim))
        sequential.append(activation(self.attention_feat_config.att_feats_embed_act,
                                     elu_alpha=self.head_config.elu_alpha))
        if self.attention_feat_config.dropout_att_embed > 0:
            sequential.append(nn.Dropout(self.attention_feat_config.dropout_att_embed))
        if self.attention_feat_config.att_feats_norm:
            sequential.append(torch.nn.LayerNorm(self.attention_feat_config.att_feats_embed_dim))
        self.att_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None

        # relation semantic embed
        self.rel_semantic_embed = nn.Embedding(self.vocab_size, self.attention_rel_feat_config.rel_embed_dim)
        vecs = obj_edge_vectors(self.vocab, wv_dir=self.word_embed_config.glove_dir,
                                wv_dim=self.attention_rel_feat_config.rel_embed_dim)
        with torch.no_grad():
            self.rel_semantic_embed.weight.copy_(vecs, non_blocking=True)
        sequential = []
        if self.attention_rel_feat_config.att_feats_embed_dim > 0:
            sequential.append(nn.Linear(self.attention_rel_feat_config.rel_embed_dim,
                                        self.attention_rel_feat_config.att_feats_embed_dim))
        sequential.append(activation(self.attention_rel_feat_config.att_feats_embed_act,
                                     elu_alpha=self.head_config.elu_alpha))
        if self.attention_rel_feat_config.dropout_att_embed > 0:
            sequential.append(nn.Dropout(self.attention_rel_feat_config.dropout_att_embed))
        if self.attention_rel_feat_config.att_feats_norm:
            sequential.append(torch.nn.LayerNorm(self.attention_rel_feat_config.att_feats_embed_dim))
        self.rel_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None

        # logit
        self.dropout_lm = nn.Dropout(
            self.head_config.dropout_lm) if self.head_config.dropout_lm > 0 else None
        self.logit = nn.Linear(self.head_config.rnn_size, self.vocab_size)
        self.p_att_feats = nn.Linear(self.att_dim, self.attention_feat_config.att_hidden_size) \
            if self.attention_feat_config.att_hidden_size > 0 else None
        self.p_rel_att_feats = nn.Linear(self.attention_rel_feat_config.att_feats_embed_dim,
                                         self.attention_rel_feat_config.att_hidden_size) \
                                         if self.attention_rel_feat_config.att_hidden_size > 0 else None

        # captioning core, lstm part
        rnn_input_size = self.head_config.rnn_size + self.word_embed_config.word_embed_dim
        self.lstm = nn.LSTMCell(rnn_input_size, self.head_config.rnn_size)
        self.roi_att = Attention(self.head_config, self.attention_feat_config)
        self.rel_att = Attention(self.head_config, self.attention_rel_feat_config)
        if self.head_config.dropout_input > 0:
            self.dropout_input = nn.Dropout(self.head_config.dropout_input)
        else:
            self.dropout_input = None

        # mixed the context
        assert self.attention_feat_config.att_hidden_size == self.attention_rel_feat_config.att_hidden_size
        self.Wc = nn.Linear(self.attention_feat_config.att_hidden_size, self.head_config.rnn_size)
        self.Wh = nn.Linear(self.head_config.rnn_size, self.head_config.rnn_size)
        self.Wm = nn.Linear(self.head_config.rnn_size, 1)

        self.ss_prob = 0.

    def preprocess_seq(self, input_seq, target_seq):
        # process the input_seq, target_seq for training
        input_seq = input_seq.view(-1, input_seq.size(-1))
        target_seq = target_seq.view(-1, target_seq.size(-1))
        seq_mask = (input_seq > 0).long()
        seq_mask[:, 0] += 1
        seq_mask_sum = seq_mask.sum(-1)
        max_len = int(seq_mask_sum.max())
        input_seq = input_seq[:, 0:max_len].contiguous()
        target_seq = target_seq[:, 0:max_len].contiguous()
        return input_seq, target_seq

    def padded_att(self, roi_feats, num_rois):
        bs = len(num_rois)
        max_num = max(num_rois)
        num_atts = torch.LongTensor(num_rois).to(roi_feats.device)
        split_roi_feats = roi_feats.split(num_rois)
        padded_att_feats = torch.zeros(bs, max_num, roi_feats.size(-1)).to(roi_feats)
        for i in range(bs):
            padded_att_feats[i, :num_rois[i], :] = split_roi_feats[i][:, :]
        att_mask = torch.arange(0, max_num, device=roi_feats.device).long().unsqueeze(0).expand(bs, max_num).lt(
            num_atts.unsqueeze(1)).long()
        return padded_att_feats, att_mask

    def network_preprocess(self, att_feats, att_mask, handle):
        p_att_feats = handle(att_feats) if handle is not None else None
        gv_feat = torch.zeros(att_feats.size(0), 1).to(att_feats)
        return gv_feat, att_feats, att_mask, p_att_feats

    def init_cap_hidden(self, batch_size, device):
        return [torch.zeros(self.num_layers, batch_size, self.head_config.rnn_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.head_config.rnn_size).to(device)]

    def forward(self,
                img,
                img_meta,
                roi_feats,
                det_result,
                gt_result=None,
                is_testing=False,
                beam_size=3):

        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]

        # process the relations
        rel_cap_seqs = det_result.rel_cap_seqs.long()
        rel_cap_seqs = rel_cap_seqs.split(num_rels)
        rel_cap_scores = det_result.rel_cap_scores
        rel_cap_scores = torch.prod(torch.exp(rel_cap_scores), dim=1)
        rel_cap_scores = rel_cap_scores.split(num_rels)
        rel_ipt_scores = det_result.rel_ipt_scores
        rel_prob_ipt_scores = [prob * F.softmax(ipt) for (prob, ipt) in zip(rel_cap_scores, rel_ipt_scores)]
        if self.attention_rel_feat_config.rel_sorted == 'prob':
            scores_for_sort = rel_cap_scores
        elif self.attention_rel_feat_config.rel_sorted == 'ipt':
            scores_for_sort = rel_ipt_scores
        elif self.attention_rel_feat_config.rel_sorted == 'prob_ipt':
            scores_for_sort = rel_prob_ipt_scores
        elif self.attention_rel_feat_config.rel_sorted == 'random':
            scores_for_sort = [torch.rand(s.size(0)).to(s.device) for s in rel_cap_scores]
        num_embed_rels, rel_semantic_feats = [], []
        for scores_i, rel_cap_seqs_i in zip(scores_for_sort, rel_cap_seqs):
            sorted_inds = torch.argsort(scores_i, descending=True)
            rel_cap_seqs_i = rel_cap_seqs_i[sorted_inds][:self.attention_rel_feat_config.num_rel_embed]  # Num x 18
            num_embed_rels.append(rel_cap_seqs_i.size(0))
            rel_semantic_feats_i = self.rel_semantic_embed(rel_cap_seqs_i)  # Num x 18 x 300
            # cutting zero
            effective_len = (rel_cap_seqs_i > 0).sum(-1, keepdim=True)
            feat_mask = (rel_cap_seqs_i > 0).unsqueeze(-1)
            rel_semantic_feats_i = rel_semantic_feats_i * feat_mask
            rel_semantic_feats_i = torch.sum(rel_semantic_feats_i, dim=1) / (effective_len + 1)
            rel_semantic_feats.append(rel_semantic_feats_i)
        rel_semantic_feats = torch.cat(rel_semantic_feats, 0)

        roi_feats = self.att_embed(roi_feats)  # 4096 --> 512
        rel_semantic_feats = self.rel_embed(rel_semantic_feats)  # 300 --> 512
        padded_att_feats, att_mask = self.padded_att(roi_feats, num_objs)
        gv_feat, padded_att_feats, att_mask, p_att_feats = self.network_preprocess(padded_att_feats, att_mask,
                                                                                   self.p_att_feats)

        padded_rel_feats, rel_att_mask = self.padded_att(rel_semantic_feats, num_embed_rels)
        gv_rel_feat, padded_rel_feats, rel_att_mask, p_rel_att_feats = self.network_preprocess(padded_rel_feats,
                                                                                               rel_att_mask,
                                                                                               self.p_rel_att_feats)
        # training or val:
        if not is_testing:
            # training: each image have 5 captions, expand the visual features
            gv_feat = expand_tensor(gv_feat, self.seq_per_img)
            padded_att_feats = expand_tensor(padded_att_feats, self.seq_per_img)
            att_mask = expand_tensor(att_mask, self.seq_per_img)
            p_att_feats = expand_tensor(p_att_feats, self.seq_per_img)

            gv_rel_feat = expand_tensor(gv_rel_feat, self.seq_per_img)
            padded_rel_feats = expand_tensor(padded_rel_feats, self.seq_per_img)
            rel_att_mask = expand_tensor(rel_att_mask, self.seq_per_img)
            p_rel_att_feats = expand_tensor(p_rel_att_feats, self.seq_per_img)

            batch_size = gv_feat.size(0)

            state = self.init_cap_hidden(batch_size, device=padded_att_feats.device)
            cap_inputs, cap_targets = gt_result.cap_inputs, gt_result.cap_targets
            cap_inputs = torch.stack(cap_inputs)
            cap_targets = torch.stack(cap_targets)
            cap_inputs, cap_targets = self.preprocess_seq(cap_inputs, cap_targets)
            gt_result.cap_inputs = cap_inputs
            gt_result.cap_targets = cap_targets
            det_result.cap_targets = cap_targets
            cap_scores = torch.zeros(batch_size, cap_inputs.size(1), self.vocab_size).to(padded_att_feats.device)

            for t in range(cap_inputs.size(1)):
                if self.training and t >= 1 and self.ss_prob > 0:
                    prob = torch.empty(batch_size).cuda().uniform_(0, 1)
                    mask = prob < self.ss_prob
                    if mask.sum() == 0:
                        wt = cap_inputs[:, t].clone()
                    else:
                        ind = mask.nonzero().view(-1)
                        wt = cap_inputs[:, t].clone()
                        prob_prev = torch.exp(cap_scores[:, t - 1].detach())
                        wt.index_copy_(0, ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, ind))
                else:
                    wt = cap_inputs[:, t].clone()

                if t >= 1 and cap_inputs[:, t].max() == 0:
                    break
                logit, state = self.caption_forward(gv_feat, padded_att_feats, att_mask, p_att_feats,
                                                    gv_rel_feat, padded_rel_feats, rel_att_mask,
                                                    p_rel_att_feats,
                                                    state, wt)
                if self.dropout_lm is not None:
                    logit = self.dropout_lm(logit)

                logit = self.logit(logit)
                cap_scores[:, t] = logit

            det_result.cap_scores = cap_scores
        else:
            # beam search
            batch_size = gv_feat.size(0)
            outputs, log_probs = self.decode_beam(beam_size, batch_size, padded_att_feats.device,
                                                  self.get_caption_logprobs_state,
                                                  gv_feat=gv_feat,
                                                  att_feats=padded_att_feats,
                                                  att_mask=att_mask,
                                                  p_att_feats=p_att_feats,
                                                  gv_rel_feat=gv_rel_feat,
                                                  rel_feats=padded_rel_feats,
                                                  rel_att_mask=rel_att_mask,
                                                  p_rel_att_feats=p_rel_att_feats)
            det_result.cap_seqs = outputs
            det_result.cap_scores = log_probs
        return det_result

    def caption_forward(self, gv_feat, att_feats, att_mask, p_att_feats,
                        gv_rel_feat, rel_feats, rel_att_mask, p_rel_att_feats, state, wt):
        xt = self.word_embed(wt)
        h_prev, c_prev = state[0][0], state[1][0]
        roi_att = self.roi_att(h_prev, att_feats, att_mask, p_att_feats)  # B * d
        rel_att = self.rel_att(h_prev, rel_feats, rel_att_mask, p_rel_att_feats)  # B * d
        stack_att = torch.stack((nn.functional.tanh(self.Wc(roi_att)+self.Wh(h_prev)),
                                 nn.functional.tanh(self.Wc(rel_att)+self.Wh(h_prev))), dim=1)  # B * 2 * d
        stack_alpha = self.Wm(stack_att).squeeze(-1)  # B * 2
        stack_alpha = nn.functional.softmax(stack_alpha, -1)
        context = torch.bmm(stack_alpha.unsqueeze(1), stack_att).squeeze(1)  # B * d
        h_new, c_new = self.lstm(torch.cat([context, xt], 1), (h_prev, c_prev))
        state = [h_new[None], c_new[None]]
        return h_new, state

    def get_caption_logprobs_state(self, state, wt, gv_feat, att_feats, att_mask, p_att_feats,
                                   gv_rel_feat, rel_feats, rel_att_mask, p_rel_att_feats):
        output, state = self.caption_forward(gv_feat, att_feats, att_mask, p_att_feats,
                                             gv_rel_feat, rel_feats, rel_att_mask, p_rel_att_feats, state, wt)
        logprobs = F.log_softmax(self.logit(output), dim=1)
        return logprobs, state

    def loss(self, det_result):
        losses = dict()
        cap_scores, cap_targets = det_result.cap_scores, det_result.cap_targets
        if isinstance(cap_targets, (list, tuple)):
            cap_targets = torch.cat(cap_targets, 0)
        losses['loss_caption'] = self.loss_caption(cap_scores.view(-1, self.vocab_size),
                                                   cap_targets.view(-1),
                                                   ignore_index=-1)
        return losses


    def _expand_state(self, batch_size, beam_size, cur_beam_size, state, selected_beam):
        shape = [int(sh) for sh in state.shape]
        beam = selected_beam
        for _ in shape[2:]:
            beam = beam.unsqueeze(-1)
        beam = beam.unsqueeze(0)

        state = torch.gather(
            state.view(*([shape[0], batch_size, cur_beam_size] + shape[2:])), 2,
            beam.expand(*([shape[0], batch_size, beam_size] + shape[2:]))
        )
        state = state.view(*([shape[0], -1, ] + shape[2:]))
        return state

    def select(self, batch_size, beam_size, t, candidate_logprob):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob

    def decode_beam(self, beam_size, batch_size, device, inference_func, **input_vars):
        seq_logprob = torch.zeros((batch_size, 1, 1)).to(device)
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).to(device)

        state = self.init_cap_hidden(batch_size, device=device)
        wt = torch.zeros(batch_size, dtype=torch.long).to(device)

        outputs = []
        for t in range(self.seq_len):
            cur_beam_size = 1 if t == 0 else beam_size
            word_logprob, state = inference_func(state, wt, **input_vars)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx / candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            for s in range(len(state)):
                state[s] = self._expand_state(batch_size, beam_size, cur_beam_size, state[s], selected_beam)

            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                                             selected_beam.unsqueeze(-1).expand(batch_size, beam_size,
                                                                                word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))

            # this_word_att_alpha = torch.gather(this_word_att_alpha, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                for k, v in input_vars.items():
                    input_vars[k] = expand_tensor(v, beam_size)

        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, self.seq_len))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, self.seq_len))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        return outputs, log_probs

    def get_result(self, det_result):
        det_result.cap_sents = decode_sequence(self.vocab, det_result.cap_seqs)

        if det_result.rel_ipt_scores is not None:
            det_result.rel_ipt_scores = F.softmax(det_result.rel_ipt_scores[0], dim=-1)

        result = det_result
        for k, v in result.__dict__.items():
            if k != 'add_losses' and k != 'head_spec_losses' and v is not None and len(v) == 1:
                _v = v[0]  # remove the outer list
            else:
                _v = v
            if isinstance(_v, torch.Tensor):
                result.__setattr__(k, _v.cpu().numpy())
            else:
                result.__setattr__(k, _v)  # e.g., img_shape, is a tuple
        return result
