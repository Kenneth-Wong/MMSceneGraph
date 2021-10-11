# ---------------------------------------------------------------
# triplelstm_head.py
# Set-up time: 2021/2/2 上午11:41
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
from mmdet.models.relation_heads.approaches.motif_util import block_orthogonal
from mmdet.models.captioners.utils import activation, expand_tensor


@HEADS.register_module
class TripleLSTMHead(RelationalCaptionHead):
    def __init__(self, **kwargs):
        super(TripleLSTMHead, self).__init__(**kwargs)

        rnn_input_size = self.head_config.rnn_input_dim + self.word_embed_config.word_embed_dim
        self.subj_lstm = nn.LSTMCell(rnn_input_size, self.head_config.hidden_dim)
        self.obj_lstm = nn.LSTMCell(rnn_input_size, self.head_config.hidden_dim)
        self.union_lstm = nn.LSTMCell(rnn_input_size, self.head_config.hidden_dim)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.proj_union = nn.Linear(self.head_config.union_feat_dim + self.head_config.union_spatial_dim,
                                    self.head_config.rnn_input_dim)
        self.proj_subj = nn.Linear(self.head_config.single_feat_dim, self.head_config.rnn_input_dim)
        self.proj_obj = nn.Linear(self.head_config.single_feat_dim, self.head_config.rnn_input_dim)

        # REM module
        if self.head_config.with_rem:
            self.W = nn.Linear(self.head_config.single_feat_dim, 3 * self.head_config.context_hidden_dim)
            self.Wz = nn.Linear(self.head_config.context_hidden_dim, self.head_config.single_feat_dim)

        # rel caption module
        sequential = [nn.Embedding(self.vocab_size, self.word_embed_config.word_embed_dim)]
        sequential.append(activation(self.word_embed_config.word_embed_act, elu_alpha=self.word_embed_config.elu_alpha))
        if self.word_embed_config.word_embed_norm:
            sequential.append(nn.LayerNorm(self.word_embed_config.word_embed_dim))
        if self.word_embed_config.dropout_word_embed > 0:
            sequential.append(nn.Dropout(self.word_embed_config.dropout_word_embed))
        self.word_embed = nn.Sequential(*sequential)

        self.logit = nn.Linear(self.head_config.hidden_dim * 3, self.vocab_size)

    def init_weights(self):
        super(TripleLSTMHead, self).init_weights()

        kaiming_init(self.proj_union, distribution='uniform', a=1)
        kaiming_init(self.proj_subj, distribution='uniform', a=1)
        kaiming_init(self.proj_obj, distribution='uniform', a=1)
        if self.head_config.with_rem:
            kaiming_init(self.Wz, distribution='uniform', a=1)
            block_orthogonal(self.W.weight.data,
                             [self.head_config.context_hidden_dim, self.head_config.single_feat_dim])

    def relcaption_forward(self, roi_subj_feats, roi_obj_feats, union_feats, state, wt):
        xt = self.word_embed(wt)
        h_subj, c_subj = self.subj_lstm(torch.cat((roi_subj_feats, xt), -1), (state[0][0], state[1][0]))
        h_obj, c_obj = self.obj_lstm(torch.cat((roi_obj_feats, xt), -1), (state[0][1], state[1][1]))
        h_union, c_union = self.union_lstm(torch.cat((union_feats, xt), -1), (state[0][2], state[1][2]))

        logit = torch.cat((h_subj, h_obj, h_union), -1)
        state = [torch.stack([h_subj, h_obj, h_union]), torch.stack([c_subj, c_obj, c_union])]

        return logit, state

    def get_relcaption_logprobs_state(self, state, wt, roi_subj_feats, roi_obj_feats, union_feats):
        output, state = self.relcaption_forward(roi_subj_feats, roi_obj_feats, union_feats, state, wt)
        logprobs = F.log_softmax(self.logit(output), dim=1)
        return logprobs, state

    def init_relcap_hidden(self, batch_size, device):
        return [torch.zeros(3, batch_size, self.head_config.hidden_dim).to(device),
                torch.zeros(3, batch_size, self.head_config.hidden_dim).to(device)]

    def decode_beam(self, beam_size, batch_size, device, inference_func, **input_vars):
        seq_logprob = torch.zeros((batch_size, 1, 1)).to(device)
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).to(device)

        state = self.init_relcap_hidden(batch_size, device=device)
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

    def forward(self,
                img,
                img_meta,
                det_result,
                gt_result=None,
                is_testing=False,
                beam_size=3):
        """
        Obtain the relation prediction results based on detection results.
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            det_result: (Result): Result containing bbox, label, mask, point, rels,
                etc. According to different mode, all the contents have been
                set correctly. Feel free to  use it.
            gt_result : (Result): The ground truth information.
            is_testing:

        Returns:
            det_result with the following newly added keys:
                refine_scores (list[Tensor]): logits of object
                rel_scores (list[Tensor]): logits of relation
                rel_pair_idxes (list[Tensor]): (num_rel, 2) index of subject and object
                relmaps (list[Tensor]): (num_obj, num_obj):
                target_rel_labels (list[Tensor]): the target relation label.
        """
        roi_feats, union_feats, det_result = self.frontend_features(img, det_result, gt_result)
        if roi_feats.shape[0] == 0:
            return det_result

        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]

        num_images = img[0].size(0)
        assert num_images == len(det_result.rel_pair_idxes)
        rel_pair_index = []
        acc_obj = 0
        for i, num_obj in enumerate(num_objs):
            rel_pair_idx_i = det_result.rel_pair_idxes[i].clone()
            rel_pair_idx_i[:, 0] += acc_obj
            rel_pair_idx_i[:, 1] += acc_obj
            acc_obj += num_obj
            rel_pair_index.append(rel_pair_idx_i)
        rel_pair_index = torch.cat(rel_pair_index, 0)

        # prepare the pairwise subj, obj feats

        # forward the REM:
        if self.head_config.with_rem:
            split_roi_feats = roi_feats.split(num_objs)
            new_roi_feats = []
            for X in split_roi_feats:
                xgate = self.relu(self.W(X))
                Xa, Xb, Xc = xgate[:, :self.head_config.context_hidden_dim], \
                             xgate[:, self.head_config.context_hidden_dim: 2 * self.head_config.context_hidden_dim], \
                             xgate[:, 2 * self.head_config.context_hidden_dim: 3 * self.head_config.context_hidden_dim]

                R = F.softmax(torch.mm(Xa, Xb.transpose(0, 1)), -1)
                A = self.Wz(torch.mm(R, Xc))
                X = X + A
                new_roi_feats.append(X)
            roi_feats = torch.cat(new_roi_feats, 0)

        roi_subj_feats, roi_obj_feats = roi_feats[rel_pair_index[:, 0], :], roi_feats[rel_pair_index[:, 1], :]
        roi_subj_feats = self.dropout(self.relu(self.proj_subj(roi_subj_feats)))
        roi_obj_feats = self.dropout(self.relu(self.proj_obj(roi_obj_feats)))
        union_feats = self.dropout(self.relu(self.proj_union(union_feats)))

        # Relational Captioning Part
        if self.with_relcaption:
            batch_size = roi_subj_feats.size(0)
            # init hidden states
            state = self.init_relcap_hidden(batch_size, roi_subj_feats.device)

            if not is_testing:
                tgt_rel_inputs, tgt_rel_targets, tgt_rel_ipts = det_result.tgt_rel_cap_inputs, det_result.tgt_rel_cap_targets, det_result.tgt_rel_ipts
                tgt_rel_inputs = torch.cat(tgt_rel_inputs, 0)
                tgt_rel_targets = torch.cat(tgt_rel_targets, 0)
                assert tgt_rel_inputs is not None and tgt_rel_targets is not None
                tgt_rel_inputs, tgt_rel_targets = self.preprocess_seq(tgt_rel_inputs, tgt_rel_targets)
                det_result.tgt_rel_cap_inputs = tgt_rel_inputs
                det_result.tgt_rel_cap_targets = tgt_rel_targets

                rel_cap_scores = torch.zeros(batch_size, tgt_rel_inputs.size(1), self.vocab_size).to(
                    roi_obj_feats.device)
                for t in range(tgt_rel_inputs.size(1)):
                    if t >= 1 and self.ss_prob > 0:
                        prob = torch.empty(batch_size).cuda().uniform_(0, 1)
                        mask = prob < self.ss_prob
                        if mask.sum() == 0:
                            wt = tgt_rel_inputs[:, t].clone()
                        else:
                            ind = mask.nonzero().view(-1)
                            wt = tgt_rel_inputs[:, t].clone()
                            prob_prev = torch.exp(rel_cap_scores[:, t - 1].detach())
                            wt.index_copy_(0, ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, ind))
                    else:
                        wt = tgt_rel_inputs[:, t].clone()

                    if t >= 1 and tgt_rel_inputs[:, t].max() == 0:
                        break

                    logit, state = self.relcaption_forward(roi_subj_feats, roi_obj_feats, union_feats, state, wt)
                    logit = self.dropout(self.logit(logit))
                    rel_cap_scores[:, t] = logit

                det_result.rel_cap_scores = rel_cap_scores

            else:
                # beam search
                outputs, log_probs = self.decode_beam(beam_size, batch_size, roi_subj_feats.device,
                                                      self.get_relcaption_logprobs_state,
                                                      roi_subj_feats=roi_subj_feats,
                                                      roi_obj_feats=roi_obj_feats,
                                                      union_feats=union_feats)
                det_result.rel_cap_scores = log_probs
                det_result.rel_cap_seqs = outputs

        return det_result
