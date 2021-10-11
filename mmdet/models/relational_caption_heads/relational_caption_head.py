# ---------------------------------------------------------------
# relational_caption_head.py
# Set-up time: 2021/1/30 17:34
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
from mmdet.models.relational_caption_heads.approaches import RelationalCapSampler
from mmdet.core import get_classes, get_tokens
import numpy as np
import mmcv
from mmdet.core import bbox2roi
from functools import reduce
from mmdet.models.captioners.utils import expand_tensor, decode_sequence


@HEADS.register_module
class RelationalCaptionHead(nn.Module):
    """
    The basic class of all the relational caption head.
    """

    def __init__(self,
                 with_relcaption,
                 with_caption,
                 head_config,
                 caption_config,
                 cross_attn,
                 bbox_roi_extractor=None,
                 relation_roi_extractor=None,
                 relation_sampler=None,
                 loss_relcaption=None,
                 loss_caption=None
                 ):
        """
        The public parameters that shared by various relation heads are
        initialized here.
        head_config: process the input feature
        caption_config: captioner
        """
        super(RelationalCaptionHead, self).__init__()

        self.with_relcaption = with_relcaption
        self.with_caption = with_caption
        self.cross_attn = cross_attn
        self.head_config = head_config
        self.num_classes = self.head_config.num_classes
        self.use_gt_box = self.head_config.use_gt_box
        self.use_gt_label = self.head_config.use_gt_label

        if self.use_gt_box:
            if self.use_gt_label:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        if bbox_roi_extractor is not None:
            self.bbox_roi_extractor = builder.build_relation_roi_extractor(bbox_roi_extractor)
        if relation_roi_extractor is not None:
            self.relation_roi_extractor = builder.build_relation_roi_extractor(relation_roi_extractor)
        if relation_sampler is not None:
            relation_sampler.update(dict(use_gt_box=self.use_gt_box))
            self.relation_sampler = RelationalCapSampler(**relation_sampler)

        if loss_relcaption is not None:
            self.loss_relcaption = builder.build_loss(loss_relcaption)

        if loss_caption is not None:
            self.loss_caption = builder.build_loss(loss_caption)

        self.obj_classes, self.vocab = get_classes('visualgenomegn'), get_tokens('visualgenomegn')
        self.obj_classes.insert(0, '__background__')
        self.vocab.insert(0, '.')

        # language part
        self.ss_prob = 0.0
        self.caption_config = caption_config
        self.seq_len = self.caption_config.seq_len
        self.seq_per_img = self.caption_config.seq_per_img
        self.vocab_size = self.caption_config.vocab_size + 1

        self.word_embed_config = self.caption_config.word_embed_config
        self.global_feat_config = self.caption_config.global_feat_config
        self.union_feat_config = self.caption_config.union_feat_config
        self.attention_feat_config = self.caption_config.attention_feat_config
        self.captioner_config = self.caption_config.head_config

        # max testing rel pair
        self.max_eval_pairs = 900

    @property
    def with_bbox_roi_extractor(self):
        return hasattr(self, 'bbox_roi_extractor') and self.bbox_roi_extractor is not None

    @property
    def with_relation_roi_extractor(self):
        return hasattr(self, 'relation_roi_extractor') and self.relation_roi_extractor is not None

    @property
    def with_loss_relcaption(self):
        return hasattr(self, 'loss_relcaption') and self.loss_relcaption is not None

    @property
    def with_loss_caption(self):
        return hasattr(self, 'loss_caption') and self.loss_caption is not None

    def init_weights(self):
        if self.with_bbox_roi_extractor:
            self.bbox_roi_extractor.init_weights()
        if self.with_relation_roi_extractor:
            self.relation_roi_extractor.init_weights()

    def frontend_features(self, img, det_result, gt_result):
        bboxes = det_result.bboxes
        # TODO: Modify the sampling methods
        # train/val or: for finetuning on the dataset without relationship annotations
        if gt_result is not None and gt_result.rels is not None:
            if self.mode in ['predcls', 'sgcls']:
                sample_function = self.relation_sampler.gtbox_relsample
            else:
                sample_function = self.relation_sampler.detect_relsample

            rel_pair_idxes, tgt_rel_cap_inputs, tgt_rel_cap_targets, tgt_rel_ipts, rel_matrix = sample_function(
                det_result, gt_result)
        else:
            tgt_rel_cap_inputs, tgt_rel_cap_targets, tgt_rel_ipts, rel_matrix = None, None, None, None
            rel_pair_idxes = self.relation_sampler.prepare_test_pairs(det_result)
            limited_rel_pair_idxes = []
            for rel_pair_idx in rel_pair_idxes:
                perm = torch.randperm(rel_pair_idx.shape[0], device=rel_pair_idx.device)[:self.max_eval_pairs]
                limited_rel_pair_idxes.append(rel_pair_idx[perm, :])
            rel_pair_idxes = limited_rel_pair_idxes

        det_result.rel_pair_idxes = rel_pair_idxes
        det_result.relmaps = rel_matrix
        det_result.tgt_rel_cap_inputs = tgt_rel_cap_inputs
        det_result.tgt_rel_cap_targets = tgt_rel_cap_targets
        det_result.tgt_rel_ipts = tgt_rel_ipts

        rois = bbox2roi(bboxes)

        # extract the unary roi features and union roi features.
        roi_feats = self.bbox_roi_extractor(img, rois)
        union_feats = self.relation_roi_extractor(img, rois, rel_pair_idx=rel_pair_idxes)
        return roi_feats, union_feats, det_result

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

    def forward(self, **kwargs):
        raise NotImplementedError

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
        alphas = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).to(device)

        state = self.init_cap_hidden(batch_size, device=device)
        wt = torch.zeros(batch_size, dtype=torch.long).to(device)

        outputs = []
        for t in range(self.seq_len):
            cur_beam_size = 1 if t == 0 else beam_size
            word_logprob, state, att_alpha = inference_func(state, wt, **input_vars)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            att_alpha = att_alpha.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                att_alpha = att_alpha * seq_mask.expand_as(att_alpha)
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
            this_word_att_alpha = torch.gather(att_alpha, 1,
                                               selected_beam.unsqueeze(-1).expand(batch_size, beam_size,
                                                                                  att_alpha.shape[-1]))
            #this_word_att_alpha = torch.gather(this_word_att_alpha, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            alphas = list(
               torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, att_alpha.shape[-1])) for o in alphas)
            alphas.append(this_word_att_alpha)
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
        alphas = torch.cat([a.unsqueeze(-1) for a in alphas], -1)
        alphas = torch.gather(alphas,  1, sort_idxs.unsqueeze(-1).expand(batch_size, beam_size, alphas.size(-2), self.seq_len))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]
        alphas = alphas.contiguous()[:, 0] # 1 * NR * T
        # transform to T * NR
        alphas = torch.transpose(alphas, dim0=2, dim1=1)

        return outputs, log_probs, alphas

    def loss(self, det_result):
        rel_cap_scores, tgt_rel_cap_targets, cap_scores, cap_targets = det_result.rel_cap_scores, \
                                                                       det_result.tgt_rel_cap_targets, \
                                                                       det_result.cap_scores, \
                                                                       det_result.cap_targets

        losses = dict()
        # do not train the caption while cross_attn is True
        if self.with_relcaption and not self.cross_attn:
            if isinstance(tgt_rel_cap_targets, (list, tuple)):
                tgt_rel_cap_targets = torch.cat(tgt_rel_cap_targets, 0)

            losses['loss_rel_caption'] = self.loss_relcaption(rel_cap_scores.view(-1, self.vocab_size),
                                                              tgt_rel_cap_targets.view(-1),
                                                              ignore_index=-1)

        if self.with_caption and not self.cross_attn:
            if isinstance(cap_targets, (list, tuple)):
                cap_targets = torch.cat(cap_targets, 0)
            losses['loss_caption'] = self.loss_caption(cap_scores.view(-1, self.vocab_size),
                                                       cap_targets.view(-1),
                                                       ignore_index=-1)
        if self.cross_attn:
            losses['loss_attn'] = 0
            rel_ipt_scores = det_result.rel_ipt_scores
            ipt_targets = det_result.rel_distribution
            if self.soft_supervised:
                bs = 0
                for rel_ipt_score, ipt_target in zip(rel_ipt_scores, ipt_targets):
                    bs += rel_ipt_score.size(0)
                    losses['loss_attn'] += torch.nn.KLDivLoss(reduction='none')(F.log_softmax(rel_ipt_score, dim=-1), ipt_target).sum(-1)
                losses['loss_attn'] = losses['loss_attn'] / bs * self.attn_loss_weight
            else:
                loss_attn = builder.build_loss(dict(type='CrossEntropyLoss', use_sigmoid=True,
                                                    loss_weight=self.attn_loss_weight))
                losses['loss_attn'] = loss_attn(torch.cat(rel_ipt_scores, 0), torch.cat(ipt_targets, 0))

        return losses

    def get_result(self, det_result, scale_factor, rescale):
        """
        for test forward
        :param det_result:
        :return:
        """
        if det_result.rel_cap_seqs is not None:
            det_result.rel_cap_sents = decode_sequence(self.vocab, det_result.rel_cap_seqs)

        if det_result.cap_seqs is not None:
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

        if rescale:
            if result.bboxes is not None:
                result.bboxes[:, :4] = result.bboxes[:, :4] / scale_factor
            if result.refine_bboxes is not None:
                result.refine_bboxes[:, :4] = result.refine_bboxes[:, :4] / scale_factor

        return result
