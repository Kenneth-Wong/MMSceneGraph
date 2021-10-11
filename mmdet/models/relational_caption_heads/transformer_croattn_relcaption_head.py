# ---------------------------------------------------------------
# transformer_croattn_relcaption_head.py
# Set-up time: 2021/3/11 10:39
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
import numpy as np
import mmcv

from .relational_caption_head import RelationalCaptionHead
from mmdet.models.captioners.utils import LowRankBilinearEncBlock, LowRankBilinearDecBlock, FeedForwardBlock
from mmdet.models.captioners.utils import activation, expand_tensor, decode_sequence
import nltk
import copy
import math
from .approaches import MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward, EncoderDecoder, Encoder, \
    EncoderLayer, Decoder, DecoderLayer, Embeddings, Generator, subsequent_mask


@HEADS.register_module
class TransformerCrossAttnRelationalCaptionHead(RelationalCaptionHead):
    def __init__(self, **kwargs):
        super(TransformerCrossAttnRelationalCaptionHead, self).__init__(**kwargs)

        self.att_dim = self.attention_feat_config.att_feats_embed_dim \
            if self.attention_feat_config.att_feats_embed_dim > 0 else self.attention_feat_config.att_feats_dim

        # union feat embed
        sequential = []
        if self.union_feat_config.union_feats_embed_dim > 0:
            sequential.append(
                nn.Linear(self.union_feat_config.union_feats_dim + self.union_feat_config.union_spatial_dim,
                          self.union_feat_config.union_feats_embed_dim))
        sequential.append(activation(self.union_feat_config.union_feats_embed_act,
                                     elu_alpha=self.captioner_config.elu_alpha))
        if self.union_feat_config.dropout_union_embed > 0:
            sequential.append(nn.Dropout(self.union_feat_config.dropout_union_embed))
        if self.union_feat_config.union_feats_norm:
            sequential.append(torch.nn.LayerNorm(self.union_feat_config.union_feats_embed_dim))
        self.union_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None

        # attention feats embed
        sequential = []
        if self.attention_feat_config.att_feats_embed_dim > 0:
            sequential.append(nn.Linear(self.attention_feat_config.att_feats_dim,
                                        self.attention_feat_config.att_feats_embed_dim))
        sequential.append(activation(self.attention_feat_config.att_feats_embed_act,
                                     elu_alpha=self.captioner_config.elu_alpha))
        if self.attention_feat_config.dropout_att_embed > 0:
            sequential.append(nn.Dropout(self.attention_feat_config.dropout_att_embed))
        if self.attention_feat_config.att_feats_norm:
            sequential.append(torch.nn.LayerNorm(self.attention_feat_config.att_feats_embed_dim))
        self.att_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None

        # encoder-decoder
        self.N_enc = self.captioner_config.N_enc
        self.N_dec = self.captioner_config.N_dec
        self.d_model = self.captioner_config.input_encoding_size  # 512
        self.d_ff = self.captioner_config.d_ff  # 512
        self.h = self.captioner_config.num_att_heads
        self.dropout = self.captioner_config.dropout  # 0.1

        self.model = self.make_model(0, self.vocab_size,
                                     N_enc=self.N_enc,
                                     N_dec=self.N_dec,
                                     d_model=self.d_model,
                                     d_ff=self.d_ff,
                                     h=self.h,
                                     dropout=self.dropout)

        # ipt estmiate
        if self.cross_attn:
            self.gather_attention = self.head_config.gather_attention
            self.ipt_estimate = self.head_config.ipt_estimate
            self.mask_obj_word = self.head_config.mask_obj_word
            self.soft_supervised = self.head_config.soft_supervised

            ipt_estm_dim = 512
            if self.ipt_estimate == 'so':
                input_dim = self.attention_feat_config.att_feats_embed_dim * 2
            elif self.ipt_estimate == 'sou':
                input_dim = self.attention_feat_config.att_feats_embed_dim * 2 + self.union_feat_config.union_feats_embed_dim
            elif self.ipt_estimate == 'u':
                input_dim = self.union_feat_config.union_feats_embed_dim
            elif self.ipt_estimate == 'sous':
                input_dim = self.attention_feat_config.att_feats_embed_dim * 2 \
                            + self.union_feat_config.union_feats_embed_dim + \
                            self.union_feat_config.union_feats_semantic_dim * 2
                self.semantic_embedding = nn.Embedding(self.num_classes,
                                                       self.union_feat_config.union_feats_semantic_dim)
            else:
                raise NotImplementedError
            self.global_feat_embed = nn.Sequential(*[
                nn.Linear(self.attention_feat_config.att_feats_embed_dim, ipt_estm_dim),
                activation('CeLU'),
                nn.Dropout(0.5),
                nn.LayerNorm(ipt_estm_dim)])
            self.local_feat_embed = nn.Sequential(*[nn.Linear(input_dim, ipt_estm_dim),
                                                    activation('CeLU'),
                                                    nn.Dropout(0.5),
                                                    nn.LayerNorm(ipt_estm_dim)])

            self.attn_loss_weight = self.head_config.attn_loss_weight

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N_dec),
            lambda x: x,
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

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

    def gather_att_region(self, padded_feats, num_rois, dim):
        return padded_feats.split(num_rois, dim=dim)

    def preproj_feats(self, roi_feats, union_feats):
        return self.att_embed(roi_feats), self.union_embed(union_feats)

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
        seq_mask = seq_mask[:, 0:max_len].contiguous()

        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & subsequent_mask(input_seq.size(-1)).to(seq_mask)
        return input_seq, target_seq, seq_mask

    def get_logprobs_state(self, state, wt, p_att_feats, att_mask):
        if len(state) == 0:
            ys = wt.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], wt.unsqueeze(1)], dim=1)
        out = self.model.decode(p_att_feats, att_mask, ys, subsequent_mask(ys.size(1)).to(p_att_feats.device))
        # use the last region-word atttention (ie, src attention) of the last decoder layer in the decoder, average
        # over the multi-heads (dimension 1),
        # the first time step: attn is 1 * num_head * 1 * num_region
        # the t time step: attn is 1 * num_head * t * num_region
        alpha = self.model.decoder.layers[-1].src_attn.attn.mean(1)[:, -1]
        return F.log_softmax(self.model.generator(out[:, -1])), [ys.unsqueeze(0)], alpha

    def init_cap_hidden(self, batch_size, device):
        return []

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
        # roi_feats: N * 4,096; union_feats: N * 512
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

        # pre encode: 512
        roi_feats, union_feats = self.preproj_feats(roi_feats, union_feats)

        # get global feats:
        split_roi_feats = roi_feats.detach().split(num_objs)
        if self.cross_attn:
            global_feats = []
            for i, srf in enumerate(split_roi_feats):
                global_feats += [self.global_feat_embed(torch.mean(srf, dim=0, keepdim=True))] * num_rels[i]
            global_feats = torch.cat(global_feats)

        # captioning part
        if self.with_caption:
            # padded feats and forward the encoder: each single roi
            padded_att_feats, att_mask = self.padded_att(roi_feats, num_objs)
            att_mask = att_mask.unsqueeze(-2)

            # training or val:
            if not is_testing:
                # training: each image have 5 captions, expand the visual features
                padded_att_feats = expand_tensor(padded_att_feats, self.seq_per_img)
                att_mask = expand_tensor(att_mask, self.seq_per_img)
                batch_size = padded_att_feats.size(0)

                cap_inputs, cap_targets = gt_result.cap_inputs, gt_result.cap_targets
                cap_inputs = torch.stack(cap_inputs)
                cap_targets = torch.stack(cap_targets)
                cap_inputs, cap_targets, seq_mask = self.preprocess_seq(cap_inputs, cap_targets)
                gt_result.cap_inputs = cap_inputs
                gt_result.cap_targets = cap_targets
                det_result.cap_targets = cap_targets
                out = self.model(padded_att_feats, cap_inputs, att_mask, seq_mask)
                cap_scores = self.model.generator(out)
                alphas = self.model.decoder.layers[-1].src_attn.attn.mean(1)  # do not need softmax
                det_result.cap_scores = cap_scores
            else:
                # beam search
                batch_size = padded_att_feats.size(0)
                # encode:
                memory = self.model.encode(padded_att_feats, att_mask)
                outputs, log_probs, alphas = self.decode_beam(beam_size, batch_size, padded_att_feats.device,
                                                              self.get_logprobs_state,
                                                              p_att_feats=memory,
                                                              att_mask=att_mask)
                det_result.cap_seqs = outputs
                det_result.cap_scores = log_probs

        # Relational Captioning Part
        if self.with_relcaption:
            roi_subj_feats, roi_obj_feats = roi_feats[rel_pair_index[:, 0], :], roi_feats[rel_pair_index[:, 1], :]
            # here the triplet features are used as the att feats in captions

            triplet_att_feats = torch.stack((roi_subj_feats, roi_obj_feats, union_feats), 1)  # N, 3, D
            batch_size = triplet_att_feats.size(0)
            num_rois = [3] * batch_size
            num_atts = torch.LongTensor(num_rois).to(triplet_att_feats.device)
            triplet_att_mask = torch.arange(0, 3, device=triplet_att_feats.device).long().unsqueeze(0).expand(
                batch_size, 3).lt(
                num_atts.unsqueeze(1)).long()
            triplet_att_mask = triplet_att_mask.unsqueeze(-2)

            if not is_testing:
                tgt_rel_inputs, tgt_rel_targets, tgt_rel_ipts = det_result.tgt_rel_cap_inputs, det_result.tgt_rel_cap_targets, det_result.tgt_rel_ipts
                tgt_rel_inputs = torch.cat(tgt_rel_inputs, 0)
                tgt_rel_targets = torch.cat(tgt_rel_targets, 0)

                assert tgt_rel_inputs is not None and tgt_rel_targets is not None
                tgt_rel_inputs, tgt_rel_targets, rel_seq_mask = self.preprocess_seq(tgt_rel_inputs, tgt_rel_targets)
                det_result.tgt_rel_cap_inputs = tgt_rel_inputs
                det_result.tgt_rel_cap_targets = tgt_rel_targets

                out = self.model(triplet_att_feats, tgt_rel_inputs, triplet_att_mask, rel_seq_mask)
                rel_cap_scores = self.model.generator(out)
                det_result.rel_cap_scores = rel_cap_scores

            else:
                # beam search
                # encode:
                triplet_memory = self.model.encode(triplet_att_feats, triplet_att_mask)
                outputs, log_probs, _ = self.decode_beam(beam_size, batch_size, roi_subj_feats.device,
                                                         self.get_logprobs_state,
                                                         att_mask=triplet_att_mask,
                                                         p_att_feats=triplet_memory)
                det_result.rel_cap_scores = log_probs
                det_result.rel_cap_seqs = outputs

        # cross attention part
        if self.cross_attn:
            assert self.with_caption and self.with_relcaption, 'No captioner! or no relational captioner!'

            # use the last hidden state to evaluate the rel_ipt_score:
            if self.ipt_estimate == 'so':
                ipt_estm_feats = torch.cat((roi_subj_feats.detach(), roi_obj_feats.detach()), -1)
            elif self.ipt_estimate == 'sou':
                ipt_estm_feats = torch.cat((roi_subj_feats.detach(), roi_obj_feats.detach(), union_feats.detach()), -1)
            elif self.ipt_estimate == 'u':
                ipt_estm_feats = union_feats.detach()
            elif self.ipt_estimate == 'sous':
                if self.mode == 'predcls':
                    labels = det_result.labels
                    if isinstance(labels, (list, tuple)):
                        labels = torch.cat(labels, 0)
                    else:
                        labels = labels.view(-1)
                    dist = torch.zeros(labels.size(0), self.num_classes).to(labels.device)
                    dist[torch.arange(labels.size(0)).to(dist.device).long(), labels] = 1
                else:
                    dist = torch.cat(det_result.dist, 0)
                semantic_feats = dist @ self.semantic_embedding.weight
                roi_subj_sem_feats, roi_obj_sem_feats = semantic_feats[rel_pair_index[:, 0], :], \
                                                        semantic_feats[rel_pair_index[:, 1]]
                ipt_estm_feats = torch.cat((roi_subj_feats.detach(), roi_obj_feats.detach(),
                                            union_feats.detach(), roi_subj_sem_feats, roi_obj_sem_feats), -1)
            else:
                raise NotImplementedError
            ipt_estm_feats = self.local_feat_embed(ipt_estm_feats)
            factor = np.sqrt(ipt_estm_feats.size(1))
            rel_ipt_scores = torch.sum(torch.mul(ipt_estm_feats, global_feats), -1) / factor
            rel_ipt_scores = rel_ipt_scores.split(num_rels)

            det_result.rel_ipt_scores = rel_ipt_scores

            if self.soft_supervised:
                # randomly select one caption and use its attention
                batch_size = alphas.size(0)  # (5B) * T * Nr (train/val) or 1 * T * Nr (test)
                att_mask = att_mask.squeeze(-2)
                num_img = batch_size // self.seq_per_img if not is_testing else batch_size
                if not is_testing:
                    sel_idx = torch.from_numpy(
                        np.arange(num_img, dtype=int) * self.seq_per_img +
                        np.random.randint(self.seq_per_img, size=num_img)).to(alphas.device).long()
                    alphas = alphas[sel_idx, :]  # B * T * Nr
                    att_mask = att_mask[sel_idx, :]  # B * Nr

                # generate the timestep mask:
                timestep_mask = torch.zeros(num_img, alphas.size(1), 1).to(att_mask)  # B * T * 1
                cap_sentences = decode_sequence(self.vocab, det_result.cap_targets[sel_idx, :]) if not is_testing \
                    else decode_sequence(self.vocab, det_result.cap_seqs)  # B * MAXT
                for i, sent in enumerate(cap_sentences):
                    tokens = sent.split(' ')  # do not use tokenize
                    if self.mask_obj_word:
                        tags = nltk.pos_tag(tokens)
                        for j, (word, tag) in enumerate(tags):
                            if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
                                timestep_mask[i, j, :] = 1
                        if torch.sum(timestep_mask[i]) == 0:
                            timestep_mask[i, :len(tokens)] = 1
                    else:
                        timestep_mask[i, :len(tokens)] = 1

                alphas = alphas * timestep_mask
                # collect this info for visualization
                det_result.word_obj_distribution = alphas

                if self.gather_attention == 'mean':
                    alphas = torch.sum(alphas, dim=1) / torch.sum(timestep_mask, dim=1)
                elif self.gather_attention == 'max':
                    alphas = torch.max(alphas, dim=1)[0]
                else:
                    raise NotImplementedError

                if not is_testing:
                    # perform softmax again and got the final object-wise attention
                    alphas = alphas.masked_fill(att_mask == 0,
                                                -1e9)  # put a large negative value, so that the exp() is almost 0
                # collect this info for visualization
                det_result.obj_distribution = F.softmax(alphas, dim=-1)

                # use the rel pair idx to construct the pair attention
                rel_pair_idxes = det_result.rel_pair_idxes
                alphas_pairs = []
                for i, rel_pair_idx in enumerate(rel_pair_idxes):
                    alphas_pairs.append(
                        F.softmax(alphas[i, rel_pair_idx[:, 0]] + alphas[i, rel_pair_idx[:, 1]], dim=-1))

                det_result.rel_distribution = alphas_pairs
            else:
                if not is_testing:
                    det_result.rel_distribution = tgt_rel_ipts
                else:
                    det_result.rel_distribution = None

        return det_result
