# ---------------------------------------------------------------
# att_base_caption_head.py
# Set-up time: 2021/1/2 18:08
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

from .base_captioner import BaseCaptioner
from mmdet.models.captioners.utils import load_vocab, decode_sequence


@CAPTIONERS.register_module
class AttBaseCaptioner(BaseCaptioner):
    def __init__(self, **kwargs):
        super(AttBaseCaptioner, self).__init__(**kwargs)
        self.ss_prob = 0.0  # Schedule sampling probability
        self.vocab_size = self.vocab_size + 1  # include <BOS>/<EOS>
        self.att_dim = self.attention_feat_config.att_feats_embed_dim \
            if self.attention_feat_config.att_feats_embed_dim > 0 else self.attention_feat_config.att_feats_dim

        # word embed
        sequential = [nn.Embedding(self.vocab_size, self.word_embed_config.word_embed_dim)]
        sequential.append(activation(self.word_embed_config.word_embed_act, elu_alpha=self.head_config.elu_alpha))
        if self.word_embed_config.word_embed_norm:
            sequential.append(nn.LayerNorm(self.word_embed_config.word_embed_dim))
        if self.word_embed_config.dropout_word_embed > 0:
            sequential.append(nn.Dropout(self.word_embed_config.dropout_word_embed))
        self.word_embed = nn.Sequential(*sequential)

        # global visual feat embed
        sequential = []
        if self.global_feat_config.gvfeat_embed_dim > 0:
            sequential.append(nn.Linear(self.global_feat_config.gvfeat_dim, self.global_feat_config.gvfeat_embed_dim))
        sequential.append(activation(self.global_feat_config.gvfeat_embed_act, elu_alpha=self.head_config.elu_alpha))
        if self.global_feat_config.dropout_gv_embed > 0:
            sequential.append(nn.Dropout(self.global_feat_config.dropout_gv_embed))
        self.gv_feat_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None

        # attention feats embed
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

        self.dropout_lm = nn.Dropout(self.head_config.dropout_lm) if self.head_config.dropout_lm > 0 else None
        self.logit = nn.Linear(self.head_config.rnn_size, self.vocab_size)
        self.p_att_feats = nn.Linear(self.att_dim, self.attention_feat_config.att_hidden_size) \
            if self.attention_feat_config.att_hidden_size > 0 else None

        # bilinear
        if self.head_config.bilinear_dim > 0:
            self.p_att_feats = None
            block_cls = {
                'FeedForward': FeedForwardBlock,
                'LowRankBilinearEnc': LowRankBilinearEncBlock,
                'LowRankBilinearDec': LowRankBilinearDecBlock,
            }
            self.encoder_layers = block_cls[self.head_config.encode_block](self.head_config)

    def init_hidden(self, batch_size, device):
        return [torch.zeros(self.num_layers, batch_size, self.head_config.rnn_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.head_config.rnn_size).to(device)]

    def make_kwargs(self, wt, gv_feat, att_feats, att_mask, p_att_feats, state, **kgs):
        kwargs = kgs
        kwargs[self.param_config.wt] = wt
        kwargs[self.param_config.global_feat] = gv_feat
        kwargs[self.param_config.att_feats] = att_feats
        kwargs[self.param_config.att_feats_mask] = att_mask
        kwargs[self.param_config.p_att_feats] = p_att_feats
        kwargs[self.param_config.state] = state
        return kwargs

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

    def preprocess_input(self, img_meta, att_feats, input_seq=None, target_seq=None):
        # 1. Transpose the att_feats:
        att_feats = torch.transpose(att_feats, 2, 1)  # N x dim x Nr --> N x Nr x dim

        # 2. generate the att_mask
        batch_size = att_feats.shape[0]
        max_num = att_feats.shape[1]
        num_atts = torch.LongTensor([img_meta[i]['num_att'] for i in range(batch_size)]).to(att_feats.device)
        att_mask = torch.arange(0, max_num, device=att_feats.device).long().unsqueeze(0).expand(batch_size, max_num).lt(
            num_atts.unsqueeze(1)).long()

        if input_seq is not None and target_seq is not None:
            input_seq, target_seq = self.preprocess_seq(input_seq, target_seq)

        return att_feats, att_mask, input_seq, target_seq

    def network_preprocess(self, gv_feat, att_feats, att_mask):
        # embed gv_feat
        if self.gv_feat_embed is not None:
            gv_feat = self.gv_feat_embed(gv_feat)

        # embed att_feats
        if self.att_embed is not None:
            att_feats = self.att_embed(att_feats)

        p_att_feats = self.p_att_feats(att_feats) if self.p_att_feats is not None else None

        # bilinear
        if self.head_config.bilinear_dim > 0:
            gv_feat, att_feats = self.encoder_layers(gv_feat, att_feats, att_mask)
            keys, value2s = self.attention.precompute(att_feats, att_feats)
            p_att_feats = torch.cat([keys, value2s], dim=-1)

        return gv_feat, att_feats, att_mask, p_att_feats

    # gv_feat -- batch_size * cfg.MODEL.GVFEAT_DIM
    # att_feats -- batch_size * att_num * att_feats_dim
    def forward_train(self, img_meta, gv_feat, att_feats, input_seq, target_seq):
        att_feats, att_mask, input_seq, target_seq = self.preprocess_input(img_meta, att_feats, input_seq, target_seq)
        gv_feat, att_feats, att_mask, p_att_feats = self.network_preprocess(gv_feat, att_feats, att_mask)
        gv_feat = expand_tensor(gv_feat, self.seq_per_img)
        att_feats = expand_tensor(att_feats, self.seq_per_img)
        att_mask = expand_tensor(att_mask, self.seq_per_img)
        p_att_feats = expand_tensor(p_att_feats, self.seq_per_img)

        batch_size = gv_feat.size(0)
        state = self.init_hidden(batch_size, device=att_feats.device)

        outputs = torch.zeros(batch_size, input_seq.size(1), self.vocab_size).to(att_feats.device)
        for t in range(input_seq.size(1)):
            if self.training and t >= 1 and self.ss_prob > 0:
                prob = torch.empty(batch_size).cuda().uniform_(0, 1)
                mask = prob < self.ss_prob
                if mask.sum() == 0:
                    wt = input_seq[:, t].clone()
                else:
                    ind = mask.nonzero().view(-1)
                    wt = input_seq[:, t].clone()
                    prob_prev = torch.exp(outputs[:, t - 1].detach())
                    wt.index_copy_(0, ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, ind))
            else:
                wt = input_seq[:, t].clone()

            if t >= 1 and input_seq[:, t].max() == 0:
                break
            output, state = self.Forward(gv_feat, att_feats, att_mask, p_att_feats, state, wt)
            if self.dropout_lm is not None:
                output = self.dropout_lm(output)

            logit = self.logit(output)
            outputs[:, t] = logit

        losses = dict()
        losses['loss_xe'] = self.loss_xe(outputs.view(-1, self.vocab_size), target_seq.view(-1), ignore_index=-1)
        return losses

    def forward_test(self, img_meta, gv_feat, att_feats, beam_size, greedy_decode):
        if beam_size > 1:
            seq, _ = self.decode_beam(img_meta, gv_feat, att_feats, beam_size)
        else:
            seq, _ = self.decode(img_meta, gv_feat, att_feats, greedy_decode)
        sents = decode_sequence(self.vocab, seq)

        results = []
        for sid, sent in enumerate(sents):
            result = {'image_id': int(img_meta[sid]['coco_id']), 'caption': sent}
            results.append(result)
        return results

    def get_logprobs_state(self, gv_feat, att_feats, att_mask, p_att_feats, state, wt):
        output, state = self.Forward(gv_feat, att_feats, att_mask, p_att_feats, state, wt)
        logprobs = F.log_softmax(self.logit(output), dim=1)
        return logprobs, state

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

    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, img_meta, gv_feat, att_feats, beam_size):
        att_feats, att_mask, _, _ = self.preprocess_input(img_meta, att_feats)
        gv_feat, att_feats, att_mask, p_att_feats = self.network_preprocess(gv_feat, att_feats, att_mask)

        batch_size = att_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).to(att_feats.device)
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).to(att_feats.device)

        state = self.init_hidden(batch_size, device=att_feats.device)
        wt = torch.zeros(batch_size, dtype=torch.long).to(att_feats.device)

        outputs = []
        for t in range(self.seq_len):
            cur_beam_size = 1 if t == 0 else beam_size
            word_logprob, state = self.get_logprobs_state(gv_feat, att_feats, att_mask, p_att_feats, state, wt)
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
                att_feats = expand_tensor(att_feats, beam_size)
                gv_feat = expand_tensor(gv_feat, beam_size)
                att_mask = expand_tensor(att_mask, beam_size)
                p_att_feats = expand_tensor(p_att_feats, beam_size)

        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, self.seq_len))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, self.seq_len))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        return outputs, log_probs

    # For the experiments of X-LAN, we use the following beam search code,
    # which achieves slightly better results but much slower.

    # def decode_beam(self, **kwargs):
    #    beam_size = kwargs['BEAM_SIZE']
    #    gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
    #    batch_size = gv_feat.size(0)
    #
    #    sents = torch.zeros((self.seq_len, batch_size), dtype=torch.long).to(att_feats.device)
    #    logprobs = torch.zeros(self.seq_len, batch_size).to(att_feats.device)
    #    self.done_beams = [[] for _ in range(batch_size)]
    #    for n in range(batch_size):
    #        state = self.init_hidden(beam_size, device=att_feats.device)
    #        gv_feat_beam = gv_feat[n:n+1].expand(beam_size, gv_feat.size(1)).contiguous()
    #        att_feats_beam = att_feats[n:n+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous()
    #        att_mask_beam = att_mask[n:n+1].expand(*((beam_size,)+att_mask.size()[1:]))
    #        p_att_feats_beam = p_att_feats[n:n+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous() if p_att_feats is not None else None
    #
    #        wt = torch.zeros(beam_size, dtype=torch.long).to(att_feats.device)
    #        kwargs = self.make_kwargs(wt, gv_feat_beam, att_feats_beam, att_mask_beam, p_att_feats_beam, state, **kwargs)
    #        logprobs_t, state = self.get_logprobs_state(**kwargs)
    #
    #        self.done_beams[n] = self.beam_search(state, logprobs_t, **kwargs)
    #        sents[:, n] = self.done_beams[n][0]['seq']
    #        logprobs[:, n] = self.done_beams[n][0]['logps']
    #    return sents.transpose(0, 1), logprobs.transpose(0, 1)

    def decode(self, img_meta, gv_feat, att_feats, greedy_decode):
        att_feats, att_mask, _, _ = self.preprocess_input(img_meta, att_feats)
        gv_feat, att_feats, att_mask, p_att_feats = self.network_preprocess(gv_feat, att_feats, att_mask)

        batch_size = gv_feat.size(0)
        state = self.init_hidden(batch_size, device=att_feats.device)

        sents = torch.zeros((batch_size, self.seq_len), dtype=torch.long).to(att_feats.device)
        logprobs = torch.zeros(batch_size, self.seq_len).to(att_feats.device)
        wt = torch.zeros(batch_size, dtype=torch.long).to(att_feats.device)
        unfinished = wt.eq(wt)
        for t in range(self.seq_len):
            logprobs_t, state = self.get_logprobs_state(gv_feat, att_feats, att_mask, p_att_feats, state, wt)

            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:, t] = wt
            logprobs[:, t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        return sents, logprobs
