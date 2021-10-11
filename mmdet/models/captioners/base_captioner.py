# ---------------------------------------------------------------
# base_caption_head.py
# Set-up time: 2021/1/2 17:05
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
from ..registry import CAPTIONERS
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from ..builder import build_loss
from mmdet.models.captioners.utils import load_vocab


@CAPTIONERS.register_module
class BaseCaptioner(nn.Module):
    def __init__(self,
                 seq_len=17,
                 seq_per_img=5,
                 vocab_size=9487,
                 vocab=None,
                 word_embed_config=None,
                 global_feat_config=None,
                 attention_feat_config=None,
                 head_config=None,
                 param_config=None,
                 loss_xe=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ):
        super(BaseCaptioner, self).__init__()
        self.seq_len = seq_len
        self.seq_per_img = seq_per_img
        self.vocab_size = vocab_size
        self.word_embed_config = word_embed_config
        self.global_feat_config = global_feat_config
        self.attention_feat_config = attention_feat_config
        self.head_config = head_config
        self.param_config = param_config
        assert vocab is not None
        self.vocab = load_vocab(vocab)
        self.loss_xe = build_loss(loss_xe)

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

    def select(self, batch_size, beam_size, t, candidate_logprob):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob

    def beam_search(self, init_state, init_logprobs, **kwargs):
        # function computes the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[
                            prev_labels]] - diversity_lambda
            return unaug_logprobsf

        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            # INPUTS:
            # logprobsf: probabilities augmented after diversity
            # beam_size: obvious
            # t        : time instant
            # beam_seq : tensor contanining the beams
            # beam_seq_logprobs: tensor contanining the beam logprobs
            # beam_logprobs_sum: tensor contanining joint logprobs
            # OUPUTS:
            # beam_seq : tensor containing the word indices of the decoded captions
            # beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            # beam_logprobs_sum : joint log-probability of each beam

            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):  # for each column (word, essentially)
                for q in range(rows):  # for each beam expansion
                    # compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q, c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    local_unaug_logprob = unaug_logprobsf[q, ix[q, c]]
                    candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': local_unaug_logprob})
            candidates = sorted(candidates, key=lambda x: -x['p'])

            new_state = [_.clone() for _ in state]
            # beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
                # we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                # fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                # rearrange recurrent states
                for state_ix in range(len(new_state)):
                    #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']]  # dimension one is time step
                # append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c']  # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r']  # the raw logprob here
                beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        beam_size = kwargs['BEAM_SIZE']
        group_size = 1  # kwargs['GROUP_SIZE']
        diversity_lambda = 0.5  # kwargs['DIVERSITY_LAMBDA']
        constraint = False  # kwargs['CONSTRAINT']
        max_ppl = False  # kwargs['MAX_PPL']
        bdash = beam_size // group_size

        beam_seq_table = [torch.LongTensor(self.seq_len, bdash).zero_() for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(self.seq_len, bdash).zero_() for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(bdash) for _ in range(group_size)]

        # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
        done_beams_table = [[] for _ in range(group_size)]
        state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state).chunk(group_size, 2)]
        logprobs_table = list(init_logprobs.chunk(group_size, 0))
        # END INIT

        for t in range(self.seq_len + group_size - 1):
            for divm in range(group_size):
                if t >= divm and t <= self.seq_len + divm - 1:
                    # add diversity
                    logprobsf = logprobs_table[divm].data.float()
                    # suppress previous word
                    if constraint and t - divm > 0:
                        logprobsf.scatter_(1, beam_seq_table[divm][t - divm - 1].unsqueeze(1).cuda(), float('-inf'))
                    # suppress UNK tokens in the decoding
                    logprobsf[:, logprobsf.size(1) - 1] -= 1000
                    # diversity is added here
                    # the function directly modifies the logprobsf values and hence, we need to return
                    # the unaugmented ones for sorting the candidates in the end. # for historical
                    # reasons :-)
                    unaug_logprobsf = add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash)

                    # infer new beams
                    beam_seq_table[divm], \
                    beam_seq_logprobs_table[divm], \
                    beam_logprobs_sum_table[divm], \
                    state_table[divm], \
                    candidates_divm = beam_step(logprobsf,
                                                unaug_logprobsf,
                                                bdash,
                                                t - divm,
                                                beam_seq_table[divm],
                                                beam_seq_logprobs_table[divm],
                                                beam_logprobs_sum_table[divm],
                                                state_table[divm])

                    # if time's up... or if end token is reached then copy beams
                    for vix in range(bdash):
                        if beam_seq_table[divm][t - divm, vix] == 0 or t == self.seq_len + divm - 1:
                            final_beam = {
                                'seq': beam_seq_table[divm][:, vix].clone(),
                                'logps': beam_seq_logprobs_table[divm][:, vix].clone(),
                                'unaug_p': beam_seq_logprobs_table[divm][:, vix].sum().item(),
                                'p': beam_logprobs_sum_table[divm][vix].item()
                            }
                            if max_ppl:
                                final_beam['p'] = final_beam['p'] / (t - divm + 1)
                            done_beams_table[divm].append(final_beam)
                            # don't continue beams from finished sequences
                            beam_logprobs_sum_table[divm][vix] = -1000

                    # move the current group one step forward in time
                    wt = beam_seq_table[divm][t - divm]
                    kwargs[self.param_config.wt] = wt.cuda()
                    kwargs[self.param_config.state] = state_table[divm]
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(**kwargs)

        # all beams are sorted by their log-probabilities
        done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
        done_beams = reduce(lambda a, b: a + b, done_beams_table)
        return done_beams

    def forward(self, img_meta, gv_feat, att_feats, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img_meta, gv_feat, att_feats, **kwargs)
        else:
            return self.forward_test(img_meta, gv_feat, att_feats, **kwargs)

    def forward_train(self, img_meta, gv_feat, att_feats, **kwargs):
        pass

    def forward_test(self, img_meta, gv_feat, att_feats, **kwargs):
        pass

