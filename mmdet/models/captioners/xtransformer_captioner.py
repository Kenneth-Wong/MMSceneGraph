# ---------------------------------------------------------------
# xtransformer.py
# Set-up time: 2021/1/3 16:29
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
from ..registry import CAPTIONERS
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.captioners.utils import LowRank, FeedForwardBlock, PositionalEncoding, activation, \
    expand_tensor, decode_sequence
from .base_captioner import BaseCaptioner


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

@CAPTIONERS.register_module
class XTransformerCaptioner(BaseCaptioner):
    def __init__(self):
        super(XTransformerCaptioner, self).__init__()
        self.vocab_size = self.vocab_size + 1

        # att_feats encoder
        sequential = []
        sequential.append(
            nn.Linear(self.attention_feat_config.att_feats_dim, self.attention_feat_config.att_feats_embed_dim))
        sequential.append(activation(self.attention_feat_config.att_feats_embed_act))
        if self.attention_feat_config.att_feats_norm:
            sequential.append(nn.LayerNorm(self.attention_feat_config.att_feats_embed_dim))
        if self.attention_feat_config.dropout_att_embed > 0:
            sequential.append(nn.Dropout(self.attention_feat_config.dropout_att_embed))
        self.att_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None

        self.encoder = Encoder(
            embed_dim=self.head_config.bilinear_dim,
            dropout=self.head_config.encode_dropout,
            att_type=self.head_config.atttype,
            att_heads=self.head_config.head,
            att_mid_dim=self.head_config.encode_att_mid_dim,
            att_mid_drop=self.head_config.encode_att_mid_dropout,
            bifeat_emb_act=self.head_config.bifeat_emb_act,
            bifeat_emb_drop=self.head_config.encode_bifeat_emb_dropout,
            ff_dropout=self.head_config.encode_ff_dropout,
            layer_num=self.head_config.encode_layers,
            act=self.head_config.act)

        self.decoder = Decoder(
            head_config=self.head_config,
            word_embed_config=self.word_embed_config,
            vocab_size=self.vocab_size,
            embed_dim=self.head_config.bilinear_dim,
            dropout=self.head_config.decode_dropout,
            att_type=self.head_config.atttype,
            att_heads=self.head_config.head,
            att_mid_dim=self.head_config.decode_att_mid_dim,
            att_mid_drop=self.head_config.decode_att_mid_dropout,
            bifeat_emb_act=self.head_config.bifeat_emb_act,
            bifeat_emb_drop=self.head_config.decode_bifeat_emb_dropout,
            ff_dropout=self.head_config.decode_ff_dropout,
            layer_num=self.head_config.decode_layers,
            act=self.head_config.act)

    def forward_train(self, img_meta, gv_feat, att_feats, input_seq, target_seq):
        att_feats, att_mask, input_seq, target_seq = self.preprocess_input(img_meta, att_feats, input_seq, target_seq)
        att_mask = expand_tensor(att_mask, self.seq_per_img)
        att_feats = expand_tensor(att_feats, self.seq_per_img)

        ##############################################
        seq_mask = (input_seq > 0).int()
        seq_mask[:, 0] += 1
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & subsequent_mask(input_seq.size(-1)).to(seq_mask)
        seq_mask = seq_mask.float()

        ##############################################
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        decoder_out = self.decoder(gx, input_seq, encoder_out, att_mask, seq_mask)

        losses = dict()
        losses['loss_xe'] = self.loss_xe(decoder_out.view(-1, self.vocab_size), target_seq.view(-1), ignore_index=-1)
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
        encoder_out = att_feats
        gx = gv_feat

        if state is None:
            ys = wt.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], wt.unsqueeze(1)], dim=1)
        seq_mask = subsequent_mask(ys.size(1)).to(encoder_out.device).float()[:, -1, :].unsqueeze(1)
        decoder_out = self.decoder(gx, ys[:, -1].unsqueeze(-1), encoder_out, att_mask, seq_mask, p_att_feats,
                                   True).squeeze(1)

        logprobs = F.log_softmax(decoder_out, dim=-1)
        return logprobs, [ys.unsqueeze(0)]

    def _expand_state(self, batch_size, beam_size, cur_beam_size, selected_beam):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([batch_size, beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s

        return fn

    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, img_meta, gv_feat, att_feats, beam_size):
        att_feats, att_mask, _, _ = self.preprocess_input(img_meta, att_feats)
        #att_feats = kwargs[self.param_config.att_feats]
        #att_mask = kwargs[self.param_config.att_feats_mask]
        #beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).to(att_feats.device)
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).to(att_feats.device)

        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        p_att_feats = self.decoder.precompute(encoder_out)

        state = None
        wt = torch.zeros(batch_size, dtype=torch.long).to(att_feats.device)
        #kwargs[self.param_config.att_feats] = encoder_out
        #kwargs[self.param_config.global_feat] = gx
        #kwargs[self.param_config.p_att_feats] = p_att_feats

        outputs = []
        self.decoder.init_buffer(batch_size)
        for t in range(self.seq_len):
            cur_beam_size = 1 if t == 0 else beam_size

            #kwargs[self.param_config.wt] = wt
            #kwargs[self.param_config.state] = state
            word_logprob, state = self.get_logprobs_state(gx, encoder_out, att_mask, p_att_feats, state, wt)
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

            self.decoder.apply_to_states(self._expand_state(batch_size, beam_size, cur_beam_size, selected_beam))
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
                encoder_out = expand_tensor(encoder_out, beam_size)
                gx = expand_tensor(gx, beam_size)
                att_mask = expand_tensor(att_mask, beam_size)
                state[0] = state[0].squeeze(0)
                state[0] = expand_tensor(state[0], beam_size)
                state[0] = state[0].unsqueeze(0)

                p_att_feats_tmp = []
                for p_feat in p_att_feats:
                    p_key, p_value2 = p_feat
                    p_key = expand_tensor(p_key, beam_size)
                    p_value2 = expand_tensor(p_value2, beam_size)
                    p_att_feats_tmp.append((p_key, p_value2))

                #kwargs[self.param_config.att_feats] = encoder_out
                #kwargs[self.param_config.global_feat] = gx
                #kwargs[self.param_config.att_feats_mask] = att_mask
                #kwargs[self.param_config.p_att_feats] = p_att_feats_tmp
                p_att_feats = p_att_feats_tmp

        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, self.seq_len))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, self.seq_len))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        self.decoder.clear_buffer()
        return outputs, log_probs

    def decode(self, img_meta, gv_feat, att_feats, greedy_decode, **kwargs):
        att_feats, att_mask, _, _ = self.preprocess_input(img_meta, att_feats)
        #beam_size = kwargs['BEAM_SIZE']
        #greedy_decode = kwargs['GREEDY_DECODE']
        #att_feats = kwargs[self.param_config.att_feats]
        #att_mask = kwargs[self.param_config.att_feats_mask]

        batch_size = att_feats.size(0)
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        p_att_feats = self.decoder.precompute(encoder_out)
        self.decoder.init_buffer(batch_size)

        state = None
        sents = torch.zeros((batch_size, self.seq_len), dtype=torch.long).to(att_feats.device)
        logprobs = torch.zeros(batch_size, self.seq_len).to(att_feats.device)
        wt = torch.zeros(batch_size, dtype=torch.long).to(att_feats.device)
        unfinished = wt.eq(wt)
        #kwargs[self.param_config.att_feats] = encoder_out
        #kwargs[self.param_config.global_feat] = gx
        #kwargs[self.param_config.p_att_feats] = p_att_feats
        for t in range(self.seq_len):
            #kwargs[self.param_config.wt] = wt
            #kwargs[self.param_config.state] = state
            logprobs_t, state = self.get_logprobs_state(gx, encoder_out, att_mask, p_att_feats, state, wt)

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
        self.decoder.clear_buffer()
        return sents, logprobs


class Encoder(nn.Module):
    def __init__(
            self,
            embed_dim,
            dropout,
            att_type,
            att_heads,
            att_mid_dim,
            att_mid_drop,
            bifeat_emb_act,
            bifeat_emb_drop,
            ff_dropout,
            layer_num,
            act
    ):
        super(Encoder, self).__init__()
        self.att_heads = att_heads
        self.layers = nn.ModuleList([])
        for i in range(layer_num):
            sublayer = EncoderLayer(
                embed_dim=embed_dim,
                dropout=dropout,
                att_type=att_type,
                att_heads=att_heads,
                att_mid_dim=att_mid_dim,
                att_mid_drop=att_mid_drop,
                bifeat_emb_act=bifeat_emb_act,
                bifeat_emb_drop=bifeat_emb_drop,
                ff_dropout=ff_dropout,
                act=act)
            self.layers.append(sublayer)

        self.proj_norm = nn.Sequential(
            nn.Linear(embed_dim * (layer_num + 1), embed_dim),
            torch.nn.LayerNorm(embed_dim))

    def forward(self, x, mask):
        gx = (torch.sum(x * mask.unsqueeze(-1), 1) / torch.sum(mask.unsqueeze(-1), 1))

        gx_arr = [gx]
        for layer in self.layers:
            gx, x = layer(gx, x, mask)
            gx_arr.append(gx)

        gx = torch.cat(gx_arr, dim=-1)
        gx = self.proj_norm(gx)
        return gx, x


class EncoderLayer(nn.Module):
    def __init__(
            self,
            embed_dim,
            dropout,
            att_type,
            att_heads,
            att_mid_dim,
            att_mid_drop,
            bifeat_emb_act,
            bifeat_emb_drop,
            ff_dropout,
            act
    ):
        super(EncoderLayer, self).__init__()
        self.encoder_attn = LowRank(
            embed_dim=embed_dim,
            att_type=att_type,
            att_heads=att_heads,
            att_mid_dim=att_mid_dim,
            att_mid_drop=att_mid_drop,
            act=act)
        self.dropout = nn.Dropout(dropout)

        self.bifeat_emb = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            activation(bifeat_emb_act),
            nn.Dropout(bifeat_emb_drop)
        )
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

        self.ff_layer = FeedForwardBlock(embed_dim=embed_dim, ffn_embed_dim=embed_dim * 4, relu_dropout=ff_dropout,
                                         dropout=ff_dropout)

    def forward(self, gx, x, mask):
        gx = self.encoder_attn(
            query=gx,
            key=x,
            mask=mask,
            value1=gx,
            value2=x
        )
        gx = self.dropout(gx)

        x_ = torch.cat([gx.unsqueeze(1).expand_as(x), x], dim=-1)
        x = self.bifeat_emb(x_) + x
        x = self.layer_norm(x)

        if self.ff_layer is not None:
            x = self.ff_layer(x)
        return gx, x


class Decoder(nn.Module):
    def __init__(
            self,
            head_config,
            word_embed_config,
            vocab_size,
            embed_dim,
            dropout,
            att_type,
            att_heads,
            att_mid_dim,
            att_mid_drop,
            bifeat_emb_act,
            bifeat_emb_drop,
            ff_dropout,
            layer_num,
            act
    ):
        super(Decoder, self).__init__()
        self.att_heads = att_heads
        self.layers = nn.ModuleList([])
        self.embed_dim = embed_dim
        for i in range(layer_num):
            sublayer = DecoderLayer(
                embed_dim=embed_dim,
                dropout=dropout,
                att_type=att_type,
                att_heads=att_heads,
                att_mid_dim=att_mid_dim,
                att_mid_drop=att_mid_drop,
                bifeat_emb_act=bifeat_emb_act,
                bifeat_emb_drop=bifeat_emb_drop,
                ff_dropout=ff_dropout,
                act=act,
                last_layer=(i == layer_num - 1))
            self.layers.append(sublayer)

        self.dropout = nn.Dropout(word_embed_config.dropout_word_embed)
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEncoding(
            embed_dim, head_config.pe_max_len
        )

        self.layer_norm_word = torch.nn.LayerNorm(embed_dim)
        self.generator = nn.Linear(embed_dim, vocab_size)

        self.wbil1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            activation(head_config.act),
            torch.nn.LayerNorm(embed_dim)
        )
        self.wbil2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            activation(head_config.act),
            torch.nn.LayerNorm(embed_dim)
        )
        self.wbi_drop = nn.Dropout(head_config.decode_dropout)
        self.dropout_lm = nn.Dropout(head_config.dropout_lm)

        self.proj_norm = nn.Sequential(
            nn.Linear(embed_dim * (layer_num + 1), 2 * embed_dim),
            nn.GLU(),
            torch.nn.LayerNorm(embed_dim))

        self.clear_buffer()

    def init_buffer(self, batch_size):
        self.seq_len = 0
        self.x = torch.zeros((batch_size, 1, self.embed_dim)).cuda()
        for layer in self.layers:
            layer.init_buffer(batch_size)

    def clear_buffer(self):
        self.seq_len = None
        self.x = None
        for layer in self.layers:
            layer.clear_buffer()

    def apply_to_states(self, fn):
        self.x = fn(self.x)
        for layer in self.layers:
            layer.apply_to_states(fn)

    def precompute(self, encoder_out):
        p_att_feats = []
        for layer in self.layers:
            key, value2 = layer.precompute(encoder_out)
            p_att_feats.append((key, value2))
        return p_att_feats

    def forward(self, gx, prev_output_tokens, encoder_out, att_mask, seq_mask=None, p_att_feats=None, precompute=False):
        att_mask = att_mask.unsqueeze(1)

        # embed positions
        seq_len = prev_output_tokens.size(1)
        if self.seq_len is not None:
            seq_len = self.seq_len + seq_len
            self.seq_len = seq_len
            positions = self.embed_positions(seq_len)[:, -1, :].unsqueeze(1)
        else:
            positions = self.embed_positions(seq_len)

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        x = x + positions
        x = self.layer_norm_word(x)
        if self.dropout is not None:
            x = self.dropout(x)

        # decoder layers
        gx = self.wbil1(gx)
        if self.x is None:
            x_gx = (torch.sum(x.unsqueeze(1) * seq_mask.unsqueeze(-1), -2) / torch.sum(seq_mask, -1).unsqueeze(-1))
        else:
            self.x = self.x + x
            x_gx = self.x / seq_len
        x_gx = self.wbil2(x_gx)
        gx = gx.unsqueeze(1)
        gx = gx * x_gx
        gx = self.wbi_drop(gx)

        gx_arr = [gx]
        for layerid, layer in enumerate(self.layers):
            if precompute == False:
                p_key = None
                p_value2 = None
            else:
                p_key, p_value2 = p_att_feats[layerid]
            gx, x = layer(gx, x, encoder_out, att_mask, seq_mask=seq_mask, p_key=p_key, p_value2=p_value2,
                          precompute=precompute)
            gx_arr.append(gx)

        gx = torch.cat(gx_arr, dim=-1)
        gx = self.proj_norm(gx)

        gx = self.dropout_lm(gx)
        out = self.generator(gx)
        return out


class DecoderLayer(nn.Module):
    def __init__(
            self,
            embed_dim,
            dropout,
            att_type,
            att_heads,
            att_mid_dim,
            att_mid_drop,
            bifeat_emb_act,
            bifeat_emb_drop,
            ff_dropout,
            act,
            last_layer=False
    ):
        super(DecoderLayer, self).__init__()
        self.last_layer = last_layer
        self.word_attn = LowRank(
            embed_dim=embed_dim,
            att_type=att_type,
            att_heads=att_heads,
            att_mid_dim=att_mid_dim,
            att_mid_drop=att_mid_drop,
            act=act)
        self.word_dropout = nn.Dropout(dropout)

        self.cross_att = LowRank(
            embed_dim=embed_dim,
            att_type=att_type,
            att_heads=att_heads,
            att_mid_dim=att_mid_dim,
            att_mid_drop=att_mid_drop,
            act=act)
        self.cross_dropout = nn.Dropout(dropout)
        self.layer_norm_cross = torch.nn.LayerNorm(embed_dim)

        if self.last_layer == False:
            self.bifeat_emb = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                activation(bifeat_emb_act),
                nn.Dropout(bifeat_emb_drop)
            )
            self.layer_norm_x = torch.nn.LayerNorm(embed_dim)

            self.ff_layer = FeedForwardBlock(embed_dim=embed_dim,ffn_embed_dim=embed_dim * 4,relu_dropout=ff_dropout,
                                             dropout=ff_dropout)

        self.layer_norm_gx = torch.nn.LayerNorm(embed_dim)

    def apply_to_states(self, fn):
        self.word_attn.apply_to_states(fn)

    def init_buffer(self, batch_size):
        self.word_attn.init_buffer(batch_size)

    def clear_buffer(self):
        self.word_attn.clear_buffer()

    def precompute(self, encoder_out):
        key, value2 = self.cross_att.precompute(encoder_out, encoder_out)
        return key, value2

    def forward(
            self,
            gx,
            x,
            encoder_out,
            att_mask,
            seq_mask,
            p_key=None,
            p_value2=None,
            precompute=False
    ):
        word_x = x
        residual = x
        x = self.word_attn.forward2(
            query=gx,
            key=x,
            mask=seq_mask,
            value1=gx,
            value2=x)
        x = self.word_dropout(x)
        x = residual + x

        residual = x
        x = self.layer_norm_cross(x)
        x = self.cross_att.forward2(
            query=x,
            key=encoder_out if precompute == False else p_key,
            mask=att_mask,
            value1=x,
            value2=encoder_out if precompute == False else p_value2,
            precompute=precompute)
        x = self.cross_dropout(x)
        gx = residual + x
        gx = self.layer_norm_gx(gx)

        if self.last_layer == False:
            x_ = torch.cat([gx, word_x], dim=-1)
            x = self.bifeat_emb(x_) + word_x
            x = self.layer_norm_x(x)

            if self.ff_layer is not None:
                x = self.ff_layer(x)
        else:
            x = None
        return gx, x
