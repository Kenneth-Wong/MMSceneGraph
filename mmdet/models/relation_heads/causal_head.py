# ---------------------------------------------------------------
# causal_head.py
# Set-up time: 2020/5/22 下午9:29
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
from ..registry import HEADS
from .relation_head import RelationHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from .approaches import (LSTMContext, get_box_info, get_box_pair_info, VCTreeLSTMContext)
from mmcv.cnn import xavier_init, normal_init
from mmdet.core import bbox2roi


@HEADS.register_module
class CausalHead(RelationHead):
    def __init__(self, **kwargs):
        super(CausalHead, self).__init__(**kwargs)

        # head config
        self.spatial_for_vision = self.head_config.causal_spatial_for_vision
        self.fusion_type = self.head_config.causal_fusion_type
        self.separate_spatial = self.head_config.separate_spatial
        self.use_vtranse = self.head_config.causal_context_layer == "vtranse"
        self.effect_type = self.head_config.causal_effect_type

        # init contextual lstm encodind

        if self.head_config.causal_context_layer == "motifs":
            self.context_layer = LSTMContext(self.head_config, self.obj_classes, self.rel_classes)
        elif self.head_config.causal_context_layer == "vctree":
            self.context_layer = VCTreeLSTMContext(self.head_config, self.obj_classes, self.rel_classes, self.statistics)
        elif self.head_config.causal_context_layer == "vtranse":
            pass
            # self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            raise NotImplementedError

        # post decoding
        self.hidden_dim = self.head_config.hidden_dim
        self.pooling_dim = self.head_config.context_pooling_dim

        if self.use_vtranse:
            self.edge_dim = self.pooling_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_predicates, bias=False)
        else:
            self.edge_dim = self.hidden_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                            nn.ReLU(inplace=True)])
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_predicates, bias=True)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_predicates)

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_predicates)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(self.hidden_dim, self.pooling_dim),
                                           nn.ReLU(inplace=True)
                                           ])
        if self.pooling_dim != self.head_config.roi_dim:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(self.head_config.roi_dim, self.pooling_dim)
        else:
            self.union_single_not_match = False

        # untreated average features
        self.effect_analysis = self.head_config.causal_effect_analysis
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))

    def init_weights(self):
        self.bbox_roi_extractor.init_weights()
        self.relation_roi_extractor.init_weights()
        self.context_layer.init_weights()

        normal_init(self.post_emb, mean=0, std=10.0 * (1.0 / self.hidden_dim) ** 0.5)
        if not self.use_vtranse:
            xavier_init(self.post_cat[0])
        xavier_init(self.ctx_compress)
        xavier_init(self.vis_compress)
        if self.fusion_type == 'gate':
            xavier_init(self.ctx_gate_fc)

        if self.spatial_for_vision:
            xavier_init(self.spt_emb[0])
            xavier_init(self.spt_emb[2])

        if self.union_single_not_match:
            xavier_init(self.up_dim)

    def pair_feature_generate(self, roi_feats, det_result, num_objs, obj_boxs, ctx_average=False):
        # encode context infomation
        refine_obj_scores, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_feats, det_result,
                                                                                  ctx_average=ctx_average)
        obj_dist_prob = F.softmax(refine_obj_scores, -1)
        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(det_result.rel_pair_idxes,
                                                                             head_reps, tail_reps,
                                                                             obj_preds, obj_boxs, obj_prob_list):
            if self.use_vtranse:
                ctx_reps.append(head_rep[pair_idx[:, 0]] - tail_rep[pair_idx[:, 1]])
            else:
                ctx_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))

            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_obj_probs.append(torch.stack((obj_prob[pair_idx[:, 0]], obj_prob[pair_idx[:, 1]]), dim=2))
            pair_bboxs_info.append(get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]]))
        pair_obj_probs = torch.cat(pair_obj_probs, dim=0)
        pair_bbox = torch.cat(pair_bboxs_info, dim=0)
        pair_pred = torch.cat(pair_preds, dim=0)
        ctx_rep = torch.cat(ctx_reps, dim=0)
        if self.use_vtranse:
            post_ctx_rep = ctx_rep
        else:
            post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, refine_obj_scores

    def forward(self, img, img_meta, det_result, gt_result=None, is_testing=False, ignore_classes=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        roi_feats, union_feats, det_result = self.frontend_features(img, img_meta, det_result, gt_result)
        if roi_feats.shape[0] == 0:
            return det_result

        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]
        assert len(num_rels) == len(num_objs)
        obj_boxs = [get_box_info(bbox, need_norm=True, size=size)
                    for (bbox, size) in zip(det_result.bboxes, det_result.img_shape)]

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, refine_obj_scores = \
            self.pair_feature_generate(roi_feats, det_result, num_objs, obj_boxs)

        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _ = self.pair_feature_generate(roi_feats,
                                                                                             det_result,
                                                                                             num_objs,
                                                                                             obj_boxs,
                                                                                             ctx_average=True)

        if self.separate_spatial:
            union_feats, spatial_conv_feats = union_feats
            post_ctx_rep = post_ctx_rep * spatial_conv_feats

        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        if self.union_single_not_match:
            union_feats = self.up_dim(union_feats)

        rel_dists = self.calculate_logits(union_feats, post_ctx_rep, pair_pred, use_label_dist=False)

        if self.training:
            det_result.target_labels = torch.cat(det_result.target_labels, dim=-1)
            det_result.target_rel_labels = torch.cat(det_result.target_rel_labels, dim=-1)
        else:
            refine_obj_scores = refine_obj_scores.split(num_objs, dim=0)
            rel_dists = rel_dists.split(num_rels, dim=0)
        det_result.refine_scores = refine_obj_scores
        det_result.rel_scores = rel_dists

        if not is_testing:  # ! only in test, not val or train
            add_for_losses = {}
            # additional loss
            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss_items = []
                for bi_gt, bi_pred in zip(det_result.relmaps, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss_items.append((bi_pred, bi_gt))
                add_for_losses["loss_vctree_binary"] = binary_loss_items

            # branch constraint: make sure each branch can predict independently
            add_for_losses['loss_auxiliary_ctx'] = (self.ctx_compress(post_ctx_rep), det_result.target_rel_labels)
            if not (self.fusion_type == 'gate'):
                add_for_losses['loss_auxiliary_vis'] = (self.vis_compress(union_feats), det_result.target_rel_labels)
                add_for_losses['loss_auxiliary_frq'] = (self.freq_bias.index_with_labels(pair_pred.long()),
                                                        det_result.target_rel_labels)
            det_result.add_losses = add_for_losses

        if self.training:
            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(self.untreated_conv_spt, spatial_conv_feats)
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_feats)

        # testing: if use effect_analysis: perform the following steps and get a new rel_dists; otherwise, do nothing.
        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep
                avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1, -1) \
                    if self.separate_spatial else avg_ctx_rep

                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type == 'TDE':  # TDE of CTX
                rel_dists = self.calculate_logits(union_feats, post_ctx_rep, pair_obj_probs) - self.calculate_logits(
                    union_feats, avg_ctx_rep, pair_obj_probs)
            elif self.effect_type == 'NIE':  # NIE of FRQ
                rel_dists = self.calculate_logits(union_feats, avg_ctx_rep, pair_obj_probs) - self.calculate_logits(
                    union_feats, avg_ctx_rep, avg_frq_rep)
            elif self.effect_type == 'TE':  # Total Effect
                rel_dists = self.calculate_logits(union_feats, post_ctx_rep, pair_obj_probs) - self.calculate_logits(
                    union_feats, avg_ctx_rep, avg_frq_rep)
            else:
                assert self.effect_type == 'none'
                pass
            det_result.rel_scores = rel_dists.split(num_rels, dim=0)

        return det_result

    def moving_average(self, holder, inp):
        assert len(inp.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * inp.mean(0).view(-1)
        return holder

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        if use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        vis_dists = self.vis_compress(vis_rep)
        ctx_dists = self.ctx_compress(ctx_rep)

        if self.fusion_type == 'gate':
            ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
            union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)
            # union_dists = (ctx_dists.exp() * torch.sigmoid(vis_dists + frq_dists + ctx_constraint) + 1e-9).log()
            #  improve on zero-shot, but low mean recall and TDE recall
            # union_dists = ctx_dists * torch.sigmoid(vis_dists * frq_dists)
            #  best conventional Recall results
            # union_dists = (ctx_dists.exp() + vis_dists.exp() + frq_dists.exp() + 1e-9).log()
            #  good zero-shot Recall
            # union_dists = ctx_dists * torch.max(torch.sigmoid(vis_dists), torch.sigmoid(frq_dists))
            #  good zero-shot Recall
            # union_dists = ctx_dists * torch.sigmoid(vis_dists) * torch.sigmoid(frq_dists)
            #  balanced recall and mean recall
            # union_dists = ctx_dists * (torch.sigmoid(vis_dists) + torch.sigmoid(frq_dists)) / 2.0
            #  good zero-shot Recall
            # union_dists = ctx_dists * torch.sigmoid((vis_dists.exp() + frq_dists.exp() + 1e-9).log())
            #  good zero-shot Recall, bad for all of the rest
        elif self.fusion_type == 'sum':
            union_dists = vis_dists + ctx_dists + frq_dists
        else:
            raise NotImplementedError('Invalid fusion type: {}'.format(self.fusion_type))

        return union_dists
