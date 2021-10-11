# ---------------------------------------------------------------
# motif_head.py
# Set-up time: 2020/4/27 下午8:08
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

from ..registry import HEADS
from .. import builder
import torch
import torch.nn as nn
import torch.nn.functional as F
from .relation_head import RelationHead
from .approaches import LSTMContext
from .approaches import GRUWriter
from .approaches import get_pattern_labels, get_internal_labels, top_down_induce
from .approaches.pointnet import feature_transform_regularizer
from mmcv.cnn import xavier_init, normal_init
from mmdet.core import get_predicate_hierarchy
from mmdet.models.losses import CrossEntropyLoss, TreeLoss
from .approaches.motif_util import obj_edge_vectors
import json
import mmcv
import numpy as np
from anytree import AnyNode

@HEADS.register_module
class MotifVRPHead(RelationHead):
    def __init__(self, **kwargs):
        super(MotifVRPHead, self).__init__(**kwargs)
        # 1. Initialize the interaction pattern templates
        self.num_point = self.head_config.num_point
        self.traditional_rel_cls = self.head_config.traditional_rel_cls
        self.group_pairs_file = self.head_config.get('group_pairs', None)
        self.group_pairs = None
        if self.group_pairs_file is not None:
            self.group_pairs = json.load(open(self.group_pairs_file))
            # transform the prior pairs into id
            pairs = []
            for pair in self.group_pairs:
                pairs.append([self.obj_classes.index(pair[0]), self.obj_classes.index(pair[1])])
            self.group_pairs = pairs

        self.context_layer = LSTMContext(self.head_config, self.obj_classes, self.rel_classes)

        # compositional augment module
        self.composition_augment = self.head_config.composition_augment
        self.use_scene = self.head_config.use_scene
        self.use_visual_sim = self.head_config.use_visual_sim
        if self.composition_augment:
            # setup the scene prediction branch
            if self.use_scene:
                self.num_scenes = self.head_config.num_scenes
                self.scene_roi_extractor = builder.build_relation_roi_extractor(self.head_config.scene_roi_extractor)
                self.scene_head = nn.Sequential(*[nn.Linear(self.head_config.roi_dim, self.num_scenes),
                                                  nn.ReLU(inplace=True)])
                # load the prior knowledge, and build the structure for composition
                scene_object_prior = mmcv.load(self.head_config.scene_object_prior)
                self.register_buffer('scene_object_prior', torch.from_numpy(scene_object_prior).float())

            predicate_object_prior = mmcv.load(self.head_config.predicate_object_prior)
            self.register_buffer('predicate_object_prior', torch.from_numpy(predicate_object_prior).float())

            obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.head_config.glove_dir,
                                              wv_dim=self.head_config.embed_dim)
            vec_norms = torch.norm(obj_embed_vecs, dim=1, keepdim=True)
            semantic_similarity = torch.div(obj_embed_vecs.mm(obj_embed_vecs.transpose(1, 0)), vec_norms.mm(
                vec_norms.transpose(1, 0)))
            semantic_similarity = (1 + semantic_similarity) / 2
            semantic_similarity[torch.arange(self.num_classes).long(), torch.arange(self.num_classes).long()] = 0
            self.register_buffer('semantic_similarity', semantic_similarity)

            self.pool_writer = GRUWriter(input_size=self.head_config.roi_dim, hidden_size=self.head_config.roi_dim)
            self.register_buffer('entity_pools', torch.zeros(self.num_classes, self.head_config.roi_dim))

        # post decoding
        self.use_vision = self.head_config.use_vision
        self.hidden_dim = self.head_config.hidden_dim
        self.context_pooling_dim = self.head_config.context_pooling_dim
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.context_pooling_dim)
        self.rel_compress = nn.Linear(self.context_pooling_dim, self.num_predicates, bias=True)

        if self.context_pooling_dim != self.head_config.roi_dim:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(self.head_config.roi_dim, self.context_pooling_dim)
        else:
            self.union_single_not_match = False

        # 3. the relation prediction head
        if not self.traditional_rel_cls:
            assert self.head_config.predicate_hierarchy is not None
            self.beta_param = self.head_config.beta_param
            self.predicate_hierarchy, self.augmented_predicates = \
                get_predicate_hierarchy(self.head_config.predicate_hierarchy)
            self.predicate_hierarchy.children = self.predicate_hierarchy.children + (AnyNode(id='__background__'),)
            self.augmented_predicates.insert(0, '__background__')
            self.num_augmented_predicates = len(self.augmented_predicates)
            self.num_pattern = len(self.predicate_hierarchy.children)
            # preapare: for all the leaf label, find its pattern label
            leaf_to_pattern = get_pattern_labels(np.arange(self.num_predicates, dtype=np.int32),
                                                 self.predicate_hierarchy,
                                                 self.augmented_predicates)
            pattern_to_consecutive_labels = np.ones(self.num_augmented_predicates, dtype=np.int32) * -1.
            pattern_indices = np.array(sorted(list(set(leaf_to_pattern.tolist()))), dtype=np.int32)
            self.register_buffer('pattern_indices', torch.from_numpy(pattern_indices).long())
            pattern_to_consecutive_labels[pattern_indices] = np.arange(self.num_pattern)
            self.register_buffer('leaf_to_pattern', torch.from_numpy(leaf_to_pattern).long())
            self.register_buffer('pattern_to_consecutive_labels',
                                 torch.from_numpy(pattern_to_consecutive_labels).long())

            self.solveroot = mmcv.load(self.head_config.solve_root)
            self.treelossinfo = mmcv.load(self.head_config.treeloss_info)

            self.proj_union = nn.Sequential(*[nn.Linear(self.head_config.roi_dim * 2, self.head_config.roi_dim),
                                              nn.ReLU(inplace=True)])
            self.out_pattern_rel = nn.Linear(self.head_config.roi_dim, self.num_pattern, bias=True)

    def init_weights(self):
        self.bbox_roi_extractor.init_weights()
        self.relation_roi_extractor.init_weights()
        self.context_layer.init_weights()

        normal_init(self.post_emb, mean=0, std=10.0 * (1.0 / self.hidden_dim) ** 0.5)
        xavier_init(self.post_cat)
        xavier_init(self.rel_compress)

        if self.union_single_not_match:
            xavier_init(self.up_dim)

    def forward(self,
                img,
                img_meta,
                det_result,
                gt_result=None,
                is_testing=False):
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
        roi_feats, roi_feats_point, single_trans_matrix, \
        union_feats, union_feats_point, union_trans_matrix, det_result = self.frontend_features(
            img, img_meta, det_result, gt_result)
        if roi_feats.shape[0] == 0:
            return det_result

        if not self.traditional_rel_cls:
            union_feats = self.proj_union(torch.cat((union_feats, union_feats_point), -1))

        refine_obj_scores, obj_preds, edge_ctx, _ = self.context_layer(roi_feats, det_result)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(det_result.rel_pair_idxes, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = torch.cat(prod_reps, dim=0)
        pair_pred = torch.cat(pair_preds, dim=0)
        num_pairs = prod_rep.shape[0]

        # OPTIONAL: begin the composition module, not in testing phase
        if self.composition_augment and not is_testing:
            # 1. scene cls: obtain the scene confidences
            if self.use_scene:
                scene_rois = np.empty((len(img_meta), 5), dtype=np.float32)
                scene_rois[:, 0] = np.arange(len(img_meta))
                for i in range(len(img_meta)):
                    scene_rois[i, 1:] = np.array([0, 0, det_result.img_shape[i][1], det_result.img_shape[i][0]])
                scene_rois = torch.from_numpy(scene_rois).to(roi_feats)
                scene_feats = self.scene_roi_extractor(img, img_meta, scene_rois)[0]
                scene_scores = self.scene_head(scene_feats)
                scene_prob = F.softmax(scene_scores, -1)
                object_in_scene_prob = torch.mm(scene_prob, self.scene_object_prior)
                object_in_scene_prob = F.softmax(object_in_scene_prob, -1)

            # 2. write the current object features to the memory
            # 1) collect the entities according to the predicted categories
            # only do it in training
            if self.training:
                obj_preds_cat = torch.cat(obj_preds, -1)
                current_inp = torch.zeros_like(self.entity_pools)
                for i in range(self.num_classes):
                    inds_i = torch.nonzero(obj_preds_cat == i).view(-1)
                    if len(inds_i) == 0:
                        continue
                    current_inp[i, :] = torch.mean(edge_ctx[inds_i], dim=0)
                # 2) use the GRU mechanism to update the object pools
                self.entity_pools = self.pool_writer(current_inp, self.entity_pools)

            # 3. sample the subj/obj and construct composited triplets
            assert det_result.target_rel_labels is not None, \
                "This can only be used on the dataset with relation annotations. "
            source_triplets = []
            for rel_pair_idx, target_label, rel_label in zip(det_result.rel_pair_idxes, det_result.target_labels,
                                                             det_result.target_rel_labels):
                source_triplets.append(torch.cat((target_label[rel_pair_idx[:, 0][:, None]],
                                                  target_label[rel_pair_idx[:, 1][:, None]],
                                                  rel_label[:, None]), -1))
            source_triplets = torch.cat(source_triplets, 0)
            positive_inds = torch.nonzero(source_triplets[:, -1] > 0).view(-1)
            # only imagine for the positive samples
            if len(positive_inds) > 0:
                positive_triplets = source_triplets[positive_inds]
                PO_prior_subj = self.predicate_object_prior[0, positive_triplets[:, -1], :].clone()  # Ntri, 151
                PO_prior_obj = self.predicate_object_prior[1, positive_triplets[:, -1], :].clone()  # Ntri, 151
                inds = torch.arange(len(positive_triplets)).long().to(positive_triplets)
                PO_prior_subj[inds, positive_triplets[:, 0]] = 0
                PO_prior_subj[inds, positive_triplets[:, 1]] = 0
                PO_prior_subj[:, 0] = 0  # exclude bg
                PO_prior_obj[inds, positive_triplets[:, 0]] = 0
                PO_prior_obj[inds, positive_triplets[:, 1]] = 0
                PO_prior_obj[:, 0] = 0  # exclude bg
                if self.use_scene:
                    PO_prior_subj_imgs = PO_prior_subj.split(num_rels)
                    PO_prior_obj_imgs = PO_prior_obj.split(num_rels)
                    PSO_prior_subj, PSO_prior_obj = [], []
                    for prior_subj, prior_obj, scene_obj_prob in zip(PO_prior_subj_imgs, PO_prior_obj_imgs,
                                                                     object_in_scene_prob):
                        PSO_prior_subj.append(prior_subj * scene_obj_prob[None])
                        PSO_prior_obj.append(prior_obj * scene_obj_prob[None])
                    PSO_prior_subj = torch.cat(PSO_prior_subj, 0)
                    PSO_prior_obj = torch.cat(PSO_prior_obj, 0)
                else:
                    PSO_prior_subj = PO_prior_subj
                    PSO_prior_obj = PO_prior_obj

                # 4. construct the similarity
                # visual similarity:
                if self.use_visual_sim:
                    visual_norms = torch.norm(self.entity_pools.detach(), dim=1, keepdim=True)
                    visual_similarity = torch.div(
                        self.entity_pools.detach().mm(self.entity_pools.detach().transpose(1, 0)), visual_norms.mm(
                            visual_norms.transpose(1, 0)))
                    visual_similarity = (1 + visual_similarity) / 2
                    visual_similarity[torch.arange(len(visual_similarity)).long().to(source_triplets),
                                      torch.arange(len(visual_similarity)).long().to(source_triplets)] = 0
                    similarity = (self.semantic_similarity + visual_similarity) / 2
                else:
                    similarity = self.semantic_similarity

                PSO_sim_subj = PSO_prior_subj.unsqueeze(1) * similarity.unsqueeze(0)
                PSO_sim_obj = PSO_prior_obj.unsqueeze(1) * similarity.unsqueeze(0)

                # 5. replace the subject or object
                max_indices_subj = torch.max(PSO_sim_subj.view(PSO_sim_subj.shape[0], -1), -1)[1][:, None]
                cand_subjs = torch.cat((max_indices_subj / self.num_classes, max_indices_subj % self.num_classes), -1)
                max_indices_obj = torch.max(PSO_sim_obj.view(PSO_sim_obj.shape[0], -1), -1)[1][:, None]
                cand_objs = torch.cat((max_indices_obj / self.num_classes, max_indices_obj % self.num_classes), -1)
                rand_idxes = torch.rand(max_indices_subj.shape[0]).ge(0.5).long()
                replace_subjs = cand_subjs[torch.arange(cand_subjs.shape[0]).to(cand_subjs), rand_idxes]
                replace_objs = cand_objs[torch.arange(cand_subjs.shape[0]).to(cand_subjs), rand_idxes]

                # 6. generate the replaced samples and perform classification
                pair_reps_subj_replacement = prod_rep.clone()[positive_inds]
                pair_reps_subj_replacement[:, :self.head_config.roi_dim] = self.entity_pools.detach()[replace_subjs, :]
                pair_preds_subj_replacement = pair_pred.clone()[positive_inds]
                pair_preds_subj_replacement[:, 0] = replace_subjs

                pair_reps_obj_replacement = prod_rep.clone()[positive_inds]
                pair_reps_obj_replacement[:, self.head_config.roi_dim:] = self.entity_pools.detach()[replace_objs, :]
                pair_preds_obj_replacement = pair_pred.clone()[positive_inds]
                pair_preds_obj_replacement[:, 1] = replace_objs

                prod_rep = torch.cat((prod_rep, pair_reps_subj_replacement, pair_reps_obj_replacement), 0)
                pair_pred = torch.cat((pair_pred, pair_preds_subj_replacement, pair_preds_obj_replacement), 0)
                union_feats = torch.cat([union_feats, union_feats[positive_inds], union_feats[positive_inds]], 0)

        prod_rep = self.post_cat(prod_rep)
        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_feats)
            else:
                prod_rep = prod_rep * union_feats

        # OPTIONAL: pattern cls, not in testing phase
        if not self.traditional_rel_cls and not is_testing:
            pattern_rel_scores = self.out_pattern_rel(union_feats_point)

        rel_scores = self.rel_compress(prod_rep)
        if self.use_bias:
            rel_scores = rel_scores + self.freq_bias.index_with_labels(pair_pred.long())
        rel_prob = F.softmax(rel_scores[:num_pairs], -1)



        # make some changes: list to tensor or tensor to tuple
        if self.training:
            det_result.target_labels = torch.cat(det_result.target_labels, dim=-1)
            det_result.target_rel_labels = torch.cat(det_result.target_rel_labels,
                                                     dim=-1) if det_result.target_rel_labels is not None else None
        else:
            refine_obj_scores = refine_obj_scores.split(num_objs, dim=0)
            rel_scores = rel_scores.split(num_rels, dim=0)

        det_result.refine_scores = refine_obj_scores
        det_result.rel_scores = rel_scores

        # additional loss
        if not is_testing:
            head_spec_losses = {}
            loss_single_trans_matrix = feature_transform_regularizer(single_trans_matrix)
            head_spec_losses['loss_single_trans_matrix'] = loss_single_trans_matrix
            loss_union_trans_matrix = feature_transform_regularizer(union_trans_matrix)
            head_spec_losses['loss_union_trans_matrix'] = loss_union_trans_matrix

            if self.composition_augment:
                if self.use_scene:
                    head_spec_losses['loss_scene'] = nn.CrossEntropyLoss(ignore_index=-1)(scene_scores,
                                                                                          torch.cat(
                                                                                              det_result.target_scenes,
                                                                                              dim=-1))

            # traditional: do not compute in the relation_head, but here!
            if self.traditional_rel_cls:
                target_rel_labels = det_result.target_rel_labels
                if self.composition_augment and len(positive_inds) > 0:
                    all_rel_labels = torch.cat([target_rel_labels, target_rel_labels[positive_inds], target_rel_labels[positive_inds]], 0)
                else:
                    all_rel_labels = target_rel_labels
                head_spec_losses['loss_rel'] = CrossEntropyLoss()(rel_scores, all_rel_labels)

            # hierarchical loss
            if not self.traditional_rel_cls:
                target_rel_labels = det_result.target_rel_labels
                rel_samples_num = torch.bincount(target_rel_labels).float()
                rel_samples_num = torch.cat((rel_samples_num,
                                             torch.zeros(self.num_predicates - len(rel_samples_num)).to(
                                                 rel_samples_num)))
                rel_samples_num += 1
                cls_weights = (1 - self.beta_param) / (1 - self.beta_param ** rel_samples_num)
                target_pattern_labels = self.pattern_to_consecutive_labels[self.leaf_to_pattern[target_rel_labels]]

                # 1. regular relation loss: add weights
                if self.composition_augment and len(positive_inds) > 0:
                    all_rel_labels = torch.cat([target_rel_labels, target_rel_labels[positive_inds], target_rel_labels[positive_inds]], 0)
                else:
                    all_rel_labels = target_rel_labels
                head_spec_losses['loss_rel'] = CrossEntropyLoss()(rel_scores, all_rel_labels,
                                                                  weight=cls_weights[all_rel_labels])

                # 2. pattern level loss
                induced_probs = top_down_induce(rel_prob, self.predicate_hierarchy, self.augmented_predicates,
                                                solveroot=self.solveroot)
                induced_weights = \
                    top_down_induce(cls_weights[None], self.predicate_hierarchy, self.augmented_predicates,
                                    solveroot=self.solveroot)[0]

                # (2) compute the predicted-pattern level loss
                induced_pattern_weights = induced_weights[target_pattern_labels]
                head_spec_losses['loss_pattern_rel'] = CrossEntropyLoss()(
                    pattern_rel_scores,
                    target_pattern_labels,
                    weight=induced_pattern_weights)

                # (3) compute the tree loss
                # """
                head_spec_losses['loss_predicate_tree'] = TreeLoss(info=self.treelossinfo)(induced_probs,
                                                                                           target_rel_labels,
                                                                                           self.predicate_hierarchy,
                                                                                           self.augmented_predicates,
                                                                                           induced_weights)
                # """
                # (4) minimize the predicted pattern and induced pattern distribution:
                # head_spec_losses['loss_KL_tree'] = nn.KLDivLoss()(F.log_softmax(pattern_rel_scores),
                #                                                  F.softmax(torch.index_select(induced_probs,
                #                                                                               1,
                #                                                                               self.pattern_indices)))

            det_result.head_spec_losses = head_spec_losses


        return det_result
