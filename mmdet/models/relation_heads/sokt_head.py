# ---------------------------------------------------------------
# sokt_head.py
# Set-up time: 2020/6/22 下午4:58
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

from ..registry import HEADS
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from .. import builder
from .relation_head import RelationHead
from mmdet.datasets import build_dataset
from .approaches import LSTMContext
from mmcv.cnn import xavier_init, normal_init
from mmdet.core import bbox2roi
from .approaches.motif_util import to_onehot


@HEADS.register_module
class SoktHead(RelationHead):
    def __init__(self, **kwargs):
        super(SoktHead, self).__init__(**kwargs)

        # post decoding
        self.roi_dim = self.head_config.roi_dim
        self.alpha = self.head_config.alpha
        self.margin = self.head_config.margin
        self.loss_dist_weight = 0.01
        self.atten_scene = nn.Linear(self.roi_dim, self.roi_dim)
        self.atten_knowledge = nn.Linear(self.roi_dim, self.roi_dim)
        self.pre_cw = nn.Linear(self.roi_dim, self.num_predicates)

        self.scene_compress = nn.Linear(self.roi_dim, self.num_classes - 1, bias=True)
        self.obj_compress = nn.Linear(self.roi_dim, self.num_classes, bias=True)
        self.initial_rel_compress = nn.Linear(self.roi_dim, self.num_predicates, bias=True)
        self.final_rel_compress = nn.Linear(self.roi_dim, self.num_predicates, bias=True)
        self.register_parameter("code_words", nn.Parameter(torch.rand(self.num_predicates, self.roi_dim)))

        self.loss_scene = builder.build_loss(dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
        self.loss_init_relation = builder.build_loss(dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))

        cache_dir = kwargs['dataset_config'].pop('object_cache', None)
        print('Loading object statistics...')
        if cache_dir is None:
            raise FileNotFoundError('The cache_dir for caching the statistics is not provided.')
        if os.path.exists(cache_dir):
            obj_statistics = torch.load(cache_dir, map_location=torch.device("cpu"))
        else:
            dataset = build_dataset(kwargs['dataset_config'])
            obj_statistics = dataset.get_object_statistics()
            torch.save(obj_statistics, cache_dir)
        self.object_weight = obj_statistics['prob']
        print('\n Object Statistics created!')

    def init_weights(self):
        self.bbox_roi_extractor.init_weights()
        self.relation_roi_extractor.init_weights()
        for module in (self.atten_scene, self.atten_knowledge, self.pre_cw,
                       self.scene_compress, self.obj_compress, self.initial_rel_compress, self.final_rel_compress):
            xavier_init(module)

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

        roi_feats, union_feats, det_result = self.frontend_features(img, img_meta, det_result, gt_result)
        if roi_feats.shape[0] == 0:
            return det_result

        # construct the global scene boxes
        scene_rois = []
        for img_id, meta in enumerate(img_meta):
            scene_box_i = torch.Tensor([img_id, 0., 0., meta['img_shape'][0]-1, meta['img_shape'][1]-1]).to(roi_feats)[None]
            scene_rois.append(scene_box_i)
        scene_rois = torch.cat(scene_rois, 0)
        scene_feats = self.bbox_roi_extractor(img, img_meta, scene_rois)
        num_objs = [len(b) for b in det_result.bboxes]
        num_rels = [len(pair_idx) for pair_idx in det_result.rel_pair_idxes]
        expand_scene_feats = torch.cat([scene_feats[i][None].expand(num_objs[i], -1) for i in range(len(num_objs))], 0)
        scene_atten = F.relu(self.atten_scene(expand_scene_feats + roi_feats))
        so_roi_feats = roi_feats + expand_scene_feats * scene_atten

        initial_triplet_feats = []
        for so_roi_feat, union_feat, pair_idx in zip(so_roi_feats.split(num_objs, 0),
                                                     union_feats.split(num_rels, 0),
                                                     det_result.rel_pair_idxes):
            triplet_feat = so_roi_feat[pair_idx[:, 0]] * so_roi_feat[pair_idx[:, 1]] * union_feat
            initial_triplet_feats.append(triplet_feat)
        initial_triplet_feats = torch.cat(initial_triplet_feats, 0)

        prob = F.softmax(self.pre_cw(initial_triplet_feats), dim=-1)
        hallu_feats = torch.mm(prob, self.code_words)
        knowledge_atten = F.relu(self.atten_knowledge(initial_triplet_feats + hallu_feats))
        hallu_feats = initial_triplet_feats + knowledge_atten * hallu_feats
        triplet_feats = self.alpha * torch.max(prob, dim=-1)[0][:, None] * hallu_feats

        if self.use_gt_label:
            refine_obj_scores = to_onehot(torch.cat(det_result.labels, -1), self.num_classes)
        else:
            refine_obj_scores = self.obj_compress(so_roi_feats)

        rel_scores = self.final_rel_compress(triplet_feats)

        # make some changes: list to tensor or tensor to tuple
        if self.training:
            det_result.target_labels = torch.cat(det_result.target_labels, dim=-1)
            det_result.target_rel_labels = torch.cat(det_result.target_rel_labels, dim=-1)
        else:
            refine_obj_scores = refine_obj_scores.split(num_objs, dim=0)
            rel_scores = rel_scores.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage

        det_result.refine_scores = refine_obj_scores
        det_result.rel_scores = rel_scores

        # prepare the specific loss for sokt_head:
        if not is_testing:
            gt_labels = gt_result.labels
            scene_labels = []
            for labels in gt_labels:
                scene_label = torch.zeros(self.num_classes - 1).to(labels)
                scene_label.index_fill_(0, labels-1, 1)
                scene_labels.append(scene_label[None])
            scene_labels = torch.cat(scene_labels, 0)
            loss_scene_input = (self.scene_compress(scene_feats), scene_labels,
                                torch.from_numpy(self.object_weight).to(scene_feats))

            initial_rel_scores = self.initial_rel_compress(initial_triplet_feats)
            target_rel_labels = det_result.target_rel_labels
            if isinstance(target_rel_labels, (tuple, list)):
                target_rel_labels = torch.cat(target_rel_labels, -1)
            loss_initial_relation_input = (initial_rel_scores, target_rel_labels)

            loss_cw_input = (initial_triplet_feats, target_rel_labels)

            det_result.head_spec_losses = [loss_scene_input, loss_initial_relation_input, loss_cw_input]

        return det_result

    def loss(self, det_result):
        losses = dict()
        basic_losses = super(SoktHead, self).loss(det_result)
        losses.update(basic_losses)

        head_spec_losses_input = det_result.head_spec_losses
        loss_scene_input, loss_init_relation_input, loss_cw_input = head_spec_losses_input[0], \
                                                                    head_spec_losses_input[1], \
                                                                    head_spec_losses_input[2]

        loss_scene = self.loss_scene(loss_scene_input[0], loss_scene_input[1], loss_scene_input[2])
        losses['loss_scene'] = loss_scene

        loss_init_relation = self.loss_init_relation(loss_init_relation_input[0], loss_init_relation_input[1])
        losses['loss_init_relaiton'] = loss_init_relation

        initial_triplet_feats, target_rel_labels = loss_cw_input
        dist = torch.norm(initial_triplet_feats[:, None, :] - self.code_words[None], p=1, dim=-1)
        indices = (torch.arange(dist.shape[0]).to(target_rel_labels), target_rel_labels)
        loss_label_dist = F.relu(self.margin - dist)
        loss_label_dist.index_put_(indices, dist[indices[0], indices[1]])
        loss_dist = self.loss_dist_weight * torch.mean(torch.sum(loss_label_dist, dim=1))
        losses['loss_dist'] = loss_dist

        return losses

