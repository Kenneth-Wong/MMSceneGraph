# ---------------------------------------------------------------
# kern_head.py
# Set-up time: 2021/4/6 22:55
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

from ..registry import HEADS
import torch
import torch.nn as nn
from .relation_head import RelationHead
from mmdet.datasets import build_dataset
from .approaches import KERNContext
from mmcv.cnn import xavier_init, normal_init
import numpy as np
import os


@HEADS.register_module
class KERNHead(RelationHead):
    def __init__(self, **kwargs):
        super(KERNHead, self).__init__(**kwargs)
        # post decoding
        self.use_vision = self.head_config.use_vision
        self.hidden_dim = self.head_config.hidden_dim
        self.context_pooling_dim = self.head_config.context_pooling_dim

        # load and preprocess the knowledge
        self.register_buffer('rel_knowledge', torch.exp(self.statistics['pred_dist']))
        cache_dir = self.dataset_config.pop('cooccur_cache', None)
        print('Loading Cooccur Statistics...')
        if cache_dir is None:
            raise FileNotFoundError('The cache_dir for caching the cooccur statistics is not provided.')
        if os.path.exists(cache_dir):
            self.register_buffer('obj_knowledge',
                                 torch.from_numpy(torch.load(cache_dir, map_location=torch.device("cpu"))))
        else:
            dataset = build_dataset(self.dataset_config)
            obj_cooccur = dataset.get_cooccur_statistics()
            self.register_buffer('obj_knowledge', torch.from_numpy(obj_cooccur))
            torch.save(obj_cooccur, cache_dir)

        self.context_layer = KERNContext(self.head_config, self.obj_classes, self.rel_classes,
                                         self.obj_knowledge, self.rel_knowledge)

    def init_weights(self):
        self.bbox_roi_extractor.init_weights()
        self.relation_roi_extractor.init_weights()
        self.context_layer.init_weights()

    def forward(self,
                img,
                img_meta,
                det_result,
                gt_result=None,
                is_testing=False,
                ignore_classes=None):
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

        refine_obj_scores, obj_preds, rel_scores = self.context_layer(roi_feats, union_feats, det_result)

        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]
        # make some changes: list to tensor or tensor to tuple
        if self.training:
            det_result.target_labels = torch.cat(det_result.target_labels, dim=-1)
            det_result.target_rel_labels = torch.cat(det_result.target_rel_labels,
                                                     dim=-1) if det_result.target_rel_labels is not None else None
        else:
            refine_obj_scores = refine_obj_scores.split(num_objs, dim=0)
            rel_scores = rel_scores.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage

        det_result.refine_scores = refine_obj_scores
        det_result.rel_scores = rel_scores
        return det_result
