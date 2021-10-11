# ---------------------------------------------------------------
# normal.py
# Set-up time: 2021/2/3 上午10:25
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------


from __future__ import division

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from mmcv.cnn import normal_init, kaiming_init
from mmdet import ops
from mmdet.core import force_fp32
from mmdet.core.utils import enumerate_by_image
from ..registry import RELATION_ROI_EXTRACTORS
import numpy as np


@RELATION_ROI_EXTRACTORS.register_module
class NormalExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 bbox_roi_layer,
                 in_channels,
                 featmap_strides,
                 roi_out_channels=256,
                 fc_out_channels=1024,
                 finest_scale=56,
                 with_avg_pool=False,
                 single=True,
                 num_fcs=2,
                 dropout=0.5,
                 spatial_cfg=dict(type='fc', fc_in_dim=6, fc_out_dim=64)):
        super(NormalExtractor, self).__init__()
        self.roi_feat_size = _pair(bbox_roi_layer.get('out_size', 7))
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.roi_out_channels = roi_out_channels
        self.fc_out_channels = fc_out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.fp16_enabled = False
        self.with_avg_pool = with_avg_pool

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area

        # set some caches
        self._roi_feats = None

        # build visual head: extract visual features.
        assert bbox_roi_layer is not None
        self.bbox_roi_layers = self.build_roi_layers(bbox_roi_layer, featmap_strides)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.single = single

        self.num_fcs = num_fcs
        self.bbox_head = nn.ModuleList()
        for i in range(self.num_fcs):
            fc_in_channels = in_channels if i == 0 else self.fc_out_channels
            self.bbox_head.append(nn.Linear(fc_in_channels, self.fc_out_channels))

        # build spatial_head
        self.with_spatial = False
        self.spatial_cfg = spatial_cfg
        if spatial_cfg is not None:
            self.with_spatial = True

        if self.with_spatial:
            if spatial_cfg['type'] == 'fc':
                self.spatial_fc = nn.Linear(spatial_cfg['fc_in_dim'], spatial_cfg['fc_out_dim'])
            else:
                raise NotImplementedError

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    @property
    def roi_feats(self):
        return self._roi_feats

    def init_weights(self):
        if hasattr(self, 'bbox_head'):
            for m in self.bbox_head:
                if isinstance(m, nn.Linear):
                    kaiming_init(m, distribution='uniform', a=1)

        if hasattr(self, 'spatial_fc'):
            kaiming_init(self.spatial_fc, distribution='uniform', a=1)

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def roi_rescale(self, rois, scale_factor):
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1] + 1
        h = rois[:, 4] - rois[:, 2] + 1
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5 + 0.5
        x2 = cx + new_w * 0.5 - 0.5
        y1 = cy - new_h * 0.5 + 0.5
        y2 = cy + new_h * 0.5 - 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    def roi_forward(self, roi_layers, feats, rois, roi_scale_factor=None):
        if len(feats) == 1:
            roi_feats = roi_layers[0](feats[0], rois)
        else:
            out_size = roi_layers[0].out_size
            num_levels = self.num_inputs
            target_lvls = self.map_roi_levels(rois, num_levels)
            roi_feats = feats[0].new_zeros(
                rois.size(0), self.roi_out_channels, *out_size)
            if roi_scale_factor is not None:
                rois = self.roi_rescale(rois, roi_scale_factor)

            for i in range(num_levels):
                inds = target_lvls == i
                if inds.any():
                    rois_ = rois[inds, :]
                    roi_feats_t = roi_layers[i](feats[i], rois_)
                    roi_feats[inds] = roi_feats_t
        return roi_feats

    def single_roi_forward(self, feats, rois, roi_scale_factor=None):
        # 1. Use the visual and spatial head to extract roi features.
        roi_feats = self.roi_forward(self.bbox_roi_layers, feats, rois, roi_scale_factor)

        # 2. use the head to vectorize the features
        if self.with_avg_pool:
            roi_feats = self.avg_pool(roi_feats)

        roi_feats = roi_feats.flatten(1)

        for fc in self.bbox_head:
            roi_feats = self.dropout(self.relu(fc(roi_feats)))

        self._roi_feats = roi_feats
        return roi_feats

    def union_roi_forward(self, feats, rois, rel_pair_idx, roi_scale_factor=None):
        num_images = feats[0].size(0)
        assert num_images == len(rel_pair_idx)
        rel_pair_index = []
        im_inds = rois[:, 0]
        acc_obj = 0
        for i, s, e in enumerate_by_image(im_inds):
            num_obj_i = e - s
            rel_pair_idx_i = rel_pair_idx[i].clone()
            rel_pair_idx_i[:, 0] += acc_obj
            rel_pair_idx_i[:, 1] += acc_obj
            acc_obj += num_obj_i
            rel_pair_index.append(rel_pair_idx_i)
        rel_pair_index = torch.cat(rel_pair_index, 0)

        # prepare the union rois
        head_rois = rois[rel_pair_index[:, 0], :]
        tail_rois = rois[rel_pair_index[:, 1], :]
        union_rois = torch.stack([head_rois[:, 0],
                                  torch.min(head_rois[:, 1], tail_rois[:, 1]),
                                  torch.min(head_rois[:, 2], tail_rois[:, 2]),
                                  torch.max(head_rois[:, 3], tail_rois[:, 3]),
                                  torch.max(head_rois[:, 4], tail_rois[:, 4])], -1)
        intersect_rois = torch.stack([head_rois[:, 0],
                                      torch.max(head_rois[:, 1], tail_rois[:, 1]),
                                      torch.max(head_rois[:, 2], tail_rois[:, 2]),
                                      torch.min(head_rois[:, 3], tail_rois[:, 3]),
                                      torch.min(head_rois[:, 4], tail_rois[:, 4])], -1)

        # Use the visual and spatial head to extract roi features.
        roi_feats = self.roi_forward(self.bbox_roi_layers, feats, union_rois, roi_scale_factor)

        if self.with_avg_pool:
            roi_feats = self.avg_pool(roi_feats)

        roi_feats = roi_feats.flatten(1)

        for fc in self.bbox_head:
            roi_feats = self.dropout(self.relu(fc(roi_feats)))

        spatial_feats = None
        if self.with_spatial:
            # construct the basic 6 dimension relative spatial feature and project it to higher dimension
            head_whs = (head_rois[:, 3:] - head_rois[:, 1:3] + 1).clamp(min=1e-7)
            tail_whs = (tail_rois[:, 3:] - tail_rois[:, 1:3] + 1).clamp(min=1e-7)
            union_areas = (union_rois[:, 3] - union_rois[:, 1] + 1) * (union_rois[:, 4] - union_rois[:, 2] + 1)
            intersect_areas = ((intersect_rois[:, 3] - intersect_rois[:, 1] + 1) \
                              * (intersect_rois[:, 4] - intersect_rois[:, 2] + 1)).clamp(min=1e-7)
            head_xys = (head_rois[:, 3:] + head_rois[:, 1:3]) * 0.5
            tail_xys = (tail_rois[:, 3:] + tail_rois[:, 1:3]) * 0.5

            geo_feats = torch.cat([(tail_xys[:, 0:1] - head_xys[:, 0:1]) / torch.sqrt(
                                        head_whs[:, 0:1] * head_whs[:, 1:2]),
                                   (tail_xys[:, 1:2] - head_xys[:, 1:2]) / torch.sqrt(
                                        head_whs[:, 0:1] * head_whs[:, 1:2]),
                                   torch.sqrt((tail_whs[:, 0:1] * tail_whs[:, 1:2]) /
                                              (head_whs[:, 0:1] * head_whs[:, 1:2])),
                                   head_whs[:, 0:1] / head_whs[:, 1:2],
                                   tail_whs[:, 0:1] / tail_whs[:, 1:2],
                                   (intersect_areas / union_areas)[:, None]
                                   ], -1)
            spatial_feats = self.spatial_fc(geo_feats)
        if spatial_feats is not None:
            return torch.cat((roi_feats, spatial_feats), -1)
        else:
            return roi_feats

    @force_fp32(apply_to=('feats',), out_fp16=True)
    def forward(self, feats, rois, rel_pair_idx=None, roi_scale_factor=None):
        if rois.shape[0] == 0:
            return torch.from_numpy(np.empty((0, self.fc_out_channels))).to(feats[0])
        if not self.single:
            assert rel_pair_idx is not None
            return self.union_roi_forward(feats, rois, rel_pair_idx, roi_scale_factor)
        else:
            return self.single_roi_forward(feats, rois, roi_scale_factor)
