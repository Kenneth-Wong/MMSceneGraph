import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmdet.core import auto_fp16, force_fp32, mask_target, get_point_from_mask
from mmdet.ops import ConvModule, build_upsample_layer
from mmdet.ops.carafe import CARAFEPack
from ..builder import build_loss
from ..registry import HEADS


@HEADS.register_module
class TransferMaskHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=81,
                 class_agnostic=False,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 transfer_cfg=dict(num_fc=2, fc_in=5120, hidden_neurons=[1024, 256], relu='LeakyReLU', mlp_fusion=True),
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(TransferMaskHead, self).__init__()
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
            None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear", "carafe"'.format(
                    self.upsample_cfg['type']))

        self.transfer_cfg = transfer_cfg.copy()
        if self.transfer_cfg['relu'] not in [None, 'ReLU', 'LeakyReLU']:
            raise ValueError('Invalid activation method {}, accpeted methods are "ReLU", "LeakyReLU"'.format(
                self.transfer_cfg['relu']))
        assert len(self.transfer_cfg['hidden_neurons']) == self.transfer_cfg['num_fc']

        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor')
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.update(
                in_channels=upsample_in_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor)
        elif self.upsample_method == 'carafe':
            upsample_cfg_.update(
                channels=upsample_in_channels, scale_factor=self.scale_factor)
        else:
            # suppress warnings
            align_corners = (None
                             if self.upsample_method == 'nearest' else False)
            upsample_cfg_.update(
                scale_factor=self.scale_factor,
                mode=self.upsample_method,
                align_corners=align_corners)
        self.upsample = build_upsample_layer(upsample_cfg_)

        transfer_modules = []
        for i in range(self.transfer_cfg.get('num_fc', 2)):
            if i == 0:
                feat_in = self.transfer_cfg.get('fc_in', 5120)
            else:
                feat_in = self.transfer_cfg['hidden_neurons'][i - 1]
            feat_out = self.transfer_cfg['hidden_neurons'][i]
            transfer_modules.append(nn.Linear(feat_in, feat_out))
            if self.transfer_cfg['relu'] == 'ReLU':
                relu = nn.ReLU(inplace=True)
            elif self.transfer_cfg['relu'] == 'LeakyReLU':
                relu = nn.LeakyReLU(inplace=True)
            transfer_modules.append(relu)
        self.transfer = nn.Sequential(*transfer_modules)

        if self.transfer_cfg['mlp_fusion']:
            if self.upsample is not None:
                feat_in = self.conv_out_channels * (roi_feat_size * 2) ** 2
                feat_out = (roi_feat_size * 2) ** 2
            else:
                feat_in = self.conv_out_channels * (roi_feat_size ** 2)
                feat_out = roi_feat_size ** 2
            self.fusion_mlp = nn.Linear(feat_in, feat_out)
        else:
            self.fusion_mlp = None

        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for m in [self.upsample]:
            if m is None:
                continue
            elif isinstance(m, CARAFEPack):
                m.init_weights()
            else:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

        for module_list in [self.transfer]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
        if self.fusion_mlp is not None:
            nn.init.xavier_uniform_(self.fusion_mlp.weight)
            nn.init.constant_(self.fusion_mlp.bias, 0)

    @auto_fp16()
    def forward(self, x, det_weight):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        batch_size, conv_out_channels, conv_out_h, conv_out_w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        if self.fusion_mlp is not None:
            mlp_x = self.fusion_mlp(x.view(batch_size, -1))
            mlp_x = mlp_x.view(batch_size, 1, conv_out_h, conv_out_w)

        mask_weight = self.transfer(det_weight).T
        mask_opt = torch.matmul(x.permute(0, 2, 3, 1).contiguous().view(-1, conv_out_channels), mask_weight)
        x = mask_opt.view(-1, conv_out_h, conv_out_w, self.num_classes)
        x = x.permute(0, 3, 1, 2).contiguous()

        if self.fusion_mlp is not None:
            x = x + mlp_x
        mask_pred = x
        return mask_pred

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    @force_fp32(apply_to=('mask_pred',))
    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets,
                                       torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale, with_point=False):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)
        # when enabling mixed precision training, mask_pred may be float16
        # numpy array
        mask_pred = mask_pred.astype(np.float32)

        format_mask_result = rcnn_test_cfg.get('format_mask_result', True)
        if format_mask_result:
            cls_segms = [[] for _ in range(self.num_classes - 1)]
        else:
            cls_segms = []
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        if with_point:
            if format_mask_result:
                contour_points = [[] for _ in range(self.num_classes - 1)]
            else:
                contour_points = []

        for i in range(bboxes.shape[0]):
            if not isinstance(scale_factor, (float, np.ndarray)):
                scale_factor = scale_factor.cpu().numpy()
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)
            # transform to the whole image mask
            im_mask_np = np.zeros((img_h, img_w), dtype=np.uint8)
            im_mask_np[bbox[1]:min(bbox[1] + h, img_h), bbox[0]:min(bbox[0] + w, img_w)] = \
                bbox_mask[:min(h, img_h - bbox[1]), :min(w, img_w - bbox[0])]

            # When for mask roi align/pooling, set the crop_mask=True; otherwise, it must be false.
            if rcnn_test_cfg.get('crop_mask', False):
                mask_result = bbox_mask
            else:
                mask_result = im_mask_np

            if rcnn_test_cfg.get('to_tensor', False):
                im_mask = torch.from_numpy(mask_result).to(det_labels)
            else:
                im_mask = mask_result.copy()

            if rcnn_test_cfg.get('rle_mask_encode', True):
                if isinstance(im_mask, torch.Tensor):
                    mask_for_encode = im_mask.cpu().numpy().astype(np.uint8)
                else:
                    mask_for_encode = im_mask
                rle = mask_util.encode(
                    np.array(mask_for_encode[:, :, np.newaxis], order='F'))[0]
                if format_mask_result:
                    cls_segms[label - 1].append(rle)
                else:
                    cls_segms.append(rle)
            else:
                if format_mask_result:
                    cls_segms[label - 1].append(im_mask)
                else:
                    cls_segms.append(im_mask)

            # point: you must use the image mask rather than the box mask to get the points
            if with_point:
                points = get_point_from_mask([im_mask_np], [bbox],
                                               rcnn_test_cfg.get('mask_size', 56),
                                               rcnn_test_cfg.get('sample_num', 729),
                                               rcnn_test_cfg.get('dist_sample_thr', 1))[0]
                if rcnn_test_cfg.get('to_tensor', False):
                    points = torch.from_numpy(points).to(det_bboxes)
                if format_mask_result:
                    contour_points[label - 1].append(points)
                else:
                    contour_points.append(points)

        if with_point:
            return cls_segms, contour_points
        else:
            return cls_segms
