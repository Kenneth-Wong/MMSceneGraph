import torch
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import cv2
from mmdet.core import get_classes, tensor2imgs

from mmdet.core import bbox2result, bbox_mapping_back, multiclass_nms
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.patches import (color_palette, imshow_det_bboxes)


@DETECTORS.register_module
class RepPointsDetector(SingleStageDetector):
    """RepPoints: Point Set Representation for Object Detection.

        This detector is the implementation of:
        - RepPoints detector (https://arxiv.org/pdf/1904.11490)
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RepPointsDetector,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained)

    def merge_aug_results(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).

        Returns:
            tuple: (bboxes, scores)
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores

    def aug_test(self, imgs, img_metas, rescale=False):
        # recompute feats to save memory
        feats = self.extract_feats(imgs)

        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.bbox_head(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, False)
            det_bboxes, det_scores = self.bbox_head.get_bboxes(*bbox_inputs)[0]
            aug_bboxes.append(det_bboxes)
            aug_scores.append(det_scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_results(
            aug_bboxes, aug_scores, img_metas)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                self.test_cfg.score_thr,
                                                self.test_cfg.nms,
                                                self.test_cfg.max_per_img)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        return bbox_results

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes_reppoints(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, _, det_labels in bbox_list
        ]

        # borrow the method for bbox result transform.
        reppoint_results = [
            bbox2result(det_reppoints, det_labels, self.bbox_head.num_classes)
            for _, det_reppoints, det_labels in bbox_list
        ]

        return bbox_results[0], reppoint_results[0]

    def show_result(self, data, result, dataset=None, score_thr=0.3, palette=dict(palette='pastel', n_colors=7)):
        if isinstance(result, tuple):
            bbox_result, reppoint_result = result
        else:
            bbox_result, reppoint_result = result, None

        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            bboxes = np.vstack(bbox_result)

            # draw reppoints

            if reppoint_result is not None:
                colors = color_palette(**palette)
                num_colors = len(colors)
                point_size = 2
                thickness = 4
                points = np.vstack(reppoint_result).astype(np.int32)
                for idx, obj in enumerate(points):
                    if idx >= 2:
                        break
                    color = colors[idx % num_colors]
                    for i in range(self.bbox_head.num_points):
                        cv2.circle(img_show, (obj[i*2], obj[i*2+1]), point_size, color, thickness, lineType=cv2.LINE_AA)

            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                class_names=class_names,
                score_thr=score_thr,
                palette=palette)
