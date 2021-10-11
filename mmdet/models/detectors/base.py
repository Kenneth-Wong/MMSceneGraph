from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch.nn as nn

from mmdet.core import auto_fp16, get_classes, tensor2imgs, get_predicates, get_point_from_mask
from mmdet.utils import print_log
from mmdet.patches import (color_palette, imshow_det_bboxes, imdraw_sg)
from mmdet.models.relation_heads.approaches import Result
import cv2
from mmcv.image import imwrite
import os


class BaseDetector(nn.Module, metaclass=ABCMeta):
    """Base class for detectors"""

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_relation(self):
        return hasattr(self, 'relation_head') and self.relation_head is not None

    @property
    def with_attr(self):
        return hasattr(self, 'attr_head') and self.attr_head is not None

    @property
    def with_relcaption(self):
        return hasattr(self, 'relcaption_head') and self.relcaption_head is not None

    @property
    def with_downstream_caption(self):
        return hasattr(self, 'downstream_caption_head') and self.downstream_caption_head is not None

    @property
    def with_saliency(self):
        return hasattr(self, 'saliency_detector') and self.saliency_detector is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (list[Tensor]): list of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

             **kwargs: specific to concrete implementation
        """
        pass

    async def async_simple_test(self, img, img_meta, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    @abstractmethod
    def relation_simple_test(self, img, img_meta, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')

    async def aforward_test(self, *, img, img_meta, **kwargs):
        for var, name in [(img, 'img'), (img_meta, 'img_meta')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img)
        if num_augs != len(img_meta):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img), len(img_meta)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = img[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return await self.async_simple_test(img[0], img_meta[0], **kwargs)
        else:
            raise NotImplementedError

    def forward_test(self, imgs, img_metas, relation_mode=False, relcaption_mode=False,
                     downstream_caption_mode=False, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
            relation_mode: (Bool): Whether to test for relation or detection.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        key_first = kwargs.pop('key_first', False)
        if relation_mode:
            assert num_augs == 1
            return self.relation_simple_test(imgs[0], img_metas[0], key_first=key_first, **kwargs)

        if relcaption_mode:
            assert num_augs == 1
            return self.relcaption_simple_test(imgs[0], img_metas[0], **kwargs)

        if downstream_caption_mode:
            assert num_augs == 1
            return self.downstream_caption_simple_test(imgs[0], img_metas[0], **kwargs)

        if num_augs == 1:
            """
            proposals (List[List[Tensor]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. The Tensor should have a shape Px4, where
                P is the number of proposals.
            """
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_meta, return_loss=True, relation_mode=False, relcaption_mode=False,
                downstream_caption_mode=False, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, relation_mode, relcaption_mode, downstream_caption_mode, **kwargs)

    def show_sg_result(self, data, result, work_dir, dataset=None, score_thr=0.3, num_rel=20, backend='graphviz'):
        """
        This API is for showing results for research rather than demo. It means that this is not the best one for
        demonstrating to humans.

        if the data contains gt information,
        """
        assert isinstance(result, Result)
        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        filenames = [m['filename'].split('/')[-1][:-4] for m in img_metas]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        if dataset is None:
            class_names = self.CLASSES
            predicate_names = self.PREDICATES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
            predicate_names = get_predicates(dataset)
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))

        gt_bboxes, gt_labels, gt_rels = None, None, None
        if 'gt_bboxes' in data:
            gt_bboxes = data['gt_bboxes'][0].data[0]
            gt_bboxes = [d.numpy() for d in gt_bboxes][0]
        if 'gt_labels' in data:
            gt_labels = data['gt_labels'][0].data[0]
            gt_labels = [d.numpy() - 1 for d in gt_labels][0]  # remember -1 for visualization
        if 'gt_rels' in data:
            gt_rels = data['gt_rels'][0].data[0]
            gt_rels[:, -1] = gt_rels[:, -1] - 1
            gt_rels = gt_rels.numpy()
            # tmp = []
            # for d in gt_rels:
            #     d[:, -1] = d[:, -1] - 1
            #     tmp.append(d.numpy())
            # gt_rels = tmp[0]

        for img, img_meta, filename in zip(imgs, img_metas, filenames):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            # main result for visualization
            bboxes = result.refine_bboxes  # (array [N, 5]): bboxes + obj_scores
            rels = result.rels.copy()
            rels[:, -1] = rels[:, -1] - 1  # (array [Nr, 3]): pair indexes, rel labels
            rels = rels[:num_rel, :]  # default
            object_classes = result.refine_labels - 1  # (array [N, ]): object classes, 1-based

            # auxiliary information, you can choose to visualize it or not
            # (array [N, ]): object confidences
            object_confs = result.refine_dists[np.arange(len(result.refine_labels)), result.refine_labels]
            # (array [Nr, ]): predicate confidences
            rel_confs = result.rel_dists[np.arange(len(result.rels)), result.rels[:, -1]][:num_rel]
            # (array [Nr, ]): triplet scores
            triplet_confs = result.triplet_scores[:num_rel]

            # display the relationship on the images and draw the abstract graph with graphviz
            imdraw_sg(img_show, bboxes, object_classes, rels, gt_bboxes, gt_labels, gt_rels,
                      pred_scores=object_confs, pred_rel_scores=rel_confs, pred_triplet_scores=triplet_confs,
                      class_names=class_names, predicate_names=predicate_names, score_thr=score_thr,
                      work_dir=work_dir, filename=filename, backend=backend)

    def show_result(self, data, result, dataset=None, score_thr=0.3, palette=dict(palette='pastel', n_colors=7),
                    **kwargs):
        # Result object, for scene graph
        if isinstance(result, Result):
            self.show_sg_result(self, data, result, dataset, score_thr, **kwargs)
            return
            # tuple, may contains 2 to three outcomes, bbox, [optional]masks, [optional]points
        elif isinstance(result, tuple):
            if len(result) == 2:
                bbox_result, segm_result = result
                point_result = None
            elif len(result) == 3:
                bbox_result, segm_result, point_result = result
            else:
                raise NotImplementedError
                # list, only single result,:bbox
        elif isinstance(result, list):
            bbox_result, segm_result, point_result = result, None, None

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

        work_dir = kwargs.get('work_dir', None)
        index = kwargs.get('index')
        if work_dir is not None:
            img_dir = os.path.join(work_dir, 'images')
            bbox_dir = os.path.join(work_dir, 'bboxes')
            point_dir = os.path.join(work_dir, 'points')

        for (img, img_meta) in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]
            if work_dir is not None:
                imwrite(img_show, os.path.join(img_dir, str(index)+'.png'))

            bboxes = np.vstack(bbox_result)
            masktrans_pts = []
            # draw segmentation masks
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                colors = color_palette(**palette)
                num_colors = len(colors)
                for idx, i in enumerate(inds):
                    # color_mask = np.random.randint(
                    #    0, 256, (1, 3), dtype=np.uint8)
                    color_mask = np.array(colors[idx % num_colors], dtype=np.uint8)
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    show_mask_or_point = kwargs.get('mask_or_point', 'mask')
                    if show_mask_or_point == 'mask':
                        img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
                    elif show_mask_or_point == 'point':
                        # directly extract the points from the mask contours
                        points = get_point_from_mask([mask.astype(np.uint8)],
                                                     [bboxes[i, :-1].astype(np.int32)], sample_num=50)[0]
                        points = points.reshape((-1, 2))
                        if work_dir is None:
                            for pt in points:
                                cv2.circle(img_show, (int(pt[0]), int(pt[1])), 2, colors[idx % num_colors], 4,
                                           lineType=cv2.LINE_AA)
                        else:
                            masktrans_pts.append(points)
                    else:
                        raise NotImplementedError
                if work_dir is not None:
                    mmcv.dump(masktrans_pts, os.path.join(point_dir, str(index)+'.pickle'))

            if point_result is not None:
                colors = color_palette(**palette)
                num_colors = len(colors)
                point_size = 2
                thickness = 4
                acc_cnt = 0
                for clss_id, clss_points in enumerate(point_result):
                    for entity_local_id, points in enumerate(clss_points):
                        color = colors[acc_cnt % num_colors]
                        for edge in points:
                            for pt in edge:
                                cv2.circle(img_show, (int(pt[0]), int(pt[1])), point_size, color, thickness,
                                           lineType=cv2.LINE_AA)
                        acc_cnt += 1

            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            if work_dir is None:
                imshow_det_bboxes(
                    img_show,
                    bboxes,
                    labels,
                    class_names=class_names,
                    score_thr=score_thr,
                    palette=palette)
            else:
                inds = bboxes[:, -1] > score_thr
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                mmcv.dump([bboxes, labels], os.path.join(bbox_dir, str(index)+'.pickle'))
