# ---------------------------------------------------------------
# vrd.py
# Set-up time: 2020/4/6 下午3:31
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import logging
import os.path as osp
import tempfile

import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import h5py
from collections import defaultdict, Counter
import random

from mmdet.core import eval_recalls
from mmdet.utils import print_log
from .custom import CustomDataset
from .coco import CocoDataset
from .visualgenome import get_VG_freq, get_VG_obj_freq, get_VG_statistics
from .registry import DATASETS
from .pipelines import Compose
from mmdet.core.bbox.geometry import bbox_overlaps
from mmdet.core import vg_evaluation
from mmdet.models.relation_heads.approaches import Result
import torch


@DATASETS.register_module
class VrdDataset(CocoDataset):

    def __init__(self,
                 ann_file,
                 dict_file,
                 image_file,
                 pipeline,
                 split,
                 num_im=-1,
                 data_root=None,
                 img_prefix='',
                 seg_prefix='',
                 test_mode=False,
                 filter_empty_rels=True,
                 filter_duplicate_rels=True,
                 filter_non_overlap=True):
        """
        Torch dataset for VRD
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        assert split in ('train', 'val', 'test')
        self.split = split
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.data_root = data_root
        self.ann_file = ann_file
        self.dict_file = dict_file
        self.image_file = image_file
        self.test_mode = test_mode
        self.filter_non_overlap = filter_non_overlap
        self.filter_duplicate_rels = filter_duplicate_rels

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.dict_file):
                self.dict_file = osp.join(self.data_root, self.dict_file)
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not osp.isabs(self.image_file):
                self.image_file = osp.join(self.data_root, self.image_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
        self.proposal_file = None
        self.proposals = None

        # contiguous 101, 71 containing __background__
        self.ind_to_classes, self.ind_to_predicates = load_info(dict_file)
        # drop background
        VrdDataset.CLASSES, VrdDataset.PREDICATES = self.ind_to_classes[1:], self.ind_to_predicates[1:]

        self.img_ids, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
            self.ann_file, num_im, filter_empty_rels=filter_empty_rels, filter_non_overlap=self.filter_non_overlap)

        img_infos = load_image_infos(self.img_prefix, self.image_file)  # all info

        self.img_infos = [img_infos[i] for i in self.img_ids]  # the img_ids are consistent with the index in list (our generated version)

        # add other infos to be used in result2json: transform the 0-base results from models to 1-base (used in GT)
        self.cat_ids = list(range(1, len(self.CLASSES) + 1))
        self.coco = self.cocoapi()

        self.pipeline = Compose(pipeline)

        if not self.test_mode:
            self._set_group_flag()

        # When printing the evaluation result, the freq maybe useful.
        # must be ensure that at least run training process once, or ensure the file exists.
        predicate_freq_file = osp.join('/'.join(self.ann_file.split('/')[:-2]), 'predicate_freq.pickle')
        if self.split == 'train':
            if osp.isfile(predicate_freq_file):
                self.predicate_freq = mmcv.load(predicate_freq_file)
            else:
                # borrow the apis from visualgenome
                self.predicate_freq = get_vrd_freq(self)
                mmcv.dump(self.predicate_freq, predicate_freq_file)
        else:
            self.predicate_freq = mmcv.load(predicate_freq_file)

    def cocoapi(self):
        """For using COCO apis.
        """
        auxcoco = COCO()
        anns = []
        for i, (cls_array, box_array) in enumerate(zip(self.gt_classes, self.gt_boxes)):
            for cls, box in zip(cls_array.tolist(), box_array):
                anns.append({'area': float((box[2] - box[0] + 1) * (box[3] - box[1] + 1)),
                             'bbox': self.xyxy2xywh(box),
                             'category_id': cls,
                             'id': len(anns),
                             'image_id': self.img_ids[i],
                             'iscrowd': 0,
                             })
        auxcoco.dataset = {'images': self.img_infos,
                           'categories': [{'id': i + 1, 'name': name} for i, name in enumerate(self.CLASSES)],
                           'annotations': anns}
        auxcoco.createIndex()
        return auxcoco

    def get_ann_info(self, idx):
        """Parse bbox and mask annotation.

        Args:
            idx: dataset index
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = self.gt_boxes[idx]
        gt_labels = self.gt_classes[idx]
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        gt_rels = self.relationships[idx].copy()
        if self.filter_duplicate_rels:
            # Filter out dupes!
            if self.split == 'train':
                all_rel_sets = defaultdict(list)
                for (o0, o1, r) in gt_rels:
                    all_rel_sets[(o0, o1)].append(r)
                gt_rels = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
                gt_rels = np.array(gt_rels, dtype=np.int32)
            else:
                # for test or val set, filter the duplicate triplets, but allow multiple labels for each pair
                all_rel_sets = []
                for (o0, o1, r) in gt_rels:
                    if (o0, o1, r) not in all_rel_sets:
                        all_rel_sets.append((o0, o1, r))
                gt_rels = np.array(all_rel_sets, dtype=np.int32)

        # add relation to target
        num_box = len(gt_bboxes)
        relation_map = np.zeros((num_box, num_box), dtype=np.int64)
        for i in range(gt_rels.shape[0]):
            if relation_map[int(gt_rels[i, 0]), int(gt_rels[i, 1])] > 0:
                if random.random() > 0.5:
                    relation_map[int(gt_rels[i, 0]), int(gt_rels[i, 1])] = int(gt_rels[i, 2])
            else:
                relation_map[int(gt_rels[i, 0]), int(gt_rels[i, 1])] = int(gt_rels[i, 2])

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            rels=gt_rels,
            rel_maps=relation_map,
            attrs=None,
            bboxes_ignore=gt_bboxes_ignore,
            masks=None,
            seg_map=None)
        return ann

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['attr_fields'] = []
        results['rel_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        # NOTE: For SGG, since the forward process may need gt_bboxes/gt_labels,
        # we should also load annotation as if in the training mode.
        anno_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=anno_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_statistics(self):
        fg_matrix, bg_matrix = get_vrd_statistics(self, self.filter_non_overlap)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': None,
        }
        return result

    def get_cooccur_statistics(self):
        mat = np.zeros((
            len(self.ind_to_classes),
            len(self.ind_to_classes),
        ), dtype=np.float32)
        sum_obj_pict = np.zeros(len(self.ind_to_classes), dtype=np.float32)

        for ex_ind in range(len(self)):
            gt_classes = self.gt_classes[ex_ind].copy()
            gt_classes_list = list(set(gt_classes))
            for i in gt_classes_list:
                sum_obj_pict[i] += 1
            inds = np.transpose(np.nonzero(1 - np.eye(len(gt_classes_list), dtype=np.int64)))
            for (i, j) in inds:
                mat[gt_classes[i], gt_classes[j]] += 1
            for key, value in dict(Counter(gt_classes)).items():
                if value >= 2:
                    mat[key, key] += 1

        sum_obj_pict += 1e-3  # because idx 0 means background, and the value is zero, divide zero will occurr an error, so add 1.
        obj_cooccurrence_matrix = mat / np.expand_dims(sum_obj_pict, axis=1)
        return obj_cooccurrence_matrix

    def get_object_statistics(self):
        object_logits, object_prob = get_vrd_obj_freq(self)
        return {'logit': object_logits, 'prob': object_prob}

    def get_gt_info(self):
        """
        We open the api for touching the groundtruth annotation and image info.
        """
        prog_bar = mmcv.ProgressBar(len(self))
        gt_results = []
        for i in range(len(self)):
            ann = self.get_ann_info(i)
            img_info = self.img_infos[i]
            gt_results.append(dict(img_info=img_info, ann_info=Result(bboxes=ann['bboxes'],
                                                                      labels=ann['labels'],
                                                                      rels=ann['rels'],
                                                                      relmaps=ann['rel_maps'],
                                                                      rel_pair_idxes=ann['rels'][:, :2],
                                                                      rel_labels=ann['rels'][:, -1],
                                                                      attrs=ann['attrs'])))
            prog_bar.update()
        return gt_results

    def get_gt(self):
        """
        api for touching the groundtruth annotation
        :return:
        """
        prog_bar = mmcv.ProgressBar(len(self))
        gt_results = []
        for i in range(len(self)):
            ann = self.get_ann_info(i)
            gt_results.append(Result(bboxes=ann['bboxes'],
                                     labels=ann['labels'],
                                     rels=ann['rels'],
                                     relmaps=ann['rel_maps'],
                                     rel_pair_idxes=ann['rels'][:, :2],
                                     rel_labels=ann['rels'][:, -1],
                                     attrs=ann['attrs']))
            prog_bar.update()
        return gt_results

    def evaluate(self,
                 results,
                 metric='predcls',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 multiple_preds=False,
                 iou_thrs=0.5,
                 nogc_thres_num=None,
                 **kwargs):
        """
        **kwargs: contain the paramteters specifically for OD, e.g., proposal_nums.
        Overwritten evaluate API:
            For each metric in metrics, it checks whether to invoke od or sg evaluation.
            if the metric is not 'sg', the evaluate method of super class is invoked
            to perform Object Detection evaluation.
            else, perform scene graph evaluation.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_sg_metrics = ['predcls', 'sgcls', 'sgdet']
        allowed_od_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        sg_metrics, od_metrics = [], []
        for m in metrics:
            if m in allowed_od_metrics:
                od_metrics.append(m)
            elif m in allowed_sg_metrics:
                sg_metrics.append(m)
            else:
                raise ValueError("Unknown metric {}.".format(m))

        if len(od_metrics) > 0:
            # invoke object detection evaluation.
            # Temporarily for bbox
            od_results = [(r.formatted_bboxes, r.formatted_masks) for r in results]
            return super(VrdDataset, self).evaluate(od_results,
                                                    metric,
                                                    logger,
                                                    jsonfile_prefix,
                                                    classwise=classwise,
                                                    iou_thrs=iou_thrs,
                                                    **kwargs)
        if len(sg_metrics) > 0:
            """ 
                Invoke scenen graph evaluation. prepare the groundtruth and predictions.
                Transform the predictions of key-wise to image-wise.
                Both the value in gt_results and det_results are numpy array.
            """
            if not hasattr(self, 'test_gt_results'):
                print('\nLooading testing groundtruth...\n')
                prog_bar = mmcv.ProgressBar(len(self))
                gt_results = []
                for i in range(len(self)):
                    ann = self.get_ann_info(i)
                    gt_results.append(Result(bboxes=ann['bboxes'],
                                             labels=ann['labels'],
                                             rels=ann['rels'],
                                             relmaps=ann['rel_maps'],
                                             rel_pair_idxes=ann['rels'][:, :2],
                                             rel_labels=ann['rels'][:, -1],
                                             attrs=ann['attrs']))
                    prog_bar.update()
                print('\n')
                self.test_gt_results = gt_results

            return vg_evaluation(sg_metrics,
                                 groundtruths=self.test_gt_results,
                                 predictions=results,
                                 iou_thrs=iou_thrs,
                                 logger=logger,
                                 ind_to_predicates=self.ind_to_predicates,
                                 multiple_preds=multiple_preds,
                                 predicate_freq=self.predicate_freq,
                                 nogc_thres_num=nogc_thres_num)


def get_vrd_statistics(vrd, must_overlap=True):
    train_data = vrd
    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)
    progbar = mmcv.ProgressBar(len(train_data))
    for ex_ind in range(len(train_data)):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_boxes = train_data.gt_boxes[ex_ind].copy()

        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
            fg_matrix[o1, o2, gtr] += 1
        # For the background, get all of the things that overlap.
        o1o2_total = gt_classes[np.array(
            box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1
        progbar.update()
    return fg_matrix, bg_matrix


def get_vrd_freq(vrd):
    predicate_freq = np.zeros(len(vrd.PREDICATES))
    for ex_ind in range(len(vrd)):
        gt_predicates = vrd.relationships[ex_ind].copy()[:, 2]
        for p in gt_predicates:
            predicate_freq[p - 1] += 1
    sorted_ids = np.argsort(predicate_freq)[::-1]
    return sorted_ids


def get_vrd_obj_freq(vrd):
    object_freq = np.zeros(len(vrd.CLASSES))
    for ex_ind in range(len(vrd)):
        gt_classes = vrd.gt_classes[ex_ind].copy()
        for c in gt_classes:
            object_freq[c - 1] += 1
    object_prob = object_freq / np.sum(object_freq)
    return object_freq, object_prob


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations.
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(torch.FloatTensor(boxes), torch.FloatTensor(boxes)).numpy() > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes


def load_info(dict_file):
    """
    Loads the file containing the vrd label meanings
    The label file have included the '__background__'
    """
    info = json.load(open(dict_file, 'r'))

    ind_to_classes = info['objects']
    ind_to_predicates = info['predicates']

    return ind_to_classes, ind_to_predicates


def load_image_infos(img_dir, image_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return:
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    img_infos = {}
    for i, img in enumerate(im_data):
        # align with coco format
        img['id'] = img['image_id']
        img['file_name'] = img['filename']
        img_infos[img['image_id']] = img
    return img_infos


def load_graphs(ann_file, num_im, filter_empty_rels, filter_non_overlap):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return:
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    anns = json.load(open(ann_file))

    # Get everything by image.
    img_ids = []
    boxes = []
    gt_classes = []
    relationships = []
    for i, ann in enumerate(anns):
        im_id = ann['image_id']
        boxes_i = np.array(ann['bboxes'], dtype=np.float32)
        gt_classes_i = np.array(ann['gt_classes'], dtype=np.int64)
        rels = np.array(ann['gt_rels'], dtype=np.int64)

        if len(rels) == 0:
            assert filter_empty_rels
            continue  # filter

        if filter_non_overlap:
            inters = bbox_overlaps(torch.FloatTensor(boxes_i), torch.FloatTensor(boxes_i)).numpy()
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]
            if inc.size > 0:
                rels = rels[inc]
            else:
                # filter out
                continue

        img_ids.append(im_id)
        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)

    if num_im > -1:
        img_ids = img_ids[:num_im]
        boxes = boxes[:num_im]
        gt_classes = gt_classes[:num_im]
        relationships = relationships[:num_im]

    return img_ids, boxes, gt_classes, relationships
