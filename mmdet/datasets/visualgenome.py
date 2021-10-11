# ---------------------------------------------------------------
# visualgenome.py
# Set-up time: 2020/3/24 下午4:13
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
import pickle
import h5py
from collections import defaultdict, Counter
import random

from mmdet.core import eval_recalls
from mmdet.utils import print_log
from .custom import CustomDataset
from .coco import CocoDataset
from .registry import DATASETS
from .pipelines import Compose
from mmdet.core.bbox.geometry import bbox_overlaps
from mmdet.core import vg_evaluation
from mmdet.models.relation_heads.approaches import Result
import torch

BOX_SCALE = 1024


@DATASETS.register_module
class VisualGenomeDataset(CocoDataset):

    def __init__(self,
                 roidb_file,
                 dict_file,
                 image_file,
                 pipeline,
                 split,
                 split_type='normal',
                 num_im=-1,
                 num_val_im=5000,
                 data_root=None,
                 img_prefix='',
                 seg_prefix='',
                 test_mode=False,
                 scene_file=None,
                 filter_empty_rels=True,
                 filter_duplicate_rels=True,
                 filter_non_overlap=True,
                 additional_files=None):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            pipeline: the image processing pipeline
            scene_file: You can choose to load the scene labels, but only part of the images possessed it.
            split_type: optional: [normal, filter_cocoval]
                filter_cocoval: filter the val images from the training split
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
            additional_files: [dict]: {'saliency_file': path to sal, 'depth_file': path to depth, etc.}
        """

        assert split in {'train', 'val', 'test'}
        assert split_type in {'normal', 'filter_cocoval'}
        self.split = split
        self.split_type = split_type
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.data_root = data_root
        self.dict_file = dict_file
        self.roidb_file = roidb_file
        self.image_file = image_file
        self.scene_file = scene_file
        self.additional_files = additional_files
        self.test_mode = test_mode
        self.filter_non_overlap = filter_non_overlap #and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels #and self.split == 'train'

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.dict_file):
                self.dict_file = osp.join(self.data_root, self.dict_file)
            if not osp.isabs(self.roidb_file):
                self.roidb_file = osp.join(self.data_root, self.roidb_file)
            if not osp.isabs(self.image_file):
                self.image_file = osp.join(self.data_root, self.image_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.scene_file is None or osp.isabs(self.scene_file)):
                self.scene_file = osp.join(self.data_root, self.scene_file)
            if not self.additional_files is None:
                for k, v in self.additional_files:
                    if not osp.isabs(v):
                        self.additional_files[k] = osp.join(self.data_root, v)
        self.proposal_file = None
        self.proposals = None

        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info(
            dict_file)  # contiguous 151, 51 containing __background__
        # drop background
        VisualGenomeDataset.CLASSES, VisualGenomeDataset.PREDICATES, \
        VisualGenomeDataset.ATTRIBUTES = self.ind_to_classes[1:], self.ind_to_predicates[1:], self.ind_to_attributes

        self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships = load_graphs(
            self.roidb_file, self.split, self.split_type, num_im, num_val_im=num_val_im,
            filter_empty_rels=filter_empty_rels,
            filter_non_overlap=self.filter_non_overlap,
        )

        self.img_ids, self.img_infos = load_image_infos(self.img_prefix,
                                                        self.image_file)  # length equals to split_mask
        self.img_ids = [self.img_ids[i] for i in np.where(self.split_mask)[0]]

        self.img_infos = [self.img_infos[i] for i in np.where(self.split_mask)[0]]

        self.scenes = None
        if self.scene_file is not None:
            self.scenes = load_scenes(self.scene_file, self.img_infos)

        # transform the gt_boxes to its original scale
        for i in range(len(self.img_infos)):
            width, height = self.img_infos[i]['width'], self.img_infos[i]['height']
            self.gt_boxes[i] = (self.gt_boxes[i] / BOX_SCALE * max(width, height)).astype(np.float32)

        # Optional: load the visual auxiliary information: saliency, depth, etc.
        if self.additional_files is not None:
            for k, v in self.additional_files.items():
                setattr(self, k.split('_')[0] + '_map', h5py.File(v, 'r')['images'])

        # add other infos to be used in result2json: transform the 0-base results from models to 1-base (used in GT)
        self.cat_ids = list(range(1, len(self.CLASSES) + 1))
        self.coco = self.cocoapi()

        self.pipeline = Compose(pipeline)

        if not self.test_mode:
            self._set_group_flag()

        # When printing the evaluation result, the freq maybe useful.
        # must be ensure that at least run training process once, or ensure the file exists.
        predicate_freq_file = osp.join('/'.join(self.roidb_file.split('/')[:-1]), 'predicate_freq.pickle')
        if self.split == 'train':
            if osp.isfile(predicate_freq_file):
                self.predicate_freq = mmcv.load(predicate_freq_file)
            else:
                self.predicate_freq = get_VG_freq(self)
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
        gt_attrs = self.gt_attributes[idx]
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        gt_scenes = self.scenes[idx] if self.scenes is not None else None

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
            attrs=gt_attrs,
            # the following keys are not exists in VG, but we still provide it in accord with coco.
            bboxes_ignore=gt_bboxes_ignore,
            masks=None,
            seg_map=None,
            scenes=gt_scenes)
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
        fg_matrix, bg_matrix = get_VG_statistics(self, self.filter_non_overlap)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': self.ind_to_attributes,
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
        object_logits, object_prob = get_VG_obj_freq(self)
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
            if not isinstance(results[0], Result):
                # it may be the reuslts from the son classes
                od_results = results
            else:
                od_results = [(r.formatted_bboxes, r.formatted_masks) for r in results]
            return super(VisualGenomeDataset, self).evaluate(od_results,
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


def get_VG_statistics(vg, must_overlap=True):
    train_data = vg
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


def get_VG_freq(vg):
    predicate_freq = np.zeros(len(vg.PREDICATES))
    for ex_ind in range(len(vg)):
        gt_predicates = vg.relationships[ex_ind].copy()[:, 2]
        for p in gt_predicates:
            predicate_freq[p - 1] += 1
    sorted_ids = np.argsort(predicate_freq)[::-1]
    return sorted_ids


def get_VG_obj_freq(vg):
    object_freq = np.zeros(len(vg.CLASSES))
    for ex_ind in range(len(vg)):
        gt_classes = vg.gt_classes[ex_ind].copy()
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


def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0
        if 'attribute_to_idx' in info:
            info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']

    ind_to_classes = list(sorted(class_to_ind, key=lambda k: class_to_ind[k]))
    ind_to_predicates = list(sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k]))
    if 'attribute_to_idx' in info:
        attribute_to_ind = info['attribute_to_idx']
        ind_to_attributes = list(sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k]))
    else:
        ind_to_attributes = None

    return ind_to_classes, ind_to_predicates, ind_to_attributes


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

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    img_ids = []
    img_infos = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        sub_dir = ['VG_100K', 'VG_100K_2']
        filename1 = osp.join(img_dir, sub_dir[0], basename)
        filename2 = osp.join(img_dir, sub_dir[1], basename)
        if osp.exists(filename1):
            img_ids.append(img['image_id'])
            # align with coco format
            img['id'] = img['image_id']
            img['filename'] = osp.join(sub_dir[0], basename)
            img['file_name'] = img['filename']
            img_infos.append(img)
        elif osp.exists(filename2):
            img_ids.append(img['image_id'])
            # align with coco format
            img['id'] = img['image_id']
            img['filename'] = osp.join(sub_dir[1], basename)
            img['file_name'] = img['filename']
            img_infos.append(img)

    assert len(img_ids) == 108073
    assert len(img_infos) == 108073
    return img_ids, img_infos


def load_graphs(roidb_file, split, split_type, num_im, num_val_im, filter_empty_rels, filter_non_overlap):
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
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    roi_h5 = h5py.File(roidb_file, 'r')
    if split_type == 'normal':
        data_split = roi_h5['split'][:]
    elif split_type == 'filter_cocoval':
        data_split = roi_h5['filter_cocoval_from_train_split'][:]
    else:
        raise NotImplementedError
    split_flag = 2 if split == 'test' else 0
    split_mask = data_split == split_flag

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if split == 'val':
            image_index = image_index[:num_val_im]
        elif split == 'train':
            image_index = image_index[num_val_im:]

    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_attributes = roi_h5['attributes'][:, :] if 'attributes' in roi_h5 else None
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = [] if all_attributes is not None else None
    relationships = []
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start: i_obj_end + 1]
        if all_attributes is not None:
            gt_attributes_i = all_attributes[i_obj_start: i_obj_end + 1, :]

        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start: i_rel_end + 1]
            obj_idx = _relations[i_rel_start: i_rel_end + 1] - i_obj_start  # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))  # (num_rel, 3), representing sub, obj, and pred
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            #assert split == 'train'
            inters = bbox_overlaps(torch.FloatTensor(boxes_i), torch.FloatTensor(boxes_i)).numpy()
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]
            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        if all_attributes is not None:
            gt_attributes.append(gt_attributes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, gt_attributes, relationships


def load_scenes(scene_file, img_infos):
    scene_ann = mmcv.load(scene_file)
    scene_anns = {}
    for ann in scene_ann:
        kid = int(ann.split('_')[-1].split('.')[0].lstrip('0'))
        scene_anns[kid] = scene_ann[ann]
        scene_anns[kid]['file_name'] = ann
    scene_labels = []
    for img_info in img_infos:
        if img_info['coco_id'] is not None and img_info['coco_id'] in scene_anns:
            scene_labels.append(scene_anns[img_info['coco_id']]['scene_class'])
        else:
            scene_labels.append(-1)
    return scene_labels