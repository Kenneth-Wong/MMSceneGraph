# ---------------------------------------------------------------
# generalized_visualgenome.py
# Set-up time: 2021/1/12 14:19
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import logging
import os.path as osp
import os
import tempfile

import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from coco_caption.pycocoevalcap.eval import COCOEvalCap
import json
import pickle
import h5py
from collections import defaultdict
import random

from mmdet.core import eval_recalls
from mmdet.utils import print_log
from .custom import CustomDataset
from .coco import CocoDataset
from .registry import DATASETS
from .pipelines import Compose
from mmdet.core.bbox.geometry import bbox_overlaps
from mmdet.core import vg_evaluation
from mmdet.core.evaluation.relcaption_eval import relcaption_evaluation
from mmdet.models.relation_heads.approaches import Result
import torch
import pandas as pd
from factories.vgkr_v2.transform_coco import transform

BOX_SCALE = 1024


@DATASETS.register_module
class GeneralizedVisualGenomeDataset(CocoDataset):
    def __init__(self,
                 roidb_file,
                 dict_file,
                 image_file,
                 pipeline,
                 split,
                 split_type='vg150_split',
                 num_im=-1,
                 num_val_im=5000,
                 num_cap_per_img=5,
                 data_root=None,
                 ann_file=None,
                 img_prefix='',
                 seg_prefix='',
                 test_mode=False,
                 scene_file=None,
                 filter_empty_rels=True,
                 filter_duplicate_rels=True,
                 filter_empty_caps=False,
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
        assert split_type in {'vg150_split', 'vgcoco_split', 'filter_coco_det_val_split',
                              'filter_coco_karpathycap_testval_split', 'vgcoco_entity_split'}
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
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels  # and self.split == 'train'
        self.num_cap_per_img = num_cap_per_img

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

        self.ind_to_classes, self.ind_to_tokens, self.ind_to_attributes = load_info(
            dict_file)
        # drop background
        GeneralizedVisualGenomeDataset.CLASSES, GeneralizedVisualGenomeDataset.TOKENS, \
        GeneralizedVisualGenomeDataset.ATTRIBUTES = self.ind_to_classes[1:], self.ind_to_tokens[1:], \
                                                    self.ind_to_attributes[1:]

        self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships, \
        self.rel_inputs, self.rel_targets, self.rel_ipt_scores, self.cap_inputs, self.cap_targets = load_graphs(
            self.roidb_file, self.image_file, self.split, self.split_type, num_im, num_val_im=num_val_im,
            filter_empty_rels=filter_empty_rels,
            filter_non_overlap=self.filter_non_overlap,
            filter_empty_caps=filter_empty_caps,
            num_cap_per_img=self.num_cap_per_img
        )

        self.img_ids, self.img_infos = load_image_infos(self.img_prefix,
                                                        self.image_file)  # length equals to split_mask
        self.img_ids = [self.img_ids[i] for i in np.where(self.split_mask)[0]]

        self.img_infos = [self.img_infos[i] for i in np.where(self.split_mask)[0]]

        self.scenes = None
        if self.scene_file is not None:
            self.scenes = load_scenes(self.scene_file, self.img_infos)

        # Optional: load the visual auxiliary information: saliency, depth, etc.
        if self.additional_files is not None:
            for k, v in self.additional_files.items():
                setattr(self, k.split('_')[0] + '_map', h5py.File(v, 'r')['images'])

        ## ann_file: used for evaluation, if this is the val/test dataset.
        self.ann_file = ann_file
        if self.ann_file is not None:
            coco_like_annots = transform(self.ann_file, self.img_ids)
            tmp_dir = tempfile.TemporaryDirectory()
            result_file = osp.join(tmp_dir.name, 'coco_annots_tmp.json')
            mmcv.dump(coco_like_annots, result_file)
            self.coco = COCO(result_file)
            os.remove(result_file)
        self.pipeline = Compose(pipeline)

        if not self.test_mode:
            self._set_group_flag()

        ######################################################################## Deprecated!!!!
        # When printing the evaluation result, the freq maybe useful.
        # must be ensure that at least run training process once, or ensure the file exists.
        # predicate_freq_file = osp.join('/'.join(self.roidb_file.split('/')[:-1]), 'predicate_freq.pickle')
        # if self.split == 'train':
        #     if osp.isfile(predicate_freq_file):
        #         self.predicate_freq = mmcv.load(predicate_freq_file)
        #     else:
        #         self.predicate_freq = get_VG_freq(self)
        #         mmcv.dump(self.predicate_freq, predicate_freq_file)
        # else:
        #     self.predicate_freq = mmcv.load(predicate_freq_file)

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
        gt_rel_inputs = self.rel_inputs[idx].copy()
        gt_rel_targets = self.rel_targets[idx].copy()
        gt_rel_ipt_scores = self.rel_ipt_scores[idx].copy()
        gt_cap_inputs = self.cap_inputs[idx].copy()
        gt_cap_targets = self.cap_targets[idx].copy()
        if self.filter_duplicate_rels:
            # Filter out dupes! keep the important ones.
            keep_ids = []
            keep_pairs = []
            for rel_id in range(len(gt_rels)):
                if gt_rel_ipt_scores[rel_id] > 0:
                    keep_ids.append(rel_id)
                else:
                    if gt_rels[rel_id].tolist() not in keep_pairs:
                        keep_pairs.append(gt_rels[rel_id].tolist())
                        keep_ids.append(rel_id)
            gt_rels = gt_rels[keep_ids]
            gt_rel_inputs = gt_rel_inputs[keep_ids]
            gt_rel_targets = gt_rel_targets[keep_ids]
            gt_rel_ipt_scores = gt_rel_ipt_scores[keep_ids]

        # add relation to target
        num_box = len(gt_bboxes)
        relation_map = np.zeros((num_box, num_box), dtype=np.int64)
        for i in range(gt_rels.shape[0]):
            relation_map[int(gt_rels[i, 0]), int(gt_rels[i, 1])] = 1

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            rels=gt_rels,
            rel_maps=relation_map,
            rel_inputs=gt_rel_inputs,
            rel_targets=gt_rel_targets,
            rel_ipt_scores=gt_rel_ipt_scores,
            attrs=gt_attrs,
            cap_inputs=gt_cap_inputs,
            cap_targets=gt_cap_targets,
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

    def _eval_img_caption(self, results, logger=None, epoch=None):
        tmp_dir = tempfile.TemporaryDirectory()
        result_file = osp.join(tmp_dir.name, 'cap_results.json')
        mmcv.dump(results, result_file)
        cocoRes = self.coco.loadRes(result_file)
        cocoEval = COCOEvalCap(self.coco, cocoRes)
        cocoEval.evaluate()
        os.remove(result_file)

        eval_results = {}
        eval_results[self.split + '_set/copystate'] = ''
        for metric, value in cocoEval.eval.items():
            eval_results[metric] = value
            eval_results[self.split + '_set/copystate'] += (metric + ': ' + str(value) + '\t')

        return eval_results

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 min_overlaps=[0.2, 0.3, 0.4, 0.5, 0.6],
                 min_scores=[-1, 0, 0.05, 0.1, 0.15, 0.2, 0.25],
                 epoch=None,
                 work_dir=None,
                 **kwargs):
        """
        **kwargs: contain the paramteters specifically for OD, e.g., proposal_nums.
        Overwritten evaluate API:
            For each metric in metrics, it checks whether to invoke od or sg evaluation.
            if the metric is not 'sg', the evaluate method of super class is invoked
            to perform Object Detection evaluation.
            else, perform scene graph evaluation.
        """
        # eval the image-wise caption
        # 1. Transform the predicted results form
        caption_results, caption_from_triplet_results = [], []
        eval_res = {}
        if results[0].cap_sents is not None:
            for img_id, r in zip(self.img_ids, results):
                caption_results.append(dict(image_id=img_id, caption=r.cap_sents))
            # save or not
            if work_dir is not None:
                res_dir = osp.join(work_dir, 'eval_results')
                mmcv.mkdir_or_exist(res_dir)
                if epoch is not None:
                    filename = osp.join(res_dir, self.split + '_' + str(epoch + 1) + '_captions.json')
                else:
                    filename = osp.join(res_dir, self.split + '_captions.json')
                mmcv.dump(caption_results, filename)
            eval_res.update(self._eval_img_caption(caption_results))

        # eval the relational captions
        if results[0].rel_cap_sents is not None:
            # TODO: use linguistic evalution protocols
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
                                             rel_cap_seqs=ann['rel_targets'],
                                             rel_ipts=ann['rel_ipt_scores'],
                                             attrs=ann['attrs']))
                    prog_bar.update()
                print('\n')
                self.test_gt_results = gt_results
            relcap_res = relcaption_evaluation(
                groundtruths=self.test_gt_results,
                predictions=results,
                min_overlaps=[0.2, 0.3, 0.4, 0.5, 0.6],
                min_scores=[-1, 0, 0.05, 0.1, 0.15, 0.2, 0.25],
                logger=logger,
                vocab=self.ind_to_tokens)
            eval_res.update(relcap_res)

        #msg = str(eval_res)
        #print_log(msg, logger=logger)
        # here: print for training and eval (eval do not have logger hook)
        for k, v in eval_res.items():
            print(k, v)
        return eval_res


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
        info['attribute_to_idx']['__background__'] = 0
        info['token_to_idx']['.'] = 0

    class_to_ind = info['label_to_idx']

    token_to_ind = info['token_to_idx']

    ind_to_classes = list(sorted(class_to_ind, key=lambda k: class_to_ind[k]))
    ind_to_tokens = list(sorted(token_to_ind, key=lambda k: token_to_ind[k]))
    if 'attribute_to_idx' in info:
        attribute_to_ind = info['attribute_to_idx']
        ind_to_attributes = list(sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k]))
    else:
        ind_to_attributes = None

    return ind_to_classes, ind_to_tokens, ind_to_attributes


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
    meta_infos = pd.read_csv(image_file, low_memory=False)

    img_ids = list(meta_infos['meta_vgids'])
    meta_paths = list(meta_infos['meta_paths'])
    meta_heights = list(meta_infos['meta_heights'])
    meta_widths = list(meta_infos['meta_widths'])
    meta_cocoids = list(meta_infos['meta_cocoids'])
    meta_flickr_ids = list(meta_infos['meta_flickr_ids'])

    img_infos = []
    for img_id, path, height, width, cocoid, flickrid in zip(img_ids, meta_paths, meta_heights, meta_widths,
                                                             meta_cocoids, meta_flickr_ids):
        assert osp.isfile(osp.join(img_dir, path))
        img = dict(id=int(img_id), img_id=int(img_id), filename=path, file_name=path,
                   height=height, width=width, coco_id=None if cocoid == 'None' else int(cocoid),
                   flickr_id=None if flickrid == 'None' else int(flickrid))

        img_infos.append(img)
    assert len(img_ids) == 108073
    assert len(img_infos) == 108073
    return img_ids, img_infos


def load_graphs(roidb_file, image_file, split, split_type, num_im, num_val_im, filter_empty_rels, filter_non_overlap,
                filter_empty_caps, num_cap_per_img=5):
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
    meta_infos = pd.read_csv(image_file, low_memory=False)
    data_split = np.array(list(meta_infos[split_type]))

    split_flag = 2 if split == 'test' else 0
    split_mask = data_split == split_flag

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0
    if filter_empty_caps:
        split_mask &= roi_h5['img_to_first_cap'][:] >= 0

    # debug: '2408674.jpg',
    #split_mask[13341] = False

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
    all_boxes = roi_h5['bboxes'][:]  # x1, y1, x2, y2
    all_boxes = np.array(all_boxes, dtype=np.float32)  # must be float32
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box
    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]

    # load relation seqs
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]
    _relations = roi_h5['relationships'][:]
    rel_inputs = np.array(roi_h5['rel_inputs'][:], dtype='int')
    rel_targets = np.array(roi_h5['rel_targets'][:], dtype='int')
    rel_ipt_scores = roi_h5['rel_ipt_scores'][:, 0]

    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])

    # load caption seqs
    im_to_first_cap = roi_h5['img_to_first_cap'][split_mask]
    im_to_last_cap = roi_h5['img_to_last_cap'][split_mask]
    cap_inputs = roi_h5['cap_inputs'][:]
    cap_targets = roi_h5['cap_targets'][:]

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = [] if all_attributes is not None else None
    gt_rels = []
    gt_rel_inputs, gt_rel_targets, gt_rel_ipt_scores = [], [], []
    gt_cap_inputs, gt_cap_targets = [], []

    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]
        i_cap_start = im_to_first_cap[i]
        i_cap_end = im_to_last_cap[i]

        boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start: i_obj_end + 1]
        if all_attributes is not None:
            gt_attributes_i = all_attributes[i_obj_start: i_obj_end + 1, :]

        if i_cap_start >= 0:
            cap_inputs_i = np.zeros((num_cap_per_img, cap_inputs.shape[1]), dtype='int')
            cap_targets_i = np.zeros((num_cap_per_img, cap_inputs.shape[1]), dtype='int')

            all_cap_inputs_i = cap_inputs[i_cap_start: i_cap_end + 1]
            all_cap_targets_i = cap_targets[i_cap_start: i_cap_end + 1]

            n = len(all_cap_inputs_i)
            if n >= num_cap_per_img:
                sid = 0
                ixs = random.sample(range(n), num_cap_per_img)
            else:
                sid = n
                ixs = np.random.choice(n, num_cap_per_img - n)

                cap_inputs_i[0:n, :] = all_cap_inputs_i
                cap_targets_i[0:n, :] = all_cap_targets_i

            for i, ix in enumerate(ixs):
                cap_inputs_i[sid + i] = all_cap_inputs_i[ix, :]
                cap_targets_i[sid + i] = all_cap_targets_i[ix, :]

        else:
            assert not filter_empty_caps
            cap_inputs_i = np.zeros((0, cap_inputs.shape[1]), dtype='int')
            cap_targets_i = np.zeros((0, cap_inputs.shape[1]), dtype='int')

        if i_rel_start >= 0:
            obj_idx = _relations[i_rel_start: i_rel_end + 1] - i_obj_start  # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = obj_idx  # (num_rel, 2), representing sub, obj

            rel_inputs_i = rel_inputs[i_rel_start: i_rel_end + 1]
            rel_targets_i = rel_targets[i_rel_start: i_rel_end + 1]
            rel_ipt_scores_i = rel_ipt_scores[i_rel_start: i_rel_end + 1]

        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 2), dtype=np.int32)
            rel_inputs_i = np.zeros((0, rel_inputs.shape[1]), dtype='int')
            rel_targets_i = np.zeros((0, rel_inputs.shape[1]), dtype='int')
            rel_ipt_scores_i = np.zeros((0,), dtype='int')

        if filter_non_overlap:
            assert split == 'train'
            inters = bbox_overlaps(torch.FloatTensor(boxes_i), torch.FloatTensor(boxes_i)).numpy()
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]
            if inc.size > 0:
                rels = rels[inc]
                rel_inputs_i = rel_inputs_i[inc]
                rel_targets_i = rel_targets_i[inc]
                rel_ipt_scores_i = rel_ipt_scores_i[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        if all_attributes is not None:
            gt_attributes.append(gt_attributes_i)
        gt_rels.append(rels)

        gt_rel_inputs.append(rel_inputs_i)
        gt_rel_targets.append(rel_targets_i)

        gt_cap_inputs.append(cap_inputs_i)
        gt_cap_targets.append(cap_targets_i)

        #: ipt scores: make the scores > 0 to 1
        rel_ipt_scores_i[np.where(rel_ipt_scores_i > 0)[0]] = 1
        gt_rel_ipt_scores.append(rel_ipt_scores_i)

    return split_mask, boxes, gt_classes, gt_attributes, gt_rels, gt_rel_inputs, gt_rel_targets, gt_rel_ipt_scores, \
           gt_cap_inputs, gt_cap_targets


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
