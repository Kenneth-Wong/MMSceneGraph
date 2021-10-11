# ---------------------------------------------------------------
# sgg_eval.py
# Set-up time: 2020/5/18 上午9:49
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import itertools
from terminaltables import AsciiTable
import numpy as np
from functools import reduce
from mmdet.core import bbox_overlaps
from .sgg_eval_util import intersect_2d, argsort_desc

from abc import ABC, abstractmethod


class SceneGraphEvaluation(ABC):
    def __init__(self, result_dict, nogc_result_dict, nogc_thres_num):
        super().__init__()
        self.result_dict = result_dict
        self.nogc_result_dict = nogc_result_dict
        self.nogc_thres_num = nogc_thres_num

    @abstractmethod
    def register_container(self, mode):
        print("Register Result Container")
        pass

    @abstractmethod
    def generate_print_string(self, mode):
        print("Generate Print String")
        pass


"""
Traditional Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""


class SGRecall(SceneGraphEvaluation):
    def __init__(self, *args, **kwargs):
        super(SGRecall, self).__init__(*args, **kwargs)

    def register_container(self, mode):
        self.result_dict[mode + '_recall'] = {protocol: {1: [], 5: [], 10: [], 20: []} for protocol in ['triplet', 'tuple']}
        self.nogc_result_dict[mode + '_recall'] = {ngc: {1: [], 5: [], 10: [], 20: []} for ngc in self.nogc_thres_num}

    def _calculate_single(self, target_dict, prediction_to_gt, gt_rels, mode, protocol='triplet', nogc_num=None):
        target = target_dict[mode + '_recall'][protocol] if nogc_num is None else target_dict[mode + '_recall'][nogc_num]
        for k in target:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, prediction_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            target[k].append(rec_i)

    def _print_single(self, target_dict, mode, protocol='triplet', nogc_num=None):
        target = target_dict[mode + '_recall'][protocol] if nogc_num is None else target_dict[mode + '_recall'][nogc_num]
        result_str = 'SGG eval: '
        for k, v in target.items():
            result_str += ' R @ %d: %.4f; ' % (k, np.mean(v))
        suffix_type = '%s Recall.' % protocol if nogc_num is None else 'NoGraphConstraint @ %d Recall.'% nogc_num
        result_str += ' for mode=%s, type=%s' % (mode, suffix_type)
        result_str += '\n'
        return result_str

    def generate_print_string(self, mode):
        result_str = self._print_single(self.result_dict, mode, 'triplet')
        result_str += self._print_single(self.result_dict, mode, 'tuple')
        # nogc
        for nogc_num in self.nogc_thres_num:
            result_str += self._print_single(self.nogc_result_dict, mode, nogc_num=nogc_num)

        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        ranking_scores = local_container['ranking_scores']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        iou_thrs = global_container['iou_thrs']

        nogc_thres_num = self.nogc_thres_num  # list: (typlically) VGKR: [80];

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes

        # compute the graph constraint setting pred_rels: these triplets has been ranked (also considered the ranking scores)
        pred_rels = np.column_stack((pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))
        pred_scores = rel_scores[:, 1:].max(1)

        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
            pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        # Compute recall. It's most efficient to match once and then do recall after
        # if mode is sgdet, report both sgdet and phrdet
        pred_to_gt = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thrs,
            phrdet=False)
        local_container['pred_to_gt'] = pred_to_gt
        self._calculate_single(self.result_dict, pred_to_gt, gt_rels, mode, protocol='triplet')

        pair_to_gt = _compute_pair_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thrs,
            phrdet=False)
        local_container['pair_to_gt'] = pair_to_gt
        self._calculate_single(self.result_dict, pair_to_gt, gt_rels, mode, protocol='tuple')

        # compute the no graph constraint setting pred_rels
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        if ranking_scores is not None and len(ranking_scores) > 0:
            obj_scores_per_rel *= ranking_scores
        nogc_overall_scores = obj_scores_per_rel[:, None] * rel_scores[:, 1:]  # Nr * 50
        sorted_inds = np.argsort(nogc_overall_scores, axis=-1)[:, ::-1]
        sorted_nogc_overall_scores = np.sort(nogc_overall_scores, axis=-1)[:, ::-1]
        gt_pair_idx = gt_rels[:, 0] * 10000 + gt_rels[:, 1]
        for nogc_num in nogc_thres_num:
            nogc_score_inds_ = argsort_desc(sorted_nogc_overall_scores[:, :nogc_num])
            nogc_pred_rels = np.column_stack((pred_rel_inds[nogc_score_inds_[:, 0]],
                                              sorted_inds[nogc_score_inds_[:, 0], nogc_score_inds_[:, 1]] + 1))
            nogc_pred_scores = rel_scores[nogc_score_inds_[:, 0],
                                          sorted_inds[nogc_score_inds_[:, 0], nogc_score_inds_[:, 1]] + 1]

            pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
                nogc_pred_rels, pred_classes, pred_boxes, nogc_pred_scores, obj_scores)

            # prepare the gt rel signal to be used in PairAccuracy:
            pred_pair_idx = nogc_pred_rels[:, 0] * 10000 + nogc_pred_rels[:, 1]
            local_container['nogc@%d_pred_pair_in_gt' % nogc_num] = \
                (pred_pair_idx[:, None] == gt_pair_idx[None, :]).sum(-1) > 0

            # Compute recall. It's most efficient to match once and then do recall after
            pred_to_gt = _compute_pred_matches(
                gt_triplets,
                pred_triplets,
                gt_triplet_boxes,
                pred_triplet_boxes,
                iou_thrs,
                phrdet=False,
            )

            # NOTE: For NGC recall, zs recall, mean recall, only need to crop the top 100 triplets.
            # While for computing the Pair Acccuracy, all of the pairs are needed here.
            local_container['nogc@%d_pred_to_gt' % nogc_num] = pred_to_gt[:100]  # for zR, mR, R
            local_container['nogc@%d_all_pred_to_gt' % nogc_num] = pred_to_gt  # for Pair accuracy
            self._calculate_single(self.nogc_result_dict, pred_to_gt[:100], gt_rels, mode, nogc_num)

        return local_container



class SGMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, nogc_result_dict, nogc_thres_num, num_rel, ind_to_predicates, print_detail=False):
        super(SGMeanRecall, self).__init__(result_dict, nogc_result_dict, nogc_thres_num)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:]  # remove __background__

    def register_container(self, mode):
        # self.result_dict[mode + '_recall_hit'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        # self.result_dict[mode + '_recall_count'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        self.result_dict[mode + '_mean_recall'] = {1: 0.0, 5: 0.0, 10: 0.0, 20: 0.0}
        self.result_dict[mode + '_mean_recall_collect'] = {1: [[] for _ in range(self.num_rel)],
                                                           5: [[] for _ in range(self.num_rel)],
                                                           10: [[] for _ in range(self.num_rel)],
                                                           20: [[] for _ in range(self.num_rel)]}
        self.result_dict[mode + '_mean_recall_list'] = {1: [], 5: [], 10: [], 20: []}

        self.nogc_result_dict[mode + '_mean_recall'] = {ngc: {1: 0.0, 5: 0.0, 10: 0.0, 20: 0.0} for ngc in
                                                        self.nogc_thres_num}
        self.nogc_result_dict[mode + '_mean_recall_collect'] = {ngc: {1: [[] for _ in range(self.num_rel)],
                                                                      5: [[] for _ in range(self.num_rel)],
                                                                      10: [[] for _ in range(self.num_rel)],
                                                                      20: [[] for _ in range(self.num_rel)]}
                                                                for ngc in self.nogc_thres_num}
        self.nogc_result_dict[mode + '_mean_recall_list'] = {ngc: {1: [], 5: [], 10: [], 20: []} for ngc in
                                                             self.nogc_thres_num}


    def _collect_single(self, target_dict, prediction_to_gt, gt_rels, mode, nogc_num=None):
        target_collect = target_dict[mode + '_mean_recall_collect'] if nogc_num is None else \
            target_dict[mode + '_mean_recall_collect'][nogc_num]

        for k in target_collect:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, prediction_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx, 2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]), 2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1

            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    target_collect[k][n].append(float(recall_hit[n] / recall_count[n]))

    def _calculate_single(self, target_dict, mode, nogc_num=None):
        target_collect = target_dict[mode + '_mean_recall_collect'] if nogc_num is None else \
            target_dict[mode + '_mean_recall_collect'][nogc_num]
        target_recall = target_dict[mode + '_mean_recall'] if nogc_num is None else \
            target_dict[mode + '_mean_recall'][nogc_num]
        target_recall_list = target_dict[mode + '_mean_recall_list'] if nogc_num is None else \
            target_dict[mode + '_mean_recall_list'][nogc_num]
        for k, v in target_recall.items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(target_collect[k][idx + 1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(target_collect[k][idx + 1])
                target_recall_list[k].append(tmp_recall)
                sum_recall += tmp_recall
            target_recall[k] = sum_recall / float(num_rel_no_bg)

    def _print_single(self, target_dict, mode, nogc_num=None, predicate_freq=None):
        target = target_dict[mode + '_mean_recall'] if nogc_num is None else \
            target_dict[mode + '_mean_recall'][nogc_num]
        target_recall_list = target_dict[mode + '_mean_recall_list'] if nogc_num is None else \
            target_dict[mode + '_mean_recall_list'][nogc_num]

        result_str = 'SGG eval: '
        for k, v in target.items():
            result_str += ' mR @ %d: %.4f; ' % (k, float(v))
        suffix_type = 'Mean Recall.' if nogc_num is None else 'NoGraphConstraint @ %d Mean Recall.' % (nogc_num)
        result_str += ' for mode=%s, type=%s' % (mode, suffix_type)
        result_str += '\n'

        # result_str is flattened for copying the data to the form, while the table is for vis.
        # Only for graph constraint, one mode for short
        if self.print_detail and mode != 'phrdet' and nogc_num is None:
            rel_name_list, res = self.rel_name_list, target_recall_list[20]
            if predicate_freq is not None:
                rel_name_list = [self.rel_name_list[sid] for sid in predicate_freq]
                res = [target_recall_list[20][sid] for sid in predicate_freq]

            result_per_predicate = []
            for n, r in zip(rel_name_list, res):
                result_per_predicate.append(('{}'.format(str(n)), '{:.4f}'.format(r)))
            result_str += '\t'.join(list(map(str, rel_name_list)))
            result_str += '\n'

            def map_float(num):
                return '{:.4f}'.format(num)

            result_str += '\t'.join(list(map(map_float, res)))
            result_str += '\n'

            num_columns = min(6, len(result_per_predicate) * 2)
            results_flatten = list(
                itertools.chain(*result_per_predicate))
            headers = ['predicate', 'Rec20'] * (num_columns // 2)
            results_2d = itertools.zip_longest(*[
                results_flatten[i::num_columns]
                for i in range(num_columns)])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            result_str += table.table + '\n'

        return result_str

    def generate_print_string(self, mode, predicate_freq=None):
        result_str = self._print_single(self.result_dict, mode, predicate_freq=predicate_freq)

        # nogc
        for nogc_num in self.nogc_thres_num:
            result_str += self._print_single(self.nogc_result_dict, mode, nogc_num, predicate_freq=predicate_freq)

        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']
        self._collect_single(self.result_dict, pred_to_gt, gt_rels, mode)

        for nogc_num in self.nogc_thres_num:
            nogc_pred_to_gt = local_container['nogc@%d_pred_to_gt' % nogc_num]
            self._collect_single(self.nogc_result_dict, nogc_pred_to_gt, gt_rels, mode, nogc_num)


    def calculate_mean_recall(self, mode):
        self._calculate_single(self.result_dict, mode)

        for nogc_num in self.nogc_thres_num:
            self._calculate_single(self.nogc_result_dict, mode, nogc_num)




def _triplet(relations, classes, boxes, predicate_scores=None, class_scores=None):
    """
    format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
    Parameters:
        relations (#rel, 3) : (sub_id, ob_id, pred_label)
        classes (#objs, ) : class labels of objects
        boxes (#objs, 4)
        predicate_scores (#rel, ) : scores for each predicate
        class_scores (#objs, ) : scores for each object
    Returns:
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplets_boxes (#rel, 8) array of boxes for the parts
        triplets_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """
    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[sub_id], predicate_scores, class_scores[ob_id],
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                          gt_boxes, pred_boxes, iou_thrs, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union).numpy()[0] >= iou_thrs

        else:
            sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4]).numpy()[0]
            obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:]).numpy()[0]

            inds = (sub_iou >= iou_thrs) & (obj_iou >= iou_thrs)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


def _compute_pair_matches(gt_triplets, pred_triplets,
                          gt_boxes, pred_boxes, iou_thrs, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(np.column_stack((gt_triplets[:, 0:1], gt_triplets[:, 2:3])),
                         np.column_stack((pred_triplets[:, 0:1], pred_triplets[:, 2:3])))
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union).numpy()[0] >= iou_thrs

        else:
            sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4]).numpy()[0]
            obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:]).numpy()[0]

            inds = (sub_iou >= iou_thrs) & (obj_iou >= iou_thrs)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt