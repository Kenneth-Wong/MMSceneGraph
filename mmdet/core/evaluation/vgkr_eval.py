# ---------------------------------------------------------------
# vg_eval.py
# Set-up time: 2020/5/18 上午9:48
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import os
import torch
import numpy as np
import json
from .sggkr_eval import SGRecall, SGMeanRecall
from mmdet.utils import print_log
import mmcv


def vgkr_evaluation(
        mode,
        groundtruths,
        predictions,
        iou_thrs,
        logger,
        ind_to_predicates,
        multiple_preds=False,
        predicate_freq=None,
        nogc_thres_num=None):
    # only conduct Recall and mRecall for VGKR, but have triplet-match and tuple-match protocols, predcls and sgcls
    modes = mode if isinstance(mode, list) else [mode]
    result_container = dict()
    for m in modes:
        msg = 'Evaluating {}...'.format(m)
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)
        single_result_dict = vgkr_evaluation_single(m,
                                                    groundtruths,
                                                    predictions,
                                                    iou_thrs,
                                                    logger,
                                                    ind_to_predicates,
                                                    multiple_preds,
                                                    predicate_freq,
                                                    nogc_thres_num)
        result_container.update(single_result_dict)
    return result_container


def vgkr_evaluation_single(
        mode,
        groundtruths,
        predictions,
        iou_thrs,
        logger,
        ind_to_predicates,
        multiple_preds=False,
        predicate_freq=None,
        nogc_thres_num=None):
    num_predicates = len(ind_to_predicates)

    assert isinstance(nogc_thres_num, (list, tuple, int)) or nogc_thres_num is None
    if nogc_thres_num is None:
        nogc_thres_num = [num_predicates - 1]  # default: all
    elif isinstance(nogc_thres_num, int):
        nogc_thres_num = [nogc_thres_num]
    else:
        pass

    result_str = '\n' + '=' * 100 + '\n'
    result_dict = {}
    nogc_result_dict = {}
    evaluator = {}
    # tradictional Recall@K
    eval_recall = SGRecall(result_dict, nogc_result_dict, nogc_thres_num)
    eval_recall.register_container(mode)
    evaluator['eval_recall'] = eval_recall

    # used for meanRecall@K
    eval_mean_recall = SGMeanRecall(result_dict, nogc_result_dict, nogc_thres_num,
                                    num_predicates, ind_to_predicates, print_detail=True)
    eval_mean_recall.register_container(mode)
    evaluator['eval_mean_recall'] = eval_mean_recall

    # prepare all inputs
    global_container = {}
    global_container['result_dict'] = result_dict
    global_container['mode'] = mode
    global_container['multiple_preds'] = multiple_preds
    global_container['num_predicates'] = num_predicates
    global_container['iou_thrs'] = iou_thrs

    pbar = mmcv.ProgressBar(len(groundtruths))
    for groundtruth, prediction in zip(groundtruths, predictions):
        evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator)
        pbar.update()

    # calculate mean recall
    eval_mean_recall.calculate_mean_recall(mode)

    # print result
    result_str += eval_recall.generate_print_string(mode)
    result_str += eval_mean_recall.generate_print_string(mode, predicate_freq)
    result_str += '=' * 100 + '\n'

    if logger is None:
        result_str = '\n' + result_str
    print_log(result_str, logger=logger)

    return format_result_dict(result_dict, result_str, mode)


def format_result_dict(result_dict, result_str, mode):
    """
    Function:
        This is used for getting the results in both float data form and text form
        so that they can be logged into tensorboard (scalar and text).

        Here we only log the graph constraint results excluding phrdet.
    """
    formatted = dict()
    copy_stat_str = ''
    # Traditional Recall
    for k, v in result_dict[mode + '_recall']['triplet'].items():
        formatted[mode + '_triplet_recall_' + 'R_%d' % k] = np.mean(v)
        copy_stat_str += (mode + '_triplet_recall_' + 'R_%d: ' % k + '{:0.3f}'.format(np.mean(v)) + '\n')

    for k, v in result_dict[mode + '_recall']['tuple'].items():
        formatted[mode + '_tuple_recall_' + 'R_%d' % k] = np.mean(v)
        copy_stat_str += (mode + '_tuple_recall_' + 'R_%d: ' % k + '{:0.3f}'.format(np.mean(v)) + '\n')

    # mean recall
    for k, v in result_dict[mode + '_mean_recall'].items():
        formatted[mode + '_mean_recall_' + 'mR_%d' % k] = float(v)
        copy_stat_str += (mode + '_mean_recall_' + 'mR_%d: ' % k + '{:0.3f}'.format(float(v)) + '\n')

    formatted[mode + '_copystat'] = copy_stat_str

    formatted[mode + '_runtime_eval_str'] = result_str
    return formatted


def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    # unpack all inputs
    mode = global_container['mode']

    local_container = {}
    rels = groundtruth.rels
    key_rels = groundtruth.key_rels
    # if there is no gt relations for current image, then skip it
    if len(rels) == 0 or key_rels is None or (key_rels is not None and len(key_rels) == 0):
        return

    local_container['gt_rels'] = rels[key_rels]

    local_container['gt_boxes'] = groundtruth.bboxes  # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth.labels  # (#gt_objs, )

    # about relations
    local_container['pred_rel_inds'] = prediction.rel_pair_idxes  # (#pred_rels, 2)
    local_container['rel_scores'] = prediction.rel_dists  # (#pred_rels, num_pred_class)
    local_container['ranking_scores'] = prediction.ranking_scores

    # about objects
    local_container['pred_boxes'] = prediction.refine_bboxes[:, :4]  # (#pred_objs, 4)
    local_container['pred_classes'] = prediction.labels  # (#pred_objs, )
    local_container['obj_scores'] = prediction.refine_bboxes[:, -1]  # (#pred_objs, )

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    else:
        raise ValueError('invalid mode')

    if local_container['pred_rel_inds'].shape[0] == 0:
        return

    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)

    # Mean Recall
    evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)

    return


def convert_relation_matrix_to_triplets(relation):
    triplets = []
    for i in range(len(relation)):
        for j in range(len(relation)):
            if relation[i, j] > 0:
                triplets.append((i, j, relation[i, j]))
    return torch.LongTensor(triplets)  # (num_rel, 3)


def generate_attributes_target(attributes, num_attributes):
    """
    from list of attribute indexs to [1,0,1,0,...,0,1] form
    """
    max_att = attributes.shape[1]
    num_obj = attributes.shape[0]

    with_attri_idx = (attributes.sum(-1) > 0).long()
    without_attri_idx = 1 - with_attri_idx
    num_pos = int(with_attri_idx.sum())
    num_neg = int(without_attri_idx.sum())
    assert num_pos + num_neg == num_obj

    attribute_targets = torch.zeros((num_obj, num_attributes), device=attributes.device).float()

    for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
        for k in range(max_att):
            att_id = int(attributes[idx, k])
            if att_id == 0:
                break
            else:
                attribute_targets[idx, att_id] = 1

    return attribute_targets
