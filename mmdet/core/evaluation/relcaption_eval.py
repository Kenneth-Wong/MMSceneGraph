# utf-8
# ---------------------------------------------------------------
# relcaption_eval.py
# Set-up time: 2021/2/26 10:19
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
from mmdet.utils import print_log
import torch
import mmcv
import numpy as np
from mmdet.models.captioners.utils import decode_sequence
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from .meteor_bridge import Meteor_


def relcaption_evaluation(
        groundtruths,
        predictions,
        logger,
        vocab,
        min_overlaps=[0.2, 0.3, 0.4, 0.5, 0.6],
        min_scores=[-1, 0, 0.05, 0.1, 0.15, 0.2, 0.25],
        topN=[-1, 2, 5],
        topNrec=[-1, 20, 50, 100]):
    m = Meteor_()
    result_container = dict()
    msg = 'Evaluating {}...'.format('mAP')
    if logger is None:
        msg = '\n' + msg
    print_log(msg, logger=logger)
    result_container.update(relcaption_evaluation_mAP(
        groundtruths,
        predictions,
        m,
        logger,
        vocab,
        min_overlaps,
        min_scores))

    msg = 'Evaluating {}...'.format('imgRecall')
    if logger is None:
        msg = '\n' + msg
    print_log(msg, logger=logger)
    result_container.update(relcaption_evaluation_recall(
        groundtruths,
        predictions,
        m,
        logger,
        vocab,
        min_overlaps,
        min_scores,
        topN))

    msg = 'Evaluating {}...'.format('imgLocRecall')
    if logger is None:
        msg = '\n' + msg
    print_log(msg, logger=logger)
    pair_orders = ['random', 'prob']
    if predictions[0].rel_ipt_scores is not None:
        pair_orders += ['score', 'mul']
    for pair_order in pair_orders:
        result_container.update(
            relcaption_evaluation_loc_recall(groundtruths, predictions, m, logger, vocab, min_overlaps, min_scores,
                                             topNrec, pair_order=pair_order))
    return result_container


def relcaption_evaluation_mAP(groundtruths, predictions, scorer, logger, vocab, min_overlaps, min_scores):
    # process every single images
    npos = 0
    candidates, references = [], []
    pair_max_ovs, pair_max_ov_gts = [], []
    for prediction, groundtruth in zip(predictions, groundtruths):
        gt_boxes = groundtruth.bboxes
        gt_rels = groundtruth.rel_pair_idxes
        gt_rel_cap_seqs = groundtruth.rel_cap_seqs
        gt_rel_cap_sents = decode_sequence(vocab, torch.from_numpy(gt_rel_cap_seqs))

        pred_boxes = prediction.bboxes[:, :4]
        pred_rels = prediction.rel_pair_idxes
        pred_rel_cap_sents = prediction.rel_cap_sents

        num_gt_rels = gt_rels.shape[0]

        # compute the overlap between the predicted boxes and gt boxes
        pred_subj_boxes, pred_obj_boxes = pred_boxes[pred_rels[:, 0], :], pred_boxes[pred_rels[:, 1], :]
        gt_subj_boxes, gt_obj_boxes = gt_boxes[gt_rels[:, 0], :], gt_boxes[gt_rels[:, 1], :]
        ious_subj, ious_obj = bbox_overlaps(pred_subj_boxes, gt_subj_boxes), bbox_overlaps(pred_obj_boxes, gt_obj_boxes)
        pred_pair_ov_gts = np.minimum(ious_subj, ious_obj)  # #predicted pairs x # gt pairs: used the smaller one
        pred_pair_max_ovs, pred_pair_max_ov_gt_pair = np.max(pred_pair_ov_gts, 1), np.argmax(pred_pair_ov_gts, 1)
        pair_max_ovs.append(pred_pair_max_ovs)
        pair_max_ov_gts.append(pred_pair_max_ov_gt_pair)
        candidates += pred_rel_cap_sents
        for i in pred_pair_max_ov_gt_pair:
            if pred_pair_max_ovs[i] == 0:
                references.append(None)
            else:
                references.append([gt_rel_cap_sents[i]])

        npos += num_gt_rels

    pair_max_ovs = np.hstack(pair_max_ovs)
    print('Computing meteor score...\n')
    meteor_scores_dict = scorer.compute(candidates, references)
    print('\ncomputing meteor score finished.\n')
    scores = meteor_scores_dict['scores']
    n = len(scores)
    ap_results = {}
    det_results = {}
    for min_overlap in min_overlaps:
        for min_score in min_scores:
            tp = np.zeros(n)
            fp = np.zeros(n)

            for i in range(n):
                if references[i] is None:
                    fp[i] = 1
                else:
                    if pair_max_ovs[i] >= min_overlap and scores[i] > min_score:
                        tp[i] = 1
                    else:
                        fp[i] = 1
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / npos
            prec = tp / (tp + fp)

            ap = 0
            apn = 0
            for t in np.arange(0, 1, 0.01):
                mask = (rec >= t).astype(np.int)
                prec_masked = prec * mask
                p = np.max(prec_masked)
                ap = ap + p
                apn = apn + 1
            ap = ap / apn
            if min_score == -1:
                det_results['ov_{}'.format(min_overlap)] = ap
            else:
                ap_results['ov_{}_score_{}'.format(min_overlap, min_score)] = ap
    return dict(map=sum(ap_results.values()) / len(ap_results),
                detmap=sum(det_results.values()) / len(det_results))


def _recall(scores, references, npos, min_overlaps, min_scores, ovs=None):
    n = len(scores)
    recall_results = {}
    det_results = {}
    for min_overlap in min_overlaps:
        for min_score in min_scores:
            tp = np.zeros(n)
            tn = np.zeros(n)

            for i in range(n):
                if references[i] is None:
                    tn[i] = 1
                else:
                    if scores[i] > min_score:
                        if ovs is not None:
                            if ovs[i] >= min_overlap:
                                tp[i] = 1
                            else:
                                tn[i] = 1
                        else:
                            tp[i] = 1
                    else:
                        tn[i] = 1
            tn = np.cumsum(tn)
            tp = np.cumsum(tp)
            rec = tp / npos
            prec = tp / (tp + tn)

            recall = 0
            recallN = 0
            for t in np.arange(0, 1, 0.01):
                mask = (rec >= t).astype(np.int)
                prec_masked = prec * mask
                p = np.max(prec_masked)
                recall = recall + p
                recallN = recallN + 1
            recall = recall / recallN
            if min_score == -1:
                det_results['ov_{}'.format(min_overlap)] = recall
            else:
                recall_results['ov_{}_score_{}'.format(min_overlap, min_score)] = recall
    if len(min_scores) == 1 and min_scores[0] == -1:
        return sum(det_results.values()) / len(det_results)
    else:
        return sum(recall_results.values()) / len(recall_results)


def relcaption_evaluation_recall(groundtruths, predictions, scorer, logger, vocab, min_overlaps, min_scores,
                                 topN=[-1, 2, 5]):
    # process every single images
    npos = 0
    candidates, ipt_flags = [], []
    references = {topn: [] for topn in topN}
    for prediction, groundtruth in zip(predictions, groundtruths):
        gt_boxes = groundtruth.bboxes
        gt_rels = groundtruth.rel_pair_idxes
        gt_rel_cap_seqs = groundtruth.rel_cap_seqs
        gt_rel_cap_sents = decode_sequence(vocab, torch.from_numpy(gt_rel_cap_seqs))
        gt_rel_ipts = groundtruth.rel_ipts

        pred_boxes = prediction.bboxes[:, :4]
        pred_rel_ipt_scores = prediction.rel_ipt_scores
        pred_rels = prediction.rel_pair_idxes
        pred_rel_cap_sents = prediction.rel_cap_sents
        if pred_rel_ipt_scores is not None:
            pair_order = np.argsort(pred_rel_ipt_scores)[::-1]
            pred_rels = pred_rels[pair_order]
            pred_rel_cap_sents = [pred_rel_cap_sents[i] for i in pair_order]

        num_gt_rels = gt_rels.shape[0]

        candidates += gt_rel_cap_sents
        ipt_flags += gt_rel_ipts.tolist()

        for topn in topN:
            if topn == -1:
                references[topn] += ([pred_rel_cap_sents] * num_gt_rels)
            else:
                references[topn] += ([pred_rel_cap_sents[:topn]] * num_gt_rels)
        npos += num_gt_rels

    meteor_scores_res = {}
    for topn in topN:
        print('Computing meteor score for top-{}...\n'.format(topn))
        meteor_scores_res[topn] = scorer.compute(candidates, references[topn])
        print('\ncomputing meteor score finished.\n')

    res_dict = {}

    # compute the recall with all predictions (topn = -1) from the angle of sentences is right or wrong,
    # the overlap is not required.
    scores = meteor_scores_res[-1]['scores']
    meteor = meteor_scores_res[-1]['average_score']
    mrecall = _recall(scores, references[-1], npos, [0], min_scores)
    res_dict['meteor'] = meteor
    res_dict['mrecall_sentences'] = mrecall

    # compute the ipt recall respect to topn > 0
    ipt_flags = np.array(ipt_flags)
    keep_idx = np.where(ipt_flags)[0]
    if len(keep_idx) == 0:
        for topn in topN:
            res_dict['mrecall_ipt_top{}'.format(topn)] = -1
    else:
        npos = len(keep_idx)
        for topn in topN:
            if topn > 0:
                scores = meteor_scores_res[topn]['scores']
                scores = [scores[i] for i in keep_idx]
                refs = references[topn]
                refs = [refs[i] for i in keep_idx]
                mr = _recall(scores, refs, npos, [0], min_scores)
                res_dict['mrecall_ipt_top{}'.format(topn)] = mr
    return res_dict


def relcaption_evaluation_loc_recall(groundtruths, predictions, scorer, logger, vocab, min_overlaps, min_scores,
                                     topN=[-1, 2, 5], pair_order='random'):
    min_overlap = 0.5
    # process every single images
    npos = 0
    candidates, ipt_flags = [], []
    pair_max_ovs, pair_max_ov_preds = [], []
    references = {topn: [] for topn in topN}
    for prediction, groundtruth in zip(predictions, groundtruths):
        gt_boxes = groundtruth.bboxes
        gt_rels = groundtruth.rel_pair_idxes
        gt_rel_cap_seqs = groundtruth.rel_cap_seqs
        gt_rel_cap_sents = decode_sequence(vocab, torch.from_numpy(gt_rel_cap_seqs))
        gt_rel_ipts = groundtruth.rel_ipts

        pred_boxes = prediction.bboxes[:, :4]
        pred_rel_ipt_scores = prediction.rel_ipt_scores
        pred_rels = prediction.rel_pair_idxes
        pred_rel_cap_sents = prediction.rel_cap_sents
        pred_rel_cap_scores = prediction.rel_cap_scores

        # use different methods to rank the rels
        # 1. sent_prob: use the prod of the words as the score of each sentences
        if pair_order == 'prob':
            order = np.argsort(np.prod(np.exp(pred_rel_cap_scores), 1))[::-1]
        # 2. random:
        elif pair_order == 'random':
            order = np.arange(len(pred_rel_cap_sents)).astype(np.int)
        # 3. if exists: prdicted ranking score,
        elif pair_order == 'score':
            assert pred_rel_ipt_scores is not None
            order = np.argsort(pred_rel_ipt_scores)[::-1]
        # 4. multiply score with the sent_prob:
        elif pair_order == 'mul':
            assert pred_rel_ipt_scores is not None
            order = np.argsort(np.prod(np.exp(pred_rel_cap_scores), 1) * pred_rel_ipt_scores)[::-1]
        else:
            raise NotImplementedError

        pred_rels = pred_rels[order]
        pred_rel_cap_sents = [pred_rel_cap_sents[i] for i in order]

        num_gt_rels = gt_rels.shape[0]

        candidates += gt_rel_cap_sents
        ipt_flags += gt_rel_ipts.tolist()

        npos += num_gt_rels

        # compute the overlap between the predicted boxes and gt boxes
        pred_subj_boxes, pred_obj_boxes = pred_boxes[pred_rels[:, 0], :], pred_boxes[pred_rels[:, 1], :]
        gt_subj_boxes, gt_obj_boxes = gt_boxes[gt_rels[:, 0], :], gt_boxes[gt_rels[:, 1], :]
        ious_subj, ious_obj = bbox_overlaps(gt_subj_boxes, pred_subj_boxes), bbox_overlaps(gt_obj_boxes, pred_obj_boxes)
        gt_pair_ov_preds = np.minimum(ious_subj, ious_obj)  # #gt pairs x # pred pairs: used the smaller one
        mask = gt_pair_ov_preds >= min_overlap
        gt_pair_max_ovs = np.max(gt_pair_ov_preds, 1)
        pair_max_ovs.append(gt_pair_max_ovs)

        for topn in topN:
            if topn == -1:
                cand_refs = pred_rel_cap_sents
                cand_mask = mask
            else:
                cand_refs = pred_rel_cap_sents[:topn]
                cand_mask = mask[:, :topn]
            for i in range(len(gt_pair_max_ovs)):
                cor_pred_idxs = np.where(cand_mask[i])[0]
                if len(cor_pred_idxs) == 0:  # could not recall
                    references[topn].append(None)
                else:
                    references[topn].append([cand_refs[j] for j in cor_pred_idxs])

    pair_max_ovs = np.hstack(pair_max_ovs)

    meteor_scores_res = {}
    for topn in topN:
        print('Computing meteor score for top-{}...\n'.format(topn))
        meteor_scores_res[topn] = scorer.compute(candidates, references[topn])
        print('\ncomputing meteor score finished.\n')

    res_dict = {}

    mrecall = _recall(meteor_scores_res[-1]['scores'], references[-1], npos, [min_overlap], min_scores, pair_max_ovs)
    res_dict['mrecall_loc_{}'.format(pair_order)] = mrecall
    # do not consider score
    mrecall = _recall(meteor_scores_res[-1]['scores'], references[-1], npos, [min_overlap], [-1], pair_max_ovs)
    res_dict['mrecall_loc_ns_{}'.format(pair_order)] = mrecall

    # compute the ipt recall respect to topn > 0
    ipt_flags = np.array(ipt_flags)
    keep_idx = np.where(ipt_flags)[0]
    if len(keep_idx) == 0:
        for topn in topN:
            res_dict['mrecall_loc_ipt_top{}_{}'.format(topn, pair_order)] = -1
    else:
        npos = len(keep_idx)
        for topn in topN:
            if topn > 0:
                scores = meteor_scores_res[topn]['scores']
                scores = [scores[i] for i in keep_idx]
                refs = references[topn]
                refs = [refs[i] for i in keep_idx]
                ovs = [pair_max_ovs[i] for i in keep_idx]
                mr = _recall(scores, refs, npos, [min_overlap], min_scores, ovs)
                res_dict['mrecall_loc_ipt_top{}_{}'.format(topn, pair_order)] = mr
                mr = _recall(scores, refs, npos, [min_overlap], [-1], ovs)
                res_dict['mrecall_loc_ns_ipt_top{}_{}'.format(topn, pair_order)] = mr
    return res_dict
