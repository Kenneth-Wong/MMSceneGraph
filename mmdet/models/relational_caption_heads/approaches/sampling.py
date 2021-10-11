# ---------------------------------------------------------------
# sampling.py
# Set-up time: 2021/2/14 上午10:53
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import torch
from torch.nn import functional as F
import numpy as np
import numpy.random as npr
from mmdet.core import bbox_overlaps
import random


class RelationalCapSampler(object):
    def __init__(
            self,
            pos_iou_thr,
            num_sample_per_gt_rel,
            num_rel_per_image,
            use_gt_box,
            label_match=True,
            test_overlap=False):
        self.pos_iou_thr = pos_iou_thr
        self.num_sample_per_gt_rel = num_sample_per_gt_rel
        self.num_rel_per_image = num_rel_per_image
        self.use_gt_box = use_gt_box
        self.label_match = label_match
        self.test_overlap = test_overlap

    def prepare_test_pairs(self, det_result):
        # prepare object pairs for relation prediction
        rel_pair_idxes = []
        device = det_result.bboxes[0].device
        for p in det_result.bboxes:
            n = len(p)
            cand_matrix = torch.ones((n, n), device=device) - torch.eye(n, device=device)
            # mode==sgdet and require_overlap
            if (not self.use_gt_box) and self.test_overlap:
                cand_matrix = cand_matrix.byte() & bbox_overlaps(p, p).gt(0).byte()
            idxs = torch.nonzero(cand_matrix).view(-1, 2)
            if len(idxs) > 0:
                rel_pair_idxes.append(idxs)
            else:
                # if there is no candidate pairs, give a placeholder of [[0, 0]]
                rel_pair_idxes.append(torch.zeros((1, 2), dtype=torch.int64, device=device))
        return rel_pair_idxes

    def gtbox_relsample(self, det_result, gt_result):
        assert self.use_gt_box
        num_per_img = self.num_rel_per_image
        rel_idx_pairs = []
        rel_cap_inputs = []
        rel_cap_targets = []
        rel_ipts = []
        rel_sym_binarys = []
        bboxes, labels = det_result.bboxes, det_result.labels
        gt_bboxes, gt_labels, gt_rels, gt_rel_cap_inputs, gt_rel_cap_targets, gt_rel_ipts = gt_result.bboxes, \
                                                                                                  gt_result.labels, \
                                                                                                  gt_result.rel_pair_idxes, \
                                                                                                  gt_result.rel_cap_inputs, \
                                                                                                  gt_result.rel_cap_targets, \
                                                                                                  gt_result.rel_ipts
        device = bboxes[0].device
        for img_id, (prp_box, prp_lab, tgt_box, tgt_lab, tgt_rels,
                     tgt_rel_cap_input, tgt_rel_cap_target, tgt_rel_ipt_score) in \
                enumerate(zip(bboxes, labels, gt_bboxes, gt_labels, gt_rels,
                              gt_rel_cap_inputs, gt_rel_cap_targets, gt_rel_ipts)):
            num_prp = prp_box.shape[0]
            assert num_prp == tgt_box.shape[0]
            tgt_pair_idxs = tgt_rels.clone()
            assert tgt_pair_idxs.shape[1] == 2
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)

            # sym_binary_rels
            binary_rel = torch.zeros((num_prp, num_prp), device=device).long()
            binary_rel[tgt_head_idxs, tgt_tail_idxs] = 1
            binary_rel[tgt_tail_idxs, tgt_head_idxs] = 1
            rel_sym_binarys.append(binary_rel)

            num_fg = tgt_pair_idxs.shape[0]
            seq_len = tgt_rel_cap_input.size(1)

            # construct sample results
            # img_rel_idxs = torch.zeros((num_per_img, 2), device=device).long()
            # img_tgt_rel_cap_inputs = torch.zeros((num_per_img, seq_len), device=device).long()
            # img_tgt_rel_cap_targets = torch.zeros((num_per_img, seq_len), device=device).long()
            # img_tgt_rel_ipts = torch.zeros(num_per_img, device=device).long()

            # if num_fg >= num_per_img:
            perm = torch.randperm(num_fg, device=device)[:num_per_img]
            img_rel_idxs = tgt_pair_idxs[perm]
            img_tgt_rel_cap_inputs = tgt_rel_cap_input[perm]
            img_tgt_rel_cap_targets = tgt_rel_cap_target[perm]
            img_tgt_rel_ipts = tgt_rel_ipt_score[perm]
            # else:
                # img_rel_idxs[:num_fg] = tgt_pair_idxs
                # img_tgt_rel_cap_inputs[:num_fg] = tgt_rel_cap_input
                # img_tgt_rel_cap_targets[:num_fg] = tgt_rel_cap_target
                # img_tgt_rel_ipts[:num_fg] = tgt_rel_ipt_score
                #
                # ixs = torch.from_numpy(np.array(random.sample(range(num_fg), num_per_img - num_fg))).to(device).long()
                # #ixs = torch.randperm(num_fg, device=device)[:(num_per_img - num_fg)]
                # pos = torch.arange(len(ixs)).long() + num_fg
                # img_rel_idxs[pos] = tgt_pair_idxs[ixs]
                # img_tgt_rel_cap_inputs[pos] = tgt_rel_cap_input[ixs]
                # img_tgt_rel_cap_targets[pos] = tgt_rel_cap_target[ixs]
                # img_tgt_rel_ipts[pos] = tgt_rel_ipt_score[ixs]

            rel_idx_pairs.append(img_rel_idxs)
            rel_cap_inputs.append(img_tgt_rel_cap_inputs)
            rel_cap_targets.append(img_tgt_rel_cap_targets)
            rel_ipts.append(img_tgt_rel_ipts)

        return rel_idx_pairs, rel_cap_inputs, rel_cap_targets, rel_ipts, rel_sym_binarys

    def detect_relsample(self, det_result, gt_result):
        # corresponding to rel_assignments function in neural-motifs
        """
        The input proposals are already processed by subsample function of box_head,
        in this function, we should only care about fg box, and sample corresponding fg/bg relations
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])  contain fields: labels, boxes(5 columns)
            targets (list[BoxList]) contain fields: labels
        """
        num_per_img = self.num_rel_per_image
        bboxes, labels = det_result.bboxes, det_result.labels
        gt_bboxes, gt_labels, gt_rels, gt_rel_cap_inputs, gt_rel_cap_targets, gt_rel_ipts = gt_result.bboxes, \
                                                                                                  gt_result.labels, \
                                                                                                  gt_result.rel_pair_idxes, \
                                                                                                  gt_result.rel_cap_inputs, \
                                                                                                  gt_result.rel_cap_targets, \
                                                                                                  gt_result.rel_ipts
        device = bboxes[0].device
        rel_idx_pairs = []
        rel_cap_inputs = []
        rel_cap_targets = []
        rel_ipts = []
        rel_sym_binarys = []
        for img_id, (prp_box, prp_lab, tgt_box, tgt_lab, tgt_rels,
                     tgt_rel_cap_input, tgt_rel_cap_target, tgt_rel_ipt_score) in \
                enumerate(zip(bboxes, labels, gt_bboxes, gt_labels, gt_rels,
                              gt_rel_cap_inputs, gt_rel_cap_targets, gt_rel_ipts)):
            # IoU matching
            ious = bbox_overlaps(tgt_box, prp_box[:, :4])  # [tgt, prp]
            is_match = (ious > self.pos_iou_thr)  # [tgt, prp]
            if self.label_match:
                is_match = is_match & (tgt_lab[:, None] == prp_lab[None])

            img_rel_idxs, img_tgt_rel_cap_inputs, img_tgt_rel_cap_targets, img_tgt_rel_ipts, binary_rel = \
                self.rel_sampling(device, tgt_rels, tgt_rel_cap_input, tgt_rel_cap_target, tgt_rel_ipt_score,
                                  num_per_img, ious, is_match)
            rel_idx_pairs.append(img_rel_idxs)
            rel_cap_inputs.append(img_tgt_rel_cap_inputs)
            rel_cap_targets.append(img_tgt_rel_cap_targets)
            rel_ipts.append(img_tgt_rel_ipts)
            rel_sym_binarys.append(binary_rel)

        return rel_idx_pairs, rel_cap_inputs, rel_cap_targets, rel_ipts, rel_sym_binarys

    def rel_sampling(self, device, tgt_rels, tgt_rel_cap_inputs, tgt_rel_cap_targets, tgt_rel_ipts,
                     num_per_img, ious, is_match):
        """
        prepare to sample fg relation triplet and bg relation triplet
        tgt_rel_matrix: # [number_target, number_target]
        ious:           # [number_target, num_proposal]
        is_match:       # [number_target, num_proposal]
        rel_possibility:# [num_proposal, num_proposal]
        """
        tgt_pair_idxs = tgt_rels.clone()
        assert tgt_pair_idxs.shape[1] == 2
        tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
        tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)

        # generate binary prp mask
        num_prp = is_match.shape[-1]
        binary_prp_head = is_match[tgt_head_idxs]  # num_fg, num_prp (matched prp head)
        binary_prp_tail = is_match[tgt_tail_idxs]  # num_fg, num_prp (matched prp head)
        binary_rel = torch.zeros((num_prp, num_prp), device=device).long()

        fg_rel_pair_idxs = []
        fg_rel_cap_input = []
        fg_rel_cap_target = []
        fg_rel_ipt_score = []
        for i in range(tgt_rels.shape[0]):
            # generate binary prp mask
            bi_match_head = torch.nonzero(binary_prp_head[i] > 0)
            bi_match_tail = torch.nonzero(binary_prp_tail[i] > 0)

            num_bi_head = bi_match_head.shape[0]
            num_bi_tail = bi_match_tail.shape[0]
            if num_bi_head > 0 and num_bi_tail > 0:
                bi_match_head = bi_match_head.view(1, num_bi_head).expand(num_bi_tail, num_bi_head).contiguous()
                bi_match_tail = bi_match_tail.view(num_bi_tail, 1).expand(num_bi_tail, num_bi_head).contiguous()
                # binary rel only consider related or not, so its symmetric
                binary_rel[bi_match_head.view(-1), bi_match_tail.view(-1)] = 1
                binary_rel[bi_match_tail.view(-1), bi_match_head.view(-1)] = 1

            tgt_head_idx = int(tgt_head_idxs[i])
            tgt_tail_idx = int(tgt_tail_idxs[i])
            tgt_rel_cap_input = tgt_rel_cap_inputs[i][None, :]
            tgt_rel_cap_target = tgt_rel_cap_targets[i][None, :]
            tgt_rel_ipt_score = tgt_rel_ipts[i][None]

            #tgt_rel_lab = int(tgt_rel_labs[i])
            # find matching pair in proposals (might be more than one)
            prp_head_idxs = torch.nonzero(is_match[tgt_head_idx]).squeeze(1)
            prp_tail_idxs = torch.nonzero(is_match[tgt_tail_idx]).squeeze(1)
            num_match_head = prp_head_idxs.shape[0]
            num_match_tail = prp_tail_idxs.shape[0]
            if num_match_head <= 0 or num_match_tail <= 0:
                continue
            # all combination pairs
            prp_head_idxs = prp_head_idxs.view(-1, 1).expand(num_match_head, num_match_tail).contiguous().view(-1)
            prp_tail_idxs = prp_tail_idxs.view(1, -1).expand(num_match_head, num_match_tail).contiguous().view(-1)
            valid_pair = prp_head_idxs != prp_tail_idxs
            if valid_pair.sum().item() <= 0:
                continue
            # remove self-pair
            # remove selected pair from rel_possibility
            prp_head_idxs = prp_head_idxs[valid_pair]
            prp_tail_idxs = prp_tail_idxs[valid_pair]

            # construct corresponding proposal triplets corresponding to i_th gt relation
            fg_rel_cap_input_i = torch.cat([tgt_rel_cap_input] * prp_tail_idxs.shape[0], 0)
            fg_rel_cap_target_i = torch.cat([tgt_rel_cap_target] * prp_tail_idxs.shape[0], 0)
            fg_rel_ipt_score_i = torch.cat([tgt_rel_ipt_score] * prp_tail_idxs.shape[0], 0)
            fg_rel_pair_idxs_i = torch.cat((prp_head_idxs.view(-1, 1), prp_tail_idxs.view(-1, 1)), dim=-1).to(torch.int64)

            # select if too many corresponding proposal pairs to one pair of gt relationship triplet
            # NOTE that in original motif, the selection is based on a ious_score score
            if fg_rel_pair_idxs_i.shape[0] > self.num_sample_per_gt_rel:
                ious_score = (ious[tgt_head_idx, prp_head_idxs] * ious[tgt_tail_idx, prp_tail_idxs]).view(
                    -1).detach().cpu().numpy()
                ious_score = ious_score / ious_score.sum()
                perm = npr.choice(ious_score.shape[0], p=ious_score, size=self.num_sample_per_gt_rel, replace=False)
                fg_rel_pair_idxs_i = fg_rel_pair_idxs_i[perm]
                fg_rel_cap_input_i = fg_rel_cap_input_i[perm]
                fg_rel_cap_target_i = fg_rel_cap_target_i[perm]
                fg_rel_ipt_score_i = fg_rel_ipt_score_i[perm]

            if fg_rel_pair_idxs_i.shape[0] > 0:
                fg_rel_pair_idxs.append(fg_rel_pair_idxs_i)
                fg_rel_cap_input.append(fg_rel_cap_input_i)
                fg_rel_cap_target.append(fg_rel_cap_target_i)
                fg_rel_ipt_score.append(fg_rel_ipt_score_i)

        num_fg = sum([p.shape[0] for p in fg_rel_pair_idxs])
        seq_len = tgt_rel_cap_inputs.size(1)

        # img_rel_idxs = torch.zeros((num_per_img, 2), device=device).long()
        # img_tgt_rel_cap_inputs = torch.zeros((num_per_img, seq_len), device=device).long()
        # img_tgt_rel_cap_targets = torch.zeros((num_per_img, seq_len), device=device).long()
        # img_tgt_rel_ipts = torch.zeros(num_per_img, device=device).long()

        if num_fg > 0:
            fg_rel_pair_idxs = torch.cat(fg_rel_pair_idxs, 0)
            fg_rel_cap_input = torch.cat(fg_rel_cap_input, 0)
            fg_rel_cap_target = torch.cat(fg_rel_cap_target, 0)
            fg_rel_ipt_score = torch.cat(fg_rel_ipt_score, 0)

            perm = torch.randperm(num_fg, device=device)[:num_per_img]
            img_rel_idxs = fg_rel_pair_idxs[perm]
            img_tgt_rel_cap_inputs = fg_rel_cap_input[perm]
            img_tgt_rel_cap_targets = fg_rel_cap_target[perm]
            img_tgt_rel_ipts = fg_rel_ipt_score[perm]
        # elif num_fg > 0:
        #     img_rel_idxs[:num_fg] = fg_rel_pair_idxs
        #     img_tgt_rel_cap_inputs[:num_fg] = fg_rel_cap_input
        #     img_tgt_rel_cap_targets[:num_fg] = fg_rel_cap_target
        #     img_tgt_rel_ipts[:num_fg] = fg_rel_ipt_score
        #
        #     #ixs = torch.randperm(num_fg, device=device)[:(num_per_img - num_fg)]
        #     ixs = torch.from_numpy(np.array(random.sample(range(num_fg), num_per_img - num_fg))).to(device).long()
        #     pos = torch.arange(len(ixs)).long() + num_fg
        #     img_rel_idxs[pos] = tgt_pair_idxs[ixs]
        #     img_tgt_rel_cap_inputs[pos] = fg_rel_cap_input[ixs]
        #     img_tgt_rel_cap_targets[pos] = fg_rel_cap_target[ixs]
        #     img_tgt_rel_ipts[pos] = fg_rel_ipt_score[ixs]
        else:
            img_rel_idxs = torch.zeros((1, 2), device=device).long()
            img_tgt_rel_cap_inputs = torch.zeros((1, seq_len), device=device).long()
            img_tgt_rel_cap_targets = torch.zeros((1, seq_len), device=device).long()
            img_tgt_rel_ipts = torch.zeros(1, device=device).long()

        return img_rel_idxs, img_tgt_rel_cap_inputs, img_tgt_rel_cap_targets, img_tgt_rel_ipts, binary_rel
