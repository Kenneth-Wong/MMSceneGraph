import torch
import numpy as np
from mmdet.ops.nms import nms_wrapper
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_dist=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
        return_dist (bool): whether to return score dist.

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)[:, 1:]
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    scores = multi_scores[:, 1:]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    valid_box_idxes = torch.nonzero(valid_mask)[:, 0].view(-1)
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    score_dists = scores[valid_box_idxes, :]
    # add bg column for later use.
    score_dists = torch.cat((torch.zeros(score_dists.size(0), 1).to(score_dists), score_dists), dim=-1)
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
        if return_dist:
            return bboxes, labels, multi_bboxes.new_zeros((0, num_classes + 1))
        else:
            return bboxes, labels

    # Modified from https://github.com/pytorch/vision/blob
    # /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = bboxes.max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    bboxes_for_nms = bboxes + offsets[:, None]
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    dets, keep = nms_op(
        torch.cat([bboxes_for_nms, scores[:, None]], 1), **nms_cfg_)
    bboxes = bboxes[keep]
    scores = dets[:, -1]  # soft_nms will modify scores
    labels = labels[keep]
    score_dists = score_dists[keep]

    if keep.size(0) > max_num:
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]
        score_dists = score_dists[inds]

    if return_dist:
        return torch.cat([bboxes, scores[:, None]], 1), labels, score_dists
    else:
        return torch.cat([bboxes, scores[:, None]], 1), labels


def multiclass_nms_pts(multi_bboxes,
                       multi_pts,
                       multi_scores,
                       multi_masks,
                       score_thr,
                       nms_cfg,
                       max_num=-1,
                       score_factors=None):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)[:, 1:]
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)

    pts = multi_pts[:, None].expand(-1, num_classes, multi_pts.shape[-1])
    masks = multi_masks[:, None].expand(-1, num_classes, multi_masks.shape[-1])
    scores = multi_scores[:, 1:]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    pts = pts[valid_mask]
    masks = masks[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        pts = multi_pts.new_zeros((0, 52))
        masks = multi_masks.new_zeros((0, 26))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)

        return bboxes, pts, masks, labels

    # Modified from https://github.com/pytorch/vision/blob
    # /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = bboxes.max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    bboxes_for_nms = bboxes + offsets[:, None]
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    dets, keep = nms_op(
        torch.cat([bboxes_for_nms, scores[:, None]], 1), **nms_cfg_)
    bboxes = bboxes[keep]
    pts = pts[keep]
    masks = masks[keep]
    scores = dets[:, -1]  # soft_nms will modify scores
    labels = labels[keep]

    if keep.size(0) > max_num:
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        pts = pts[inds]
        masks = masks[inds]
        scores = scores[inds]
        labels = labels[inds]

    bboxes = torch.cat([bboxes, scores[:, None]], 1)
    pts = torch.cat([pts, scores[:, None]], 1)
    masks = torch.cat([masks, scores[:, None]], 1)

    return bboxes, pts, masks, labels


def multiclass_nms_for_cluster(multi_bboxes,
                               multi_scores,
                               labels,
                               nms_thres=0.5):
    """NMS for multi-class bboxes.

        Args:
            multi_bboxes (np.array): shape (n, #class*4) or (n, 4)
            multi_scores (np.array): shape (n, ),
            score_thr (float): bbox threshold, bboxes with scores lower than it
                will not be considered.
            nms_cfg (float): NMS IoU threshold
            max_num (int): if there are more than max_num bboxes after NMS,
                only top max_num will be kept.

        Returns:
            tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
                are 0-based.
        """
    # Modified from https://github.com/pytorch/vision/blob
    # /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = multi_bboxes.max()
    offsets = labels * (max_coordinate + 1)
    bboxes_for_nms = multi_bboxes + offsets[:, None]
    order = np.argsort(multi_scores)[::-1]
    num_box = len(multi_bboxes)
    suppressed = np.zeros(num_box)
    gathered = (np.ones(num_box) * -1).astype(np.int32)
    ious = bbox_overlaps(bboxes_for_nms, bboxes_for_nms)
    for i in range(num_box):
        if suppressed[order[i]]:
            continue
        for j in range(i+1, num_box):
            if suppressed[order[j]]:
                continue
            iou = ious[order[i], order[j]]
            if iou >= nms_thres:
                suppressed[order[j]] = 1
                gathered[order[j]] = order[i]
    keep = np.where(suppressed == 0)[0]
    return keep, gathered