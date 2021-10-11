# ---------------------------------------------------------------
# het_util.py
# Set-up time: 2021/4/1 11:40
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import torch
import numpy as np

from .vctree_util import ArbitraryTree
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


def generate_forest(det_result, pick_parent, isc_thresh, child_order='leftright', num_embed_depth=None, need_depth=False):
    """
    generate a list of trees that covers all the objects in a batch
    im_inds: [obj_num]
    box_priors: [obj_num, (x1, y1, x2, y2)]
    pair_scores: [obj_num, obj_num]

    output: list of trees, each present a chunk of overlaping objects
    """
    output_forest = []  # the list of trees, each one is a chunk of overlapping objects
    output_depths = []  # the list of tree depths,

    all_bboxes, all_dists, all_labels = det_result.bboxes, det_result.dists, det_result.labels
    num_objs = [len(b) for b in all_bboxes]
    if all_dists is not None:
        node_scores = [dist.max(1)[0] for dist in all_dists]
    else:
        node_scores = [torch.ones(len(b)).to(all_bboxes[0]) for b in all_bboxes]

    areas = torch.cat(all_bboxes, 0)
    areas = (areas[:, 3] - areas[:, 1] + 1) * (areas[:, 2] - areas[:, 0] + 1)
    split_areas = areas.split(num_objs)
    split_areas = [a.cpu().numpy() for a in split_areas]
    all_sorted_idxes = [np.argsort(a)[::-1] for a in split_areas]

    bbox_intersections = [bbox_overlaps(boxes.cpu().numpy()[:, :4], boxes.cpu().numpy()[:, :4], mode='iof')
                          for boxes in all_bboxes]

    offset = 0
    for img_id, (scores, bboxes, labels, areas, sorted_idxes, intersection, num_obj) in enumerate(zip(node_scores,
                                                                                                      all_bboxes,
                                                                                                      all_labels,
                                                                                                      split_areas,
                                                                                                      all_sorted_idxes,
                                                                                                      bbox_intersections,
                                                                                                      num_objs)):
        # select the nodes from the same image
        node_container = []
        depth_labels = np.zeros(num_objs[img_id], dtype=np.int32)

        # note: the index of root is the N+tree_id
        root = ArbitraryTree(sum(num_objs)+img_id, -1, -1, is_root=True)

        bboxes = bboxes[:, :4]
        # put all nodes into node container
        for idx in range(num_obj):
            new_node = ArbitraryTree(offset + idx, scores[idx], labels[idx], bboxes[idx])
            node_container.append(new_node)

        # iteratively generate tree
        gen_het(node_container, root, areas, sorted_idxes, intersection,
                pick_parent=pick_parent, isc_thresh=isc_thresh, child_order=child_order)

        if need_depth:
            get_tree_depth(root, depth_labels, offset, num_embed_depth)
            output_depths.append(torch.from_numpy(depth_labels).long().to(bboxes.device))

        output_forest.append(root)
        offset += num_obj
    if need_depth:
        output_depths = torch.cat(output_depths, 0)
    return output_forest, output_depths


def gen_het(node_container, root, areas, sorted_idxes, intersection, pick_parent='area', isc_thresh=0.9,
            child_order='leftright'):
    num_nodes = len(node_container)
    if num_nodes == 0:
        return

    # first step: sort the rois according to areas
    sorted_node_container = [node_container[i] for i in sorted_idxes]

    if pick_parent == 'isc':
        sort_key = 1
    elif pick_parent == 'area':
        sort_key = 2
    else:
        raise NotImplementedError
    # i, j for sorted_node_container, origin_i, origin_j for node_container
    for i in range(num_nodes):
        current_node = sorted_node_container[i]
        possible_parent = []
        origin_i = sorted_idxes[i]
        for j in range(0, i):  # all nodes that are larger than current_node
            origin_j = sorted_idxes[j]
            M = intersection[origin_i, origin_j]
            N = intersection[origin_j, origin_i]
            if M > isc_thresh:
                possible_parent.append((j, N, areas[origin_j]))
        if len(possible_parent) == 0:
            # assign the parrent of i as root
            root.add_child(current_node)
        else:
            if pick_parent != 'area' and pick_parent != 'isc':
                raise NotImplementedError('%s for pick_parent not implemented' % pick_parent)

            parent_id = sorted(possible_parent, key=lambda d: d[sort_key], reverse=True)[0][0]
            sorted_node_container[parent_id].add_child(current_node)
    # sort the children
    sort_childs(root, child_order)


def sort_childs(root, order='leftright'):
    if len(root.children) == 0:
        return
    children = root.children
    boxes = np.vstack([n.box.cpu().numpy() for n in children])
    node_scores = np.array([n.score for n in children])
    if order == 'leftright':
        scores = (boxes[:, 0] + boxes[:, 2]) / 2
        scores = scores / (np.max(scores) + 1)
    elif order == 'size':
        scores = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        scores = scores / (np.max(scores) + 1)
    elif order == 'confidence':
        scores = node_scores
    elif order == 'random':
        scores = np.random.rand(len(children))
    else:
        raise NotImplementedError('Unknown sorting method: %s' % order)
    sorted_id = np.argsort(-scores)
    root.children = [children[i] for i in sorted_id]
    for i in range(len(root.children)):
        sort_childs(root.children[i], order)


def get_tree_depth(root, tree_depths, offset, num_embed_depth):
    if root.parent is not None:
        depth = root.depth()
        if num_embed_depth is not None and depth >= num_embed_depth:
            depth = num_embed_depth - 1
        tree_depths[root.index - offset] = depth
    for c in root.children:
        get_tree_depth(c, tree_depths, offset, num_embed_depth)