# ---------------------------------------------------------------
# image.py
# Set-up time: 2020/4/26 上午8:41
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import mmcv
import cv2
import numpy as np
from mmcv.image import imread, imwrite
from .color import color_palette, float_palette
import os.path as osp
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
import time
from collections import defaultdict

# entity_color_pools:
#   the first two are for gt: refered/not refered in the graph
#   the last two are for prediction: grounded/not grounded to the groundtruth entities.
entity_color_pools = ['SandyBrown', 'PaleGreen', 'LightCoral', 'GreenYellow']
rel_color_pools = ['Violet', 'SkyBlue']


def draw_abstract_graph(name_dict, rels, predicate_names, work_dir, filename, entity_scores=None, rel_scores=None,
                        triplet_scores=None, entity_colors_dict=None, rel_colors_dict=None):
    type = 'gt' if rel_scores is None else 'pred'
    from graphviz import Digraph
    u = Digraph('GT Scene Graph', format='pdf')
    u.body.append('size="6, 6"')
    u.body.append('rankdir="LR"')
    u.node_attr.update(style='filled')
    for i, name in name_dict.items():
        c = entity_color_pools[entity_colors_dict[i]]
        entity_label = name_dict[i]
        if entity_scores is not None:
            entity_label += '|{:.02f}'.format(entity_scores[i])
        u.node(str(i), label=entity_label, color=c)
    for i, rel in enumerate(rels):
        edge_key = '%s_%s_%s' % (rel[0], rel[1], rel[2])
        edge_label = predicate_names[rel[2]]
        if rel_scores is not None:
            edge_label += '|{:.02f}'.format(rel_scores[i])
        if triplet_scores is not None:
            edge_label += '|{:.02f}'.format(triplet_scores[i])
        u.node(edge_key, label=edge_label, color=rel_color_pools[rel_colors_dict[i]])
        u.edge(str(rel[0]), edge_key)
        u.edge(edge_key, str(rel[1]))
    u.render(osp.join(work_dir, filename + '_{}_sg'.format(type)))
    sg_im = convert_from_path(osp.join(work_dir, filename + '_{}_sg.pdf'.format(type)))  # PIL list
    return sg_im[0]


def get_name_dict(class_names, labels):
    name_cnt = {n: 1 for n in class_names}
    name_dict = {}
    for idx, l in enumerate(labels):
        name = class_names[l]
        suffix = name_cnt[name]
        name_cnt[name] += 1
        name_dict[idx] = name + '_' + str(suffix)
    return name_dict


def imdraw_sg(img,
              pred_bboxes,
              pred_labels,
              pred_rels,
              gt_bboxes=None,
              gt_labels=None,
              gt_rels=None,
              pred_scores=None,
              pred_rel_scores=None,
              pred_triplet_scores=None,
              class_names=None,
              predicate_names=None,
              score_thr=0.3,
              iou_thr=0.5,
              work_dir=None,
              filename=None,
              backend='graphviz'):
    """
    TODO: Currently the backend: networkx has some bugs. You'd better use graphviz.
    """
    img = imread(img)
    h, w = img.shape[:2]
    # create the figure
    if gt_rels is None:
        nrows, ncols = 1, 2
    else:
        nrows, ncols = 2, 2
    figsize = [50, 20]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axflat = ax.flat

    # for networkx
    node_size = 4000
    font_size = 20
    line_width = 2
    arrowsize = 20

    if score_thr > 0 and pred_scores is not None:
        inds = np.where(pred_scores >= score_thr)[0]
        rel_inds = np.ones(len(pred_rels), dtype=np.bool)
        for i, rel in enumerate(pred_rels):
            if rel[0] not in inds or rel[1] not in inds:
                rel_inds[i] = 0
        pred_bboxes = pred_bboxes[inds, :]
        pred_scores = pred_scores[inds]
        pred_labels = pred_labels[inds]

        pred_rels = pred_rels[rel_inds]
        pred_rel_scores = pred_rel_scores[rel_inds]
        pred_triplet_scores = pred_triplet_scores[rel_inds]

        # adjust the box id in the pred_rels
        entity_mapping_ = {ind: i for i, ind in enumerate(inds)}
        tmp = []
        for rel in pred_rels:
            tmp.append([entity_mapping_[rel[0]], entity_mapping_[rel[1]], rel[2]])
        pred_rels = np.array(tmp)

    subplot_offset = 0
    gt_to_pred = None
    if gt_rels is not None:
        subplot_offset = 2
        gt_entity_colors_dict, gt_rel_colors_dict = {}, {}
        # draw the gt scene graph: both on image and abstract graph
        gt_name_dict = get_name_dict(class_names, gt_labels)
        gt_rel_inds = gt_rels[:, :2].ravel().tolist()
        axflat[0].imshow(img)
        axflat[0].axis('off')
        for i, (bbox, label) in enumerate(zip(gt_bboxes, gt_labels)):
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            gt_entity_colors_dict[i] = 0 if i in gt_rel_inds else 1
            gt_rel_colors_dict[i] = 0
            axflat[0].add_patch(
                plt.Rectangle(left_top, bbox_int[2] - bbox_int[0], bbox_int[3] - bbox_int[1], fill=False,
                              edgecolor=entity_color_pools[gt_entity_colors_dict[i]], linewidth=4.5))
            axflat[0].text(bbox_int[0], bbox_int[1] + 2, gt_name_dict[i],
                           bbox=dict(facecolor=entity_color_pools[gt_entity_colors_dict[i]], alpha=0.5),
                           fontsize=15, color='black')
        axflat[0].set_title('GT Object Visualization', fontsize=25)

        if backend == 'graphviz':
            gt_abstract_graph = draw_abstract_graph(gt_name_dict, gt_rels, predicate_names, work_dir, filename,
                                                    entity_colors_dict=gt_entity_colors_dict,
                                                    rel_colors_dict=gt_rel_colors_dict)

            gt_abstract_graph = gt_abstract_graph.resize((w, h))
            axflat[1].imshow(np.asarray(gt_abstract_graph))
        elif backend == 'networkx':
            import networkx as nx
            nodes, node_colors, edges, edge_labels, edge_colors = [], [], [], {}, []
            for idx in range(len(gt_name_dict)):
                nodes.append(gt_name_dict[idx])
                node_colors.append(entity_color_pools[gt_entity_colors_dict[idx]])
            for idx, rel in enumerate(gt_rels):
                edges.append([gt_name_dict[rel[0]], gt_name_dict[rel[1]]])
                edge_labels[(gt_name_dict[rel[0]], gt_name_dict[rel[1]])] = predicate_names[rel[2]]
                edge_colors.append(rel_color_pools[gt_rel_colors_dict[idx]])
            plt.sca(axflat[1])
            g = nx.DiGraph()
            g.add_nodes_from(nodes)
            g.add_edges_from(edges)
            pos = nx.circular_layout(g)
            nx.draw(g, pos, edge_color=edge_colors, width=line_width, ax=axflat[1],
                    linewidth=1, node_size=node_size, node_color=node_colors, font_size=font_size,
                    labels={node: node for node in g.nodes()}, arrowsize=arrowsize, connectionstyle='arc3, rad = 0.2')
            nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=font_size, font_color='black',
                                         ax=axflat[1])

        axflat[1].axis('off')
        axflat[1].set_title('GT Scene Graph Visualization', fontsize=25)

        """
        step 1: (Find the equivalent boxes) group the prediction box: because there may be more than one boxes can be 
                grounded to the same gt box. 
        """
        ious = bbox_overlaps(pred_bboxes, gt_bboxes)
        pred_to_gt = np.zeros(len(pred_bboxes), dtype=np.int32)
        pred_to_gt_iou = np.zeros(len(pred_bboxes))
        pred_to_gt_iou.fill(-1)
        for pred_ind, (box, label) in enumerate(zip(pred_bboxes, pred_labels)):
            cand_gt_inds = np.where(gt_labels == label)[0]
            if len(cand_gt_inds) == 0:
                pred_to_gt[pred_ind] = -1
                continue
            target_ious = ious[pred_ind, cand_gt_inds]
            max_gt_iou, max_gt_index = np.max(target_ious), np.argmax(target_ious)
            pred_to_gt[pred_ind] = cand_gt_inds[max_gt_index]
            pred_to_gt_iou[pred_ind] = max_gt_iou
        # for each gt, find all the qualified predicted boxes
        qualified_inds = np.where(pred_to_gt_iou > iou_thr)[0]
        gt_to_pred = defaultdict(list)
        for pred_ind in qualified_inds:
            gt_to_pred[pred_to_gt[pred_ind]].append((pred_ind, pred_to_gt_iou[pred_ind]))
        gt_to_pred = dict(gt_to_pred)
        for k, v in gt_to_pred.items():
            gt_to_pred[k] = sorted(v, key=lambda x: x[1], reverse=True)

    """
    Step 2: For each predicted relation R, evaluate whether it is grounded to a gt relation:
        1). The subject and object can be grounded to a gt object;
        2). The relation can be found in the gt relations;
        3). If the gt relation associated by R has been shown, R will not be shown. 
    """
    # (1) map the all the predicted boxes to its equivalent boxes
    _eq_inner_mapping = {}
    _eq_gt_mapping = {}
    if gt_to_pred is not None and len(gt_to_pred) > 0:
        for k, v in gt_to_pred.items():
            for _v in v:
                _eq_inner_mapping[_v[0]] = v[0][0]  # the first one has the highest iou, so this is the flag
            _eq_gt_mapping[v[0][0]] = k

    # (2) replace the predicted relation indexes and scores:
    new_rels = {}
    for rel, rel_score, triplet_score in zip(pred_rels, pred_rel_scores, pred_triplet_scores):
        new_rel_pair = (_eq_inner_mapping.get(rel[0], rel[0]), _eq_inner_mapping.get(rel[1], rel[1]))
        if new_rel_pair in new_rels and rel[2] not in [v[0] for v in new_rels[new_rel_pair]]:
            new_rels[new_rel_pair].append((rel[2], rel_score, triplet_score))
        else:
            new_rels[new_rel_pair] = [(rel[2], rel_score, triplet_score)]
    # find the visible bbox idx, and adjust the relations, assign the entity colors and relation colors
    pred_entity_colors_dict, pred_rel_colors_dict = {}, {}
    vis_pred_idxes = np.ones(len(pred_bboxes), dtype=np.bool)
    for pred_ind in range(len(vis_pred_idxes)):
        if pred_ind in _eq_inner_mapping.keys() and pred_ind not in _eq_inner_mapping.values():
            vis_pred_idxes[pred_ind] = 0
    pred_bboxes = pred_bboxes[vis_pred_idxes, :]
    pred_labels = pred_labels[vis_pred_idxes]
    pred_scores = pred_scores[vis_pred_idxes] if pred_scores is not None else None
    _o2n_mapping = {idx: i for i, idx in enumerate(np.where(vis_pred_idxes)[0])}
    grd_idxes = [_o2n_mapping[i] for i in list(set(list(_eq_inner_mapping.values())))]
    for i in range(len(pred_bboxes)):
        pred_entity_colors_dict[i] = 2 if i in grd_idxes else 3

    new_pred_rels = []
    new_pred_rel_scores = [] if pred_rel_scores is not None else None
    new_pred_triplet_scores = [] if pred_triplet_scores is not None else None
    grounded_gtrel_idxes = []
    gt_rel_lists = gt_rels.tolist() if gt_rels is not None else None
    for rel_pair, cand_predicates in new_rels.items():
        subj, obj = _o2n_mapping[rel_pair[0]], _o2n_mapping[rel_pair[1]]
        if rel_pair[0] in _eq_gt_mapping and rel_pair[1] in _eq_gt_mapping:
            # NOTE: there may be one of them do not match the gt box, check it !!!!
            for cand_predicate in cand_predicates:
                cand_rel = [_eq_gt_mapping[rel_pair[0]], _eq_gt_mapping[rel_pair[1]], cand_predicate[0]]
                if cand_rel in gt_rel_lists and gt_rel_lists.index(cand_rel) not in grounded_gtrel_idxes:
                    grounded_gtrel_idxes.append(gt_rel_lists.index(cand_rel))
                    pred_rel_colors_dict[len(new_pred_rels)] = 0
                    new_pred_rels.append([subj, obj, cand_predicate[0]])
                    new_pred_rel_scores.append(cand_predicate[1])
                    new_pred_triplet_scores.append(cand_predicate[2])
        else:
            for cand_predicate in cand_predicates:
                pred_rel_colors_dict[len(new_pred_rels)] = 1
                new_pred_rels.append([subj, obj, cand_predicate[0]])
                new_pred_rel_scores.append(cand_predicate[1])
                new_pred_triplet_scores.append(cand_predicate[2])

    pred_name_dict = get_name_dict(class_names, pred_labels)
    axflat[subplot_offset].imshow(img)
    axflat[subplot_offset].axis('off')
    for i, (bbox, label, score) in enumerate(zip(pred_bboxes, pred_labels, pred_scores)):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        c = entity_color_pools[pred_entity_colors_dict[i]]
        axflat[subplot_offset].add_patch(
            plt.Rectangle(left_top, bbox_int[2] - bbox_int[0], bbox_int[3] - bbox_int[1], fill=False,
                          edgecolor=c, linewidth=4.5))
        axflat[subplot_offset].text(bbox_int[0], bbox_int[1] + 2, pred_name_dict[i] + '|{:.02f}'.format(score),
                                    bbox=dict(facecolor=c, alpha=0.5),
                                    fontsize=15, color='black')
    axflat[subplot_offset].set_title('Predicted Object Visualization', fontsize=25)
    # axflat[subplot_offset].savefig(osp.join(work_dir, filename + '_vis_pred_object.png'), bbox_inches='tight')

    if backend == 'graphviz':
        pred_abstract_graph = draw_abstract_graph(pred_name_dict, new_pred_rels, predicate_names, work_dir, filename,
                                                  entity_scores=pred_scores,
                                                  rel_scores=new_pred_rel_scores,
                                                  triplet_scores=new_pred_triplet_scores,
                                                  entity_colors_dict=pred_entity_colors_dict,
                                                  rel_colors_dict=pred_rel_colors_dict)

        pred_abstract_graph = pred_abstract_graph.resize((w, h))
        axflat[subplot_offset + 1].imshow(np.asarray(pred_abstract_graph))
    elif backend == 'networkx':
        import networkx as nx
        nodes, node_colors, edges, edge_labels, edge_colors = [], [], [], {}, []
        for idx in range(len(pred_name_dict)):
            nodes.append(pred_name_dict[idx])
            node_colors.append(entity_color_pools[pred_entity_colors_dict[idx]])
        for idx, rel in enumerate(new_pred_rels):
            edges.append([pred_name_dict[rel[0]], pred_name_dict[rel[1]]])
            edge_labels[(pred_name_dict[rel[0]], pred_name_dict[rel[1]])] = predicate_names[rel[2]]
            edge_colors.append(rel_color_pools[pred_rel_colors_dict[idx]])
        plt.sca(axflat[subplot_offset + 1])
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges[:5])
        pos = nx.circular_layout(g)
        nx.draw(g, pos, edge_color=edge_colors, width=line_width, ax=axflat[subplot_offset + 1],
                linewidth=1, node_size=node_size, node_color=node_colors, font_size=font_size,
                labels={node: node for node in g.nodes()}, arrowsize=arrowsize, connectionstyle='arc3, rad = 0.05')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=font_size, font_color='black',
                                     ax=axflat[subplot_offset + 1], connectionstyle='arc3, rad = 0.05')

    axflat[subplot_offset + 1].axis('off')
    axflat[subplot_offset + 1].set_title('Predicted Scene Graph Visualization', fontsize=25)

    plt.tight_layout()
    fig.savefig(osp.join(work_dir, filename + '_vis_sg.png'), bbox_inches='tight')


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      palette=dict(palette='pastel', n_colors=7),
                      thickness=2,
                      font_scale=0.7,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        palette (dict): Palette parameters containing 'palette', 'n_colors', and 'desat'.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    palette = color_palette(**palette)
    num_colors = len(palette)

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, palette[i % num_colors], thickness=thickness, lineType=cv2.LINE_AA)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 3),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, palette[i % num_colors])

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)
