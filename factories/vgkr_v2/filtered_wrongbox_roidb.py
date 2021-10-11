# ---------------------------------------------------------------
# filtered_wrongbox_roidb.py
# Set-up time: 2021/3/1 15:15
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import argparse, json, string
from collections import Counter
import math
import os.path as osp
from math import floor
import numpy as np
import pprint
from factories.vgkr_v2.config_v2 import *
from factories.utils.tools import make_alias_dict, make_alias_dict_from_synset
import mmcv
import pandas as pd
from random import shuffle, seed
import h5py


def main(params):
    f = h5py.File(params['h5_filtered'], 'w')
    roi_h5 = h5py.File(params['h5_input'], 'r')

    all_labels = roi_h5['labels'][:, 0]
    all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['bboxes'][:]  # x1, y1, x2, y2
    all_boxes = np.array(all_boxes, dtype=np.float32)  # must be float32
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box
    im_to_first_box = roi_h5['img_to_first_box'][:]
    im_to_last_box = roi_h5['img_to_last_box'][:]

    # load relation seqs
    im_to_first_rel = roi_h5['img_to_first_rel'][:]
    im_to_last_rel = roi_h5['img_to_last_rel'][:]
    _relations = roi_h5['relationships'][:]
    rel_inputs = np.array(roi_h5['rel_inputs'][:], dtype='int')
    rel_targets = np.array(roi_h5['rel_targets'][:], dtype='int')
    rel_ipt_scores = roi_h5['rel_ipt_scores'][:, 0]

    # load caption seqs
    im_to_first_cap = roi_h5['img_to_first_cap'][:]
    im_to_last_cap = roi_h5['img_to_last_cap'][:]
    cap_inputs = roi_h5['cap_inputs'][:]
    cap_targets = roi_h5['cap_targets'][:]

    keep_boxes_ids = np.where(np.bitwise_and(all_boxes[:, 0]<=all_boxes[:, 2], all_boxes[:, 1]<=all_boxes[:, 3]))[0]
    print('filtered %d boxes'%(len(all_boxes)-len(keep_boxes_ids)))

    old_to_new_map = dict(zip(keep_boxes_ids.tolist(), list(range(len(keep_boxes_ids)))))
    bboxes = all_boxes[keep_boxes_ids]
    attributes = all_attributes[keep_boxes_ids]
    labels = all_labels[keep_boxes_ids]
    new_im_to_first_box = []
    new_im_to_last_box = []
    pbar = mmcv.ProgressBar(len(im_to_first_box))
    for i in range(len(im_to_first_box)):
        if im_to_first_box[i] == -1:
            assert im_to_last_box[i] == -1
            new_im_to_first_box.append(im_to_first_box[i])
            new_im_to_last_box.append(im_to_last_box[i])
        else:
            consecutive_ids = list(range(im_to_first_box[i], im_to_last_box[i]+1))
            segs = []
            for j in consecutive_ids:
                if j in old_to_new_map:
                    segs.append(old_to_new_map[j])
            if len(segs) == 0:
                new_im_to_first_box.append(-1)
                new_im_to_last_box.append(-1)
            else:
                new_im_to_first_box.append(min(segs))
                new_im_to_last_box.append(max(segs))
        pbar.update()

    new_rels = []
    new_im_to_first_rel = []
    new_im_to_last_rel = []
    keep_rel_ids = []
    #pbar = mmcv.ProgressBar(len(_relations))

    keep_rel_ids = np.where(np.logical_and(np.isin(_relations[:, 0], keep_boxes_ids),
                                           np.isin(_relations[:, 1], keep_boxes_ids)))[0]
    box_id_map = np.zeros(all_boxes.shape[0])
    box_id_map[keep_boxes_ids] = np.arange(len(keep_boxes_ids))
    box_id_map = box_id_map.astype(np.int)
    new_rels = _relations[keep_rel_ids]
    new_rels[:, 0] = box_id_map[new_rels[:, 0]]
    new_rels[:, 1] = box_id_map[new_rels[:, 1]]

    # for idx, rel in enumerate(_relations):
    #     if rel[0] not in keep_boxes_ids or rel[1] not in keep_boxes_ids:
    #         continue
    #     new_rels.append([old_to_new_map[rel[0]], old_to_new_map[rel[1]]])
    #     keep_rel_ids.append(idx)
    #     pbar.update()
    # new_rels = np.array(new_rels)
    keep_rel_idxes = keep_rel_ids
    rel_inputs = rel_inputs[keep_rel_idxes]
    rel_targets = rel_targets[keep_rel_idxes]
    rel_ipt_scores = rel_ipt_scores[keep_rel_idxes]
    old_to_new_map = dict(zip(keep_rel_ids, list(range(len(keep_rel_ids)))))
    pbar = mmcv.ProgressBar(len(im_to_first_rel))
    for i in range(len(im_to_first_rel)):
        if im_to_first_rel[i] == -1:
            assert im_to_last_rel[i] == -1
            new_im_to_first_rel.append(im_to_first_rel[i])
            new_im_to_last_rel.append(im_to_last_rel[i])
        else:
            consecutive_ids = list(range(im_to_first_rel[i], im_to_last_rel[i]+1))
            segs = []
            for j in consecutive_ids:
                if j in old_to_new_map:
                    segs.append(old_to_new_map[j])
            if len(segs) == 0:
                new_im_to_first_rel.append(-1)
                new_im_to_last_rel.append(-1)
            else:
                new_im_to_first_rel.append(min(segs))
                new_im_to_last_rel.append(max(segs))
        pbar.update()

    f.create_dataset('labels', data=labels[:, None])
    f.create_dataset('bboxes', data=bboxes)
    f.create_dataset('attributes', data=attributes)
    f.create_dataset('img_to_first_box', data=new_im_to_first_box)
    f.create_dataset('img_to_last_box', data=new_im_to_last_box)
    f.create_dataset('relationships', data=new_rels)
    f.create_dataset('rel_inputs', data=rel_inputs)
    f.create_dataset('rel_targets', data=rel_targets)
    f.create_dataset('rel_ipt_scores', data=rel_ipt_scores[:, None])
    f.create_dataset('img_to_first_rel', data=new_im_to_first_rel)
    f.create_dataset('img_to_last_rel', data=new_im_to_last_rel)


    f.create_dataset('cap_inputs', data=cap_inputs)
    f.create_dataset('cap_targets', data=cap_targets)
    f.create_dataset('img_to_first_cap', data=im_to_first_cap)
    f.create_dataset('img_to_last_cap', data=im_to_last_cap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--meta_info', default=meta_form_file)

    # output database
    parser.add_argument('--h5_input', default=vggn_roidb_file)
    parser.add_argument('--h5_filtered', default=osp.join('/'.join(vggn_roidb_file.split('/')[:-1]),
                                                          'debugged_' + vggn_roidb_file.split('/')[-1]))

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)