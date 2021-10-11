# ---------------------------------------------------------------
# sg_handler.py
# Set-up time: 2020/11/5 10:25
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import cv2
import random
import numpy as np
from pandas import DataFrame, Series
from graphviz import Digraph
import torchtext  # 0. install torchtext==0.2.3 (pip install torchtext==0.2.3)
import os.path as osp
import os
import mmcv
from .visualization_tools import SameNodeDetection, FindObjectClassColor, \
    CompareObjects, BboxSizeResample, Visualization

fasttext = torchtext.vocab.FastText()
_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)
colorlist = [(random.randint(0, 230), random.randint(0, 230), random.randint(0, 230)) for i in range(10000)]

SND = SameNodeDetection()
FOCC = FindObjectClassColor()
TFCO = CompareObjects()
RBS = BboxSizeResample()
TFV = Visualization()


class SceneGraphHandler(object):
    def __init__(self, args, object_classes, predicate_classes):
        # self.data = DataFrame({"node_feature":[]}, index=[])
        # [class, index, score, bounding_box, 3d_pose, mean, var, pt_number, color]
        self.vocab_objects = object_classes
        self.vocab_predicates = predicate_classes
        self.num_objects = len(self.vocab_objects) - 1
        self.num_predicates = len(self.vocab_predicates) - 1
        self.data = DataFrame({"class": Series([], dtype='int'),
                               "idx": Series([], dtype='int'),
                               "score": [],  # check
                               "bounding_box": [], "3d_pose": [], "mean": [],
                               "var": [], "pt_num": [], "color_hist": [],
                               "detection_cnt:": []},
                              columns=['class', 'idx', 'score', 'bounding_box', '3d_pose', 'mean',
                                       'var', 'pt_num', 'color_hist', 'detection_cnt'])
        self.rel_data = DataFrame({"relation": []}, index=[])
        self.covered_objects = []  # the global object id, maintain the least set of rels that cover objects
        self.nodeid2drawid = {}  # global draw id assignemt
        self.newest_drawid = 0   # newest draw id
        self.img_count = 0
        self.pt_num = 0
        self.mean = [0, 0, 0]
        self.var = [0, 0, 0]
        self.args = args
        self.detect_cnt_thres = args.detect_cnt_thres
        self.disable_samenode = self.args.disable_samenode
        if self.disable_samenode:
            self.detect_cnt_thres = 0
        self.format = args.format

    # test_set, obj_labels, obj_boxes, obj_scores,
    # subject_inds, predicate_inds, object_inds,
    # subject_IDs, object_IDs, triplet_scores, relationships, ):
    def vis_scene_graph(self, image_scene, result, idx, camera_pose,
                        pix_depth=None, inv_p_matrix=None, inv_R=None, Trans=None, dataset='scannet',
                        mode='local_demo', return_mode='image',
                        h_ratio=None, w_ratio=None, image_scene_original=None):
        updated_image_scene = image_scene.copy()
        sg = Digraph('structs', format=self.format)  # initialize scene graph tool
        print('-ID--|----Object-----|Score|3D_position (x, y, z)|---var-------------|---color------')

        obj_boxes = result.refine_bboxes[:, :-1].copy()
        # add a column to obj_boxes
        obj_boxes = np.hstack((obj_boxes, np.zeros(len(obj_boxes))[:, None]))
        obj_labels = result.refine_labels
        obj_scores = result.refine_bboxes[:, -1]

        threshold = 0.8127

        updated_nodes = []      # In the same frame, updating of the same node is not allowed.
        filtered_new_objs = []  # When an Error is threw during adding a new object,  this object is dropped
        newly_inserted_nodes = []
        newly_inserted_rels = []
        global_num = len(self.data)
        for i, (obj_box, obj_label, obj_score) in enumerate(
                zip(obj_boxes, obj_labels, obj_scores)):  # loop for bounding boxes on each images
            '''1. Get Color Histogram'''
            color_hist = None  # TFCO.get_color_hist2(box_whole_img)

            '''2. Get Center Patch '''
            # Define bounding box info
            width = int(obj_box[2]) - int(obj_box[0])
            height = int(obj_box[3]) - int(obj_box[1])
            box_center_x = int(obj_box[0]) + width / 2
            box_center_y = int(obj_box[1]) + height / 2
            # using belows to find mean and variance of each bounding boxes
            # pop 1/5 size window_box from object bounding boxes
            range_x_min, range_x_max, range_y_min, range_y_max = RBS.make_window_size(width, height, obj_box)

            '''3. Get 3D positions of the Centor Patch'''
            window_3d_pts = []
            for pt_x in range(range_x_min, range_x_max):
                for pt_y in range(range_y_min, range_y_max):
                    pose_2d_window = np.array([pt_x, pt_y, 1])[None]
                    pose_3d_window = (pix_depth[pt_y][pt_x]) * np.matmul(inv_p_matrix, pose_2d_window.transpose())
                    pose_3d = np.array([pose_3d_window.item(0), pose_3d_window.item(1), pose_3d_window.item(2), 1],
                                       dtype='float')[None]
                    pose_3d_world_coord_window = np.matmul(camera_pose, pose_3d.transpose())
                    # pose_3d_world_coord_window = np.matmul(inv_R, pose_3d_window[0:3] - Trans.transpose())

                    if not RBS.isNoisyPoint(pose_3d_world_coord_window):
                        # save several points in window_box to calculate mean and variance
                        window_3d_pts.append(
                            [pose_3d_world_coord_window.item(0), pose_3d_world_coord_window.item(1),
                             pose_3d_world_coord_window.item(2)])

            window_3d_pts = RBS.outlier_filter(window_3d_pts)

            '''4. Get a 3D position of the Center Patch's Center point'''
            # find 3D point of the bounding box(the center patch)'s center
            curr_pt_num, curr_mean, curr_var, flag = TFCO.Measure_new_Gaussian_distribution(window_3d_pts)
            if not flag:
                filtered_new_objs.append(i)
                continue

            # ex: np.matrix([[X_1],[Y_1],[Z_1]])

            # get object class names as strings
            obj_name = self.vocab_objects[obj_label]

            '''5. Save Object Recognition Results in DataFrame Format'''
            if (self.img_count == 0):
                # first image -> make new node
                self.pt_num, self.mean, self.var, flag = TFCO.Measure_new_Gaussian_distribution(window_3d_pts)
                if not flag:
                    filtered_new_objs.append(i)
                    continue
                box_id = len(self.data)
                # check
                start_data = {"class": int(obj_label), "idx": int(box_id), "score": obj_score,
                              "bounding_box": obj_box.tolist()[:-1],
                              "3d_pose": [int(self.mean[0]), int(self.mean[1]), int(self.mean[2])],
                              "mean": self.mean, "var": self.var, "pt_num": self.pt_num,
                              "color_hist": color_hist, "detection_cnt": 1}
                obj_box[4] = box_id
                newly_inserted_nodes.append(len(self.data))
                self.data.loc[len(self.data)] = start_data
            else:
                # get node similarity score: Only compare with the current nodes, not with the nodes detected on this frame
                node_score, max_score_index = SND.node_update(window_3d_pts, self.data[:global_num], curr_mean, curr_var,
                                                              obj_name, obj_scores[i], self.vocab_objects)

                # double check: the object class must be the same, it can be updated
                if not self.disable_samenode and node_score > threshold \
                        and self.data.at[max_score_index, "class"] == obj_label and max_score_index not in updated_nodes:
                    # change value of global_node
                    # change global_node[max_score_index]
                    print("node updated!!!")
                    updated_nodes.append(max_score_index)
                    self.data.at[max_score_index, "score"] = obj_score
                    self.pt_num, self.mean, self.var = \
                        TFCO.Measure_added_Gaussian_distribution(window_3d_pts, self.data.iloc[max_score_index]["mean"],
                                                                 self.data.iloc[max_score_index]["var"],
                                                                 self.data.iloc[max_score_index]["pt_num"],
                                                                 len(window_3d_pts))
                    self.data.at[max_score_index, "mean"] = self.mean
                    self.data.at[max_score_index, "var"] = self.var
                    self.data.at[max_score_index, "pt_num"] = self.pt_num
                    self.data.at[max_score_index, "color_hist"] = color_hist
                    self.data.at[max_score_index, "detection_cnt"] = self.data.iloc[max_score_index]["detection_cnt"] + 1
                    box_id = self.data.iloc[max_score_index]["idx"]
                    obj_box[4] = box_id
                else:
                    # make new_node in global_node
                    # [class, index, score, bounding_box, 3d_pose, mean, var, pt_number, color_hist]
                    self.pt_num, self.mean, self.var, flag = TFCO.Measure_new_Gaussian_distribution(window_3d_pts)
                    if not flag:
                        filtered_new_objs.append(i)
                        continue
                    box_id = len(self.data)
                    obj_box[4] = box_id
                    add_node_list = [int(obj_label), int(box_id), obj_score, obj_box.tolist()[:-1],
                                     [self.mean[0], self.mean[1], self.mean[2]],
                                     self.mean, self.var, self.pt_num, color_hist, 1]
                    newly_inserted_nodes.append(len(self.data))
                    self.data.loc[len(self.data)] = add_node_list

            # if object index was changed, update relation's object index also

            '''6. Print object info'''
            print('{obj_ID:5} {obj_cls:15}  {obj_score:4.2f}'.format(
                obj_ID=box_id, obj_cls=obj_name, obj_score=obj_score) + str(self.mean) + '  ' + str(self.var))

            '''7. Plot '''
            # updated object_detection
            cv2.rectangle(updated_image_scene,
                          (int(obj_box[0]), int(obj_box[1])),
                          (int(obj_box[2]), int(obj_box[3])),
                          colorlist[int(obj_box[4])], 2)
            font_scale = 0.5
            txt = str(box_id) + '. ' + str(obj_name) + ' ' + str(round(obj_score, 2))
            ((txt_w, txt_h), _) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            # Place text background.
            x0, y0 = int(obj_box[0]), int(obj_box[3])
            back_tl = x0, y0 - int(1.3 * txt_h)
            back_br = x0 + txt_w, y0
            cv2.rectangle(updated_image_scene, back_tl, back_br, colorlist[int(obj_box[4])], -1)
            cv2.putText(updated_image_scene,
                        txt,
                        (x0, y0 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        1)

        # filter the relationships
        keep_obj_ids = np.where(np.isin(np.arange(len(obj_boxes), dtype=np.int32), filtered_new_objs) == 0)[0]
        obj_boxes = obj_boxes[keep_obj_ids]
        obj_labels = obj_labels[keep_obj_ids]
        obj_scores = obj_scores[keep_obj_ids]
        old_to_new = dict(zip(keep_obj_ids.tolist(), list(range(len(keep_obj_ids)))))
        new_rel_pair_idxes = []
        rel_pair_idxes = result.rel_pair_idxes
        keep_rel_idxes = []
        for rel_idx, rel in enumerate(rel_pair_idxes):
            if rel[0] not in keep_obj_ids or rel[1] not in keep_obj_ids:
                continue
            new_rel_pair_idxes.append([old_to_new[rel[0]], old_to_new[rel[1]]])
            keep_rel_idxes.append(rel_idx)

        new_rel_pair_idxes = np.array(new_rel_pair_idxes).astype(np.int32)
        result.rel_pair_idxes = new_rel_pair_idxes
        result.rel_labels = result.rel_labels[keep_rel_idxes]
        if len(keep_rel_idxes) > 0:
            result.rels = np.hstack((result.rel_pair_idxes, result.rel_labels[:, None]))
        else:
            result.rels = np.array([]).astype(np.int32)
        result.rel_dists = result.rel_dists[keep_rel_idxes]
        result.triplet_scores = result.triplet_scores[keep_rel_idxes]

        rel_prev_num = len(self.rel_data)
        print('-------Subject--------|-------Predicate-----|--------Object---------|--Score-')
        for i, (relation, triplet_score) in enumerate(zip(result.rels, result.triplet_scores)):
            # update relation's class also
            subj_local_idx, obj_local_idx = int(relation[0]), int(relation[1])
            subj_global_idx, obj_global_idx = int(obj_boxes[int(relation[0]), 4]), int(obj_boxes[int(relation[1]), 4])
            # accumulate relation_list
            if subj_global_idx != obj_global_idx:
                if subj_global_idx in self.covered_objects and obj_global_idx in self.covered_objects:
                    continue
                else:
                    if subj_global_idx not in self.covered_objects:
                        self.covered_objects.append(subj_global_idx)
                    if obj_global_idx not in self.covered_objects:
                        self.covered_objects.append(obj_global_idx)

                    newly_inserted_rels.append(len(self.rel_data))
                    self.rel_data.loc[len(self.rel_data)] = [[subj_global_idx, int(relation[2]), obj_global_idx]]



                print('{sbj_cls:9} {sbj_ID:4} {sbj_score:1.3f}  |  '
                      '{pred_cls:11} {pred_score:1.3f}  |  '
                      '{obj_cls:9} {obj_ID:4} {obj_score:1.3f}  |  '
                      '{triplet_score:1.3f}'.format(
                    sbj_cls=self.vocab_objects[obj_labels[subj_local_idx]],
                    sbj_score=obj_scores[subj_local_idx],
                    sbj_ID=str(subj_global_idx),
                    pred_cls=self.vocab_predicates[int(relation[2])],
                    pred_score=triplet_score / obj_scores[subj_local_idx] / obj_scores[obj_local_idx],
                    obj_cls=self.vocab_objects[obj_labels[obj_local_idx]],
                    obj_score=obj_scores[obj_local_idx],
                    obj_ID=str(obj_global_idx),
                    triplet_score=triplet_score))

        rel_new_num = len(self.rel_data)
        # it's help to select starting point of first image manually
        self.img_count += 1

        # Return the result:
        # . 1. Normal: for local demo: return
        if mode == 'local_demo':
            if (rel_prev_num != rel_new_num):
                TFV.Draw_connected_scene_graph(self.data, self.rel_data, self.img_count-1, sg, idx,
                                               self.vocab_objects, self.vocab_predicates,
                                               self.detect_cnt_thres, self.args.plot_graph, show=True)
                return updated_image_scene

        elif mode == 'remote_demo':
            if return_mode == 'image':
                TFV.Draw_connected_scene_graph(self.data, self.rel_data, self.img_count-1, sg, idx,
                                               self.vocab_objects, self.vocab_predicates,
                                               self.detect_cnt_thres, self.args.plot_graph,
                                               save_path=self.args.save_path_sg, show=False)

            elif return_mode == 'json':
                assert h_ratio is not None and w_ratio is not None
                sg_json = dict()
                obj_boxes[:, 0::2] /= w_ratio
                obj_boxes[:, 1::2] /= h_ratio

                # version 2:
                sg_json['links'] = []
                sg_json['nodes'] = []
                sg_json['id'] = self.img_count - 1
                #nodeid2drawid = {}
                for i in newly_inserted_nodes:
                    node = self.data.iloc[i]
                    # only write the newly added
                    assign_id = self.newest_drawid
                    self.nodeid2drawid[node["idx"]] = assign_id
                    self.newest_drawid += 1
                    sg_json['nodes'].append({"group": 0, "group_name": "object", 'id': assign_id,
                                             "name": self.vocab_objects[int(node["class"])] + "_" + str(int(node["idx"])),
                                             "score": float(node["score"]),
                                             "new": 1})
                relation_list = []
                for num in range(len(self.rel_data)):
                    relation_list.append(
                        (self.rel_data.iloc[num]["relation"][0], self.rel_data.iloc[num]["relation"][1],
                         self.rel_data.iloc[num]["relation"][2]))

                # add link
                for i in newly_inserted_rels:
                    rel = relation_list[i]
                    # only write the newly added
                    assign_id = self.newest_drawid
                    self.newest_drawid += 1
                    self.nodeid2drawid['rel_%d' % i] = assign_id
                    sg_json['nodes'].append({"group": 0, "group_name": "relation", 'id': assign_id,
                                             "name": self.vocab_predicates[int(relation_list[i][1])],
                                             "new": 1})
                    sg_json['links'].append({"source": int(self.nodeid2drawid[int(rel[0])]), "target": assign_id})
                    sg_json['links'].append({"source": assign_id, "target": int(self.nodeid2drawid[int(rel[2])])})

                # writing the detection result
                mmcv.imwrite(image_scene_original,
                             osp.join(self.args.save_path_frame, str(self.img_count - 1) + '.png'))
                det_result = {"obj_boxes": obj_boxes[:, :-1].tolist(),
                              "obj_labels": [self.vocab_objects[int(l_index)] for l_index in obj_labels],
                              "obj_scores": obj_scores.tolist(), "pose": camera_pose.tolist()}
                mmcv.dump(det_result, osp.join(self.args.save_path_det, str(self.img_count - 1) + '.json'))

                mmcv.dump(sg_json, osp.join(self.args.save_path_sg, str(self.img_count - 1) + '.json'))
                return sg_json

            else:
                raise NotImplementedError