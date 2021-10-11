# ---------------------------------------------------------------
# visualization.py
# Set-up time: 2020/11/4 20:57
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import cv2
import random
import numpy as np
from pandas import DataFrame
import pandas as pd
from graphviz import Digraph
import webcolors
import pprint
import math
from scipy.stats import norm
from color_histogram.core.hist_3d import Hist3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torchtext  # 0. install torchtext==0.2.3 (pip install torchtext==0.2.3)
from torch.nn.functional import cosine_similarity
from collections import Counter
import pcl  # cd python-pcl -> python setup.py build-ext -i -> python setup.py install
import os.path as osp
import os

fasttext = torchtext.vocab.FastText()
_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)
colorlist = [(random.randint(0, 230), random.randint(0, 230), random.randint(0, 230)) for i in range(10000)]


class SameNodeDetection(object):
    def __init__(self):
        self.compare_all = False
        self.class_weight = 10.0 / 20.0
        self.pose_weight = 8.0 / 20.0
        self.color_weight = 2.0 / 20.0

    def compare_class(self, curr_cls, prev_cls, cls_score):
        score = 0.
        score = cosine_similarity(fasttext.vectors[fasttext.stoi[curr_cls]].cuda(),
                                  fasttext.vectors[fasttext.stoi[prev_cls]].cuda(), dim=0).cpu().item()
        score = (score + 1) / 2.
        return score

    def compare_position(self, curr_mean, curr_var, prev_mean, prev_var, prev_pt_num, new_pt_num):
        I_x, I_y, I_z = TFCO.check_distance(curr_mean, curr_var, prev_mean, prev_var)
        # score = (I_x * I_y * I_z)
        score = (I_x / 3.0) + (I_y / 3.0) + (I_z / 3.0)
        score = float(score)
        return score

    def compare_color(self, curr_hist, prev_hist):
        curr_rgb = webcolors.name_to_rgb(curr_hist[0][1])
        prev_rgb = webcolors.name_to_rgb(prev_hist[0][1])
        dist = np.sqrt(np.sum(np.power(np.subtract(curr_rgb, prev_rgb), 2))) / (255 * np.sqrt(3))
        score = 1 - dist
        return score

    def node_update(self, window_3d_pts, global_node, curr_mean, curr_var, curr_cls, cls_score,
                    object_classes):
        # temporary do not use the curr_color_hist
        try:
            new_pt_num = len(window_3d_pts)
            global_node_num = len(global_node)
            #print(global_node_num)
            score = []
            score_pose = []
            w1, w2, w3 = self.class_weight, self.pose_weight, self.color_weight
            # print("current object : {cls:3}".format(cls=curr_cls[0]))
            for i in range(global_node_num):
                #import pdb
                #pdb.set_trace()
                prev_cls = object_classes[global_node.iloc[i]["class"]]
                prev_mean, prev_var, prev_pt_num = global_node.iloc[i]["mean"], global_node.iloc[i]["var"], \
                                                   global_node.iloc[i]["pt_num"]
                prev_color_hist = global_node.iloc[i]["color_hist"]
                cls_sc = self.compare_class(curr_cls, prev_cls, cls_score)
                pos_sc = self.compare_position(curr_mean, curr_var, prev_mean, prev_var, prev_pt_num, new_pt_num)
                # col_sc = self.compare_color(curr_color_hist, prev_color_hist)
                # print("class_score {cls_score:3.2f}".format(cls_score=cls_sc))
                # print("pose_score {pos_score:3.2f}".format(pos_score=pos_sc))
                # print("color_score {col_score:3.2f}".format(col_score=col_sc))
                tot_sc = (w1 * cls_sc) + (w2 * pos_sc) + w3  # (w3 * col_sc)
                # print("total_score {tot_score:3.2f}".format(tot_score=tot_sc))
                score.append(tot_sc)
                # score_pose.append(pos_sc)
            node_score = max(score)
            print("node_score {score:3.4f}".format(score=node_score))
            max_score_index = score.index(max(score))
            # node_score_pose = score_pose[max_score_index]
            # print("node_score_pose {score_pose:3.2f}".format(score_pose=node_score_pose))
            return node_score, max_score_index
        except:
            return 0, 0


class FindObjectClassColor(object):
    def __init__(self):
        self.power = 2

    def get_class_string(self, class_index, score, dataset):
        class_text = dataset[class_index] if dataset is not None else \
            'id{:d}'.format(class_index)
        return class_text + ' {:0.2f}'.format(score).lstrip('0')

    def closest_colour(self, requested_colour):
        min_colours = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** self.power
            gd = (g_c - requested_colour[1]) ** self.power
            bd = (b_c - requested_colour[2]) ** self.power
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]

    def get_colour_name(self, requested_colour):
        try:
            closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
        except ValueError:
            closest_name = self.closest_colour(requested_colour)
            actual_name = None
        return actual_name, closest_name


class CompareObjects(object):
    def __init__(self):
        self.meter = 5000.
        self.th_x = 0.112
        self.th_y = 0.112
        self.th_z = 0.112

    def check_distance(self, x, curr_var, mean, var):
        Z_x = (x[0] - mean[0]) / self.meter
        Z_y = (x[1] - mean[1]) / self.meter
        Z_z = (x[2] - mean[2]) / self.meter
        # Z_x = (x[0]-mean[0])/np.sqrt(curr_var[0])
        # Z_y = (x[1]-mean[1])/np.sqrt(curr_var[1])
        # Z_z = (x[2]-mean[2])/np.sqrt(curr_var[2])
        # In Standardized normal gaussian distribution
        # Threshold : 0.9 --> -1.65 < Z < 1.65
        #           : 0.8 --> -1.29 < Z < 1.29
        #           : 0.7 --> -1.04 < Z < 1.04
        #           : 0.6 --> -0.845 < Z < 0.845
        #           : 0.5 --> -0.675 < Z < 0.675
        #           : 0.4 --> -0.53 < Z < 0.53
        # print("    pos {pos_x:3.2f} {pose_y:3.2f} {pose_z:3.2f}".format(pos_x=Z_x, pose_y=Z_y, pose_z=Z_z))
        # print("pos_y {pos_y:3.2f}".format(pos_y=Z_y))
        # print("pos_z {pos_z:3.2f}".format(pos_z=Z_z))
        # th_x = np.sqrt(np.abs(var[0])) *beta
        # th_y = np.sqrt(np.abs(var[1])) *beta
        # th_z = np.sqrt(np.abs(var[2])) *beta
        x_check = -self.th_x < Z_x < self.th_x
        y_check = -self.th_y < Z_y < self.th_y
        z_check = -self.th_z < Z_z < self.th_z

        if (x_check):
            I_x = 1.0
        else:
            # I_x = norm.cdf(-np.abs(Z_x)) / norm.cdf(-self.th_x)
            # I_x = (norm.cdf(self.th_x) - norm.cdf(-self.th_x)) / (norm.cdf(np.abs(Z_x)) - norm.cdf(-np.abs(Z_x)))
            I_x = self.th_x / np.abs(Z_x)
            # if (np.abs(self.th_x - Z_x)<1):
            #     I_x = np.abs(self.th_x - Z_x)
            # else:
            #     I_x = 1/np.abs(self.th_x-Z_x)
        if (y_check):
            I_y = 1.0
        else:
            # I_y = norm.cdf(-np.abs(Z_y)) / norm.cdf(-self.th_y)
            # I_y = (norm.cdf(self.th_y) - norm.cdf(-self.th_y)) / (norm.cdf(np.abs(Z_y)) - norm.cdf(-np.abs(Z_y)))
            I_y = self.th_y / np.abs(Z_y)
            # if (np.abs(self.th_y - Z_y)<1):
            #     I_y = np.abs(self.th_y - Z_y)
            # else:
            #     I_y = 1/np.abs(self.th_y-Z_y)
        if (z_check):
            I_z = 1.0
        else:
            # I_z = norm.cdf(-np.abs(Z_z)) / norm.cdf(-self.th_z)
            # I_z = (norm.cdf(self.th_z) - norm.cdf(-self.th_z)) / (norm.cdf(np.abs(Z_z)) - norm.cdf(-np.abs(Z_z)))
            I_z = self.th_z / np.abs(Z_z)
            # if (np.abs(self.th_z - Z_z)<1):
            #     I_z = np.abs(self.th_z - Z_z)
            # else:
            #     I_z = 1/np.abs(self.th_x-Z_z)

        # print("    score {score_x:3.2f} {score_y:3.2f} {score_z:3.2f}".format(score_x=I_x, score_y=I_y, score_z=I_z))
        # print("    tot_score {score:3.2f} ".format(score=(I_x+I_y+I_z)/3.))
        # print("pose_score_y {pos_score_y:3.2f}".format(pos_score_y=I_y))
        # print("pose_score_z {pos_score_z:3.2f}".format(pos_score_z=I_z))
        return I_x, I_y, I_z

    def Measure_new_Gaussian_distribution(self, new_pts):
        try:
            pt_num = len(new_pts)
            mu = np.sum(new_pts, axis=0) / pt_num
            mean = [int(mu[0]), int(mu[1]), int(mu[2])]
            var = np.sum(np.power(new_pts, 2), axis=0) / pt_num - np.power(mu, 2)
            var = [int(var[0]), int(var[1]), int(var[2])]
            return pt_num, mean, var, True
        except:
            return 1, [0, 0, 0], [1, 1, 1], False

    def Measure_added_Gaussian_distribution(self, new_pts, prev_mean, prev_var, prev_pt_num, new_pt_num):
        # update mean and variance
        pt_num = prev_pt_num + new_pt_num
        mu = np.sum(new_pts, axis=0)
        mean = np.divide((np.multiply(prev_mean, prev_pt_num) + mu), pt_num)
        mean = [int(mean[0]), int(mean[1]), int(mean[2])]
        var = np.subtract(np.divide(
            (np.multiply((prev_var + np.power(prev_mean, 2)), prev_pt_num) + np.sum(np.power(new_pts, 2), axis=0)),
            pt_num), np.power(mean, 2))
        var = [int(var[0]), int(var[1]), int(var[2])]
        return pt_num, mean, var

    def get_color_hist(self, img):
        '''
        # return color_hist
        # format: [[num_pixels1,color1],[num_pixels2,color2],...,[num_pixelsN,colorN]]
        # ex:     [[362        ,'red' ],[2          ,'blue'],...,[3          ,'gray']]
        '''

        img = img[..., ::-1]  # BGR to RGB
        img = img.flatten().reshape(-1, 3).tolist()  # shape: ((640x480)*3)

        color_hist = []
        start = 0
        new_color = False
        actual_name, closest_name = FOCC.get_colour_name(img[0])
        if (actual_name == None):
            color_hist.append([0, closest_name])
        else:
            color_hist.append([0, actual_name])

        for i in range(len(img)):
            actual_name, closest_name = FOCC.get_colour_name(img[i])
            for k in range(len(color_hist)):
                if (color_hist[k][1] == actual_name or color_hist[k][1] == closest_name):
                    color_hist[k][0] += 1
                    new_color = False
                    break
                else:
                    new_color = True
            if (new_color == True):
                if (actual_name == None):
                    color_hist.append([1, closest_name])
                    new_color = False
                else:
                    color_hist.append([1, actual_name])
                    new_color = False
        color_hist = sorted(color_hist, reverse=True)
        return color_hist

    def get_color_hist2(self, img):
        '''
        # return color_hist
        # format: [[density1,color1],[density2,color2],[density3,color3]]
        # ex:     [[362     ,'red' ],[2       ,'blue'],[3       ,'gray']]
        '''
        try:
            hist3D = Hist3D(img[..., ::-1], num_bins=8, color_space='rgb')  # BGR to RGB
            # print('sffsd:', img.shape)
            # cv2.imshow('a',img)
            # cv2.waitKey(1)
        except:

            return TFCO.get_color_hist(img)
        else:
            densities = hist3D.colorDensities()
            order = densities.argsort()[::-1]
            densities = densities[order]
            colors = (255 * hist3D.rgbColors()[order]).astype(int)
            color_hist = []
            for density, color in zip(densities, colors)[:4]:
                actual_name, closest_name = FOCC.get_colour_name(color.tolist())
                if (actual_name == None):
                    color_hist.append([density, closest_name])
                else:
                    color_hist.append([density, actual_name])

        return color_hist


class BboxSizeResample(object):
    def __init__(self):
        self.range = 10.0
        self.mean_k = 10
        self.thres = 1.0

    def isNoisyPoint(self, point):
        return -self.range < point[0] < self.range and -self.range < point[1] < self.range and -self.range < point[
            2] < self.range

    def outlier_filter(self, points):
        try:
            points_ = np.array(points, dtype=np.float32)
            cloud = pcl.PointCloud()
            cloud.from_array(points_)
            filtering = cloud.make_statistical_outlier_filter()
            filtering.set_mean_k(min(len(points_), self.mean_k))
            filtering.set_std_dev_mul_thresh(self.thres)
            cloud_filtered = filtering.filter()
            return cloud_filtered.to_array().tolist()
        except:
            return points

    def make_window_size(self, width, height, obj_boxes):
        if (width < 30):
            range_x_min = int(obj_boxes[0]) + int(width * 3. / 10.)
            range_x_max = int(obj_boxes[0]) + int(width * 7. / 10.)
        elif (width < 60):
            range_x_min = int(obj_boxes[0]) + int(width * 8. / 20.)
            range_x_max = int(obj_boxes[0]) + int(width * 12. / 20.)
        else:
            range_x_min = int(obj_boxes[0]) + int(width * 12. / 30.)
            range_x_max = int(obj_boxes[0]) + int(width * 18. / 30.)

        if (height < 30):
            range_y_min = int(obj_boxes[1]) + int(height * 3. / 10.)
            range_y_max = int(obj_boxes[1]) + int(height * 7. / 10.)
        elif (height < 60):
            range_y_min = int(obj_boxes[1]) + int(height * 8. / 20.)
            range_y_max = int(obj_boxes[1]) + int(height * 12. / 20.)
        else:
            range_y_min = int(obj_boxes[1]) + int(height * 12. / 30.)
            range_y_max = int(obj_boxes[1]) + int(height * 18. / 30.)

        return range_x_min, range_x_max, range_y_min, range_y_max


class Visualization(object):
    def __init(self):
        self.color = _GREEN
        self.thick = 1

    def vis_bbox_opencv(self, img, bbox):
        """Visualizes a bounding box."""
        (x0, y0, w, h) = bbox
        x1, y1 = int(x0 + w), int(y0 + h)
        x0, y0 = int(x0), int(y0)
        cv2.rectangle(img, (x0, y0), (x1, y1), self.color, thickness=self.thick)
        return img

    def Draw_connected_scene_graph(self, node_feature, relation, img_count, sg, idx, object_classes, predicate_classes,
                                   cnt_thres=2, view=True, save_path='./vis_result/', show=True):
        # load all of saved node_feature
        # if struct ids are same, updated to newly typed object
        # print('node_feature:',node_feature)
        tile_idx = []
        handle_idx = []
        # tomato_rgb = [255,99,71]
        tomato_rgb = [236, 93, 87]
        blue_rgb = [81, 167, 250]
        tomato_hex = webcolors.rgb_to_hex(tomato_rgb)
        blue_hex = webcolors.rgb_to_hex(blue_rgb)
        for node_num in range(len(node_feature)):
            if node_feature.iloc[node_num]['detection_cnt'] < cnt_thres:
                continue

            obj_cls = object_classes[int(node_feature.iloc[node_num]["class"])]
            node = node_feature.iloc[node_num]
            if (obj_cls == "tile"):
                tile_idx.append(str(node["idx"]))
            elif (obj_cls == "handle"):
                handle_idx.append(str(node["idx"]))
            else:
                sg.node('struct' + str(node["idx"]), shape='box', style='filled,rounded',
                        label=obj_cls + "_" + str(node["idx"]), margin='0.11, 0.0001', width='0.11', height='0',
                        fillcolor=tomato_hex, fontcolor='black')
                # sg.node('attribute_pose_' + str(node["idx"]), shape='box', style='filled, rounded',
                #         label="(" + str(round(float(node["3d_pose"][0]) / 5000.0, 1)) + "," +
                #               str(round(float(node["3d_pose"][1]) / 5000.0, 1)) + "," +
                #               str(round(float(node["3d_pose"][2]) / 5000.0, 1)) + ")",
                #         margin='0.11, 0.0001', width='0.11', height='0',
                #         fillcolor=blue_hex, fontcolor='black')
                # sg.edge('struct' + str(node["idx"]), 'attribute_pose_' + str(node["idx"]))

        tile_idx = set(tile_idx)
        tile_idx = list(tile_idx)
        handle_idx = set(handle_idx)
        handle_idx = list(handle_idx)

        relation_list = []
        for num in range(len(relation)):
            relation_list.append(
                (relation.iloc[num]["relation"][0], relation.iloc[num]["relation"][1],
                 relation.iloc[num]["relation"][2]))

        #relation_list = [rel for rel in relation_list if (
        #        node_feature.loc[node_feature['idx'] == int(rel[0])]['detection_cnt'].item() >= min(cnt_thres,
        #                                                                                            idx))]
        #relation_list = [rel for rel in relation_list if (
        #        node_feature.loc[node_feature['idx'] == int(rel[2])]['detection_cnt'].item() >= min(cnt_thres,
        #                                                                                            idx))]

        relation_set = set(relation_list)  # remove duplicate relations

        repeated_idx = []
        relation_array = np.array(list(relation_set))
        for i in range(len(relation_array)):
            for j in range(len(relation_array)):
                res = relation_array[i] == relation_array[j]
                if res[0] and res[2] and i != j:
                    repeated_idx.append(i)
        repeated_idx = set(repeated_idx)
        repeated_idx = list(repeated_idx)
        if len(repeated_idx) > 0:
            repeated = relation_array[repeated_idx]
            # print repeated.shape, repeated_idx
            for i, (pos, x, y) in enumerate(zip(repeated_idx, repeated[:, 0], repeated[:, 2])):
                position = np.where((x == repeated[:, 0]) & (y == repeated[:, 2]))[0]
                triplets = repeated[position].astype(int).tolist()
                preds = [t[1] for t in triplets]
                counted = Counter(preds)
                voted_pred = counted.most_common(1)
                # print(i, idx, triplets, voted_pred)
                relation_array[pos, 1] = voted_pred[0][0]

            relation_set = [tuple(rel) for rel in relation_array.astype(int).tolist()]
            relation_set = set(relation_set)
            # print(len(relation_set))

        # pale_rgb = [152,251,152]
        pale_rgb = [112, 191, 64]
        pale_hex = webcolors.rgb_to_hex(pale_rgb)
        for rel_num in range(len(relation_set)):
            rel = relation_set.pop()
            tile = False
            handle = False
            for t_i in tile_idx:
                if (str(rel[0]) == t_i or str(rel[2]) == t_i):
                    tile = True
            for h_i in handle_idx:
                if (str(rel[0]) == h_i or str(rel[2]) == h_i):
                    handle = True
            if ((not tile) and (not handle)):
                sg.node('rel' + str(rel_num), shape='box', style='filled, rounded', fillcolor=pale_hex,
                        fontcolor='black',
                        margin='0.11, 0.0001', width='0.11', height='0', label=str(predicate_classes[rel[1]]))
                sg.edge('struct' + str(rel[0]), 'rel' + str(rel_num))
                sg.edge('rel' + str(rel_num), 'struct' + str(rel[2]))

        if view and sg.format == 'pdf':
            sg.render(osp.join(save_path, 'scene_graph' + str(idx)), view=view)
        elif view and sg.format == 'png':
            sg.render(osp.join(save_path, 'scene_graph' + str(idx)), view=False)
            if show:
                img = cv2.imread(osp.join(save_path, 'scene_graph' + str(idx) + '.png'), cv2.IMREAD_COLOR)
                resize_x = 0.65
                resize_y = 0.9
                if img.shape[1] < int(1920 * resize_x) and img.shape[0] < int(1080 * resize_y):
                    pad = cv2.resize(img.copy(), (int(1920 * resize_x), int(1080 * resize_y)))
                    pad.fill(255)
                    pad[:img.shape[0], :img.shape[1], :] = img
                    resized = pad
                elif img.shape[1] < int(1920 * resize_x):
                    pad = cv2.resize(img.copy(), (int(1920 * resize_x), int(1080 * resize_y)))
                    pad.fill(255)
                    img = cv2.resize(img, (img.shape[1], int(1080 * resize_y)))
                    pad[:img.shape[0], :img.shape[1], :] = img
                    resized = pad
                elif img.shape[0] < int(1080 * resize_y):
                    pad = cv2.resize(img.copy(), (int(1920 * resize_x), int(1080 * resize_y)))
                    pad.fill(255)
                    img = cv2.resize(img, (int(1920 * resize_x), img.shape[0]))
                    pad[:img.shape[0], :img.shape[1], :] = img
                    resized = pad
                else:
                    resized = cv2.resize(img, (int(1920 * resize_x), int(1080 * resize_y)))
                cv2.imshow('3D Scene Graph', resized)
                # cv2.moveWindow('3D Scene Graph',1920,0)
                cv2.moveWindow('3D Scene Graph', 650, 0)
                cv2.waitKey(1)

        # node_feature.to_json(osp.join(save_path, 'json', 'scene_graph_node' + str(idx) + '.json'), orient='index')
        # relation.to_json(osp.join(save_path, 'json', 'scene_graph_relation' + str(idx) + '.json'), orient='index')
        # sg.clear()

    def vis_object_detection(self, image_scene, object_classes,
                             obj_inds, obj_boxes, obj_scores):

        for i, obj_ind in enumerate(obj_inds):
            cv2.rectangle(image_scene,
                          (int(obj_boxes[i][0]), int(obj_boxes[i][1])),
                          (int(obj_boxes[i][2]), int(obj_boxes[i][3])),
                          colorlist[int(obj_boxes[i][4])],
                          2)
            font_scale = 0.5
            txt = FOCC.get_class_string(obj_ind, obj_scores[i], object_classes)
            ((txt_w, txt_h), _) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            # Place text background.
            x0, y0 = int(obj_boxes[i][0]), int(obj_boxes[i][3])
            back_tl = x0, y0 - int(1.3 * txt_h)
            back_br = x0 + txt_w, y0
            cv2.rectangle(image_scene, back_tl, back_br, colorlist[int(obj_boxes[i][4])], -1)
            cv2.putText(image_scene,
                        txt,
                        (x0, y0 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        1)

        return image_scene


FOCC = FindObjectClassColor()
TFCO = CompareObjects()
