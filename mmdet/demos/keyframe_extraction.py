# ---------------------------------------------------------------
# keyframe_extraction.py
# Set-up time: 2020/11/3 23:08
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import numpy as np
import os
import os.path as osp
import cv2
import math
import time


def blurryness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def calculate_overlap(depth_org, rel_pose, intrinsic_org, pixel_coordinates, num_coordinate_samples=1000):
    """
        Description
            - Calculate overlap between two images based on projection
            - Projection of img2 to img1
                - p' = K * T_21 * depth * K^(-1) * p

        Parameter
            - depth: information on depth image
            - pose: relative pose to the reference image
            - intrinsic: camera intrinsic

        Return
            - amount_overlap: estimated amount of overlap in percentage

    """

    ## Pixel coordinates (p in the above eq.)
    intrinsic = np.copy(intrinsic_org)
    x_ratio = 0.1
    y_ratio = 0.1
    intrinsic[0] *= x_ratio
    intrinsic[1] *= y_ratio
    depth = cv2.resize(depth_org, None, fx=x_ratio, fy=y_ratio)
    height, width = depth.shape

    ## Calculate the amount of the overlapping area
    num_total = height * width
    num_overlap = 0

    for i in range(height):  # y-direction
        for j in range(width):  # x-direction
            temp = np.dot(np.linalg.inv(intrinsic), np.array([j, i, 1], dtype=float).reshape(3, 1))  # temp = (X, Y, 1)
            temp = (depth[i, j]) * temp  # temp = (X', Y', Z')
            temp = np.dot(rel_pose, np.append(temp, [1]).reshape(4, 1))
            temp = temp / float(temp[3] + 1e-10)
            temp = np.dot(intrinsic, temp[:3])
            temp = temp / float(temp[2] + 1e-10)
            x, y = int(temp[0]), int(temp[1])
            if x >= 0 and x < width and y >= 0 and y < height:
                num_overlap += 1

    overlapping_area = num_overlap / num_total

    return overlapping_area


def relative_pose(pose1, pose2):
    """
        Description
            - Calculate relative pose between a pair of poses
            - To avoid calculating matrix inverse, the calculation is based on
                - P_12 = [R_2^(-1) R_2^(-1)(t_1 - t_2); 0, 0, 0, 1],
                - where R_2^(-1) = R_2.T

        Parameter
            - pose1, pose2: 4 x 4 pose matrix

        Return
            - p_2_to_1 (relative_pose): estimated relative pose

    """
    p_2_to_1 = np.dot(np.linalg.inv(pose2), pose1)
    return p_2_to_1


class KeyFrameChecker(object):
    def __init__(self, args,
                 intrinsic_depth=None,
                 depth_shape=(480, 640),
                 num_coordinate_samples=1000,
                 BLURRY_REJECTION_ONLY=False):
        self.args = args
        self.frame_num = 0

        self.BLURRY_REJECTION_ONLY = BLURRY_REJECTION_ONLY

        # Blurry Image Rejection: Hyper parameters
        self.blurry_gain = args.gain
        self.blurry_offset = args.offset
        self.alpha = args.alpha

        # Key frame/ anchor frame Selection: Hyper parameters
        if not self.BLURRY_REJECTION_ONLY:
            self.intrinsic_depth = intrinsic_depth[:3, :3]
            self.thresh_key = args.thresh_key  # 0.2
            self.thresh_anchor = args.thresh_anchor  # 0.68
            self.max_group_len = args.max_group_len  # 10
            self.depth_shape = depth_shape  # (480, 640)
            pixel_coordinates = np.array(
                [[x, y, 1] for x in np.arange(depth_shape[0]) for y in np.arange(depth_shape[1])])
            self.pixel_coordinates = np.swapaxes(pixel_coordinates, 0, 1)
            self.key_frame_groups, self.curr_key_frame_group = [], [0]
            self.num_cooridnate_samples = num_coordinate_samples

    def check_frame(self, img, depth, pose):
        if self.args.disable_keyframe: return True, 0.0, 0.0
        if self.frame_num == 0:
            self.average_of_blurryness = blurryness(img)
            self.key_pose, self.anchor_pose, = [pose] * 2

        # 1. reject blurry images
        curr_blurry = blurryness(img)
        self.average_of_blurryness = self.alpha * self.average_of_blurryness + (1 - self.alpha) * curr_blurry
        threshold = self.blurry_gain * math.log(self.average_of_blurryness) + self.blurry_offset
        # threshold = self.blurry_gain * self.average_of_blurryness + self.blurry_offset
        if curr_blurry < threshold: return False, curr_blurry, threshold
        # if self.BLURRY_REJECTION_ONLY: return curr_blurry > threshold, curr_blurry, threshold

        # 2. calculate the relative pose to key & anchor frames
        rel_pose_to_key = relative_pose(pose, self.key_pose)
        rel_pose_to_anchor = relative_pose(pose, self.anchor_pose)

        # 3. calculate the ratio of the overlapping area
        depth = np.asarray(depth, dtype='uint16')

        overlap_with_key = calculate_overlap(depth, rel_pose_to_key, self.intrinsic_depth,
                                             self.pixel_coordinates, self.num_cooridnate_samples)
        overlap_with_anchor = calculate_overlap(depth, rel_pose_to_anchor, self.intrinsic_depth,
                                                self.pixel_coordinates, self.num_cooridnate_samples)

        # 4. update anchor and key frames
        if overlap_with_anchor < self.thresh_anchor:
            self.curr_key_frame_group.append(self.frame_num)
            self.anchor_pose = pose
            IS_ANCHOR = True
        else:
            IS_ANCHOR = False

        if overlap_with_key < self.thresh_key or len(self.curr_key_frame_group) > self.max_group_len:
            self.key_frame_groups.append(self.curr_key_frame_group)
            self.curr_key_frame_group = []
            self.key_pose, self.anchor_pose = [pose] * 2
            IS_KEY = True
        else:
            IS_KEY = False

        self.frame_num += 1

        return IS_KEY or IS_ANCHOR, curr_blurry, threshold
