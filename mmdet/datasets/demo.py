# ---------------------------------------------------------------
# demo.py
# Set-up time: 2020/11/3 16:30
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import os.path as osp
import os
import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmdet.core import eval_map, eval_recalls
from .pipelines import Compose
from .registry import DATASETS
from PIL import Image
import cv2


@DATASETS.register_module
class DemoDataset(Dataset):
    """Demo dataset for loading images and other auxiliary information.
        We provide some probable image-wise axuiliary information interfaces, including depth,
        intrinsic matrix, pose matrix, etc.
        """

    def __init__(self,
                 pipeline,
                 data_root=None,
                 image_file=None,
                 img_prefix=None,
                 depth_prefix=None,
                 intrinsic_prefix=None,
                 pose_prefix=None,
                 start_id=0,
                 end_id=-1):
        self.data_root = data_root
        self.image_file = image_file
        self.img_prefix = img_prefix
        self.depth_prefix = depth_prefix  # each frame
        self.intrinsic_prefix = intrinsic_prefix  # each video
        self.pose_prefix = pose_prefix  # each frame
        self.start_id = start_id
        self.end_id = end_id

        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.depth_prefix is None or osp.isabs(self.depth_prefix)):
                self.depth_prefix = osp.join(self.data_root, self.depth_prefix)
            if not (self.intrinsic_prefix is None or osp.isabs(self.intrinsic_prefix)):
                self.intrinsic_prefix = osp.join(self.data_root, self.intrinsic_prefix)
            if not (self.pose_prefix is None or osp.isabs(self.pose_prefix)):
                self.pose_prefix = osp.join(self.data_root, self.pose_prefix)

            if not (self.image_file is None or osp.isabs(self.image_prefix)):
                self.image_file = osp.join(self.data_root, self.image_file)

        if self.image_file is not None:  # temporarily not utilized
            pass
        else:
            all_img_names = sorted(os.listdir(self.img_prefix), key=lambda x: int(x[:-4]))
            self.img_ids = [int(x[:-4]) for x in all_img_names][self.start_id:self.end_id]
            self.img_infos = [dict(image_id=int(x[:-4]), id=int(x[:-4]), file_name=x, filename=x) for x in
                              all_img_names][self.start_id:self.end_id]

        if self.intrinsic_prefix is not None:
            self.intrinsic_color = open(osp.join(self.intrinsic_prefix, 'intrinsic_color.txt')).read()
            self.intrinsic_color = np.array([item.split() for item in self.intrinsic_color.split('\n')[:-1]],
                                            dtype=np.float32)
            self.intrinsic_depth = open(osp.join(self.intrinsic_prefix, 'intrinsic_depth.txt')).read()
            self.intrinsic_depth = np.array([item.split() for item in self.intrinsic_depth.split('\n')[:-1]],
                                            dtype=np.float32)

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        return self.prepare_img(idx)

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix

    def prepare_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def load_image(self, filename):
        """
        Open this API for single image checking: Whether the image should be fed into the model
        """

        img_path = osp.join(self.img_prefix, filename)
        if self.pose_prefix is not None:
            camera_pose = open(osp.join(self.pose_prefix, filename[:-4] + '.txt')).read()
            camera_pose = [item.split() for item in camera_pose.split('\n')[:-1]]
            camera_pose = np.array(camera_pose, dtype=np.float32)
            R = np.array([camera_pose[0][0:3], camera_pose[1][0:3], camera_pose[2][0:3]], dtype=np.float32)
            inv_R = np.linalg.inv(R)
            Trans = np.array([camera_pose[0][3], camera_pose[1][3], camera_pose[2][3]], dtype=np.float32)
        else:
            inv_R, Trans, camera_pose = None, None, None

        if self.depth_prefix is not None:
            depth_img = Image.open(osp.join(self.depth_prefix, filename[:-4] + '.png'))
            # Preprocess loaded camera parameter and depth info
            depth_pix = depth_img.load()
            pix_depth = []
            for ii in range(depth_img.size[1]):
                pix_row = []
                for jj in range(depth_img.size[0]):
                    pix_row.append(depth_pix[jj, ii])
                pix_depth.append(pix_row)
        else:
            depth_img, pix_depth = None, None

        if self.intrinsic_prefix is not None:
            p_matrix = [self.intrinsic_color[0][:], self.intrinsic_color[1][:], self.intrinsic_color[2][:]]
            p_matrix = np.array(p_matrix, dtype=np.float32)
            inv_p_matrix = np.linalg.pinv(p_matrix)
        else:
            inv_p_matrix = None
        img_scene = cv2.imread(img_path)
        return img_scene, depth_img, pix_depth, inv_p_matrix, inv_R, Trans, camera_pose
