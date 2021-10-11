# ---------------------------------------------------------------
# soc.py
# Set-up time: 2021/5/12 21:59
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import logging
import os
import os.path as osp
import tempfile
import mmcv
import numpy as np
from collections import defaultdict, Counter
import random
from torch.utils.data import Dataset
from mmdet.utils import print_log
from .registry import DATASETS
from .pipelines import Compose

import torch


@DATASETS.register_module
class SaliencyDataset(Dataset):
    def __init__(self, img_root, name_file, pipeline=None, target_root=None, test_mode=False, dataset_name=None):
        super(SaliencyDataset, self).__init__()
        self.img_prefix = img_root
        self.map_prefix = target_root if target_root is not None else img_root
        if isinstance(name_file, str):
            img_name_list = open(name_file).read().splitlines()
        else:
            img_name_list = name_file
        self.img_list = [{'filename': img_name + '.jpg'} for img_name in img_name_list]
        self.target_list = [{'saliency_map': img_name + '.png'} for img_name in img_name_list]

        self.pipeline = Compose(pipeline)

        self.test_mode = test_mode

        self.dataset_name = dataset_name

        if not self.test_mode:
            self._set_group_flag()

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.ones(len(self), dtype=np.uint8)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def __len__(self):
        return len(self.img_list)

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['map_prefix'] = self.map_prefix
        results['map_fields'] = []

    def prepare_train_img(self, idx):
        img_info = self.img_list[idx]
        ann_info = self.target_list[idx]
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_list[idx]
        # NOTE: For SGG, since the forward process may need gt_bboxes/gt_labels,
        # we should also load annotation as if in the training mode.
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
