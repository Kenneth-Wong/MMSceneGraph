# ---------------------------------------------------------------
# gradcheck.py
# Set-up time: 2020/4/16 下午9:57
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import os.path as osp
import sys

import numpy as np
import torch
from torch.autograd import gradcheck


  # noqa: E402, isort:skip
from mmdet.ops.shape_aware_roi_align import ShapeAwareRoIAlign

feat_size = 15
spatial_scale = 1.0 / 8
img_size = feat_size / spatial_scale
num_imgs = 2
num_rois = 20

batch_ind = np.random.randint(num_imgs, size=(num_rois, 1))
rois = (np.random.rand(num_rois, 4) * img_size * 0.5).astype(np.int32)
rois[:, 2:] += int(img_size * 0.5)
rois = np.hstack((batch_ind, rois))

roi_heights = rois[:, 4] - rois[:, 2] + 1
roi_widths = rois[:, 3] - rois[:, 1] + 1
masks = []
for i in range(num_rois):
    masks.append(torch.from_numpy(np.random.rand(roi_heights[i], roi_widths[i])).float().cuda())

feat = torch.randn(
    num_imgs, 16, feat_size, feat_size, requires_grad=True, device='cuda:0')
rois = torch.from_numpy(rois).float().cuda()
inputs = (feat, rois, masks)
print('Gradcheck for roi align...')
test = gradcheck(ShapeAwareRoIAlign(3, spatial_scale), inputs, atol=1e-3, eps=1e-3)
print(test)
test = gradcheck(ShapeAwareRoIAlign(3, spatial_scale, 2), inputs, atol=1e-3, eps=1e-3)
print(test)
