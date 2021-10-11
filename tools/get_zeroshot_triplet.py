# ---------------------------------------------------------------
# get_zeroshot_triplet.py
# Set-up time: 2021/4/12 19:29
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
from mmcv import Config
from mmdet.datasets import build_dataset
import numpy as np
import torch

def get_triplets(dataset):
    rels, classes = dataset.relationships, dataset.gt_classes
    triplets = []
    for rel, cls in zip(rels, classes):
        for r in rel:
            t = [cls[r[0]], cls[r[1]], r[2]]
            if t not in triplets:
                triplets.append(t)
    return triplets


cfg = Config.fromfile('configs/visualgenome_kr/VGCOCO_PredCls_heth_faster_rcnn_x101_64x4d_fpn_1x.py')

dataset = build_dataset(cfg.data.train)

test_dataset = build_dataset(cfg.data.test)

train_triplets = get_triplets(dataset)
test_triplets = get_triplets(test_dataset)
print(len(train_triplets))
print(len(test_triplets))

zeroshot_triplets = []
for t in test_triplets:
    if t not in train_triplets:
        zeroshot_triplets.append(t)
zeroshot_triplets = np.array(zeroshot_triplets, dtype=np.int64)
torch.save(torch.from_numpy(zeroshot_triplets), 'data/visualgenomekr_evaluation/zeroshot_triplet.pytorch')

