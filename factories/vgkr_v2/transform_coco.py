# ---------------------------------------------------------------
# transform_coco.py
# Set-up time: 2021/2/26 9:19
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

"""Transform the annotations of captions to coco pattern"""

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

def transform(ori_file, tgt_img_ids):
    ori_annots = mmcv.load(ori_file)

    annotations, images = [], []
    caption_id = 0
    for annot in ori_annots:
        if annot['vg_id'] in tgt_img_ids:
            for sent in annot['cap_sentences']:
                annotations.append(dict(image_id=annot['vg_id'], id=caption_id, caption=sent))
                caption_id += 1
            images.append(dict(id=annot['vg_id']))
    coco_annots = dict(images=images, annotations=annotations)
    return coco_annots

