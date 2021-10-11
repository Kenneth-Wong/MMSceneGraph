# ---------------------------------------------------------------
# transform_captions_to_sg.py
# Set-up time: 2020/12/6 22:11
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import mmcv

from factories.utils import sng_parser
from factories.vgkr_v2.config_v2 import *

captions = mmcv.load(captions_vg_file)

captions_sg = {}
bar = mmcv.ProgressBar(len(captions))
for coco_id, caption_list in captions.items():
    captions_sg[coco_id] = []
    for caption in caption_list:
        graph = sng_parser.parse(caption)
        node = [n['lemma_head'] for n in graph['entities']]
        edge = [[node[r['subject']], r['lemma_relation'], node[r['object']]] for r in graph['relations']]
        captions_sg[coco_id].append({'node': node, 'edge': edge, 'caption': caption})
    bar.update()

mmcv.dump(captions_sg, cap_to_sg_file)



