# ---------------------------------------------------------------
# config_v1.py
# Set-up time: 2020/12/6 18:32
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import os.path as osp

DATA_DIR = osp.join('./', 'data')
VG_DIR = osp.join(DATA_DIR, 'visualgenome')
IMAGE_ROOT_DIR = osp.join(VG_DIR, 'Images')
IMAGE_DIR1 = osp.join(IMAGE_ROOT_DIR, 'VG_100K')
IMAGE_DIR2 = osp.join(IMAGE_ROOT_DIR, 'VG_100K_2')
imdb_file = osp.join(VG_DIR, 'imdb_1024.h5')
imdb512_file = osp.join(VG_DIR, 'imdb_512.h5')
roidb_file = osp.join(VG_DIR, 'VG-SGG-with-attri.h5')
dict_file = osp.join(VG_DIR, 'VG-SGG-dicts-with-attri.json')
meta_file = osp.join(VG_DIR, 'recsize_image_data.json')
sal_file = osp.join(VG_DIR, 'saliency_512.h5')


GLOVE_DIR = osp.join(DATA_DIR, 'glove')

RAW_DIR = osp.join(DATA_DIR, 'visualgenomekr_ingredients')
RAW_PUBLIC_DIR = osp.join(RAW_DIR, 'public_inputs')
RAW_V1_DIR = osp.join(RAW_DIR, 'v1')

#merge_rel_file = osp.join(RAW_DIR, 'merge_relationships.json')
"""Reading file"""
captions_vg_file = osp.join(RAW_PUBLIC_DIR, 'captions_vgOVcoco.json')
objects_file = osp.join(RAW_PUBLIC_DIR, 'objects.json')
rels_file = osp.join(RAW_PUBLIC_DIR, 'relationships.json')
obj_alias_file = osp.join(RAW_PUBLIC_DIR, 'object_alias.txt')
pred_alias_file = osp.join(RAW_PUBLIC_DIR, 'relationship_alias.txt')
object_list_file = osp.join(RAW_PUBLIC_DIR, 'object_list.txt')
predicate_list_file = osp.join(RAW_PUBLIC_DIR, 'predicate_list.txt')
predicate_stem_file = osp.join(RAW_PUBLIC_DIR, 'predicate_stem.txt')
cap_to_sg_file = osp.join(RAW_PUBLIC_DIR, 'captions_to_sg_v1.json')

"""Writing Public file"""
cleanse_objects_file = osp.join(RAW_V1_DIR, 'cleanse_objects.json')
cleanse_rels_file = osp.join(RAW_V1_DIR, 'cleanse_relationships.json')

"""Writing V1 file"""

cleanse_triplet_match_file = osp.join(RAW_V1_DIR, 'cleanse_triplet_match.json')
#triplet_match_file = osp.join(RAW_V1_DIR, 'triplet_match.json')


# ====================== output VGKR roidb
VG_KR_DIR = osp.join(DATA_DIR, 'visualgenomekr')
vgkr_roidb_file = osp.join(VG_KR_DIR, 'VG200-SGG.h5')
vgkr_dict_file = osp.join(VG_KR_DIR, 'VG200-SGG-dicts.json')