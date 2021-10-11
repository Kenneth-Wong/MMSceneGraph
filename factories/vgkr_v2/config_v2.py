# ---------------------------------------------------------------
# config_v2.py
# Set-up time: 2020/12/6 21:57
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
RAW_V2_DIR = osp.join(RAW_DIR, 'v2')

"""Reading file"""
captions_vg_file = osp.join(RAW_PUBLIC_DIR, 'captions_vgOVcoco.json')
objects_file = osp.join(RAW_PUBLIC_DIR, 'objects.json')
attributes_file = osp.join(RAW_PUBLIC_DIR, 'attributes.json')
attribute_synset_file = osp.join(RAW_PUBLIC_DIR, 'attribute_synsets.json')
rels_file = osp.join(RAW_PUBLIC_DIR, 'relationships.json')
obj_alias_file = osp.join(RAW_PUBLIC_DIR, 'object_alias.txt')
pred_alias_file = osp.join(RAW_PUBLIC_DIR, 'relationship_alias.txt')
object_list_file = osp.join(RAW_PUBLIC_DIR, 'object_list.txt')
predicate_list_file = osp.join(RAW_PUBLIC_DIR, 'predicate_list.txt')
predicate_stem_file = osp.join(RAW_PUBLIC_DIR, 'predicate_stem.txt')
cap_to_sg_file_v1 = osp.join(RAW_PUBLIC_DIR, 'captions_to_sg_v1.json')
"""Writing """
cap_to_sg_file_v2 = osp.join(RAW_PUBLIC_DIR, 'captions_to_sg_v2.json')

coco_caption_entities_file = osp.join(RAW_PUBLIC_DIR, 'coco_entities_release.json')

"""Writing/Reading Public file"""
cleanse_objects_file = osp.join(RAW_V2_DIR, 'cleanse_objects.json')
cleanse_rels_file = osp.join(RAW_V2_DIR, 'cleanse_relationships.json')
cleanse_attrs_file = osp.join(RAW_V2_DIR, 'cleanse_attributes.json')

"""Writing V2 file"""
cleanse_triplet_match_file = osp.join(RAW_V2_DIR, 'cleanse_triplet_match.json')

"""Writing/Reading Public meta form"""
meta_form_file = osp.join(VG_DIR, 'meta_form.csv')

"""Writing/Reading V2 cap_sgentities_vgcoco file"""
cap_sgentities_vgcoco_file = osp.join(RAW_V2_DIR, 'cap_sgentities_vgcoco.json')

"""Out file from notebooks that is useful"""
info_from_objects_file = osp.join(RAW_V2_DIR, 'infoFromObjects.pickle')
padded_info_from_objects_file = osp.join(RAW_V2_DIR, 'padded_infoFromObjects.pickle')
info_from_rels_file = osp.join(RAW_V2_DIR, 'infoFromRels.pickle')
info_from_rels_match_seq_file = osp.join(RAW_V2_DIR, 'infoFromRels_match_and_seq.pickle')
padded_info_from_rels_match_seq_file = osp.join(RAW_V2_DIR, 'padded_infoFromRels_match_and_seq.pickle')
info_from_caps_file = osp.join(RAW_V2_DIR, 'infoFromCaps.pickle')
padded_info_from_caps_file = osp.join(RAW_V2_DIR, 'padded_infoFromCaps.pickle')

"""generalized VG ouput file"""
VG_GN_DIR = osp.join(DATA_DIR, 'visualgenomegn')
vggn_roidb_file = osp.join(VG_GN_DIR, 'VGGN-SGG.h5')
vggn_dict_file = osp.join(VG_GN_DIR, 'VGGN-SGG-dicts.json')
all_sentence_file = osp.join(VG_GN_DIR, 'VGGN-SGG-sentences.json')

# ====================== output VGKR roidb
VG_KR_DIR = osp.join(DATA_DIR, 'visualgenomekrV2')
vgkr_roidb_file = osp.join(VG_KR_DIR, 'VG200-SGG.h5')
vgkr_dict_file = osp.join(VG_KR_DIR, 'VG200-SGG-dicts.json')