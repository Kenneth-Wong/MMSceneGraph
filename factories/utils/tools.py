# ---------------------------------------------------------------
# tools.py
# Set-up time: 2020/12/6 21:33
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import json
import os.path as osp
import torch
import mmcv
from nltk.corpus import wordnet as wn


def load_vg_iminfos(meta_file, IMAGE_DIR1, IMAGE_DIR2):
    """
    :return: a list containing im tuple (path, vg_id, and coco_id)
    """
    image_meta = json.load(open(meta_file))
    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    image_infos = []
    for i in image_meta:
        basename = str(i['image_id']) + '.jpg'
        if basename in corrupted_ims:
            continue
        if osp.isfile(osp.join(IMAGE_DIR1, basename)):
            im_path = osp.join(IMAGE_DIR1, basename)
        else:
            im_path = osp.join(IMAGE_DIR2, basename)
        image_infos.append((im_path, i['image_id'], i['coco_id']))

    assert len(image_infos) == 108073
    return image_infos


def vecDist(vec1, vec2):
    v1sum = torch.sum(vec1 ** 2, 1, keepdim=True)
    v2sum = torch.transpose(torch.sum(vec2 ** 2, 1, keepdim=True), 0, 1)
    product = torch.matmul(vec1, torch.transpose(vec2, 0, 1))
    dist = v1sum + v2sum - 2 * product
    return dist


def db2vg(meta_file):
    """transfer corrupted-not-removed db indices (108077) to vg indexes"""
    dbidx2vgidx = {}
    image_meta = json.load(open(meta_file))
    for i, item in enumerate(image_meta):
        dbidx2vgidx[i] = item['image_id']
    vgidx2dbidx = {dbidx2vgidx[i]: i for i in dbidx2vgidx}
    return dbidx2vgidx, vgidx2dbidx


def cleaneddb2vg(meta_file):
    """transfer corrupted-removed db indices (108073) to vg indexes"""
    dbidx2vgidx = {}
    image_meta = json.load(open(meta_file))
    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    keep = []
    for i, item in enumerate(image_meta):
        basename = str(item['image_id']) + '.jpg'
        if basename in corrupted_ims:
            continue
        keep.append(item)
    for i, item in enumerate(keep):
        dbidx2vgidx[i] = item['image_id']
    vgidx2dbidx = {dbidx2vgidx[i]: i for i in dbidx2vgidx}
    return dbidx2vgidx, vgidx2dbidx


def make_alias_dict(dict_file):
    """create an alias dictionary from a file"""
    out_dict = {}
    vocab = []
    for line in open(dict_file, 'r'):
        alias = line.strip('\n').strip('\r').split(',')
        alias_target = alias[0] if alias[0] not in out_dict else out_dict[alias[0]]
        for a in alias:
            out_dict[a] = alias_target  # use the first term as the aliasing target
        vocab.append(alias_target)
    return out_dict, vocab


def make_alias_dict_from_synset(dict_file):
    out_dict = mmcv.load(dict_file)
    for key, v in out_dict.items():
        out_dict[key] = wn.synset(v).lemma_names()[0]
    return out_dict