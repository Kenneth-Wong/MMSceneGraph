# ---------------------------------------------------------------
# triplet_match.py
# Set-up time: 2020/12/6 21:29
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

from factories.vgkr_v1.config_v1 import *
from factories.utils.tools import vecDist, cleaneddb2vg
from factories.utils.word_embeddings import load_word_vectors, obj_edge_vectors
import os
import os.path as osp
import torch
import json
import numpy as np
from nltk.corpus import wordnet as wn
import sys
import string
from factories.utils.tools import make_alias_dict
import mmcv

def preprocess_labels(label, alias_dict={}):

    label = sentence_preprocess(label)
    if label in alias_dict:
        label = alias_dict[label]
    return label

def sentence_preprocess(phrase):
    """ preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
    replacements = {
      '½': 'half',
      '—' : '-',
      '™': '',
      '¢': 'cent',
      'ç': 'c',
      'û': 'u',
      'é': 'e',
      '°': ' degree',
      'è': 'e',
      '…': '',
    }
    table1 = str.maketrans("", "", string.punctuation)
    table2 = str.maketrans("", "", "0123456789")
    #phrase = phrase.encode('utf-8')
    phrase = phrase.lstrip(' ').rstrip(' ')
    for k, v in replacements.items():
        phrase = phrase.replace(k, v)
    return str(phrase).lower().translate(table1).translate(table2)

def load_CapSG_vectors(wv_dict, wv_arr, wv_size, object_alias={}, pred_alias={}, vg_idxes=None, file=cap_to_sg_file):
    print('Loading relations from captions....')
    with open(file, 'r') as f:
        cap_sg_dict = json.load(f)

    vectors_dict = {}
    if vg_idxes is None:
        vg_idxes = list(cap_sg_dict.keys())

    for cnt, vg_idx in enumerate(vg_idxes):
        assert vg_idx in cap_sg_dict
        sg_list = cap_sg_dict[vg_idx]
        num_triplets = sum([len(sg['edge']) for sg in sg_list])
        cumsum = np.cumsum([len(sg['edge']) for sg in sg_list])
        vectors = torch.Tensor(num_triplets, wv_size * 3)
        rels = []
        for i, sg in enumerate(sg_list):
            edges =  sg['edge']
            for j, edge in enumerate(edges):
                sub, rel, obj = edge[0].lower().split('-')[0], \
                                edge[1].lower().split('-')[0], \
                                edge[2].lower().split('-')[0]
                sub = preprocess_labels(sub, object_alias)
                rel = preprocess_labels(rel, pred_alias)
                obj = preprocess_labels(obj, object_alias)
                vecs = obj_edge_vectors([sub, rel, obj], wv_dict, wv_arr, wv_size)  # 3, dim
                vecs = vecs.view(1, -1)
                if i == 0:
                    vectors[j, :] = vecs
                else:
                    vectors[cumsum[i-1] + j, :] = vecs
                rels.append((sub, rel, obj, i))  # use i to mark the rel belongs to which captions

        vectors_dict[int(vg_idx)] = (vectors, rels)
        sys.stdout.write('Loaded: {:d}/{:d}   \r'.format(cnt, len(vg_idxes)))
        sys.stdout.flush()
    print('\n')
    return vectors_dict


def load_SG_vectors(wv_dict, wv_arr, wv_size, dbidxes=None, file=cleanse_rels_file):
    print('Loading relations from annotations....')
    with open(file, 'r') as f:
        sgs = json.load(f)

    vectors_dict = {}
    if dbidxes is None:
        dbidxes = list(range(len(sgs)))
    for cnt, sgidx in enumerate(dbidxes):
        sg = sgs[sgidx]
        vg_idx = sg['image_id']
        relationships = sg['relationships']
        vectors = torch.Tensor(len(relationships), wv_size*3)
        rels = []
        for j, rel_item in enumerate(relationships):
            subj = rel_item['subject']
            obj = rel_item['object']
            predicate = rel_item['predicate']
            #triplet = rel_item['relation']
            #sub, rel, obj = triplet[0].lower(), triplet[1].lower(), triplet[2].lower()
            if 'names' in subj:
                sub_name = subj['names'][0]
            else:
                sub_name = subj['name']
            if 'names' in obj:
                obj_name = obj['names'][0]
            else:
                obj_name = obj['name']
            vecs = obj_edge_vectors([sub_name, predicate, obj_name], wv_dict, wv_arr, wv_size)
            vecs = vecs.view(1, -1)
            vectors[j, :] = vecs
            rels.append((sub_name, predicate, obj_name))

        vectors_dict[vg_idx] = (vectors, rels)
        sys.stdout.write('Loaded: {:d}/{:d}   \r'.format(cnt, len(dbidxes)))
        sys.stdout.flush()
    print('\n')
    selected_sgs = [sgs[sgidx] for sgidx in dbidxes]
    return selected_sgs, vectors_dict


def wordnet_preprocess(vocab_A, vocab_B, pos=wn.NOUN):
    base_vocab_A = wn.morphy(vocab_A)
    base_vocab_B = wn.morphy(vocab_B)
    if base_vocab_A is None or base_vocab_B is None:
        return False
    synsets_A = wn.synsets(base_vocab_A, pos)
    synsets_B = wn.synsets(base_vocab_B, pos)

    # justify whether two synsets overlap with each other
    for s_a in synsets_A:
        for s_b in synsets_B:
            if s_a == s_b or len(list(set(s_a.lowest_common_hypernyms(s_b)).intersection(set([s_a, s_b])))) > 0 :
                return True
    return False

def triplet_match(wv_dir=GLOVE_DIR, wv_type='glove.6B', wv_dim=300, include_rel=True):
    obj_alias_dict, _ = make_alias_dict(obj_alias_file)
    pred_alias_dict, _ = make_alias_dict(pred_alias_file)
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)
    dbidx2vgidx, vgidx2dbidx = cleaneddb2vg(meta_file) #db2vg()
    cap_sg_vectors = load_CapSG_vectors(wv_dict, wv_arr, wv_size, obj_alias_dict, pred_alias_dict)
    vg_idxes = cap_sg_vectors.keys()
    sgs, sg_vectors = load_SG_vectors(wv_dict, wv_arr, wv_size, [vgidx2dbidx[vg_idx] for vg_idx in vg_idxes])

    print('Matching...')
    #write_data_cache = []
    #num = len(vg_idxes)

    for sgdb_idx, vg_idx in enumerate(vg_idxes):
        ori_vectors, ori_rels = sg_vectors[vg_idx]
        cap_vectors, cap_rels = cap_sg_vectors[vg_idx]

        if len(ori_vectors) == 0 or len(cap_vectors) == 0:
            continue

        ori_rel_refcounter = {i:[] for i in range(len(ori_rels))}

        # firstly we use the wordnet to align the caption triplets with annotated triplets
        match_pairs = {i:[] for i in range(len(cap_rels))}
        for cap_rel_idx, cap_triplet in enumerate(cap_rels):
            c_sub, c_rel, c_obj, cap_idx = cap_triplet
            for ori_idx, ori_triplet in enumerate(ori_rels):
                o_sub, o_rel, o_obj = ori_triplet
                # shot
                if (c_sub == o_sub and c_obj == o_obj) or (wordnet_preprocess(c_sub, o_sub) and wordnet_preprocess(c_obj, o_obj)):
                    match_pairs[cap_rel_idx].append(ori_idx)
                    if cap_idx not in ori_rel_refcounter[ori_idx]:
                        ori_rel_refcounter[ori_idx].append(cap_idx)
        ori_rel_refcounter = {k:len(v) for k, v in ori_rel_refcounter.items()}

        # now extract the cap triplet indexes which has not been matched
        not_matched = []
        for i in sorted(list(match_pairs.keys())):
            if len(match_pairs[i]) == 0:
                not_matched.append(i)
        if len(not_matched) == 0:
            ori_rel_refcounter_by_dist = None
            assert sgs[sgdb_idx]['image_id'] == vg_idx
            sgs[sgdb_idx]['cap_ref'] = ori_rel_refcounter
            sgs[sgdb_idx]['cap_ref_by_dist'] = ori_rel_refcounter_by_dist
            continue

        not_matched_dict = {i:not_matched[i] for i in range(len(not_matched))}
        cap_vectors = cap_vectors[not_matched, :]
        if not include_rel:
            ori_vectors = torch.cat((ori_vectors[:, :300], ori_vectors[:, 600:]), 1)
            cap_vectors = torch.cat((cap_vectors[:, :300], cap_vectors[:, 600:]), 1)

        # compute distance
        distance = vecDist(cap_vectors, ori_vectors)
        _, indices = torch.sort(distance, 1)

        #for j, rel in enumerate([cap_rels[k] for k in not_matched]):
        #    print(rel, '-----', ori_rels[indices[j, 0]])

        match_by_dist_pairs = {not_matched_dict[i]: int(indices[i, 0]) for i in range(len(not_matched))}
        ori_rel_refcounter_by_dist = {}
        for k, v in match_by_dist_pairs.items():
            if v not in ori_rel_refcounter_by_dist:
                ori_rel_refcounter_by_dist[v] = 1
            else:
                ori_rel_refcounter_by_dist[v] += 1

        assert sgs[sgdb_idx]['image_id'] == vg_idx
        sgs[sgdb_idx]['cap_ref'] = ori_rel_refcounter
        sgs[sgdb_idx]['cap_ref_by_dist'] = ori_rel_refcounter_by_dist

        sys.stdout.write('Loaded: {:d}/{:d}   \r'.format(sgdb_idx, len(vg_idxes)))
        sys.stdout.flush()

    mmcv.dump(sgs, cleanse_triplet_match_file)



triplet_match(include_rel=False)


