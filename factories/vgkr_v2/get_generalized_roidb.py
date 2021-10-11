# ---------------------------------------------------------------
# get_generalized_roidb.py
# Set-up time: 2021/1/18 15:42
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
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




def build_vocab(ifrs_ms, ifcs, count_thr):
    # count up the number of words
    token_counter = Counter()
    print('Building vocab...')
    pbar = mmcv.ProgressBar(len(ifrs_ms))
    for ifr, ifc in zip(ifrs_ms, ifcs):
        tokens = ifr['tokens']
        keep_flags = ifr['keep_flags']
        for rel_id in range(len(ifr)):
            if keep_flags[rel_id]:
                token_counter.update(tokens[rel_id])

        if ifc is not None:
            tokens = ifc['tokens']
            keep_flags = ifc['keep_flags']
            for cap_id in range(len(ifc)):
                if keep_flags[cap_id]:
                    token_counter.update(tokens[cap_id])
        pbar.update()

    print('\n top words and their counts:')
    print(token_counter.most_common(20))

    # print some stats
    total_words = sum(token_counter.values())
    print('total words:', total_words)

    vocab = set()
    for token, count in token_counter.items():
        if count > count_thr:
            vocab.add(token)
    vocab = list(vocab)

    print('\n Keeping %d / %d tokens with enough instances' % (len(vocab), len(token_counter)))

    bad_words = [w for w, n in token_counter.items() if n <= count_thr]
    bad_count = sum(token_counter[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (
    len(bad_words), len(token_counter), len(bad_words) * 100.0 / len(token_counter)))
    print('number of words in vocab would be %d' % (len(vocab)))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special <UNK> token')
        vocab.append('<UNK>')

    return vocab


def build_vocab_dict(vocab):
    token_to_idx, idx_to_token = {}, {}
    next_idx = 1

    for token in vocab:
        token_to_idx[token] = next_idx
        idx_to_token[next_idx] = token
        next_idx = next_idx + 1

    return token_to_idx, idx_to_token


def split_caption_tokens(ifrs_ms, ifcs, max_rel_num, max_rel_token_len, max_cap_token_len):
    pbar = mmcv.ProgressBar(len(ifrs_ms))
    rel_captions_kept, captions_kept = 0, 0
    max_rel_captions_kept, min_rel_captions_kept, min_key_rel_captions_kept, max_key_rel_captions_kept = 0, 100, 100, 0
    for ifr, ifc in zip(ifrs_ms, ifcs):
        rel_seqs = list(ifr['seqs'])
        match = list(ifr['match'])

        times = []
        for rel_id in range(len(ifr)):
            times.append(len(match[rel_id]) if match[rel_id] is not None else 0)

        sorted_idxes = (np.argsort(times)[::-1]).tolist()
        ranks = [None] * len(times)
        for i, j in enumerate(sorted_idxes):
            ranks[j] = i

        num_key_seq, num_all_seq = len(np.where(np.array(times) > 0)[0]), len(rel_seqs)

        if num_key_seq > 0:
            max_key_rel_captions_kept = max(max_key_rel_captions_kept, num_key_seq)
            min_key_rel_captions_kept = min(min_key_rel_captions_kept, num_key_seq)

        rel_seq_tokens = []
        keep_flag = []
        for i, seq in enumerate(rel_seqs):
            tokens = seq.strip().split()
            if times[i] > 0:
                assert max_rel_token_len >= len(tokens), "adjust the max rel token len: now the token len is %d" % len(
                    tokens)
            if len(tokens) <= max_rel_token_len and ranks[i] < max_rel_num:
                rel_seq_tokens.append(tokens)
                rel_captions_kept += 1
                keep_flag.append(1)
            else:
                assert times[i] == 0
                rel_seq_tokens.append(None)
                keep_flag.append(0)

        max_rel_captions_kept = max(max_rel_captions_kept, len(rel_seq_tokens))
        min_rel_captions_kept = min(min_rel_captions_kept, len(rel_seq_tokens))

        ifr['tokens'] = rel_seq_tokens
        ifr['ipt_scores'] = times
        ifr['ipt_ranks'] = ranks
        ifr['keep_flags'] = keep_flag

        caption_seq_tokens = []
        cap_keep_flag = []
        if ifc is not None:
            captions = list(ifc['captions'])
            for cap in captions:
                tokens = cap.strip().split()
                if len(tokens) <= max_cap_token_len:
                    caption_seq_tokens.append(tokens)
                    captions_kept += 1
                    cap_keep_flag.append(1)
                else:
                    caption_seq_tokens.append(None)
                    cap_keep_flag.append(0)
            ifc['tokens'] = caption_seq_tokens
            ifc['keep_flags'] = cap_keep_flag

        pbar.update()

    print(
        '\n Keeping %d rel captions, max rel captions: %d,  min rel captions: %d, max key rel captions: %d,  min key rel captions: %d' %
        (rel_captions_kept, max_rel_captions_kept, min_rel_captions_kept, max_key_rel_captions_kept,
         min_key_rel_captions_kept))
    print('Keeping %d captions' % captions_kept)


def encode_objects_relationships_captions(meta_infos, ifos, ifrs_ms, ifcs, label_to_idx, attr_to_idx, token_to_idx,
                                          max_attr_num, max_length):
    num_data = len(ifos)
    assert len(ifos) == len(ifrs_ms)
    pbar = mmcv.ProgressBar(num_data)
    final_bboxes, final_labels, final_attrs, \
    final_rels, final_rel_ipts, final_rel_tgts, final_ipt_scores = [], [], [], [], [], [], []
    img_to_first_box, img_to_last_box, img_to_first_rel, img_to_last_rel = \
        np.zeros(num_data, dtype=np.int32), np.zeros(num_data, dtype=np.int32), \
        np.zeros(num_data, dtype=np.int32), np.zeros(num_data, dtype=np.int32)

    final_cap_ipts, final_cap_tgts = [], []
    img_to_first_cap, img_to_last_cap = np.zeros(num_data, dtype=np.int32), np.zeros(num_data, dtype=np.int32)

    all_sentences = []

    obj_counter = 0
    rel_idx_counter = 0
    no_rel_counter = 0
    obj_filtered = 0
    duplicate_filtered = 0
    cap_idx_counter = 0
    no_cap_counter_for_vg = 0
    no_cap_counter_for_filtered = 0
    for i, (vg_id, ifo, ifr, ifc) in enumerate(zip(list(meta_infos['meta_vgids']), ifos, ifrs_ms, ifcs)):
        bboxes, all_ids, object_ids, attrs, names = ifo['bboxes'], ifo['all_ids'], \
                                                    ifo['object_ids'], ifo['attrs'], ifo['names']
        # process the objects
        objid2dbidx = {}
        img_to_first_box[i] = obj_counter
        for bbox_idx in range(len(ifo)):
            if names[bbox_idx] in label_to_idx:
                final_labels.append(label_to_idx[names[bbox_idx]])
                final_bboxes.append(bboxes[bbox_idx])

                attr_list = []
                if attrs[bbox_idx] is not None:
                    for j, a in enumerate(list(set(attrs[bbox_idx]))):
                        if a in attr_to_idx:
                            attr_list.append(attr_to_idx[a])
                attr_list = attr_list[:max_attr_num]
                if len(attr_list) < max_attr_num:
                    attr_list += [0] * (max_attr_num - len(attr_list))
                final_attrs.append(attr_list)

                for cand_id in all_ids[bbox_idx]:
                    objid2dbidx[cand_id] = obj_counter

                obj_counter += 1

        if img_to_first_box[i] == obj_counter:
            img_to_first_box[i] = -1
            img_to_last_box[i] = -1
            no_rel_counter += 1
        else:
            img_to_last_box[i] = obj_counter - 1

        # process the relationships

        seqs, ipt_scores, ipt_ranks, keep_flags, subj_ids, obj_ids = ifr['seqs'], ifr['ipt_scores'], ifr['ipt_ranks'], \
                                                                     ifr['keep_flags'], ifr['subj_ids'], ifr['obj_ids']
        rel_sentences = []
        traverse_order = np.argsort(ipt_ranks)
        img_to_first_rel[i] = rel_idx_counter
        for rel_idx in traverse_order:
            if keep_flags[rel_idx]:
                if ipt_scores[rel_idx] > 0:
                    assert subj_ids[rel_idx] in objid2dbidx and obj_ids[rel_idx] in objid2dbidx and \
                           subj_ids[rel_idx] != obj_ids[rel_idx], \
                        'Please check %d image: key relation %d is not about the kept objects.' % (i, rel_idx)
                if subj_ids[rel_idx] not in objid2dbidx or obj_ids[rel_idx] not in objid2dbidx:
                    obj_filtered += 1
                    continue
                elif objid2dbidx[subj_ids[rel_idx]] == objid2dbidx[obj_ids[rel_idx]]:
                    duplicate_filtered += 1
                    continue
                else:
                    final_rels.append([objid2dbidx[subj_ids[rel_idx]], objid2dbidx[obj_ids[rel_idx]]])
                    final_ipt_scores.append(ipt_scores[rel_idx])
                    inp, tgt = encode_inp_tgt_caption(seqs[rel_idx].split(), max_length, token_to_idx)
                    final_rel_ipts.append(inp)
                    final_rel_tgts.append(tgt)
                    rel_sentences.append(seqs[rel_idx])

                    rel_idx_counter += 1
        if img_to_first_rel[i] == rel_idx_counter:
            img_to_first_rel[i] = -1
            img_to_last_rel[i] = -1
        else:
            img_to_last_rel[i] = rel_idx_counter - 1

        # process the captions
        cap_sentences = []
        if ifc is None:
            img_to_first_cap[i] = -1
            img_to_last_cap[i] = -1
            no_cap_counter_for_vg += 1
        else:
            cap_seqs, cap_tokens, cap_keep_flags = ifc['captions'], ifc['tokens'], ifc['keep_flags']
            img_to_first_cap[i] = cap_idx_counter
            for cap_idx in range(len(ifc)):
                if cap_keep_flags[cap_idx]:
                    inp, tgt = encode_inp_tgt_caption(cap_tokens[cap_idx], max_length, token_to_idx)
                    final_cap_ipts.append(inp)
                    final_cap_tgts.append(tgt)
                    cap_sentences.append(cap_seqs[cap_idx])

                    cap_idx_counter += 1

            if img_to_first_cap[i] == cap_idx_counter:
                img_to_first_cap[i] = -1
                img_to_last_cap[i] = -1
                no_cap_counter_for_filtered += 1
            else:
                img_to_last_cap[i] = cap_idx_counter - 1

        all_sentences.append(dict(dbidx=i, vg_id=vg_id, rel_sentences=rel_sentences, cap_sentences=cap_sentences))

        pbar.update()

    print('\n %i rel is filtered by object' % obj_filtered)
    print('%i rel is filtered by duplicate' % duplicate_filtered)
    print('%i rel remains ' % len(final_rels))
    print('%i out of %i valid images have relationships' % (num_data - no_rel_counter, num_data))

    print('%i cap remains ' % len(final_cap_ipts))
    print('%i out of %i valid images have cap, %i for not in coco, %i for filtered by length' % (
            num_data - no_cap_counter_for_vg, num_data, no_cap_counter_for_vg, no_cap_counter_for_filtered))

    return np.vstack(final_bboxes), np.vstack(final_labels), np.vstack(final_attrs), img_to_first_box, img_to_last_box, \
           np.vstack(final_rels), np.vstack(final_rel_ipts), np.vstack(final_rel_tgts), np.vstack(final_ipt_scores), \
           img_to_first_rel, img_to_last_rel, np.vstack(final_cap_ipts), np.vstack(final_cap_tgts), \
           img_to_first_cap, img_to_last_cap, all_sentences


def encode_inp_tgt_caption(seq, max_length, wtoi):
    L = np.zeros(max_length, dtype=np.int32)
    for k, w in enumerate(seq):
        if k < max_length:
            L[k] = wtoi.get(w, wtoi['<UNK>'])

    L_inp = np.hstack((np.zeros((1,)), L)).astype(np.int32)
    L_tgt = np.hstack((L, np.zeros((1,)))).astype(np.int32)

    first_zero_ind = np.where(L_tgt == 0)[0][0]
    L_tgt[first_zero_ind + 1:] = -1

    return L_inp, L_tgt


def getObjectAll(ifos):
    obj_vocabs = Counter()
    attr_vocabs = Counter()
    pbar = mmcv.ProgressBar(len(ifos))
    for i, ifo in enumerate(ifos):
        obj_vocabs.update(list(ifo['names']))
        all_attrs = []
        for attrs in ifo['attrs']:
            if attrs is not None:
                all_attrs += attrs
        all_attrs = list(set(all_attrs))
        attr_vocabs.update(all_attrs)
        pbar.update()
    return obj_vocabs, attr_vocabs


def getObjectMustIn(ifos, ifrs_match_seq):
    obj_vocabs_must_in = Counter()
    pbar = mmcv.ProgressBar(len(ifos))
    for i, (ifo, ifr_ms) in enumerate(zip(ifos, ifrs_match_seq)):
        ifo_names = dict(zip(list(ifo['object_ids']), list(ifo['names'])))
        cand_id_to_name = {}
        for cand_ids, object_id in zip(ifo['all_ids'], ifo['object_ids']):
            cand_ids = list(set(cand_ids))
            for cand_id in cand_ids:
                cand_id_to_name[cand_id] = ifo_names[object_id]
        for rel_id in range(len(ifr_ms)):
            if ifr_ms['match'][rel_id] is not None:
                obj_vocabs_must_in.update([cand_id_to_name[ifr_ms['subj_ids'][rel_id]]])
                obj_vocabs_must_in.update([cand_id_to_name[ifr_ms['obj_ids'][rel_id]]])
        pbar.update()
    return obj_vocabs_must_in


def main(params):
    print('Loading...')
    padded_infoFromCaps = mmcv.load(params['padded_cap_annots'])
    infoFromObjects = mmcv.load(params['obj_annots'])
    padded_infoFromObjects = mmcv.load(params['padded_obj_annots'])
    infoFromRels_match_and_seq = mmcv.load(params['rel_annots'])
    padded_infoFromRels_match_and_seq = mmcv.load(params['padded_rel_annots'])
    meta_infos = pd.read_csv(meta_form_file, low_memory=False)

    seed(123)  # make reproducible

    # step 1: get the object / attr vocabs for pretraining the detection models
    print('get Objects must in...')
    obj_vocabs_must_in = getObjectMustIn(infoFromObjects, infoFromRels_match_and_seq)  # about 2760
    # obj_vocabs, attr_vocabs = getObjectAll(infoFromObjects)  # objs from 51,208 images,  31,352 objs, 24,727 attrs
    print('get all objects and attrs...')
    all_obj_vocabs, all_attr_vocabs = getObjectAll(
        padded_infoFromObjects)  # objs from 108,073 images, 50,967 objs, 41,185 attrs

    # enlarge the obj_vocabs
    if params['max_obj_vocab'] > len(obj_vocabs_must_in):
        total_adds = params['max_obj_vocab'] - len(obj_vocabs_must_in)
        num_add = 0
        add_list = []
        for item in all_obj_vocabs.most_common():
            if item[0] not in obj_vocabs_must_in:
                add_list.append(item)
                num_add += 1
            if num_add >= total_adds:
                break
        obj_vocabs_must_in += Counter(dict(add_list))
    obj_vocabs = obj_vocabs_must_in

    attr_vocabs = all_attr_vocabs
    if params['max_attr_vocab'] < len(all_attr_vocabs):
        attr_vocabs = Counter(dict(all_attr_vocabs.most_common(params['max_attr_vocab'])))

    # create the vocab: 1~3000, 1-index
    idx_to_label = {i + 1: w for i, w in enumerate(obj_vocabs)}  # a 1-indexed vocab translation table
    label_to_idx = {w: i + 1 for i, w in enumerate(obj_vocabs)}  # inverse table

    # create the attr vocab: 1~800, 1-index
    idx_to_attribute = {i + 1: w for i, w in enumerate(attr_vocabs)}
    attribute_to_idx = {w: i + 1 for i, w in enumerate(attr_vocabs)}

    # step 2: create the language vocabs
    ### step 2.1: create the tokens, ranks, rank scores, keep flags for triplet sentences for 108073 images,
    # and 2w images with key rels
    # also, for the 5w images with full captions.
    split_caption_tokens(padded_infoFromRels_match_and_seq, padded_infoFromCaps, params['max_rel_num'],
                         params['max_length'], params['max_length'])

    ### step 2.2: build the token vocabs for triplet sentences and captions
    vocabs = build_vocab(padded_infoFromRels_match_and_seq, padded_infoFromCaps, params['word_count_threshold'])

    ### step 2.3: build vocab dict
    token_to_idx, idx_to_token = build_vocab_dict(vocabs)

    # step 3: write the imdb
    ### step 3.1: encode
    f = h5py.File(params['h5_output'], 'w')

    bboxes, labels, attrs, img_to_first_box, img_to_last_box, \
    rels, rel_ipts, rel_tgts, rel_ipt_scores, img_to_first_rel, img_to_last_rel, \
    cap_ipts, cap_tgts, img_to_first_cap, img_to_last_cap, all_sentences = encode_objects_relationships_captions(
        meta_infos, padded_infoFromObjects, padded_infoFromRels_match_and_seq, padded_infoFromCaps,
        label_to_idx, attribute_to_idx, token_to_idx, params['max_attr_num'], params['max_length'])

    f.create_dataset('labels', data=labels)
    f.create_dataset('bboxes', data=bboxes)
    f.create_dataset('attributes', data=attrs)
    f.create_dataset('img_to_first_box', data=img_to_first_box)
    f.create_dataset('img_to_last_box', data=img_to_last_box)
    f.create_dataset('relationships', data=rels)
    f.create_dataset('rel_inputs', data=rel_ipts)
    f.create_dataset('rel_targets', data=rel_tgts)
    f.create_dataset('rel_ipt_scores', data=rel_ipt_scores)
    f.create_dataset('img_to_first_rel', data=img_to_first_rel)
    f.create_dataset('img_to_last_rel', data=img_to_last_rel)
    f.create_dataset('cap_inputs', data=cap_ipts)
    f.create_dataset('cap_targets', data=cap_tgts)
    f.create_dataset('img_to_first_cap', data=img_to_first_cap)
    f.create_dataset('img_to_last_cap', data=img_to_last_cap)

    ### step 3.2: write the dict
    mmcv.dump(dict(label_to_idx=label_to_idx, idx_to_label=idx_to_label,
                   attribute_to_idx=attribute_to_idx, idx_to_attribute=idx_to_attribute,
                   token_to_idx=token_to_idx, idx_to_token=idx_to_token), vggn_dict_file)

    mmcv.dump(all_sentences, all_sentence_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--obj_annots', default=info_from_objects_file)
    parser.add_argument('--padded_obj_annots', default=padded_info_from_objects_file)
    parser.add_argument('--rel_annots', default=info_from_rels_match_seq_file)
    parser.add_argument('--padded_rel_annots', default=padded_info_from_rels_match_seq_file)
    parser.add_argument('--cap_annots', default=info_from_caps_file)
    parser.add_argument('--padded_cap_annots', default=padded_info_from_caps_file)
    parser.add_argument('--meta_info', default=meta_form_file)

    parser.add_argument('--max_obj_vocab', default=3000, type=int)
    parser.add_argument('--max_attr_vocab', default=800, type=int)
    # options
    parser.add_argument('--max_length', default=17, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--max_rel_num', default=50, type=int)
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--max_attr_num', default=10, type=int)
    # output database
    parser.add_argument('--h5_output', default=vggn_roidb_file)
    parser.add_argument('--dict_output', default=vggn_dict_file)

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
