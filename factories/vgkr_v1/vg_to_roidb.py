# ---------------------------------------------------------------
# vg_to_roidb.py
# Set-up time: 2020/12/6 21:43
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
# coding=utf8
# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# --------------------------------------------------------

import argparse, json, string
from collections import Counter
import math

from math import floor
import h5py as h5
import numpy as np
import pprint
from factories.vgkr_v1.config_v1 import *
from factories.utils.tools import cleaneddb2vg, db2vg
import mmcv

import nltk.stem.porter as pt
import nltk.stem.lancaster as lc
import nltk.stem.snowball as sb
import nltk.stem as ns

pt_stemmer = pt.PorterStemmer()
lc_stemmer = lc.LancasterStemmer()
sb_stemmer = sb.SnowballStemmer("english")
lemmatizer = ns.WordNetLemmatizer()

"""
A script for generating an hdf5 ROIDB from the VisualGenome dataset
"""

def map_pred_to_list(pred, pred_list, pred_stem_list):
    # split the pred
    lw_tokens = sorted(pred.split(' '), key=lambda x: len(x), reverse=True)
    # try different stemmer
    for lw_token in lw_tokens:
        lm_stem = lemmatizer.lemmatize(lw_token, pos='v')
        pt_stem = pt_stemmer.stem(lw_token)
        lc_stem = lc_stemmer.stem(lw_token)
        sb_stem = sb_stemmer.stem(lw_token)
        token_stems = [lm_stem, pt_stem, lc_stem, sb_stem]
        for token_stem in token_stems:
            for idx, stem in enumerate(pred_stem_list):
                if token_stem == stem:
                    return {pred: pred_list[idx]}
    return {}

def map_object_to_list(obj, obj_list):
    lw_token = sorted(obj.split(' '), key=lambda x: len(x), reverse=True)[0]
    stem = lemmatizer.lemmatize(lw_token, pos='n')
    for o in obj_list:
        if stem == o:
            return {obj: o}
    return {}



def extract_object_token(data, num_tokens, obj_list=[], verbose=True):
    """ Builds a set that contains the object names. Filters infrequent tokens. """
    token_counter = Counter()
    for img in data:
        for region in img['objects']:
            for name in region['names']:
                if not obj_list or name in obj_list:
                    token_counter.update([name])
    tokens = set()
    # pick top N tokens
    token_counter_return = {}
    for token, count in token_counter.most_common():
        tokens.add(token)
        token_counter_return[token] = count
        if len(tokens) == num_tokens:
            break
    if verbose:
        print(('Keeping %d / %d objects'
               % (len(tokens), len(token_counter))))
    return tokens, token_counter_return


def extract_predicate_token(data, num_tokens, pred_list=[], obj_list=[], pred_stem_list=[], old_to_new_alias={}, pred_old_to_new_alias={}, verbose=True):
    """ Builds a set that contains the relationship predicates. Filters infrequent tokens. """
    token_counter = Counter()
    total = 0
    total_key_rel = 0
    has_key_rel_imgs = 0
    all_key_rel_imgs = 0
    filter_rel_by_predicate_counter = Counter()
    filter_subj_counter = Counter()
    filter_obj_counter = Counter()
    for img in data:
        key_rel_idxes = []
        if 'cap_ref' in img:
            cap_ref = img['cap_ref']
            for i in cap_ref:
                if cap_ref[i] > 0:
                    key_rel_idxes.append(int(i))
        if len(key_rel_idxes):
            all_key_rel_imgs += 1
        keep_key = []
        for idx, relation in enumerate(img['relationships']):
            predicate = relation['predicate']
            if not pred_list or predicate in pred_list:
                token_counter.update([predicate])
            else:  ## not in pred_list
                if predicate in pred_old_to_new_alias:
                    predicate = pred_old_to_new_alias[predicate]
                else:
                    pred_stem_dict = map_pred_to_list(predicate, pred_list, pred_stem_list)
                    predicate = pred_stem_dict.get(predicate, predicate)
                    pred_old_to_new_alias.update(pred_stem_dict)
                if idx in key_rel_idxes and predicate not in pred_list:
                    filter_rel_by_predicate_counter.update([predicate])
            # consider objects
            if idx in key_rel_idxes:
                subj, obj = relation['subject'], relation['object']
                # subj_name, obj_name = subj['names'][0], obj['names'][0]
                if 'names' in subj:
                    subj_name = subj['names'][0]
                else:
                    subj_name = subj['name']
                if 'names' in obj:
                    obj_name = obj['names'][0]
                else:
                    obj_name = obj['name']
                # try best to align the name to the existing lists
                if subj_name not in obj_list:
                    if subj_name in old_to_new_alias:
                        subj_name = old_to_new_alias[subj_name]
                    else:
                        object_stem_dict = map_object_to_list(subj_name, obj_list)
                        subj_name = object_stem_dict.get(subj_name, subj_name)
                        old_to_new_alias.update(object_stem_dict)
                if obj_name not in obj_list:
                    if obj_name in old_to_new_alias:
                        obj_name = old_to_new_alias[obj_name]
                    else:
                        object_stem_dict = map_object_to_list(obj_name, obj_list)
                        obj_name = object_stem_dict.get(obj_name, obj_name)
                        old_to_new_alias.update(object_stem_dict)

                if predicate in pred_list:
                    if subj_name not in obj_list:
                        filter_subj_counter.update([subj_name])
                    elif obj_name not in obj_list:
                        filter_obj_counter.update([obj_name])
                if predicate in pred_list and subj_name in obj_list and obj_name in obj_list:
                    total_key_rel += 1
                    keep_key.append(idx)
            total += 1
        if len(keep_key):
            has_key_rel_imgs += 1
    all_filter = filter_subj_counter + filter_obj_counter
    with open('filter_by_objects.txt', 'w') as f:
        for token, count in all_filter.most_common():
            f.write(token + ' ' + str(count) + '\n')
    with open('filter_by_preds.txt', 'w') as f:
        for token, count in filter_rel_by_predicate_counter.most_common():
            f.write(token + ' ' + str(count) + '\n')
    tokens = set()
    token_counter_return = {}
    for token, count in token_counter.most_common():
        tokens.add(token)
        token_counter_return[token] = count
        if len(tokens) == num_tokens:
            break
    if verbose:
        print(('Keeping %d / %d predicates with enough instances'
               % (len(tokens), len(token_counter))))
    return tokens, token_counter_return


def build_token_dict(vocab):
    """ build bi-directional mapping between index and token"""
    token_to_idx, idx_to_token = {}, {}
    next_idx = 1
    vocab_sorted = sorted(list(vocab))  # make sure it's the same order everytime
    for token in vocab_sorted:
        token_to_idx[token] = next_idx
        idx_to_token[next_idx] = token
        next_idx = next_idx + 1

    return token_to_idx, idx_to_token


def encode_box(region, org_h, org_w, im_long_size):
    x = region['x']
    y = region['y']
    w = region['w']
    h = region['h']
    scale = float(im_long_size) / max(org_h, org_w)
    image_size = im_long_size
    # recall: x,y are 1-indexed
    x, y = math.floor(scale * (region['x'] - 1)), math.floor(scale * (region['y'] - 1))
    w, h = math.ceil(scale * region['w']), math.ceil(scale * region['h'])

    # clamp to image
    if x < 0: x = 0
    if y < 0: y = 0

    # box should be at least 2 by 2
    if x > image_size - 2:
        x = image_size - 2
    if y > image_size - 2:
        y = image_size - 2
    if x + w >= image_size:
        w = image_size - x
    if y + h >= image_size:
        h = image_size - y

    # also convert to center-coord oriented
    box = np.asarray([x + floor(w / 2), y + floor(h / 2), w, h], dtype=np.int32)
    assert box[2] > 0  # width height should be positive numbers
    assert box[3] > 0
    return box


def encode_objects(obj_data, token_to_idx, token_counter, org_h, org_w, im_long_sizes, object_alias):
    encoded_labels = []
    encoded_boxes = {}
    for size in im_long_sizes:
        encoded_boxes[size] = []
    im_to_first_obj = np.zeros(len(obj_data), dtype=np.int32)
    im_to_last_obj = np.zeros(len(obj_data), dtype=np.int32)
    obj_counter = 0

    for i, img in enumerate(obj_data):
        im_to_first_obj[i] = obj_counter
        img['id_to_idx'] = {}  # object id to region idx
        for obj in img['objects']:
            # pick a label for the object
            max_occur = 0
            obj_label = None
            for name in obj['names']:
                # pick the name that has maximum occurance
                # try to alias
                if name not in token_to_idx:
                    name = object_alias.get(name, name)
                if name in token_to_idx and token_counter[name] > max_occur:
                    obj_label = name
                    max_occur = token_counter[obj_label]

            if obj_label is not None:
                # encode region
                for size in im_long_sizes:
                    encoded_boxes[size].append(encode_box(obj, org_h[i], org_w[i], size))

                encoded_labels.append(token_to_idx[obj_label])

                for obj_id in obj['ids']:  # assign same index for merged ids
                    img['id_to_idx'][obj_id] = obj_counter

                obj_counter += 1

        if im_to_first_obj[i] == obj_counter:
            im_to_first_obj[i] = -1
            im_to_last_obj[i] = -1
        else:
            im_to_last_obj[i] = obj_counter - 1

    for k, boxes in encoded_boxes.items():
        encoded_boxes[k] = np.vstack(boxes)
    return np.vstack(encoded_labels), encoded_boxes, im_to_first_obj, im_to_last_obj


def encode_relationship(sub_id, obj_id, id_to_idx):
    # builds a tuple of the index of object and subject in the object list
    sub_idx = id_to_idx[sub_id]
    obj_idx = id_to_idx[obj_id]
    return np.asarray([sub_idx, obj_idx], dtype=np.int32)


def encode_relationships(rel_data, token_to_idx, obj_data, pred_alias):
    """MUST BE CALLED AFTER encode_objects!!!"""
    encoded_pred = []  # encoded predicates
    encoded_rel = []  # encoded relationship tuple
    im_to_first_rel = np.zeros(len(rel_data), dtype=np.int32)
    im_to_last_rel = np.zeros(len(rel_data), dtype=np.int32)
    rel_idx_counter = 0

    encoded_key_rel_index = []
    im_to_first_key_index = np.zeros(len(rel_data), dtype=np.int32)
    im_to_last_key_index = np.zeros(len(rel_data), dtype=np.int32)
    key_idx_counter = 0

    no_rel_counter = 0
    obj_filtered = 0
    predicate_filtered = 0
    duplicate_filtered = 0
    no_key_rel_counter = 0
    for i, img in enumerate(rel_data):
        im_to_first_rel[i] = rel_idx_counter
        im_to_first_key_index[i] = key_idx_counter
        id_to_idx = obj_data[i]['id_to_idx']  # object id to object list idx
        cap_ref = img['cap_ref'] if 'cap_ref' in img else None
        key_idxes = []
        if cap_ref:
            for k, v in cap_ref.items():
                if v > 0:
                    key_idxes.append(int(k))
        for relidx, relation in enumerate(img['relationships']):
            subj = relation['subject']
            obj = relation['object']
            predicate = relation['predicate']
            if predicate not in token_to_idx:
                predicate = pred_alias.get(predicate, predicate)
            if subj['object_id'] not in id_to_idx or obj['object_id'] not in id_to_idx:
                obj_filtered += 1
                continue
            elif predicate not in token_to_idx:
                predicate_filtered += 1
                continue
            elif id_to_idx[subj['object_id']] == id_to_idx[obj['object_id']]:  # sub and obj can't be the same box
                duplicate_filtered += 1
                continue
            else:
                encoded_pred.append(token_to_idx[predicate])
                encoded_rel.append(
                    encode_relationship(subj['object_id'],
                                        obj['object_id'],
                                        id_to_idx
                                        ))

                if relidx in key_idxes:
                    encoded_key_rel_index.append(rel_idx_counter)
                    key_idx_counter += 1

                rel_idx_counter += 1  # accumulate counter

        if im_to_first_rel[i] == rel_idx_counter:
            # if no qualifying relationship
            im_to_first_rel[i] = -1
            im_to_last_rel[i] = -1
            no_rel_counter += 1
        else:
            im_to_last_rel[i] = rel_idx_counter - 1
        if im_to_first_key_index[i] == key_idx_counter:
            im_to_first_key_index[i] = -1
            im_to_last_key_index[i] = -1
            no_key_rel_counter += 1
        else:
            im_to_last_key_index[i] = key_idx_counter - 1

    print('%i rel is filtered by object' % obj_filtered)
    print('%i rel is filtered by predicate' % predicate_filtered)
    print('%i rel is filtered by duplicate' % duplicate_filtered)
    print('%i rel remains ' % len(encoded_pred))
    print('%i key rel remains ' % len(encoded_key_rel_index))

    print('%i out of %i valid images have relationships' % (len(rel_data) - no_rel_counter, len(rel_data)))
    print('%i out of %i valid images have key relationships' % (len(rel_data) - no_key_rel_counter, len(rel_data)))
    return np.vstack(encoded_pred), np.vstack(encoded_rel), np.vstack(encoded_key_rel_index),\
           im_to_first_rel, im_to_last_rel, im_to_first_key_index, im_to_last_key_index


def encode_splits(obj_data, im_to_first_key_index, opt=None):
    if opt is not None:
        test_begin_idx = opt['test_begin_idx']
        keyrel_test_begin_num = opt['keyrel_test_begin_num']
    split = np.zeros(len(obj_data), dtype=np.int32)
    keyrel_split = np.zeros(len(obj_data), dtype=np.int32)
    keyrel_counter = 0
    for i, info in enumerate(obj_data):
        splitix = 0

        if opt is None:  # use encode from input file
            s = info['split']
            if s == 'test': splitix = 2
        else:  # use portion split
            if i >= test_begin_idx: splitix = 2
        split[i] = splitix

        keysplitix = -1
        if im_to_first_key_index[i] > -1: # has key idx
            # make sure that keyrel train/val split is the subset of train/test split
            if i >= test_begin_idx or keyrel_counter >= keyrel_test_begin_num:
                keysplitix = 2
            else:
                keysplitix = 0
            keyrel_counter += 1
        keyrel_split[i] = keysplitix

    if opt is not None and opt['shuffle']:
        np.random.shuffle(split)

    print(('assigned %d/%d/%d to train/val/test split' % (np.sum(split == 0), np.sum(split == 1), np.sum(split == 2))))
    print(('assigned %d/%d/%d to train/val/test keyrel_split' % (np.sum(keyrel_split == 0), np.sum(keyrel_split == 1), np.sum(keyrel_split == 2))))
    return split, keyrel_split


def make_list(list_file):
    """create a blacklist list from a file"""
    return [line.strip('\n').strip('\r') for line in open(list_file)]


def filter_by_idx(data, valid_list):
    return [data[i] for i in valid_list]


def main(args):
    print('start')
    pprint.pprint(args)

    obj_list = []
    if len(args.object_list) > 0:
        print('using object list from %s' % (args.object_list))
        obj_list = make_list(args.object_list)
        assert (len(obj_list) >= args.num_objects)

    pred_list = []
    if len(args.pred_list) > 0:
        print('using predicate list from %s' % (args.pred_list))
        pred_list = make_list(args.pred_list)
        assert (len(pred_list) >= args.num_predicates)

    pred_stem = []
    if len(args.pred_stem) > 0:
        print('using predicate stem from %s' % (args.pred_stem))
        pred_stem = make_list(args.pred_stem)
        assert (len(pred_stem) >= args.num_predicates)

    # read in the annotation data
    print('loading json files..')
    _, cleaned_vgidx2dbidx = cleaneddb2vg(args.meta_file)
    _, vgidx2dbidx = db2vg(args.meta_file)

    obj_data = json.load(open(args.object_input))

    rel_data = json.load(open(args.relationship_input))
    img_data = json.load(open(args.metadata_input))
    sel_dbidx = [cleaned_vgidx2dbidx[rel_item['image_id']] for rel_item in rel_data]
    obj_data = [obj_data[cleaned_vgidx2dbidx[rel_item['image_id']]] for rel_item in rel_data]
    img_data = [img_data[vgidx2dbidx[rel_item['image_id']]] for rel_item in rel_data]

    # check
    for i in range(len(rel_data)):
        assert rel_data[i]['image_id'] == obj_data[i]['image_id']
        assert rel_data[i]['image_id'] == obj_data[i]['image_id']

    print('read image db from %s' % args.imdb)
    imdb = h5.File(args.imdb, 'r')
    num_im, _, _, _ = imdb['images'].shape
    img_long_sizes = [512, 1024]
    #valid_im_idx = imdb['valid_idx'][:]  # valid image indices
    #img_ids = imdb['image_ids'][:]
    #img_data = filter_by_idx(img_data, valid_im_idx)

    # may only load a fraction of the data
    if args.load_frac < 1:
        num_im = int(num_im * args.load_frac)
        obj_data = obj_data[:num_im]
        rel_data = rel_data[:num_im]
    print('processing %i images' % num_im)

    heights, widths = imdb['original_heights'][:][sel_dbidx], imdb['original_widths'][:][sel_dbidx]

    old_to_new_alias = {'tracks': 'track',
                        'baby': 'child',
                        'bicycle': 'bike',
                        'skis': 'ski',
                        'cellphone': 'phone',
                        'waves': 'wave',
                        'flowers': 'flower',
                        'bananas': 'banana',
                        'lines': 'line',
                        'doughnut': 'donut',
                        'doughnuts': 'donut',
                        'trees': 'tree',
                        'animals': 'animal',
                        'jet': 'airplane',
                        'tv': 'television',
                        }
    pred_old_to_new_alias = {'crossing': 'across',
                             'talking on': 'says',
                             'talking to': 'says',
                             'leaning on': 'against',
                             'leaning against': 'against',
                             'filled with': 'full of',
                             'contains': 'full of',
                             'containing': 'full of',
                             'in middle of': 'between',
                             'underneath': 'under',
                             'taking': 'carrying',
                             'close to': 'near',
                             'going down': 'walking on',
                             'beneath': 'under',
                             'atop': 'on',
                             'next': 'near',
                             'doing': 'working on',
                             'jumping on': 'on',
                             'jumping': 'on',
                             'on top': 'on',
                             'surrounded by': 'between',
                             }

    predicate_tokens, predicate_token_counter = extract_predicate_token(rel_data,
                                                                        args.num_predicates,
                                                                        pred_list, obj_list, pred_stem,
                                                                        old_to_new_alias=old_to_new_alias,
                                                                        pred_old_to_new_alias=pred_old_to_new_alias)
    predicate_to_idx, idx_to_predicate = build_token_dict(predicate_tokens)

    # build vocabulary
    object_tokens, object_token_counter = extract_object_token(obj_data, args.num_objects,
                                                               obj_list)

    label_to_idx, idx_to_label = build_token_dict(object_tokens)

    # print out vocabulary
    print('objects: ')
    print(object_token_counter)
    print('relationships: ')
    print(predicate_token_counter)

    # write the h5 file
    f = h5.File(args.h5_file, 'w')

    # encode object
    encoded_label, encoded_boxes, im_to_first_obj, im_to_last_obj = \
        encode_objects(obj_data, label_to_idx, object_token_counter, \
                       heights, widths, img_long_sizes, old_to_new_alias)

    f.create_dataset('labels', data=encoded_label)
    for k, boxes in encoded_boxes.items():
        f.create_dataset('boxes_%i' % k, data=boxes)
    f.create_dataset('img_to_first_box', data=im_to_first_obj)
    f.create_dataset('img_to_last_box', data=im_to_last_obj)

    encoded_predicate, encoded_rel, encoded_key_idx, im_to_first_rel, im_to_last_rel, im_to_first_key, im_to_last_key = \
        encode_relationships(rel_data, predicate_to_idx, obj_data, pred_old_to_new_alias)

    f.create_dataset('predicates', data=encoded_predicate)
    f.create_dataset('relationships', data=encoded_rel)
    f.create_dataset('key_relationship_idxes', data=encoded_key_idx)
    f.create_dataset('img_to_first_rel', data=im_to_first_rel)
    f.create_dataset('img_to_last_rel', data=im_to_last_rel)
    f.create_dataset('img_to_first_keyrel', data=im_to_first_key)
    f.create_dataset('img_to_last_keyrel', data=im_to_last_key)

    # build train/val/test splits

    print('num objects = %i' % encoded_label.shape[0])
    print('num relationships = %i' % encoded_predicate.shape[0])

    opt = None
    if not args.use_input_split:
        opt = {}
        opt['val_begin_idx'] = int(len(obj_data) * args.train_frac)
        opt['test_begin_idx'] = int(len(obj_data) * args.val_frac)
        opt['keyrel_test_begin_num'] = int(np.sum(im_to_first_key != -1) * args.val_frac)
        opt['shuffle'] = args.shuffle
    split, keyrel_split = encode_splits(obj_data, im_to_first_key, opt)

    if split is not None:
        f.create_dataset('split', data=split)  # 2 = test, 0 = train
    if keyrel_split is not None:
        f.create_dataset('keyrel_split', data=keyrel_split)

    # and write the additional json file
    json_struct = {
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'predicate_to_idx': predicate_to_idx,
        'idx_to_predicate': idx_to_predicate,
        'predicate_count': predicate_token_counter,
        'object_count': object_token_counter,
        'subset_dbidx': sel_dbidx,
        'vgidx2dbidx': cleaned_vgidx2dbidx
    }

    mmcv.dump(json_struct, args.json_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imdb', default=imdb_file, type=str)
    parser.add_argument('--object_input', default=cleanse_objects_file, type=str)
    parser.add_argument('--relationship_input', default=cleanse_triplet_match_file, type=str)
    parser.add_argument('--metadata_input', default=meta_file, type=str)
    parser.add_argument('--object_alias', default=obj_alias_file, type=str)
    parser.add_argument('--pred_alias', default=pred_alias_file, type=str)
    parser.add_argument('--object_list',
                        default=object_list_file,
                        type=str)
    parser.add_argument('--pred_list',
                        default=predicate_list_file,
                        type=str)
    parser.add_argument('--pred_stem',
                        default=predicate_stem_file,
                        type=str)
    parser.add_argument('--num_objects', default=200, type=int, help="set to 0 to disable filtering")
    parser.add_argument('--num_predicates', default=80, type=int, help="set to 0 to disable filtering")
    parser.add_argument('--min_box_area_frac', default=0.002, type=float)
    parser.add_argument('--json_file', default=vgkr_dict_file)
    parser.add_argument('--h5_file', default=vgkr_roidb_file)
    parser.add_argument('--load_frac', default=1, type=float)
    parser.add_argument('--use_input_split', default=False, type=bool)
    parser.add_argument('--train_frac', default=0.7, type=float)
    parser.add_argument('--val_frac', default=0.7, type=float)
    parser.add_argument('--shuffle', default=False, type=bool)

    args = parser.parse_args()
    main(args)
