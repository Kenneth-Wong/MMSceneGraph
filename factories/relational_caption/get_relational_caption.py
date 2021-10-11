# ---------------------------------------------------------------
# get_relational_caption.py
# Set-up time: 2021/1/11 16:37
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------


import argparse, os, json, string
import os.path as osp
from collections import Counter
import pandas as pd
import mmcv
from threading import Thread, Lock
from math import floor
import h5py
import numpy as np
from nltk.corpus import wordnet as wn



def build_vocab(data, min_token_instances, verbose=True):
    """ Builds a set that contains the vocab. Filters infrequent tokens. """
    token_counter = Counter()
    print('Building vocab...')
    pbar = mmcv.ProgressBar(len(data))
    for img in data:
        for region in img['relationships']:
            if region['tokens'] is not None:
                token_counter.update(region['tokens'])
        pbar.update()

    vocab = set()
    for token, count in token_counter.items():
        if count >= min_token_instances:
            vocab.add(token)

    if verbose:
        print('\n Keeping %d / %d tokens with enough instances'
              % (len(vocab), len(token_counter)))

    if len(vocab) < len(token_counter):
        vocab.add('<UNK>')
        if verbose:
            print('adding special <UNK> token.')
    else:
        if verbose:
            print('no <UNK> token needed.')
    print('VOCAB num: %s'%(len(vocab)))
    return vocab


def build_vocab_dict(vocab):
    token_to_idx, idx_to_token = {}, {}
    next_idx = 1

    for token in vocab:
        token_to_idx[token] = next_idx
        idx_to_token[next_idx] = token
        next_idx = next_idx + 1

    return token_to_idx, idx_to_token


def encode_caption(tokens, token_to_idx, max_token_length):
    encoded = np.zeros(max_token_length, dtype=np.int32)
    for i, token in enumerate(tokens):
        if i == max_token_length:
            return encoded
        if token in token_to_idx:
            encoded[i] = token_to_idx[token]
        else:
            encoded[i] = token_to_idx['<UNK>']

    return encoded


def encode_captions(data, token_to_idx, max_token_length):
    encoded_list = []
    parts_list = []
    lengths = []
    for img in data:
        for region in img['relationships']:
            tokens = region['tokens']
            if tokens is None: continue
            tokens_encoded = encode_caption(tokens, token_to_idx, max_token_length)
            encoded_list.append(tokens_encoded)
            parts_list.append(encode_caption(region['parts'], {1: 1, 2: 2, 3: 3}, max_token_length))
            lengths.append(len(tokens))
    return np.vstack(encoded_list), np.vstack(parts_list), np.asarray(lengths, dtype=np.int32)


def encode_boxes(data, original_heights, original_widths):
    all_boxes = []
    all_names = []
    xwasbad = 0
    ywasbad = 0
    wwasbad = 0
    hwasbad = 0
    print('Building boxes...')
    pbar = mmcv.ProgressBar(len(data))
    for i, img in enumerate(data):
        H, W = original_heights[i], original_widths[i]
        for region in img['relationships']:
            if region['tokens'] is None:
                continue
            # recall: x,y are 1-indexed

            # ----------------------------------------------subject-----------------------------------------------------
            x, y = round((region['subject']['x'] - 1) + 1), round((region['subject']['y'] - 1) + 1)
            w, h = round(region['subject']['w']), round(region['subject']['h'])
            if 'synsets' in region['subject'] and len(region['subject']['synsets']) > 0:
                subject_name = wn.synset(region['subject']['synsets'][0]).lemma_names()[0]
            else:
                subject_name = region['subject']['name']
            region['subject']['synset_name'] = subject_name
            all_names.append(subject_name)
            # clamp to image
            if x < 1: x = 1
            if y < 1: y = 1
            if x > W - 1:
                x = W - 1
                xwasbad += 1
            if y > H - 1:
                y = H - 1
                ywasbad += 1
            if x + w > W:
                w = W - x
                wwasbad += 1
            if y + h > H:
                h = H - y
                hwasbad += 1

            box = np.asarray([x + floor(w / 2), y + floor(h / 2), w, h],
                             dtype=np.int32)  # also convert to center-coord oriented
            assert box[2] >= 0  # width height should be positive numbers
            assert box[3] >= 0
            all_boxes.append(box)

            # ------------------------------------------------object--------------------------------------------------------
            x, y = round((region['object']['x'] - 1) + 1), round((region['object']['y'] - 1) + 1)
            w, h = round(region['object']['w']), round(region['object']['h'])
            if 'synsets' in region['object'] and len(region['object']['synsets']) > 0:
                object_name = wn.synset(region['object']['synsets'][0]).lemma_names()[0]
            else:
                object_name = region['object']['name']
            region['object']['synset_name'] = object_name
            all_names.append(object_name)
            # clamp to image
            if x < 1: x = 1
            if y < 1: y = 1
            if x > W - 1:
                x = W - 1
                xwasbad += 1
            if y > H - 1:
                y = H - 1
                ywasbad += 1
            if x + w > W:
                w = W - x
                wwasbad += 1
            if y + h > H:
                h = H - y
                hwasbad += 1

            box = np.asarray([x + floor(w / 2), y + floor(h / 2), w, h],
                             dtype=np.int32)  # also convert to center-coord oriented
            assert box[2] >= 0  # width height should be positive numbers
            assert box[3] >= 0
            all_boxes.append(box)
        pbar.update()
    print('\n number of bad x,y,w,h: ', xwasbad, ywasbad, wwasbad, hwasbad)

    return np.vstack(all_boxes), all_names


def build_img_idx_to_box_idxs(data):
    box_idx = 0
    num_images = len(data)
    img_to_first_box = np.zeros(num_images, dtype=np.int32)
    img_to_last_box = np.zeros(num_images, dtype=np.int32)
    for i, img in enumerate(data):
        start_counter = box_idx
        end_counter = box_idx
        for region in img['relationships']:
            if region['tokens'] is None:
                continue
            end_counter += 2
        if start_counter != end_counter:
            img_to_first_box[i] = start_counter
            img_to_last_box[i] = end_counter - 1 # the last box, inclusive
            box_idx = end_counter
        else:
            img_to_first_box[i] = -1
            img_to_last_box[i] = -1

    return img_to_first_box, img_to_last_box


def build_filename_dict(data):
    # First make sure all filenames
    filenames_list = ['%d.jpg' % img['image_id'] for img in data]
    assert len(filenames_list) == len(set(filenames_list))

    next_idx = 0
    filename_to_idx, idx_to_filename = {}, {}
    for img in data:
        filename = '%d.jpg' % img['image_id']
        filename_to_idx[filename] = next_idx
        idx_to_filename[next_idx] = filename
        next_idx += 1
    return filename_to_idx, idx_to_filename


def encode_filenames(data, filename_to_idx):
    filename_idxs = []
    for img in data:
        filename = '%d.jpg' % img['image_id']
        idx = filename_to_idx[filename]
        for region in img['relationships']:
            if region['tokens'] is None: continue
            filename_idxs.append(idx)

    return np.asarray(filename_idxs, dtype=np.int32)


def build_entity_dict(entity_list):
    entity_list = list(set(entity_list))
    label_to_idx = {e: i+1 for i, e in enumerate(entity_list)}
    return label_to_idx

def encode_labels(data, label_to_idx):
    labels = []
    for img in data:
        for region in img['relationships']:
            if region['tokens'] is None: continue
            label = label_to_idx[region['subject']['synset_name']]
            label = label_to_idx[region['object']['synset_name']]
            labels.append(label)
    return np.asarray(labels, dtype=np.int32)[:, None]



# def add_images(data, h5_file, args):
#     num_images = len(data)
#
#     shape = (num_images, 3, args.image_size, args.image_size)
#     image_dset = h5_file.create_dataset('images', shape, dtype=np.uint8)
#     original_heights = np.zeros(num_images, dtype=np.int32)
#     original_widths = np.zeros(num_images, dtype=np.int32)
#     image_heights = np.zeros(num_images, dtype=np.int32)
#     image_widths = np.zeros(num_images, dtype=np.int32)
#
#     lock = Lock()
#     q = Queue()
#
#     for i, img in enumerate(data):
#         filename = os.path.join(args.image_dir, '%s.jpg' % img['image_id'])
#         q.put((i, filename))
#
#     def worker():
#         while True:
#             i, filename = q.get()
#             img = imread(filename)
#             # handle grayscale
#             if img.ndim == 2:
#                 img = img[:, :, None][:, :, [0, 0, 0]]
#             H0, W0 = img.shape[0], img.shape[1]
#             img = imresize(img, float(args.image_size) / max(H0, W0))
#             H, W = img.shape[0], img.shape[1]
#             # swap rgb to bgr. Is this the best way?
#             r = img[:, :, 0].copy()
#             img[:, :, 0] = img[:, :, 2]
#             img[:, :, 2] = r
#
#             lock.acquire()
#             if i % 1000 == 0:
#                 print('Writing image %d / %d' % (i, len(data)))
#             original_heights[i] = H0
#             original_widths[i] = W0
#             image_heights[i] = H
#             image_widths[i] = W
#             image_dset[i, :, :H, :W] = img.transpose(2, 0, 1)
#             lock.release()
#             q.task_done()
#
#     print('adding images to hdf5.... (this might take a while)')
#     for i in range(args.num_workers):
#         t = Thread(target=worker)
#         t.daemon = True
#         t.start()
#     q.join()
#
#     h5_file.create_dataset('image_heights', data=image_heights)
#     h5_file.create_dataset('image_widths', data=image_widths)
#     h5_file.create_dataset('original_heights', data=original_heights)
#     h5_file.create_dataset('original_widths', data=original_widths)


def words_preprocess(phrase):
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
    for k, v in replacements.items():
        phrase = phrase.replace(k, v)

    return str(phrase).lower().translate(str.maketrans("", "", string.punctuation)).strip().split()


def split_filter_captions(data, max_token_length, tokens_type, verbose=True):
    """
    Modifies data in-place by adding a 'tokens' field to each region.
    If the region's label is too long, 'tokens' will be None; otherwise
    it will be a list of strings.
    Splits by space when tokens_type = "words", or lists all chars when "chars"
    """
    captions_kept = 0
    img_kept = 0
    img_removed = 0
    captions_removed = 0
    pbar = mmcv.ProgressBar(len(data))
    for i, img in enumerate(data):
        regions_per_image = 0
        for region in img['relationships']:
            # create tokens array
            if tokens_type == 'words':
                tokens = words_preprocess(region['phrase'])
            elif tokens_type == 'chars':
                tokens = list(region['label'])
            else:
                assert False, 'tokens_type must be "words" or "chars"'

            # filter by length
            if max_token_length > 0 and len(tokens) <= max_token_length:
                region['tokens'] = tokens
                # pdb.set_trace()
                region['parts'] = [1] * len(words_preprocess(region['subject']['name'])) + [2] * len(
                    words_preprocess(region['predicate'])) + [3] * len(words_preprocess(region['object']['name']))

                captions_kept += 1
                regions_per_image = regions_per_image + 1
            else:
                region['tokens'] = None
                captions_removed += 1
        if regions_per_image == 0:
            img_removed += 1
        else:
            img_kept += 1
        pbar.update()

    print('\n ###### WANRING: kept %d, removed %d' % (img_kept, img_removed))

    if verbose:
        print('Keeping %d captions' % captions_kept)
        print('Skipped %d captions for being too long' % captions_removed)


def encode_splits(data, split_data):
    """ Encode splits as intetgers and return the array. """
    lookup = {'train': 0, 'val': 1, 'test': 2}
    id_to_split = {}
    split_array = np.zeros(len(data))
    for split, idxs in split_data.iteritems():
        for idx in idxs:
            id_to_split[idx] = split
    for i, img in enumerate(data):
        split_array[i] = lookup[id_to_split[img['image_id']]]
    return split_array


def filter_images(data, vgid2idx, meta_vgids):
    """ Keep only images that are in some split and have some captions """
    new_data = []
    for vgid in meta_vgids:
        new_data.append(data[vgid2idx[vgid]])
    return new_data


def main(args):
    # read in the data
    data = mmcv.load(args.relationship_data)  # 108,077
    vgid2idx = {item['image_id']: i for i, item in enumerate(data)}
    meta_infos = pd.read_csv(args.meta_info, low_memory=False)
    meta_vgids = list(meta_infos['meta_vgids'])

    # Only keep images that are in a split
    print('There are %d images total' % len(data))
    data = filter_images(data, vgid2idx, meta_vgids)
    print('After filtering for splits there are %d images' % len(data))

    if args.max_images > 0:
        data = data[:args.max_images]

    # create the output hdf5 file handle
    f = h5py.File(args.h5_output, 'w')

    # add split information
    split = np.array(list(meta_infos['vg150_split']))
    f.create_dataset('split', data=split)

    # process "label" field in each region to a "tokens" field, and cap at some max length
    split_filter_captions(data, args.max_token_length, args.tokens_type)

    # build vocabulary
    vocab = build_vocab(data, args.min_token_instances)  # vocab is a set()
    token_to_idx, idx_to_token = build_vocab_dict(vocab)  # both mappings are dicts

    # encode labels
    captions_matrix, parts_matrix, lengths_vector = encode_captions(data, token_to_idx, args.max_token_length)

    f.create_dataset('labels', data=np.concatenate((captions_matrix, parts_matrix), axis=1))
    f.create_dataset('lengths', data=lengths_vector)

    # encode boxes
    original_heights = np.array(list(meta_infos['meta_heights']))
    original_widths = np.array(list(meta_infos['meta_widths']))
    boxes_matrix, entity_names = encode_boxes(data, original_heights, original_widths)

    f.create_dataset('boxes', data=boxes_matrix)

    # integer mapping between image ids and box ids
    img_to_first_box, img_to_last_box = build_img_idx_to_box_idxs(data)
    f.create_dataset('img_to_first_box', data=img_to_first_box)
    f.create_dataset('img_to_last_box', data=img_to_last_box)
    filename_to_idx, idx_to_filename = build_filename_dict(data)
    box_to_img = encode_filenames(data, filename_to_idx)

    f.create_dataset('box_to_img', data=box_to_img)

    #label_to_idx = build_entity_dict(entity_names)
    #print('NUM labels: %d' % len(label_to_idx))
    #labels = encode_labels(data, label_to_idx)
    #f.create_dataset('labels', data=labels)

    f.close()

    # and write the additional json file
    json_struct = {
        'token_to_idx': token_to_idx,
        'idx_to_token': idx_to_token,
        'filename_to_idx': filename_to_idx,
        'idx_to_filename': idx_to_filename,
        #'label_to_idx': label_to_idx
    }

    mmcv.dump(json_struct, args.json_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # INPUT settings
    parser.add_argument('--relationship_data',
                        default='data/relational_caption/relational_captions.json',
                        help='Input JSON file with relationships')
    parser.add_argument('--meta_info',
                        default='data/visualgenome/meta_form.csv',
                        help='splits')

    # OUTPUT settings
    parser.add_argument('--json_output',
                        default='data/relational_caption/VG-regions-dicts.json',
                        help='Path to output JSON file')
    parser.add_argument('--h5_output',
                        default='data/relational_caption/VG-regions.h5',
                        help='Path to output HDF5 file')

    # OPTIONS
    parser.add_argument('--image_size',
                        default=720, type=int,
                        help='Size of longest edge of preprocessed images')
    parser.add_argument('--max_token_length',
                        default=15, type=int,
                        help="Set to 0 to disable filtering")
    parser.add_argument('--min_token_instances',
                        default=3, type=int,
                        help="When token appears less than this times it will be mapped to <UNK>")
    parser.add_argument('--tokens_type', default='words',
                        help="Words|chars for word or char split in captions")
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--max_images', default=-1, type=int,
                        help="Set to a positive number to limit the number of images we process")
    args = parser.parse_args()
    main(args)
