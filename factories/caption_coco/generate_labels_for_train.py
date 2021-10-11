# ---------------------------------------------------------------
# generate_caption_labels.py
# Set-up time: 2021/1/8 9:56
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------



"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.py

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image,
  such as in particular the 'split' it was assigned to.
"""
import os
import os.path as osp
import mmcv
import numpy as np
import argparse
from random import shuffle, seed
import json
import string
# non-standard dependencies:
import h5py
import torch
from PIL import Image
from mmdet.models.captioners.utils import load_vocab, load_ids

def prepro_captions(imgs):
    # preprocess all the captions
    print('example processed tokens:')
    for i, img in enumerate(imgs):
        img['processed_tokens'] = []
        for j, s in enumerate(img['captions']):
            txt = str(s).lower().translate(str.maketrans("", "", string.punctuation)).strip().split()
            img['processed_tokens'].append(txt)


def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']  # 5

    # count up the number of words
    counts = {}
    for img in imgs:
        for sent in img['sentences']:
            for w in sent['tokens']:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for sent in img['sentences']:
            txt = sent['tokens']
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len + 1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')

    for img in imgs:
        img['final_captions'] = []
        for sent in img['sentences']:
            txt = sent['tokens']
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            img['final_captions'].append(caption)

    return vocab


def encode_captions(imgs, params, wtoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    max_length = params['max_length']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs)  # total number of captions

    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32')  # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    for i, img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')
        for j, s in enumerate(img['final_captions']):
            label_length[caption_counter] = min(max_length, len(s))  # record the length of this sequence
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]

        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0)  # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    print('encoded captions to array of size ', L.shape)
    return L, label_start_ix, label_end_ix, label_length


def encode_inp_tgt_captions(imgs, params, wtoi, cocoids):
    max_length = params['max_length']  # 16
    N = len(imgs)  # 123,287
    M = sum(len(img['sentences']) for img in imgs)  # total number of captions

    label_arrays = []

    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    pbar = mmcv.ProgressBar(N)
    output_inp, output_tgt = {}, {}
    for i, img in enumerate(imgs):
        pbar.update()
        if img['cocoid'] not in cocoids:
            continue
        num_caps = len(img['sentences'])
        assert num_caps > 0, 'error: some image has no captions'

        Li = np.zeros((num_caps, max_length), dtype='uint32')
        for j, s in enumerate(img['sentences']):
            label_length[caption_counter] = min(max_length, len(s['tokens']))  # record the length of this sequence
            caption_counter += 1
            for k, w in enumerate(s['tokens']):
                if k < max_length:
                    Li[j, k] = wtoi.get(w, wtoi['UNK'])

        Li_inp = np.hstack((np.zeros((num_caps, 1)), Li)).astype(np.int32)
        Li_tgt = np.hstack((Li, np.zeros((num_caps, 1)))).astype(np.int32)
        for seq in Li_tgt:
            first_zero_ind = np.where(seq == 0)[0][0]
            seq[first_zero_ind+1:] = -1

        output_inp[str(img['cocoid'])] = Li_inp
        output_tgt[str(img['cocoid'])] = Li_tgt

    return output_inp, output_tgt


def main(params):
    imgs = mmcv.load(params['input_json'])
    imgs = imgs['images']

    vocab = load_vocab(params['input_vocab_json'])

    seed(123)  # make reproducible

    # create the vocab
    # the load_vocab has added a '.' to align. So no need to plus 1.
    itow = {i: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i for i, w in enumerate(vocab)}  # inverse table

    cocoids = load_ids(params['input_id_json'])

    # encode captions in large arrays, ready to ship to hdf5 file
    output_inp, output_tgt = encode_inp_tgt_captions(imgs, params, wtoi, cocoids)

    mmcv.dump(output_inp, params['output_inp_pkl'])
    mmcv.dump(output_tgt, params['output_tgt_pkl'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='data/caption_coco/karpathy_captions/dataset_coco.json')
    parser.add_argument('--input_id_json', default='data/caption_coco/karpathy_captions/coco_val_image_id.txt')
    parser.add_argument('--input_vocab_json', default='data/caption_coco/karpathy_captions/coco_vocabulary.txt')
    parser.add_argument('--output_inp_pkl', default='data/caption_coco/karpathy_captions/coco_val_input.pkl')
    parser.add_argument('--output_tgt_pkl', default='data/caption_coco/karpathy_captions/coco_val_target.pkl')
    # options
    parser.add_argument('--max_length', default=16, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
