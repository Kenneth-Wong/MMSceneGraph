# ---------------------------------------------------------------
# relation_saliency.py
# Set-up time: 2021/6/5 12:34
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import h5py
import numpy as np
from factories.vgkr_v1.config_v1 import *
from factories.vgkr_v2.config_v2 import meta_form_file
import mmcv
import argparse
import torch
from mmdet.models import build_saliency_detector
from mmcv.runner import load_checkpoint
from torchvision import transforms
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from PIL import Image
import pandas as pd
import os
import time
from collections import Counter



def init_detector(device='cuda:0'):
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    # build the model and load checkpoint
    model = build_saliency_detector(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cfg = cfg
    model.to(device)
    model.eval()
    return model


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        #results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        #results['pad_shape'] = img.shape
        results['flip'] = False
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = [[0.0] * num_channels, [1.0] * num_channels,
                                   False]
        return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')

    args = parser.parse_args()

    return args


def relation_saliency(model, thresh=0.5, saltype=1, npart=50, sample_num=50000):
    triplet_match_data = mmcv.load(cleanse_triplet_match_file)  # 51498
    meta_form = pd.read_csv(meta_form_file, low_memory=False)
    vgids = list(meta_form['meta_vgids'])
    meta_paths = list(meta_form['meta_paths'])

    if model is not None:
        cfg = model.cfg
        device = next(model.parameters()).device  # model device
        # build the data pipeline
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)

        to_pil = transforms.ToPILImage()

    all_triplet_names = []
    all_samples = []
    tmp_sample_dir = osp.join(os.getcwd(), 'tmp', 'samples')
    tmp_sal_dir = osp.join(os.getcwd(), 'tmp', 'salmaps')
    mmcv.mkdir_or_exist(tmp_sample_dir)
    mmcv.mkdir_or_exist(tmp_sal_dir)

    print('Infering saliency...\n')
    probar = mmcv.ProgressBar(len(triplet_match_data))
    for idx, sg in enumerate(triplet_match_data):
        probar.update()

        # begin to process the saliency
        relations = sg['relationships']
        if 'cap_ref' not in sg or 'cap_ref_by_dist' not in sg:
            continue
        cap_ref = sg['cap_ref']
        key_rel_idxes = []
        for i in cap_ref:
            if cap_ref[i] > 0:
                key_rel_idxes.append((int(i), cap_ref[i]))
        if len(key_rel_idxes) == 0:
            continue

        vgid = sg['image_id']
        img = osp.join(IMAGE_ROOT_DIR, meta_paths[vgids.index(int(vgid))])
        tgt_sal_img = osp.join(tmp_sal_dir, '%d.png' % int(vgid))
        if not osp.isfile(tgt_sal_img) and model is not None:
            img = mmcv.imread(img)
            h, w = img.shape[:2]
            # prepare data
            data = dict(img=img)
            data = test_pipeline(data)
            data = scatter(collate([data], samples_per_gpu=1), [device])[0]
            # forward the model
            with torch.no_grad():
                result = model(return_loss=False, **data).squeeze(0).cpu()
            salmap = to_pil(result)  # to PIL Image, H * W, 0~255, uint8
            salmap = np.array(salmap)  # uint8
            img_shape = data['img_meta'][0][0]['img_shape']
            salmap = salmap[:img_shape[0], :img_shape[1]]
            salmap = mmcv.imresize(salmap, size=(w, h))
            Image.fromarray(salmap).save(tgt_sal_img)
            salmap = mmcv.imnormalize(salmap, std=np.array([255.], dtype=np.float32),
                                      mean=np.array([0.], dtype=np.float32), to_rgb=False)
        else:
            salmap = mmcv.imread(tgt_sal_img, flag='unchanged').squeeze()
            salmap = mmcv.imnormalize(salmap, std=np.array([255.], dtype=np.float32),
                                      mean=np.array([0.], dtype=np.float32), to_rgb=False)
            h, w = salmap.shape[:2]

        area = h * w
        # now begin to compute the overlap between triplets and saliency
        for item in key_rel_idxes:
            rel_id, ref_num = item
            triplet_info = relations[rel_id]
            subj = triplet_info['subject']
            obj = triplet_info['object']
            predicate = triplet_info['predicate']
            subj_name = subj['names'][0] if 'names' in subj else subj['name']
            obj_name = obj['names'][0] if 'names' in obj else obj['name']
            all_triplet_names.append((subj_name, predicate, obj_name))

            sub_box = np.array([subj['x'], subj['y'], subj['w'], subj['h']])
            obj_box = np.array([obj['x'], obj['y'], obj['w'], obj['h']])
            sub_area = sub_box[2] * sub_box[3]
            obj_area = obj_box[2] * obj_box[3]
            sub_box[2:] = sub_box[:2] + sub_box[2:] - 1
            obj_box[2:] = obj_box[:2] + obj_box[2:] - 1

            sub_box = np.floor(sub_box).astype(np.int32)
            sub_box[0] = np.maximum(0, sub_box[0])
            sub_box[1] = np.maximum(0, sub_box[1])
            sub_box[2] = np.minimum(w - 1, sub_box[2])
            sub_box[3] = np.minimum(h - 1, sub_box[3])
            obj_box = np.floor(obj_box).astype(np.int32)
            obj_box[0] = np.maximum(0, obj_box[0])
            obj_box[1] = np.maximum(0, obj_box[1])
            obj_box[2] = np.minimum(w - 1, obj_box[2])
            obj_box[3] = np.minimum(h - 1, obj_box[3])
            # compute the saliency points num inside boxes
            sub_sal = salmap[sub_box[1]:(sub_box[3] + 1), sub_box[0]:(sub_box[2] + 1)]
            obj_sal = salmap[obj_box[1]:(obj_box[3] + 1), obj_box[0]:(obj_box[2] + 1)]

            # subsal_degree = np.sum(sub_sal) / ((sub_box[3] - sub_box[1] + 1) * (sub_box[2] - sub_box[0] + 1))
            # objsal_degree = np.sum(obj_sal) / ((obj_box[3] - obj_box[1] + 1) * (obj_box[2] - obj_box[0] + 1))
            if saltype == 1:
                subsal = len(np.where(sub_sal > thresh)[0]) / (
                        (sub_box[3] - sub_box[1] + 1) * (sub_box[2] - sub_box[0] + 1))
                objsal = len(np.where(obj_sal > thresh)[0]) / (
                        (obj_box[3] - obj_box[1] + 1) * (obj_box[2] - obj_box[0] + 1))
            elif saltype == 2:
                # type 2:
                subsal = np.mean(sub_sal)
                objsal = np.mean(obj_sal)
            else:
                raise NotImplementedError

            vis_saliency = subsal + objsal
            saliency = sub_area / area + obj_area / area
            all_samples.append((ref_num, saliency))

    # sampling
    all_samples = np.array(all_samples)
    # compute the distribution of ref_num
    unique_ref, ref_cnts = np.unique(all_samples[:, 0], return_counts=True)
    weights = np.zeros(all_samples.shape[0])
    for r, c in zip(unique_ref, ref_cnts):
        inds = np.where(all_samples[:, 0] == r)[0]
        weights[inds] = c / all_samples.shape[0] / inds.shape[0]
    sample_idxes = np.random.choice(range(all_samples.shape[0]), size=sample_num, p=weights.tolist())
    s_samples = all_samples[sample_idxes]
    all_triplet_names = [all_triplet_names[ind] for ind in sample_idxes]

    # grouping
    unique_ref, ref_cnts = np.unique(s_samples[:, 0], return_counts=True)
    group_samples_by_ref = {}
    for r in unique_ref:
        group_samples_by_ref[r] = s_samples[np.where(s_samples[:, 0] == r)[0]]

    print('\nCorref: %.5f' % np.corrcoef(s_samples.T)[0, 1])

    for r in group_samples_by_ref:
        print(r, np.mean(group_samples_by_ref[r]))

    with open(osp.join(tmp_sample_dir, 'samples_%d.txt' % int(time.time())), 'w') as f:
        for pt in s_samples:
            f.write(str(pt[0]) + ' ' + str(pt[1]) + '\n')


    # reverse: visual saliency as x axis, cognitive saliency as y axis
    sorted_inds = np.argsort(s_samples[:, 1])
    sorted_s_samples = s_samples[sorted_inds]
    all_triplet_names = [all_triplet_names[ind] for ind in sorted_inds]
    part_num = sorted_s_samples.shape[0] // npart
    vs_cs_points = []
    for i in range(npart):
        if i == npart - 1:
            vs_cs_points.append([sorted_s_samples[-1, 1], np.mean(sorted_s_samples[i * part_num:, 0])])
            print(sorted_s_samples[-1, 1], np.mean(sorted_s_samples[i * part_num:, 0]))
        else:
            vs_cs_points.append(
                [sorted_s_samples[(i + 1) * part_num, 1], np.mean(sorted_s_samples[i * part_num:(i + 1) * part_num, 0])])
            print(sorted_s_samples[(i + 1) * part_num, 1], np.mean(sorted_s_samples[i * part_num:(i + 1) * part_num, 0]))

    vs_cs_points = np.array(vs_cs_points)
    print('\nCorref: %.5f' % np.corrcoef(vs_cs_points.T)[0, 1])
    # max_cs_ind = np.argmax(vs_cs_points[:, 1])
    # max_cs_point = vs_cs_points[max_cs_ind, 1]
    # vsseg_point = vs_cs_points[max_cs_ind, 0]
    vsseg_point = vs_cs_points[vs_cs_points.shape[0] // 2, 0]
    low_vs_point = (vs_cs_points[0, 0] + vsseg_point) / 2
    high_vs_point = (vs_cs_points[-1, 0] + vsseg_point) / 2
    low_inds = list(set(np.where(sorted_s_samples[:, 1] <= low_vs_point)[0]))
    high_inds = list(set(np.where(sorted_s_samples[:, 1] > high_vs_point)[0]))
    # list(set(np.where(sample_points[:, 0] <= max_cs_point / 2)[0]).intersection(
    low_triplets = [all_triplet_names[ind] for ind in low_inds]
    high_triplets = [all_triplet_names[ind] for ind in high_inds]

    low_counter = Counter()
    high_counter = Counter()
    low_counter.update(low_triplets)
    high_counter.update(high_triplets)
    print(low_counter.most_common(30))
    print('====')
    print(high_counter.most_common(30))
    return low_counter, high_counter


if __name__ == '__main__':
    # if needed to run the model, uncomment it!
    # model = init_detector()
    low_counter, high_counter = relation_saliency(model=None, saltype=2)


