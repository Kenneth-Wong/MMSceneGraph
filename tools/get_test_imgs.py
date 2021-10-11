# ---------------------------------------------------------------
# get_test_imgs.py
# Set-up time: 2021/3/8 20:07
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import argparse

import mmcv
import torch

from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--test_set', help='select the dataset for testing',
                        default='test', type=str, choices=['train', 'test', 'val'])
    parser.add_argument('--out', help='output result file in pickle format')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.__getitem__(args.test_set).__setitem__('test_mode', True)
    # change the pipeline setting if testing on training set.
    if args.test_set == 'train':
        cfg.data.train.pipeline = cfg.test_pipeline

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.__getitem__(args.test_set))

    img_infos = dataset.img_infos
    print(len(img_infos))
    mmcv.dump(img_infos, args.out)


if __name__ == '__main__':
    main()
