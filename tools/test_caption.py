# ---------------------------------------------------------------
# test_caption.py
# Set-up time: 2021/1/7 20:41
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist
from mmdet.patches import load_checkpoint

from mmdet.apis import caption_single_gpu_test, caption_multi_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_captioner


class MultipleKVAction(argparse.Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options should
    be passed as comma separated values, i.e KEY=V1,V2,V3
    """

    def _parse_int_float_bool(self, val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        return val

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            val = [self._parse_int_float_bool(v) for v in val.split(',')]
            if len(val) == 1:
                val = val[0]
            options[key] = val
        setattr(namespace, self.dest, options)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--test_set', help='select the dataset for testing',
                        default='test', type=str, choices=['train', 'test', 'val'])
    parser.add_argument('--out', help='output result file in json format')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--greedy_decode', action='store_true', help='show results')
    parser.add_argument('--beam_size', default=3, type=int, help='show results')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=MultipleKVAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--eval_file', type=str, default='',
                        help='eval an output file and avoid running the model.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.show, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results) with the argument "--out", "--eval", "--format_only" '
         'or "--show"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle', '.json')):
        raise ValueError('Invalid output format.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.data.__getitem__(args.test_set).__setitem__('test_mode', True)
    # change the pipeline setting if testing on training set.
    if args.test_set == 'train':
        cfg.data.train.pipeline = cfg.test_pipeline

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.__getitem__(args.test_set))

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    if len(args.eval_file) > 0:
        outputs = mmcv.load(args.eval_file)
        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                print('\nwriting results to {}'.format(args.out))
                mmcv.dump(outputs, args.out)
            kwargs = {} if args.options is None else args.options
            dataset.evaluate(outputs, **kwargs)
        exit(0)

    # build the model and load checkpoint
    model = build_captioner(cfg.model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = caption_single_gpu_test(model, data_loader, args.beam_size, args.greedy_decode)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = caption_multi_gpu_test(model, data_loader, args.beam_size, args.greedy_decode,
                                         args.gpu_collect, args.tmpdir)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print('\nwriting results to {}'.format(args.out))
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.options is None else args.options
        dataset.evaluate(outputs, **kwargs)


if __name__ == '__main__':
    main()
