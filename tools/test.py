import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


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
                        default='test', type=str, choices=['train', 'test', 'val', 'test_kr'])
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format_only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
             ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--save', action='store_true', help='especially for OD results: if True, save the results.')
    parser.add_argument('--relation_mode', action='store_true', help='test the relation module.')
    parser.add_argument('--key_first', action='store_true', help='use the ranking scores or not')
    parser.add_argument('--relcaption_mode', action='store_true', help='test the relation module.')
    parser.add_argument('--downstream_caption_mode', action='store_true', help='test the relation module.')
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

    assert args.out or args.eval or args.format_only or args.show, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results) with the argument "--out", "--eval", "--format_only" '
         'or "--show"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
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
    """
       train_dataset = build_dataset(cfg.data.train)  # 26298 triplets
       zs_triplets = torch.load("data/vg_evaluation/zeroshot_triplet.pytorch",
                                     map_location=torch.device("cpu")).long().numpy()
       from mmdet.core.evaluation.sgg_eval_util import intersect_2d
       import numpy as np
       our_zs = dataset.all_triplets[np.where(intersect_2d(dataset.all_triplets, train_dataset.all_triplets).sum(-1) == 0)[0]]
       match_idxes = np.where(intersect_2d(our_zs, zs_triplets).sum(-1) > 0)[0]  # 5971, all match
       """
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    if len(args.eval_file) > 0:
        outputs = mmcv.load(os.path.join(cfg.work_dir, args.eval_file))
        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                print('\nwriting results to {}'.format(os.path.join(cfg.work_dir,args.out)))
                mmcv.dump(outputs, os.path.join(cfg.work_dir,args.out))
            kwargs = {} if args.options is None else args.options
            if args.format_only:
                dataset.format_results(outputs, **kwargs)
            if args.eval:
                if dataset.__class__.__name__ in ['GeneralizedVisualGenomeDataset', 'CaptionCocoDataset']:
                    eval_scores = dataset.evaluate(outputs, args.eval, work_dir=cfg.work_dir, **kwargs)
                else:
                    eval_scores = dataset.evaluate(outputs, args.eval, **kwargs)
                if args.out:
                    print('\nwriting eval scores to {}'.format(
                        os.path.join(cfg.work_dir, args.out.split('.')[0] + '_scores.pickle')))
                    mmcv.dump(eval_scores, os.path.join(cfg.work_dir, args.out.split('.')[0] + '_scores.pickle'))
        exit(0)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    if 'PREDICATES' in checkpoint['meta']:
        model.PREDICATES = checkpoint['meta']['PREDICATES']
    elif hasattr(dataset, 'PREDICATES'):
        model.PREDICATES = dataset.PREDICATES

    if hasattr(dataset, 'ATTRIBUTES'):
        model.ATTRIBUTES = dataset.ATTRIBUTES
    if hasattr(dataset, 'TOKENS'):
        model.TOKENS = dataset.TOKENS

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.relation_mode, args.relcaption_mode,
                                  args.downstream_caption_mode, args.show, args.save,
                                  cfg, key_first=args.key_first)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.relation_mode, args.relcaption_mode,
                                 args.downstream_caption_mode,
                                 args.key_first, args.gpu_collect)

    """
        # for debug: we generate the fake output results
        import numpy as np
        num_classes = len(model.CLASSES)
        outputs = []
        num_img = len(dataset)
        for i in range(num_img):
            result = [np.random.rand(2, 5) for _ in range(num_classes)]
            for j in range(len(result)):
                result[j][:, :4] *= 800
                result[j][:, -1] = np.sort(result[j][:, -1])[::-1]

            outputs.append(result)
        """

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print('\nwriting results to {}'.format(os.path.join(cfg.work_dir, args.out)))
            mmcv.dump(outputs, os.path.join(cfg.work_dir, args.out))
        kwargs = {} if args.options is None else args.options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            # for caption dataset:
            if dataset.__class__.__name__ in ['GeneralizedVisualGenomeDataset', 'CaptionCocoDataset']:
                eval_scores = dataset.evaluate(outputs, args.eval, work_dir=cfg.work_dir, **kwargs)
            else:
                eval_scores = dataset.evaluate(outputs, args.eval, **kwargs)
            if args.out:
                print('\nwriting eval scores to {}'.format(os.path.join(cfg.work_dir, args.out.split('.')[0]+'_scores.pickle')))
                mmcv.dump(eval_scores, os.path.join(cfg.work_dir, args.out.split('.')[0]+'_scores.pickle'))



if __name__ == '__main__':
    main()
