# ---------------------------------------------------------------
# caption_eval_hooks.py
# Set-up time: 2021/1/6 22:04
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import os.path as osp
import mmcv
from mmcv.runner import Hook
from torch.utils.data import DataLoader
import os.path as osp

class CaptionEvalHook(Hook):
    """Evaluation hook.
    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        from mmdet.apis import caption_single_gpu_test
        results = caption_single_gpu_test(runner.model, self.dataloader, **self.eval_kwargs)

        # save the results:
        res_dir = osp.join(runner.work_dir, 'eval_results')
        mmcv.mkdir_or_exist(res_dir)
        mmcv.dump(results, osp.join(res_dir, self.dataloader.dataset.split + '_' + str(runner.epoch + 1) + '.json'))
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(
            results, epoch=runner.epoch, logger=runner.logger)
        # NOTE: Add suffix to distinguish evaluation on test set or val set.
        for name, val in eval_res.items():
            runner.log_buffer.output[name+'/'+self.dataloader.dataset.split] = val
        runner.log_buffer.ready = True


class CaptionDistEvalHook(Hook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.eval_kwargs = eval_kwargs

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        from mmdet.apis import caption_multi_gpu_test
        results = caption_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect,
            **self.eval_kwargs)
        if runner.rank == 0:
            print('\n')
            # save the results:
            res_dir = osp.join(runner.work_dir, 'eval_results')
            mmcv.mkdir_or_exist(res_dir)
            mmcv.dump(results, osp.join(res_dir, self.dataloader.dataset.split + '_' + str(runner.epoch + 1) + '.json'))

            self.evaluate(runner, results)

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(
            results, epoch=runner.epoch, logger=runner.logger)
        for name, val in eval_res.items():
            runner.log_buffer.output[name+'/'+self.dataloader.dataset.split] = val
        runner.log_buffer.ready = True