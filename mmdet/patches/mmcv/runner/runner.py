# ---------------------------------------------------------------
# runner.py
# Set-up time: 2020/5/14 下午11:17
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import torch
from mmcv.runner import Runner
from .checkpoint import load_checkpoint, save_checkpoint
from mmcv.runner.hooks import IterTimerHook
import os.path as osp
import mmcv

class PatchRunner(Runner):
    def load_checkpoint(self, filename, load_mapping=None, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, load_mapping, strict,
                               self.logger)

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        save_scheduler=False,
                        meta=None,
                        create_symlink=True):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        scheduler = None
        if save_scheduler:
            for hook in self._hooks:
                if hook.__class__.__name__.endswith('LrUpdateHook'):
                    scheduler = hook
                    save_checkpoint(self.model, filepath, optimizer=optimizer, scheduler=scheduler, meta=meta)
                    break
        else:
            save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            mmcv.symlink(filename, osp.join(out_dir, 'latest.pth'))

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               resume_scheduler=False,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        if 'scheduler' in checkpoint and resume_scheduler:
            for hook in self._hooks:
                if hook.__class__.__name__.endswith('LrUpdateHook'):
                    hook.load_state_dict(checkpoint['scheduler'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                lr_first=True):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        if lr_first:
            self.register_lr_hook(lr_config)
            self.register_optimizer_hook(optimizer_config)
        else:
            self.register_optimizer_hook(optimizer_config)
            self.register_lr_hook(lr_config)
        self.register_checkpoint_hook(checkpoint_config)
        self.register_hook(IterTimerHook())
        self.register_logger_hooks(log_config)