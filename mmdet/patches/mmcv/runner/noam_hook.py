# ---------------------------------------------------------------
# noam_hook.py
# Set-up time: 2021/1/3 22:42
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

from mmcv.runner import Hook
from torch.optim.lr_scheduler import _LRScheduler

class NoamLrUpdateHook(Hook, _LRScheduler):
    def __init__(
        self,
        optimizer,
        model_size,
        factor,
        warmup,
        step_type='iter',
        last_epoch=-1):

        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self.step_type = step_type
        Hook.__init__(self)

        # this has made the LRScheduler's step added by 1
        _LRScheduler.__init__(self, optimizer, last_epoch)

    def get_lr(self):
        # the self.last_epoch and self.base_lrs are from LRScheduler
        return [
            self.factor * \
            (self.model_size ** (-0.5) *
            min((self.last_epoch + 1) ** (-0.5), (self.last_epoch + 1) * self.warmup ** (-1.5)))
            for base_lr in self.base_lrs
        ]

    def after_train_epoch(self, runner):
        if self.step_type == 'epoch':
            self.step(epoch=None)

    def after_train_iter(self, runner):
        self.step(epoch=None)

