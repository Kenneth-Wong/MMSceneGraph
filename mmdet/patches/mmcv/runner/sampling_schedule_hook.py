# ---------------------------------------------------------------
# sampling_schedule_hook.py
# Set-up time: 2021/1/13 20:32
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

from mmcv.runner import Hook


class SamplingScheduleHook(Hook):
    def __init__(self, start, inc_every, inc_prob, max_prob):
        self.start = start
        self.inc_every = inc_every
        self.inc_prob = inc_prob
        self.max_prob = max_prob

    def after_train_epoch(self, runner):
        epoch = runner.epoch
        if epoch > self.start:
            frac = (epoch - self.start) // self.inc_every
            ss_prob = min(self.inc_prob * frac, self.max_prob)

            if hasattr(runner.model, 'module'):
                runner.model.module.ss_prob = ss_prob
            else:
                runner.model.ss_prob = ss_prob

