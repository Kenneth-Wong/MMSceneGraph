#!/usr/bin/env python
# ---------------------------------------------------------------
# meteor_bridge.py
# Set-up time: 2021/2/26 15:22
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import os
import sys
import subprocess
import threading
import mmcv

from coco_caption.pycocoevalcap.meteor.meteor import Meteor


class Meteor_(Meteor):
    def compute_score(self, c, r):
        eval_line = 'EVAL'
        self.lock.acquire()
        stat = self._stat(c, r)
        eval_line += ' ||| {}'.format(stat)

        self.meteor_p.stdin.write(eval_line + '\n')
        score = float(self.meteor_p.stdout.readline().strip())
        final_score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()
        return final_score

    def compute(self, candidates, references):
        scores = []
        pbar = mmcv.ProgressBar(len(candidates))
        for c, r in zip(candidates, references):
            score = self.compute_score(c, r if r is not None else [])
            scores.append(score)
            pbar.update()

        out = dict()
        out['scores'] = scores
        out['average_score'] = sum(scores) / len(scores)
        return out
