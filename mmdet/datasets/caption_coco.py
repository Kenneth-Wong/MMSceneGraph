# ---------------------------------------------------------------
# caption_coco.py
# Set-up time: 2020/12/29 11:30
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import itertools
import logging
import os.path as osp
import os
import tempfile

import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from coco_caption.pycocoevalcap.eval import COCOEvalCap
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from mmdet.utils import print_log
from torch.utils.data import Dataset
from .registry import DATASETS
from .pipelines import Compose
import random


@DATASETS.register_module
class CaptionCocoDataset(Dataset):
    """
    COCO dataset for image captioning.

    Generally, if both gv_feat_prefix or att_feats_prefix are not None, it directly use
    the pre-extracted features, such as bottom-up features.
    """

    def __init__(
            self,
            image_ids_path,
            box_size_path,
            pipeline,
            seq_per_img=5,
            max_feat_num=-1,
            input_seq=None,
            target_seq=None,
            gv_feat_prefix='',
            att_feats_prefix='',
            split='train',
            num_img=-1,
            test_mode=False,
            ann_file=None):

        assert split in ['train', 'val', 'test']
        self.split = split
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)

        self.box_size_prefix = box_size_path if len(box_size_path) > 0 else None
        self.att_feats_prefix = att_feats_prefix if len(att_feats_prefix) > 0 else None
        self.gv_feat = mmcv.load(gv_feat_prefix) if len(gv_feat_prefix) > 0 else None
        self.max_feat_num = max_feat_num
        self.seq_per_img = seq_per_img
        self.img_ids = [line.strip() for line in open(image_ids_path)]
        if num_img > 0:
            self.img_ids = self.img_ids[:num_img]
        if input_seq is not None and target_seq is not None:
            self.input_seq = mmcv.load(input_seq)
            self.target_seq = mmcv.load(target_seq)
            self.seq_len = len(self.input_seq[self.img_ids[0]][0, :])
        else:
            self.seq_len = -1
            self.input_seq = None
            self.target_seq = None

        if not self.test_mode:
            self._set_group_flag()

        ## ann_file: used for evaluation, if this is the val/test dataset.
        self.ann_file = ann_file
        if self.ann_file is not None:
            self.coco = COCO(self.ann_file)


    def set_seq_per_img(self, seq_per_img):
        self.seq_per_img = seq_per_img

    def __len__(self):
        return len(self.img_ids)

    def get_ann_info(self, idx):
        image_id = self.img_ids[idx]

        input_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
        target_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')

        n = len(self.input_seq[image_id])
        if n >= self.seq_per_img:
            sid = 0
            ixs = random.sample(range(n), self.seq_per_img)
        else:
            sid = n
            ixs = random.sample(range(n), self.seq_per_img - n)
            input_seq[0:n, :] = self.input_seq[image_id]
            target_seq[0:n, :] = self.target_seq[image_id]

        for i, ix in enumerate(ixs):
            input_seq[sid + i] = self.input_seq[image_id][ix, :]
            target_seq[sid + i] = self.target_seq[image_id][ix, :]

        return dict(input_seq=input_seq, target_seq=target_seq)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        A fake flag
        """
        self.flag = np.ones(len(self), dtype=np.uint8)


    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def load_imginfo(self, idx):
        # Here, all the pre-extracted features and the bboxes are regarded as the image infos, i.e.,
        # every image has these attributes, no matter train/val/test
        img_id = self.img_ids[idx]
        att_feat = np.array(np.load(osp.join(self.att_feats_prefix, str(img_id) + '.npz'))['feat'],
                            dtype=np.float32)
        if self.max_feat_num > 0 and att_feat.shape[0] > self.max_feat_num:
            att_feat = att_feat[:self.max_feat_num, :]
        # trick: as the collate fucntion in mmcv only pad the last few dims, we transpose the att_feat so that
        # the dim that needs to be padded (#region) are is the last dim.
        att_feat = att_feat.T  # (ndim, num_region)

        box_size = mmcv.load(osp.join(self.box_size_prefix, str(img_id) + '.pickle'))
        img_info = {'height': box_size['h'], 'width': box_size['w'], 'bboxes': box_size['boxes'],
                    'att_feats': att_feat, 'gv_feat': self.gv_feat[img_id] if self.gv_feat is not None else np.zeros((1,)),
                    'coco_id': img_id}
        return img_info

    def prepare_train_img(self, idx):
        img_info = self.load_imginfo(idx)
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.load_imginfo(idx)
        results = dict(img_info=img_info)
        return self.pipeline(results)

    def evaluate(self, results, epoch=None, logger=None):
        assert hasattr(self, 'coco')

        tmp_dir = tempfile.TemporaryDirectory()

        result_file = osp.join(tmp_dir.name, 'cap_results.json')
        mmcv.dump(results, result_file)

        cocoRes = self.coco.loadRes(result_file)
        cocoEval = COCOEvalCap(self.coco, cocoRes)
        cocoEval.evaluate()
        os.remove(result_file)

        msg = '######## Epoch ({}) {} ########'.format(self.split, epoch if epoch is not None else 'X')
        if logger is None:
            msg = '\n' + msg
        msg = msg + '\n' + str(cocoEval.eval)
        print_log(msg, logger=logger)

        eval_results = {}
        eval_results[self.split + '_set/copystate'] = ''
        for metric, value in cocoEval.eval.items():
            eval_results[metric] = value
            eval_results[self.split + '_set/copystate'] += (metric + ': ' + str(value) + '\t')

        return eval_results


