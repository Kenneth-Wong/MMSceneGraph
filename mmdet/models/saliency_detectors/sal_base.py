from abc import ABCMeta, abstractmethod
import torch.nn as nn
from mmdet.utils import print_log
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm


class BaseSalineyDetector(nn.Module, metaclass=ABCMeta):
    """Base class for saliency detector"""

    def __init__(self, eval_mode=False):
        super(BaseSalineyDetector, self).__init__()
        self.eval_mode = eval_mode

    def forward(self, img, img_meta, gt_saliency_map=None, return_loss=True):
        if return_loss:
            return self.forward_train(img, img_meta, gt_saliency_map)
        else:
            if not self.eval_mode:
                return self.forward_test(img[0], img_meta[0])
            else:
                # specifically used as a 3rd party saliency detector in other tasks, e.g., SGG
                return self.forward_test(img, img_meta)


    @abstractmethod
    def forward_train(self, img, img_meta, gt_maps):
        pass

    @abstractmethod
    def forward_test(self, img, img_meta):
        pass

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)