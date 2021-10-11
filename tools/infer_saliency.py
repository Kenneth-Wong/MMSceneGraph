"""
This script is for inferring the saliency map, but not for evaluation.
"""
import argparse
import os
import os.path as osp
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_saliency_detector
from torchvision import transforms
import numpy as np
from PIL import Image
import pydensecrf.densecrf as dcrf

# codes of this function are borrowed from https://github.com/Andrew-Qibin/dss_crf
def crf_refine(img, annos):
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')

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
    parser.add_argument('testdata_config', help='a file written with the dataset config you want to test.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()


    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    # build the model and load checkpoint
    model = build_saliency_detector(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    to_pil = transforms.ToPILImage()

    # build datasets
    test_data_cfg = mmcv.Config.fromfile(args.testdata_config)
    datasets = [build_dataset(ds_cfg) for ds_cfg in test_data_cfg.test_data]
    data_loaders = [build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False) for dataset in datasets]

    # begin inferring
    method_name = cfg.model.type[:-16]

    for data_loader in data_loaders:
        dataset = data_loader.dataset
        prediction_dir = osp.join(cfg.work_dir, method_name, dataset.dataset_name)
        mmcv.mkdir_or_exist(prediction_dir)
        print('\n inferring the {}\n'.format(dataset.dataset_name))
        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                sal_map = model(return_loss=False,  **data).squeeze(0).cpu() # 1(channel) * H * W, float
                sal_map = to_pil(sal_map)  # to PIL Image, H * W, 0~255, uint8
                sal_map = np.array(sal_map)  # uint8

                filename = data['img_meta'][0].data[0][0]['filename']
                img = Image.open(filename).convert('RGB')
                #sal_map = crf_refine(np.array(img), sal_map)
                Image.fromarray(sal_map).save(osp.join(prediction_dir,
                                                       filename.split('/')[-1].split('.')[0] + '.png'))
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size):
                prog_bar.update()

    mmcv.symlink(osp.abspath(osp.join(cfg.work_dir, method_name)),
                 osp.join(test_data_cfg.SOC_data_root, 'SOCToolbox/maps/Prediction', method_name))




if __name__ == '__main__':
    main()
