from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from ..registry import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


@PIPELINES.register_module
class ToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)


@PIPELINES.register_module
class ImageToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)


@PIPELINES.register_module
class Transpose(object):

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={}, order={})'.format(
            self.keys, self.order)


@PIPELINES.register_module
class ToDataContainer(object):

    def __init__(self,
                 fields=(dict(key='img', stack=True), dict(key='gt_bboxes'),
                         dict(key='gt_labels'))):
        self.fields = fields

    def __call__(self, results):
        for field in self.fields:
            field = field.copy()
            key = field.pop('key')
            results[key] = DC(results[key], **field)
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(fields={})'.format(self.fields)


@PIPELINES.register_module
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
        if 'gt_saliency_map' in results:
            results['gt_saliency_map'] = DC(
                to_tensor(results['gt_saliency_map'][None, ...]), stack=True)
        if 'rel_fields' in results and len(results['rel_fields']) > 0:
            for key in results['rel_fields']:
                results[key] = DC(to_tensor(results[key]))
        if 'attr_fields' in results and len(results['attr_fields']) > 0:
            for key in results['attr_fields']:
                results[key] = DC(to_tensor(results[key]))
        if 'gt_scenes' in results:
            results['gt_scenes'] = DC(to_tensor(results['gt_scenes']))

        # for image captioning
        if 'att_feats' in results:
            results['att_feats'] = DC(to_tensor(results['att_feats']), stack=True, pad_dims=1)
        # no need to pad gv_fat, input_seq, target_seq any more
        if 'gv_feat' in results:
            results['gv_feat'] = DC(to_tensor(results['gv_feat']), stack=True, pad_dims=None)
        if 'input_seq' in results:
            results['input_seq'] = DC(to_tensor(results['input_seq']), stack=True, pad_dims=None)
        if 'target_seq' in results:
            results['target_seq'] = DC(to_tensor(results['target_seq']), stack=True, pad_dims=None)
        if 'bboxes' in results:
            results['bboxes'] = DC(to_tensor(results['bboxes']))

        # for relational captioning
        if 'gt_rel_inputs' in results:
            results['gt_rel_inputs'] = DC(to_tensor(results['gt_rel_inputs']))
        if 'gt_rel_targets' in results:
            results['gt_rel_targets'] = DC(to_tensor(results['gt_rel_targets']))
        if 'gt_rel_ipt_scores' in results:
            results['gt_rel_ipt_scores'] = DC(to_tensor(results['gt_rel_ipt_scores']))
        if 'gt_cap_inputs' in results:
            results['gt_cap_inputs'] = DC(to_tensor(results['gt_cap_inputs']))
        if 'gt_cap_targets' in results:
            results['gt_cap_targets'] = DC(to_tensor(results['gt_cap_targets']))

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module
class Collect(object):
    """
    Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple
            (h, w, c).  Note that images may be zero padded on the bottom/right
            if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:
            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_meta'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(keys={}, meta_keys={})'.format(
            self.keys, self.meta_keys)
