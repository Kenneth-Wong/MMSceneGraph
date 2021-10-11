import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from ..registry import PIPELINES


@PIPELINES.register_module
class LoadImageFromFile(object):

    def __init__(self, to_float32=False, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img = mmcv.imread(filename, self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['flip'] = False
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = [[0.0] * num_channels, [1.0] * num_channels,
                                   False]
        return results

    def __repr__(self):
        return '{} (to_float32={}, color_type={})'.format(
            self.__class__.__name__, self.to_float32, self.color_type)


@PIPELINES.register_module
class LoadMultiChannelImageFromFiles(object):
    """ Load multi channel images from a list of separate channel files.
    Expects results['filename'] to be a list of filenames
    """

    def __init__(self, to_float32=True, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = [
                osp.join(results['img_prefix'], fname)
                for fname in results['img_info']['filename']
            ]
        else:
            filename = results['img_info']['filename']
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return '{} (to_float32={}, color_type={})'.format(
            self.__class__.__name__, self.to_float32, self.color_type)


@PIPELINES.register_module
class LoadAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 with_map=False,
                 with_rel=False,
                 with_keyrel=False,
                 with_attr=False,
                 with_relcap=False,
                 with_cap=False,
                 poly2mask=True,
                 with_scene=False):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.with_map = with_map
        self.with_rel = with_rel
        self.with_keyrel = with_keyrel
        self.with_attr = with_attr
        self.with_relcap = with_relcap
        self.with_cap = with_cap
        self.poly2mask = poly2mask
        self.with_scene = with_scene

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def _load_maps(self, results):
        results['gt_saliency_map'] = mmcv.imread(
            osp.join(results['map_prefix'], results['ann_info']['saliency_map']),
            flag='unchanged').squeeze()
        if len(results['gt_saliency_map'].shape) == 3:
            results['gt_saliency_map'] = results['gt_saliency_map'][:, :, 0]
        #!! NOTE: some map may have 24 bits, so there are 3 channels. But actually they are all 255 in three channels
        results['map_fields'].append('gt_saliency_map')
        return results

    def _load_rels(self, results):
        ann_info = results['ann_info']
        results['gt_rels'] = ann_info['rels']
        results['gt_relmaps'] = ann_info['rel_maps']

        assert 'rel_fields' in results

        results['rel_fields'] += ['gt_rels', 'gt_relmaps']
        return results

    def _load_keyrels(self, results):
        ann_info = results['ann_info']
        results['gt_keyrels'] = ann_info['key_rels']
        assert 'rel_fields' in results
        results['rel_fields'] += ['gt_keyrels']
        return results

    def _load_relcaps(self, results):
        ann_info = results['ann_info']
        results['gt_rel_inputs'] = ann_info['rel_inputs']
        results['gt_rel_targets'] = ann_info['rel_targets']
        results['gt_rel_ipt_scores'] = ann_info['rel_ipt_scores']
        return results

    def _load_caps(self, results):
        ann_info = results['ann_info']
        results['gt_cap_inputs'] = ann_info['cap_inputs']
        results['gt_cap_targets'] = ann_info['cap_targets']
        return results

    def _load_attrs(self, results):
        ann_info = results['ann_info']
        results['gt_attrs'] = ann_info['attrs']
        assert 'attr_fields' in results
        results['attr_fields'].append('gt_attrs')
        return results

    def _load_scene(self, results):
        ann_info = results['ann_info']
        results['gt_scenes'] = ann_info['scenes']
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        if self.with_map:
            results = self._load_maps(results)
        if self.with_rel:
            results = self._load_rels(results)
        if self.with_keyrel:
            results = self._load_keyrels(results)
        if self.with_attr:
            results = self._load_attrs(results)
        if self.with_relcap:
            results = self._load_relcaps(results)
        if self.with_cap:
            results = self._load_caps(results)
        if self.with_scene:
            results = self._load_scene(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={}, with_map={}, with_rel={}, with_keyrel={}, with_attr={}, with_relcap={},'
                     ' with_cap={}, with_scene={})').format(self.with_bbox, self.with_label,
                                                            self.with_mask, self.with_seg,
                                                            self.with_map,
                                                            self.with_rel, self.with_attr,
                                                            self.with_keyrel,
                                                            self.with_relcap, self.with_cap, self.with_scene)
        return repr_str


@PIPELINES.register_module
class LoadProposals(object):

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                'but found {}'.format(proposals.shape))
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(num_max_proposals={})'.format(
            self.num_max_proposals)


#  For Image Captioning
@PIPELINES.register_module
class LoadCaptionVisuals(object):
    def __init__(self,
                 with_gv_feat=False,
                 with_att_feat=True,
                 with_bboxes=True):
        self.with_gv_feat = with_gv_feat
        self.with_att_feat = with_att_feat
        self.with_bboxes = with_bboxes

    def __call__(self, results):
        img_info = results['img_info']
        # load the basic info
        results['height'] = img_info['height']
        results['width'] = img_info['width']
        results['coco_id'] = img_info['coco_id']
        if self.with_gv_feat:
            results['gv_feat'] = img_info['gv_feat']
        if self.with_att_feat:
            results['att_feats'] = img_info['att_feats']
            results['num_att'] = img_info['att_feats'].shape[-1]
        if self.with_bboxes:
            results['bboxes'] = img_info['bboxes']
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_gv_feat={}, with_att_feat={}, with_bboxes={}').format(self.with_gv_feat, self.with_att_feat,
                                                                                  self.with_bboxes)
        return repr_str


@PIPELINES.register_module
class LoadCaptionAnnotations(object):

    def __init__(self,
                 with_input_seq=True,
                 with_target_seq=True):
        self.with_input_seq = with_input_seq
        self.with_target_seq = with_target_seq

    def _load_input_seq(self, results):
        ann_info = results['ann_info']
        results['input_seq'] = ann_info['input_seq']
        return results

    def _load_target_seq(self, results):
        ann_info = results['ann_info']
        results['target_seq'] = ann_info['target_seq']
        return results

    def __call__(self, results):
        if self.with_input_seq:
            results = self._load_input_seq(results)
        if self.with_target_seq:
            results = self._load_target_seq(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_input_seq={}, with_target_seq={}').format(self.with_input_seq, self.with_target_seq)
        return repr_str
