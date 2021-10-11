from mmdet.utils import Registry

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
SHARED_HEADS = Registry('shared_head')
HEADS = Registry('head')
LOSSES = Registry('loss')
DETECTORS = Registry('detector')

RELATION_ROI_EXTRACTORS = Registry('relation_roi_extractor')

CAPTIONERS = Registry('captioner')

SALIENCY_DETECTORS = Registry('saliency_detector')
