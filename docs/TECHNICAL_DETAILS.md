# Technical Details

In this section, we will introduce the main units of training a detector:
data pipeline, model and iteration pipeline.

## Data pipeline

Following typical conventions, we use `Dataset` and `DataLoader` for data loading
with multiple workers. `Dataset` returns a dict of data items corresponding
the arguments of models' forward method.
Since the data in object detection may not be the same size (image size, gt bbox size, etc.),
we introduce a new `DataContainer` type in MMCV to help collect and distribute
data of different size.
See [here](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py) for more details.

The data preparation pipeline and the dataset is decomposed. Usually a dataset
defines how to process the annotations and a data pipeline defines all the steps to prepare a data dict.
A pipeline consists of a sequence of operations. Each operation takes a dict as input and also output a dict for the next transform.

We present a classical pipeline in the following figure. The blue blocks are pipeline operations. With the pipeline going on, each operator can add new keys (marked as green) to the result dict or update the existing keys (marked as orange).
![pipeline figure](../demo/data_pipeline.png)

The operations are categorized into data loading, pre-processing, formatting and test-time augmentation.

**NOTE**

For scene graph generation, you should load the ground truth relationships (`gt_rels`) as well. 
Since there are three evaluation protocols, `gt_bboxes` and `gt_labels`  are necessary for either `predcls` or `sgcls`,
they should be loaded even in the test pipeline. 

Here is an pipeline example for scene graph generation. If you want to compare the ground truth and predicted
scene graph, you must keep the `with_rel=True` in `test_pipeline`.
```python
dataset_type = 'VisualGenomeDataset'
data_root = 'data/visualgenome/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_rel=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_rels', 'gt_relmaps']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # Since the forward process may need gt info, annos must be loaded.
    dict(type='LoadAnnotations', with_bbox=True, with_rel=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            # NOTE: Do not change the img to DC.
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels']),
            dict(type='ToDataContainer', fields=(dict(key='gt_bboxes'), dict(key='gt_labels'))),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ])
]
```

For each operation, we list the related dict fields that are added/updated/removed.

### Data loading

`LoadImageFromFile`
- add: img, img_shape, ori_shape

`LoadAnnotations`
- add: gt_bboxes, gt_bboxes_ignore, gt_labels, gt_masks, gt_semantic_seg, bbox_fields, mask_fields

`LoadProposals`
- add: proposals

### Pre-processing

`Resize`
- add: scale, scale_idx, pad_shape, scale_factor, keep_ratio
- update: img, img_shape, *bbox_fields, *mask_fields, *seg_fields

`RandomFlip`
- add: flip
- update: img, *bbox_fields, *mask_fields, *seg_fields

`Pad`
- add: pad_fixed_size, pad_size_divisor
- update: img, pad_shape, *mask_fields, *seg_fields

`RandomCrop`
- update: img, pad_shape, gt_bboxes, gt_labels, gt_masks, *bbox_fields

`Normalize`
- add: img_norm_cfg
- update: img

`SegRescale`
- update: gt_semantic_seg

`PhotoMetricDistortion`
- update: img

`Expand`
- update: img, gt_bboxes

`MinIoURandomCrop`
- update: img, gt_bboxes, gt_labels

`Corrupt`
- update: img

### Formatting

`ToTensor`
- update: specified by `keys`.

`ImageToTensor`
- update: specified by `keys`.

`Transpose`
- update: specified by `keys`.

`ToDataContainer`
- update: specified by `fields`.

`DefaultFormatBundle`
- update: img, proposals, gt_bboxes, gt_bboxes_ignore, gt_labels, gt_masks, gt_semantic_seg

`Collect`
- add: img_meta (the keys of img_meta is specified by `meta_keys`)
- remove: all other keys except for those specified by `keys`

### Test time augmentation

`MultiScaleFlipAug`

## Model

In MMDetection, model components are basically categorized as 4 types.

- backbone: usually a FCN network to extract feature maps, e.g., ResNet.
- neck: the part between backbones and heads, e.g., FPN, ASPP.
- head: the part for specific tasks, e.g., bbox prediction and mask prediction.
- roi extractor: the part for extracting features from feature maps, e.g., RoI Align.

We also write implement some general detection pipelines with the above components,
such as `SingleStageDetector` and `TwoStageDetector`.

For a model in MMSceneGraph, we temporarily develop it based on `TwoStageDetector`. It may contain
the following additional components:

- relation roi extractor: the part for extracting the relation features which may include the visual and spatial features.
- relation head: a basic class for leading different scene graph generation methods.
- relational caption head: a basic class for leading different linguistic scene graph generation methods (refer to topicSG ICCV 2021).

### Others

We provide the models for saliency object detection (`mmdet/models/saliency_detectors`) and image captioning (`mmdet/models/captioners`), feel free to try it.


## Other information

For more information, please refer to our [technical report](https://arxiv.org/abs/1906.07155).


