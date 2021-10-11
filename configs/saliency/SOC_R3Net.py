# ---------------------------------------------------------------
# SOC_R3Net.py
# Set-up time: 2021/5/13 0:24
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

# dataset settings
dataset_type = 'SaliencyDataset'
data_root = 'data/SOC/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_map=True, with_bbox=False, with_label=False),
    dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='MapNormalize', mean=[0.], std=[255.]),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_saliency_map']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 480),
        flip=False,
        transforms=[
            dict(type='Normalize', **img_norm_cfg),
            # NOTE: Do not change the img to DC.
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_root=data_root + 'Train/SOC/image',
        name_file=data_root + 'Train/SOC/train.txt',
        pipeline=train_pipeline,
        target_root=data_root + 'Train/SOC/mask'),
    # take a set for example, you can additionally set the test set in test.py
    val=dict(
        type=dataset_type,
        img_root=data_root + 'Test/SOC/SOC-AC/image',
        name_file=data_root + 'Test/SOC/SOC-AC/test.txt',
        pipeline=test_pipeline,
        target_root=data_root + 'Test/SOC/SOC-AC/mask'),
    test=dict(
        type=dataset_type,
        img_root=data_root + 'Test/SOC/SOC-AC/image',
        name_file=data_root + 'Test/SOC/SOC-AC/test.txt',
        pipeline=test_pipeline,
        target_root=data_root + 'Test/SOC/SOC-AC/mask'))
# model settings
dataset_config = data['train'].copy()
model = dict(
    type='R3NetSaliencyDetector',
    pretrained='checkpoints/mmlab/imnet/resnext101_64x4d-ee2c6f71.pth',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'))

find_unused_parameters = True
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005,
                 paramwise_options=dict(bias_lr_mult=2.))
optimizer_config = dict()   # !!! NOTE: It cannot be NONE!!!!
# learning policy
lr_config = dict(
    policy='poly',
    by_epoch=False,
    power=0.9)
checkpoint_config = dict(interval=1)

# yapf:enable
# runtime settings
total_epochs = 9
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './saliency_experiments/SOC_R3Net'

workflow = [('train', 1), ('val', 1)]

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 project=work_dir.split('/')[-1],
                 name='train-1',
                 config=work_dir + '/cfg.yaml'))
    ])
