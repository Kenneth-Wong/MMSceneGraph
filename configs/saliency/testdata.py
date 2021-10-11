# ---------------------------------------------------------------
# testdata.py
# Set-up time: 2021/5/13 20:23
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
dataset_type = 'SaliencyDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# set some data roots
MSRA10K_data_root = 'data/MSRA10K/'
SOC_data_root = 'data/SOC/'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 480),
        flip=False,
        transforms=[
            # if you want to evaluate the performance, do not use Resize, otherwise, use it for visualization.
            #dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            # NOTE: Do not change the img to DC.
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# SOC_attrs = ['AC', 'BO', 'CL', 'HO', 'MB', 'OC', 'OV', 'SC', 'SO']
test_data = [
    # SOC data
    dict(
        type=dataset_type,
        img_root=SOC_data_root + 'Test/SOC/SOC-AC/image',
        name_file=SOC_data_root + 'Test/SOC/SOC-AC/test.txt',
        pipeline=test_pipeline,
        test_mode=True,
        target_root=SOC_data_root + 'Test/SOC/SOC-AC/mask',
        dataset_name='SOC/SOC-AC'),
    dict(
        type=dataset_type,
        img_root=SOC_data_root + 'Test/SOC/SOC-BO/image',
        name_file=SOC_data_root + 'Test/SOC/SOC-BO/test.txt',
        pipeline=test_pipeline,
        test_mode=True,
        target_root=SOC_data_root + 'Test/SOC/SOC-BO/mask',
        dataset_name='SOC/SOC-BO'),
    dict(
        type=dataset_type,
        img_root=SOC_data_root + 'Test/SOC/SOC-CL/image',
        name_file=SOC_data_root + 'Test/SOC/SOC-CL/test.txt',
        pipeline=test_pipeline,
        test_mode=True,
        target_root=SOC_data_root + 'Test/SOC/SOC-CL/mask',
        dataset_name='SOC/SOC-CL'),
    dict(
        type=dataset_type,
        img_root=SOC_data_root + 'Test/SOC/SOC-HO/image',
        name_file=SOC_data_root + 'Test/SOC/SOC-HO/test.txt',
        pipeline=test_pipeline,
        test_mode=True,
        target_root=SOC_data_root + 'Test/SOC/SOC-HO/mask',
        dataset_name='SOC/SOC-HO'),
    dict(
        type=dataset_type,
        img_root=SOC_data_root + 'Test/SOC/SOC-MB/image',
        name_file=SOC_data_root + 'Test/SOC/SOC-MB/test.txt',
        pipeline=test_pipeline,
        test_mode=True,
        target_root=SOC_data_root + 'Test/SOC/SOC-MB/mask',
        dataset_name='SOC/SOC-MB/'),
    dict(
        type=dataset_type,
        img_root=SOC_data_root + 'Test/SOC/SOC-OC/image',
        name_file=SOC_data_root + 'Test/SOC/SOC-OC/test.txt',
        pipeline=test_pipeline,
        test_mode=True,
        target_root=SOC_data_root + 'Test/SOC/SOC-OC/mask',
        dataset_name='SOC/SOC-OC'),
    dict(
        type=dataset_type,
        img_root=SOC_data_root + 'Test/SOC/SOC-OV/image',
        name_file=SOC_data_root + 'Test/SOC/SOC-OV/test.txt',
        pipeline=test_pipeline,
        test_mode=True,
        target_root=SOC_data_root + 'Test/SOC/SOC-OV/mask',
        dataset_name='SOC/SOC-OV'),
    dict(
        type=dataset_type,
        img_root=SOC_data_root + 'Test/SOC/SOC-SC/image',
        name_file=SOC_data_root + 'Test/SOC/SOC-SC/test.txt',
        pipeline=test_pipeline,
        test_mode=True,
        target_root=SOC_data_root + 'Test/SOC/SOC-SC/mask',
        dataset_name='SOC/SOC-SC'),
    dict(
        type=dataset_type,
        img_root=SOC_data_root + 'Test/SOC/SOC-SO/image',
        name_file=SOC_data_root + 'Test/SOC/SOC-SO/test.txt',
        pipeline=test_pipeline,
        test_mode=True,
        target_root=SOC_data_root + 'Test/SOC/SOC-SO/mask',
        dataset_name='SOC/SOC-SO'),

    # add other testing data here!

]
