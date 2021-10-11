# model settings
model = dict(
    type='FasterRCNN',
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
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=3001,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    attr_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        with_avg_pool=True,
        num_classes=801,
        with_reg=False,
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # The following are for the relational caption /  captioning
    relcaption_head=dict(
        type='UpDownCrossAttnRelationalCaptionHead',
        with_relcaption=True,
        with_caption=True,
        cross_attn=False,
        head_config=dict(
            num_classes=3001,
            use_gt_box=True,
            use_gt_label=True,
            context_hidden_dim=512,
            hidden_dim=512,
            single_feat_dim=4096,
            union_feat_dim=512,
            union_spatial_dim=64,
            rnn_input_dim=512,
            gather_attention='mean',
            ipt_estimate='sou',
            soft_supervised=False,
            attn_loss_weight=1.0),
        # this is for both the relational caption or whole image caption
        caption_config=dict(
            seq_len=18,
            seq_per_img=5,
            vocab_size=11437,
            word_embed_config=dict(
                word_embed_dim=1024,
                word_embed_act='CeLU',
                word_embed_norm=False,
                dropout_word_embed=0.5,
                elu_alpha=1.3),
            global_feat_config=dict(
                gvfeat_dim=4096,
                gvfeat_embed_dim=-1,
                gvfeat_embed_act=None,
                dropout_gv_embed=0.0),
            union_feat_config=dict(
                union_feats_dim=512,
                union_spatial_dim=64,
                union_feats_embed_dim=1024,
                union_feats_embed_act='CeLU',
                dropout_union_embed=0.5,
                union_feats_norm=False
            ),
            attention_feat_config=dict(
                att_feats_dim=4096,
                att_feats_embed_dim=1024,
                att_feats_embed_act='CeLU',
                dropout_att_embed=0.5,
                att_feats_norm=False,
                att_hidden_size=1024,
                att_hidden_drop=0.5,
                att_act='Tanh',
                need_attn=True, # return the attention
            ),
            head_config=dict(
                ########## rnn param ##########,
                rnn_size=1024,
                dropout_lm=0.5,
                ########## transformer ########
                pe_max_len=5000,
                ########## bottom_up ##########,
                dropout_first_input=0.5,
                dropout_sec_input=0.5,
                ########## bilinear ##########
                bilinear_dim=-1,  # not use bilinear
                act='CeLU',
                atttype='scatt',  # scatt, basicatt,
                att_dim=1024,
                bifeat_emb_act='ReLU',
                decode_att_mid_dim=[128, 64, 128],
                decode_att_mid_dropout=0.1,
                decode_bifeat_emb_dropout=0.3,
                decode_block='LowRankBilinearDec',
                decode_dropout=0.5,
                decode_ff_dropout=0.1,
                decode_layers=1,
                elu_alpha=1.3,
                encode_att_mid_dim=[128, 64, 128],
                encode_att_mid_dropout=0.1,
                encode_bifeat_emb_dropout=0.3,
                encode_block='LowRankBilinearEnc',
                encode_dropout=0.5,
                encode_ff_dropout=0.1,
                encode_layers=4,
                head=8,
                type='LowRank')

        ),
        bbox_roi_extractor=dict(
            type='NormalExtractor',
            bbox_roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            in_channels=256,
            fc_out_channels=4096,
            featmap_strides=[4, 8, 16, 32],
            single=True,
            num_fcs=2,
            spatial_cfg=None),
        relation_roi_extractor=dict(
            type='NormalExtractor',
            bbox_roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            in_channels=256,
            fc_out_channels=512,
            featmap_strides=[4, 8, 16, 32],
            single=False,
            num_fcs=2,
            spatial_cfg=dict(type='fc', fc_in_dim=6, fc_out_dim=64)),
        relation_sampler=dict(
            pos_iou_thr=0.5,
            num_sample_per_gt_rel=4,
            num_rel_per_image=64,
            label_match=False,
            test_overlap=False),
        loss_relcaption=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.3),
        loss_caption=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
)
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=30)
)
# dataset settings
dataset_type = 'GeneralizedVisualGenomeDataset'
data_root = 'data/visualgenomegn/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_attr=True, with_rel=True, with_relcap=True, with_cap=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_attrs', 'gt_rels', 'gt_relmaps',
                               'gt_rel_inputs', 'gt_rel_targets',
                               'gt_rel_ipt_scores', 'gt_cap_inputs', 'gt_cap_targets']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_attr=True, with_rel=True, with_relcap=True, with_cap=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_attrs', 'gt_rels', 'gt_relmaps',
                                       'gt_rel_inputs', 'gt_rel_targets',
                                       'gt_rel_ipt_scores', 'gt_cap_inputs', 'gt_cap_targets']),
        ])
]
data = dict(
    imgs_per_gpu=12,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        roidb_file=data_root + 'VGGN-SGG.h5',
        dict_file=data_root + 'VGGN-SGG-dicts.json',
        image_file=data_root + 'meta_form.csv',
        pipeline=train_pipeline,
        num_im=-1,
        num_val_im=1000,
        split='train',
        split_type='filter_coco_karpathycap_testval_split',
        img_prefix=data_root + 'Images/',
        filter_empty_caps=True,
        filter_non_overlap=False),
    val=dict(
        type=dataset_type,
        roidb_file=data_root + 'VGGN-SGG.h5',
        dict_file=data_root + 'VGGN-SGG-dicts.json',
        image_file=data_root + 'meta_form.csv',
        pipeline=test_pipeline,
        num_im=-1,
        num_val_im=1000,
        split='val',
        split_type='filter_coco_karpathycap_testval_split',
        ann_file=data_root + 'VGGN-SGG-sentences.json',
        img_prefix=data_root + 'Images/',
        filter_empty_caps=True,
        filter_non_overlap=False),
    test=dict(
        type=dataset_type,
        roidb_file=data_root + 'VGGN-SGG.h5',
        dict_file=data_root + 'VGGN-SGG-dicts.json',
        image_file=data_root + 'meta_form.csv',
        pipeline=test_pipeline,
        num_im=5000,
        split='test',
        split_type='filter_coco_karpathycap_testval_split',
        ann_file=data_root + 'VGGN-SGG-sentences.json',
        img_prefix=data_root + 'Images/',
        filter_empty_caps=True,
        filter_non_overlap=False))
find_unused_parameters=True
evaluation = dict(interval=100, relcaption_mode=True, min_overlaps=[0.2, 0.3, 0.4, 0.5, 0.6],
                  min_scores=[-1, 0, 0.05, 0.1, 0.15, 0.2, 0.25], topN=[-1, 2, 5])
# optimizer
optimizer = dict(type='Adam', lr=0.0005, betas=[0.9, 0.98], eps=1e-9, weight_decay=0.0000,
                 freeze_modules=['backbone', 'neck', 'rpn_head', 'bbox_head', 'attr_head'])
optimizer_config = dict(grad_clip=dict(max_norm=0.5, norm_type=2))
# for the order of hook
lr_first = False
# learning policy
lr_config = dict(
    policy='noam',
    warmup=10000,
    factor=1.0,
    model_size=1024,
    step_type='iter')
checkpoint_config = dict(interval=1, save_scheduler=True)

# yapf:enable
# runtime settings
total_epochs = 30
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './experiments/iccv21/VGGN_updowncrossattn_predcls_Xcroatt_relcap0.3_faster_rcnn_x101_64x4d_fpn_1x'
load_from = 'experiments/VGGN_Detection_faster_rcnn_x101_64x4d_fpn_1x/latest.pth'
resume_from = None
resume_config = dict(resume_scheduler=True)
workflow = [('train', 1), ('val', 1)]

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 project=work_dir.split('/')[-1],
                 name='train-1',
                 config=work_dir+'/cfg.yaml'))
    ])
