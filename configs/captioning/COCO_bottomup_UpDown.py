# dataset settings
dataset_type = 'CaptionCocoDataset'
data_root = 'data/caption_coco/'
ann_root = data_root + 'karpathy_captions/'
feat_root = data_root + 'bottomup/up_down_10_100/'

train_pipeline = [
    dict(type='LoadCaptionVisuals', with_gv_feat=True, with_att_feat=True, with_bboxes=False),
    dict(type='LoadCaptionAnnotations', with_input_seq=True, with_target_seq=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['att_feats', 'gv_feat', 'input_seq', 'target_seq'], meta_keys=('height', 'width', 'coco_id', 'num_att')),
]
test_pipeline = [
    dict(type='LoadCaptionVisuals', with_gv_feat=True, with_att_feat=True, with_bboxes=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['att_feats', 'gv_feat'], meta_keys=('height', 'width', 'coco_id', 'num_att')),
]
data = dict(
    imgs_per_gpu=6, # * 2 gpus, bs=12,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        image_ids_path=ann_root + 'coco_train_image_id.txt',
        box_size_path=feat_root + 'train/boxsize/',
        pipeline=train_pipeline,
        input_seq=ann_root + 'coco_train_input.pkl',
        target_seq=ann_root + 'coco_train_target.pkl',
        att_feats_prefix=feat_root + 'train/feature/',
        seq_per_img=5,
        max_feat_num=-1,
        num_img=-1,
        split='train'),
    val=dict(
        type=dataset_type,
        image_ids_path=ann_root + 'coco_val_image_id.txt',
        box_size_path=feat_root + 'val/boxsize/',
        pipeline=test_pipeline,
        input_seq=ann_root + 'coco_val_input.pkl',
        target_seq=ann_root + 'coco_val_target.pkl',
        att_feats_prefix=feat_root + 'val/feature/',
        seq_per_img=5,
        max_feat_num=-1,
        num_img=-1,
        ann_file=ann_root + 'captions_val5k.json',
        split='val'),
    test=dict(
        type=dataset_type,
        image_ids_path=ann_root + 'coco_test_image_id.txt',
        box_size_path=feat_root + 'test/boxsize/',
        pipeline=test_pipeline,
        input_seq=ann_root + 'coco_test_input.pkl',
        target_seq=ann_root + 'coco_test_target.pkl',
        att_feats_prefix=feat_root + 'test/feature/',
        seq_per_img=5,
        max_feat_num=-1,
        ann_file=ann_root + 'captions_test5k.json',
        split='test'))
# model settings
param = dict(
    att_feats='ATT_FEATS',
    att_feats_mask='ATT_FEATS_MASK',
    global_feat='GV_FEAT',
    indices='INDICES',
    input_sent='INPUT_SENT',
    p_att_feats='P_ATT_FEATS',
    state='STATE',
    target_sent='TARGET_SENT',
    wt='WT')
model = dict(
    type='UpDownCaptioner',
    #pretrained='checkpoints/mmlab/imnet/resnext101_64x4d-ee2c6f71.pth',
    seq_len=17,      # include <EOS>/<BOS>
    seq_per_img=5,
    vocab_size=9487,
    vocab=ann_root + 'coco_vocabulary.txt',
    param_config=param,
    word_embed_config=dict(
        word_embed_dim=1024,
        word_embed_act='CeLU',
        word_embed_norm=False,
        dropout_word_embed=0.5),
    ########## global features ##########
    global_feat_config=dict(
        gvfeat_dim=2048,
        gvfeat_embed_dim=-1,
        gvfeat_embed_act=None,
        dropout_gv_embed=0.0),
    attention_feat_config=dict(
        att_feats_dim=2048,
        att_feats_embed_dim=1024,
        att_feats_embed_act='CeLU',
        dropout_att_embed=0.5,
        att_feats_norm=False,
        att_hidden_size=1024,
        att_hidden_drop=0.5,
        att_act='Tanh',
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
        type='LowRank'),
    loss_xe=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0))

find_unused_parameters=True
# yapf:enable
# runtime settings
total_epochs = 10 #60
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './experiments/COCO_bottomup_UpDown/train_2ndstage_Adam_poly'
evaluation = dict(interval=5, beam_size=3, greedy_decode=True)
load_from = './experiments/COCO_bottomup_UpDown/epoch_10.pth'
resume_from = None
#resume_config = dict(resume_scheduler=True)
workflow = [('train', 1), ('val', 1)]

# optimizer
#optimizer = dict(type='Adam', lr=0.0005, betas=[0.9, 0.98], eps=1e-9, weight_decay=0.0000)
optimizer = dict(type='Adam', lr=0.00001, betas=[0.9, 0.98], eps=1e-9, weight_decay=0.0000)

optimizer_config = dict(grad_clip=dict(max_norm=0.5, norm_type=2))
# learning policy
# lr_config = dict(
#     policy='noam',
#     warmup=10000,
#     factor=1.0,
#     model_size=1024,
#     step_type='iter')
lr_config = dict(
    policy='poly',
    by_epoch=False,
    power=2,
)
checkpoint_config = dict(interval=5)
#checkpoint_config = dict(interval=5, save_scheduler=True)
sampling_schedule_config = dict(start=6, inc_every=5, inc_prob=0.05, max_prob=0.5)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 project='captioning_COCO_bottomup_UpDown',
                 name='train_2ndstage_Adam_poly',
                 config=work_dir+'/cfg.yaml'))
    ])
