_base_ = '../configs/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py'
model = dict(
    bbox_head=dict(
        type='VFNetHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)))

# dataset settings
dataset_type = 'CocoDataset'
classes = ('opacity', )

# yapf:disable
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# yapf:enable

# Albumentation training transform settings
albu_train_transforms = [
    dict(type='ShiftScaleRotate', rotate_limit=25, p=0.80),
    dict(
        type="RandomBrightnessContrast",
        p=0.80,
        brightness_limit=0.25,
        contrast_limit=0.25),
    dict(type='HorizontalFlip', p=0.5),
]

# Data Input Pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        keymap=dict(img="image", gt_bboxes="bboxes"),
        update_pad_shape=False,
        skip_img_without_anno=True,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=['gt_labels'],
            filter_lost_elements=True,
            min_visibility=0.1,
            min_area=1)),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Dataset Settings
# yapf:disable
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        img_prefix='/home/shivgup/ayush/mmdetection/train/',
        classes=classes,
        ann_file='/home/shivgup/ayush/mmdetection/kaggle/coco_annotations/train_annotations_fold1.json',
        pipeline=train_pipeline,
        filter_empty_gt=True),
    val=dict(
        img_prefix='/home/shivgup/ayush/mmdetection/train/',
        classes=classes,
        ann_file='/home/shivgup/ayush/mmdetection/kaggle/coco_annotations/val_annotations_fold1.json',
        pipeline=test_pipeline,
        filter_empty_gt=True),
    test=dict(
        img_prefix='/home/shivgup/ayush/mmdetection/train/',
        classes=classes,
        ann_file='/home/shivgup/ayush/mmdetection/kaggle/coco_annotations/val_annotations_fold1.json',
        pipeline=test_pipeline,
        filter_empty_gt=True))
# yapf:enable
# optimizer
optimizer = dict(lr=0.01 * 8 / 16)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.001,
    min_lr=0.0)

log_config = dict(interval=30, hooks=[dict(type='TextLoggerHook')])
checkpoint_config = dict(interval=30)
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')

total_epochs = 50
runner = dict(type='EpochBasedRunner', max_epochs=50)

load_from = 'https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_1x_coco/vfnet_r101_fpn_1x_coco_20201027pth-c831ece7.pth'
work_dir = 'runs/vfnet_r50_augs_without_empty_fold1'

log_level = 'INFO'

# fp16 settings
#fp16 = dict(loss_scale=512.)