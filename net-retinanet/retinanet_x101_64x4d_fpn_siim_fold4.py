model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=None,
        groups=64,
        base_width=4),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.1,
                rotate_limit=20,
                scale_limit=0.2,
                p=0.6),
            dict(
                type='RandomBrightnessContrast',
                p=0.7,
                brightness_limit=0.25,
                contrast_limit=0.25),
            dict(type='IAAAffine', shear=(-10.0, 10.0), p=0.4),
            dict(type='Blur', p=1.0, blur_limit=7),
            dict(type='CLAHE', p=0.5),
            dict(type='Equalize', mode='cv', p=0.4),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='GaussianBlur', p=1.0, blur_limit=7),
                    dict(type='MedianBlur', p=1.0, blur_limit=7)
                ],
                p=0.4)
        ],
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            filter_lost_elements=True,
            min_visibility=0.1,
            min_area=1)),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type='CocoDataset',
        ann_file=
        'data/annotations/coco/train_annotations_fold4.json',
        img_prefix='data/images/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.1,
                        rotate_limit=20,
                        scale_limit=0.2,
                        p=0.6),
                    dict(
                        type='RandomBrightnessContrast',
                        p=0.7,
                        brightness_limit=0.25,
                        contrast_limit=0.25),
                    dict(type='IAAAffine', shear=(-10.0, 10.0), p=0.4),
                    dict(type='Blur', p=1.0, blur_limit=7),
                    dict(type='CLAHE', p=0.5),
                    dict(type='Equalize', mode='cv', p=0.4),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='GaussianBlur', p=1.0, blur_limit=7),
                            dict(type='MedianBlur', p=1.0, blur_limit=7)
                        ],
                        p=0.4)
                ],
                keymap=dict(img='image', gt_bboxes='bboxes'),
                update_pad_shape=False,
                skip_img_without_anno=True,
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    filter_lost_elements=True,
                    min_visibility=0.1,
                    min_area=1)),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=1),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=('opacity', ),
        filter_empty_gt=True),
    val=dict(
        type='CocoDataset',
        ann_file=
        'data/annotations/coco/val_annotations_fold4.json',
        img_prefix='data/images/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('opacity', ),
        filter_empty_gt=True),
    test=dict(
        type='CocoDataset',
        ann_file=
        'data/annotations/coco/val_annotations_fold4.json',
        img_prefix='data/images/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('opacity', ),
        filter_empty_gt=True))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')
optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=0.0)
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=10,
    hooks=[dict(type='TensorboardLoggerHook'),
           dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth'
resume_from = None
workflow = [('train', 1)]
classes = ('opacity', )
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.1,
        rotate_limit=20,
        scale_limit=0.2,
        p=0.6),
    dict(
        type='RandomBrightnessContrast',
        p=0.7,
        brightness_limit=0.25,
        contrast_limit=0.25),
    dict(type='IAAAffine', shear=(-10.0, 10.0), p=0.4),
    dict(type='Blur', p=1.0, blur_limit=7),
    dict(type='CLAHE', p=0.5),
    dict(type='Equalize', mode='cv', p=0.4),
    dict(
        type='OneOf',
        transforms=[
            dict(type='GaussianBlur', p=1.0, blur_limit=7),
            dict(type='MedianBlur', p=1.0, blur_limit=7)
        ],
        p=0.4)
]
fp16 = dict(loss_scale=512.0)
total_epochs = 20
work_dir = 'runs/retinanet_x101_64x4d_fpn_without_empty/fold4'
gpu_ids = range(0, 1)
