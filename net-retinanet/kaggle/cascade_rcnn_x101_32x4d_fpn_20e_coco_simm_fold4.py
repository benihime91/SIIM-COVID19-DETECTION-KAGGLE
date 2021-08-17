_base_ = '/home/shivgup/mmdetection/mmdetection/configs/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_20e_coco.py'
model = dict(
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
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
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
# dataset settings
dataset_type = 'CocoDataset'
classes = ('opacity', )

# yapf:disable
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# yapf:enable

# Albumentation training transform settings
#albu_train_transforms = [
#    dict(type='ShiftScaleRotate', rotate_limit=25, p=0.80),
#    dict(
#        type="RandomBrightnessContrast",
#        p=0.80,
#        brightness_limit=0.25,
#        contrast_limit=0.25),
#    dict(type='HorizontalFlip', p=0.5),
#]

albu_train_transforms = [
    dict(type='ShiftScaleRotate', shift_limit=0.0625,
         scale_limit=0.15, rotate_limit=15, p=0.7),
    dict(type='RandomBrightnessContrast', brightness_limit=0.2,
         contrast_limit=0.2, p=0.7),
    dict(type='IAAAffine', shear=(-10.0, 10.0), p=0.7),
#     dict(type='MixUp', p=0.2, lambd=0.5),
    dict(type="Blur", p=0.7, blur_limit=7),
    #dict(type='CLAHE', p=0.5),
    dict(type='Equalize', mode='cv', p=0.7),
    dict(type='HorizontalFlip', p=0.5),
    dict(
        type="OneOf",
        transforms=[
            dict(type="GaussianBlur", p=1.0, blur_limit=7),
            dict(type="MedianBlur", p=1.0, blur_limit=7),
        ],
        p=0.7,
    ),
    
#     dict(type='MixUp', p=0.2, lambd=0.5),
#     dict(type='RandomRotate90', p=0.5),
#     dict(type='CLAHE', p=0.5),
#     dict(type='InvertImg', p=0.5),
#     dict(type='Equalize', mode='cv', p=0.4),
#     dict(type='MedianBlur', blur_limit=3, p=0.1)
    ]

# Data Input Pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
        type='BboxParams',
        format='pascal_voc',
        label_fields=['gt_labels'],
        min_visibility=0.0,
        filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
#train_pipeline = [
#    dict(type='LoadImageFromFile'),
#    dict(type='LoadAnnotations', with_bbox=True),
#    dict(
#        type='Albu',
#        transforms=albu_train_transforms,
#        keymap=dict(img="image", gt_bboxes="bboxes"),
#        update_pad_shape=False,
#        skip_img_without_anno=True,
#        bbox_params=dict(
#            type="BboxParams",
#            format="pascal_voc",
#            label_fields=['gt_labels'],
#            filter_lost_elements=True,
#            min_visibility=0.1,
#            min_area=1)),
#    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
#    dict(type='RandomFlip', flip_ratio=0.0),
#    dict(type='Normalize', **img_norm_cfg),
#    dict(type='Pad', size_divisor=1),
#    dict(type='DefaultFormatBundle'),
#    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
#]

#test_pipeline = [
#    dict(type='LoadImageFromFile'),
#    dict(
#        type='MultiScaleFlipAug',
#        img_scale=(1024, 1024),
#        flip=False,
#        transforms=[
#            dict(type='Resize', keep_ratio=True),
#            dict(type='RandomFlip'),
#            dict(type='Normalize', **img_norm_cfg),
#            dict(type='Pad', size_divisor=32),
#            dict(type='ImageToTensor', keys=['img']),
#            dict(type='Collect', keys=['img']),
#        ])
#]

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


# Dataset Settings
# yapf:disable
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        img_prefix='/home/shivgup/yolo/train/',
        classes=classes,
        ann_file='/home/shivgup/mmdetection/mmdetection/kaggle/coco_annotations/train_annotations_fold4.json',
        pipeline=train_pipeline,
        filter_empty_gt=True),
    val=dict(
        img_prefix='/home/shivgup/yolo/train/',
        classes=classes,
        ann_file='/home/shivgup/mmdetection/mmdetection/kaggle/coco_annotations/val_annotations_fold4.json',
        pipeline=test_pipeline,
        filter_empty_gt=True),
    test=dict(
        img_prefix='/home/shivgup/yolo/train/',
        classes=classes,
        ann_file='/home/shivgup/mmdetection/mmdetection/kaggle/coco_annotations/val_annotations_fold4.json',
        pipeline=test_pipeline,
        filter_empty_gt=True))
# yapf:enable
# optimizer
#optimizer = dict(lr=0.01 * 8 / 16)
#optimizer_config = dict(grad_clip=None)
optimizer = dict(lr=0.01/8)
# learning policy
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=700,
    warmup_ratio=0.001,
    min_lr=1e-07)



log_config = dict(interval=200, hooks=[dict(type='TextLoggerHook')])
checkpoint_config = dict(interval=30)
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')

total_epochs = 15
runner = dict(type='EpochBasedRunner', max_epochs=15)

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_20e_coco/cascade_rcnn_x101_32x4d_fpn_20e_coco_20200906_134608-9ae0a720.pth'
work_dir = 'runs/cascade_rcnn_x101_32x4d_fpn_20e_coco_fold4_new'

log_level = 'INFO'

# fp16 settings
#fp16 = dict(loss_scale=512.)