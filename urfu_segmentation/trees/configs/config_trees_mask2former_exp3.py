_base_ = [
    '../../../configs/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512.py',
]

dataset_type = 'TreesDataset'

data_root = '/misc/home6/m_imm_freedata/Segmentation/Projects/mmseg_trees/Merge_TreesDFC512_MiniFrance'

num_classes = 2
batch_size = 20
num_workers = 12
auto_scale_lr = dict(enable=True, base_batch_size=16)

model = dict(
    decode_head=dict(
        num_classes=num_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])
    )
)


experiment_name = f'Mask2_{dataset_type}_Merge_TreesDFC512_MiniFrance_bsize_{batch_size}'
logs_dir = 'logs'
work_dir = f'{logs_dir}/{experiment_name}'
log_interval = 10


crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomChoiceResize',
        scales=[int(512 * x * 0.1) for x in range(6, 14)],
        resize_type='ResizeShortestEdge',
        max_size=2048),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=30, seg_pad_val=0),
    dict(type='RandomCutOut', n_holes=(2, 6), cutout_ratio=(0.05, 0.15), seg_fill_in=0, prob=0.4),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]


val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]


train_dataloader = dict(
    pin_memory=True,
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/images',
            seg_map_path='train/gt'),
        pipeline=train_pipeline,
        )
    )

val_dataloader = dict(
    pin_memory=True,
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/images',
            seg_map_path='val/gt'),
        pipeline=val_pipeline,
    )
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

custom_imports = dict(
    imports=['mmdet.models', 'mmengine.runner'], 
    allow_failed_imports=False
)

test_dataloader = val_dataloader
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend', scalar_save_file='../../scalars.json', save_dir=work_dir),
                dict(type='TensorboardVisBackend', save_dir=work_dir)]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')