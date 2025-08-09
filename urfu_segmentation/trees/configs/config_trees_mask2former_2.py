_base_ = [
    '../../../configs/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512.py',
]


########### for optim_wrapper


pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'  # noqa

depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        depths=depths, init_cfg=dict(type='Pretrained',
                                     checkpoint=pretrained)))

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})


################


seed = 887

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=16000,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=144000,  # 160000 - 16000 (после warmup)
        eta_min=1e-7,
        begin=16000,
        end=160000,
        by_epoch=False,
    )
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=160000,
    val_interval=5000
)
checkpoint_config = dict(interval=5000, by_epoch=False, max_keep_ckpts=5)


dataset_type = 'TreesDataset'

data_root = '/misc/home6/m_imm_freedata/Segmentation/Projects/mmseg_trees/Trees_DFC_512'

num_classes = 2

crop_size = (512, 512)

loss = [
    dict(type='FocalLoss', class_weight=[0.9, 1.1], loss_weight=0.7),
    dict(type='DiceLoss', loss_weight=0.3)
]

batch_size = 16
gradient_accumulation_steps = 4
actual_batch_size = batch_size * gradient_accumulation_steps

num_workers = 12


experiment_name = f'Mask2_{dataset_type}_{crop_size[0]}_{"_".join([loss_["type"] for loss_ in loss])}_bsize_{actual_batch_size}'
logs_dir = 'logs'
work_dir = f'{logs_dir}/{experiment_name}'
log_interval = 10


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.001),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0),
    accumulative_counts=gradient_accumulation_steps,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),

    dict(
        type='RandomChoiceResize',
        scales=[int(crop_size[0] * s / 10) for s in range(5, 25)],
        resize_type='ResizeShortestEdge',
        max_size=4096),

    dict(type='RandomRotate', degree=20, prob=0.5),

    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),

    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),

    dict(type='PhotoMetricDistortion'),

    dict(type='RandomCutOut', n_holes=6, cutout_shape=(64, 64), prob=0.4),

    dict(type='PackSegInputs')
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]


train_dataloader = dict(
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
        pipeline=test_pipeline,
        )
    )

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend', scalar_save_file='../../scalars.json', save_dir=work_dir),
                dict(type='TensorboardVisBackend', save_dir=work_dir)]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')