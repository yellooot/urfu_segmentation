# для выбора модели, расписания необходимо наследовать один из файлов из репозитория mmsegmentation
# базовые файлы для наследования можно посмотреть по пути mmsegmentation/configs/_base_/
_base_ = [
    '../../../configs/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512.py',
]

load_from = "/home/s0171/_scratch2/urfu_segmentation/urfu_segmentation/trees/logs/Mask2_TreesDataset_512_FocalLoss_bsize_128/iter_160000.pth"
param_scheduler = [
    dict(
        type='LinearLR',  # warmup
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500,
    ),
    dict(
        type='PolyLR',  # основной decay
        power=0.9,
        eta_min=0.0,
        begin=1500,
        end=50000,  # max_iters из train_cfg
        by_epoch=False,
    )
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=50000,    # теперь всего 50k итераций
    val_interval=5000
)
checkpoint_config = dict(interval=5000, by_epoch=False, max_keep_ckpts=5)

# ----------------------------------------------------------------
# Изменение гиперпараметров

# Поскольку мы используем только один графический процессор, вместо SyncBN используется BN
norm_cfg = dict(type='BN', requires_grad=True)
# Название датасета из файла urfu_project/dataset.py
dataset_type = 'TreesDataset'
# Путь к папке с преобразованным набором данных
data_root = '/misc/home6/m_imm_freedata/Segmentation/Projects/mmseg_trees/Trees_DFC_512'
# Количество классов для сегментации
num_classes = 2
# Размер изображения, который принимает на вход сеть
crop_size = (512, 512)
# Функция потерь
loss = dict(type='FocalLoss', class_weight=[0.9, 1.1])
# Размер батча
batch_size = 16
gradient_accumulation_steps = 4
actual_batch_size = batch_size * gradient_accumulation_steps
# num_workers
num_workers = 12

# Параметры логирования 
experiment_name = f'Mask2_{dataset_type}_{crop_size[0]}_{loss["type"]}_bsize_{actual_batch_size}_tune'
logs_dir = 'logs'
work_dir = f'{logs_dir}/{experiment_name}'  # директория для сохранения логов
log_interval = 10  # интервал в итерациях для печати логов


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
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.001),
    # clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0),
    accumulative_counts=gradient_accumulation_steps
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomChoiceResize',
        scales=[int(512 * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=2048),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
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