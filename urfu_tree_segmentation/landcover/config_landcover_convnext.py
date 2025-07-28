# для выбора модели, расписания необходимо наследовать один из файлов из репозитория mmsegmentation
# базовые файлы для наследования можно посмотреть по пути mmsegmentation/configs/_base_/
_base_ = [
    '../../configs/_base_/models/upernet_convnext.py',
    '../../configs/_base_/default_runtime.py',
    # '../../configs/_base_/schedules/schedule_160k.py'
]

# ----------------------------------------------------------------
# Изменение гиперпараметров

# Поскольку мы используем только один графический процессор, вместо SyncBN используется BN
norm_cfg = dict(type='BN', requires_grad=True)

# Название датасета из файла urfu_project/dataset.py
dataset_type = 'LandcoverAI'

# Путь к папке с преобразованным набором данных
data_root = '/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512'

# Количество классов для сегментации
num_classes = 2

# Размер изображения, который принимает на вход сеть
crop_size = (512, 512)

# Количичество эпох для обучения
max_epochs = 100

# Функция потерь
loss = dict(type='FocalLoss', class_weight=[0.9, 1.1])

# Размер батча
batch_size = 16
gradient_accumulation_steps = 8
actual_batch_size = batch_size * gradient_accumulation_steps

# num_workers
num_workers = 8

# Оптимизатор
# optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0005)
optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)

# Параметры логирования 
experiment_name = f'ConvNext_{dataset_type}_{crop_size[0]}_{loss["type"]}_{optimizer["type"]}_bsize_{actual_batch_size}'
logs_dir = 'logs'
work_dir = f'{logs_dir}/{experiment_name}'  # директория для сохранения логов
log_interval = 10  # интервал в итерациях для печати логов

# Директория, где хрянятся файлы с списком изображений train и val
splits = 'splits'

# Имя эксперимента
# ----------------------------------------------------------------
# Параметры модели
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

data_preprocessor = dict(size=crop_size)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='ResizeToMultiple', size_divisor=32),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmpretrain.ConvNeXt',
        arch='small',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.3,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=150,
    ),
    auxiliary_head=dict(in_channels=384, num_classes=150),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)


# ----------------------------------------------------------------
# Параметры default_runtime
log_processor = dict(by_epoch=True)
# ----------------------------------------------------------------
# Параметры scheduler
# optimizer
optim_wrapper = dict(
    optimizer=optimizer,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    },
    constructor='LearningRateDecayOptimizerConstructor',
)
# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=1)
# В default_hooks draw=False для того, чтобы не выводить изображения из val с результатами модели после эпохи в логи
# Если поставить True, то увеличьте интервал до более высокого, чтоб не сохранял часто много изображений
default_hooks = dict(logger=dict(type='LoggerHook', log_metric_by_epoch=True, interval=log_interval),
                     checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, max_keep_ckpts=1, save_best='mIoU'),
                     visualization=dict(type='SegVisualizationHook', draw=False, interval=500))
# ----------------------------------------------------------------

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
        # ann_file=f'{splits}/train.txt'
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
        # ann_file=f'{splits}/val.txt'
        )
    )

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
# tta_pipeline = [  # Test Time Augmentation (TTA)
#     dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
#     dict(
#         type='TestTimeAug',
#         transforms=[
#             [
#                 dict(type='Resize', scale_factor=r, keep_ratio=True)
#                 for r in img_ratios
#             ],
#             [
#                 dict(type='RandomFlip', prob=0., direction='horizontal'),
#                 dict(type='RandomFlip', prob=1., direction='horizontal')
#             ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
#         ])
# ]
# ----------------------------------------------------------------
# Tensorboard visualization
vis_backends = [dict(type='LocalVisBackend', scalar_save_file='../../scalars.json', save_dir=work_dir),
                dict(type='TensorboardVisBackend', save_dir=work_dir)]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# ----------------------------------------------------------------
