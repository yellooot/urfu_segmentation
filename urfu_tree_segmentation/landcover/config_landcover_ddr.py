# для выбора модели, расписания необходимо наследовать один из файлов из репозитория mmsegmentation
# базовые файлы для наследования можно посмотреть по пути mmsegmentation/configs/_base_/
_base_ = [
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
class_weight = [
    0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786,
    1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
    1.0507
]
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/ddrnet/pretrain/ddrnet23s-in1kpre_3rdparty-1ccac5b1.pth'  # noqa
crop_size = (1024, 1024)

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
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)

# Параметры логирования 
experiment_name = f'DDR_{dataset_type}_{crop_size[0]}_{loss["type"]}_{optimizer["type"]}_bsize_{actual_batch_size}'
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

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='ResizeToMultiple', size_divisor=32),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DDRNet',
        in_channels=3,
        channels=32,
        ppm_channels=128,
        norm_cfg=norm_cfg,
        align_corners=False,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(
        type='DDRHead',
        in_channels=32 * 4,
        channels=64,
        dropout_ratio=0.,
        num_classes=19,
        align_corners=False,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=0.4),
        ]),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

iters = 120000
# ----------------------------------------------------------------
# Параметры default_runtime
log_processor = dict(by_epoch=False)
# ----------------------------------------------------------------
# Параметры scheduler
# optimizer
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=iters, val_interval=iters // 10)
# В default_hooks draw=False для того, чтобы не выводить изображения из val с результатами модели после эпохи в логи
# Если поставить True, то увеличьте интервал до более высокого, чтоб не сохранял часто много изображений
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=iters // 10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

randomness = dict(seed=304)
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
