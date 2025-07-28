# для выбора модели, расписания необходимо наследовать один из файлов из репозитория mmsegmentation
# базовые файлы для наследования можно посмотреть по пути mmsegmentation/configs/_base_/
_base_ = [
    '../../configs/san/san-vit-b16_coco-stuff164k-640x640.py',
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
crop_size = (640, 640)

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
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001)

# Параметры логирования 
experiment_name = f'SAN_{dataset_type}_{crop_size[0]}_{loss["type"]}_{optimizer["type"]}_bsize_{actual_batch_size}'
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

# data_preprocessor = dict(size=crop_size)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomChoiceResize',
        scales=[int(640 * x * 0.1) for x in range(5, 16)],
        resize_type='ResizeShortestEdge',
        max_size=2560),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeShortestEdge', scale=crop_size, max_size=2560),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# ----------------------------------------------------------------
# Параметры default_runtime
log_processor = dict(by_epoch=True)
# ----------------------------------------------------------------
# Параметры scheduler
# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=0.01, norm_type=2))
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=60000,
        by_epoch=False,
    )
]

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=60000,
    val_interval=500,
    val_begin=55000)
# В default_hooks draw=False для того, чтобы не выводить изображения из val с результатами модели после эпохи в логи
# Если поставить True, то увеличьте интервал до более высокого, чтоб не сохранял часто много изображений
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=10000,
        save_best='mIoU'))

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
