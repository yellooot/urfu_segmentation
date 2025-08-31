import argparse
import math
import os
import pandas as pd
import cv2

from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

from mmseg.apis import MMSegInferencer

TREES_PIXEL_VALUE = 128
BACKGROUND_CLASS_IDX = 0
TREES_CLASS_IDX = 1

PALETTE = [
    [0, 0, 0],
    [128, 128, 128],
]

dfc_path = Path('/misc/home6/m_imm_freedata/Segmentation/Projects/mmseg_trees/Trees_DFC_512/val')
minifrance_path = Path('/misc/home6/m_imm_freedata/Segmentation/Projects/mmseg_trees/MiniFrance/val')

batch_size = 20

datasets = [
    {'name': 'dfc', 'images': dfc_path / 'images', 'gt': dfc_path / 'gt'},
    {'name': 'mini_france', 'images': minifrance_path / 'images', 'gt': minifrance_path / 'gt'},
]

def find_config_path(experiment_path: Path) -> Path:
    return next(experiment_path.glob("*.py"))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path")
    parser.add_argument("-V", "--visualize", required=False, default=False, action="store_true")
    parser.add_argument("-e", "--experiment-path")
    parser.add_argument("--device", required=False, default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    experiment_path = Path(args.experiment_path)
    config_path = str(find_config_path(experiment_path))
    weights_path = (experiment_path / 'last_checkpoint').read_text()

    inferencer = MMSegInferencer(
        model=config_path,
        weights=weights_path,
        device=args.device,
    )
    inferencer.show_progress = False

    results = {}

    for dataset in datasets:
        dataset_name = dataset['name']
        images_path, gt_path = dataset['images'], dataset['gt']

        images = [str(path) for path in images_path.glob("*.tif")]
        print(f'Inferencing {dataset_name}. Images {len(images)}')

        # --- инференс батчами ---
        predictions = []
        for i in range(math.ceil(len(images) / batch_size)):
            batch_images = images[i * batch_size:(i + 1) * batch_size]
            inference_result = inferencer(batch_images, batch_size=batch_size)
            predictions.extend(inference_result['predictions'])

        # --- подготовка GT ---
        print(f'Scoring {dataset_name}')
        gts = []
        for gt_image in gt_path.glob("*.tif"):
            gt = cv2.imread(str(gt_image), cv2.IMREAD_GRAYSCALE)
            gt[gt != TREES_PIXEL_VALUE] = BACKGROUND_CLASS_IDX
            gt[gt == TREES_PIXEL_VALUE] = TREES_CLASS_IDX
            gts.append(gt)

        # --- метрики ---
        acc_values = []
        iou_values = []
        for pred, gt in zip(predictions, gts):
            iou, acc = calculate_iou_and_accuracy(pred, gt)
            iou_values.append(iou)
            acc_values.append(acc)

        # --- визуализация ---
        if args.visualize:
            print(f'Visualizing {dataset_name}')
            for idx, (image_path, pred, gt) in enumerate(zip(images, predictions, gts)):
                image = Image.open(image_path)
                pred_image = Image.fromarray(pred.astype(np.uint8)).convert('P')
                pred_image.putpalette(np.array(PALETTE, dtype=np.uint8))

                plot_name = (
                    Path(image_path).stem
                    + f"_iou_{iou_values[idx]:.2f}_acc_{acc_values[idx]:.2f}.png"
                )
                save_path = output_path / 'visualization' / dataset_name / plot_name
                save_path.parent.mkdir(parents=True, exist_ok=True)

                plots = {
                    'image': image,
                    'predicted': np.array(pred_image),
                    'mask': gt,
                }
                write_plots_and_visualize(str(save_path), **plots)

                mask_save_path = (
                    output_path / 'masks' / dataset_name / (Path(image_path).stem + "_mask.tif")
                )
                mask_save_path.parent.mkdir(parents=True, exist_ok=True)
                pred_image.save(mask_save_path, format='TIFF')

        mIoU, mAcc = calculate_dataset_iou_and_accuracy(predictions, gts)
        results[dataset_name] = {
            'mAcc': mAcc,
            'mIoU': mIoU,
        }
        print(f'{dataset_name} results: {results[dataset_name]}')

    output_df = pd.DataFrame.from_dict(results, orient='index')
    print(output_df)
    output_df.to_csv(output_path / 'results.csv')


def calculate_dataset_iou_and_accuracy(predictions, gts, num_classes=2, eps=1e-8):
    """
    Считает mIoU и mAcc по всему датасету.
    
    predictions: список np.array с предсказанными масками
    gts: список np.array с ground truth масками
    """
    # инициализация сумм для каждого класса
    intersection_sum = np.zeros(num_classes, dtype=np.float64)
    union_sum = np.zeros(num_classes, dtype=np.float64)
    gt_sum = np.zeros(num_classes, dtype=np.float64)

    for pred, gt in zip(predictions, gts):
        for cls in range(num_classes):
            pred_cls = (pred == cls).astype(float)
            gt_cls = (gt == cls).astype(float)

            intersection = (pred_cls * gt_cls).sum()
            union = pred_cls.sum() + gt_cls.sum() - intersection

            intersection_sum[cls] += intersection
            union_sum[cls] += union
            gt_sum[cls] += gt_cls.sum()

    ious = intersection_sum / (union_sum + eps)
    accs = intersection_sum / (gt_sum + eps)

    mIoU = np.mean(ious)
    mAcc = np.mean(accs)

    return mIoU, mAcc



def calculate_iou_and_accuracy(pred, gt, num_classes=2, eps=1e-8):
    ious = []
    accs = []

    for cls in range(num_classes):
        pred_cls = (pred == cls).astype(float)
        gt_cls = (gt == cls).astype(float)

        intersection = (pred_cls * gt_cls).sum()
        union = pred_cls.sum() + gt_cls.sum() - intersection
        iou = intersection / (union + eps)
        acc = intersection / (gt_cls.sum() + eps)

        ious.append(iou)
        accs.append(acc)

    mIoU = np.mean(ious)
    mAcc = np.mean(accs)
    return mIoU, mAcc

def write_plots_and_visualize(path_to_res, **images):
    n = len(images)
    plt.figure(figsize=(16, 5))

    for i, (name, image) in enumerate(images.items()):
        if image is not None:
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)

    plt.savefig(path_to_res, bbox_inches='tight')

if __name__ == "__main__":
    main()
