#!/bin/sh

source ../.venv/bin/activate


NODE_PARAMS="-p hiperf --gres=gpu:a100:1 -t 05:00:00"
DEVICE="cuda"

EXPERIMENT_PATH="./logs/Mask2_TreesDataset_TreesDFC512_bsize_40"

sbatch -n1 \
    --cpus-per-task=12 \
    --mem=45000 \
    $NODE_PARAMS \
    --job-name="mmsegm-trees-eval-exp4-dfc-minifrance" \
    --wrap="python ./evaluate.py \
            --output-path=./eval_exp4_dfc_minifrance \
            --experiment-path=$EXPERIMENT_PATH \
            --device=$DEVICE"