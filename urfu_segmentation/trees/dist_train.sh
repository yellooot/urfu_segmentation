#!/bin/sh

source ../.venv/bin/activate

GPUS=2
NODE_PARAMS="-p hiperf --gres=gpu:a100:$GPUS --nodelist=tesla-a101 -t 20:00:00"

sbatch -n1 \
    --cpus-per-task=12 \
    --mem=45000 \
    $NODE_PARAMS \
    --job-name=mmsegm-trees \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS} \
    --wrap="srun python ./train.py ./configs/config_trees_mask2former_exp4.py --resume --launcher slurm"
