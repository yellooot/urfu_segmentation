#!/bin/sh

source ../.venv/bin/activate

GPUS=2
NODE_PARAMS="-p hiperf --account=students --gres=gpu:a100:$GPUS --nodelist=tesla-a101 -t 00:30:00"

sbatch -n1 \
    --cpus-per-task=8 \
    --mem=45000 \
    $NODE_PARAMS \
    --job-name=mmsegm-debug-trees \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS} \
    --wrap="srun python ./train.py ./configs/config_trees_debug.py --launcher slurm"
