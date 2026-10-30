#!/bin/bash
#SBATCH --job-name=dreamer-long
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_sku:H100|gpu_sku:L40S|gpu_sku:A100"
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -e
cd "$DATA/world_models_paper"

srun .venv/bin/python -u scripts/train_dreamer.py \
  --name "arc-${SLURM_JOB_ID}-long" \
  --backend cuda \
  --collect-interval 2000 \
  --steps-per-eval 10 \
  --steps-per-checkpoint 5 \
  --num-eval-episodes 64 \
  --cfg.actor-entropy-weight 0.0005 \
  --seq-length 64 \
  --batch-size 32 \
  --outer-train-steps 200 \
  --cfg.dtype bfloat16
