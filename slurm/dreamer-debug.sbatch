#!/bin/bash
#SBATCH --job-name=dreamer-test
#SBATCH --partition=short
#SBATCH --reservation=a2i2026-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -e
cd "$DATA/world_models_paper"

srun .venv/bin/python -u scripts/train_dreamer.py \
  --name "arc-${SLURM_JOB_ID}" \
  --backend cuda \
  --debug \
  --buffer-size 8
