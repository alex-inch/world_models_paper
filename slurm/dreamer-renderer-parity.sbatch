#!/bin/bash
#SBATCH --job-name=renderer-parity
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_sku:L40S"
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/dreamer-renderer-parity-%j.out

set -euo pipefail
cd "$DATA/world_models_paper"

srun .venv/bin/python -u scripts/check_dreamer_renderer.py \
  --backend cuda \
  --checkpoint experiments/dreamer/checkpoints/fh6igknc/00022000 \
  --output-dir "experiments/dreamer/diagnostics/renderer-parity-${SLURM_JOB_ID}"
