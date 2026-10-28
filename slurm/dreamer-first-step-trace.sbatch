#!/bin/bash
#SBATCH --job-name=dreamer-first-step
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_sku:L40S"
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/dreamer-first-step-%j.out

set -euo pipefail
cd "$DATA/world_models_paper"

srun .venv/bin/python -u scripts/trace_dreamer_first_step.py \
  --backend cuda \
  --stock-checkpoint experiments/dreamer/checkpoints/gp8l0xmb/00000000 \
  --optimized-checkpoint experiments/dreamer/checkpoints/fh6igknc/00000000 \
  --output-dir "experiments/dreamer/diagnostics/first-step-${SLURM_JOB_ID}" \
  "$@"
