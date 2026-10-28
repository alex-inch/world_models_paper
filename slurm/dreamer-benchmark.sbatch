#!/bin/bash
#SBATCH --job-name=dreamer-bench
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/dreamer-benchmark-%j.out

set -euo pipefail
cd "$DATA/world_models_paper"

OUTPUT="experiments/dreamer/benchmarks/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT"

# Preserve the training job's thread settings for the first diagnostic run.
# A repeat with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# can test oversubscription once the recorded worker affinity/CPU data is known.
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1

MONITOR_PID=""
cleanup() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,power.draw \
    --format=csv -l 1 > "$OUTPUT/gpu.csv" 2> "$OUTPUT/gpu-monitor.err" &
  MONITOR_PID=$!
fi

srun .venv/bin/python -u scripts/benchmark_dreamer.py \
  --backend cuda \
  --output-dir "$OUTPUT" \
  "$@"
