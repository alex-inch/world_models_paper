# Benchmarking Dreamer collection on ARC

From the repository root on ARC:

```bash
sbatch slurm/dreamer-benchmark.sbatch
```

The job requests one GPU, 16 CPUs, 32 GB RAM and a two-hour limit. It uses the
`short` partition, without a reservation or GPU-model constraint. Add the same
constraint as a training job when comparing hardware. It does not submit
training jobs, create W&B runs, or change training code/checkpoints.

Results are saved incrementally under
`experiments/dreamer/benchmarks/<job-id>/`. Start with `report.md`; keep
`results.json` and the Slurm log so we can inspect the underlying measurements.
`gpu.csv` samples GPU utilisation once a second. Case completion timestamps in
the JSON help align it with those samples. A killed or failed run retains results
from completed cases.

`train_dreamer.py` uses the existing optimized CarRacing renderer for collection
and in-run evaluation when `env_id` is `CarRacing-v3`. Checkpoint metadata keeps
the canonical `CarRacing-v3` ID, so the separate offline evaluation, inspection
and rendering scripts continue to construct stock CarRacing. Other environment
IDs pass through unchanged. The benchmark's `stock` cases still use stock
CarRacing for comparison.

## Order and questions answered

The lower-complexity candidates run before the full worker-count sweep:

1. **Font I/O:** compare the exact Pygame default-resource lookup, an absolute
   path to that same package font, a copy on node scratch, and an already loaded
   font. Font construction and text rendering are timed separately. These are
   warm-cache measurements; the benchmark does not flush filesystem caches.
2. **Eight-worker environment comparison:** identical seeds and fixed actions
   (`[0, 0.5, 0]`) with stock rendering, cached fonts only, the existing
   `ControllerTrainingCarRacing` renderer, and no rendering. The last case returns
   black observations and is a physics-cost diagnostic, not a proposed training
   environment. The existing renderer retains Box2D physics and the cropped model
   input; caching the font retains the complete stock observation.
3. **Policy and copies:** run the actual `choose_action` on device-resident
   inputs; time synchronized image upload, action download and recurrent-state
   download separately. Downloads use fresh device outputs so repeated NumPy
   conversions cannot accidentally measure a cached host copy. First compilation
   and warmup are reported separately.
4. **Learning and overlap:** call the actual `model_learning_step` with batch 16,
   sequence length 64, comparing per-update host synchronization with waiting at
   the end. Then compare `env.step → learn` against
   `env.step_async → learn → env.step_wait`, with the same steps, seeds, frozen
   acting weights and number of updates (1.6 per vector step, matching 1,600
   updates per 1,000 vector steps). Both stock and optimized renderers are tested.
   This tests the simple scheduling change, not Hafner's full independent
   actor/replay/learner architecture. Acting weights are separate from donated
   learner weights, and the learner uses valid synthetic replay episodes. The
   overlap loop omits replay insertion and rollout staging in both schedules to
   isolate scheduling; use the production collection cases for those costs.
5. **Production collection:** call the actual `collect_new_rollouts` with stock,
   cached-font and optimized renderers, including the original host staging and
   replay insertion. Default replay capacity is 100 episodes and collection is
   1,000 vector steps, as in training. This gives a direct collection speedup for
   the renderer changes, plus Python-stage timings and the fraction of interaction
   before each slot's first episode boundary.
6. **Worker scaling:** repeat environment-only cases with 1, 12 and 16 workers
   (8 was already measured). Compare **transitions/second**, since each worker
   count collects a different number of transitions per vector step. This sweep
   isolates simulation/IPC scaling; it does not predict full learner scaling.

Default runs repeat each comparison twice. Environment-only runs use 200 steps
per case, full collection uses 1,000, and overlap uses 100. Each repeat uses a
different deterministic seed. The overlap pair alternates execution order to
help expose drift. Other comparisons share the same trace across renderers but
do not randomize order; inspect individual repeats when cluster load varies.
Without `--checkpoint`, acting models use deterministic random initialization.
Supply a trained checkpoint to measure a representative driving workload:

```bash
sbatch slurm/dreamer-benchmark.sbatch \
  --checkpoint experiments/dreamer/checkpoints/RUN/STEP
```

## Reading the results

`report.md` contains median speedup comparisons followed by individual cases.
All speedups are measured within the stated scope: a font-only speedup is not a
whole-training speedup. `results.json` contains mean/p50/p95/p99/max durations,
worker CPU time and inherited CPU affinity, package versions/paths, filesystem
information, GPU identity, Slurm environment and the Git revision/status.
`stock-render.prof` and `stock-render.txt` contain a separate cProfile run;
profiling is excluded from the throughput results.

Worker timings are nested: `step` includes `render`; `render` includes `road`,
`dashboard`, `font_load` and `image_conversion`. Do not sum those columns.
`non_render` is step time minus rendering, including Python vehicle dynamics,
Box2D and reward logic. Worker timings are recorded locally and fetched after
the loop, avoiding telemetry on every IPC message. Instrumentation has some
overhead, shared by the renderer comparisons. Environment startup, warmup and
the environment-only reset are reported separately. Render totals also include
frames generated inside resets; worker `reset` timings expose those separately.

Production collection's `policy_dispatch` measures Python/JAX dispatch rather
than completed GPU compute. Its residual includes the waits caused by NumPy
conversion of actions/carry, key generation, allocations, host staging and other
unwrapped work. The isolated synchronized policy/copy tests help distinguish
those causes. Replay insertion includes a completion barrier and is warmed
before measurement. Collection's own reset and allocations remain inside its
wall time, as in production. The `learning` cases exclude W&B serialization and
network activity, so their synchronization comparison is not a full logging
benchmark. There are no quality-of-learning conclusions from synthetic replay.

Useful signals:

- Stock slow, cached-font fast, and package-path fonts slower than scratch:
  investigate package resource/filesystem access first.
- Stock and cached-font similar, optimized faster, road timing dominant:
  rendering/culling is the next candidate.
- No-render still slow: inspect the non-render remainder and CPU allocation.
- Individual worker steps fast but vector steps slow: IPC, CPU scheduling or a
  slowest-worker barrier may dominate. Low worker CPU/wall ratios in collection
  also include time waiting for GPU inference; compare the environment-only runs.
- Fast GPU policy but slow collection residual: inspect synchronization and
  host staging next. Low measured GPU utilization alone cannot identify the cause.
- Overlap improves stock but not optimized: faster rendering has exposed GPU
  contention or made the learner the limiting stage.
- Poor worker scaling with narrow affinity or many threads per worker:
  repeat with explicit thread limits. Do not change those settings in the first
  run, so the baseline resembles the training job.

## Shorter runs and local verification

Skip model/learning work for an environment-only cluster investigation:

```bash
sbatch slurm/dreamer-benchmark.sbatch --skip-model
```

Skip learner compilation but retain policy and production collection:

```bash
sbatch slurm/dreamer-benchmark.sbatch --skip-learning
```

A small CPU smoke run, including the real learner and overlap schedules:

```bash
.venv/bin/python scripts/benchmark_dreamer.py --backend cpu \
  --env-counts 1 2 --policy-envs 1 --repeats 1 --warmup 1 \
  --steps 3 --collection-steps 4 --replay-capacity 2 \
  --font-iterations 3 --profile-steps 3 --learning-steps 2 \
  --overlap-steps 2 --batch-size 1 --seq-length 2 \
  --output-dir /tmp/dreamer-benchmark-smoke
```

Use a new output directory for each run. Do not combine results from different
GPU models as if they were repeated measurements of the same workload.
