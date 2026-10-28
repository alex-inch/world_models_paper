# First-step and renderer diagnostics on ARC

Queue the first-update comparison from the repository root:

```bash
sbatch slurm/dreamer-first-step-trace.sbatch
```

This requests an L40S, matching the two original runs, and uses their existing
initial checkpoints:

- Stock: `experiments/dreamer/checkpoints/gp8l0xmb/00000000`
- Optimized: `experiments/dreamer/checkpoints/fh6igknc/00000000`

It compares all saved initial parameters and RNG states bit for bit, reconstructs
the first collection independently with each renderer, and saves the sampled
batch, sequence indices, starting states and random keys. It then captures the
world-model losses and gradients, updated world-model parameters, posterior
encoder output and recurrent states, imagined states/actions/rewards/continuation
probabilities, bootstrap values, critic targets and weights, critic predictions,
loss and gradients, replay refresh and updated model states.

A third pass repeats the stock case's **same frozen replay, post-collection model
and RNG states, step key and newly initialized optimizer states**, using the same
traced update. This distinguishes a difference in inputs from execution
variability. That control does not run collection again.

The historical critic-learning-rate behavior is reconstructed from metadata:
when only `policy_lr` is present, it was unused and the critic used `actor_lr`.
Checkpoints with the newer `critic_lr` field use its explicit value. The diagnostic
is independent of the current training script's logging changes and does not
start W&B runs or write training checkpoints.
It also preserves the original world-model objective and its second posterior
pass after the optimizer update, rather than adopting newer changes that reuse
pre-update states or consume RNG keys differently.

## What can and cannot be recovered

The initial model checkpoints are genuine historical tensors. **The original
first replay batches and critic targets were never saved**, so those stages are
reconstructions, not recovered originals. All captured arrays are saved in NPZ
files with manifests and SHA-256 hashes. The comparison reports the first
differing stage, differing tensor paths, element counts, coordinates and maximum
absolute errors; bitwise equality is separate from approximate numerical equality.

Returning intermediate arrays changes the XLA graph and can change kernel fusion
and autotuning. Consequently the reconstructed critic loss may not equal the
historical W&B scalar exactly. A fully matching reconstructed pair rules out
differences in those reproduced inputs; it cannot prove the original independent
GPU compilations used identical kernels. A mismatch in the repeated frozen-input
pass demonstrates variability in the traced update on that device.

Default dimensions and learning rates come from each checkpoint's metadata.
Collection is 1,000 steps with the original replay capacity, preserving production
shapes. Full tensor output can occupy several GB; it stays on ARC. The job saves
partial reports as it progresses and logs collection progress every 200 steps.
The wall-time request is one hour; NFS font stalls make runtime variable.

## Results

Read `experiments/dreamer/diagnostics/first-step-<job-id>/report.md`. Return
`report.json`, `report.md` and the Slurm stdout log for analysis. There is no need
to transfer all NPZ files initially; retain them on ARC for detailed inspection.

Interpret the earliest mismatch in order:

1. `initial`: real saved parameters or RNG state already differ.
2. `post_collection` / `replay`: policy collection, observations, actions, rewards,
   stored recurrent states or RNG progression differ.
3. `sample`: sequence indices, batch data, starting states or keys differ.
4. `wm_forward` / `wm_gradients` / `wm_updated`: the initial model computation,
   gradients or first world-model optimizer update differs.
5. `posterior_embedding` / `posterior`: the recomputed encoder or posterior differs.
6. Imagination through `critic_targets`: latent dynamics, rewards, discounting or
   target values differ. These occur before the critic optimizer update.
7. `critic_predictions` / `critic_loss`: the critic forward calculation or weighted
   squared-error reduction differs.
8. Later gradients/replay outputs: subsequent effects, not the origin of an
   already differing critic loss.

For a **separate** reconstruction controlling XLA GPU nondeterminism:

```bash
sbatch slurm/dreamer-first-step-trace.sbatch --deterministic
```

This sets `--xla_gpu_exclude_nondeterministic_ops=true` before JAX initialization.
It can be slower, and XLA may reject operations without deterministic support.
It changes execution relative to the original jobs; compare it as a controlled
experiment rather than silently enabling it for the normal diagnostic.

## Paired renderer check with the current policy

The short launcher for the earlier renderer diagnostic hardcodes the current
optimized run's existing 22k checkpoint:

```bash
sbatch slurm/dreamer-renderer-parity.sbatch
```

It drives stock and optimized environments with the same actions from
`fh6igknc/00022000`, comparing pixels after preprocessing, rewards, termination
and truncation. Results go to `renderer-parity-<job-id>/results.json`. On failure,
`mismatch.npz` contains the first differing tensors. This is an environment check,
not a first-update reconstruction or an additional training run.
