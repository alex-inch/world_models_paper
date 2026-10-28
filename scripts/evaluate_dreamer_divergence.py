"""Measure latent prediction drift across episodes; no image decoding is needed."""

import json
from pathlib import Path
from typing import Literal

import jax
import numpy as np

from dreamer.checkpoints import load_checkpoint
from dreamer.divergence import CONDITIONS, bootstrap_curve_mean, collect_divergence


def main(
    checkpoint: Path,
    output: Path,
    num_episodes: int = 64,
    num_envs: int = 8,
    max_steps: int = 1000,
    context: int = 5,
    seed: int = 500_000,
    bootstrap_reps: int = 10_000,
    backend: Literal["cpu", "mps", "cuda"] = "cpu",
):
    """Save per-episode KL curves and a mean/CI summary over the common horizon."""
    if num_episodes < 2 or min(num_envs, context, bootstrap_reps) < 1 or max_steps < context:
        raise ValueError("Use at least two episodes, positive settings and max_steps >= context")
    if any(output.with_suffix(suffix).exists() for suffix in (".npz", ".json")):
        raise FileExistsError("Choose a fresh --output filename prefix")
    jax.config.update("jax_platforms", backend)
    print(f"Loading {checkpoint} on {jax.devices()}", flush=True)
    models, information = load_checkpoint(checkpoint)
    curves, lengths, returns = collect_divergence(
        models, information["train_args"]["env_id"], num_episodes=num_episodes,
        num_envs=num_envs, max_steps=max_steps, context=context, seed=seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output.with_suffix(".npz"), kl_curves=curves, episode_lengths=lengths,
        episode_returns=returns, episode_seeds=np.arange(seed, seed + num_episodes),
    )
    # At the first matched-action prediction the real and imagined histories are identical.
    np.testing.assert_allclose(curves[:, context, 0], curves[:, context, 1], atol=1e-5)
    common_end = int(lengths.min())
    if common_end < context:
        raise ValueError("Some episodes ended before the initial context was complete")
    steps = np.arange(context, common_end + 1)
    print(f"Bootstrapping {num_episodes} complete trajectories through step {common_end}", flush=True)
    mean, interval = bootstrap_curve_mean(
        curves[:, steps], reps=bootstrap_reps, seed=seed,
    )
    report = {
        "checkpoint": str(checkpoint.resolve()), "checkpoint_metadata": information,
        "num_episodes": num_episodes, "num_envs": min(num_envs, num_episodes),
        "backend": backend, "context": context, "max_steps": max_steps,
        "episode_seeds": list(range(seed, seed + num_episodes)),
        "episode_lengths": lengths.tolist(), "episode_returns": returns.tolist(),
        "metric": "KL(observation-conditioned posterior || predicted prior)",
        "units": "nats per categorical latent factor", "aggregation": "mean across episodes",
        "conditions": list(CONDITIONS), "bootstrap_reps": bootstrap_reps,
        "bootstrap_library": "rliable", "bootstrap_method": "percentile",
        "bootstrap_seed": seed,
        "confidence_level": 0.95, "confidence_scope": "pointwise whole-episode bootstrap; fixed checkpoint",
        "common_horizon_end": common_end,
        "horizon_note": "All episodes contribute at every plotted step. Longer trajectories are retained in the NPZ.",
        "steps": steps.tolist(), "mean": mean.tolist(),
        "ci_low": interval[0].tolist(), "ci_high": interval[1].tolist(),
        "condition_descriptions": {
            "posterior_corrected": "One-step prior from the preceding real posterior and real action; corrected by observations every step.",
            "open_loop_matched_actions": "No observation corrections after context; uses the real policy's actions to isolate dynamics drift.",
            "open_loop_actor": "No observation corrections after context; actor chooses actions in the imagined state, matching the GIF condition.",
        },
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output.with_suffix('.npz')} and {output.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
