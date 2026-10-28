"""Evaluate conditioning length with a shared forecast origin and paired episodes."""

import json
from pathlib import Path
from typing import Literal

import jax
import numpy as np

from dreamer.checkpoints import load_checkpoint
from dreamer.conditioning import CONTEXTS, collect_conditioning
from dreamer.divergence import bootstrap_curve_mean


def main(checkpoint: Path, output: Path, num_episodes: int = 64, num_envs: int = 8,
         max_steps: int = 1000, seed: int = 500_000, bootstrap_reps: int = 10_000,
         backend: Literal["cpu", "mps", "cuda"] = "cpu"):
    if min(num_envs, bootstrap_reps) < 1 or num_episodes < 2:
        raise ValueError("Use at least two episodes and positive settings")
    if any(output.with_suffix(suffix).exists() for suffix in (".json", ".npz")):
        raise FileExistsError("Choose a fresh output filename prefix")
    jax.config.update("jax_platforms", backend)
    models, metadata = load_checkpoint(checkpoint)
    print(f"Loaded {checkpoint} on {jax.devices()}", flush=True)
    curves, lengths, returns = collect_conditioning(
        models, metadata["train_args"]["env_id"], num_episodes=num_episodes,
        num_envs=num_envs, max_steps=max_steps, seed=seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), kl_curves=curves,
                        episode_lengths=lengths, episode_returns=returns,
                        episode_seeds=np.arange(seed, seed + num_episodes))
    common_end = int(lengths.min())
    environment_steps = np.arange(max(CONTEXTS), common_end + 1)
    if not len(environment_steps):
        raise ValueError("An episode ended before the common forecast origin")
    print(f"Bootstrapping all {num_episodes} episodes through environment step {common_end}", flush=True)
    mean, interval = bootstrap_curve_mean(curves[:, environment_steps], reps=bootstrap_reps, seed=seed)
    report = {
        "checkpoint": str(checkpoint.resolve()), "checkpoint_metadata": metadata,
        "contexts": list(CONTEXTS), "num_episodes": num_episodes,
        "num_envs": min(num_envs, num_episodes), "backend": backend,
        "seed": seed, "max_steps": max_steps,
        "episode_lengths": lengths.tolist(), "episode_returns": returns.tolist(),
        "metric": "KL(real observation-conditioned posterior || open-loop dynamics prior)",
        "units": "nats per categorical latent factor", "aggregation": "mean across episodes",
        "actions": "Actor chooses actions inside each imagined state; real policy unchanged",
        "conditioning": "Trailing observation windows ending at the same environment frame",
        "context_windows": [[max(CONTEXTS) - context, max(CONTEXTS) - 1] for context in CONTEXTS],
        "forecast_origin_environment_step": max(CONTEXTS) - 1,
        "paired_randomness": "Same model latent draws and actor sampling keys across contexts",
        "common_horizon_end": common_end, "environment_steps": environment_steps.tolist(),
        "steps": (environment_steps - max(CONTEXTS) + 1).tolist(),
        "mean": mean.tolist(), "ci_low": interval[0].tolist(), "ci_high": interval[1].tolist(),
        "bootstrap_library": "rliable", "bootstrap_method": "percentile",
        "bootstrap_reps": bootstrap_reps, "bootstrap_seed": seed,
        "confidence_level": 0.95, "confidence_scope": "pointwise whole-episode bootstrap; fixed checkpoint",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
