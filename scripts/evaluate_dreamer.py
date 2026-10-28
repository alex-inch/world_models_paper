"""Evaluate frozen checkpoints using raw episode returns and rliable IQM intervals.

Every checkpoint uses the same environment seeds and sampled training policy.
The confidence interval resamples evaluation episodes for a fixed checkpoint;
it does not measure variation across independent training seeds.
"""

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from rliable import library, metrics

from dreamer.buffer import choose_action
from dreamer.checkpoints import load_checkpoint
from wm.utils.obs import crop_obs, normalise_obs


def summarize_returns(returns, *, bootstrap_reps=50_000, seed=500_000):
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 1 or len(returns) < 2 or not np.isfinite(returns).all():
        raise ValueError("Expected at least two finite episode returns")

    # rliable's resampling uses NumPy's global RNG. Preserve the caller's state.
    random_state = np.random.get_state()
    try:
        np.random.seed(seed)
        points, intervals = library.get_interval_estimates(
            {"checkpoint": returns[:, None]},
            metrics.aggregate_iqm,
            reps=bootstrap_reps,
            confidence_interval_size=0.95,
        )
    finally:
        np.random.set_state(random_state)

    return {
        "mean": float(returns.mean()),
        "median": float(np.median(returns)),
        "iqm": float(points["checkpoint"]),
        "iqm_ci_95": intervals["checkpoint"].reshape(-1).tolist(),
    }


def evaluate_checkpoint(checkpoint, *, num_episodes, num_envs, max_steps, seed):
    models, metadata = load_checkpoint(checkpoint)
    wm, actor = models["world_model"], models["actor"]
    env_id = metadata["train_args"]["env_id"]
    num_envs = min(num_envs, num_episodes)
    envs = gym.make_vec(env_id, num_envs=num_envs, vectorization_mode="sync")
    returns, lengths = [], []
    try:
        for start in range(0, num_episodes, num_envs):
            count = min(num_envs, num_episodes - start)
            batch_seed = seed + start
            observation, _ = envs.reset(
                seed=[batch_seed + index for index in range(num_envs)]
            )
            nnx.reseed(wm, default=batch_seed)
            carry = wm.rssm.initialize_state(num_envs)
            previous_action = jnp.zeros(
                (num_envs, wm.cfg.action_dim), dtype=jnp.float32
            )
            key = jax.random.key(batch_seed)
            active = np.arange(num_envs) < count
            batch_returns = np.zeros(num_envs, dtype=np.float64)
            batch_lengths = np.zeros(num_envs, dtype=np.int32)

            for step in range(max_steps):
                image = normalise_obs(crop_obs(observation)) - 0.5
                key, action_key = jax.random.split(key)
                carry, action = choose_action(
                    wm,
                    actor,
                    carry,
                    jnp.asarray(image),
                    previous_action,
                    jnp.full(num_envs, step == 0),
                    action_key,
                )
                observation, reward, terminated, truncated, _ = envs.step(
                    np.asarray(action)
                )
                batch_returns += np.where(active, reward, 0.0)
                batch_lengths += active
                # Ignore autoresets and all subsequent rewards after an episode ends.
                active &= ~(terminated | truncated)
                previous_action = action
                if not active.any():
                    break

            returns.extend(batch_returns[:count].tolist())
            lengths.extend(batch_lengths[:count].tolist())
            print(
                f"{checkpoint.parent.name}/{checkpoint.name}: "
                f"{len(returns)}/{num_episodes} episodes",
                flush=True,
            )
    finally:
        envs.close()
    return np.asarray(returns), lengths, metadata


def main(
    checkpoints: list[Path],
    output: Path | None = None,
    num_episodes: int = 64,
    num_envs: int = 8,
    max_steps: int = 1000,
    seed: int = 500_000,
    backend: Literal["cpu", "mps", "cuda"] = "cpu",
    bootstrap_reps: int = 50_000,
):
    """Evaluate checkpoint directories and save summaries plus individual returns."""
    if not checkpoints or num_episodes < 2:
        raise ValueError("Provide checkpoints and at least two evaluation episodes")
    if min(num_envs, max_steps, bootstrap_reps) < 1:
        raise ValueError("num_envs, max_steps and bootstrap_reps must be positive")
    checkpoints = list(dict.fromkeys(path.resolve() for path in checkpoints))
    for checkpoint in checkpoints:
        if not (checkpoint / "metadata.json").is_file():
            raise FileNotFoundError(f"Not a completed checkpoint: {checkpoint}")
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = Path("experiments/dreamer/evaluations") / f"{stamp}.json"
    if output.exists():
        raise FileExistsError(f"Choose a new --output path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_platforms", backend)
    print(f"Devices: {jax.devices()}", flush=True)
    report = {
        "num_episodes": num_episodes,
        "num_envs": min(num_envs, num_episodes),
        "max_steps": max_steps,
        "episode_seeds": list(range(seed, seed + num_episodes)),
        "policy": "sampled",
        "backend": backend,
        "bootstrap_reps": bootstrap_reps,
        "confidence_level": 0.95,
        "confidence_scope": "evaluation episodes, conditional on a fixed checkpoint",
        "results": [],
    }
    for checkpoint in checkpoints:
        started = perf_counter()
        returns, lengths, metadata = evaluate_checkpoint(
            checkpoint,
            num_episodes=num_episodes,
            num_envs=num_envs,
            max_steps=max_steps,
            seed=seed,
        )
        result = {
            "checkpoint": str(checkpoint),
            "name": metadata["train_args"]["name"],
            "train_step": metadata["train_step"],
            "checkpoint_metadata": metadata,
            "returns": returns.tolist(),
            "episode_lengths": lengths,
            **summarize_returns(returns, bootstrap_reps=bootstrap_reps, seed=seed),
            "elapsed_seconds": perf_counter() - started,
        }
        report["results"].append(result)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    print("\nCheckpoint                        Mean    Median       IQM [95% CI]")
    for result in report["results"]:
        checkpoint = Path(result["checkpoint"])
        label = f"{checkpoint.parent.name}/{checkpoint.name}"
        low, high = result["iqm_ci_95"]
        print(
            f"{label:30} {result['mean']:8.2f} {result['median']:9.2f} "
            f"{result['iqm']:9.2f} [{low:.2f}, {high:.2f}]"
        )
    print(f"\nSaved episode returns and statistics to {output.resolve()}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
