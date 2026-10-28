"""Compare stock and optimized CarRacing under identical Dreamer actions.

No learning or W&B logging is performed. A single policy supplies both vector
environments, so a pixel/reward mismatch identifies an environment difference
without two separately trained policies introducing their own divergence.
"""

import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import numpy as np


def check(args):
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from dreamer.actor_critic import Actor
    from dreamer.buffer import choose_action
    from dreamer.wm import DreamerConfig, DreamerWM
    from wm.envs import CONTROLLER_TRAINING_ENV_SPEC
    from wm.utils.obs import crop_obs, normalise_obs

    jax.config.update("jax_platforms", args.backend)
    if args.checkpoint:
        from dreamer.checkpoints import load_checkpoint

        models, _ = load_checkpoint(args.checkpoint)
        wm, actor = models["world_model"], models["actor"]
        nnx.reseed(wm, default=args.seed + 1)
        del models
    else:
        cfg = DreamerConfig()
        wm = DreamerWM(cfg, rngs=nnx.Rngs(args.seed + 1))
        actor = Actor(cfg, rngs=nnx.Rngs(args.seed + 2))

    # Match the reset seed used by the first production collection call.
    action_key, _ = jax.random.split(jax.random.key(args.seed))
    reset_key, key = jax.random.split(action_key)
    env_seed = int(jax.random.randint(reset_key, (), 0, np.iinfo(np.int32).max))
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    result = {
        "seed": args.seed,
        "environment_seed": env_seed,
        "backend": str(jax.devices()),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "flax", "gymnasium", "pygame-ce", "Box2D")
        },
        "cases": [],
    }
    envs = []
    started = perf_counter()

    def mismatch(case, step, field, reference, optimized):
        result["mismatch"] = {"case": case, "step": step, "field": field}
        np.savez(directory / "mismatch.npz", reference=reference, optimized=optimized)
        raise AssertionError(
            f"{field} differs in {case} at step {step}; see {directory}"
        )

    try:
        for spec in ("CarRacing-v3", CONTROLLER_TRAINING_ENV_SPEC):
            envs.append(
                gym.make_vec(
                    spec,
                    num_envs=args.num_envs,
                    vectorization_mode="async",
                    vector_kwargs={"context": "spawn"},
                )
            )
        reference, optimized = envs
        for case in ("policy", "driving_and_autoresets"):
            obs, _ = reference.reset(seed=env_seed)
            fast, _ = optimized.reset(seed=env_seed)
            carry = wm.rssm.initialize_state(args.num_envs)
            previous = jnp.zeros((args.num_envs, 3), dtype=jnp.float32)
            first = jnp.ones(args.num_envs, dtype=bool)
            endings = 0
            for step in range(args.steps + 1):
                pixels, fast_pixels = crop_obs(obs), crop_obs(fast)
                if not np.array_equal(pixels, fast_pixels):
                    mismatch(case, step, "model_input_pixels", pixels, fast_pixels)
                if step == args.steps:
                    break
                if case == "policy":
                    subkey, key = jax.random.split(key)
                    carry, action = choose_action(
                        wm,
                        actor,
                        carry,
                        jnp.asarray(normalise_obs(pixels) - 0.5),
                        previous,
                        first,
                        subkey,
                    )
                    actions = np.asarray(action, np.float32)
                    previous = action
                    # Deliberately mirror production collection's reset handling.
                    first = jnp.zeros(args.num_envs, dtype=bool)
                else:
                    steer = np.resize(
                        np.array(
                            [
                                0,
                                0.1,
                                -0.1,
                                0.25 * np.sin(step / 50),
                                -0.25 * np.sin(step / 50),
                                0,
                                0.6,
                                -0.6,
                            ],
                            np.float32,
                        ),
                        args.num_envs,
                    )
                    actions = np.stack(
                        [
                            steer,
                            np.full(args.num_envs, 0.8, np.float32),
                            np.zeros(args.num_envs, np.float32),
                        ],
                        axis=-1,
                    )
                reference.step_async(actions)
                optimized.step_async(actions)
                stock_step, fast_step = reference.step_wait(), optimized.step_wait()
                obs, fast = stock_step[0], fast_step[0]
                for index, field in (
                    (1, "reward"),
                    (2, "terminated"),
                    (3, "truncated"),
                ):
                    if not np.array_equal(stock_step[index], fast_step[index]):
                        mismatch(
                            case, step + 1, field, stock_step[index], fast_step[index]
                        )
                endings += int(np.count_nonzero(stock_step[2] | stock_step[3]))
                if (step + 1) % 200 == 0:
                    print(
                        f"{case}: {step + 1} steps match across {args.num_envs} lanes",
                        flush=True,
                    )
            result["cases"].append({
                "case": case,
                "vector_steps": args.steps,
                "lanes": args.num_envs,
                "matched_images": args.num_envs * (args.steps + 1),
                "episode_end_events": endings,
            })
        result["status"] = "passed"
    except BaseException:
        result["status"] = "failed"
        raise
    finally:
        for vector in envs:
            vector.close()
        result["elapsed_s"] = perf_counter() - started
        (directory / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved {directory / 'results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--backend", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    if arguments.num_envs < 1 or arguments.steps < 1:
        parser.error("--num-envs and --steps must be positive")
    if arguments.output_dir is None:
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        arguments.output_dir = (
            Path("experiments/dreamer/diagnostics")
            / f"renderer-parity-{stamp}-{os.getpid()}"
        )
    check(arguments)
