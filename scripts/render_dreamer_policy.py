"""Render a checkpoint at native FPS with periodic and uninterrupted imagination."""

import json
import os
import subprocess
from pathlib import Path
from typing import Literal

import gymnasium as gym
import imageio_ffmpeg
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from dreamer.checkpoints import load_checkpoint
from dreamer.imagine import run_imagination_rollout
from dreamer.inspection import _branch_world_model, frames_to_uint8, record_step
from dreamer.wm.rssm import RSSMOut
from wm.utils.obs import crop_obs, normalise_obs

INK = "#343633"
PAPER = "#F7F4EB"


def record_policy(models, metadata, *, seed=500_000, max_steps=1000):
    """Recreate the first training-evaluation lane and stop at its episode end."""
    wm, actor = models["world_model"], models["actor"]
    args = metadata["train_args"]
    num_envs = args["num_envs"]
    envs = gym.make_vec(args["env_id"], num_envs=num_envs, vectorization_mode="sync")
    collection_key, _ = jax.random.split(jax.random.key(seed))
    reset_key, key = jax.random.split(collection_key)
    env_seed = int(jax.random.randint(reset_key, (), 0, np.iinfo(np.int32).max))
    episode = {
        name: []
        for name in (
            "images",
            "hidden",
            "latent_sample",
            "prior_logits",
            "actions",
            "rewards",
        )
    }
    nnx.reseed(wm, default=seed)
    carry = wm.rssm.initialize_state(num_envs)
    previous_action = jnp.zeros((num_envs, wm.cfg.action_dim), dtype=jnp.float32)
    done = False
    try:
        observation, _ = envs.reset(seed=env_seed)
        fps = int(envs.metadata["render_fps"])
        for step in range(max_steps + 1):
            pixels = crop_obs(observation)
            action_key, key = jax.random.split(key)
            carry, action, state, _ = record_step(
                wm,
                actor,
                carry,
                jnp.asarray(normalise_obs(pixels) - 0.5),
                previous_action,
                jnp.full(num_envs, step == 0),
                action_key,
            )
            episode["images"].append(pixels[0])
            for name, value in (
                ("hidden", state.hidden),
                ("latent_sample", state.latent_sample),
                ("prior_logits", state.prior),
            ):
                episode[name].append(np.asarray(value[0]))
            if done or step == max_steps:
                break
            observation, reward, terminated, truncated, _ = envs.step(
                np.asarray(action)
            )
            episode["actions"].append(np.asarray(action[0]))
            episode["rewards"].append(float(reward[0]))
            previous_action = action
            done = bool(terminated[0] or truncated[0])
            if (step + 1) % 100 == 0:
                print(f"Recorded {step + 1} environment steps", flush=True)
    finally:
        envs.close()
    return {name: np.asarray(values) for name, values in episode.items()}, fps, env_seed


@nnx.jit(static_argnames="horizon")
def _imagine_features(wm, actor, critic, initial, *, horizon, key):
    trajectory, _, _ = run_imagination_rollout(wm, actor, critic, initial, horizon, key)
    return trajectory.state.joint_latent


@nnx.jit
def _decode(wm, features):
    return wm.decoder(features)


def _decode_features(wm, features, chunk_size=64):
    features = np.asarray(features)
    shape = features.shape[:-1]
    flattened = features.reshape(-1, features.shape[-1])
    decoded = []
    for start in range(0, len(flattened), chunk_size):
        batch = flattened[start : start + chunk_size]
        count = len(batch)
        batch = np.pad(batch, ((0, chunk_size - count), (0, 0)))
        decoded.append(frames_to_uint8(_decode(wm, jnp.asarray(batch)))[:count])
    return np.concatenate(decoded).reshape(*shape, 64, 64, 3)


def prediction_panels(models, episode, *, context=5, full_context=None, horizon=15, seed=500_001):
    """The full branch consumes only the initial context, never future observations."""
    wm = models["world_model"]
    count = len(episode["images"])
    context = min(context, count)
    full_context = min(context if full_context is None else full_context, count)
    if min(context, full_context, horizon) < 1:
        raise ValueError("Use positive context lengths and imagination horizon")
    reconstruction = _decode_features(
        wm, np.concatenate([episode["hidden"], episode["latent_sample"]], axis=-1)
    )
    periodic = reconstruction.copy()
    full = reconstruction.copy()

    def initial(indices):
        return RSSMOut(
            hidden=jnp.asarray(episode["hidden"][indices]),
            latent_sample=jnp.asarray(episode["latent_sample"][indices]),
            prior=jnp.asarray(episode["prior_logits"][indices]),
            posterior=None,
        )

    starts = np.arange(context - 1, count - 1, context + horizon)
    if len(starts):
        branch = _branch_world_model(models, seed)
        features = _imagine_features(
            branch,
            models["actor"],
            models["target_critic"],
            initial(starts),
            horizon=horizon,
            key=jax.random.key(seed),
        )
        frames = _decode_features(branch, features)
        for index, start in enumerate(starts):
            end = min(start + horizon + 1, count)
            periodic[start + 1 : end] = frames[index, 1 : end - start]

    if count > full_context:
        branch = _branch_world_model(models, seed + 1)
        features = _imagine_features(
            branch,
            models["actor"],
            models["target_critic"],
            initial(np.array([full_context - 1])),
            horizon=count - full_context,
            key=jax.random.key(seed + 1),
        )
        full[full_context:] = _decode_features(branch, features)[0, 1:]
    return {
        "policy": episode["images"],
        "reconstruction": reconstruction,
        "periodic_imagination": periodic,
        "full_imagination": full,
    }


def encode_video(panels, output, information):
    """Write matching GIF and MP4 files; one simulation frame per encoded frame."""
    fps = information["fps"]
    count = len(panels["policy"])
    font = font_manager.findfont(
        font_manager.FontProperties(family="Routed Gothic"), fallback_to_default=False
    )
    fonts = {size: ImageFont.truetype(font, size) for size in (12, 14)}
    base = Image.new("RGB", (864, 264), PAPER)
    draw = ImageDraw.Draw(base)
    # draw.text(
    #     (24, 54), f"{information['run_name']}  /  {information['train_step']:,} UPDATES",
    #     font=fonts[14], fill=INK,
    # )
    labels = [
        ("policy", "POLICY"),
        ("reconstruction", "RECONSTRUCTION"),
        ("periodic_imagination", "PARTIAL IMAGINATION"),
        ("full_imagination", "FULL IMAGINATION"),
    ]
    for index, (_, title) in enumerate(labels):
        left = 24 + index * 208
        draw.text((left, 24), title, font=fonts[14], fill=INK)
        draw.rectangle((left - 1, 47, left + 192, 240), outline=INK)
    # draw.text(
    #     (24, 414), "IMAGINATION PANES USE THEIR OWN ACTOR ACTIONS; FULL IMAGINATION NEVER RECORRECTS.",
    #     font=fonts[12], fill=INK,
    # )
    # Avoid a preexec_fn/fork after JAX has started its worker threads.
    writer = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "864x264",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output.with_suffix(".mp4")),
        ],
        stdin=subprocess.PIPE,
        close_fds=False,  # Allows subprocess to use posix_spawn on macOS.
    )
    assert writer.stdin is not None
    gif_frames = []
    try:
        for index in range(count):
            frame = base.copy()
            for panel, (name, _) in enumerate(labels):
                pixels = Image.fromarray(panels[name][index]).resize(
                    (192, 192), Image.Resampling.NEAREST
                )
                frame.paste(pixels, (24 + panel * 208, 48))
            writer.stdin.write(frame.tobytes())
            gif_frames.append(
                frame.quantize(
                    colors=256,
                    method=Image.Quantize.FASTOCTREE,
                    dither=Image.Dither.NONE,
                )
            )
            if (index + 1) % 100 == 0:
                print(f"Rendered {index + 1}/{count} frames", flush=True)
    finally:
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError("FFmpeg failed to encode the policy video")
    gif_frames[0].save(
        output.with_suffix(".gif"),
        save_all=True,
        append_images=gif_frames[1:],
        duration=round(1000 / fps),
        loop=0,
        disposal=1,
        optimize=False,
    )


def main(
    checkpoint: Path,
    output: Path,
    backend: Literal["cpu", "mps", "cuda"] = "cpu",
    max_steps: int = 1000,
    seed: int = 500_000,
    full_context: int = 50,
):
    """Regenerate the first policy-evaluation episode with four labelled panes."""
    if full_context < 1 or max_steps < max(5, full_context):
        raise ValueError("Use a positive context no longer than max_steps")
    if any(
        output.with_suffix(suffix).exists()
        for suffix in (".gif", ".mp4", ".npz", ".json")
    ):
        raise FileExistsError("Choose a new --output filename prefix")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    jax.config.update("jax_platforms", backend)
    print(f"Loading {checkpoint} on {jax.devices()}", flush=True)
    models, metadata = load_checkpoint(checkpoint)
    episode, fps, env_seed = record_policy(
        models, metadata, seed=seed, max_steps=max_steps
    )
    print("Generating reconstructions and imagination branches", flush=True)
    panels = prediction_panels(models, episode, full_context=full_context, seed=seed + 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    information = {
        "checkpoint": str(checkpoint.resolve()),
        "run_name": metadata["train_args"]["name"],
        "train_step": metadata["train_step"],
        "backend": backend,
        "seed": seed,
        "environment_seed": env_seed,
        "fps": fps,
        "context": 5,
        "full_context": full_context,
        "horizon": 15,
        "frames": len(episode["images"]),
        "environment_steps": len(episode["actions"]),
        "episode_return": float(episode["rewards"].sum()),
        "duration_seconds": len(episode["images"]) / fps,
        "full_imagination": f"actor chooses actions inside the model; no observations after initial {full_context} frames",
        "header": False,
        "status_line": False,
        "timer": False,
        "subheads": False,
        "resolution": [864, 264],
    }
    np.savez_compressed(
        output.with_suffix(".npz"),
        **panels,
        actions=episode["actions"],
        rewards=episode["rewards"],
    )
    output.with_suffix(".json").write_text(json.dumps(information, indent=2) + "\n")
    encode_video(panels, output, information)
    print(
        f"Saved {output.with_suffix('.gif')} and {output.with_suffix('.mp4')}",
        flush=True,
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
