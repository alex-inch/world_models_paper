"""Render a selected benchmark episode using CarRacing's native video renderer."""

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
from PIL import Image

from dreamer.buffer import choose_action
from dreamer.checkpoints import load_checkpoint
from wm.utils.obs import crop_obs, normalise_obs

from render_dreamer_policy import encode_video, prediction_panels


@nnx.jit
def prior_logits(wm, hidden):
    return wm.rssm.trans_model(hidden).reshape(
        hidden.shape[0], wm.cfg.num_latent_dists, wm.cfg.num_latent_classes,
    )


def record_episode(models, env_id, *, episode_seed, num_envs=8, max_steps=1000):
    """Reproduce the benchmark's batch RNG while recording one selected lane."""
    if episode_seed < 500_000 or num_envs < 1:
        raise ValueError("Use a benchmark episode seed >= 500000 and positive num_envs")
    batch_seed = 500_000 + (episode_seed - 500_000) // num_envs * num_envs
    lane = episode_seed - batch_seed
    wm, actor = models["world_model"], models["actor"]
    envs = gym.make_vec(env_id, num_envs=num_envs, vectorization_mode="sync",
                        render_mode="rgb_array")
    nnx.reseed(wm, default=batch_seed)
    carry = wm.rssm.initialize_state(num_envs)
    previous_action = jnp.zeros((num_envs, wm.cfg.action_dim), dtype=jnp.float32)
    key = jax.random.key(batch_seed)
    history = {name: [] for name in (
        "images", "rgb_frames", "hidden", "latent_sample", "prior_logits", "actions", "rewards",
    )}
    done = False
    ended = {"terminated": False, "truncated": False}
    try:
        observation, _ = envs.reset(seed=[batch_seed + i for i in range(num_envs)])
        fps = int(envs.metadata["render_fps"])
        for step in range(max_steps + 1):
            pixels = crop_obs(observation)
            key, action_key = jax.random.split(key)
            carry, action = choose_action(
                wm, actor, carry, jnp.asarray(normalise_obs(pixels) - 0.5),
                previous_action, jnp.full(num_envs, step == 0), action_key,
            )
            history["images"].append(pixels[lane])
            history["rgb_frames"].append(envs.envs[lane].render())
            history["hidden"].append(np.asarray(carry[0][lane]))
            history["latent_sample"].append(np.asarray(carry[1][lane]))
            history["prior_logits"].append(np.asarray(prior_logits(wm, carry[0])[lane]))
            if done or step == max_steps:
                break
            observation, reward, terminated, truncated, _ = envs.step(np.asarray(action))
            history["actions"].append(np.asarray(action[lane]))
            history["rewards"].append(float(reward[lane]))
            previous_action = action
            done = bool(terminated[lane] or truncated[lane])
            ended = {"terminated": bool(terminated[lane]), "truncated": bool(truncated[lane])}
            if (step + 1) % 100 == 0:
                print(f"Recorded {step + 1} steps for track {episode_seed}", flush=True)
    finally:
        envs.close()
    return ({name: np.asarray(values) for name, values in history.items()}, fps,
            {"batch_seed": batch_seed, "lane": lane, "num_envs": num_envs, **ended})


def encode_raw(frames, output, fps):
    """Encode native pixels with no added text, borders, cropping or resizing."""
    height, width = frames.shape[1:3]
    # A shared palette prevents per-frame colour shifts in the GIF.
    samples = np.linspace(0, len(frames) - 1, 24, dtype=int)
    palette_sheet = Image.new("RGB", (900, 400))
    for slot, index in enumerate(samples):
        thumbnail = Image.fromarray(frames[index]).resize((150, 100), Image.Resampling.NEAREST)
        palette_sheet.paste(thumbnail, ((slot % 6) * 150, (slot // 6) * 100))
    palette = palette_sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    writer = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "16",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output.with_suffix(".mp4")),
    ], stdin=subprocess.PIPE, close_fds=False)
    assert writer.stdin is not None
    gif_frames = []
    try:
        for index, pixels in enumerate(frames):
            writer.stdin.write(pixels.tobytes())
            gif_frames.append(Image.fromarray(pixels).quantize(palette=palette, dither=Image.Dither.NONE))
            if (index + 1) % 200 == 0:
                print(f"Encoded {index + 1}/{len(frames)} raw frames", flush=True)
    finally:
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError("FFmpeg failed to encode the raw policy video")
    gif_frames[0].save(output.with_suffix(".gif"), save_all=True,
                       append_images=gif_frames[1:], duration=round(1000 / fps),
                       loop=0, disposal=1, optimize=False)


def main(checkpoint: Path, output: Path, hero_output: Path | None = None,
         episode_seed: int = 500_039, num_envs: int = 8, max_steps: int = 1000,
         backend: Literal["cpu", "mps", "cuda"] = "cpu"):
    outputs = [output] + ([hero_output] if hero_output is not None else [])
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("Choose different output prefixes")
    if any(path.with_suffix(suffix).exists() for path in outputs
           for suffix in (".gif", ".mp4", ".npz", ".json")):
        raise FileExistsError("Choose fresh output filename prefixes")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    jax.config.update("jax_platforms", backend)
    models, metadata = load_checkpoint(checkpoint)
    print(f"Loaded {checkpoint} on {jax.devices()}", flush=True)
    episode, fps, setup = record_episode(
        models, metadata["train_args"]["env_id"], episode_seed=episode_seed,
        num_envs=num_envs, max_steps=max_steps,
    )
    information = {
        "checkpoint": str(checkpoint.resolve()), "run_name": metadata["train_args"]["name"],
        "train_step": metadata["train_step"], "backend": backend,
        "environment_seed": episode_seed, **setup, "fps": fps,
        "frames": len(episode["images"]), "environment_steps": len(episode["actions"]),
        "episode_return": float(episode["rewards"].sum()),
        "duration_seconds": len(episode["images"]) / fps,
        "renderer": "Gymnasium CarRacing-v3 rgb_array", "resolution": [600, 400],
        "added_text": False, "native_hud": True,
        "selection": "Selected hero episode from the existing 64-episode benchmark",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), **episode)
    output.with_suffix(".json").write_text(json.dumps(information, indent=2) + "\n")
    encode_raw(episode["rgb_frames"], output, fps)
    print(f"Saved raw policy GIF: {output.with_suffix('.gif')}", flush=True)
    if hero_output is not None:
        panels = prediction_panels(models, episode, full_context=50, seed=setup["batch_seed"] + 1)
        hero_info = {**information, "context": 5, "full_context": 50, "horizon": 15,
                     "header": False, "status_line": False, "timer": False,
                     "subheads": False, "resolution": [864, 264],
                     "added_text": True, "native_hud": False,
                     "renderer": "64x64 model observations and decoded model predictions",
                     "raw_policy_recording": str(output.resolve())}
        hero_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(hero_output.with_suffix(".npz"), **panels,
                            actions=episode["actions"], rewards=episode["rewards"])
        hero_output.with_suffix(".json").write_text(json.dumps(hero_info, indent=2) + "\n")
        encode_video(panels, hero_output, hero_info)
        print(f"Saved four-pane hero GIF: {hero_output.with_suffix('.gif')}", flush=True)
    print(f"Episode return: {information['episode_return']:.2f}; {information['environment_steps']} steps", flush=True)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
