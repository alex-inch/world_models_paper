"""Render four imagination-only histories from the same cached policy episode."""

import json
import subprocess
from pathlib import Path
from typing import Literal

import imageio_ffmpeg
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from dreamer.checkpoints import load_checkpoint
from dreamer.conditioning import CONTEXTS, condition_recording, imagine_step
from dreamer.inspection import frames_to_uint8

INK = "#343633"
PAPER = "#F7F4EB"


@nnx.jit
def decode(wm, features):
    return wm.decoder(features)


def decode_features(wm, features, chunk_size=64):
    shape = features.shape[:-1]
    flattened = features.reshape(-1, features.shape[-1])
    frames = []
    for start in range(0, len(flattened), chunk_size):
        batch = flattened[start : start + chunk_size]
        count = len(batch)
        batch = np.pad(batch, ((0, chunk_size - count), (0, 0)))
        frames.append(frames_to_uint8(decode(wm, jnp.asarray(batch)))[:count])
    return np.concatenate(frames).reshape(*shape, 64, 64, 3)


def encode(panels, output, fps):
    font = font_manager.findfont(
        font_manager.FontProperties(family="Routed Gothic"), fallback_to_default=False,
    )
    heading = ImageFont.truetype(font, 24)
    label = ImageFont.truetype(font, 14)
    subtitle = ImageFont.truetype(font, 12)
    base = Image.new("RGB", (864, 360), PAPER)
    draw = ImageDraw.Draw(base)
    draw.text((24, 18), "DREAMER - OPEN-LOOP CONDITIONING", font=heading, fill=INK)
    for index, context in enumerate(CONTEXTS):
        left = 24 + index * 208
        draw.text((left, 64), f"{context} FRAME{'S' if context != 1 else ''}", font=label, fill=INK)
        draw.text((left, 86), "CONDITIONING", font=subtitle, fill=INK)
        draw.rectangle((left - 1, 103, left + 192, 296), outline=INK)
    writer = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "864x360",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "16",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output.with_suffix(".mp4")),
    ], stdin=subprocess.PIPE, close_fds=False)
    assert writer.stdin is not None
    gif_frames = []
    try:
        for step in range(panels.shape[1]):
            frame = base.copy()
            for index in range(len(CONTEXTS)):
                pixels = Image.fromarray(panels[index, step]).resize((192, 192), Image.Resampling.NEAREST)
                frame.paste(pixels, (24 + index * 208, 104))
            ImageDraw.Draw(frame).text(
                (24, 322), f"OPEN-LOOP TIME {(step + 1) / fps:05.2f} S", font=label, fill=INK,
            )
            writer.stdin.write(frame.tobytes())
            gif_frames.append(frame.quantize(colors=256, method=Image.Quantize.FASTOCTREE,
                                             dither=Image.Dither.NONE))
            if (step + 1) % 200 == 0:
                print(f"Encoded {step + 1}/{panels.shape[1]} frames", flush=True)
    finally:
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError("FFmpeg failed to encode the conditioning video")
    gif_frames[0].save(output.with_suffix(".gif"), save_all=True,
                       append_images=gif_frames[1:], duration=round(1000 / fps),
                       loop=0, disposal=1, optimize=False)


def main(checkpoint: Path, recording: Path, output: Path, seed: int = 500_000,
         backend: Literal["cpu", "mps", "cuda"] = "cpu"):
    if any(output.with_suffix(suffix).exists() for suffix in (".gif", ".mp4", ".npz", ".json")):
        raise FileExistsError("Choose a fresh output filename prefix")
    jax.config.update("jax_platforms", backend)
    models, metadata = load_checkpoint(checkpoint)
    source = json.loads(recording.with_suffix(".json").read_text())
    if Path(source["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError("Use a recording from the same checkpoint")
    with np.load(recording.with_suffix(".npz")) as data:
        images, actions = data["policy"], data["actions"]
    branches, states = condition_recording(models, images, actions, seed=seed)
    horizon = len(images) - max(CONTEXTS)
    key = jax.random.key(seed + 3_000_000)
    features = []
    for step in range(horizon):
        key, action_key = jax.random.split(key)
        states = imagine_step(branches, models["actor"], states, action_key)
        features.append(np.stack([np.asarray(state.joint_latent[0]) for state in states]))
        if (step + 1) % 200 == 0:
            print(f"Imagined {step + 1}/{horizon} steps", flush=True)
    features = np.stack(features, axis=1)
    panels = decode_features(models["world_model"], features)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), panels=panels, features=features)
    information = {
        "checkpoint": str(checkpoint.resolve()), "recording": str(recording.resolve()),
        "run_name": metadata["train_args"]["name"], "train_step": metadata["train_step"],
        "backend": backend, "seed": seed, "environment_seed": source["environment_seed"],
        "fps": source["fps"], "contexts": list(CONTEXTS),
        "context_windows": [[max(CONTEXTS) - context, max(CONTEXTS) - 1] for context in CONTEXTS],
        "forecast_origin_environment_step": max(CONTEXTS) - 1,
        "frames": horizon, "duration_seconds": horizon / source["fps"],
        "conditioning": "Trailing windows ending at the same frame; recurrent state reset at each window start",
        "imagination": "Own actor actions; no further observations; shared random draws across histories",
    }
    output.with_suffix(".json").write_text(json.dumps(information, indent=2) + "\n")
    encode(panels, output, source["fps"])
    print(f"Saved {output.with_suffix('.gif')} and {output.with_suffix('.mp4')}", flush=True)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
