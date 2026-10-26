"""Drive a real CarRacing environment with a frozen Dreamer checkpoint."""

import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import gymnasium as gym
import jax
from loguru import logger

from dreamer.checkpoints import load_checkpoint
from dreamer.interactive import InspectorSession


def main(
    checkpoint: Path,
    output: Path | None = None,
    seed: int = 500_000,
    horizon: int = 15,
    fps: int = 15,
    backend: Literal["cpu", "mps", "cuda"] | None = None,
    headless_steps: int | None = None,
    screenshot: Path | None = None,
):
    """Open the inspector paused; arrows or draggable actor axes supply actions.

    Headless mode runs a short actor-controlled recording and optionally saves
    a screenshot, for environments without a display.
    """
    if fps < 1 or horizon < 1 or (headless_steps is not None and headless_steps < 0):
        raise ValueError("fps/horizon must be positive and headless_steps nonnegative")
    if backend is not None:
        jax.config.update("jax_platforms", backend)
    if headless_steps is not None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if output is None:
        output = Path("experiments/dreamer/interactive") / datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    if (output / "metadata.json").exists():
        raise ValueError(
            f"Recording already exists at {output}; choose a new output directory"
        )
    logger.info("Loading checkpoint {} on {}", checkpoint, jax.devices())
    models, information = load_checkpoint(checkpoint)
    env_id = information.get("train_args", {}).get("env_id", "CarRacing-v3")
    env = gym.make(env_id)
    session = None
    saved_revision = -1
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_metadata": information,
        "env_id": env_id,
        "untrained_demo": information.get("untrained_demo", False),
    }

    def save():
        nonlocal saved_revision
        if not (session.history or session.archived):
            return "No transitions to save yet."
        count = session.save(output, metadata)
        saved_revision = session.revision
        logger.info("Saved {} episodes to {}", count, output)
        return f"Saved {count} episode(s) to {output}"

    try:
        logger.info(
            "Preparing live posterior and diagnostics (first call compiles JAX functions)"
        )
        session = InspectorSession(models, env, seed=seed, horizon=horizon)
        from dreamer.interactive_viewer import (
            InspectorControls,
            InspectorRenderer,
            run_inspector,
        )

        if headless_steps is None:
            run_inspector(
                session,
                save=save,
                checkpoint_label=f"{checkpoint.parent.name}/{checkpoint.name}",
                fps=fps,
            )
        else:
            import pygame

            pygame.init()
            for _ in range(headless_steps):
                if session.done:
                    break
                session.step(session.actor_action(), source="actor")
            start = max(0, len(session.history) - horizon)
            if session.history:
                session.compare_returns(start)
            renderer = InspectorRenderer()
            surface = renderer.draw(
                session,
                InspectorControls(
                    actor_control=True,
                    pending=not bool(session.history),
                    selection=start if session.history else None,
                ),
                checkpoint_label=str(checkpoint),
            )
            if screenshot is not None:
                screenshot.parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(surface, screenshot)
                logger.info("Saved screenshot to {}", screenshot)
    finally:
        try:
            if session is not None and session.revision != saved_revision:
                save()
        finally:
            env.close()
            import pygame

            pygame.quit()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
