from pathlib import Path
from time import perf_counter

import gymnasium as gym
from loguru import logger

from dreamer.checkpoints import load_checkpoint
from dreamer.inspection import analyse_episode, record_episode, save_recording


def main(
    checkpoint: Path,
    output: Path,
    num_episodes: int = 2,
    max_steps: int = 1000,
    seed: int = 500_000,
):
    started = perf_counter()
    models, information = load_checkpoint(checkpoint)
    env_id = information["train_args"]["env_id"]
    envs = gym.make_vec(env_id, num_envs=1, vectorization_mode="sync")
    episodes = []
    try:
        for episode_index in range(num_episodes):
            episode = record_episode(
                models, envs, seed=seed + episode_index, max_steps=max_steps
            )
            episode = analyse_episode(models, episode)
            episodes.append(episode)
            logger.info(
                "Episode {}: {:.2f} reward over {} steps",
                episode_index,
                episode["next_rewards"].sum(),
                len(episode["images"]),
            )
    finally:
        envs.close()

    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_metadata": information,
        "seed": seed,
        "env_id": env_id,
        "elapsed_seconds": perf_counter() - started,
        "untrained_demo": information.get("untrained_demo", False),
    }
    save_recording(output, episodes, metadata)
    logger.info("Saved recording to {} in {:.1f}s", output, metadata["elapsed_seconds"])


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
