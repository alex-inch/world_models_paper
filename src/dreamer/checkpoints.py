"""Save model snapshots for evaluation and inspection."""

import json
from dataclasses import asdict
from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx

from dreamer.actor_critic import Actor, Critic
from dreamer.wm import DreamerConfig, DreamerWM


def save_checkpoint(
    directory: str | Path,
    models: dict[str, nnx.Module],
    config: DreamerConfig,
    metadata: dict,
):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True)
    state = {name: nnx.state(model) for name, model in models.items()}

    # Finish writing before training reuses any donated model buffers.
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(directory / "models", state)
        checkpointer.wait_until_finished()

    information = {"model_config": asdict(config), **metadata}
    (directory / "metadata.json").write_text(json.dumps(information, indent=2))


def load_checkpoint(directory: str | Path):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "metadata.json").read_text())
    config = DreamerConfig(**metadata["model_config"])
    models = {
        "world_model": DreamerWM(config, rngs=nnx.Rngs(0)),
        "actor": Actor(config, nnx.Rngs(1)),
        "critic": Critic(config, nnx.Rngs(2)),
        "target_critic": Critic(config, nnx.Rngs(3)),
    }
    state = {name: nnx.state(model) for name, model in models.items()}
    with ocp.StandardCheckpointer() as checkpointer:
        restored = checkpointer.restore(directory / "models", target=state)

    for name, model in models.items():
        nnx.update(model, restored[name])
    return models, metadata
