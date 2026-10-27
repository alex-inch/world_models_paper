from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from time import perf_counter
from typing import Literal

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from einops import rearrange
from flax import nnx
from jax import Array
from loguru import logger

import wandb
from dreamer.actor_critic import Actor, Critic
from dreamer.buffer import (
    BufferState,
    collect_new_rollouts,
    get_batch,
    init_buffer,
    refresh_model_states,
)
from dreamer.checkpoints import save_checkpoint
from dreamer.diagnostics import evaluate_rollouts, new_episode_cumulative_rewards
from dreamer.training import actor_objective, critic_objective, wm_objective
from dreamer.wm import DreamerBatch, DreamerWM
from dreamer.wm.config import DreamerConfig


@chex.dataclass(frozen=True)  # freeze so it's hashable
class TrainArgs:
    # Run config
    name: str
    cfg: DreamerConfig
    backend: Literal["mps", "cpu", "cuda"] | None = None
    # Env config
    seed: int = 20
    num_envs: int = 8
    env_id: str = "CarRacing-v3"
    # Training hparams
    buffer_size: int = 100
    outer_train_steps: int = 100
    batch_size: int = 16
    seq_length: int = 25
    lr: float = 3e-4
    adamw_eps: float = 1e-5
    collect_interval: int = 100
    imagination_horizon: int = 15
    steps_per_eval: int = 1
    num_eval_episodes: int = 64
    debug: bool = False
    checkpoint_dir: str = "experiments/dreamer/checkpoints"


wm_value_and_grad = nnx.value_and_grad(wm_objective, has_aux=True)
actor_value_and_grad = nnx.value_and_grad(actor_objective, has_aux=True)
critic_value_and_grad = nnx.value_and_grad(critic_objective)


@nnx.jit(
    donate_argnames=[
        "wm",
        "wm_optim",
        "actor",
        "actor_optim",
        "critic",
        "critic_optim",
        "target_critic",
        "replay_buffer",
    ],
    static_argnames="args",
)
def model_learning_step(
    wm: DreamerWM,
    wm_optim: nnx.Optimizer,
    actor: Actor,
    actor_optim: nnx.Optimizer,
    critic: Critic,
    critic_optim: nnx.Optimizer,
    target_critic: Critic,
    replay_buffer: BufferState,
    args: TrainArgs,
    key: Array,
):
    batch_key, imagination_key = jax.random.split(key)
    batch, ctxt = get_batch(replay_buffer, args.batch_size, args.seq_length, batch_key)

    # World model learning
    initial_state = ctxt.initial_state
    (wm_loss, wm_sub_losses), wm_grads = wm_value_and_grad(wm, batch, initial_state)
    wm_optim.update(wm, wm_grads)

    # Behaviour learning
    model_states = wm.run_rssm(batch, initial_state)
    # Note - to keep the stored RSSM states fresh, we update the buffer with whatever we've computed
    replay_buffer = refresh_model_states(replay_buffer, ctxt, model_states)
    model_states = model_states.replace(posterior=None)  # pyrefly: ignore[missing-attribute]

    # Concat the batch and time dimensions. We'll run the actor in imagination across
    # all of these in parallel.
    initial_states = jax.tree.map(
        lambda x: rearrange(x, "B T ... -> (B T) ..."), model_states
    )
    aux, actor_grads = actor_value_and_grad(
        actor,
        target_critic,
        wm,
        initial_states,
        horizon=args.imagination_horizon,
        key=imagination_key,
    )
    actor_optim.update(actor, actor_grads)
    actor_loss, (actor_losses, critic_data) = aux
    states, targets, weights = critic_data

    critic_loss, critic_grads = critic_value_and_grad(critic, states, targets, weights)
    critic_optim.update(critic, critic_grads)

    # Update the target network
    nnx.update(
        target_critic,
        optax.incremental_update(
            new_tensors=nnx.state(critic, nnx.Param),
            old_tensors=nnx.state(target_critic, nnx.Param),
            step_size=0.02,
        ),
    )

    # Some bookkeeping variables to track training
    metrics = {
        # WM Variables
        "wm/losses/loss": wm_loss,
        "wm/losses/image_loss": wm_sub_losses.image_loss,
        "wm/losses/reward_loss": wm_sub_losses.reward_loss,
        "wm/losses/continue_loss": wm_sub_losses.continue_loss,
        "wm/losses/kl_loss": wm_sub_losses.kl_loss,
        "wm/grad_norm": optax.tree.norm(wm_grads),
        "wm/grad_max": optax.tree.norm(wm_grads, jnp.inf),
        # Actor Critic Variables
        "ac/targets": targets,
        "ac/actor/loss": actor_loss,
        "ac/actor/reward_loss": actor_losses.reward_loss,
        "ac/actor/entropy_loss": actor_losses.entropy_loss,
        "ac/actor/grad_norm": optax.tree.norm(actor_grads),
        "ac/actor/grad_max": optax.tree.norm(actor_grads, jnp.inf),
        "ac/critic/loss": critic_loss,
        "ac/critic/grad_norm": optax.tree.norm(critic_grads),
        "ac/critic/grad_max": optax.tree.norm(critic_grads, jnp.inf),
    }

    return metrics, replay_buffer


def eval_step(
    envs: gym.vector.VectorEnv,
    world_model: DreamerWM,
    actor: Actor,
    critic: Critic,
    target_critic: Critic,
    num_eps: int,
    fps: int = 10,
    imagination_horizon: int = 15,
    fixed_batch: DreamerBatch | None = None,
):
    logger.info("Beginning eval step")
    world_model = nnx.clone(world_model, arrays=True)
    nnx.reseed(world_model, default=500_000)
    key = jax.random.key(500_000)  # Fixed seed for all eval renders
    collection_key, imagination_key = jax.random.split(key)
    buffer = init_buffer(capacity=num_eps, cfg=world_model.cfg)
    buffer = collect_new_rollouts(
        buffer, envs, world_model, actor, num_eps=num_eps, key=collection_key
    )

    metrics, fixed_batch = evaluate_rollouts(
        buffer,
        world_model,
        actor,
        critic,
        target_critic,
        imagination_key,
        fps=fps,
        imagination_horizon=imagination_horizon,
        fixed_batch=fixed_batch,
    )
    logger.info(
        f"Eval step completed with mean return {metrics['eval/return_mean']:.2f}"
    )
    return metrics, fixed_batch


def setup_logging(args: TrainArgs):
    if args.debug:  # also disables wandb logging
        logger.warning("Running in debug mode.")
        args = args.replace(  # pyrefly: ignore[missing-attribute]
            outer_train_steps=1,
            collect_interval=1,
            batch_size=3,
            seq_length=5,
            num_envs=2,
            num_eval_episodes=2,
        )

    # WANDB setup
    run = wandb.init(
        name=args.name,
        project="dreamer-reimplementation",
        config=asdict(args),
        mode="disabled" if args.debug else "online",
    )
    run.define_metric("train_step")
    run.define_metric("wm/*", step_metric="train_step")
    run.define_metric("ac/*", step_metric="train_step")
    run.define_metric("eval/*", step_metric="train_step")
    return args, run


def train(args: TrainArgs, envs: gym.vector.VectorEnv, run: wandb.Run):
    replay_buffer = init_buffer(capacity=args.buffer_size, cfg=args.cfg)

    # Init the models and their optimisers. For now just use a shared learning rate
    # fmt: off
    wm           = DreamerWM(cfg=args.cfg, rngs=nnx.Rngs(args.seed + 1))
    actor        = Actor(cfg=args.cfg,     rngs=nnx.Rngs(args.seed + 2))
    critic       = Critic(cfg=args.cfg,    rngs=nnx.Rngs(args.seed + 3))
    target_critic = nnx.clone(critic, arrays=True) # slow version of the critic

    tx = optax.chain(
        optax.transforms.adaptive_grad_clip(0.3),
        optax.adamw(args.lr, eps=args.adamw_eps),
    )
    wm_optim     = nnx.Optimizer(wm,     tx=tx, wrt=nnx.Param)
    actor_optim  = nnx.Optimizer(actor,  tx=tx, wrt=nnx.Param)
    critic_optim = nnx.Optimizer(critic, tx=tx, wrt=nnx.Param)
    # fmt: on

    parameter_counts = {
        name: sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
        for name, model in (("world_model", wm), ("actor", actor), ("critic", critic))
    }
    for name, count in parameter_counts.items():
        logger.info(f"{name}: {count:,} trainable parameters")
    run.config.update({"parameter_counts": parameter_counts})

    models = {
        "world_model": wm,
        "actor": actor,
        "critic": critic,
        "target_critic": target_critic,
    }
    checkpoint_root = Path(args.checkpoint_dir) / run.id
    checkpoint_metadata = {
        "train_args": asdict(args),
        "wandb_run_id": run.id,
        "train_step": 0,
        "env_steps": 0,
        "elapsed_seconds": 0.0,
    }
    save_checkpoint(checkpoint_root / "00000000", models, args.cfg, checkpoint_metadata)
    logger.info(f"Saving model checkpoints in {checkpoint_root}")
    last_checkpoint_step = 0
    train_step = 0
    fixed_eval_batch = None
    started = perf_counter()
    key = jax.random.key(args.seed)

    # Train loop
    for outer_step in range(args.outer_train_steps):
        # Environment Interaction
        action_key, key = jax.random.split(key)

        logger.info(f"Step {outer_step + 1}: Commencing data collection")
        replay_buffer = collect_new_rollouts(
            replay_buffer,
            envs,
            wm,
            actor,
            num_eps=args.num_envs,
            key=action_key,
        )
        # log the mean reward of the collected episodes
        mean_rwd, median_rwd = new_episode_cumulative_rewards(
            replay_buffer, args.num_envs
        )
        run.log({
            "eval/episodic_reward/train/mean": mean_rwd,
            "eval/episodic_reward/train/median": median_rwd,
            "train_step": train_step,
        })

        # Model Learning
        logger.info(f"Step {outer_step + 1}: Commencing model learning")
        for inner_step in range(args.collect_interval):
            # Dynamics Learning
            subkey, key = jax.random.split(key)
            metrics, replay_buffer = model_learning_step(
                wm,
                wm_optim,
                actor,
                actor_optim,
                critic,
                critic_optim,
                target_critic,
                replay_buffer,
                args,
                subkey,
            )

            # Bookkeeping - wandb needs the targets to be a scalar
            targets = metrics.pop("ac/targets")
            metrics["ac/target_mean"] = float(targets.mean())
            train_step = outer_step * args.collect_interval + inner_step + 1
            metrics["train_step"] = train_step
            run.log(metrics)

        #           logger.info(f"""Substep {inner_step + 1} complete with key metrics:
        #     WM Loss: {metrics["wm/losses/loss"]:.3f}
        #  Actor Loss: {metrics["ac/actor/loss"]:.3f}
        # Critic Loss: {metrics["ac/critic/loss"]:.3f}""")

        if outer_step % args.steps_per_eval == 0:
            checkpoint_metadata["train_step"] = train_step
            checkpoint_metadata["env_steps"] = (
                (outer_step + 1) * args.num_envs * (replay_buffer.episode_length - 1)
            )
            checkpoint_metadata["elapsed_seconds"] = perf_counter() - started
            save_checkpoint(
                checkpoint_root / f"{train_step:08d}",
                models,
                args.cfg,
                checkpoint_metadata,
            )
            last_checkpoint_step = train_step
            eval_metrics, fixed_eval_batch = eval_step(
                envs,
                wm,
                actor,
                critic,
                target_critic,
                args.num_eval_episodes,
                fixed_batch=fixed_eval_batch,
            )
            eval_metrics["train_step"] = train_step
            run.log(eval_metrics)

    # Final run of the eval at the end
    if train_step != last_checkpoint_step:
        checkpoint_metadata["train_step"] = train_step
        checkpoint_metadata["env_steps"] = (
            args.outer_train_steps * args.num_envs * (replay_buffer.episode_length - 1)
        )
        checkpoint_metadata["elapsed_seconds"] = perf_counter() - started
        save_checkpoint(
            checkpoint_root / f"{train_step:08d}", models, args.cfg, checkpoint_metadata
        )
    eval_metrics, _ = eval_step(
        envs,
        wm,
        actor,
        critic,
        target_critic,
        args.num_eval_episodes,
        fixed_batch=fixed_eval_batch,
    )
    eval_metrics["train_step"] = train_step
    run.log(eval_metrics)


def main(args: TrainArgs):
    if args.backend is not None:
        jax.config.update("jax_platforms", args.backend)
    logger.info(f"Using JAX devices: {jax.devices()}")

    args, run = setup_logging(args)
    args_pprint = pformat(asdict(args), sort_dicts=False)
    logger.info(f"Beginning training run with args:\n{args_pprint}")

    try:
        envs = gym.make_vec(
            id=args.env_id,
            num_envs=args.num_envs,
            vectorization_mode="async",
            vector_kwargs={"context": "spawn"},
        )
        train(args, envs, run)
    except BaseException:  # BaseException is normally wrong, but ok cos we re-raise
        run.finish(exit_code=1)
        raise
    finally:
        envs.close()
    # If no exceptions
    run.finish(exit_code=0)


if __name__ == "__main__":
    import tyro

    args = tyro.cli(TrainArgs)
    main(args)
