from dataclasses import asdict
from pprint import pformat

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from einops import rearrange
from flax import nnx
from jax import Array
from jax.lax import stop_gradient as sg
from jaxtyping import Float
from loguru import logger

import wandb
from dreamer.actor_critic import Actor, Critic
from dreamer.buffer import (
    BufferState,
    collect_new_rollouts,
    get_batch,
    init_buffer,
)
from dreamer.imagine import run_imagination_rollout
from dreamer.wm import DreamerBatch, DreamerWM
from dreamer.wm.config import DreamerConfig
from dreamer.wm.rssm import RSSMOut


@chex.dataclass(frozen=True)  # freeze so it's hashable
class TrainArgs:
    # Run config
    name: str
    cfg: DreamerConfig
    # Env config
    seed: int = 20
    num_envs: int = 8
    env_id: str = "CarRacing-v3"
    # Training hparams
    buffer_size: int = 100
    outer_train_steps: int = 100
    batch_size: int = 16
    seq_length: int = 25
    lr: float = 1e-4
    adamw_eps: float = 1e-5
    collect_interval: int = 10
    imagination_horizon: int = 15
    steps_per_eval = 5
    num_eval_episodes: int = 64
    debug: bool = False


@nnx.value_and_grad(has_aux=True)
def wm_loss_fn(wm: DreamerWM, batch: DreamerBatch):
    preds = wm(batch)
    losses = wm.loss_fn(preds, batch, wm.cfg)
    loss = (
        wm.cfg.kl_beta * losses.kl_loss
        + losses.continue_loss
        + losses.reward_loss
        + losses.image_loss
    )
    return loss, losses


@nnx.value_and_grad(has_aux=True)
def actor_loss_fn(
    actor: Actor,
    target_critic: Critic,
    wm: DreamerWM,
    initial_states: RSSMOut,  # [B H], [B Z]
    horizon: int,
    key: Array,
):
    rollout, rewards, conts = run_imagination_rollout(
        wm,
        actor,
        target_critic,
        sg(initial_states),
        horizon=horizon,
        key=key,
    )
    losses, targets, weights = actor.loss_fn(
        actor.cfg, rewards, conts, rollout.value, rollout.action_dist
    )
    # Split out the data for the critic to use in its loss function
    critic_data = (sg(rollout.state.joint_latent[:, :-1]), sg(targets), weights)
    # which contains all the terms for posterity
    return losses.loss, (losses, critic_data)


@nnx.value_and_grad
def critic_loss_fn(
    critic: Critic,
    features: Float[Array, "B T D"],
    targets: Float[Array, "B T"],
    weights: Float[Array, "B T"],
):
    values = critic(features)
    return critic.loss_fn(values, targets, weights)


@nnx.jit(
    donate_argnames=[
        "wm",
        "wm_optim",
        "actor",
        "actor_optim",
        "critic",
        "critic_optim",
        "target_critic",
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
    batch = get_batch(replay_buffer, args.batch_size, args.seq_length, batch_key)

    # World model learning
    (wm_loss, wm_sub_losses), wm_grads = wm_loss_fn(wm, batch)
    wm_optim.update(wm, wm_grads)

    # Behaviour learning
    model_states = wm.run_rssm(batch)
    model_states = model_states.replace(posterior=None)  # pyrefly: ignore[missing-attribute]

    # Concat the batch and time dimensions. We'll run the actor in imagination across
    # all of these in parallel.
    initial_states = jax.tree.map(
        lambda x: rearrange(x, "B T ... -> (B T) ..."), model_states
    )
    aux, actor_grads = actor_loss_fn(
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

    critic_loss, critic_grads = critic_loss_fn(critic, states, targets, weights)
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
    return {
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


def format_rollout_gif_data(observations: Float[Array, "T H W C"], fps: int = 50):
    "Get a jax array of frames into shape for wandb upload"
    # OG is 50fps - we can downsample a bit to save space
    frame_skip = 50 // fps
    if 50 % fps != 0:
        logger.warning(
            f"Provided eval GIF FPS {fps} does not neatly divide env fps {50}, using fps {int(50 / frame_skip)}"
        )
    frames = np.asarray(observations[::frame_skip])
    frames_rescaled = 255 * (frames + 0.5)
    frames_rescaled = np.clip(frames_rescaled, 0, 255).astype(np.uint8)
    return rearrange(frames_rescaled, "T H W C -> T C H W")


def eval_step(
    envs: gym.vector.VectorEnv,
    world_model: DreamerWM,
    actor: Actor,
    num_eps: int,
    fps: int = 10,
):
    # TODO(Alex): Add imagination gif
    # TODO(Alex): Add reconstruction gif
    logger.info("Beginning eval step")
    key = jax.random.key(500_000)  # Fixed seed for all eval renders
    buffer = init_buffer(capacity=num_eps)
    buffer = collect_new_rollouts(
        buffer, envs, world_model, actor, num_eps=num_eps, key=key
    )

    rollout = format_rollout_gif_data(buffer.obs[0], fps)

    rewards = buffer.rewards  # [T H]
    returns = rewards.sum(axis=1)

    logger.info(f"Eval step completed with mean return {returns.mean():.2f}")
    return {
        "eval/return_mean": float(returns.mean()),
        "eval/return_median": float(np.median(returns)),
        "eval/return_std": float(returns.std()),
        "eval/returns": wandb.Histogram(returns, num_bins=16),  # type: ignore
        "eval/policy": wandb.Video(
            rollout,
            caption=f"Mean Return: {float(returns.mean()):.0f}",
            fps=fps,
            format="gif",
        ),
    }


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
    replay_buffer = init_buffer(capacity=args.buffer_size)

    # Init the models and their optimisers. For now just use a shared learning rate
    # fmt: off
    wm           = DreamerWM(cfg=args.cfg, rngs=nnx.Rngs(args.seed + 1))
    actor        = Actor(cfg=args.cfg,     rngs=nnx.Rngs(args.seed + 2))
    critic       = Critic(cfg=args.cfg,    rngs=nnx.Rngs(args.seed + 3))
    target_critic = nnx.clone(critic, arrays=True) # slow version of the critic

    tx = optax.chain(
        optax.clip_by_global_norm(100),
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

        # Model Learning
        logger.info(f"Step {outer_step + 1}: Commencing model learning")
        for inner_step in range(args.collect_interval):
            # Dynamics Learning
            subkey, key = jax.random.split(key)
            metrics = model_learning_step(
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
            wandb_metrics = jax.device_get(metrics)
            targets = wandb_metrics.pop("ac/targets")
            wandb_metrics["ac/target_mean"] = float(targets.mean())
            train_step = outer_step * args.collect_interval + inner_step + 1
            wandb_metrics["train_step"] = train_step
            run.log(wandb_metrics)

            logger.info(f"""Substep {inner_step + 1} complete with key metrics:
      WM Loss: {metrics["wm/losses/loss"]:.3f}
   Actor Loss: {metrics["ac/actor/loss"]:.3f}
  Critic Loss: {metrics["ac/critic/loss"]:.3f}""")

        if outer_step % args.steps_per_eval == 0:
            eval_metrics = eval_step(envs, wm, actor, args.num_eval_episodes)
            run.log(eval_metrics)

    # Final run of the eval at the end
    eval_metrics = eval_step(envs, wm, actor, args.num_eval_episodes)
    run.log(eval_metrics)


def main(args: TrainArgs):
    args, run = setup_logging(args)
    args_pprint = pformat(asdict(args), sort_dicts=False)
    logger.info(f"Beginning training run with args:\n{args_pprint}")

    try:
        envs = gym.make_vec(
            id=args.env_id, num_envs=args.num_envs, vectorization_mode="async"
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
