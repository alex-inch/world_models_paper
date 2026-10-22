from dataclasses import dataclass

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array
from jax.lax import stop_gradient as sg
from jaxtyping import Float
from tqdm import trange

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


@dataclass
class TrainArgs:
    cfg: DreamerConfig
    num_envs: int = 8
    env_id: str = "CarRacing-v3"
    buffer_size: int = 100
    outer_train_steps: int = 100
    batch_size: int = 16
    collect_interval: int = 10
    seq_length: int = 50
    imagination_horizon: int = 15
    lr: float = 2e-4
    adamw_eps: float = 1e-5


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
    critic: Critic,
    wm: DreamerWM,
    initial_states: RSSMOut,  # [B H], [B Z]
    horizon: int,
    key: Array,
):
    rollout, rewards, conts = run_imagination_rollout(
        wm,
        actor,
        critic,
        initial_states,
        horizon=horizon,
        key=key,
    )
    losses, targets = actor.loss_fn(
        actor.cfg, rewards, conts, rollout.value, rollout.action_dist
    )
    # Split out the data for the critic to use in its loss function
    critic_data = (sg(rollout.state.joint_latent[:, :-1]), sg(targets))
    # which contains all the terms for posterity
    return losses.loss, (losses, critic_data)


@nnx.value_and_grad
def critic_loss_fn(
    critic: Critic,
    features: Float[Array, "B T D"],
    targets: Float[Array, "B T"],
):
    values = critic(features)
    return critic.loss_fn(values, targets)


# @nnx.jit(donate_argnames=['wm', 'wm_optim', 'actor', 'actor_optim', 'critic', 'critic_optim'])
def model_learning_step(
    wm: DreamerWM,
    wm_optim: nnx.Optimizer,
    actor: Actor,
    actor_optim: nnx.Optimizer,
    critic: Critic,
    critic_optim: nnx.Optimizer,
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
    batch_model_states = wm.run_rssm(batch)
    batch_model_states = batch_model_states.replace(posterior=None)  # pyrefly: ignore[missing-attribute]

    initial_states = jax.tree.map(lambda x: sg(x[:, -1]), batch_model_states)
    aux, actor_grads = actor_loss_fn(
        actor,
        critic,
        wm,
        initial_states,
        horizon=args.imagination_horizon,
        key=imagination_key,
    )
    actor_optim.update(actor, actor_grads)
    actor_loss, (actor_losses, critic_data) = aux
    states, targets = critic_data

    critic_loss, critic_grads = critic_loss_fn(critic, states, targets)
    critic_optim.update(critic, critic_grads)

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


def main(args: TrainArgs):
    envs = gym.make_vec(
        id=args.env_id, num_envs=args.num_envs, vectorization_mode="async"
    )

    replay_buffer = init_buffer(capacity=args.buffer_size)

    # Init the models and their optimisers. For now just use a shared learning rate
    # fmt: off
    wm           = DreamerWM(cfg=args.cfg, rngs=nnx.Rngs(20))
    actor        = Actor(cfg=args.cfg,     rngs=nnx.Rngs(21))
    critic       = Critic(cfg=args.cfg,    rngs=nnx.Rngs(22))

    tx = optax.adamw(args.lr, eps=args.adamw_eps)
    wm_optim     = nnx.Optimizer(wm,     tx=tx, wrt=nnx.Param)
    actor_optim  = nnx.Optimizer(actor,  tx=tx, wrt=nnx.Param)
    critic_optim = nnx.Optimizer(critic, tx=tx, wrt=nnx.Param)
    # fmt: on

    key = jax.random.key(42)

    all_losses = []
    for i in trange(args.outer_train_steps):
        # Environment Interaction
        action_key, key = jax.random.split(key)

        replay_buffer = collect_new_rollouts(
            replay_buffer,
            envs,
            wm,
            actor,
            num_eps=args.num_envs,
            key=action_key,
        )

        # Model Learning
        for update_step in range(args.collect_interval):
            # Dynamics Learning
            subkey, key = jax.random.split(key)
            step_losses = model_learning_step(
                wm,
                wm_optim,
                actor,
                actor_optim,
                critic,
                critic_optim,
                replay_buffer,
                args,
                subkey,
            )
            all_losses.append(step_losses)


if __name__ == "__main__":
    import tyro

    args = tyro.cli(TrainArgs)
    main(args)
