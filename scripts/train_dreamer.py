from dataclasses import dataclass

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax import nnx
from jax import Array
from jaxtyping import Float

from dreamer.actor_critic import ActorCritic
from dreamer.buffer import BufferState, add_episodes, get_batch, init_buffer
from dreamer.imagine import ImaginationTrajectory, imagination_rollout
from dreamer.wm import DreamerBatch, DreamerWM
from dreamer.wm.config import DreamerConfig


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
    imagination_horizon: int = 10
    lr: float = 2e-4


def wm_train_step(wm: DreamerWM): ...


def policy_train_step(ac: ActorCritic): ...


@nnx.value_and_grad(has_aux=True)
def wm_loss_fn(model: DreamerWM, batch: DreamerBatch):
    preds = model(batch)
    losses = model.loss_fn(preds, batch, model.cfg)
    loss = (
        model.cfg.kl_beta * losses.kl_loss
        + losses.continue_loss
        + losses.reward_loss
        + losses.image_loss
    )
    return loss, losses


@nnx.value_and_grad(has_aux=True)
def ac_loss_fn(
    model: ActorCritic,
    rewards: Float[Array, "B T"],
    conts: Float[Array, "B T"],
    rollout: ImaginationTrajectory,
):
    losses = model.loss_fn(
        model.cfg, rewards, conts, rollout.value, rollout.action_dist
    )

    # Split out the total loss (sum of the actor and critic losses) while returning the object
    # which contains all the terms for posterity
    return losses.loss, losses


# @nnx.jit(donate_argnames=['world_model', 'wm_optim', 'actor_critic', 'ac_optim'])
def model_learning_step(
    world_model: DreamerWM,
    wm_optim: nnx.Optimizer,
    actor_critic: ActorCritic,
    ac_optim: nnx.Optimizer,
    replay_buffer: BufferState,
    key: Array,
):
    # World model learning
    batch_key, imagination_key = jax.random.split(key)
    batch = get_batch(replay_buffer, args.batch_size, args.seq_length, batch_key)
    (_wm_loss, wm_sub_losses), wm_grads = wm_loss_fn(world_model, batch)
    wm_optim.update(world_model, wm_grads)

    # Behaviour Learning
    # TODO(Alex): Should we run this N times, selecting different starting states in the batch?
    # TODO(Alex): I think maybe this isn't totally right? Should we run up to this point to prime the GRU?
    #             Just thinking that the images might not be Markovian
    initial_states = world_model.init_imagination_state(batch.images[:, 0])
    imagination_key, key = jax.random.split(key)
    rollout, rewards, conts = imagination_rollout(
        world_model,
        actor_critic,
        initial_states,
        horizon=args.imagination_horizon,
        key=imagination_key,
    )
    (_ac_loss, ac_sub_losses), ac_grads = ac_loss_fn(
        world_model.cfg, rewards, conts, rollout.value, rollout.action_dist
    )
    ac_optim.update(actor_critic, ac_grads)

    return {"wm_losses": wm_sub_losses, "ac_losses": ac_sub_losses}


# @nnx.jit()
def get_action(
    world_model: DreamerWM, actor_critic: ActorCritic, batch: DreamerBatch, action_key
):
    wm_state = world_model.run_rssm(batch)
    ac_pred = actor_critic(wm_state)
    action_sample = ac_pred.action_dist.sample(seed=action_key)
    return action_sample


def main(args: TrainArgs):
    envs = gym.make_vec(
        id=args.env_id, num_envs=args.num_envs, vectorization_mode="async"
    )

    replay_buffer = init_buffer(capacity=args.buffer_size)

    # Init the models and their optimisers. For now just use a shared learning rate
    world_model = DreamerWM(cfg=args.cfg, rngs=nnx.Rngs(20))
    wm_optim = nnx.Optimizer(world_model, tx=optax.adam(args.lr), wrt=nnx.Param)
    actor_critic = ActorCritic(cfg=args.cfg, rngs=nnx.Rngs(21))
    ac_optim = nnx.Optimizer(actor_critic, tx=optax.adam(args.lr), wrt=nnx.Param)

    key = jax.random.key(42)

    for i in range(args.outer_train_steps):
        # Model Learning
        for update_step in range(args.collect_interval):
            # Dynamics Learning
            key, subkey = jax.random.split(key)
            losses = model_learning_step(
                world_model, wm_optim, actor_critic, ac_optim, replay_buffer, subkey
            )

        # Environment Interaction
        obs, _infos = envs.reset()
        # construct fake batch for the first step
        action = jnp.zeros(shape=(envs.num_envs, world_model.cfg.action_dim))
        reward = jnp.zeros(shape=(envs.num_envs,))
        cont = jnp.ones(shape=(envs.num_envs,))
        is_first = jnp.ones(shape=(envs.num_envs,)).astype(jnp.bool)
        batch = DreamerBatch(obs, reward, cont, action, is_first)

        for t in range(1000):
            action_key, key = jax.random.split(key)
            action_sample = get_action(world_model, batch, action, action_key)
            action_sample = np.array(action_sample)  # convert from JAX
            _next_obs, rewards, terminations, truncations, _infos = envs.step(
                action_sample
            )

            # If terminated or truncated 0, otherwise 1
            cont = 1 - jnp.logical_or(terminations, truncations).astype(jnp.int32)

            replay_buffer = add_episodes(replay_buffer, obs, action, cont, rewards)


if __name__ == "__main__":
    args = tyro.cli(TrainArgs)
    main(args)
