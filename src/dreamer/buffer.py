from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector import VectorEnv
from jaxtyping import Array, Bool, Float, Int
from loguru import logger

from wm.utils.obs import crop_obs, normalise_obs

from .wm.world_model import DreamerBatch


@chex.dataclass
class BufferState:
    cursor: Int[Array, ""]
    full: Bool[Array, ""]
    obs: Float[Array, "Capacity Time Height Width Channels"]
    acts: Float[Array, "Capacity Time Action"]
    discounts: Float[Array, "Capacity Time"]
    rewards: Float[Array, "Capacity Time"]

    @property
    def capacity(self) -> int:
        return self.obs.shape[0]

    @property
    def episode_length(self) -> int:
        return self.obs.shape[1]

    @property
    def num_eps(self) -> Array:
        return jnp.where(
            self.full,
            self.capacity,
            self.cursor,
        )


def init_buffer(capacity: int = 1000) -> BufferState:
    N, T, H, W, C = capacity, 1000, 64, 64, 3
    action_size = 3

    return BufferState(
        cursor=jnp.asarray(0, dtype=jnp.int32),
        full=jnp.asarray(False),
        obs=jnp.zeros((N, T, H, W, C), dtype=jnp.float32),
        acts=jnp.zeros((N, T, action_size), dtype=jnp.float32),
        discounts=jnp.zeros((N, T), dtype=jnp.float32),
        rewards=jnp.zeros((N, T), dtype=jnp.float32),
    )


@jax.jit
def add_episodes(
    state: BufferState,
    obs: Float[Array, "B T H W C"],
    acts: Float[Array, "B T A"],
    discounts: Float[Array, "B T"],
    rewards: Float[Array, "B T"],
) -> BufferState:
    # These are static assertions, so fine during JIT tracing.
    chex.assert_type(obs, float)
    chex.assert_equal_shape_prefix([obs, acts, discounts, rewards], prefix_len=2)
    chex.assert_rank(obs, 5)

    batch_size = obs.shape[0]
    capacity = state.capacity

    end = state.cursor + batch_size

    # We can't write this as one arange: if both the start and end are traced, JAX
    # doesn't know the size - instead it's clearly fixed to `batch_size`.
    dt = state.cursor.dtype
    indices = (state.cursor + jnp.arange(batch_size, dtype=dt)) % capacity

    return state.replace(  # pyright: ignore[reportAttributeAccessIssue]
        cursor=end % capacity,
        full=jnp.logical_or(state.full, end >= capacity),
        obs=state.obs.at[indices].set(obs),
        acts=state.acts.at[indices].set(acts),
        discounts=state.discounts.at[indices].set(discounts),
        rewards=state.rewards.at[indices].set(rewards),
    )


@jax.jit(static_argnames=["batch_size", "seq_length"])
def get_batch(state, batch_size, seq_length, key):
    ep_key, seq_key = jax.random.split(key)

    episode_indices = jax.random.randint(
        ep_key,
        shape=(batch_size,),
        minval=0,
        maxval=state.num_eps,
    )

    T = state.obs.shape[1]

    seq_starts = jax.random.randint(
        seq_key,
        shape=(batch_size,),
        minval=0,
        maxval=T - seq_length + 1,
    )

    # [B, L]
    time_indices = seq_starts[:, None] + jnp.arange(seq_length)[None, :]

    # [B, 1], broadcasts against [B, L]
    batch_indices = episode_indices[:, None]

    return DreamerBatch(
        image=state.obs[batch_indices, time_indices],
        prev_action=state.acts[batch_indices, time_indices],
        discount=state.discounts[batch_indices, time_indices],
        reward=state.rewards[batch_indices, time_indices],
        is_first=jnp.zeros((batch_size, seq_length), dtype=jnp.bool_),
    )


def collect_new_rollouts(
    buffer: BufferState,
    envs: VectorEnv,
    policy: Callable[[Array, Array], Array],
    num_eps: int,
    key: Array,
) -> BufferState:
    # Technically this function overfills if num_eps doesn't neatly divide by num_envs.
    num_envs = envs.num_envs
    assert envs.spec is not None
    assert envs.spec.max_episode_steps is not None
    max_timesteps = envs.spec.max_episode_steps
    logger.info(f"Collecting {num_eps} rollouts with policy '{policy.__name__}'")

    for ep_ix in range(0, num_eps, num_envs):
        rollout_obs = np.empty(shape=(num_envs, 1000, 64, 64, 3))
        rollout_acts = np.empty(shape=(num_envs, 1000, 3))
        rollout_discounts = np.empty(shape=(num_envs, 1000))
        rollout_rewards = np.empty(shape=(num_envs, 1000))

        obs, _ = envs.reset(seed=ep_ix * 250)

        for t in range(max_timesteps):
            subkey, key = jax.random.split(key)
            acts = policy(obs, subkey)
            obs, rwds, _terminated, _truncated, _infos = envs.step(np.array(acts))

            # TODO: process obs, rewards and acts properly
            # TODO: also don't really have a proper continue predictor atm...
            rollout_obs[:num_envs, t] = normalise_obs(crop_obs(obs))
            rollout_acts[:num_envs, t] = acts
            rollout_discounts[:num_envs, t] = 0.997
            rollout_rewards[:num_envs, t] = rwds[0]

        buffer = add_episodes(
            buffer,
            rollout_obs,
            rollout_acts,
            rollout_discounts,
            rollout_rewards,
        )

    return buffer
