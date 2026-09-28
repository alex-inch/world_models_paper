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
    conts: Float[Array, "Capacity Time"]
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


def init_buffer(
    capacity: int = 100,
    seq_length: int = 1001,
    action_dim: int = 3,
) -> BufferState:
    N, T, H, W, C = capacity, seq_length, 64, 64, 3

    return BufferState(
        cursor=jnp.asarray(0, dtype=jnp.int32),
        full=jnp.asarray(False),
        obs=jnp.zeros((N, T, H, W, C), dtype=jnp.float32),
        acts=jnp.zeros((N, T, action_dim), dtype=jnp.float32),
        conts=jnp.zeros((N, T), dtype=jnp.float32),
        rewards=jnp.zeros((N, T), dtype=jnp.float32),
    )


@jax.jit
def add_episodes(
    state: BufferState,
    obs: Float[Array, "B T H W C"],
    acts: Float[Array, "B T A"],
    conts: Float[Array, "B T"],
    rewards: Float[Array, "B T"],
) -> BufferState:
    chex.assert_type(obs, float)
    chex.assert_equal_shape_prefix([obs, acts, conts, rewards], prefix_len=2)
    chex.assert_rank(obs, 5)

    batch_size = obs.shape[0]
    capacity = state.capacity

    end = state.cursor + batch_size

    # We can't write this as arange(start, end), because if both the start and end are traced, JAX
    # doesn't know the size. This way, it's clearly fixed to `batch_size`.
    dt = state.cursor.dtype
    indices = (state.cursor + jnp.arange(batch_size, dtype=dt)) % capacity

    return state.replace(  # pyright: ignore[reportAttributeAccessIssue]
        cursor=end % capacity,
        full=jnp.logical_or(state.full, end >= capacity),
        obs=state.obs.at[indices].set(obs),
        acts=state.acts.at[indices].set(acts),
        conts=state.conts.at[indices].set(conts),
        rewards=state.rewards.at[indices].set(rewards),
    )


@jax.jit(static_argnames=["batch_size", "seq_length"])
def get_batch(state: BufferState, batch_size: int, seq_length: int, key: Array):
    # First we select episodes. For each episode, selected, sample a sequence uniformly in the range
    # between [0,ep_end], where ep_end is marked by the first instance of the continue value going to
    # zero (cont looks like [1,1,1,0,0,0])
    ep_key, seq_key = jax.random.split(key)

    episode_indices = jax.random.randint(
        ep_key, shape=(batch_size,), minval=0, maxval=state.num_eps
    )

    T = state.episode_length
    selected_conts = state.conts[episode_indices]
    # The first zero marks the end of the episode.
    episode_ends = jnp.min(
        jnp.where(selected_conts == 0, jnp.arange(T), T - 1), axis=-1
    )

    seq_starts = jax.random.randint(
        seq_key,
        shape=(batch_size,),
        minval=0,
        maxval=episode_ends - seq_length + 2,
    )

    # This is a bit vestigial. The original Dreamer implementation uses it to pack multiple sequences into one
    # batch, but my implementation only uses it for the first steps of the batch
    is_first = jnp.zeros((batch_size, seq_length), dtype=jnp.bool)
    is_first = is_first.at[:, 0].set(True)

    # [B, L]
    time_indices = seq_starts[:, None] + jnp.arange(seq_length)[None, :]
    # [B, 1] - broadcasts against [B, L] when we used these to slice the arrays
    batch_indices = episode_indices[:, None]

    return DreamerBatch(
        image=state.obs[batch_indices, time_indices],
        prev_action=state.acts[batch_indices, time_indices],
        cont=state.conts[batch_indices, time_indices],
        reward=state.rewards[batch_indices, time_indices],
        is_first=is_first,
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
        rollout_obs = np.zeros(shape=(num_envs, 1001, 64, 64, 3))
        rollout_acts = np.zeros(shape=(num_envs, 1001, 3))
        rollout_conts = np.zeros(shape=(num_envs, 1001))
        rollout_rewards = np.zeros(shape=(num_envs, 1001))

        obs, _ = envs.reset(seed=ep_ix * 250)
        # The first entry for each episode is the initial state. Note we're implicitly setting the initial action and reward to zero, these
        # correspond to the action and reward "before" the initial state, so that's fine.
        rollout_obs[:, 0] = normalise_obs(crop_obs(obs))
        rollout_conts[:, 0] = 1

        for t in range(1, max_timesteps + 1):
            subkey, key = jax.random.split(key)
            acts = policy(obs, subkey)
            obs, rwds, terminated, truncated, _infos = envs.step(np.array(acts))

            # We store the action a_t and the following observation o_t+1 in the same time step, since that's what
            # we'll what to exist in a single batch later when we train the model
            rollout_obs[:, t] = normalise_obs(crop_obs(obs))
            rollout_acts[:, t] = acts
            # store raw reward, transform happens in the loss calc
            rollout_rewards[:, t] = rwds

            # Add the episode end to train the continue predictor
            episode_ended = np.logical_or(terminated, truncated)
            rollout_conts[:, t] = np.where(episode_ended, 0, 1)

        buffer = add_episodes(
            buffer,
            rollout_obs,
            rollout_acts,
            rollout_conts,
            rollout_rewards,
        )

    return buffer
