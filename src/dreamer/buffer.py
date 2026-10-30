import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from gymnasium.vector import VectorEnv
from jaxtyping import Array, Bool, Float, Int
from loguru import logger

from dreamer.actor_critic import Actor
from dreamer.wm import DreamerConfig, DreamerWM
from dreamer.wm.rssm import RSSMOut
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
    model_state_hidden: Float[Array, "Capacity Time HiddenDim"]
    model_state_latent: Float[Array, "Capacity Time LatentDim"]

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


@chex.dataclass
class ReplayContext:
    initial_state_hidden: Float[Array, "Batch HiddenDim"]
    initial_state_latent: Float[Array, "Batch LatentDim"]
    batch_indices: Int[Array, " Batch #Time"]
    time_indices: Int[Array, "Batch Time"]

    @property
    def initial_state(
        self,
    ) -> tuple[Float[Array, "Batch HiddenDim"], Float[Array, "Batch LatentDim"]]:
        return self.initial_state_hidden, self.initial_state_latent


def init_buffer(
    capacity: int = 100,
    seq_length: int = 1001,
    action_dim: int = 3,
    *,
    cfg: DreamerConfig,
) -> BufferState:
    N, T, H, W, C = capacity, seq_length, 64, 64, 3

    return BufferState(
        cursor=jnp.asarray(0, dtype=jnp.int32),
        full=jnp.asarray(False),
        obs=jnp.zeros((N, T, H, W, C), dtype=cfg.dtype),
        acts=jnp.zeros((N, T, action_dim), dtype=cfg.dtype),
        conts=jnp.zeros((N, T), dtype=cfg.dtype),
        rewards=jnp.zeros((N, T), dtype=jnp.float32),
        model_state_hidden=jnp.zeros((N, T, cfg.wm_hidden_dim), dtype=cfg.dtype),
        model_state_latent=jnp.zeros((N, T, cfg.latent_dim), dtype=cfg.dtype),
    )


@jax.jit
def add_episodes(
    state: BufferState,
    obs: Float[Array, "B T H W C"],
    acts: Float[Array, "B T A"],
    conts: Float[Array, "B T"],
    rewards: Float[Array, "B T"],
    model_state_hidden: Float[Array, "B T H"],
    model_state_latent: Float[Array, "B T Z"],
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

    return state.replace(  # pyrefly: ignore[missing-attribute] - .replace is synthesised so missed by static analysis
        cursor=end % capacity,
        full=jnp.logical_or(state.full, end >= capacity),
        obs=state.obs.at[indices].set(obs),
        acts=state.acts.at[indices].set(acts),
        conts=state.conts.at[indices].set(conts),
        rewards=state.rewards.at[indices].set(rewards),
        model_state_hidden=state.model_state_hidden.at[indices].set(model_state_hidden),
        model_state_latent=state.model_state_latent.at[indices].set(model_state_latent),
    )


@jax.jit
def refresh_model_states(
    buffer: BufferState,
    context: ReplayContext,
    states: RSSMOut,
) -> BufferState:
    # note that we want the *previous* model state associated with each step in the batch.
    # To that end: offset by one.
    positions = context.batch_indices, context.time_indices[:, 1:]
    sg = jax.lax.stop_gradient

    # fmt: off
    new_hiddens = buffer.model_state_hidden.at[positions].set(sg(states.hidden[:, :-1]))
    new_latents = buffer.model_state_latent.at[positions].set(sg(states.latent_sample[:, :-1]))

    # pyrefly: ignore[missing-attribute] - pyrefly misses synthesised method
    return buffer.replace(model_state_hidden=new_hiddens, model_state_latent=new_latents)
    # fmt: on


@jax.jit(static_argnames=["batch_size", "seq_length"])
def get_batch(
    state: BufferState, batch_size: int, seq_length: int, key: Array
) -> tuple[DreamerBatch, ReplayContext]:
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

    # [B, L]
    time_indices = seq_starts[:, None] + jnp.arange(seq_length)[None, :]
    # [B, 1] - broadcasts against [B, L] when we used these to slice the arrays
    batch_indices = episode_indices[:, None]

    # This is a bit vestigial. The original Dreamer implementation uses it to
    # cram samples from multiple episodes into one sequence within a batch, but my
    # implementation only uses it for the first steps of the batch
    is_first = time_indices == 0

    positions = (batch_indices, time_indices)
    batch = DreamerBatch(
        image=state.obs[positions],
        prev_action=state.acts[positions],
        cont=state.conts[positions],
        reward=state.rewards[positions],
        is_first=is_first,
    )

    context = ReplayContext(
        initial_state_hidden=state.model_state_hidden[episode_indices, seq_starts],
        initial_state_latent=state.model_state_latent[episode_indices, seq_starts],
        batch_indices=batch_indices,
        time_indices=time_indices,
    )
    return batch, context


def policy_step(
    world_model: DreamerWM,
    actor: Actor,
    carry: tuple[Array, Array],
    image: Array,
    prev_action: Array,
    is_first: Array,
    key: Array,
):
    """Observe an image and return the action together with its state and distribution."""
    embedding = world_model.encoder(image)
    carry, outputs = world_model.rssm.unroll(
        carry,
        (embedding[:, None], prev_action[:, None], is_first[:, None]),
    )
    hidden, latent, prior, posterior = jax.tree.map(lambda x: x[:, 0], outputs)
    shape = (
        image.shape[0],
        world_model.cfg.num_latent_dists,
        world_model.cfg.num_latent_classes,
    )
    state = RSSMOut(
        hidden=hidden,
        latent_sample=latent,
        prior=prior.reshape(shape),
        posterior=posterior.reshape(shape),
    )
    action_dist = actor(state.joint_latent)
    action = action_dist.sample(seed=key)
    return carry, action, state, action_dist


@nnx.jit
def choose_action(
    world_model: DreamerWM,
    actor: Actor,
    carry: tuple[Array, Array],
    image: Array,
    prev_action: Array,
    is_first: Array,
    key: Array,
) -> tuple[tuple[Array, Array], Array]:
    """Get the policy's action, based on the world_model's recurrent state."""
    carry, action, _model_state, _action_dist = policy_step(
        world_model, actor, carry, image, prev_action, is_first, key
    )
    return carry, action


def collect_new_rollouts(
    buffer: BufferState,
    envs: VectorEnv,
    world_model: DreamerWM,
    actor: Actor,
    num_eps: int,
    key: Array,
) -> BufferState:
    # Technically this function overfills if num_eps doesn't neatly divide by num_envs. But too much
    # data is rarely a bad thing
    num_envs = envs.num_envs
    max_timesteps = buffer.episode_length - 1
    action_dim = world_model.cfg.action_dim
    logger.info(f"Collecting {num_eps} rollouts of {max_timesteps} transitions")
    dtype = world_model.cfg.dtype

    for _ in range(0, num_eps, num_envs):
        rollout_obs = np.zeros(
            (num_envs, buffer.episode_length, *buffer.obs.shape[2:]), dtype=dtype
        )
        rollout_acts = np.zeros(
            (num_envs, buffer.episode_length, action_dim), dtype=dtype
        )
        rollout_conts = np.zeros((num_envs, buffer.episode_length), dtype=dtype)
        rollout_rewards = np.zeros_like(rollout_conts, dtype=np.float32)
        rollout_hidden = np.zeros(
            (num_envs, buffer.episode_length, world_model.cfg.wm_hidden_dim),
            dtype=dtype,
        )
        rollout_latent = np.zeros(
            (num_envs, buffer.episode_length, world_model.cfg.latent_dim),
            dtype=dtype,
        )

        reset_key, key = jax.random.split(key)
        seed = int(jax.random.randint(reset_key, (), 0, np.iinfo(np.int32).max))
        obs, _ = envs.reset(seed=seed)
        image = normalise_obs(crop_obs(obs)) - 0.5
        carry = world_model.rssm.initialize_state(num_envs)
        # The first entry for each episode is the initial state. Note we're implicitly
        # setting the initial action and reward to zero, these correspond to the action
        # and reward "before" the initial state, so that's fine.
        rollout_obs[:, 0] = image
        rollout_conts[:, 0] = 1
        rollout_hidden[:, 0] = carry[0]
        rollout_latent[:, 0] = carry[1]
        prev_action = jnp.zeros((num_envs, action_dim), dtype=dtype)
        is_first = jnp.ones(num_envs, dtype=jnp.bool_)

        for t in range(1, max_timesteps + 1):
            subkey, key = jax.random.split(key)
            carry, acts = choose_action(
                world_model,
                actor,
                carry,
                jnp.asarray(image),
                prev_action,
                is_first,
                subkey,
            )
            acts = np.asarray(acts, dtype=np.float32)
            obs, rwds, terminated, truncated, _infos = envs.step(acts)
            image = normalise_obs(crop_obs(obs)) - 0.5

            # We store the action a_t and the following observation o_t+1 in the same
            # time step, since that's what we'll what to exist in a single batch later when we train
            rollout_obs[:, t] = image
            rollout_acts[:, t] = acts
            # store raw reward, transform happens in the loss calc
            rollout_rewards[:, t] = rwds
            # Here we're storing the previous carry
            rollout_hidden[:, t] = carry[0]
            rollout_latent[:, t] = carry[1]

            # Add the episode end to train the continue predictor
            rollout_conts[:, t] = ~(terminated | truncated)
            prev_action = jnp.asarray(acts, dtype=dtype)
            is_first = jnp.zeros(num_envs, dtype=jnp.bool_)

        buffer = add_episodes(
            buffer,
            rollout_obs,
            rollout_acts,
            rollout_conts,
            rollout_rewards,
            rollout_hidden,
            rollout_latent,
        )

    return buffer
