import distrax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from gymnasium.vector import SyncVectorEnv

from dreamer.buffer import (
    ReplayContext,
    collect_new_rollouts,
    get_batch,
    init_buffer,
    refresh_model_states,
)
from dreamer.wm import DreamerConfig
from dreamer.wm.rssm import RSSMOut


class ImageEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (96, 96, 3), dtype=np.uint8)
    action_space = gym.spaces.Box(
        np.array([-1, 0, 0], np.float32), np.ones(3, np.float32)
    )

    def __init__(self, horizon=4):
        self.horizon = horizon
        self.resets = 0
        self.seeds = []

    def image(self):
        return np.full((96, 96, 3), 20 + 50 * self.resets + 5 * self.t, np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resets += 1
        self.seeds.append(seed)
        self.t = 0
        return self.image(), {}

    def step(self, action):
        assert self.action_space.contains(action)
        self.t += 1
        return (
            self.image(),
            10 * self.t + float(action[1]),
            False,
            self.t == self.horizon,
            {},
        )


class CountingRSSM(nnx.Module):
    def initialize_state(self, batch_dim):
        return jnp.zeros((batch_dim, 1)), jnp.zeros((batch_dim, 1))

    @nnx.scan(in_axes=(None, nnx.Carry, 1), out_axes=(nnx.Carry, 1))
    def unroll(self, state, inputs):
        embedding, action, first = inputs
        hidden = state[0] * (1 - first[:, None]) + 1 + action[:, 1:2]
        latent = embedding
        logits = jnp.zeros_like(latent)
        return (hidden, latent), (hidden, latent, logits, logits)


class MeanEncoder(nnx.Module):
    def __call__(self, image):
        return image.mean(axis=(1, 2, 3))[:, None]


class CountingWorldModel(nnx.Module):
    def __init__(self):
        self.cfg = DreamerConfig(
            wm_hidden_dim=1, num_latent_dists=1, num_latent_classes=1
        )
        self.encoder = MeanEncoder()
        self.rssm = CountingRSSM()


class CountingPolicy(nnx.Module):
    def __call__(self, features):
        throttle = features[:, 0] / 10
        actions = jnp.stack(
            [jnp.zeros_like(throttle), throttle, jnp.zeros_like(throttle)], -1
        )
        return distrax.Independent(distrax.Deterministic(actions), 1)


def test_collection_retains_history_and_aligned_complete_episodes():
    envs = SyncVectorEnv([ImageEnv, ImageEnv])
    world_model = CountingWorldModel()
    try:
        buffer = collect_new_rollouts(
            init_buffer(capacity=5, seq_length=5, cfg=world_model.cfg),
            envs,
            world_model,
            CountingPolicy(),
            num_eps=4,
            key=jax.random.key(0),
        )
        assert int(buffer.num_eps) == 4
        expected_actions = np.array([0, 0.1, 0.21, 0.331, 0.4641], np.float32)
        for episode in range(4):
            np.testing.assert_allclose(
                buffer.acts[episode, :, 1], expected_actions, atol=1e-6
            )
            np.testing.assert_allclose(
                buffer.rewards[episode, 1:],
                10 * np.arange(1, 5) + expected_actions[1:],
                atol=1e-6,
            )
            np.testing.assert_array_equal(buffer.acts[episode, 0], np.zeros(3))
            np.testing.assert_array_equal(buffer.conts[episode], [1, 1, 1, 1, 0])
            np.testing.assert_array_equal(buffer.rewards[episode, 0], 0.0)
            np.testing.assert_allclose(
                buffer.model_state_hidden[episode, :, 0],
                10 * expected_actions,
                atol=1e-6,
            )
            np.testing.assert_array_equal(buffer.model_state_latent[episode, 0], 0)
            np.testing.assert_allclose(
                buffer.model_state_latent[episode, 1:, 0],
                np.asarray(buffer.obs[episode, :-1]).mean(axis=(1, 2, 3)),
                atol=1e-6,
            )
            pixels = np.asarray(buffer.obs[episode, :, 0, 0, 0])
            np.testing.assert_allclose(np.diff(pixels), np.full(4, 5 / 255), atol=1e-6)
        assert buffer.obs.dtype == jnp.float32
        assert np.max(np.abs(buffer.obs)) <= 0.5
        assert envs.envs[0].seeds[0] != envs.envs[0].seeds[-1]

        batch, context = get_batch(buffer, 16, 3, jax.random.key(1))
        embeddings = batch.image.mean(axis=(-3, -2, -1))[..., None]
        _, (hidden, latent, _, _) = world_model.rssm.unroll(
            context.initial_state, (embeddings, batch.prev_action, batch.is_first)
        )
        following = context.batch_indices, context.time_indices[:, 1:]
        np.testing.assert_allclose(
            hidden[:, :-1], buffer.model_state_hidden[following], atol=1e-6
        )
        np.testing.assert_allclose(
            latent[:, :-1], buffer.model_state_latent[following], atol=1e-6
        )
    finally:
        envs.close()


@pytest.fixture
def replay_buffer():
    cfg = DreamerConfig(wm_hidden_dim=3, num_latent_dists=2, num_latent_classes=3)
    buffer = init_buffer(capacity=3, seq_length=5, cfg=cfg)
    return buffer.replace(
        cursor=jnp.asarray(2, dtype=jnp.int32),
        full=jnp.asarray(True),
        obs=jnp.full_like(buffer.obs, 0.2),
        acts=jnp.full_like(buffer.acts, 0.3),
        rewards=jnp.arange(15, dtype=jnp.float32).reshape(3, 5),
        conts=jnp.ones_like(buffer.conts).at[:, -1].set(0),
        model_state_hidden=jnp.arange(45, dtype=jnp.float32).reshape(3, 5, 3),
        model_state_latent=1000 + jnp.arange(90, dtype=jnp.float32).reshape(3, 5, 6),
    )


@pytest.mark.parametrize("seq_length", [3, 5])
def test_sampling_returns_starting_memory_and_actual_episode_resets(
    replay_buffer, seq_length
):
    batch, context = get_batch(replay_buffer, 16, seq_length, jax.random.key(0))
    positions = context.batch_indices, context.time_indices
    starts = context.batch_indices[:, 0], context.time_indices[:, 0]
    np.testing.assert_array_equal(batch.image, replay_buffer.obs[positions])
    np.testing.assert_array_equal(batch.prev_action, replay_buffer.acts[positions])
    np.testing.assert_array_equal(batch.reward, replay_buffer.rewards[positions])
    np.testing.assert_array_equal(batch.cont, replay_buffer.conts[positions])
    np.testing.assert_array_equal(batch.is_first, context.time_indices == 0)
    np.testing.assert_array_equal(
        context.initial_state_hidden, replay_buffer.model_state_hidden[starts]
    )
    np.testing.assert_array_equal(
        context.initial_state_latent, replay_buffer.model_state_latent[starts]
    )
    assert context.initial_state_hidden.shape == (16, 3)
    assert context.initial_state_latent.shape == (16, 6)
    if seq_length == 5:
        assert np.all(batch.is_first[:, 0])
    else:
        assert np.any(~np.asarray(batch.is_first[:, 0]))


def refresh_inputs(buffer):
    episodes = jnp.array([[0], [2]])
    times = jnp.array([[1, 2, 3], [2, 3, 4]])
    context = ReplayContext(
        initial_state_hidden=buffer.model_state_hidden[episodes[:, 0], times[:, 0]],
        initial_state_latent=buffer.model_state_latent[episodes[:, 0], times[:, 0]],
        batch_indices=episodes,
        time_indices=times,
    )
    states = RSSMOut(
        hidden=2000 + jnp.arange(18, dtype=jnp.float32).reshape(2, 3, 3),
        latent_sample=3000 + jnp.arange(36, dtype=jnp.float32).reshape(2, 3, 6),
        prior=jnp.zeros((2, 3, 2, 3)),
    )
    return context, states


def test_refresh_stores_memory_before_the_following_frame(replay_buffer):
    context, states = refresh_inputs(replay_buffer)
    updated = refresh_model_states(replay_buffer, context, states)

    expected_hidden = np.array(replay_buffer.model_state_hidden, copy=True)
    expected_latent = np.array(replay_buffer.model_state_latent, copy=True)
    for row, (episode, following_frames) in enumerate(((0, (2, 3)), (2, (3, 4)))):
        for step, frame in enumerate(following_frames):
            expected_hidden[episode, frame] = states.hidden[row, step]
            expected_latent[episode, frame] = states.latent_sample[row, step]

    np.testing.assert_array_equal(updated.model_state_hidden, expected_hidden)
    np.testing.assert_array_equal(updated.model_state_latent, expected_latent)


@pytest.mark.parametrize("seq_length", [1, 3])
def test_refresh_preserves_starting_memory_and_other_replay_data(
    replay_buffer, seq_length
):
    context, states = refresh_inputs(replay_buffer)
    context = context.replace(time_indices=context.time_indices[:, :seq_length])
    states = jax.tree.map(lambda x: x[:, :seq_length], states)
    before = [np.array(x, copy=True) for x in jax.tree.leaves(replay_buffer)]
    updated = refresh_model_states(replay_buffer, context, states)

    starts = context.batch_indices[:, 0], context.time_indices[:, 0]
    np.testing.assert_array_equal(
        updated.model_state_hidden[starts], context.initial_state_hidden
    )
    np.testing.assert_array_equal(
        updated.model_state_latent[starts], context.initial_state_latent
    )
    for name in ("obs", "acts", "rewards", "conts", "cursor", "full"):
        np.testing.assert_array_equal(getattr(updated, name), getattr(replay_buffer, name))
    for expected, actual in zip(before, jax.tree.leaves(replay_buffer), strict=True):
        np.testing.assert_array_equal(actual, expected)
    if seq_length == 1:
        for expected, actual in zip(before, jax.tree.leaves(updated), strict=True):
            np.testing.assert_array_equal(actual, expected)
