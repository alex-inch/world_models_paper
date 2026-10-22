import distrax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from gymnasium.vector import SyncVectorEnv

from dreamer.buffer import collect_new_rollouts, init_buffer
from dreamer.wm import DreamerConfig


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
    try:
        buffer = collect_new_rollouts(
            init_buffer(capacity=5, seq_length=5),
            envs,
            CountingWorldModel(),
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
            pixels = np.asarray(buffer.obs[episode, :, 0, 0, 0])
            np.testing.assert_allclose(np.diff(pixels), np.full(4, 5 / 255), atol=1e-6)
        assert buffer.obs.dtype == jnp.float32
        assert np.max(np.abs(buffer.obs)) <= 0.5
        assert envs.envs[0].seeds[0] != envs.envs[0].seeds[-1]
    finally:
        envs.close()
