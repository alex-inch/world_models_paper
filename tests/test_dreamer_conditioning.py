import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamer.actor_critic import Actor
from dreamer.conditioning import (
    collect_conditioning, condition_recording, forecast_step, imagine_step,
)
from dreamer.divergence import observe
from dreamer.wm import DreamerConfig, DreamerWM
from wm.utils.obs import normalise_obs


def tiny_models():
    cfg = DreamerConfig(wm_hidden_dim=4, num_latent_dists=2,
                        num_latent_classes=2, agent_hidden_dim=8)
    return {"world_model": DreamerWM(cfg, rngs=nnx.Rngs(0)),
            "actor": Actor(cfg, nnx.Rngs(1))}


def test_short_context_forgets_earlier_images_and_actions():
    models = tiny_models()
    images = np.zeros((3, 64, 64, 3), np.uint8)
    actions = np.zeros((2, 3), np.float32)
    _, before = condition_recording(models, images, actions, contexts=(1, 3), seed=2)
    changed_images = images.copy()
    changed_images[:2] = 255
    _, after = condition_recording(models, changed_images, actions + 0.9,
                                   contexts=(1, 3), seed=2)
    np.testing.assert_array_equal(before[0].hidden, after[0].hidden)
    np.testing.assert_array_equal(before[0].latent_sample, after[0].latent_sample)
    assert not np.allclose(before[1].hidden, after[1].hidden)


def test_full_context_matches_real_history_and_random_draws_are_shared():
    models = tiny_models()
    images = np.full((4, 64, 64, 3), 120, np.uint8)
    actions = np.full((3, 3), 0.2, np.float32)
    branches, states = condition_recording(models, images, actions, contexts=(1, 3), seed=2)
    wm = models["world_model"]
    nnx.reseed(wm, default=2)
    carry = wm.rssm.initialize_state(1)
    key = jax.random.key(2)
    for step in range(3):
        key, action_key = jax.random.split(key)
        previous = jnp.zeros((1, 3)) if step == 0 else jnp.asarray(actions[step - 1 : step])
        carry, _, real, _ = observe(
            wm, models["actor"], carry,
            jnp.asarray(normalise_obs(images[step : step + 1]) - 0.5),
            previous, jnp.asarray([step == 0]), action_key,
        )
    np.testing.assert_array_equal(states[1].hidden, real.hidden)
    np.testing.assert_array_equal(states[1].latent_sample, real.latent_sample)

    # Equal histories have identical predicted distributions and shared samples.
    branches, equal = condition_recording(models, images, actions, contexts=(3, 3), seed=2)
    predicted = imagine_step(branches, models["actor"], equal, jax.random.key(4))
    np.testing.assert_array_equal(predicted[0].prior, predicted[1].prior)
    np.testing.assert_array_equal(predicted[0].latent_sample, predicted[1].latent_sample)


def test_future_real_image_cannot_change_any_imagination_branch():
    models = tiny_models()
    images = np.zeros((3, 64, 64, 3), np.uint8)
    branches, states = condition_recording(models, images, np.zeros((2, 3)),
                                          contexts=(1, 3), seed=2)
    carry = models["world_model"].rssm.initialize_state(1)
    predictions = []
    for image in (jnp.zeros((1, 64, 64, 3)), jnp.ones((1, 64, 64, 3))):
        expected_carry, expected_action, _, _ = observe(
            nnx.clone(models["world_model"], arrays=True), models["actor"],
            carry, image, jnp.zeros((1, 3)), jnp.zeros(1, dtype=bool), jax.random.key(3),
        )
        real_carry, real_action, next_states, _ = forecast_step(
            nnx.clone(models["world_model"], arrays=True), models["actor"],
            [nnx.clone(branch, arrays=True) for branch in branches],
            carry, states, image, jnp.zeros((1, 3)), jax.random.key(3), jax.random.key(4),
        )
        for expected, actual in zip(expected_carry, real_carry, strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-6)
        np.testing.assert_allclose(real_action, expected_action, atol=1e-6)
        predictions.append(next_states)
    for before, after in zip(*predictions, strict=True):
        np.testing.assert_array_equal(before.prior, after.prior)
        np.testing.assert_array_equal(before.hidden, after.hidden)
        np.testing.assert_array_equal(before.latent_sample, after.latent_sample)


class ShortImageEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (96, 96, 3), dtype=np.uint8)
    action_space = gym.spaces.Box(-1., 1., (3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.length = 2 if seed is None or seed % 2 == 0 else 4
        return np.zeros((96, 96, 3), np.uint8), {}

    def step(self, action):
        self.steps += 1
        return (np.zeros((96, 96, 3), np.uint8), float(self.steps),
                self.steps == self.length, False, {})


def test_common_forecast_origin_and_terminal_observation_mask(monkeypatch):
    import dreamer.conditioning as conditioning

    envs = gym.vector.SyncVectorEnv([ShortImageEnv, ShortImageEnv])
    monkeypatch.setattr(conditioning.gym, "make_vec", lambda *args, **kwargs: envs)
    curves, lengths, returns = collect_conditioning(
        tiny_models(), "test", contexts=(1, 2), num_episodes=3,
        num_envs=2, max_steps=5, seed=0,
    )
    np.testing.assert_array_equal(lengths, [2, 4, 2])
    np.testing.assert_array_equal(returns, [3, 10, 3])
    assert np.isnan(curves[:, :2]).all()
    for index, end in enumerate(lengths):
        assert np.isfinite(curves[index, 2 : end + 1]).all()
        assert np.isnan(curves[index, end + 1 :]).all()
    assert envs.closed
