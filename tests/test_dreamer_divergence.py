import jax
import jax.numpy as jnp
import numpy as np

from dreamer.divergence import _advance, bootstrap_curve_mean, collect_divergence, latent_kl, observe
from dreamer.actor_critic import Actor
from dreamer.wm import DreamerConfig, DreamerWM
from flax import nnx


def test_kl_direction_factor_average_and_identity():
    q = np.array([0.8, 0.2])
    p = np.array([0.1, 0.9])
    q_logits = jnp.asarray(np.tile(np.log(q), (1, 3, 1)))
    p_logits = jnp.asarray(np.tile(np.log(p), (1, 3, 1)))
    expected = (q * np.log(q / p)).sum()
    np.testing.assert_allclose(latent_kl(q_logits, p_logits), [expected], rtol=1e-6)
    np.testing.assert_array_equal(latent_kl(q_logits, q_logits), [0])


def test_kl_stays_finite_for_overconfident_disagreement():
    value = latent_kl(jnp.array([[[1000., -1000.]]]), jnp.array([[[-1000., 1000.]]]))
    np.testing.assert_allclose(value, [2000.])


def test_bootstrap_resamples_whole_curves_and_keeps_conditions_paired():
    curves = np.arange(20)[:, None, None] + np.arange(6)[None, :, None]
    curves = np.concatenate([curves, curves + 10], axis=2)
    mean, interval = bootstrap_curve_mean(curves, reps=500, seed=20)
    again = bootstrap_curve_mean(curves, reps=500, seed=20)
    np.testing.assert_allclose(mean[:, 1] - mean[:, 0], 10)
    np.testing.assert_allclose(interval[:, :, 1] - interval[:, :, 0], 10)
    np.testing.assert_allclose(interval[:, :, 0] - interval[:, :1, 0], np.tile(np.arange(6), (2, 1)))
    np.testing.assert_array_equal(interval, again[1])


def test_first_matched_prediction_has_same_prior_as_observation_corrected_model():
    cfg = DreamerConfig(wm_hidden_dim=4, num_latent_dists=2, num_latent_classes=2, agent_hidden_dim=8)
    wm = DreamerWM(cfg, rngs=nnx.Rngs(0))
    actor = Actor(cfg, nnx.Rngs(1))
    matched_wm = nnx.clone(wm, arrays=True)
    actor_wm = nnx.clone(wm, arrays=True)
    carry = wm.rssm.initialize_state(2)
    image = jnp.zeros((2, 64, 64, 3))
    carry, action, state, _ = observe(
        wm, actor, carry, image, jnp.zeros((2, 3)), jnp.ones(2, dtype=bool), jax.random.key(2)
    )
    _, _, _, _, scores = _advance(
        wm, actor, matched_wm, actor_wm, carry,
        state.replace(posterior=None), state.replace(posterior=None),
        image + 0.1, action, jax.random.key(3), jax.random.key(4),
    )
    np.testing.assert_allclose(scores[:, 0], scores[:, 1], atol=1e-6)
import gymnasium as gym


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
        done = self.steps == self.length
        return np.zeros((96, 96, 3), np.uint8), float(self.steps), done, False, {}


def test_terminal_observation_is_scored_but_autoreset_is_excluded(monkeypatch):
    import dreamer.divergence as divergence

    cfg = DreamerConfig(wm_hidden_dim=4, num_latent_dists=2, num_latent_classes=2, agent_hidden_dim=8)
    models = {"world_model": DreamerWM(cfg, rngs=nnx.Rngs(0)), "actor": Actor(cfg, nnx.Rngs(1))}
    envs = gym.vector.SyncVectorEnv([ShortImageEnv, ShortImageEnv])
    monkeypatch.setattr(divergence.gym, "make_vec", lambda *args, **kwargs: envs)
    curves, lengths, returns = collect_divergence(
        models, "test", num_episodes=3, num_envs=2, max_steps=5, context=1, seed=0,
    )
    np.testing.assert_array_equal(lengths, [2, 4, 2])
    np.testing.assert_array_equal(returns, [3, 10, 3])
    for index, end in enumerate(lengths):
        assert np.isfinite(curves[index, 1 : end + 1]).all()
        assert np.isnan(curves[index, end + 1 :]).all()
    assert envs.closed
