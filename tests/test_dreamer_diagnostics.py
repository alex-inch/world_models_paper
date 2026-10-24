from types import SimpleNamespace

import distrax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamer.actor_critic import Actor
from dreamer.diagnostics import (
    _reference_diagnostics,
    policy_metrics,
    reference_metrics,
)
from dreamer.wm import DreamerBatch, DreamerConfig
from dreamer.wm.rssm import RSSMOut


class CountingWorldModel(nnx.Module):
    def __init__(self):
        self.cfg = DreamerConfig(
            wm_hidden_dim=1,
            num_latent_dists=1,
            num_latent_classes=2,
            agent_hidden_dim=8,
            discount=0.9,
            td_lambda=0.5,
        )
        self.rngs = nnx.Rngs(20)

    def run_rssm(self, batch):
        self.rngs()  # Like the real RSSM, observing a sequence advances an RNG stream.
        shape = batch.image.shape[:2]
        return RSSMOut(
            hidden=batch.image[..., 0, 0, :1],
            latent_sample=jnp.broadcast_to(jnp.array([1.0, 0.0]), (*shape, 2)),
            prior=jnp.zeros((*shape, 1, 2)),
            posterior=jnp.zeros((*shape, 1, 2)),
        )

    def imagine_step(self, state, action):
        return state.replace(hidden=state.hidden + action[:, :1])

    def _predict_reward_and_continue(self, features):
        return features[..., :1], jnp.full(features.shape[:-1], 30.0)

    def decode_reward_logits(self, logits):
        return logits[..., 0]


class ConstantPolicy(nnx.Module):
    def __call__(self, features):
        actions = jnp.ones((*features.shape[:-1], 3))
        return distrax.Independent(
            distrax.Deterministic(
                actions, atol=jnp.zeros_like(actions), rtol=jnp.zeros_like(actions)
            ),
            1,
        )


class LinearCritic(nnx.Module):
    def __init__(self, scale):
        self.scale = nnx.Param(jnp.asarray(scale, dtype=jnp.float32))

    def __call__(self, features):
        return self.scale[...] * features[..., 0]


def reference_batch():
    time = jnp.arange(8, dtype=jnp.float32)
    return DreamerBatch(
        image=time[None, :, None, None, None],
        prev_action=jnp.stack([time, time * 0, time * 0], axis=-1)[None],
        reward=100 + time[None],
        cont=jnp.ones((1, 8)),
        is_first=(time == 0)[None],
    )


def test_action_metrics_exclude_dummy_actions_and_steps_after_episode_end():
    wm = CountingWorldModel()
    actor = Actor(wm.cfg, nnx.Rngs(0))
    actions = np.array(
        [
            [[99, 99, 99], [-1, 0, 0], [1, 1, 1], [99, 99, 99]],
            [[99, 99, 99], [0, 0.5, 0.5], [0, 0.5, 0.5], [0, 0.5, 0.5]],
        ],
        dtype=np.float32,
    )
    buffer = SimpleNamespace(acts=actions, conts=np.array([[1, 1, 0, 0], [1, 1, 1, 0]]))
    predictions = SimpleNamespace(
        hidden=jnp.zeros((1, 4, 1)), latent_sample=jnp.zeros((1, 4, 2))
    )
    metrics = policy_metrics(buffer, actor, predictions)
    assert metrics["eval/actions/steering_mean"] == 0
    np.testing.assert_allclose(metrics["eval/actions/throttle_mean"], 0.5)
    np.testing.assert_allclose(metrics["eval/actions/brake_mean"], 0.5)
    np.testing.assert_allclose(
        metrics["eval/actions/steering_near_bound_fraction"], 0.4
    )
    assert sum(metrics["eval/actions/steering_histogram"].histogram) == 5
    np.testing.assert_allclose(
        metrics["eval/policy_entropy"], actor(jnp.zeros((1, 3))).entropy()[0]
    )


def test_recorded_action_alignment_and_lambda_return_decomposition():
    data = _reference_diagnostics(
        CountingWorldModel(),
        ConstantPolicy(),
        LinearCritic(1),
        LinearCritic(2),
        reference_batch(),
        jnp.array([1, 3]),
        horizon=3,
    )
    np.testing.assert_allclose(data["open_loop_reward"], [[3, 6, 10], [7, 12, 18]])
    np.testing.assert_allclose(
        data["open_loop_actual_reward"], [[102, 103, 104], [104, 105, 106]]
    )
    np.testing.assert_allclose(data["posterior_reward"], [np.arange(1, 8)])
    np.testing.assert_allclose(data["lambda_return"][:, 0], [8.633, 15.277], rtol=1e-6)
    np.testing.assert_allclose(data["reward_return"][:, 0], [4.16, 7.465], rtol=1e-6)
    np.testing.assert_allclose(
        data["reward_return"] + data["bootstrap_return"], data["lambda_return"]
    )
    np.testing.assert_allclose(data["weights"], [[1, 0.9, 0.81]] * 2)
    np.testing.assert_allclose(data["prior_entropy"], np.log(2))
    np.testing.assert_allclose(data["posterior_entropy"], np.log(2))
    np.testing.assert_allclose(data["kl"], 0)
    np.testing.assert_allclose(data["category_usage"], [[1, 0]])


def test_reference_metrics_are_repeatable_and_leave_model_state_unchanged():
    wm, actor = CountingWorldModel(), ConstantPolicy()
    critic, target = LinearCritic(1), LinearCritic(2)
    before = np.asarray(wm.rngs.default.count[...]).copy()
    first = reference_metrics(wm, actor, critic, target, reference_batch())
    second = reference_metrics(wm, actor, critic, target, reference_batch())
    np.testing.assert_array_equal(wm.rngs.default.count[...], before)
    for name, value in first.items():
        if isinstance(value, float):
            assert np.isfinite(value)
            assert value == second[name]
    np.testing.assert_allclose(first["eval/latents/category_fraction_used"], 0.5)
    assert first["eval/critic/online_target_disagreement"] > 0
    assert first["eval/reward/posterior_mae"] == 100
    assert first["eval/reward/posterior_bias"] == -100
    for name in (
        "eval/reward/posterior_trace",
        "eval/reward/prior_trace",
        "eval/critic/return_components",
        "eval/critic/predictions",
        "eval/imagination/continuation_and_weights",
        "eval/latents/category_usage",
    ):
        assert name in first
