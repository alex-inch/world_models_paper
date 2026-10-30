from types import SimpleNamespace

import distrax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from dreamer.actor_critic import Actor
from dreamer.diagnostics import (
    _reference_diagnostics,
    evaluate_rollouts,
    policy_metrics,
    reference_metrics,
)
from dreamer.wm import DreamerBatch, DreamerConfig
from dreamer.wm.rssm import RSSMOut


class ZeroStateRSSM(nnx.Module):
    def initialize_state(self, batch_size):
        return jnp.zeros((batch_size, 1)), jnp.zeros((batch_size, 2))


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
        self.rssm = ZeroStateRSSM()

    def run_rssm(self, batch, initial_state):
        self.rngs()  # Like the real RSSM, observing a sequence advances an RNG stream.
        shape = batch.image.shape[:2]
        hidden, latent = initial_state
        assert hidden.shape == (shape[0], 1)
        assert latent.shape == (shape[0], 2)
        return RSSMOut(
            hidden=batch.image[..., 0, 0, :1] + hidden[:, None],
            latent_sample=jnp.broadcast_to(jnp.array([1.0, 0.0]), (*shape, 2))
            + latent[:, None],
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


@pytest.mark.parametrize("last_continue", [0, 1])
def test_eval_returns_stop_at_first_episode_end(monkeypatch, last_continue):
    class RewardWorldModel(CountingWorldModel):
        def __call__(self, batch, initial_state):
            state = self.run_rssm(batch, initial_state)
            return SimpleNamespace(
                hidden=state.hidden,
                latent_sample=state.latent_sample,
                latent_prior=state.prior,
                image=jnp.zeros_like(batch.image),
            )

        def decoder(self, features):
            return jnp.zeros((*features.shape[:-1], 1, 1, 3))

    monkeypatch.setattr("dreamer.diagnostics.policy_metrics", lambda *args: {})
    monkeypatch.setattr("dreamer.diagnostics.reference_metrics", lambda *args: {})
    monkeypatch.setattr(
        "dreamer.diagnostics.format_eval_video", lambda *args: np.zeros((1, 3, 1, 1))
    )
    monkeypatch.setattr(
        "dreamer.diagnostics.wandb.Video",
        lambda frames, **kwargs: SimpleNamespace(frames=frames, **kwargs),
    )
    monkeypatch.setattr(
        "dreamer.diagnostics.wandb.Histogram",
        lambda values, **kwargs: np.asarray(values).copy(),
    )
    buffer = SimpleNamespace(
        obs=jnp.zeros((3, 8, 1, 1, 3)),
        acts=jnp.zeros((3, 8, 3)),
        # Sentinel initial rewards and large rewards from later episodes must
        # be ignored, while the first episode's terminal reward is included.
        rewards=jnp.array([
            [1234, 2, 5, 100, 100, 100, 100, 100],
            [999, 1, 2, 3, 4, 200, 200, 200],
            [777, 1, 2, 3, 4, 5, 6, 7],
        ]),
        conts=jnp.array([
            [1, 1, 0, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, last_continue],
        ]),
    )
    wm = RewardWorldModel()
    metrics, _ = evaluate_rollouts(
        buffer, wm, ConstantPolicy(), LinearCritic(1), LinearCritic(2), wm.rngs()
    )
    expected = np.array([7, 10, 28])
    np.testing.assert_array_equal(metrics["eval/returns"], expected)
    assert metrics["eval/return_mean"] == 15
    assert metrics["eval/return_median"] == 10
    np.testing.assert_allclose(metrics["eval/return_std"], expected.std())
    assert "Mean Return: 15." in metrics["eval/policy"].caption


@pytest.mark.parametrize("fps", [10, 50])
def test_eval_video_repeats_context_and_imagination_in_sync(monkeypatch, fps):
    class VideoWorldModel(CountingWorldModel):
        def __call__(self, batch, initial_state):
            # Posterior states jump ten units per frame; imagination advances
            # only one. This makes resets and off-by-one errors visible.
            hidden = (10 * jnp.arange(batch.image.shape[1], dtype=jnp.float32))[
                None, :, None
            ] + initial_state[0][:, None]
            latent = jnp.broadcast_to(jnp.array([1.0, 0.0]), (1, len(hidden[0]), 2))
            return SimpleNamespace(
                hidden=hidden,
                latent_sample=latent,
                latent_prior=jnp.zeros((1, len(hidden[0]), 1, 2)),
                image=self.decoder(jnp.concatenate([hidden, latent], axis=-1)),
            )

        def decoder(self, features):
            pixels = features[..., :1] / 1000 - 0.5
            return jnp.broadcast_to(
                pixels[..., None, None, :], (*features.shape[:-1], 2, 4, 3)
            )

    monkeypatch.setattr("dreamer.diagnostics.policy_metrics", lambda *args: {})
    monkeypatch.setattr("dreamer.diagnostics.reference_metrics", lambda *args: {})
    monkeypatch.setattr(
        "dreamer.diagnostics.wandb.Video",
        lambda frames, **kwargs: SimpleNamespace(frames=frames, **kwargs),
    )
    # Two complete cycles, then five context frames and a partial imagined tail.
    buffer = SimpleNamespace(
        obs=jnp.zeros((1, 47, 2, 4, 3)),
        acts=jnp.zeros((1, 47, 3)),
        rewards=jnp.zeros((1, 47)),
        conts=jnp.ones((1, 47)),
    )
    wm = VideoWorldModel()
    metrics, _ = evaluate_rollouts(
        buffer,
        wm,
        ConstantPolicy(),
        LinearCritic(1),
        LinearCritic(2),
        wm.rngs(),
        fps=fps,
    )
    video = metrics["eval/policy"]
    expected_states = np.concatenate([
        10 * np.arange(5),
        40 + np.arange(1, 16),
        10 * np.arange(20, 25),
        240 + np.arange(1, 16),
        10 * np.arange(40, 45),
        440 + np.arange(1, 3),
    ])
    expected_pixels = (255 * expected_states / 1000).astype(np.uint8)[:: 50 // fps]
    imagination_panel = video.frames[:, :, 20:, 8:]
    np.testing.assert_allclose(
        imagination_panel,
        np.broadcast_to(expected_pixels[:, None, None, None], imagination_panel.shape),
        atol=1,
    )
    assert len(video.frames) == len(expected_pixels)
    assert "5 posterior reconstructions followed by 15 imagined steps" in video.caption
