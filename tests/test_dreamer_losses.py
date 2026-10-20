from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from dreamer.preprocessing import symexp
from dreamer.wm import DreamerBatch, DreamerConfig, DreamerWM, DreamerWMOut
from dreamer.wm.losses import dreamer_wm_loss_fn, twohot_loss


@pytest.fixture
def reward_case():
    cfg = DreamerConfig(
        num_latent_dists=2,
        num_latent_classes=3,
        reward_encoding_bins=5,
        reward_encoding_max_exp=4,
        kl_alpha=0.3,
    )
    batch = DreamerBatch(
        image=jnp.zeros((1, 5, 1, 1, 1)),
        reward=symexp(jnp.array([[-10, -1.5, 0, 1.5, 10]])),
        cont=jnp.ones((1, 5)),
        prev_action=jnp.zeros((1, 5, cfg.action_dim)),
        is_first=jnp.array([[True, False, False, False, False]]),
    )
    preds = DreamerWMOut(
        image=jnp.zeros_like(batch.image),
        reward=jnp.array([[
            [2, -1, 0, 1, -2],
            [0, 3, -1, 2, -2],
            [-3, -1, 4, 2, 0],
            [-2, 0, 3, 1, -1],
            [0, 1, 0, -1, 3],
        ]], dtype=jnp.float32),
        cont=jnp.zeros_like(batch.cont),
        latent_prior=jnp.zeros((1, 5, 2, 3)),
        latent_posterior=jnp.ones((1, 5, 2, 3)),
        latent_sample=jnp.zeros((1, 5, cfg.latent_dim)),
        hidden=jnp.zeros((1, 5, cfg.wm_hidden_dim)),
    )
    # Centres [-4, -2, 0, 2, 4]: +/-1.5 split mass 3:1; +/-10 clip.
    labels = jnp.array([[
        [1, 0, 0, 0, 0],
        [0, 0.75, 0.25, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0.25, 0.75, 0],
        [0, 0, 0, 0, 1],
    ]])
    return cfg, batch, preds, labels


def test_twohot_loss_applies_symlog_and_requested_limits(reward_case):
    cfg, batch, preds, labels = reward_case
    expected = -(labels * jax.nn.log_softmax(preds.reward)).sum(axis=-1).mean()
    actual = twohot_loss(preds, batch, max_exp=cfg.reward_encoding_max_exp)
    assert jnp.allclose(actual, expected)


def test_twohot_loss_gradient_matches_target_distribution(reward_case):
    cfg, batch, preds, labels = reward_case

    def loss(logits):
        return twohot_loss(
            replace(preds, reward=logits), batch, cfg.reward_encoding_max_exp
        )

    actual = jax.grad(loss)(preds.reward)
    expected = (jax.nn.softmax(preds.reward) - labels) / batch.reward.size
    assert jnp.allclose(actual, expected)


@pytest.mark.parametrize("loss_fn", [dreamer_wm_loss_fn, DreamerWM.loss_fn])
def test_world_model_loss_uses_configured_reward_limits(loss_fn, reward_case):
    cfg, batch, preds, labels = reward_case
    expected = -(labels * jax.nn.log_softmax(preds.reward)).sum(axis=-1).mean()
    losses = loss_fn(preds, batch, cfg)
    assert jnp.allclose(losses.reward_loss, expected)
    assert losses.image_loss == 0
    assert jnp.allclose(losses.continue_loss, jnp.log(2))
    assert losses.kl_loss == 0
