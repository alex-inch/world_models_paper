import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from dreamer.actor_critic import Actor, Critic
from dreamer.imagine import run_imagination_rollout
from dreamer.wm import DreamerConfig
from dreamer.wm.rssm import RSSMOut
from scripts.train_dreamer import actor_loss_fn, critic_loss_fn


class CountingWorldModel(nnx.Module):
    def imagine_step(self, state, action):
        return state.replace(hidden=state.hidden + action[:, :1], prior=state.prior + 1)

    def _predict_reward_and_continue(self, features):
        return features[..., :1], jnp.zeros(features.shape[:-1])

    def decode_reward_logits(self, logits):
        return logits[..., 0]


class ConstantPolicy(nnx.Module):
    def __init__(self):
        self.action = nnx.Param(jnp.asarray(1.0))

    def __call__(self, features):
        actions = jnp.broadcast_to(self.action[...], (features.shape[0], 3))
        return distrax.Independent(
            distrax.Deterministic(
                actions, atol=jnp.zeros_like(actions), rtol=jnp.zeros_like(actions)
            ),
            1,
        )


class LinearCritic(nnx.Module):
    def __call__(self, features):
        return 2 * features[..., 0]


def initial_state():
    return RSSMOut(
        hidden=jnp.array([[0.0], [10.0]]),
        latent_sample=jnp.zeros((2, 2)),
        prior=jnp.zeros((2, 1, 2)),
    )


@pytest.mark.parametrize("horizon", [1, 3])
def test_rollout_keeps_final_state_and_aligns_transition_rewards(horizon):
    wm, policy, critic = CountingWorldModel(), ConstantPolicy(), LinearCritic()

    @nnx.jit
    def run(wm, policy, critic):
        return run_imagination_rollout(
            wm,
            policy,
            critic,
            initial_state(),
            horizon,
            jax.random.key(0),
        )

    trajectory, rewards, continues = run(wm, policy, critic)
    expected_states = np.array([[0.0], [10.0]]) + np.arange(horizon + 1)
    np.testing.assert_allclose(trajectory.state.hidden[..., 0], expected_states)
    np.testing.assert_allclose(trajectory.value, 2 * expected_states)
    np.testing.assert_allclose(rewards, expected_states[:, 1:])
    np.testing.assert_allclose(continues, np.full((2, horizon), 0.5))
    assert trajectory.action.shape == (2, horizon, 3)
    assert trajectory.state.prior.shape == (2, horizon + 1, 1, 2)
    assert trajectory.state.posterior is None


@pytest.mark.parametrize("signal,expected_gradient", [("reward", 2.0), ("value", 6.0)])
def test_gradient_reaches_policy_through_imagination(signal, expected_gradient):
    wm, policy, critic = CountingWorldModel(), ConstantPolicy(), LinearCritic()

    @nnx.grad
    def objective(policy, critic, wm):
        trajectory, rewards, _ = run_imagination_rollout(
            wm,
            policy,
            critic,
            initial_state(),
            3,
            jax.random.key(0),
        )
        return rewards.mean() if signal == "reward" else trajectory.value[:, -1].mean()

    gradient = objective(policy, critic, wm)
    np.testing.assert_allclose(gradient.action[...], expected_gradient)


def test_training_losses_update_only_their_own_module():
    cfg = DreamerConfig(
        wm_hidden_dim=1, num_latent_dists=1, num_latent_classes=2, agent_hidden_dim=8
    )
    actor = Actor(cfg, nnx.Rngs(0))
    critic = Critic(cfg, nnx.Rngs(1))
    wm = CountingWorldModel()
    actor_optim = nnx.Optimizer(actor, optax.sgd(0.01), wrt=nnx.Param)
    critic_optim = nnx.Optimizer(critic, optax.sgd(0.01), wrt=nnx.Param)

    def parameters(model):
        return [
            np.array(x, copy=True) for x in jax.tree.leaves(nnx.state(model, nnx.Param))
        ]

    actor_before, critic_before = parameters(actor), parameters(critic)
    (loss, (_, (features, targets))), grads = actor_loss_fn(
        actor, critic, wm, initial_state(), horizon=2, key=jax.random.key(0)
    )
    assert np.isfinite(loss)
    assert set(grads) == {"actor_net"}
    assert features.shape == (2, 2, 3)
    assert targets.shape == (2, 2)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))
    actor_optim.update(actor, grads)
    assert any(
        not np.array_equal(a, b) for a, b in zip(actor_before, parameters(actor))
    )
    for before, after in zip(critic_before, parameters(critic)):
        np.testing.assert_array_equal(before, after)

    actor_after = parameters(actor)
    loss, grads = critic_loss_fn(critic, features, targets)
    assert np.isfinite(loss)
    assert set(grads) == {"critic_net"}
    critic_optim.update(critic, grads)
    assert any(
        not np.array_equal(a, b) for a, b in zip(critic_before, parameters(critic))
    )
    for before, after in zip(actor_after, parameters(actor)):
        np.testing.assert_array_equal(before, after)
