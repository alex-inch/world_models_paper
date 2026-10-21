import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from dreamer.actor_critic import DreamerActorCriticOut
from dreamer.imagine import imagination_rollout
from dreamer.wm.rssm import RSSMOut


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

    def __call__(self, state, prev_action, key):
        del prev_action, key
        actions = jnp.broadcast_to(self.action[...], (state.hidden.shape[0], 3))
        return DreamerActorCriticOut(
            action_dist=distrax.Independent(
                distrax.Deterministic(
                    actions, atol=jnp.zeros_like(actions), rtol=jnp.zeros_like(actions)
                ),
                1,
            ),
            value_pred=2 * state.hidden,
        )


def initial_state():
    return RSSMOut(
        hidden=jnp.array([[0.0], [10.0]]),
        latent_sample=jnp.zeros((2, 2)),
        prior=jnp.zeros((2, 1, 2)),
    )


@pytest.mark.parametrize("horizon", [1, 3])
def test_rollout_keeps_final_state_and_aligns_transition_rewards(horizon):
    wm, policy = CountingWorldModel(), ConstantPolicy()

    @nnx.jit
    def run(wm, policy):
        return imagination_rollout(
            wm,
            policy,
            initial_state(),
            jnp.zeros((2, 3)),
            jax.random.key(0),
            horizon,
        )

    trajectory, rewards, continues = run(wm, policy)
    expected_states = np.array([[0.0], [10.0]]) + np.arange(horizon + 1)
    np.testing.assert_allclose(trajectory.state.hidden[..., 0], expected_states)
    np.testing.assert_allclose(trajectory.value, 2 * expected_states)
    np.testing.assert_allclose(rewards, expected_states[:, 1:])
    np.testing.assert_allclose(continues, np.full((2, horizon), 0.5))
    assert trajectory.action.shape == (2, horizon, 3)
    assert trajectory.state.prior.shape == (2, horizon + 1, 1, 2)
    assert trajectory.state.posterior is None


def test_reward_gradient_reaches_policy_through_imagination():
    wm, policy = CountingWorldModel(), ConstantPolicy()

    @nnx.grad(argnums=1)
    def objective(wm, policy):
        _, rewards, _ = imagination_rollout(
            wm,
            policy,
            initial_state(),
            jnp.zeros((2, 3)),
            jax.random.key(0),
            3,
        )
        return rewards.mean()

    gradient = objective(wm, policy)
    np.testing.assert_allclose(gradient.action[...], 2.0)
