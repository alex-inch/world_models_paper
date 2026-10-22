import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from dreamer.actor_critic import Actor, Critic, compute_td_lambda_returns
from dreamer.wm import DreamerConfig
from dreamer.wm.rssm import RSSMOut


def test_actor_and_critic_return_bounded_actions_and_scalar_values():
    cfg = DreamerConfig(
        wm_hidden_dim=4, num_latent_dists=2, num_latent_classes=2, agent_hidden_dim=8
    )
    actor = Actor(cfg, nnx.Rngs(0))
    critic = Critic(cfg, nnx.Rngs(1))
    state = RSSMOut(
        hidden=jnp.zeros((2, 3, 4)),
        latent_sample=jnp.zeros((2, 3, 4)),
        prior=jnp.zeros((2, 3, 2, 2)),
    )
    action_dist = actor(state.joint_latent)
    values = critic(state.joint_latent)
    actions = action_dist.sample(seed=jax.random.key(0))
    assert actions.shape == (2, 3, 3)
    assert values.shape == (2, 3)
    assert action_dist.entropy().shape == (2, 3)
    assert np.all(np.asarray(actions) >= [-1, 0, 0])
    assert np.all(np.asarray(actions) <= [1, 1, 1])


def reference_td_lambda(rewards, continues, values, discount, lam):
    """Explicitly mix every n-step return at every starting time step."""
    batch_size, length = rewards.shape
    result = np.zeros_like(rewards)

    for batch in range(batch_size):
        for start in range(length):
            horizon = length - start
            n_step_returns = []
            discounted_continuation = 1.0
            reward_sum = 0.0

            for n in range(1, horizon + 1):
                step = start + n - 1
                reward_sum += discounted_continuation * rewards[batch, step]
                discounted_continuation *= discount * continues[batch, step]
                n_step_returns.append(
                    reward_sum + discounted_continuation * values[batch, start + n]
                )

            for n in range(1, horizon):
                result[batch, start] += (
                    (1 - lam) * lam ** (n - 1) * n_step_returns[n - 1]
                )
            result[batch, start] += lam ** (horizon - 1) * n_step_returns[-1]

    return result


def test_td_lambda_scan_matches_explicit_n_step_returns():
    cfg = DreamerConfig(discount=0.93, td_lambda=0.7)
    rewards = np.array(
        [[1.0, -0.5, 2.0, 0.25], [-1.0, 0.75, 0.5, 1.5]], dtype=np.float32
    )
    continues = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    values = np.array(
        [[0.2, 1.5, -0.3, 0.8, 2.0], [0.4, -0.7, 1.2, 0.1, -1.0]],
        dtype=np.float32,
    )

    actual = compute_td_lambda_returns(
        cfg, jnp.asarray(rewards), jnp.asarray(continues), jnp.asarray(values)
    )
    expected = reference_td_lambda(
        rewards, continues, values, cfg.discount, cfg.td_lambda
    )

    assert actual.shape == rewards.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_td_lambda_scan_matches_reference_at_lambda_endpoints(lam):
    cfg = DreamerConfig(discount=0.8, td_lambda=lam)
    rewards = np.array([[2.0, -1.0, 3.0], [0.5, 1.5, -2.0]], dtype=np.float32)
    continues = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    values = np.array(
        [[5.0, -0.25, 0.75, 4.0], [0.0, 2.0, -3.0, 1.0]], dtype=np.float32
    )

    actual = compute_td_lambda_returns(
        cfg, jnp.asarray(rewards), jnp.asarray(continues), jnp.asarray(values)
    )
    expected = reference_td_lambda(rewards, continues, values, cfg.discount, lam)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
