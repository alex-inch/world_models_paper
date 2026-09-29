import jax.numpy as jnp
import numpy as np
import pytest

from dreamer.actor_critic import compute_td_lambda_returns
from dreamer.wm import DreamerConfig


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
