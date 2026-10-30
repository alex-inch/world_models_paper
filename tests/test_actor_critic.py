import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from dreamer.actor_critic import Actor, Critic, compute_td_lambda_returns
from dreamer.wm import DreamerConfig
from dreamer.wm.rssm import RSSMOut


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_actor_and_critic_return_bounded_actions_and_scalar_values(dtype):
    cfg = DreamerConfig(
        wm_hidden_dim=4,
        num_latent_dists=2,
        num_latent_classes=2,
        agent_hidden_dim=8,
        dtype=dtype,
    )
    actor = Actor(cfg, nnx.Rngs(0))
    critic = Critic(cfg, nnx.Rngs(1))
    state = RSSMOut(
        hidden=jnp.zeros((2, 3, 4), dtype=dtype),
        latent_sample=jnp.zeros((2, 3, 4), dtype=dtype),
        prior=jnp.zeros((2, 3, 2, 2), dtype=dtype),
    )
    action_dist = actor(state.joint_latent)
    values = critic(state.joint_latent)
    actions = action_dist.sample(seed=jax.random.key(0))
    assert actions.shape == (2, 3, 3)
    assert values.shape == (2, 3)
    assert action_dist.entropy().shape == (2, 3)
    assert actions.dtype == values.dtype == action_dist.entropy().dtype == jnp.float32
    assert action_dist.loc.dtype == action_dist.scale.dtype == jnp.float32
    assert np.all(np.asarray(actions) >= [-1, 0, 0])
    assert np.all(np.asarray(actions) <= [1, 1, 1])


def test_actor_extreme_outputs_keep_finite_bounded_parameters_and_gradients():
    cfg = DreamerConfig(
        wm_hidden_dim=4, num_latent_dists=2, num_latent_classes=2, agent_hidden_dim=8
    )
    actor = Actor(cfg, nnx.Rngs(0))
    head = actor.actor_net.layers[-1]
    head.kernel[...] = jnp.zeros_like(head.kernel[...])
    head.bias[...] = jnp.array([-1000.0, 0.0, 1000.0, -1000.0, 0.0, 1000.0])
    features = jnp.zeros((2, 3, cfg.latent_dim + cfg.wm_hidden_dim))

    distribution = actor(features)
    assert np.all(np.asarray(distribution.loc) >= [-1, 0, 0])
    assert np.all(np.asarray(distribution.loc) <= [1, 1, 1])
    assert np.all(np.asarray(distribution.scale) >= [0.1, 0.05, 0.05])
    assert np.all(np.asarray(distribution.scale) <= [2.1, 1.05, 1.05])

    @nnx.jit
    @nnx.value_and_grad
    def objective(policy):
        dist = policy(features)
        return dist.entropy().mean() + dist.sample(seed=jax.random.key(0)).mean()

    value, gradients = objective(actor)
    assert np.isfinite(value)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(gradients))


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


def test_diagnostic_loss_contributions_average_to_training_losses():
    cfg = DreamerConfig(wm_hidden_dim=4, num_latent_dists=2, num_latent_classes=2, agent_hidden_dim=8)
    actor = Actor(cfg, nnx.Rngs(0))
    distribution = actor(jnp.zeros((2, 3, 8)))
    rewards = jnp.array([[1.0, 2.0, -0.1], [0.0, 0.5, 1.0]])
    continues = jnp.array([[1.0, 0.9, 0.0], [1.0, 1.0, 1.0]])
    values = jnp.ones((2, 4))
    average, targets, weights = Actor.loss_fn(cfg, rewards, continues, values, distribution)
    contributions, _, _ = Actor.loss_fn(cfg, rewards, continues, values, distribution, per_step=True)
    np.testing.assert_allclose(contributions.loss.mean(), average.loss, rtol=1e-6)
    critic_average = Critic.loss_fn(values[:, :-1], targets, weights)
    critic_contributions = Critic.loss_fn(values[:, :-1], targets, weights, per_step=True)
    np.testing.assert_allclose(critic_contributions.mean(), critic_average, rtol=1e-6)


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
