import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from dreamer.truncated_normal import TruncatedNormal


def _quadrature_stats(loc, scale, lower, upper):
    """Independent density integration, rescaled to work in remote tails."""
    anchor = np.clip(loc, lower, upper)
    anchor_z = (anchor - loc) / scale

    def weight(x):
        return np.exp(-0.5 * ((x - loc) / scale) ** 2 + 0.5 * anchor_z**2)

    mass = quad(weight, lower, upper, epsabs=1e-12)[0]
    mean = quad(lambda x: x * weight(x), lower, upper, epsabs=1e-12)[0] / mass
    entropy = (
        np.log(mass)
        + quad(
            lambda x: 0.5 * (((x - loc) / scale) ** 2 - anchor_z**2) * weight(x),
            lower,
            upper,
            epsabs=1e-12,
        )[0]
        / mass
    )
    return mean, entropy


def test_mean_and_entropy_match_quadrature():
    lower = np.array([-1.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 1.0])
    loc = np.array([[0.0, 0.5, 0.0], [10.0, -10.0, 100.0]])
    scale = np.array([[1.0, 0.3, 100.0], [1.0, 1.0, 1.0]])
    dist = TruncatedNormal(loc, scale, lower, upper)

    expected = [
        _quadrature_stats(loc[i, j], scale[i, j], lower[j], upper[j])
        for i in range(2)
        for j in range(3)
    ]
    expected_mean, expected_entropy = (
        np.array(expected).reshape(2, 3, 2).transpose(2, 0, 1)
    )
    assert isinstance(dist, distrax.Independent)
    assert dist.batch_shape == (2,)
    assert dist.event_shape == (3,)
    np.testing.assert_allclose(dist.mean(), expected_mean, rtol=2e-4, atol=6e-5)
    np.testing.assert_allclose(
        dist.entropy(), expected_entropy.sum(axis=-1), rtol=2e-4, atol=5e-4
    )

    narrow = TruncatedNormal(jnp.array([-10.0]), 1.0, 0.0, 0.001)
    narrow_mean, narrow_entropy = _quadrature_stats(-10.0, 1.0, 0.0, 0.001)
    np.testing.assert_allclose(narrow.mean()[0], narrow_mean, atol=1e-6)
    np.testing.assert_allclose(narrow.entropy(), narrow_entropy, atol=1e-5)


def test_sampling_log_prob_and_jax_transforms():
    loc = jnp.array([[0.0, 8.0, -8.0], [0.3, -2.0, 2.0]])
    scale = jnp.ones_like(loc)
    lower = jnp.array([-1.0, 0.0, 0.0])
    upper = jnp.array([1.0, 1.0, 1.0])
    dist = TruncatedNormal(loc, scale, lower, upper)

    @jax.jit
    def draw_and_score(distribution, key):
        return distribution.sample_and_log_prob(seed=key, sample_shape=(20_000,))

    samples, log_probs = draw_and_score(dist, jax.random.key(0))
    assert samples.shape == (20_000, 2, 3)
    assert log_probs.shape == (20_000, 2)
    assert jnp.isfinite(samples).all()
    assert jnp.isfinite(log_probs).all()
    assert jnp.all((samples >= lower) & (samples <= upper))
    np.testing.assert_allclose(log_probs, dist.log_prob(samples), atol=1e-6)
    np.testing.assert_allclose(
        log_probs, dist.distribution.log_prob(samples).sum(axis=-1), atol=1e-6
    )
    np.testing.assert_allclose(samples.mean(axis=0), dist.mean(), atol=0.02)
    assert jnp.isneginf(dist.log_prob(jnp.array([2.0, -1.0, 2.0]))).all()
    np.testing.assert_allclose(dist.entropy(), dist.distribution.entropy().sum(axis=-1))

    vmapped = jax.jit(
        jax.vmap(lambda x: TruncatedNormal(x, 1.0, lower, upper).entropy())
    )
    np.testing.assert_allclose(vmapped(loc), dist.entropy(), atol=1e-6)

    grad = jax.jit(
        jax.grad(
            lambda x: (
                TruncatedNormal(x, scale, lower, upper).mean().sum()
                + TruncatedNormal(x, scale, lower, upper).entropy().sum()
            )
        )
    )(loc)
    assert jnp.isfinite(grad).all()

    far = TruncatedNormal(jnp.array([-100.0]), 0.01, 0.0, 1.0)
    far_samples = jax.jit(lambda key: far.sample(seed=key, sample_shape=(2_000,)))(
        jax.random.key(1)
    )
    assert 0.8e-6 < far_samples.mean() < 1.2e-6
    np.testing.assert_allclose(far.mean()[0], 1e-6, rtol=1e-4)
    np.testing.assert_allclose(far.entropy(), 1 - np.log(1e6), atol=1e-4)
    assert jnp.isfinite(
        jax.grad(lambda x: TruncatedNormal(jnp.array([x]), 0.01, 0.0, 1.0).entropy())(
            -100.0
        )
    )

    shallow_tail = TruncatedNormal(jnp.array([-1e7]), 1e4, 0.0, 1.0)
    shallow_samples = shallow_tail.sample(seed=jax.random.key(2), sample_shape=(5_000,))
    np.testing.assert_allclose(shallow_samples.mean(), shallow_tail.mean(), atol=0.02)


def test_action_parameter_shapes():
    loc = jnp.zeros((2, 4, 3))
    TruncatedNormal(loc, 1.0, jnp.array([-1.0, 0.0, 0.0]), 1.0)

    with pytest.raises(AssertionError):
        TruncatedNormal(jnp.array(0.0), 1.0, 0.0, 1.0)
    with pytest.raises(AssertionError):
        TruncatedNormal(loc, jnp.ones((4, 2)), 0.0, 1.0)
    with pytest.raises(AssertionError):
        TruncatedNormal(loc, 1.0, jnp.zeros(2), jnp.ones(3))
