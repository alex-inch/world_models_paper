import jax
import jax.numpy as jnp
import pytest

from dreamer.preprocessing import get_bins, symexp, symlog, twohot_decode, twohot_encode
from dreamer.wm import DreamerConfig


@pytest.fixture(params=[
    pytest.param((19, 9), id="19-bins-max9"),
    pytest.param((127, 9), id="127-bins-max9"),
    pytest.param((255, 9), id="255-bins-max9"),
    pytest.param((255, 20), id="255-bins-max20"),
])
def inputs(request):
    n_bins, max_exp = request.param
    B = get_bins(n_bins=n_bins, max_exp=max_exp)
    scaled = jax.random.uniform(
        jax.random.key(0), shape=(4, 5), minval=-max_exp, maxval=max_exp
    )
    x = symexp(scaled)
    return x, B


@pytest.mark.parametrize("n_bins,max_exp", [(19, 3), (127, 9), (255, 4), (255, 20)])
def test_get_bins_uses_requested_count_and_limits(n_bins, max_exp):
    B = get_bins(n_bins=n_bins, max_exp=max_exp)
    assert B.shape == (n_bins,)
    assert B[0] == -max_exp
    assert B[-1] == max_exp
    assert jnp.isclose(B[n_bins // 2], 0)
    assert jnp.all(jnp.diff(B) > 0)


def test_default_bins_match_config():
    cfg = DreamerConfig()
    B = get_bins()
    assert B.shape == (cfg.reward_encoding_bins,)
    assert B[0] == -cfg.reward_encoding_max_exp
    assert B[-1] == cfg.reward_encoding_max_exp


def test_twohot_encoding_roundtrip(inputs):
    x, B = inputs
    encoded = twohot_encode(symlog(x), B)
    decoded = symexp(twohot_decode(encoded, B))
    assert jnp.allclose(decoded, x)


def test_twohot_encoding_sums_to_one(inputs):
    x, B = inputs
    encoded = twohot_encode(symlog(x), B)
    assert encoded.shape == (*x.shape, B.size)
    assert jnp.all(encoded >= 0)
    assert jnp.allclose(encoded.sum(axis=-1), 1)


def test_twohot_has_at_most_two_nonzero_weights(inputs):
    x, B = inputs
    encoded = twohot_encode(symlog(x), B)
    assert jnp.all((encoded > 0).sum(axis=-1) <= 2)


def test_twohot_bin_centres_are_onehot(inputs):
    _, B = inputs
    encoded = twohot_encode(B, B)
    assert jnp.allclose(encoded, jnp.eye(B.size))


def test_twohot_midpoints_split_mass_equally(inputs):
    _, B = inputs
    midpoints = (B[:-1] + B[1:]) / 2
    expected = (jnp.eye(B.size)[:-1] + jnp.eye(B.size)[1:]) / 2
    encoded = twohot_encode(midpoints, B)
    # Float32 midpoints can round slightly towards one of the enclosing bins.
    assert jnp.allclose(encoded, expected, atol=1e-5)


def test_twohot_clips_to_endpoints(inputs):
    _, B = inputs
    scaled = jnp.array([B[0] - 100, B[0], B[-1], B[-1] + 100])
    encoded = twohot_encode(scaled, B)
    expected = jnp.eye(B.size)[jnp.array([0, 0, B.size - 1, B.size - 1])]
    assert jnp.allclose(encoded, expected)
    assert jnp.allclose(twohot_decode(encoded, B), jnp.clip(scaled, B[0], B[-1]))


def test_symlog_symexp_roundtrip():
    # invertible mappings
    assert symexp(symlog(1)) == 1
    assert symexp(symlog(0.5)) == 0.5
    assert symexp(symlog(-1)) == -1
    assert symexp(symlog(0)) == 0


def test_symlog_expected_values():
    assert symlog(0) == 0
    assert symlog(-1) == -symlog(1)
    assert symlog(2) == jnp.log(3)


def test_symexp_expected_values():
    assert symexp(0) == 0
    assert symexp(-1) == -symexp(1)
    assert symexp(2) == jnp.exp(2) - 1
