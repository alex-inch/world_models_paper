import jax
import jax.numpy as jnp
import pytest

from dreamer.preprocessing import get_bins, symexp, symlog, twohot_decode, twohot_encode


@pytest.fixture
def inputs():
    B = get_bins(19)
    x = jax.random.uniform(jax.random.key(0), shape=(4, 5)) * 100
    return x, B


def test_twohot_encoding_roundtrip(inputs):
    x, B = inputs
    encoded = twohot_encode(symlog(x), B)
    decoded = symexp(twohot_decode(encoded, B))
    assert jnp.allclose(decoded, x)


def test_twohot_encoding_sums_to_one(inputs):
    x, B = inputs
    encoded = twohot_encode(symlog(x), B)
    assert jnp.all(encoded >= 0)
    assert jnp.allclose(encoded.sum(axis=-1), 1)


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
