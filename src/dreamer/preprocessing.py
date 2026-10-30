import jax
import jax.numpy as jnp
from einops import einsum
from jaxtyping import Array, ArrayLike, Shaped


def get_bins(n_bins: int = 127, max_exp: int = 9):
    "Construct the vector of bins B used in twohot encoding."
    assert n_bins % 2 == 1  # should be odd for how this is being constructed.
    half = n_bins // 2
    return jnp.arange(-half, half + 1, dtype=jnp.float32) / half * max_exp


def twohot_encode(
    x: Shaped[Array, "..."],
    B: Shaped[Array, "N"],  # ruff: ignore[undefined-name]
) -> Shaped[Array, "... N"]:
    """The twohot encoding is a generalisation of onehot encoding which can smoothly represent any real value.

    The input to this function should be in symlog space for Dreamer-style reward scaling.

    B must contain at least two strictly increasing bin centres. Values outside
    the bin range are clipped to its endpoints.
    """
    x = jnp.clip(x, B[0], B[-1])
    hi = jnp.clip(jnp.searchsorted(B, x, side="right"), 1, B.size - 1)
    lo = hi - 1

    # Use the enclosing bins' actual spacing: even an evenly spaced grid has
    # slightly different interval widths after rounding to float32.
    w_hi = (x - B[lo]) / (B[hi] - B[lo])
    return (
        jax.nn.one_hot(lo, B.size) * (1 - w_hi[..., None])
        + jax.nn.one_hot(hi, B.size) * w_hi[..., None]
    )


def twohot_decode(
    x: Shaped[Array, "... N"],
    B: Shaped[Array, "N"],  # ruff: ignore[undefined-name]
) -> Shaped[Array, "..."]:
    """Reverse process of the twohot encoding.

    If doing Dreamer-style use, may need to symexp the output for appropriate scaling.
    """
    # This one is easier - just return a weighted sum of the activated bins with a dot product
    return einsum(x, B, "... i, i -> ...")


def symlog(x: ArrayLike):
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def symexp(x: ArrayLike):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)
