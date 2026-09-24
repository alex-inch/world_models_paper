import jax.numpy as jnp
from einops import einsum, rearrange
from jaxtyping import Array, ArrayLike, Shaped


def get_bins(n_classes=19):
    "Construct the vector of bins B used in twohot encoding."
    assert n_classes % 2 == 1  # should be odd for how this is being constructed.
    minval = -(n_classes // 2)
    maxval = n_classes // 2
    B = jnp.arange(minval, maxval + 1)
    return B


def twohot_encode(
    x: Shaped[Array, "..."],
    B: Shaped[Array, "N"],  # ruff: ignore[undefined-name]
) -> Shaped[Array, "... N"]:
    """The twohot encoding is a generalisation of onehot encoding which can smoothly represent any real value.

    The input to this function should be in symlog space for Dreamer-style reward scaling.

    This function also assumes that bins in B have a constant spacing of 1.
    """
    # This implementation works by computing the distance between x and the bin centroids.
    # We say that, for centroid Bi, |x - Bi| gives the unsigned distance to the centroid. If we take
    # 1 - |x - Bi|, we'll get values in [0, 1] for the two closest centroids, or a negative
    # number otherwise. By setting negative numbers to 0 we get a two hot encoding.
    # To reason about it - consider the case where x is directly on a centroid, or equidistant between two.
    # Note that in general, we'd need to divide by the distance between centroids (x - Bi / w), but we just fix it to 1 for this.
    # Idk if there's a better way to do it, this is just what I cooked up.

    # Set up axes so each reward gets broadcast to the N classes
    # assert B[1] - B[0] == 1

    x_bc = rearrange(x, "... -> ... 1")
    distances = 1 - jnp.abs(x_bc - B)  # broadcasts [... 1] - [N] -> [... N]
    encoding = jnp.maximum(distances, 0)
    return encoding


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
