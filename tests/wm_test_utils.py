import jax
import jax.numpy as jnp
from jaxtyping import PyTree


def all_zero(pytree: PyTree):
    bool_tree = jax.tree.map(lambda a: jnp.allclose(a, jnp.zeros_like(a)).all(), pytree)
    return jax.tree.all(bool_tree)
