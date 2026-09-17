import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from wm_test_utils import all_zero

from dreamer.wm import RSSM, DreamerConfig

BATCH_SIZE = 5
SEQ_LENGTH = 5

CFG = DreamerConfig()


def make_data(
    cfg: DreamerConfig,
    rngs: nnx.Rngs,
    batch_size: int = BATCH_SIZE,
    seq_length: int = SEQ_LENGTH,
):
    embedding = rngs.normal(shape=(batch_size, seq_length, cfg.embedding_dim))
    prev_action = rngs.normal(shape=(batch_size, seq_length, cfg.action_dim))
    is_first = jnp.zeros((batch_size, seq_length), dtype=jnp.bool)
    is_first = is_first.at[:, 0].set(True)

    return embedding, prev_action, is_first


def test_valid_grads():
    """Confirm that all elements of the RSSM receive nonzero gradients, and the straight-through
    estimator works correctly for propagating gradients through the latent sample."""
    rngs = nnx.Rngs(0)
    rssm = RSSM(CFG, rngs)
    inputs = make_data(CFG, rngs)

    @nnx.value_and_grad
    def loss_fn(rssm: RSSM, data: tuple[jax.Array, ...]):
        "This is not the true loss, just a dummy to check gradient flows"
        embedding, prev_action, is_first = data
        out = rssm(embedding, prev_action, is_first)
        hidden, latent, prior, posterior = out.to_tuple()  # type: ignore
        return jnp.mean(hidden**2 + latent**2 + prior**2)

    loss, grads = loss_fn(rssm, inputs)
    # loss is nonzero
    assert loss > 0
    # Check the grads are nonzero for each layer
    assert not all_zero(grads)

    # Check that the loss is larger for longer sequences - sniff test that we are
    # accumulating gradients across time steps
    inputs = make_data(CFG, rngs, seq_length=SEQ_LENGTH * 2)
    loss_long, _ = loss_fn(rssm, inputs)
    assert loss_long > loss


def test_shape_handling():
    "Check that the model raises when incorrectly shaped inputs are provided"
    rngs = nnx.Rngs(0)
    rssm = RSSM(CFG, rngs)
    embedding, prev_action, is_first = make_data(CFG, rngs)

    with pytest.raises(AssertionError):
        # screw up the dimensions - should raise
        rssm(embedding[0], prev_action, is_first)


def test_config_inferred_vals():
    """Check that some of the inferred values for the RSSM config:
    e.g. that the latent dimension is the product of number and class count of the categorical encodings"""
    cfg = DreamerConfig(num_latent_classes=5, num_latent_dists=6)
    assert cfg.latent_dim == cfg.num_latent_classes * cfg.num_latent_dists
