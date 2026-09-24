import jax.numpy as jnp
import pytest
from flax import nnx
from wm_test_utils import all_zero

from dreamer.wm import DreamerBatch, DreamerConfig, DreamerWM
from dreamer.wm.losses import compute_balanced_kl

BATCH_SIZE = 10
SEQ_LENGTH = 5
N_CHANNELS = 3
ACTION_DIM = 5
HIDDEN_DIM = 50


def get_data(rngs) -> DreamerBatch:
    imgs = rngs.normal((BATCH_SIZE, SEQ_LENGTH, 64, 64, N_CHANNELS))
    reward = rngs.normal((BATCH_SIZE, SEQ_LENGTH))
    cont = rngs.normal((BATCH_SIZE, SEQ_LENGTH))
    action = rngs.normal((BATCH_SIZE, SEQ_LENGTH, ACTION_DIM))
    is_first = jnp.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=jnp.bool)
    is_first = is_first.at[:, 0].set(True)
    is_first = is_first.at[:, 2].set(True)

    return DreamerBatch(
        image=imgs,
        reward=reward,
        cont=cont,
        prev_action=action,
        is_first=is_first,
    )


@pytest.fixture(scope="module")
def run_wm():
    rngs = nnx.Rngs(0)
    cfg = DreamerConfig(
        num_latent_classes=3,
        num_latent_dists=5,
        action_dim=ACTION_DIM,
        wm_hidden_dim=HIDDEN_DIM,
        img_channels=N_CHANNELS,
    )
    model = DreamerWM(cfg, rngs=rngs)
    batch = get_data(rngs)
    out = model(batch)

    return model, cfg, batch, out


def test_world_model_shapes(run_wm):
    _model, cfg, _batch, out = run_wm

    assert out.image.shape == (
        BATCH_SIZE,
        SEQ_LENGTH,
        cfg.img_height,
        cfg.img_width,
        N_CHANNELS,
    )
    assert out.reward.shape == (BATCH_SIZE, SEQ_LENGTH)
    assert out.cont.shape == (BATCH_SIZE, SEQ_LENGTH)
    assert out.latent_prior.shape == (BATCH_SIZE, SEQ_LENGTH, cfg.latent_dim)
    assert out.latent_posterior.shape == (BATCH_SIZE, SEQ_LENGTH, cfg.latent_dim)
    assert out.latent_sample.shape == (BATCH_SIZE, SEQ_LENGTH, cfg.latent_dim)
    assert out.hidden.shape == (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)


def test_kl_loss(run_wm):
    "Test basic KL loss function working."
    _model, _cfg, _batch, out = run_wm

    # KL should be finite for normal inputs
    posterior = out.latent_posterior
    prior = out.latent_prior
    assert jnp.isfinite(compute_balanced_kl(posterior, prior, 0.5)).all()

    # KL of a distribution to itself should be 0
    assert all_zero(compute_balanced_kl(posterior, posterior, 0.5))
    assert all_zero(compute_balanced_kl(prior, prior, 0.5))

    # TODO: add test - can we compute something for a known distribution, not just test extremities


def test_kl_gradient_flow(run_wm):
    "Tests appropriate gradient flow through the KL stop grad terms"
    model, _cfg, batch, _out = run_wm

    # If alpha is 1.0, there should be no gradient to the prior network, and vice versa
    @nnx.grad
    def test_loss(model: DreamerWM, batch, alpha):
        out = model(batch)
        kl = compute_balanced_kl(out.latent_posterior, out.latent_prior, alpha)
        return kl

    grads = test_loss(model, batch, alpha=0.5)
    assert not all_zero(grads.rssm.repr_model)  # representation model <=> posterior
    assert not all_zero(grads.rssm.trans_model)  # trans model <=> posterior

    # if alpha is 0, no gradient flows to the transition model.
    grads_zero = test_loss(model, batch, alpha=0)
    assert not all_zero(grads_zero.rssm.repr_model)
    assert all_zero(grads_zero.rssm.trans_model)

    # if alpha is 1, gradient flows to the transition model, and no gradient
    # flows *directly* to the representation model...
    grads_one = test_loss(model, batch, alpha=1)
    assert not all_zero(grads_one.rssm.trans_model)

    # ...however, since the representation model contributes to the deterministic hidden state,
    # which is used by the transition model, it still has some, smaller gradients.
    assert not all_zero(grads_one.rssm.repr_model)
    assert (
        jnp.abs(grads_zero.rssm.repr_model.kernel).mean()
        > jnp.abs(grads_one.rssm.repr_model.kernel).mean()
    )
