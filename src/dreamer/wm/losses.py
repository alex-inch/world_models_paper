import chex
import distrax
import jax
import optax
from jaxtyping import Array, Shaped

from .world_model import DreamerBatch, DreamerWM


def nll(
    pred: Shaped[Array, "B T"] | Shaped[Array, "B T H W C"],
    true: Shaped[Array, "B T"] | Shaped[Array, "B T H W C"],
) -> Shaped[Array, ""]:
    "Computes the negative log likelihood for a unit Gaussian (i.e. the MSE)"
    chex.assert_rank([true, pred], {2, 5})  # expecting either scalar or image inputs

    # For Gaussians with unit covariance, the logprob is just the mean squared error,
    # and it's more efficient to compute that directly.
    mse = 0.5 * optax.losses.squared_error(pred, true)
    return mse.mean()


def _compute_kl(
    posterior: Shaped[Array, "B T Z"], prior: Shaped[Array, "B T Z"]
) -> Shaped[Array, "B T"]:
    # don't I need to reshape? Because it's actually K categoricals
    posterior_d = distrax.Categorical(logits=posterior)
    prior_d = distrax.Categorical(logits=prior)
    return posterior_d.kl_divergence(prior_d)  # type: ignore


def compute_balanced_kl(
    posterior: Shaped[Array, "B T Z"], prior: Shaped[Array, "B T Z"], alpha: float
) -> Shaped[Array, ""]:
    """Compute the 'balanced KL' described in DreamerV2. If alpha is 1, all the
    gradient will flow to the prior network, and none to the posterior. Vice-versa for alpha of 0"""
    sg = jax.lax.stop_gradient

    # fmt: off
    kl =        alpha  * _compute_kl(sg(posterior), prior) \
         + (1 - alpha) * _compute_kl(posterior, sg(prior))
    return kl.mean()
    # fmt : on


@chex.dataclass
class DreamerWMLosses:
    image_loss: Shaped[Array, ""]
    reward_loss: Shaped[Array, ""]
    discount_loss: Shaped[Array, ""]
    kl_loss: Shaped[Array, ""]


def dreamer_wm_loss_fn(
    model: DreamerWM, x: DreamerBatch, kl_alpha: float
) -> DreamerWMLosses:
    "Compute all the losses for the Dreamer world model, returning a dict of jax Arrays"
    chex.assert_rank(x.image, 5)  # ensure batch and time dimension are present

    preds = model(x)

    image_loss = nll(preds.image, x.image)
    reward_loss = nll(preds.reward, x.reward)  # TODO: symlog rewards
    discount_loss = nll(preds.discount, x.discount)  # TODO: should this be binary?

    kl_loss = compute_balanced_kl(preds.latent_posterior, preds.latent_prior, kl_alpha)

    chex.assert_rank([image_loss, reward_loss, discount_loss, kl_loss], 0)

    return DreamerWMLosses(
        image_loss=image_loss,
        reward_loss=reward_loss,
        discount_loss=discount_loss,
        kl_loss=kl_loss,
    )
