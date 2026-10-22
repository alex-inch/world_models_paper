import chex
import distrax
import einops
import jax
import optax
from jaxtyping import Array, Shaped

from dreamer.preprocessing import get_bins, symlog, twohot_encode
from dreamer.wm import DreamerConfig

from .config import DreamerWMLosses
from .world_model import DreamerBatch, DreamerWMOut


def nll(
    pred: Shaped[Array, "B T"] | Shaped[Array, "B T H W C"],
    true: Shaped[Array, "B T"] | Shaped[Array, "B T H W C"],
) -> Shaped[Array, ""]:
    "Computes the negative log likelihood for a unit Gaussian (i.e. the MSE)"
    chex.assert_rank([true, pred], {2, 5})  # expecting either scalar or image inputs

    # For Gaussians with unit covariance, the logprob is just the mean squared error,
    # and it's more efficient to compute that directly.
    mse = 0.5 * optax.losses.squared_error(pred, true)
    # sum over pixels
    mse = einops.reduce(mse, "B T ... -> B T", reduction="sum")
    return mse.mean()


def _compute_kl(
    posterior: Shaped[Array, "B T N K"], prior: Shaped[Array, "B T N K"]
) -> Shaped[Array, "B T N"]:
    posterior_d = distrax.Categorical(logits=posterior)
    prior_d = distrax.Categorical(logits=prior)
    return posterior_d.kl_divergence(prior_d)  # type: ignore


def compute_balanced_kl(
    posterior: Shaped[Array, "B T N K"], prior: Shaped[Array, "B T N K"], alpha: float
) -> Shaped[Array, ""]:
    """Compute the 'balanced KL' described in DreamerV2. If alpha is 1, all the
    gradient will flow to the prior network, and none to the posterior. Vice-versa for alpha of 0"""
    sg = jax.lax.stop_gradient

    # fmt: off
    kl =        alpha  * _compute_kl(sg(posterior),    prior) \
         + (1 - alpha) * _compute_kl(   posterior , sg(prior))
    # fmt : on

    # Sum the KL over each of the N distributions
    kl = kl.sum(axis=-1)
    # Then average over the batch/time dims
    return kl.mean()


def twohot_loss(preds, x, max_exp=9):
    "Compute the symexp categorical loss described in Dreamerv3."
    B = get_bins(n_bins=preds.reward.shape[-1], max_exp=max_exp)
    r_scaled = symlog(x.reward)
    true_reward_encoded = twohot_encode(r_scaled, B=B)
    reward_loss = optax.softmax_cross_entropy(preds.reward, true_reward_encoded)
    reward_loss = reward_loss.mean()
    return reward_loss


def dreamer_wm_loss_fn(
    preds: DreamerWMOut, x: DreamerBatch, cfg: DreamerConfig
) -> DreamerWMLosses:
    "Compute all the losses for the Dreamer world model, returning an object containing each one."
    chex.assert_rank(x.image, 5)  # ensure batch and time dimension are present

    image_loss = nll(preds.image, x.image)
    continue_loss = optax.sigmoid_binary_cross_entropy(preds.cont, x.cont).mean()
    reward_loss = twohot_loss(preds, x, cfg.reward_encoding_max_exp)
    kl_loss = compute_balanced_kl(
        preds.latent_posterior, preds.latent_prior, cfg.kl_alpha
    )

    chex.assert_rank([image_loss, reward_loss, continue_loss, kl_loss], 0)

    return DreamerWMLosses(
        image_loss=image_loss,
        reward_loss=reward_loss,
        continue_loss=continue_loss,
        kl_loss=kl_loss,
    )
