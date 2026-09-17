import chex
import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from jaxtyping import Array, Shaped

from .config import DreamerConfig
from .rssm import RSSM, Decoder, Encoder


# fmt: off
@chex.dataclass
class DreamerBatch:
    image:         Shaped[Array, "B T H W C"]
    reward:        Shaped[Array, "B T"]
    discount:      Shaped[Array, "B T"]
    prev_action:   Shaped[Array, "B T A"]
    is_first:      Shaped[Array, "B T"]


@chex.dataclass
class DreamerWMOut: 
    # Dim definitions
    #   B - batch
    #   T - timestep
    #   H - image height
    #   W - image width
    #   C - image channels
    #   N - number of latent categorical distributions
    #   K - number of classes per latent categorical distribution
    #   R - hidden dimension of recurrent network
    image:            Shaped[Array, "B T H W C"]  # p(x_t | h_t, z_t)
    reward:           Shaped[Array, "B T"]        # p(r_t | h_t, z_t)
    discount:         Shaped[Array, "B T"]        # p(ɣ_t | h_t, z_t)
    latent_prior:     Shaped[Array, "B T Z"]      # p(z_t | h_t)
    latent_posterior: Shaped[Array, "B T Z"]      # p(z_t | h_t, x_t)
    latent_sample:    Shaped[Array, "B T Z"]      # z ~ p(z_t | h_t, x_t)
    hidden:           Shaped[Array, "B T R"]      # f(h_t | h_t1, z_t1, a_t1)
# fmt: on


class DreamerWM(nnx.Module):
    def __init__(self, cfg: DreamerConfig, *, rngs: nnx.Rngs):
        self.encoder = Encoder(cfg=cfg, rngs=rngs)
        self.decoder = Decoder(cfg=cfg, rngs=rngs)
        self.rssm = RSSM(cfg=cfg, rngs=rngs)

        self.pred_layer = nnx.Linear(cfg.wm_hidden_dim + cfg.latent_dim, 2, rngs=rngs)

    def predict_reward_and_gamma(
        self, joint_latent: Shaped[Array, "B T Z+H"]
    ) -> tuple[Shaped[Array, "B T"], ...]:
        scalars = self.pred_layer(joint_latent)
        reward, gamma = jnp.split(scalars, 2, axis=-1)

        # Scale discount into the range [0, 1].
        gamma = nnx.sigmoid(gamma)

        # Remove singleton final dimension
        reward = rearrange(reward, "... 1 -> ...")
        gamma = rearrange(gamma, "... 1 -> ...")
        return reward, gamma

    def __call__(self, input: DreamerBatch) -> DreamerWMOut:
        embedding = self.encoder(input.image)
        wm_pred = self.rssm(embedding, input.prev_action, input.is_first)

        joint_latent = wm_pred.joint_latent
        image_pred = self.decoder(joint_latent)
        reward_pred, discount_pred = self.predict_reward_and_gamma(joint_latent)
        return DreamerWMOut(
            image=image_pred,
            reward=reward_pred,
            discount=discount_pred,
            latent_prior=wm_pred.prior,
            latent_posterior=wm_pred.posterior,
            latent_sample=wm_pred.latent,
            hidden=wm_pred.hidden,
        )


# Loss funcs for DreamerV2
