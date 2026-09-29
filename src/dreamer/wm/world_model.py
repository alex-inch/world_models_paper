import chex
import optax
from einops import rearrange
from flax import nnx
from jaxtyping import Array, Shaped

from .config import DreamerBatch, DreamerConfig, DreamerWMLosses, DreamerWMOut
from .losses import compute_balanced_kl, nll, twohot_loss
from .rssm import RSSM, Decoder, Encoder


class DreamerWM(nnx.Module):
    def __init__(self, cfg: DreamerConfig, *, rngs: nnx.Rngs):
        self.encoder = Encoder(cfg=cfg, rngs=rngs)
        self.decoder = Decoder(cfg=cfg, rngs=rngs)
        self.rssm = RSSM(cfg=cfg, rngs=rngs)

        state_dims = cfg.wm_hidden_dim + cfg.latent_dim
        self.continue_pred = nnx.Linear(state_dims, 1, rngs=rngs)
        self.reward_pred = nnx.Linear(state_dims, cfg.reward_encoding_bins, rngs=rngs)

    def predict_reward_and_continue(
        self, joint_latent: Shaped[Array, "B T Z+H"]
    ) -> tuple[Shaped[Array, "B T K"], Shaped[Array, "B T"]]:
        cont = self.continue_pred(joint_latent)
        reward = self.reward_pred(joint_latent)

        # Remove singleton final dimension. Note we're returning `cont` as logits instead of
        # passing it through a sigmoid - for more numerically stable loss computation.
        cont = rearrange(cont, "... 1 -> ...")
        return reward, cont

    def __call__(self, input: DreamerBatch) -> DreamerWMOut:
        embedding = self.encoder(input.image)
        wm_pred = self.rssm(embedding, input.prev_action, input.is_first)

        joint_latent = wm_pred.joint_latent
        image_pred = self.decoder(joint_latent)
        reward_pred, cont_pred = self.predict_reward_and_continue(joint_latent)
        return DreamerWMOut(
            image=image_pred,
            reward=reward_pred,
            cont=cont_pred,
            latent_prior=wm_pred.prior,
            latent_posterior=wm_pred.posterior,
            latent_sample=wm_pred.latent,
            hidden=wm_pred.hidden,
        )

    @staticmethod
    def loss_fn(
        preds: DreamerWMOut, x: DreamerBatch, kl_alpha: float
    ) -> DreamerWMLosses:
        "Compute all the losses for the Dreamer world model, returning an object containing each one."
        chex.assert_rank(x.image, 5)  # ensure batch and time dimension are present

        image_loss = nll(preds.image, x.image)
        continue_loss = optax.sigmoid_binary_cross_entropy(preds.cont, x.cont).mean()

        # Symlog + twohot encoded reward
        reward_loss = twohot_loss(preds, x)

        kl_loss = compute_balanced_kl(
            preds.latent_posterior, preds.latent_prior, kl_alpha
        )

        chex.assert_rank([image_loss, reward_loss, continue_loss, kl_loss], 0)

        return DreamerWMLosses(
            image_loss=image_loss,
            reward_loss=reward_loss,
            continue_loss=continue_loss,
            kl_loss=kl_loss,
        )
