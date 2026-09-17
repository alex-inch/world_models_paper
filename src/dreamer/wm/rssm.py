from typing import Any

import chex
import jax
import jax.numpy as jnp
from einops import einsum, pack, rearrange
from flax import nnx
from flax.nnx.nn.recurrent import Carry
from jaxtyping import Array, Shaped

from .config import DreamerConfig


@chex.dataclass
class RSSMOut:
    hidden: Shaped[Array, "B T H"]
    latent: Shaped[Array, "B T Z"]
    prior: Shaped[Array, "B T Z"]
    posterior: Shaped[Array, "B T Z"]

    @property
    def joint_latent(self) -> Shaped[Array, "B T Z+H"]:
        return jnp.concatenate([self.hidden, self.latent], axis=-1)


class RSSM(nnx.Module):
    def __init__(self, cfg: DreamerConfig, rngs: nnx.Rngs):
        self.cfg = cfg
        self.rngs = rngs

        repr_in_features = cfg.embedding_dim + cfg.wm_hidden_dim

        self.gru = nnx.GRUCell(cfg.rnn_in_features, cfg.wm_hidden_dim, rngs=self.rngs)
        self.trans_model = nnx.Linear(cfg.wm_hidden_dim, cfg.latent_dim, rngs=self.rngs)
        self.repr_model = nnx.Linear(repr_in_features, cfg.latent_dim, rngs=self.rngs)

    def sample(self, logits: Shaped[Array, "B Z"]):
        "Given the flattened logits of a latent posterior, return"
        # need to reshape each logit to 32x32 per example. Then we can sample
        # from the defined categorical distribtuions, and unflatten back to the full
        # latent dimension
        logits = rearrange(logits, "B (N K) -> B N K", N=self.cfg.num_latent_dists)
        probs = nnx.softmax(logits, axis=-1)
        B, N, K = logits.shape

        draw = self.rngs.categorical(logits, axis=-1, shape=(B, N))
        draw = nnx.one_hot(draw, self.cfg.num_latent_classes)
        # straight-through estimator to get gradients through the sampling step
        draw = draw + probs - jax.lax.stop_gradient(probs)
        draw = rearrange(draw, "B N K -> B (N K)")
        return draw

    def initialize_state(self, batch_dim: int) -> Carry:
        rnn_input_shape = (batch_dim, self.cfg.rnn_in_features)
        hidden = self.gru.initialize_carry(rnn_input_shape, self.rngs)

        latent_shape = (batch_dim, self.cfg.latent_dim)
        latent = jnp.zeros(latent_shape, dtype=jnp.float32)

        return (hidden, latent)

    # This scan order expects batch-major inputs
    @nnx.scan(in_axes=(None, nnx.Carry, 1), out_axes=(nnx.Carry, 1))
    def unroll(self, state, inputs):
        embedding, prev_action, is_first = inputs
        # The sequences are packed in end-to-end, so sometimes they will contain the first
        # states of a sequence. In that case, we want to reset the model state, containing
        # the deterministic hidden and the stochastic latent, as well as ignoring the action.
        # For a branchless approach, we can mult by 0.0 for starts, using einsum to broadcast
        mask = 1.0 - is_first.astype(prev_action.dtype)
        state, prev_action = jax.tree.map(
            lambda a: einsum(mask, a, "B, B ... -> B ..."), (state, prev_action)
        )
        prev_hidden, prev_latent = state

        gru_input, _ = pack([prev_latent, prev_action], "B *")
        hidden, _ = self.gru(prev_hidden, gru_input)

        prior = self.trans_model(hidden)
        repr_in, _ = pack([hidden, embedding], "B *")
        posterior = self.repr_model(repr_in)

        latent = self.sample(posterior)

        state = (hidden, latent)  # carry reused for next timestep
        outputs = (hidden, latent, prior, posterior)  # These are what nnx will stack
        return state, outputs

    def __call__(
        self,
        embedding: Shaped[Array, "B T E"],
        action: Shaped[Array, "B T A"],
        is_first: Shaped[Array, "B T"],
    ) -> RSSMOut:
        chex.assert_rank([embedding, action], 3)
        chex.assert_equal_shape_prefix([embedding, action, is_first], 2)
        chex.assert_axis_dimension(embedding, 2, self.cfg.embedding_dim)
        chex.assert_axis_dimension(action, 2, self.cfg.action_dim)
        B, T, E = embedding.shape

        state = self.initialize_state(B)
        _, out = self.unroll(state, (embedding, action, is_first))

        hidden, latent, prior, posterior = out
        return RSSMOut(hidden=hidden, prior=prior, posterior=posterior, latent=latent)


class Encoder(nnx.Module):
    """Embeds image inputs into a latent vector for use by the Dreamer RSSM."""

    def __init__(self, cfg: DreamerConfig, *, rngs: nnx.Rngs):
        # The original implementation has branches for both images (CNNs) and proprioception (MLPs). Since we're only
        # solving car racing I've simplified it to just the image path.
        # fmt: off
        conv_params: dict[str, Any] = dict(kernel_size=(4, 4), strides=2, padding="VALID", rngs=rngs)
        self.layers= nnx.Sequential(
            nnx.Conv(in_features=cfg.img_channels, out_features=48,  **conv_params),  # 31 x 31
            cfg.activation_fn,
            nnx.Conv(in_features=48,  out_features=96,  **conv_params),  # 14 x 14
            cfg.activation_fn,
            nnx.Conv(in_features=96,  out_features=192, **conv_params),  #  6 x 6
            cfg.activation_fn,
            nnx.Conv(in_features=192, out_features=384, **conv_params)   #  2 x 2
        )
        # fmt: on

    def __call__(self, x: Shaped[Array, "... H W C"]) -> Shaped[Array, "... E"]:
        x = self.layers(x)
        x = rearrange(x, "... H W C -> ... (H W C)")  # flatten
        return x


class Decoder(nnx.Module):
    def __init__(
        self,
        cfg: DreamerConfig,
        *,
        rngs: nnx.Rngs,
    ):
        in_feat = cfg.wm_hidden_dim + cfg.latent_dim
        conv_params: dict[str, Any] = dict(strides=2, padding="VALID", rngs=rngs)
        self.layers = nnx.Sequential(
            nnx.Linear(in_feat, 384, rngs=rngs),  # 1x1
            cfg.activation_fn,
            nnx.ConvTranspose(384, 192, kernel_size=(5, 5), **conv_params),  # 5x5
            cfg.activation_fn,
            nnx.ConvTranspose(192, 96, kernel_size=(5, 5), **conv_params),  # 13x13
            cfg.activation_fn,
            nnx.ConvTranspose(96, 48, kernel_size=(6, 6), **conv_params),  # 30x30
            cfg.activation_fn,
            nnx.ConvTranspose(
                48, cfg.img_channels, kernel_size=(6, 6), **conv_params
            ),  # 64x64
        )

    def __call__(
        self, joint_latent: Shaped[Array, "... H+Z"]
    ) -> Shaped[Array, "... H W C"]:
        joint_latent = rearrange(joint_latent, "... (1 1 C) -> ... 1 1 C")
        img_means = self.layers(joint_latent)
        # TOOD: output? sample?  What do?
        return img_means
