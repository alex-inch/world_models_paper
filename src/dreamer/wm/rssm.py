from typing import Any

import chex
import jax
from jax.lax import stop_gradient as sg
import jax.numpy as jnp
from einops import einsum, pack, rearrange
from flax import nnx
from flax.nnx.nn.recurrent import Carry
from jaxtyping import Array, Shaped

from .config import DreamerConfig


@chex.dataclass
class RSSMOut:
    """Stores the state and outputs of each step of the RSSM. The posterior, which represents the updated
    latent distribution p(z) given an image, should be non-null over real data, and None during imagination."""

    hidden: Shaped[Array, "B T H"]
    latent_sample: Shaped[Array, "B T Z"]
    prior: Shaped[Array, "B T N K"]
    posterior: Shaped[Array, "B T N K"] | None = None

    @property
    def joint_latent(self) -> Shaped[Array, "B T Z+H"]:
        return jnp.concatenate([self.hidden, self.latent_sample], axis=-1)


class RSSM(nnx.Module):
    def __init__(self, cfg: DreamerConfig, rngs: nnx.Rngs):
        self.cfg = cfg
        self.rngs = rngs

        repr_in_features = cfg.embedding_dim + cfg.wm_hidden_dim

        self.gru = nnx.GRUCell(
            cfg.rnn_in_features, cfg.wm_hidden_dim, rngs=self.rngs, dtype=cfg.dtype
        )
        self.trans_model = nnx.Linear(
            cfg.wm_hidden_dim, cfg.latent_dim, rngs=self.rngs, dtype=cfg.dtype
        )
        self.repr_model = nnx.Linear(
            repr_in_features, cfg.latent_dim, rngs=self.rngs, dtype=cfg.dtype
        )

    def sample(self, logits: Shaped[Array, "B N*K"]) -> Shaped[Array, "B N*K"]:
        "Given the flattened logits of a latent posterior, return"
        # need to reshape each logit to 32x32 per example. Then we can sample
        # from the defined categorical distribtuions, and unflatten back to the full
        # latent dimension

        # For numerical stability, keeping this in f32
        logits = rearrange(logits, "B (N K) -> B N K", N=self.cfg.num_latent_dists)
        probs = nnx.softmax(logits.astype(jnp.float32), axis=-1)
        B, N, K = logits.shape

        draw = self.rngs.categorical(logits.astype(jnp.float32), axis=-1, shape=(B, N))
        draw = nnx.one_hot(draw, K, dtype=self.cfg.dtype)
        # straight-through estimator to get gradients through the sampling step
        # wrap the (probs - sg(probs)) in brackets to encourage a more numerically stably computation:
        # instead of (draw + probs) - probs, the bracketed expression will evaluate to zero correctly
        draw = draw + (probs - sg(probs))
        draw = rearrange(draw, "B N K -> B (N K)")
        return draw.astype(self.cfg.dtype)

    def initialize_state(self, batch_dim: int) -> Carry:
        rnn_input_shape = (batch_dim, self.cfg.rnn_in_features)
        hidden = self.gru.initialize_carry(rnn_input_shape, self.rngs)
        hidden = hidden.astype(self.cfg.dtype)  # nnx always inits to f32

        latent_shape = (batch_dim, self.cfg.latent_dim)
        latent = jnp.zeros(latent_shape, dtype=self.cfg.dtype)

        return (hidden, latent)

    # This scan order expects batch-major inputs
    @nnx.scan(in_axes=(None, nnx.Carry, 1), out_axes=(nnx.Carry, 1))
    def unroll(
        self,
        state: tuple[Shaped[Array, "B H"], Shaped[Array, "B Z"]],
        inputs: tuple[Shaped[Array, "B ..."], ...],
    ):
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

        gru_in, _ = pack([prev_latent, prev_action], "B *")
        hidden, _ = self.gru(prev_hidden, gru_in)

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
        initial_state: tuple[Shaped[Array, "B H"], Shaped[Array, "B Z"]],
    ) -> RSSMOut:
        chex.assert_rank([embedding, action], 3)
        chex.assert_equal_shape_prefix([embedding, action, is_first], 2)
        chex.assert_axis_dimension(embedding, 2, self.cfg.embedding_dim)
        chex.assert_axis_dimension(action, 2, self.cfg.action_dim)

        initial_state = jax.lax.stop_gradient(initial_state)
        _carry, out = self.unroll(initial_state, (embedding, action, is_first))

        hidden, latent, prior, posterior = out

        # unwrap the latent and posterior into N categorical distributions of K classes
        N = self.cfg.num_latent_dists
        K = self.cfg.num_latent_classes
        prior = rearrange(prior, "... (N K) -> ... N K", N=N, K=K)
        posterior = rearrange(posterior, "... (N K) -> ... N K", N=N, K=K)
        return RSSMOut(
            hidden=hidden, prior=prior, posterior=posterior, latent_sample=latent
        )

    def init_imagination(self, embedding: Shaped[Array, "B E"]) -> RSSMOut:
        chex.assert_rank(embedding, 2)
        B, _E = embedding.shape

        # Compute the first hidden state using the image input
        initial_hidden, initial_latent = self.initialize_state(B)
        zero_action = jnp.zeros(shape=(B, self.cfg.action_dim), dtype=embedding.dtype)
        gru_in, _ = pack([initial_latent, zero_action], "B *")
        hidden, _ = self.gru(initial_hidden, gru_in)

        # Pass the initial hidden state to the representation model to get the posterior sample z
        repr_in, _ = pack([hidden, embedding], "B *")
        posterior_logits = self.repr_model(repr_in)
        latent = self.sample(posterior_logits)

        prior = self.trans_model(hidden)

        # Reshape
        N = self.cfg.num_latent_dists
        K = self.cfg.num_latent_classes
        prior = rearrange(prior, "... (N K) -> ... N K", N=N, K=K)
        return RSSMOut(hidden=hidden, latent_sample=latent, prior=prior)

    def imagine_step(self, rssm_state: RSSMOut, action: Array):
        h, z = rssm_state.hidden, rssm_state.latent_sample

        # Update hidden state
        gru_in, _ = pack([z, action], "B *")
        h, _ = self.gru(h, gru_in)

        # Update and sample a new latent
        prior = self.trans_model(h)
        z = self.sample(prior)

        # Reshape
        N = self.cfg.num_latent_dists
        K = self.cfg.num_latent_classes
        prior = rearrange(prior, "... (N K) -> ... N K", N=N, K=K)
        # Note: no posterior because we don't have ground truth images embeddings any more!
        return RSSMOut(hidden=h, latent_sample=z, prior=prior)


class Encoder(nnx.Module):
    """Embeds image inputs into a latent vector for use by the Dreamer RSSM."""

    def __init__(self, cfg: DreamerConfig, *, rngs: nnx.Rngs):
        # The original implementation has branches for both images (CNNs) and proprioception (MLPs). Since we're only
        # solving car racing I've simplified it to just the image path.
        # fmt: off
        # ruff: ignore[unnecessary-collection-call]
        conv_params: dict[str, Any] = dict(kernel_size=(4, 4), strides=2, padding="VALID", rngs=rngs, dtype=cfg.dtype)
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
        # ruff: ignore[unnecessary-collection-call]
        conv_params: dict[str, Any] = dict(
            strides=2, padding="VALID", rngs=rngs, dtype=cfg.dtype
        )
        self.layers = nnx.Sequential(
            nnx.Linear(in_feat, 384, rngs=rngs, dtype=cfg.dtype),  # 1x1
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
        return img_means
