from collections.abc import Callable

import chex
from flax import nnx
from jaxtyping import Array, Shaped


@chex.dataclass
class DreamerConfig:
    "Defines the latent, action, hidden and embedding dimensions for a given model instantiation."

    num_latent_classes: int = 32  # latent dim = 1024
    num_latent_dists: int = 32
    action_dim: int = 3
    wm_hidden_dim: int = 1024
    img_height: int = 64
    img_width: int = 64
    img_channels: int = 3
    activation_fn: Callable = nnx.elu
    kl_alpha: float = 0.8
    kl_beta: float = 0.1
    td_lambda: float = 0.9
    agent_hidden_dim: int = 100
    discount: float = 0.997
    reward_encoding_bins: int = 19

    @property
    def latent_dim(self) -> int:
        return self.num_latent_classes * self.num_latent_dists

    @property
    def embedding_dim(self) -> int:
        return 2 * 2 * 384  # TODO: do this properly from the img_dims

    @property
    def rnn_in_features(self) -> int:
        return self.latent_dim + self.action_dim


@chex.dataclass
class HyperParams:
    alpha_kl: float = 0.8
    beta_kl: float = 1.0


# fmt: off
@chex.dataclass
class DreamerBatch:
    image:         Shaped[Array, "B T H W C"]
    reward:        Shaped[Array, "B T"]
    cont:          Shaped[Array, "B T"]
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
    cont:             Shaped[Array, "B T"]        # p(ɣ_t | h_t, z_t)
    latent_prior:     Shaped[Array, "B T Z"]      # p(z_t | h_t)
    latent_posterior: Shaped[Array, "B T Z"]      # p(z_t | h_t, x_t)
    latent_sample:    Shaped[Array, "B T Z"]      # z ~ p(z_t | h_t, x_t)
    hidden:           Shaped[Array, "B T R"]      # f(h_t | h_t1, z_t1, a_t1)
# fmt: on


@chex.dataclass
class DreamerWMLosses:
    image_loss: Shaped[Array, ""]
    reward_loss: Shaped[Array, ""]
    continue_loss: Shaped[Array, ""]
    kl_loss: Shaped[Array, ""]
