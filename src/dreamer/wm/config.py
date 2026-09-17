from collections.abc import Callable

import chex
from flax import nnx


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
