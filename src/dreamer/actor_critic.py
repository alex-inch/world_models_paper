import chex
import jax
import jax.numpy as jnp
from einops import pack, rearrange
from flax import nnx
from jaxtyping import Array, Float

from dreamer.truncated_normal import TruncatedNormal

from .wm import DreamerConfig


@chex.dataclass
class ActorLoss:
    reward_loss: Float[Array, ""]
    entropy_loss: Float[Array, ""]

    @property
    def loss(self) -> Float[Array, ""]:
        return self.reward_loss + self.entropy_loss


class Actor(nnx.Module):
    def __init__(self, cfg: DreamerConfig, rngs: nnx.Rngs):
        self.cfg = cfg
        input_dim = cfg.latent_dim + cfg.wm_hidden_dim

        self.actor_net = nnx.Sequential(
            nnx.Linear(input_dim, cfg.agent_hidden_dim, rngs=rngs),
            cfg.activation_fn,
            nnx.Linear(cfg.agent_hidden_dim, cfg.agent_hidden_dim, rngs=rngs),
            cfg.activation_fn,
            nnx.Linear(cfg.agent_hidden_dim, 2 * cfg.action_dim, rngs=rngs),
        )

    def __call__(self, x: Float[Array, "... Z+H"]) -> TruncatedNormal:
        raw_mean, raw_std = jnp.split(self.actor_net(x), 2, axis=-1)
        lower = jnp.array([-1.0, 0.0, 0.0])
        upper = jnp.array([1.0, 1.0, 1.0])
        midpoint = (lower + upper) / 2
        half_range = (upper - lower) / 2

        # Bound the parameters in [-1, 1] coordinates, then rescale for CarRacing.
        mean = midpoint + half_range * jnp.tanh(raw_mean)
        std = half_range * (2 * jax.nn.sigmoid(raw_std / 2) + 0.1)
        # Batch the bounds so the imagination scan stacks all distribution parameters consistently.
        return TruncatedNormal(
            mean,
            std,
            lower=jnp.broadcast_to(lower, mean.shape),
            upper=jnp.broadcast_to(upper, mean.shape),
        )

    @staticmethod
    def loss_fn(
        cfg: DreamerConfig,
        rewards: Float[Array, "Batch Time"],
        continues: Float[Array, "Batch Time"],
        values: Float[Array, "Batch Time+1"],
        action_dists: TruncatedNormal,  # Samples are shape [B T A]
    ) -> tuple[ActorLoss, Float[Array, "Batch Time"], Float[Array, "Batch Time"]]:
        # We need an extra value - one for each transition + the terminal value
        B, T = rewards.shape
        chex.assert_shape(continues, (B, T))
        chex.assert_shape(values, (B, T + 1))

        td_return = compute_td_lambda_returns(cfg, rewards, continues, values)

        # The actor *maximises* the return, so we return the negative value,
        # which the optimiser will minimise, plus an entropy term to encourage exploration
        # The hardcoded 1e-4 is from the Dreamer paper. TODO(Alex): stick it on the config
        weights = compute_step_weights(continues, cfg.discount)
        reward_loss = -(weights * td_return).mean()
        entropy_loss = -1e-4 * (weights * action_dists.entropy()).mean()

        return (
            ActorLoss(
                reward_loss=reward_loss,
                entropy_loss=entropy_loss,
            ),
            td_return,
            weights,
        )


class Critic(nnx.Module):
    def __init__(self, cfg: DreamerConfig, rngs: nnx.Rngs):
        self.cfg = cfg
        input_dim = cfg.latent_dim + cfg.wm_hidden_dim

        self.critic_net = nnx.Sequential(
            nnx.Linear(input_dim, cfg.agent_hidden_dim, rngs=rngs),
            cfg.activation_fn,
            nnx.Linear(cfg.agent_hidden_dim, cfg.agent_hidden_dim, rngs=rngs),
            cfg.activation_fn,
            nnx.Linear(cfg.agent_hidden_dim, 1, rngs=rngs),
        )

    def __call__(self, features: Float[Array, "... Z+H"]) -> Float[Array, "..."]:
        value_pred = self.critic_net(features)
        return rearrange(value_pred, "... 1 -> ...")

    @staticmethod
    def loss_fn(
        values: Float[Array, "B T"],
        targets: Float[Array, "B T"],
        weights: Float[Array, "B T"],
    ):
        return (weights * (values - targets) ** 2).mean()


def compute_td_lambda_returns(
    cfg: DreamerConfig,
    rewards: Float[Array, "Batch Time"],
    continues: Float[Array, "Batch Time"],
    values: Float[Array, "Batch Time+1"],
) -> Float[Array, "Batch Time"]:
    discount = cfg.discount
    lam = cfg.td_lambda

    # Compute the TD(lambda) return from the final state, summing backwards recursively
    @nnx.scan(
        in_axes=(1, 1, 1, nnx.Carry),
        out_axes=(1, nnx.Carry),
        reverse=True,
    )
    def _compute_lambda_return(r_t: Array, c_t: Array, v_t1: Array, G_t1: Array):
        G_t = r_t + discount * c_t * ((1 - lam) * v_t1 + lam * G_t1)
        # first output stores the results per-timestep, the second acts as the carry
        return G_t, G_t

    # Take the terminal value as the initial input for the backwards loop
    carry = values[..., -1]
    target, _ = _compute_lambda_return(rewards, continues, values[..., 1:], carry)
    return target


def compute_step_weights(continues: Float[Array, "B T"], discount: float):
    """Dreamer discounts subsequent steps contributing to the loss based on their continue
    prediction. Ie. if we think a step should come *after* the end of an episode (cont_i=0),
    then we want to disregard any contribution it makes to the loss.
    """
    chex.assert_rank(continues, 2)
    discounts = discount * continues
    cumulative_discounts = jnp.cumprod(discounts, axis=1)  # cumprod over time dimension

    # the discounts are associated with the probability of proceeding from a state,
    # but we want weights for the state before. So we offset our array of weights from:
    # [w0, w1, ..., wT] -> [1, w0, ..., wT-1]
    B, _T = continues.shape
    first = jnp.ones(shape=(B, 1))
    rest = cumulative_discounts[:, :-1]
    weights, _ = pack([first, rest], "B *")
    return jax.lax.stop_gradient(weights)
