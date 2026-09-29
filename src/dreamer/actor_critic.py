import chex
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxtyping import Array, Float

from dreamer.truncated_normal import TruncatedNormal
from dreamer.wm.world_model import DreamerWMOut

from .wm import DreamerConfig


@chex.dataclass
class DreamerActorCriticOut:
    action_dist: TruncatedNormal
    value_pred: Float[Array, "Batch Time"]


@chex.dataclass
class DreamerActorCriticLoss:
    critic_loss: Float[Array, ""]
    actor_loss: Float[Array, ""]
    loss: Float[Array, ""]
    target: Float[Array, "Batch Time"]


class ActorCritic(nnx.Module):
    def __init__(self, cfg: DreamerConfig, rngs: nnx.Rngs):
        input_dim = cfg.latent_dim + cfg.wm_hidden_dim + cfg.action_dim

        self.backbone = nnx.Sequential(
            nnx.Linear(input_dim, cfg.agent_hidden_dim, rngs=rngs),
            cfg.activation_fn,
            nnx.Linear(cfg.agent_hidden_dim, cfg.agent_hidden_dim, rngs=rngs),
            cfg.activation_fn,
        )

        self.critic_head = nnx.Linear(cfg.agent_hidden_dim, 1, rngs=rngs)
        self.actor_head = nnx.Linear(
            cfg.agent_hidden_dim, 2 * cfg.action_dim, rngs=rngs
        )

    def actor(self, z, key):
        # Predict Gaussian actions
        action_stats = self.actor_head(z)
        action_mean, action_std = jnp.split(action_stats, 2, axis=-1)
        action_std = jnp.exp(action_std)  # make the standard deviation positive

        # Output Gaussian actions in [-1:1, 0:1, 0:1]
        lower = jnp.array([-1, 0, 0])
        upper = jnp.array([1, 1, 1])
        return TruncatedNormal(action_mean, action_std, lower, upper)

    def __call__(
        self,
        wm_out: DreamerWMOut,
        prev_action: Float[Array, "... ActionDim"],
        key: Array,
    ) -> DreamerActorCriticOut:
        z = jnp.concat([wm_out.hidden, wm_out.latent_sample, prev_action], axis=-1)
        z = self.backbone(z)

        action_dist = self.actor(z, key)
        value_pred = self.critic_head(z)  # idk if we need to transform this at all?

        return DreamerActorCriticOut(
            action_dist=action_dist,
            value_pred=value_pred,
        )

    @staticmethod
    def loss_fn(
        cfg: DreamerConfig,
        rewards: Float[Array, "Batch Time"],
        continues: Float[Array, "Batch Time"],
        values: Float[Array, "Batch Time+1"],
        action_dists: TruncatedNormal,  # Samples are shape [B T A]
    ) -> DreamerActorCriticLoss:
        # We need an extra value - one for each transition + the terminal value
        B, T = rewards.shape
        chex.assert_shape(continues, (B, T))
        chex.assert_shape(values, (B, T + 1))

        target = compute_td_lambda_returns(cfg, rewards, continues, values)

        target = jax.lax.stop_gradient(target)
        # Compute the actor and critic losses:
        #  - The critic regresses its value predictions towards the TD(lambda) values
        critic_loss = optax.losses.squared_error(target, values[..., :-1]).mean()
        #  - The actor *maximises* the return, so we return the negative value (which the optimiser will minimise) plus an entropy term
        actor_loss = -target.mean() - 1e-4 * action_dists.entropy().mean()

        return DreamerActorCriticLoss(
            critic_loss=critic_loss,
            actor_loss=actor_loss,
            target=target,
            loss=critic_loss + actor_loss,
        )


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
