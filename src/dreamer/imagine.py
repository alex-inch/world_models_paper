from typing import Any

import chex
import jax
import jax.numpy as jnp
from einops import pack
from flax import nnx
from jaxtyping import Array, Float, Shaped

from dreamer.actor_critic import Actor, Critic
from dreamer.truncated_normal import TruncatedNormal
from dreamer.wm import DreamerWM
from dreamer.wm.rssm import RSSMOut


@chex.dataclass
class ImaginationTrajectory:
    """Contains all the associated values at a given time step: (s_t, a_t, v_t, p(a_t|s_t)"""

    state: RSSMOut
    action: Any
    value: Array
    action_dist: TruncatedNormal


@nnx.scan(in_axes=(None, None, None, nnx.Carry, 0), out_axes=(nnx.Carry, 1))
def imagination_step(
    wm: DreamerWM, actor: Actor, critic: Critic, state: RSSMOut, key: Array
) -> tuple[RSSMOut, ImaginationTrajectory]:

    features = state.joint_latent
    action_dist = actor(features)
    action = action_dist.sample(seed=key)

    next_state = wm.imagine_step(state, action)

    output = ImaginationTrajectory(
        state=state,
        action=action,
        value=critic(features),
        action_dist=action_dist,  # would be less overhead to just send the entropy
    )

    return next_state, output


def run_imagination_rollout(
    wm: DreamerWM,
    actor: Actor,
    critic: Critic,
    initial_state: RSSMOut,
    horizon: int,
    key: Shaped[Array, ""],
) -> tuple[ImaginationTrajectory, Array, Array]:
    assert initial_state.posterior is None

    # TODO(Alex): check how batching works here
    keys = jax.random.split(key, horizon)

    # Run imagination
    final_carry, trajectory = imagination_step(wm, actor, critic, initial_state, keys)

    # Our trajectory has N actions, but N+1 states - we need to add the final state onto the trajectory
    #  so we get (s0, s1, ..., sT)
    final_state = final_carry
    final_value_pred = critic(final_state.joint_latent)
    trajectory.state = jax.tree.map(
        lambda sequence, final: jnp.concatenate(
            [sequence, final[:, None, ...]], axis=1
        ),
        trajectory.state,
        final_state,
    )
    trajectory.value, _ = pack([trajectory.value, final_value_pred], "B *")  # [B T+1]

    # Predict the reward and continue values over the trajectory (note we skip the initial state)
    # rewards: [B T R], cont: [B T]
    joint_latent: Float[Array, "B T Z+H"] = trajectory.state.joint_latent[:, 1:]
    reward_logits, cont_logits = wm._predict_reward_and_continue(joint_latent)
    # convert out of categorical prediction in symlog space to scalar prediction
    rewards = wm.decode_reward_logits(reward_logits)
    conts = jax.nn.sigmoid(cont_logits)

    return trajectory, rewards, conts
