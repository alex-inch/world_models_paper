from typing import Any

import chex
import jax
import jax.numpy as jnp
from einops import pack
from flax import nnx
from jaxtyping import Array, Float, Shaped

from dreamer.actor_critic import ActorCritic
from dreamer.truncated_normal import TruncatedNormal
from dreamer.wm import DreamerWM
from dreamer.wm.rssm import RSSMOut

type ImaginationCarry = tuple[RSSMOut, Array]


@chex.dataclass
class ImaginationTrajectory:
    """Contains all the associated values at a given time step: (s_t, a_t, v_t, p(a_t|s_t)"""

    state: RSSMOut
    action: Any
    value: Array
    action_dist: TruncatedNormal


@nnx.scan(in_axes=(None, None, nnx.Carry, 0), out_axes=(nnx.Carry, 1))
def imagination_step(
    wm: DreamerWM, actor_critic: ActorCritic, carry: ImaginationCarry, key
) -> tuple[ImaginationCarry, ImaginationTrajectory]:
    state, prev_action = carry
    policy_key, action_sample_key = jax.random.split(key)

    prediction = actor_critic(state, prev_action, policy_key)
    action = prediction.action_dist.sample(seed=action_sample_key)

    next_state = wm.imagine_step(state, action)

    output = ImaginationTrajectory(
        state=state,
        action=action,
        value=prediction.value_pred,
        action_dist=prediction.action_dist,  # would be less overhead to just send the entropy
    )

    next_carry = (next_state, action)
    return next_carry, output


def imagination_rollout(
    wm: DreamerWM,
    actor_critic: ActorCritic,
    initial_state: RSSMOut,
    prev_action: Shaped[Array, "B A"],
    key: Shaped[Array, ""],
    horizon: int,
) -> tuple[ImaginationTrajectory, Array, Array]:
    assert initial_state.posterior is None

    # TODO(Alex): check how batching works here
    bootstrap_key, key = jax.random.split(key)
    keys = jax.random.split(key, horizon)

    # Run imagination
    init_carry = (initial_state, prev_action)
    final_carry, trajectory = imagination_step(wm, actor_critic, init_carry, keys)

    # Our trajectory has N actions, but N+1 states - we need to add the final state onto the trajectory
    #  so we get (s0, s1, ..., sT)
    final_state, last_action = final_carry
    final_value_pred = actor_critic(final_state, last_action, bootstrap_key).value_pred
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
