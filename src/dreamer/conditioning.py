"""Compare observation histories at a shared open-loop forecast origin."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamer.buffer import policy_step
from dreamer.divergence import latent_kl, observe
from wm.utils.obs import crop_obs, normalise_obs

CONTEXTS = (1, 5, 20, 50)


def initialize_branches(wm, contexts, num_envs, seed):
    """Share random draws across histories, including subsequent imagination."""
    branches = [nnx.clone(wm, arrays=True) for _ in contexts]
    for branch in branches:
        nnx.reseed(branch, default=seed)
    carries = [branch.rssm.initialize_state(num_envs) for branch in branches]
    return branches, carries


@nnx.jit
def condition_step(branches, actor, carries, image, previous_action, resets, key):
    next_carries, states = [], []
    for branch, carry, reset in zip(branches, carries, resets, strict=True):
        carry, _, state, _ = policy_step(
            branch, actor, carry, image, previous_action, reset, key,
        )
        next_carries.append(carry)
        states.append(state.replace(posterior=None))
    return next_carries, states


def _imagine_step(branches, actor, states, key):
    next_states = []
    for branch, state in zip(branches, states, strict=True):
        action = actor(state.joint_latent).sample(seed=key)
        next_states.append(branch.imagine_step(state, action))
    return next_states


imagine_step = nnx.jit(_imagine_step)


@nnx.jit
def forecast_step(wm, actor, branches, carry, states, image,
                  previous_action, action_key, imagined_action_key):
    carry, action, posterior, _ = policy_step(
        wm, actor, carry, image, previous_action,
        jnp.zeros(image.shape[0], dtype=jnp.bool_), action_key,
    )
    states = _imagine_step(branches, actor, states, imagined_action_key)
    scores = jnp.stack([
        latent_kl(posterior.posterior, state.prior) for state in states
    ], axis=-1)
    return carry, action, states, scores


def context_resets(step, contexts, num_envs):
    """Erase recurrent history at the start of each trailing observation window."""
    starts = max(contexts) - np.asarray(contexts)
    return jnp.asarray(np.broadcast_to(
        ((step == 0) | (step == starts))[:, None], (len(contexts), num_envs),
    ))


def condition_recording(models, images, actions, *, contexts=CONTEXTS, seed=500_000):
    """Initialize histories from cached pixels/actions, ending at one common frame.

    Earlier observations are erased by the window-start reset. Each branch retains
    only its last N observations and the actions between them. All branches consume
    the same random draws at each time, reducing sampling noise in the comparison.
    """
    if min(contexts) < 1 or len(images) < max(contexts):
        raise ValueError("The recording must contain the largest positive context")
    branches, carries = initialize_branches(models["world_model"], contexts, 1, seed)
    key = jax.random.key(seed)
    for step in range(max(contexts)):
        key, action_key = jax.random.split(key)
        previous = np.zeros((1, models["world_model"].cfg.action_dim), np.float32)
        if step:
            previous[0] = actions[step - 1]
        carries, states = condition_step(
            branches, models["actor"], carries,
            jnp.asarray(normalise_obs(images[step : step + 1]) - 0.5),
            jnp.asarray(previous), context_resets(step, contexts, 1), action_key,
        )
    return branches, states


def collect_conditioning(models, env_id, *, contexts=CONTEXTS, num_episodes=64,
                         num_envs=8, max_steps=1000, seed=500_000):
    """Compare uninterrupted actor-driven predictions against the real posterior."""
    if min(contexts) < 1 or max(contexts) > max_steps:
        raise ValueError("Use positive contexts shorter than the rollout")
    wm, actor = models["world_model"], models["actor"]
    num_envs = min(num_envs, num_episodes)
    envs = gym.make_vec(env_id, num_envs=num_envs, vectorization_mode="sync")
    curves = np.full((num_episodes, max_steps + 1, len(contexts)), np.nan, np.float32)
    lengths = np.zeros(num_episodes, np.int32)
    returns = np.zeros(num_episodes, np.float64)
    try:
        for start in range(0, num_episodes, num_envs):
            count = min(num_envs, num_episodes - start)
            batch_seed = seed + start
            observation, _ = envs.reset(seed=[batch_seed + i for i in range(num_envs)])
            nnx.reseed(wm, default=batch_seed)
            carry = wm.rssm.initialize_state(num_envs)
            branches, branch_carries = initialize_branches(wm, contexts, num_envs, batch_seed)
            previous_action = jnp.zeros((num_envs, wm.cfg.action_dim), dtype=jnp.float32)
            key = jax.random.key(batch_seed)
            imagined_key = jax.random.key(batch_seed + 3_000_000)
            active = np.arange(num_envs) < count
            valid_observation = active.copy()
            batch_lengths = np.zeros(num_envs, np.int32)
            batch_returns = np.zeros(num_envs, np.float64)
            for step in range(max_steps + 1):
                image = jnp.asarray(normalise_obs(crop_obs(observation)) - 0.5)
                key, action_key = jax.random.split(key)
                if step < max(contexts):
                    carry, action, _, _ = observe(
                        wm, actor, carry, image, previous_action,
                        jnp.full(num_envs, step == 0), action_key,
                    )
                    branch_carries, states = condition_step(
                        branches, actor, branch_carries, image, previous_action,
                        context_resets(step, contexts, num_envs), action_key,
                    )
                else:
                    imagined_key, imagined_action_key = jax.random.split(imagined_key)
                    carry, action, states, scores = forecast_step(
                        wm, actor, branches, carry, states, image,
                        previous_action, action_key, imagined_action_key,
                    )
                    curves[start : start + count, step] = np.where(
                        valid_observation[:count, None], np.asarray(scores)[:count], np.nan,
                    )
                if step == max_steps or not active.any():
                    break
                observation, reward, terminated, truncated, _ = envs.step(np.asarray(action))
                batch_returns += np.where(active, reward, 0.0)
                batch_lengths += active
                valid_observation = active.copy()
                active &= ~(terminated | truncated)
                previous_action = action
            lengths[start : start + count] = batch_lengths[:count]
            returns[start : start + count] = batch_returns[:count]
            print(f"Scored {start + count}/{num_episodes} episodes for {contexts} frames", flush=True)
    finally:
        envs.close()
    return curves, lengths, returns
