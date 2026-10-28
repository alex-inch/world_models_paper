"""Posterior-to-prior drift and whole-episode bootstrap confidence intervals."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from rliable import library

from dreamer.buffer import policy_step
from wm.utils.obs import crop_obs, normalise_obs

CONDITIONS = ("posterior_corrected", "open_loop_matched_actions", "open_loop_actor")
observe = nnx.jit(policy_step)


@jax.jit
def latent_kl(posterior_logits, prior_logits):
    """KL(q || p), summed over classes and averaged over categorical factors.

    Inputs end in (factors, classes); outputs are in nats per factor. Log-softmax
    keeps unlikely predictions finite without adding an arbitrary probability floor.
    """
    log_q = jax.nn.log_softmax(posterior_logits, axis=-1)
    log_p = jax.nn.log_softmax(prior_logits, axis=-1)
    per_factor = jnp.sum(jnp.exp(log_q) * (log_q - log_p), axis=-1)
    return jnp.maximum(per_factor.mean(axis=-1), 0.0)


@nnx.jit
def _advance(wm, actor, matched_wm, actor_wm, carry, matched, imagined,
             image, previous_action, action_key, imagined_action_key):
    carry, action, posterior, _ = policy_step(
        wm, actor, carry, image, previous_action,
        jnp.zeros(image.shape[0], dtype=jnp.bool_), action_key,
    )
    matched = matched_wm.imagine_step(matched, previous_action)
    imagined_action = actor(imagined.joint_latent).sample(seed=imagined_action_key)
    imagined = actor_wm.imagine_step(imagined, imagined_action)
    scores = jnp.stack([
        latent_kl(posterior.posterior, posterior.prior),
        latent_kl(posterior.posterior, matched.prior),
        latent_kl(posterior.posterior, imagined.prior),
    ], axis=-1)
    return carry, action, matched, imagined, scores


def collect_divergence(models, env_id, *, num_episodes=64, num_envs=8,
                       max_steps=1000, context=5, seed=500_000):
    """Score real episodes without resetting either open-loop branch after context."""
    wm, actor = models["world_model"], models["actor"]
    matched_wm = nnx.clone(wm, arrays=True)
    actor_wm = nnx.clone(wm, arrays=True)
    num_envs = min(num_envs, num_episodes)
    envs = gym.make_vec(env_id, num_envs=num_envs, vectorization_mode="sync")
    curves = np.full((num_episodes, max_steps + 1, len(CONDITIONS)), np.nan, np.float32)
    lengths = np.zeros(num_episodes, np.int32)
    returns = np.zeros(num_episodes, np.float64)
    try:
        for start in range(0, num_episodes, num_envs):
            count = min(num_envs, num_episodes - start)
            batch_seed = seed + start
            observation, _ = envs.reset(seed=[batch_seed + i for i in range(num_envs)])
            nnx.reseed(wm, default=batch_seed)
            nnx.reseed(matched_wm, default=batch_seed + 1_000_000)
            nnx.reseed(actor_wm, default=batch_seed + 2_000_000)
            carry = wm.rssm.initialize_state(num_envs)
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
                if step < context:
                    carry, action, posterior, _ = observe(
                        wm, actor, carry, image, previous_action,
                        jnp.full(num_envs, step == 0), action_key,
                    )
                    matched = posterior.replace(posterior=None)
                    imagined = posterior.replace(posterior=None)
                    scores = np.full((num_envs, len(CONDITIONS)), np.nan, np.float32)
                    scores[:, 0] = np.asarray(latent_kl(posterior.posterior, posterior.prior))
                else:
                    imagined_key, imagined_action_key = jax.random.split(imagined_key)
                    carry, action, matched, imagined, scores = _advance(
                        wm, actor, matched_wm, actor_wm, carry, matched, imagined,
                        image, previous_action, action_key, imagined_action_key,
                    )
                    scores = np.asarray(scores)
                curves[start : start + count, step] = np.where(
                    valid_observation[:count, None], scores[:count], np.nan
                )
                if step == max_steps or not active.any():
                    break
                observation, reward, terminated, truncated, _ = envs.step(np.asarray(action))
                batch_returns += np.where(active, reward, 0.0)
                batch_lengths += active
                # A terminal observation is still valid; ignore only subsequent resets.
                valid_observation = active.copy()
                active &= ~(terminated | truncated)
                previous_action = action
            lengths[start : start + count] = batch_lengths[:count]
            returns[start : start + count] = batch_returns[:count]
            print(f"Scored {start + count}/{num_episodes} episodes", flush=True)
    finally:
        envs.close()
    return curves, lengths, returns


def bootstrap_curve_mean(curves, *, reps=10_000, seed=500_000):
    """Pointwise 95% intervals, resampling complete episodes with shared weights.

    Time points and conditions stay paired within every bootstrap replicate.
    These are pointwise intervals, not a simultaneous confidence band.
    """
    curves = np.asarray(curves, dtype=np.float64)
    if len(curves) < 2 or not np.isfinite(curves).all() or reps < 1:
        raise ValueError("Expected at least two finite trajectories and positive reps")
    count = len(curves)
    # Bootstrap episode indices as one stratum, then apply them to entire curves.
    # Passing time columns directly would resample each time point independently.
    def aggregate(indices):
        return curves[indices[:, 0]].mean(axis=0).reshape(-1)

    random_state = np.random.get_state()
    try:
        np.random.seed(seed)
        points, intervals = library.get_interval_estimates(
            {"episodes": np.arange(count)[:, None]}, aggregate,
            reps=reps, confidence_interval_size=0.95, method="percentile",
        )
    finally:
        np.random.set_state(random_state)
    return (
        points["episodes"].reshape(curves.shape[1:]),
        intervals["episodes"].reshape(2, *curves.shape[1:]),
    )
