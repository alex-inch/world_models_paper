"""Record the states used by the policy and inspect a frozen Dreamer checkpoint."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamer.actor_critic import Actor, Critic, compute_td_lambda_returns
from dreamer.buffer import policy_step
from dreamer.imagine import run_imagination_rollout
from dreamer.wm import DreamerBatch, DreamerWM, DreamerWMOut
from dreamer.wm.losses import nll
from dreamer.wm.rssm import RSSMOut
from wm.utils.obs import crop_obs, normalise_obs

record_step = nnx.jit(policy_step)


def record_episode(models, envs, *, seed: int, max_steps: int = 1000):
    """Record decision states s_t, actions a_t, and their following observations/rewards."""
    wm, actor = models["world_model"], models["actor"]
    observation, _ = envs.reset(seed=seed)
    carry = wm.rssm.initialize_state(1)
    prev_action = jnp.zeros((1, wm.cfg.action_dim), dtype=jnp.float32)
    previous_reward = 0.0
    key = jax.random.key(seed)
    history = {}

    for step in range(max_steps):
        pixels = crop_obs(observation)
        image = normalise_obs(pixels) - 0.5
        is_first = jnp.asarray([step == 0])
        action_key, key = jax.random.split(key)
        carry, action, state, distribution = record_step(
            wm, actor, carry, jnp.asarray(image), prev_action, is_first, action_key
        )
        next_observation, reward, terminated, truncated, _ = envs.step(
            np.asarray(action)
        )

        # Keep the actual sampled latent and distribution. Re-running the RSSM later
        # could sample a different latent, which would change the policy's answer.
        row = {
            "images": pixels[0],
            "next_images": crop_obs(next_observation)[0],
            "prev_actions": prev_action[0],
            "rewards": previous_reward,
            "continues": 1.0,
            "is_first": step == 0,
            "actions": action[0],
            "next_rewards": reward[0],
            "terminated": terminated[0],
            "truncated": truncated[0],
            "hidden": state.hidden[0],
            "latent_sample": state.latent_sample[0],
            "prior_logits": state.prior[0],
            "posterior_logits": state.posterior[0],
            "action_loc": distribution.loc[0],
            "action_scale": distribution.scale[0],
            "action_lower": distribution.lower[0],
            "action_upper": distribution.upper[0],
            "action_mean": distribution.mean()[0],
            "action_entropy": distribution.entropy()[0],
        }
        for name, value in jax.device_get(row).items():
            history.setdefault(name, []).append(value)

        observation = next_observation
        prev_action = action
        previous_reward = float(reward[0])
        if terminated[0] or truncated[0]:
            break

    return {name: np.asarray(values) for name, values in history.items()}


def _state_from_arrays(arrays):
    return RSSMOut(
        hidden=jnp.asarray(arrays["hidden"])[None],
        latent_sample=jnp.asarray(arrays["latent_sample"])[None],
        prior=jnp.asarray(arrays["prior_logits"])[None],
        posterior=jnp.asarray(arrays["posterior_logits"])[None],
    )


def _frame_losses(predictions, batch, config):
    def loss_at_frame(prediction, observation):
        prediction = jax.tree.map(lambda x: x[:, None], prediction)
        observation = jax.tree.map(lambda x: x[:, None], observation)
        return DreamerWM.loss_fn(prediction, observation, config)

    return jax.vmap(loss_at_frame, in_axes=(1, 1))(predictions, batch)


def frames_to_uint8(frames):
    return np.clip(255 * (np.asarray(frames) + 0.5), 0, 255).astype(np.uint8)


@nnx.jit
def _analyse_chunk(wm, critic, target_critic, arrays):
    state = _state_from_arrays(arrays)
    features = state.joint_latent
    reconstruction = wm.decoder(features)
    reward, continuation = wm._predict_reward_and_continue(features)
    predictions = DreamerWMOut(
        image=reconstruction,
        reward=reward,
        cont=continuation,
        latent_prior=state.prior,
        latent_posterior=state.posterior,
        latent_sample=state.latent_sample,
        hidden=state.hidden,
    )
    batch = DreamerBatch(
        image=arrays["images"][None].astype(jnp.float32) / 255 - 0.5,
        prev_action=arrays["prev_actions"][None],
        reward=arrays["rewards"][None],
        cont=arrays["continues"][None],
        is_first=arrays["is_first"][None],
    )
    losses = _frame_losses(predictions, batch, wm.cfg)
    prior_log_probs = jax.nn.log_softmax(state.prior)
    posterior_log_probs = jax.nn.log_softmax(state.posterior)
    prior_entropy = -(jnp.exp(prior_log_probs) * prior_log_probs).sum(axis=(-1, -2))
    posterior_entropy = -(jnp.exp(posterior_log_probs) * posterior_log_probs).sum(
        axis=(-1, -2)
    )
    return {
        "reconstruction": reconstruction[0],
        "predicted_reward": wm.decode_reward_logits(reward)[0],
        "predicted_continue": jax.nn.sigmoid(continuation)[0],
        "value": critic(features)[0],
        "target_value": target_critic(features)[0],
        "image_loss": losses.image_loss,
        "reward_loss": losses.reward_loss,
        "continue_loss": losses.continue_loss,
        "kl_loss": losses.kl_loss,
        "weighted_kl_loss": wm.cfg.kl_beta * losses.kl_loss,
        "world_model_loss": losses.image_loss
        + losses.reward_loss
        + losses.continue_loss
        + wm.cfg.kl_beta * losses.kl_loss,
        "prior_entropy": prior_entropy[0],
        "posterior_entropy": posterior_entropy[0],
    }


@jax.jit
def _recorded_returns(rewards, discounts):
    def backward(value, inputs):
        reward, discount = inputs
        value = reward + discount * value
        return value, value

    _, returns = jax.lax.scan(
        backward, jnp.asarray(0.0), (rewards, discounts), reverse=True
    )
    return returns


def analyse_episode(models, episode, *, chunk_size: int = 64):
    """Evaluate the recorded states without sampling new posterior latents."""
    fields = (
        "images",
        "prev_actions",
        "rewards",
        "continues",
        "is_first",
        "hidden",
        "latent_sample",
        "prior_logits",
        "posterior_logits",
    )
    diagnostics = {}
    for start in range(0, len(episode["images"]), chunk_size):
        arrays = {
            name: jnp.asarray(episode[name][start : start + chunk_size])
            for name in fields
        }
        values = _analyse_chunk(
            models["world_model"], models["critic"], models["target_critic"], arrays
        )
        for name, value in jax.device_get(values).items():
            if name == "reconstruction":
                value = frames_to_uint8(value)
            diagnostics.setdefault(name, []).append(value)

    result = dict(episode)
    result.update({
        name: np.concatenate(values) for name, values in diagnostics.items()
    })
    discounts = models["world_model"].cfg.discount * (~episode["terminated"])
    # This is the discounted reward in the recording, with a zero tail. It is
    # a useful reference, not an unbiased value target at a recording cutoff.
    result["recorded_return"] = np.asarray(
        _recorded_returns(
            jnp.asarray(episode["next_rewards"], dtype=jnp.float32),
            jnp.asarray(discounts, dtype=jnp.float32),
        )
    )
    return result


def save_recording(directory: str | Path, episodes, metadata: dict):
    directory = Path(directory)
    directory.mkdir(parents=True)
    for index, episode in enumerate(episodes):
        np.savez_compressed(directory / f"episode_{index:03d}.npz", **episode)
    information = {**metadata, "num_episodes": len(episodes)}
    (directory / "metadata.json").write_text(json.dumps(information, indent=2))


def load_recording(directory: str | Path):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    episodes = []
    for index in range(metadata["num_episodes"]):
        with np.load(
            directory / f"episode_{index:03d}.npz", allow_pickle=False
        ) as archive:
            episodes.append(dict(archive))
    return episodes, metadata


def _branch_world_model(models, seed):
    # Branching must not change the loaded checkpoint's RNG state.
    wm = nnx.clone(models["world_model"], arrays=True)
    nnx.reseed(wm, default=seed)
    return wm


def imagine_from_frame(
    models, episode, frame: int, *, horizon: int = 15, seed: int = 0
):
    wm = _branch_world_model(models, seed)
    actor, critic = models["actor"], models["critic"]
    initial = _state_from_arrays({
        name: values[frame] for name, values in episode.items()
    })
    initial = initial.replace(posterior=None)
    trajectory, rewards, continues = run_imagination_rollout(
        wm, actor, models["target_critic"], initial, horizon, jax.random.key(seed)
    )
    features = trajectory.state.joint_latent
    losses, targets, weights = Actor.loss_fn(
        wm.cfg,
        rewards,
        continues,
        trajectory.value,
        trajectory.action_dist,
        per_step=True,
    )
    # Lambda returns are linear in rewards and values. Keep the same lambda
    # when splitting them, so the two contributions add up to the actual target.
    reward_return = compute_td_lambda_returns(
        wm.cfg, rewards, continues, jnp.zeros_like(trajectory.value)
    )
    bootstrap_return = targets - reward_return
    values = critic(features[:, :-1])
    critic_losses = Critic.loss_fn(values, targets, weights, per_step=True)
    frames = frames_to_uint8(wm.decoder(features)[0])
    distribution = trajectory.action_dist
    result = {
        "images": frames[:-1],
        "next_images": frames[1:],
        "reference_images": episode["images"][frame : frame + horizon],
        "actions": trajectory.action[0],
        "action_loc": distribution.loc[0],
        "action_scale": distribution.scale[0],
        "action_lower": distribution.lower[0],
        "action_upper": distribution.upper[0],
        "action_mean": distribution.mean()[0],
        "action_entropy": distribution.entropy()[0],
        "predicted_reward": rewards[0],
        "predicted_continue": continues[0],
        "value": values[0],
        "target_value": trajectory.value[0, :-1],
        "lambda_return": targets[0],
        "reward_return": reward_return[0],
        "bootstrap_return": bootstrap_return[0],
        "weight": weights[0],
        "actor_reward_loss": losses.reward_loss[0],
        "actor_entropy_loss": losses.entropy_loss[0],
        "actor_loss": losses.loss[0],
        "critic_loss": critic_losses[0],
        "prior_logits": trajectory.state.prior[0, :-1],
        "latent_sample": trajectory.state.latent_sample[0, :-1],
    }
    return {name: np.asarray(value) for name, value in jax.device_get(result).items()}


@nnx.scan(in_axes=(None, nnx.Carry, 1), out_axes=(nnx.Carry, 1))
def _predict_step(wm, state, action):
    state = wm.imagine_step(state, action)
    return state, state


def predict_recorded_actions(
    models, episode, frame: int, *, horizon: int = 15, seed: int = 0
):
    wm = _branch_world_model(models, seed)
    horizon = min(horizon, len(episode["actions"]) - frame)
    initial = _state_from_arrays({
        name: values[frame] for name, values in episode.items()
    })
    initial = initial.replace(posterior=None)
    actions = jnp.asarray(episode["actions"][frame : frame + horizon])[None]
    _, states = _predict_step(wm, initial, actions)
    predicted = wm.decoder(states.joint_latent)[0]
    actual = episode["next_images"][frame : frame + horizon]
    image_losses = jax.vmap(
        lambda prediction, target: nll(prediction[None, None], target[None, None])
    )(predicted, jnp.asarray(actual, dtype=jnp.float32) / 255 - 0.5)
    reward, continuation = wm._predict_reward_and_continue(states.joint_latent)
    return {
        "images": frames_to_uint8(predicted),
        "reference_images": actual,
        "predicted_reward": np.asarray(wm.decode_reward_logits(reward)[0]),
        "rewards": episode["next_rewards"][frame : frame + horizon],
        "predicted_continue": np.asarray(jax.nn.sigmoid(continuation)[0]),
        "image_loss": np.asarray(image_losses),
    }
