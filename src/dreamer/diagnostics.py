"""Evaluation diagnostics for policy behaviour, reward prediction and imagination."""

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import nnx
from jax import Array
from jaxtyping import Float
from loguru import logger
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw

import wandb
from dreamer.actor_critic import Actor, compute_td_lambda_returns
from dreamer.imagine import run_imagination_rollout
from dreamer.wm import DreamerBatch
from dreamer.wm.rssm import RSSMOut


def policy_metrics(buffer, actor, predictions):
    """Summarise collected actions, excluding the dummy action and episode padding."""
    valid = np.cumprod(np.asarray(buffer.conts[:, :-1]), axis=1).astype(bool)
    actions = np.asarray(buffer.acts[:, 1:])[valid]

    # Entropy is evaluated on replayed posterior states. The actions above are
    # the actual environment actions, rather than new samples from this replay.
    features = jnp.concatenate([predictions.hidden, predictions.latent_sample], axis=-1)
    distribution = actor(features[:, :-1])
    entropy = np.asarray(distribution.distribution.entropy())[valid[:1]]
    metrics = {"eval/policy_entropy": float(entropy.sum(axis=-1).mean())}
    lower, upper = np.array([-1, 0, 0]), np.ones(3)
    relative_actions = (actions - lower) / (upper - lower)
    near_bounds = (relative_actions < 0.05) | (relative_actions > 0.95)
    for index, name in enumerate(("steering", "throttle", "brake")):
        prefix = f"eval/actions/{name}"
        metrics.update({
            f"{prefix}_mean": float(actions[:, index].mean()),
            f"{prefix}_std": float(actions[:, index].std()),
            f"{prefix}_near_bound_fraction": float(near_bounds[:, index].mean()),
            f"{prefix}_entropy": float(entropy[:, index].mean()),
            f"{prefix}_histogram": wandb.Histogram(actions[:, index]),
        })
    return metrics


@nnx.scan(in_axes=(None, nnx.Carry, 1), out_axes=(nnx.Carry, 1))
def _predict_action(wm, state, action):
    state = wm.imagine_step(state, action)
    return state, state


@nnx.jit(static_argnames="horizon")
def _reference_diagnostics(wm, actor, critic, target_critic, batch, starts, horizon):
    states = wm.run_rssm(batch)
    reward_logits, _ = wm._predict_reward_and_continue(states.joint_latent)

    # Every branch starts from an observed posterior sample with its full history.
    initial_states = jax.tree.map(
        lambda x: x[0, starts], states.replace(posterior=None)
    )
    # prev_action[t + 1] is the action that took state t to state t + 1.
    times = starts[:, None] + jnp.arange(1, horizon + 1)[None]
    recorded_actions = batch.prev_action[0, times]
    _, predicted_states = _predict_action(wm, initial_states, recorded_actions)
    predicted_logits, _ = wm._predict_reward_and_continue(predicted_states.joint_latent)

    trajectory, rewards, continues = run_imagination_rollout(
        wm, actor, target_critic, initial_states, horizon, jax.random.key(500_001)
    )
    _, targets, weights = Actor.loss_fn(
        wm.cfg, rewards, continues, trajectory.value, trajectory.action_dist
    )
    # Keeping lambda unchanged makes these two signed contributions add up to
    # the actual lambda target, including all intermediate critic bootstraps.
    reward_return = compute_td_lambda_returns(
        wm.cfg, rewards, continues, jnp.zeros_like(trajectory.value)
    )
    features = trajectory.state.joint_latent[:, :-1]
    prior_log_probs = jax.nn.log_softmax(states.prior, axis=-1)
    posterior_log_probs = jax.nn.log_softmax(states.posterior, axis=-1)
    posterior_probs = jnp.exp(posterior_log_probs)
    prior_entropy = -(jnp.exp(prior_log_probs) * prior_log_probs).sum(axis=(-1, -2))
    posterior_entropy = -(posterior_probs * posterior_log_probs).sum(axis=(-1, -2))
    kl = (posterior_probs * (posterior_log_probs - prior_log_probs)).sum(axis=(-1, -2))
    usage = states.latent_sample.reshape(
        *states.latent_sample.shape[:2],
        wm.cfg.num_latent_dists,
        wm.cfg.num_latent_classes,
    ).mean(axis=(0, 1))
    return {
        "posterior_reward": wm.decode_reward_logits(reward_logits)[:, 1:],
        "recorded_reward": batch.reward[:, 1:],
        "open_loop_reward": wm.decode_reward_logits(predicted_logits),
        "open_loop_actual_reward": batch.reward[0, times],
        "imagined_reward": rewards,
        "continues": continues,
        "weights": weights,
        "lambda_return": targets,
        "reward_return": reward_return,
        "bootstrap_return": targets - reward_return,
        "value": critic(features),
        "target_value": trajectory.value[:, :-1],
        "prior_entropy": prior_entropy.mean(),
        "posterior_entropy": posterior_entropy.mean(),
        "kl": kl.mean(),
        "category_usage": usage,
    }


def _line_plot(x, series, title, xname):
    return wandb.plot.line_series(
        xs=np.asarray(x).tolist(),
        ys=[np.asarray(values).tolist() for values in series.values()],
        keys=list(series),
        title=title,
        xname=xname,
    )


def _category_plot(usage):
    figure = Figure(figsize=(7, 4), layout="constrained")
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots()
    image = axes.imshow(usage, aspect="auto", vmin=0, vmax=1)
    axes.set(
        xlabel="Category",
        ylabel="Latent distribution",
        title="Posterior sampled category usage",
    )
    figure.colorbar(image, ax=axes, label="Fraction of recorded states")
    canvas.draw()
    return wandb.Image(np.asarray(canvas.buffer_rgba()).copy())


def reference_metrics(wm, actor, critic, target_critic, batch, *, horizon=15):
    """Compare checkpoints on the same recording and deterministic sampling seeds."""
    # Evaluating diagnostics must not advance the training model's RNG streams.
    wm = nnx.clone(wm, arrays=True)
    nnx.reseed(wm, default=500_001)
    context = min(5, batch.image.shape[1] - 1)
    horizon = min(horizon, batch.image.shape[1] - context)
    starts = np.unique(
        np.linspace(context - 1, batch.image.shape[1] - horizon - 1, 8, dtype=int)
    )
    data = jax.device_get(
        _reference_diagnostics(
            wm, actor, critic, target_critic, batch, jnp.asarray(starts), horizon
        )
    )
    metrics = {}

    error = data["posterior_reward"] - data["recorded_reward"]
    metrics["eval/reward/posterior_mae"] = float(np.abs(error).mean())
    metrics["eval/reward/posterior_bias"] = float(error.mean())
    metrics["eval/reward/posterior_trace"] = _line_plot(
        np.arange(1, batch.image.shape[1]),
        {
            "Actual": data["recorded_reward"][0],
            "Posterior prediction": data["posterior_reward"][0],
        },
        "Reward prediction on the fixed recording",
        "Recorded step",
    )
    error = data["open_loop_reward"] - data["open_loop_actual_reward"]
    for step in (1, 5, 15):
        if step <= horizon:
            metrics[f"eval/reward/prior_mae_{step}"] = float(
                np.abs(error[:, step - 1]).mean()
            )
            metrics[f"eval/reward/prior_bias_{step}"] = float(error[:, step - 1].mean())
    steps = np.arange(1, horizon + 1)
    metrics["eval/reward/prior_trace"] = _line_plot(
        steps,
        {
            "Actual": data["open_loop_actual_reward"].mean(axis=0),
            "Prior prediction": data["open_loop_reward"].mean(axis=0),
        },
        "Reward prediction using recorded actions (mean over starts)",
        "Prediction horizon",
    )
    metrics["eval/reward/prior_error_by_horizon"] = _line_plot(
        steps,
        {"MAE": np.abs(error).mean(axis=0), "Signed bias": error.mean(axis=0)},
        "Reward error using recorded actions",
        "Prediction horizon",
    )

    for name in (
        "value",
        "target_value",
        "lambda_return",
        "reward_return",
        "bootstrap_return",
    ):
        metrics[f"eval/critic/{name}_mean"] = float(data[name].mean())
        metrics[f"eval/critic/{name}_std"] = float(data[name].std())
    critic_error = data["value"] - data["lambda_return"]
    metrics["eval/critic/target_rmse"] = float(np.sqrt((critic_error**2).mean()))
    metrics["eval/critic/target_bias"] = float(critic_error.mean())
    metrics["eval/critic/online_target_disagreement"] = float(
        np.abs(data["value"] - data["target_value"]).mean()
    )
    metrics["eval/critic/predictions"] = _line_plot(
        steps,
        {
            "Online critic": data["value"].mean(axis=0),
            "Target critic": data["target_value"].mean(axis=0),
            "Lambda target": data["lambda_return"].mean(axis=0),
        },
        "Critic predictions on imagined states (mean over starts)",
        "Imagined transition",
    )
    metrics["eval/critic/return_components"] = _line_plot(
        steps,
        {
            "Reward contribution": data["reward_return"].mean(axis=0),
            "Bootstrap contribution": data["bootstrap_return"].mean(axis=0),
            "Lambda target": data["lambda_return"].mean(axis=0),
        },
        "Signed contributions to the lambda return",
        "Imagined transition",
    )

    metrics["eval/imagination/reward_mean"] = float(data["imagined_reward"].mean())
    metrics["eval/imagination/continue_mean"] = float(data["continues"].mean())
    metrics["eval/imagination/weight_mean"] = float(data["weights"].mean())
    metrics["eval/imagination/effective_horizon"] = float(
        data["weights"].sum(axis=1).mean()
    )
    metrics["eval/imagination/continuation_and_weights"] = _line_plot(
        steps,
        {
            "Continuation probability": data["continues"].mean(axis=0),
            "Loss weight": data["weights"].mean(axis=0),
        },
        "Imagined continuation and loss weights",
        "Imagined transition",
    )
    for name in ("prior_entropy", "posterior_entropy", "kl"):
        metrics[f"eval/latents/{name}"] = float(data[name])
    metrics["eval/latents/entropy_max"] = float(
        wm.cfg.num_latent_dists * np.log(wm.cfg.num_latent_classes)
    )
    metrics["eval/latents/category_fraction_used"] = float(
        (data["category_usage"] > 0).mean()
    )
    metrics["eval/latents/category_usage"] = _category_plot(data["category_usage"])
    return metrics


def format_rollout_gif_data(observations: Float[Array, "T H W C"], fps: int = 50):
    "Get a jax array of frames into shape for wandb upload"
    # OG is 50fps - we can downsample a bit to save space
    frame_skip = 50 // fps
    if 50 % fps != 0:
        logger.warning(
            f"Provided eval GIF FPS {fps} does not neatly divide env fps {50}, using fps {int(50 / frame_skip)}"
        )
    frames = np.asarray(observations[::frame_skip])
    frames_rescaled = 255 * (frames + 0.5)
    frames_rescaled = np.clip(frames_rescaled, 0, 255).astype(np.uint8)
    return rearrange(frames_rescaled, "T H W C -> T C H W")


def format_eval_video(rollout, reconstruction, imagination, imagination_start, fps):
    # Align imagination with its starting observation before downsampling the video.
    imagined_rollout = np.full(rollout.shape, -0.5, dtype=np.float32)
    imagination_end = min(imagination_start + len(imagination), len(rollout))
    imagined_rollout[imagination_start:imagination_end] = np.asarray(
        imagination[: imagination_end - imagination_start]
    )
    comparison = np.concatenate([rollout, reconstruction, imagined_rollout], axis=2)
    frames = format_rollout_gif_data(comparison, fps)

    panel_width = rollout.shape[2]
    header_height = 20
    header = Image.new("RGB", (3 * panel_width, header_height))
    draw = ImageDraw.Draw(header)
    for panel, label in enumerate(["Policy", "Reconstruction", "Imagination"]):
        draw.text((panel * panel_width + 4, 3), label, fill="white")
    header = rearrange(np.asarray(header), "H W C -> C H W")

    labelled_frames = np.zeros(
        (len(frames), 3, frames.shape[2] + header_height, frames.shape[3]),
        dtype=np.uint8,
    )
    labelled_frames[:, :, :header_height] = header
    labelled_frames[:, :, header_height:] = frames
    return labelled_frames


def evaluate_rollouts(
    buffer,
    world_model,
    actor,
    critic,
    target_critic,
    imagination_key,
    *,
    fps=10,
    imagination_horizon=100,
    fixed_batch=None,
):
    """Build evaluation metrics and videos, retaining one fixed reference recording."""
    rollout = buffer.obs[0]

    # Construct a buffer with only one episode. Note we slide to retain the
    # spurious batch dim ~ [B T ...]
    images = buffer.obs[:1]
    is_first = jnp.zeros(images.shape[:2], dtype=jnp.bool_)
    is_first = is_first.at[:, 0].set(True)

    batch = DreamerBatch(
        image=images,
        prev_action=buffer.acts[:1],
        reward=buffer.rewards[:1],
        cont=buffer.conts[:1],
        is_first=is_first,
    )

    if fixed_batch is None:
        # Keep the first evaluation recording fixed across checkpoints. Stop at
        # the episode boundary, and cap its length to keep diagnostics inexpensive.
        episode_ends = np.flatnonzero(np.asarray(batch.cont[0]) == 0)
        episode_length = (
            int(episode_ends[0]) + 1 if len(episode_ends) else images.shape[1]
        )
        reference_length = min(256, episode_length)
        fixed_batch = jax.tree.map(lambda x: x[:, :reference_length].copy(), batch)

    predictions = world_model(batch)
    reconstruction = predictions.image[0]

    # Start from a posterior state after five observations, then stop using images.
    context_steps = 5
    start = context_steps - 1
    initial_state = RSSMOut(
        hidden=predictions.hidden[:, start],
        latent_sample=predictions.latent_sample[:, start],
        prior=predictions.latent_prior[:, start],
    )
    trajectory, _, _ = run_imagination_rollout(
        world_model,
        actor,
        critic,
        initial_state,
        horizon=imagination_horizon,
        key=imagination_key,
    )
    imagined_frames = world_model.decoder(trajectory.state.joint_latent)[0]
    rollout_video = format_eval_video(
        rollout, reconstruction, imagined_frames, start, fps
    )

    rewards = buffer.rewards  # [T H]
    returns = rewards.sum(axis=1)

    metrics = {
        "eval/return_mean": float(returns.mean()),
        "eval/return_median": float(np.median(returns)),
        "eval/return_std": float(returns.std()),
        "eval/returns": wandb.Histogram(returns, num_bins=16),  # type: ignore
        "eval/policy": wandb.Video(
            rollout_video,
            caption=(
                f"Mean Return: {float(returns.mean()):.0f}. "
                f"Policy | Reconstruction | Imagination. "
                f"Imagination starts after {context_steps} observations and runs for "
                f"{imagination_horizon} steps; its panel is blank outside that window."
            ),
            fps=fps,
            format="gif",
        ),
    }
    metrics.update(policy_metrics(buffer, actor, predictions))
    metrics.update(
        reference_metrics(world_model, actor, critic, target_critic, fixed_batch)
    )
    return metrics, fixed_batch
