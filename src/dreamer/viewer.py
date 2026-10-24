"""An offline notebook viewer for recorded observations and imagined branches."""

import base64
import gzip
import html
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from IPython.display import HTML
from PIL import Image
from scipy.special import softmax
from scipy.stats import truncnorm


def _images_as_png(images):
    encoded = []
    for image in images:
        stream = BytesIO()
        Image.fromarray(image).save(stream, format="PNG")
        encoded.append(
            "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
        )
    return encoded


def _action_densities(episode):
    densities = []
    for action in range(3):
        lower = episode["action_lower"][:, action]
        upper = episode["action_upper"][:, action]
        location = episode["action_loc"][:, action]
        scale = episode["action_scale"][:, action]
        grid = np.linspace(lower[0], upper[0], 101)
        density = truncnorm.pdf(
            grid[None],
            ((lower - location) / scale)[:, None],
            ((upper - location) / scale)[:, None],
            loc=location[:, None],
            scale=scale[:, None],
        )
        densities.append({"grid": grid.tolist(), "density": density.tolist()})
    return densities


def _payload(episode, title, caption, panels, plots, metrics):
    image_panels = []
    for key, label in panels:
        image_panels.append({"label": label, "frames": _images_as_png(episode[key])})

    history_plots = []
    for plot_title, series in plots:
        lines = []
        for key, label in series:
            lines.append({"label": label, "values": episode[key].tolist()})
        history_plots.append({"title": plot_title, "series": lines})

    metric_values = []
    for key, label in metrics:
        metric_values.append({"label": label, "values": episode[key].tolist()})

    data = {
        "title": title,
        "caption": caption,
        "length": len(episode["images"]),
        "panels": image_panels,
        "plots": history_plots,
        "metrics": metric_values,
    }
    if "actions" in episode:
        data["actions"] = episode["actions"].tolist()
        data["action_mean"] = episode["action_mean"].tolist()
        data["densities"] = _action_densities(episode)
    if "prior_logits" in episode:
        data["prior"] = softmax(episode["prior_logits"], axis=-1).round(4).tolist()
        sample = episode["latent_sample"].reshape(episode["prior_logits"].shape)
        data["samples"] = sample.argmax(axis=-1).tolist()
    if "posterior_logits" in episode:
        data["posterior"] = (
            softmax(episode["posterior_logits"], axis=-1).round(4).tolist()
        )
    return data


def _render(payloads):
    template = Path(__file__).with_name("inspector.html").read_text()
    # Full episodes contain many images and categorical probabilities. Compress
    # the payload so notebook output stays small; the browser decompresses it.
    data = json.dumps(payloads).encode()
    compressed = base64.b64encode(gzip.compress(data)).decode()
    document = template.replace("__RECORDINGS__", compressed)
    title = html.escape(payloads[0]["title"], quote=True)
    # Each view has its own document, so several viewers can coexist in a notebook.
    return HTML(
        f'<div aria-label="{title}"><iframe sandbox="allow-scripts" '
        'style="width:100%;height:1800px;border:0" '
        f'srcdoc="{html.escape(document, quote=True)}"></iframe></div>'
    )


def view_recording(episodes, metadata):
    plots = [
        (
            "Reward attached to observation",
            [("rewards", "Observed r_t"), ("predicted_reward", "Predicted r_t")],
        ),
        (
            "Value estimates",
            [
                ("value", "Critic"),
                ("target_value", "Target critic"),
                ("recorded_return", "Recorded rewards, zero tail"),
            ],
        ),
        (
            "Image and KL losses",
            [("image_loss", "Image"), ("weighted_kl_loss", "Weighted KL")],
        ),
        (
            "Reward and continuation losses",
            [("reward_loss", "Reward"), ("continue_loss", "Continue")],
        ),
        (
            "Latent entropy",
            [("prior_entropy", "Prior"), ("posterior_entropy", "Posterior")],
        ),
        (
            "Policy entropy and continuation",
            [
                ("action_entropy", "Action entropy"),
                ("predicted_continue", "Continue probability"),
            ],
        ),
    ]
    metrics = [
        ("next_rewards", "Reward after this action"),
        ("terminated", "Terminates after this action"),
        ("truncated", "Truncates after this action"),
        ("image_loss", "Image loss"),
        ("reward_loss", "Reward loss"),
        ("continue_loss", "Continue loss"),
        ("kl_loss", "KL loss"),
        ("world_model_loss", "Total world-model loss"),
        ("action_entropy", "Action entropy"),
    ]
    payloads = []
    for index, episode in enumerate(episodes):
        total = episode["next_rewards"].sum()
        caption = (
            "Each frame is the observation before choosing an action. "
            "The distribution and categorical sample are the ones actually used. "
            "Reward r_t belongs to the current observation; the table also shows the following reward. "
            "The recorded return has a zero tail at the end of the recording."
        )
        if metadata.get("untrained_demo"):
            caption = "UNTRAINED DEMO — use this to try the controls. " + caption
        payloads.append(
            _payload(
                episode,
                f"Episode {index}: {total:.2f} reward, {len(episode['images'])} steps",
                caption,
                [
                    ("images", "Observation o_t"),
                    ("reconstruction", "Posterior reconstruction"),
                    ("next_images", "Observation after action"),
                ],
                plots,
                metrics,
            )
        )
    return _render(payloads)


def view_imagination(branch):
    plots = [
        (
            "Imagined reward and continuation",
            [("predicted_reward", "Reward"), ("predicted_continue", "Continue")],
        ),
        (
            "Values and lambda return",
            [
                ("value", "Critic"),
                ("target_value", "Target critic"),
                ("lambda_return", "Lambda return"),
            ],
        ),
        (
            "Lambda return contributions",
            [
                ("lambda_return", "Total"),
                ("reward_return", "Reward contribution"),
                ("bootstrap_return", "Critic-value contribution"),
            ],
        ),
        (
            "Actor loss contributions",
            [
                ("actor_reward_loss", "Return term"),
                ("actor_entropy_loss", "Entropy term"),
            ],
        ),
        (
            "Critic loss and step weight",
            [("critic_loss", "Critic loss"), ("weight", "Step weight")],
        ),
    ]
    metrics = []
    for _, series in plots:
        metrics.extend(series)
    return _render([
        _payload(
            branch,
            "Policy imagination",
            "The policy chooses new actions from the recorded posterior start. "
            "The real future is a reference, not a prediction target: its actions may differ. "
            "Losses are per-step contributions to the same objectives used during training.",
            [
                ("images", "Imagined state"),
                ("next_images", "Imagined next state"),
                ("reference_images", "Real future reference"),
            ],
            plots,
            metrics,
        )
    ])


def view_prediction(prediction):
    return _render([
        _payload(
            prediction,
            "Prediction under recorded actions",
            "Both trajectories start at the selected posterior state and use the same recorded actions. "
            "After the starting state, predictions use the prior without observing more images.",
            [
                ("images", "Predicted next observation"),
                ("reference_images", "Actual next observation"),
            ],
            [
                ("Reward", [("predicted_reward", "Predicted"), ("rewards", "Actual")]),
                ("Image prediction loss", [("image_loss", "Image loss")]),
            ],
            [
                ("image_loss", "Image loss"),
                ("predicted_continue", "Continue probability"),
            ],
        )
    ])
