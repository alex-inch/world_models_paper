"""Stateful, UI-independent Dreamer driving and matched prediction diagnostics."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from scipy.stats import truncnorm

from dreamer.actor_critic import compute_td_lambda_returns
from dreamer.buffer import policy_step
from dreamer.inspection import analyse_episode, frames_to_uint8, save_recording
from dreamer.wm.rssm import RSSMOut
from wm.utils.obs import crop_obs, normalise_obs

_returns = jax.jit(compute_td_lambda_returns, static_argnums=0)


@nnx.jit
def _observe(wm, actor, carry, image, prev_action, is_first):
    # Reuse the existing inference path. Its sampled action is not applied to
    # the environment; manual controls or the actor can independently supply it.
    carry, _, state, _ = policy_step(
        wm, actor, carry, image, prev_action, is_first, jax.random.key(0)
    )
    return carry, state


def _heads(wm, actor, critic, target_critic, state):
    features = state.joint_latent
    reward, continuation = wm._predict_reward_and_continue(features)
    distribution = actor(features)
    return {
        "reconstruction": wm.decoder(features)[0],
        "predicted_reward": wm.decode_reward_logits(reward)[0],
        "predicted_continue": jax.nn.sigmoid(continuation)[0],
        "value": critic(features)[0],
        "target_value": target_critic(features)[0],
        "action_loc": distribution.loc[0],
        "action_scale": distribution.scale[0],
        "action_lower": distribution.lower[0],
        "action_upper": distribution.upper[0],
        "action_mean": distribution.mean()[0],
        "action_entropy": distribution.entropy()[0],
    }


_diagnose = nnx.jit(_heads)


@nnx.jit
def _forecast(wm, actor, critic, target_critic, state, action):
    state = wm.imagine_step(state, action)
    return state, _heads(wm, actor, critic, target_critic, state)


@nnx.scan(in_axes=(None, nnx.Carry, 1), out_axes=(nnx.Carry, 1))
def _action_step(wm, state, action):
    state = wm.imagine_step(state, action)
    return state, state


@nnx.jit
def _action_branch(wm, target_critic, initial, actions):
    _, states = _action_step(wm, initial, actions)
    reward, continuation = wm._predict_reward_and_continue(states.joint_latent)
    values = jnp.concatenate(
        [
            target_critic(initial.joint_latent)[:, None],
            target_critic(states.joint_latent),
        ],
        axis=1,
    )
    return wm.decode_reward_logits(reward), jax.nn.sigmoid(continuation), values


@nnx.jit
def _sample_actor(actor, state, key):
    return actor(state.joint_latent).sample(seed=key)[0]


def _host_diagnostics(values):
    result = {name: np.asarray(value) for name, value in jax.device_get(values).items()}
    result["reconstruction"] = frames_to_uint8(result["reconstruction"])
    loc, scale = result["action_loc"], result["action_scale"]
    # The normal's scale is not the standard deviation after truncation.
    result["action_std"] = np.asarray(
        truncnorm.std(
            (result["action_lower"] - loc) / scale,
            (result["action_upper"] - loc) / scale,
            loc=loc,
            scale=scale,
        ),
        dtype=np.float32,
    )
    return result


@dataclass
class ObservedFrame:
    index: int
    image: np.ndarray
    state: RSSMOut
    diagnostics: dict
    incoming_reward: float
    incoming_action: np.ndarray


@dataclass
class Forecast:
    action: np.ndarray
    state: RSSMOut
    diagnostics: dict
    open_loop_start: int | None


@dataclass
class Transition:
    before: ObservedFrame
    forecast: Forecast
    after: ObservedFrame
    reward: float
    terminated: bool
    truncated: bool
    source: str

    @property
    def factor_kl(self):
        """Compare distributions at the same time; histories differ in open loop."""
        p = np.asarray(self.forecast.state.prior[0], dtype=np.float64)
        q = np.asarray(self.after.state.posterior[0], dtype=np.float64)
        p = p - np.logaddexp.reduce(p, axis=-1, keepdims=True)
        q = q - np.logaddexp.reduce(q, axis=-1, keepdims=True)
        return (np.exp(q) * (q - p)).sum(axis=-1)


class InspectorSession:
    """Run one real environment while observing a frozen checkpoint.

    Live observations, previews, and retrospective branches use separate RNGs.
    Previewing an action never steps the environment or changes the posterior.
    """

    def __init__(self, models, env, *, seed=500_000, horizon=15):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.models = {
            name: nnx.clone(model, arrays=True) for name, model in models.items()
        }
        self.wm = self.models["world_model"]
        self.actor = self.models["actor"]
        self.critic = self.models["critic"]
        self.target_critic = self.models["target_critic"]
        if self.wm.cfg.action_dim != 3 or env.action_space.shape != (3,):
            raise ValueError("The driving inspector requires three CarRacing actions")
        self.forecast_wm = nnx.clone(self.wm, arrays=True)
        self.branch_wm = nnx.clone(self.wm, arrays=True)
        self.env = env
        self.horizon = horizon
        self.archived = []
        self.bookmarks = []
        self.started = perf_counter()
        self.revision = 0
        self.reset(seed=seed)

    def _diagnostics(self, state):
        return _host_diagnostics(
            _diagnose(self.wm, self.actor, self.critic, self.target_critic, state)
        )

    def _frame(self, observation, action, reward, *, first):
        image = crop_obs(observation)
        self.carry, state = _observe(
            self.wm,
            self.actor,
            self.carry,
            jnp.asarray(normalise_obs(image)[None] - 0.5),
            jnp.asarray(action[None]),
            jnp.asarray([first]),
        )
        return ObservedFrame(
            len(self.history),
            image,
            state,
            self._diagnostics(state),
            reward,
            action.copy(),
        )

    def reset(self, *, seed=None, archive=True):
        if archive and getattr(self, "history", None):
            self.archived.append((self.to_episode(), self.seed))
        self.seed = int(seed if seed is not None else self.seed)
        nnx.reseed(self.wm, default=self.seed)
        self.carry = self.wm.rssm.initialize_state(1)
        self.history = []
        self.comparisons = {}
        self.open_loop = False
        self.open_loop_start = None
        self._cached_preview = None
        self.done = False
        observation, _ = self.env.reset(seed=self.seed)
        self.current = self._frame(
            observation, np.zeros(3, np.float32), 0.0, first=True
        )
        self.free_state = self.current.state.replace(posterior=None)
        self.revision += 1

    def actor_action(self):
        key = jax.random.fold_in(jax.random.key(self.seed), self.current.index)
        return np.asarray(_sample_actor(self.actor, self.current.state, key))

    def set_open_loop(self, enabled):
        self.open_loop = bool(enabled)
        self.correct()

    def correct(self):
        self.free_state = self.current.state.replace(posterior=None)
        self.open_loop_start = self.current.index if self.open_loop else None
        self._cached_preview = None

    def preview(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("action must be a finite three-element vector")
        if np.any(action < self.env.action_space.low) or np.any(
            action > self.env.action_space.high
        ):
            raise ValueError("action is outside the environment's bounds")
        if self._cached_preview is not None and np.array_equal(
            action, self._cached_preview.action
        ):
            return self._cached_preview
        # Reuse the same noise for candidate actions at this timestep.
        nnx.reseed(
            self.forecast_wm, default=(self.seed + self.current.index + 1) % 2**32
        )
        initial = (
            self.free_state
            if self.open_loop
            else self.current.state.replace(posterior=None)
        )
        state, diagnostics = _forecast(
            self.forecast_wm,
            self.actor,
            self.critic,
            self.target_critic,
            initial,
            jnp.asarray(action[None]),
        )
        self._cached_preview = Forecast(
            action.copy(), state, _host_diagnostics(diagnostics), self.open_loop_start
        )
        return self._cached_preview

    def step(self, action, *, source="human"):
        if self.done:
            raise RuntimeError("The episode has ended; reset before stepping again")
        forecast = self.preview(action)
        # Commit the forecast before requesting the next real observation.
        before = self.current
        observation, reward, terminated, truncated, _ = self.env.step(
            forecast.action.copy()
        )
        after = self._frame(observation, forecast.action, float(reward), first=False)
        after.index = before.index + 1
        transition = Transition(
            before,
            forecast,
            after,
            float(reward),
            bool(terminated),
            bool(truncated),
            source,
        )
        self.history.append(transition)
        self.current = after
        self.free_state = forecast.state
        self.done = bool(terminated or truncated)
        self._cached_preview = None
        self.revision += 1
        return transition

    def compare_returns(self, start):
        """Pair real and imagined lambda targets under exactly the recorded actions."""
        if not 0 <= start < len(self.history):
            raise ValueError("Select a recorded transition to compare returns")
        transitions = self.history[start : start + self.horizon]
        count = len(transitions)
        cached = self.comparisons.get(start)
        if cached is not None and cached["steps"] == count:
            return cached
        first = transitions[0].before
        nnx.reseed(self.branch_wm, default=(self.seed + start + 100_000) % 2**32)
        actions = np.stack([t.forecast.action for t in transitions])[None]
        predicted = _action_branch(
            self.branch_wm,
            self.target_critic,
            first.state.replace(posterior=None),
            jnp.asarray(actions),
        )
        real_rewards = jnp.asarray([[t.reward for t in transitions]])
        # Match the continuation labels used by this repository's training buffer.
        real_conts = jnp.asarray(
            [[not (t.terminated or t.truncated) for t in transitions]],
            dtype=jnp.float32,
        )
        real_values = jnp.asarray([
            [
                float(first.diagnostics["target_value"]),
                *[float(t.after.diagnostics["target_value"]) for t in transitions],
            ]
        ])
        result = {
            "steps": count,
            "horizon": self.horizon,
            "value": float(first.diagnostics["value"]),
        }
        for name, (rewards, conts, values) in (
            ("recorded", (real_rewards, real_conts, real_values)),
            ("imagined", predicted),
        ):
            total = float(_returns(self.wm.cfg, rewards, conts, values)[0, 0])
            reward_part = float(
                _returns(self.wm.cfg, rewards, conts, jnp.zeros_like(values))[0, 0]
            )
            result[name] = total
            result[name + "_reward"] = reward_part
            result[name + "_bootstrap"] = total - reward_part
        self.comparisons[start] = result
        self.revision += 1
        return result

    def bookmark(self, frame=None):
        frame = self.current.index if frame is None else int(frame)
        if not 0 <= frame <= len(self.history):
            raise ValueError("bookmark is outside the current recording")
        item = {"episode": len(self.archived), "seed": self.seed, "frame": frame}
        if item not in self.bookmarks:
            self.bookmarks.append(item)
            self.revision += 1
        return item

    def restore(self, frame):
        """Restore by seed and action replay, checking pixels rather than cloning Box2D."""
        if not 0 <= frame <= len(self.history):
            raise ValueError("frame is outside the current recording")
        prefix = self.history[:frame]
        expected = (
            (prefix[-1].after if prefix else self.history[0].before)
            if self.history
            else self.current
        )
        was_open = self.open_loop
        self.reset(seed=self.seed)
        for transition in prefix:
            self.step(transition.forecast.action, source=transition.source)
            if not np.array_equal(self.current.image, transition.after.image):
                self.done = True
                raise RuntimeError(
                    "Environment replay diverged; refusing to branch from a different state"
                )
        if not np.array_equal(self.current.image, expected.image):
            raise RuntimeError("Restored observation differs from the bookmark")
        self.set_open_loop(was_open)

    def to_episode(self):
        if not self.history:
            raise ValueError("No transitions have been recorded")
        rows = []
        for transition in self.history:
            frame, prediction = transition.before, transition.forecast
            row = {
                **frame.diagnostics,
                "images": frame.image,
                "next_images": transition.after.image,
                "prev_actions": frame.incoming_action,
                "rewards": frame.incoming_reward,
                "continues": 1.0,
                "is_first": frame.index == 0,
                "actions": prediction.action,
                "next_rewards": transition.reward,
                "terminated": transition.terminated,
                "truncated": transition.truncated,
                "hidden": np.asarray(frame.state.hidden[0]),
                "latent_sample": np.asarray(frame.state.latent_sample[0]),
                "prior_logits": np.asarray(frame.state.prior[0]),
                "posterior_logits": np.asarray(frame.state.posterior[0]),
                "action_source": transition.source,
                "forecast_image": prediction.diagnostics["reconstruction"],
                "forecast_reward": prediction.diagnostics["predicted_reward"],
                "forecast_continue": prediction.diagnostics["predicted_continue"],
                "forecast_value": prediction.diagnostics["value"],
                "forecast_prior_logits": np.asarray(prediction.state.prior[0]),
                "forecast_latent_sample": np.asarray(prediction.state.latent_sample[0]),
                "open_loop_start": -1
                if prediction.open_loop_start is None
                else prediction.open_loop_start,
            }
            rows.append(row)
        episode = {name: np.stack([row[name] for row in rows]) for name in rows[0]}
        for key in (
            "recorded",
            "imagined",
            "recorded_reward",
            "recorded_bootstrap",
            "imagined_reward",
            "imagined_bootstrap",
        ):
            episode[key + "_lambda_return"] = np.asarray([
                self.comparisons.get(index, {}).get(key, np.nan)
                for index in range(len(rows))
            ])
        episode["lambda_steps"] = np.asarray([
            self.comparisons.get(index, {}).get("steps", 0)
            for index in range(len(rows))
        ])
        return episode

    def save(self, directory: Path, metadata: dict):
        recordings = [*self.archived]
        if self.history:
            recordings.append((self.to_episode(), self.seed))
        if not recordings:
            raise ValueError("No transitions have been recorded")
        episodes = [analyse_episode(self.models, episode) for episode, _ in recordings]
        save_recording(
            directory,
            episodes,
            {
                **metadata,
                "seed": self.seed,
                "episode_seeds": [seed for _, seed in recordings],
                "bookmarks": self.bookmarks,
                "elapsed_seconds": perf_counter() - self.started,
                "interactive": True,
                "lambda_horizon": self.horizon,
                "lambda_continuation": "zero on termination or truncation, matching training",
            },
        )
        return len(episodes)
