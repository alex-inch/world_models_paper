"""Integration checks for the inspector's temporal alignment and state isolation."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from dreamer.actor_critic import Actor
from dreamer.inspection import load_recording
from dreamer.interactive import InspectorSession
from dreamer.wm import DreamerConfig, DreamerWM


class DrivingEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (96, 96, 3), dtype=np.uint8)
    action_space = gym.spaces.Box(
        np.array([-1, 0, 0], np.float32), np.ones(3, np.float32)
    )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.position = 0.0
        self.actions = []
        return self.image(), {}

    def image(self):
        return np.full((96, 96, 3), 80 + self.t * 10 + int(self.position * 5), np.uint8)

    def step(self, action):
        self.actions.append(action.copy())
        self.t += 1
        self.position += float(action[0])
        return self.image(), float(self.t), False, self.t == 3, {}


class ConstantCritic(nnx.Module):
    def __init__(self):
        self.value = nnx.Param(jnp.asarray(5.0))

    def __call__(self, features):
        return jnp.broadcast_to(self.value[...], features.shape[:-1])


@pytest.fixture(scope="module")
def models():
    config = DreamerConfig(
        wm_hidden_dim=4,
        num_latent_dists=2,
        num_latent_classes=3,
        agent_hidden_dim=8,
        discount=0.8,
        td_lambda=0.7,
    )
    return {
        "world_model": DreamerWM(config, rngs=nnx.Rngs(0)),
        "actor": Actor(config, nnx.Rngs(1)),
        "critic": ConstantCritic(),
        "target_critic": ConstantCritic(),
    }


@pytest.fixture
def session(models):
    env = DrivingEnv()
    try:
        yield InspectorSession(models, env, seed=20, horizon=3)
    finally:
        env.close()


def state_arrays(model):
    result = []
    for value in jax.tree.leaves(nnx.state(model)):
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            value = jax.random.key_data(value)
        result.append(np.array(value, copy=True))
    return result


def test_preview_does_not_step_env_or_consume_live_rng(session, models):
    before = state_arrays(session.wm)
    original = state_arrays(models["world_model"])
    selected = np.array([0.25, 0.75, 0], np.float32)
    first = session.preview(selected)
    session.preview(np.array([-0.5, 0, 0.5], np.float32))
    again = session.preview(selected)
    assert session.env.t == 0
    assert session.current.index == 0
    np.testing.assert_array_equal(first.state.latent_sample, again.state.latent_sample)
    for expected, actual in zip(before, state_arrays(session.wm), strict=True):
        np.testing.assert_array_equal(expected, actual)
    transition = session.step(selected)
    np.testing.assert_array_equal(session.env.actions, [selected])
    assert transition.reward == 1
    assert transition.before.index == 0
    assert transition.after.index == 1
    # Both paths must advance h exactly once, using the same action and previous z.
    np.testing.assert_allclose(
        transition.forecast.state.hidden, transition.after.state.hidden, atol=1e-6
    )
    np.testing.assert_allclose(
        transition.forecast.state.prior, transition.after.state.prior, atol=1e-6
    )
    for expected, actual in zip(
        original, state_arrays(models["world_model"]), strict=True
    ):
        np.testing.assert_array_equal(expected, actual)


def test_forecast_branches_and_return_diagnostics_do_not_change_real_policy(
    session, models
):
    reference = InspectorSession(models, DrivingEnv(), seed=20, horizon=3)
    try:
        for _ in range(3):
            np.testing.assert_array_equal(
                session.actor_action(), reference.actor_action()
            )
            action = reference.actor_action()
            session.preview(np.array([1, 1, 0], np.float32))
            session.preview(np.array([-1, 0, 1], np.float32))
            if session.history:
                session.compare_returns(0)
            session.step(action)
            reference.step(action)
            np.testing.assert_array_equal(
                session.current.state.latent_sample,
                reference.current.state.latent_sample,
            )
            np.testing.assert_array_equal(
                session.current.state.hidden, reference.current.state.hidden
            )
    finally:
        reference.env.close()


def test_open_loop_preserves_its_own_state_until_corrected(session):
    session.set_open_loop(True)
    action = np.array([0.3, 0.5, 0], np.float32)
    first = session.step(action)
    second = session.step(action)
    assert first.forecast.open_loop_start == second.forecast.open_loop_start == 0
    session.correct()
    third = session.preview(action)
    assert third.open_loop_start == 2
    # Once corrected, the next prediction again shares h with observed inference.
    final = session.step(action)
    np.testing.assert_allclose(third.state.hidden, final.after.state.hidden, atol=1e-6)


def test_real_lambda_targets_bootstrap_at_cutoff_and_stop_at_training_truncations(
    session,
):
    action = np.array([0, 0.5, 0], np.float32)
    session.step(action)
    partial = session.compare_returns(0)
    # A one-transition nonterminal cutoff must use the entire terminal V,
    # irrespective of lambda: r + gamma V = 1 + .8 * 5.
    assert partial["steps"] == 1
    assert partial["recorded"] == pytest.approx(5)
    assert partial["recorded_reward"] == pytest.approx(1)
    assert partial["recorded_bootstrap"] == pytest.approx(4)
    session.step(action)
    session.step(action)
    complete = session.compare_returns(0)
    assert complete["steps"] == 3
    assert complete["recorded"] == pytest.approx(4.9328)
    assert session.compare_returns(2)["recorded"] == pytest.approx(3)
    for prefix in ("recorded", "imagined"):
        assert complete[prefix] == pytest.approx(
            complete[prefix + "_reward"] + complete[prefix + "_bootstrap"]
        )
    with pytest.raises(RuntimeError, match="episode has ended"):
        session.step(action)


def test_restore_replays_actions_and_exports_previous_branch(session, tmp_path):
    session.step(np.array([0.4, 0.5, 0], np.float32))
    bookmarked = session.current
    session.bookmark()
    session.step(np.array([-0.2, 0.5, 0.1], np.float32))
    session.restore(1)
    assert session.current.index == 1
    assert len(session.archived) == 1
    np.testing.assert_array_equal(session.current.image, bookmarked.image)
    np.testing.assert_array_equal(
        session.current.state.latent_sample, bookmarked.state.latent_sample
    )
    session.step(np.array([0.8, 0, 0.5], np.float32))
    session.compare_returns(0)
    session.save(tmp_path / "recording", {"env_id": "test"})
    episodes, metadata = load_recording(tmp_path / "recording")
    assert metadata["num_episodes"] == 2
    assert metadata["bookmarks"][0]["frame"] == 1
    np.testing.assert_array_equal(
        episodes[1]["prev_actions"][1:], episodes[1]["actions"][:-1]
    )
    np.testing.assert_array_equal(episodes[1]["next_rewards"], [1, 2])
    assert episodes[1]["action_std"].shape == (2, 3)
    assert episodes[1]["lambda_steps"][0] == 2
    assert np.isfinite(episodes[1]["recorded_lambda_return"][0])


@pytest.mark.parametrize("action", [[2, 0, 0], [0, -1, 0], [0, 0, np.nan], [0, 0]])
def test_invalid_action_is_rejected_before_environment_step(session, action):
    with pytest.raises(ValueError):
        session.step(action)
    assert session.env.t == 0


def test_actor_whiskers_use_truncated_std_and_pygame_controls_drive_real_steps(
    session, monkeypatch
):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from dreamer.interactive_viewer import (
        InspectorControls,
        InspectorRenderer,
        run_inspector,
    )

    diagnostics = session.current.diagnostics
    assert np.isfinite(diagnostics["action_std"]).all()
    assert (diagnostics["action_std"] < diagnostics["action_scale"]).all()
    renderer = InspectorRenderer()
    surface = renderer.draw(session, InspectorControls())
    assert surface.get_size() == (1400, 1020)
    assert renderer.action_at((148, 490)) == (0, -1)
    assert renderer.action_at((453, 551)) == (1, 1)
    batches = iter([
        [
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT),
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_PERIOD),
        ],
        [
            pygame.event.Event(pygame.KEYUP, key=pygame.K_RIGHT),
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_i),
        ],
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_PERIOD)],
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t)],
        [pygame.event.Event(pygame.QUIT)],
    ])
    monkeypatch.setattr(pygame.event, "get", lambda: next(batches))
    try:
        run_inspector(session, save=lambda: "saved")
        assert len(session.history) == 2
        assert session.env.actions[0][0] == 1
        assert session.env.actions[1][0] == 0
        assert session.open_loop
        assert session.comparisons[1]["steps"] == 1
    finally:
        pygame.quit()
