import importlib.util
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pytest

script = Path(__file__).resolve().parents[1] / "scripts/evaluate_dreamer.py"
spec = importlib.util.spec_from_file_location("evaluate_dreamer", script)
assert spec is not None and spec.loader is not None
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_episode_statistics_and_reproducible_iqm_interval():
    returns = np.array([0] * 32 + [4] * 16 + [100] * 16)
    before = np.random.get_state()
    result = evaluation.summarize_returns(returns, bootstrap_reps=200, seed=20)
    after = np.random.get_state()
    assert result["mean"] == 26
    assert result["median"] == 2  # rliable.aggregate_median is a different estimator.
    assert result["iqm"] == 2
    assert result["iqm_ci_95"][0] <= 2 <= result["iqm_ci_95"][1]
    assert result == evaluation.summarize_returns(
        returns, bootstrap_reps=200, seed=20
    )
    np.testing.assert_array_equal(before[1], after[1])
    assert before[0] == after[0] and before[2:] == after[2:]


@pytest.mark.parametrize("returns", [[1], [1, np.nan], [[1], [2]]])
def test_invalid_returns_are_rejected(returns):
    with pytest.raises(ValueError):
        evaluation.summarize_returns(returns)


class ShortEpisodeEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (96, 96, 3), dtype=np.uint8)
    action_space = gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.length = 1 if seed is None or seed % 2 == 0 else 3
        return np.zeros((96, 96, 3), np.uint8), {}

    def step(self, action):
        self.steps += 1
        done = self.steps == self.length
        return (
            np.zeros((96, 96, 3), np.uint8),
            float(self.steps),
            done and self.length == 1,
            done and self.length == 3,
            {},
        )


@pytest.mark.parametrize("max_steps, expected_returns, expected_lengths", [
    (5, [1, 6, 1], [1, 3, 1]),
    (2, [1, 3, 1], [1, 2, 1]),
])
def test_evaluation_counts_each_episode_once_and_handles_a_partial_batch(
    monkeypatch, max_steps, expected_returns, expected_lengths,
):
    envs = gym.vector.SyncVectorEnv([ShortEpisodeEnv, ShortEpisodeEnv])
    wm = SimpleNamespace(
        cfg=SimpleNamespace(action_dim=3),
        rssm=SimpleNamespace(initialize_state=lambda n: (jnp.zeros((n, 1)),) * 2),
    )
    metadata = {"train_args": {"env_id": "test"}}
    monkeypatch.setattr(
        evaluation, "load_checkpoint", lambda _: (
            {"world_model": wm, "actor": None}, metadata,
        ),
    )
    monkeypatch.setattr(evaluation.gym, "make_vec", lambda *args, **kwargs: envs)
    monkeypatch.setattr(evaluation.nnx, "reseed", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluation, "choose_action",
        lambda wm, actor, carry, image, previous_action, is_first, key: (
            carry, previous_action,
        ),
    )
    returns, lengths, _ = evaluation.evaluate_checkpoint(
        Path("checkpoint"), num_episodes=3, num_envs=2,
        max_steps=max_steps, seed=0,
    )
    np.testing.assert_array_equal(returns, expected_returns)
    assert lengths == expected_lengths
    assert envs.closed
