import json

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from gymnasium.vector import SyncVectorEnv

from dreamer.actor_critic import Actor, Critic
from dreamer.buffer import choose_action
from dreamer.checkpoints import load_checkpoint, save_checkpoint
from dreamer.inspection import (
    _frame_losses,
    _state_from_arrays,
    analyse_episode,
    imagine_from_frame,
    load_recording,
    predict_recorded_actions,
    record_episode,
    record_step,
    save_recording,
)
from dreamer.viewer import view_imagination, view_prediction, view_recording
from dreamer.wm import DreamerBatch, DreamerConfig, DreamerWM, DreamerWMOut


class ImageEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (96, 96, 3), dtype=np.uint8)
    action_space = gym.spaces.Box(
        np.array([-1, 0, 0], np.float32), np.ones(3, np.float32)
    )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.full((96, 96, 3), 50, np.uint8), {}

    def step(self, action):
        self.t += 1
        image = np.full((96, 96, 3), 50 + 10 * self.t, np.uint8)
        return image, float(self.t), False, self.t == 4, {}


@pytest.fixture(scope="module")
def models():
    cfg = DreamerConfig(
        wm_hidden_dim=4, num_latent_dists=2, num_latent_classes=2, agent_hidden_dim=8
    )
    return {
        "world_model": DreamerWM(cfg, rngs=nnx.Rngs(0)),
        "actor": Actor(cfg, nnx.Rngs(1)),
        "critic": Critic(cfg, nnx.Rngs(2)),
        "target_critic": Critic(cfg, nnx.Rngs(3)),
    }


@pytest.fixture(scope="module")
def episode(models):
    envs = SyncVectorEnv([ImageEnv])
    try:
        recording = record_episode(models, envs, seed=20, max_steps=10)
    finally:
        envs.close()
    return analyse_episode(models, recording, chunk_size=4)


def comparable_state(model):
    values = []
    for value in jax.tree.leaves(nnx.state(model)):
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            value = jax.random.key_data(value)
        values.append(np.array(value, copy=True))
    return values


def test_checkpoint_round_trip_preserves_models_and_configuration(models, tmp_path):
    directory = tmp_path / "checkpoint"
    before = {name: comparable_state(model) for name, model in models.items()}
    metadata = {"train_step": 10, "train_args": {"env_id": "CarRacing-v3"}}
    save_checkpoint(directory, models, models["world_model"].cfg, metadata)
    restored, information = load_checkpoint(directory)
    assert information["train_step"] == 10
    assert restored["world_model"].cfg == models["world_model"].cfg
    assert set(restored) == set(models)
    for name, model in restored.items():
        for original, loaded in zip(before[name], comparable_state(model)):
            np.testing.assert_array_equal(original, loaded)
    assert (
        json.loads((directory / "metadata.json").read_text())["model_config"][
            "wm_hidden_dim"
        ]
        == 4
    )


def test_recording_uses_training_policy_and_keeps_transition_alignment(models, episode):
    assert len(episode["images"]) == 4
    np.testing.assert_array_equal(episode["rewards"], [0, 1, 2, 3])
    np.testing.assert_array_equal(episode["next_rewards"], [1, 2, 3, 4])
    np.testing.assert_array_equal(episode["prev_actions"][1:], episode["actions"][:-1])
    np.testing.assert_array_equal(episode["images"][1:], episode["next_images"][:-1])
    np.testing.assert_array_equal(episode["truncated"], [False, False, False, True])
    state = _state_from_arrays(episode)
    distribution = models["actor"](state.joint_latent)
    # Batched and single-step GPU matmuls can differ by a float32 rounding unit.
    np.testing.assert_allclose(
        distribution.loc[0], episode["action_loc"], rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        distribution.mean()[0], episode["action_mean"], rtol=1e-6
    )

    wm = nnx.clone(models["world_model"], arrays=True)
    other = nnx.clone(wm, arrays=True)
    carry = wm.rssm.initialize_state(1)
    other_carry = other.rssm.initialize_state(1)
    image = jnp.asarray(episode["images"][:1], dtype=jnp.float32) / 255 - 0.5
    action = jnp.zeros((1, 3))
    first = jnp.asarray([True])
    _, training_action = choose_action(
        wm, models["actor"], carry, image, action, first, jax.random.key(0)
    )
    _, recorded_action, _, _ = record_step(
        other, models["actor"], other_carry, image, action, first, jax.random.key(0)
    )
    np.testing.assert_array_equal(training_action, recorded_action)


def test_per_frame_world_model_losses_match_training_reductions(models, episode):
    wm = models["world_model"]
    state = _state_from_arrays(episode)
    features = state.joint_latent
    reward, continuation = wm._predict_reward_and_continue(features)
    predictions = DreamerWMOut(
        image=wm.decoder(features),
        reward=reward,
        cont=continuation,
        latent_prior=state.prior,
        latent_posterior=state.posterior,
        latent_sample=state.latent_sample,
        hidden=state.hidden,
    )
    batch = DreamerBatch(
        image=jnp.asarray(episode["images"])[None].astype(jnp.float32) / 255 - 0.5,
        prev_action=jnp.asarray(episode["prev_actions"])[None],
        reward=jnp.asarray(episode["rewards"])[None],
        cont=jnp.asarray(episode["continues"])[None],
        is_first=jnp.asarray(episode["is_first"])[None],
    )
    expected = wm.loss_fn(predictions, batch, wm.cfg)
    actual = _frame_losses(predictions, batch, wm.cfg)
    for mean_loss, frame_losses in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual)
    ):
        np.testing.assert_allclose(np.mean(frame_losses), mean_loss, rtol=1e-5)
    np.testing.assert_allclose(
        episode["image_loss"].mean(), expected.image_loss, rtol=1e-5
    )
    np.testing.assert_allclose(
        episode["reward_loss"].mean(), expected.reward_loss, rtol=1e-5
    )


def test_imagination_reuses_posterior_start_and_does_not_mutate_checkpoint(
    models, episode
):
    before = {name: comparable_state(model) for name, model in models.items()}
    first = imagine_from_frame(models, episode, 1, horizon=3, seed=42)
    second = imagine_from_frame(models, episode, 1, horizon=3, seed=42)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_array_equal(first["images"][0], episode["reconstruction"][1])
    for name, model in models.items():
        for original, after in zip(before[name], comparable_state(model)):
            np.testing.assert_array_equal(original, after)

    expected_distribution = models["actor"](
        _state_from_arrays(episode).joint_latent[:, 1:2]
    )
    np.testing.assert_array_equal(first["action_loc"][:1], expected_distribution.loc[0])
    np.testing.assert_allclose(
        first["actor_loss"], first["actor_reward_loss"] + first["actor_entropy_loss"]
    )
    np.testing.assert_allclose(
        first["lambda_return"], first["reward_return"] + first["bootstrap_return"]
    )


def test_recorded_action_predictions_and_recording_files(models, episode, tmp_path):
    prediction = predict_recorded_actions(models, episode, 2, horizon=15, seed=10)
    assert prediction["images"].shape == (2, 64, 64, 3)
    np.testing.assert_array_equal(
        prediction["reference_images"], episode["next_images"][2:]
    )
    np.testing.assert_array_equal(prediction["rewards"], episode["next_rewards"][2:])

    directory = tmp_path / "recording"
    save_recording(directory, [episode], {"seed": 20})
    loaded, metadata = load_recording(directory)
    assert metadata["num_episodes"] == 1
    for key in episode:
        np.testing.assert_array_equal(episode[key], loaded[0][key])

    assert "<iframe" in view_recording(loaded, metadata).data
    assert (
        "Policy imagination"
        in view_imagination(imagine_from_frame(models, episode, 0, horizon=2)).data
    )
    assert "Prediction under recorded actions" in view_prediction(prediction).data
