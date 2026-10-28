from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamer.first_step_trace import (
    TraceConfig,
    array_difference,
    compare_snapshots,
    historical_wm_objective,
    write_snapshot,
)


def test_historical_objective_keeps_original_auxiliary_outputs():
    losses = SimpleNamespace(
        image_loss=1.0, reward_loss=2.0, continue_loss=3.0, kl_loss=4.0
    )

    class Model:
        cfg = SimpleNamespace(kl_beta=0.1)

        def __call__(self, batch, initial_state):
            return object()

        @staticmethod
        def loss_fn(predictions, batch, config):
            return losses

    total, auxiliary = historical_wm_objective(Model(), object(), object())
    assert total == pytest.approx(6.4)
    assert auxiliary is losses


def test_historical_critic_rate_uses_actor_rate_not_unused_policy_rate():
    config = TraceConfig.from_metadata({
        "train_args": {
            "actor_lr": 4e-5,
            "policy_lr": 1e-4,
            "batch_size": 32,
        }
    })
    assert config.critic_lr == pytest.approx(4e-5)
    assert config.batch_size == 32
    modern = TraceConfig.from_metadata({
        "train_args": {
            "actor_lr": 4e-5,
            "critic_lr": 2e-4,
        }
    })
    assert modern.critic_lr == pytest.approx(2e-4)


def test_comparison_finds_earliest_changed_tensor_and_keeps_small_differences(tmp_path):
    left, right = tmp_path / "stock", tmp_path / "optimized"
    for directory in (left, right):
        write_snapshot(
            directory,
            "initial",
            {"weights": jnp.ones((2, 3)), "key": jax.random.key(20)},
        )
        write_snapshot(
            directory,
            "sample",
            {"image": jnp.zeros((1, 2)), "index": jnp.array([1, 2])},
        )
    values = np.array([[1, 2], [3, 4]], np.float32)
    changed = values.copy()
    changed[1, 0] = np.nextafter(changed[1, 0], np.float32(4))
    write_snapshot(left, "critic_targets", {"targets": values})
    write_snapshot(right, "critic_targets", {"targets": changed})
    comparison = compare_snapshots(left, right)
    assert comparison["first_differing_stage"] == "critic_targets"
    assert comparison["stages"]["initial"]["bitwise_equal"]
    assert comparison["stages"]["sample"]["bitwise_equal"]
    detail = comparison["stages"]["critic_targets"]["different_tensors"]["['targets']"]
    assert detail["different_elements"] == 1
    assert detail["first_index"] == [1, 0]
    assert detail["max_abs"] > 0
    assert detail["allclose_rtol_1e-5_atol_1e-7"]
    assert np.load(left / "initial.npz", allow_pickle=False).files


def test_comparison_handles_missing_stages_and_signed_zero(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    write_snapshot(left, "initial", {"x": jnp.asarray(0.0)})
    write_snapshot(right, "sample", {"x": jnp.asarray(0.0)})
    result = compare_snapshots(left, right)
    assert result["first_differing_stage"] == "initial"
    assert result["stages"]["initial"]["reason"] == "missing_stage"
    signed = array_difference(np.array(0.0, np.float32), np.array(-0.0, np.float32))
    assert not signed["bitwise_equal"]
    assert signed["first_index"] == []
    assert signed["max_abs"] == 0


def test_comparison_does_not_hide_a_different_rng_key(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    write_snapshot(left, "initial", {"key": jax.random.key(20)})
    write_snapshot(right, "initial", {"key": jax.random.key(21)})
    result = compare_snapshots(left, right)
    assert result["first_differing_stage"] == "initial"
    assert "['key']" in result["stages"]["initial"]["different_tensors"]
