import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np

script = Path(__file__).resolve().parents[1] / "scripts/render_dreamer_policy.py"
spec = importlib.util.spec_from_file_location("render_dreamer_policy", script)
assert spec is not None and spec.loader is not None
video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video)


def test_full_imagination_never_reads_future_posterior_states(monkeypatch):
    count = 71
    episode = {
        "images": np.zeros((count, 64, 64, 3), dtype=np.uint8),
        "hidden": np.arange(count, dtype=np.float32)[:, None],
        "latent_sample": np.zeros((count, 1), dtype=np.float32),
        "prior_logits": np.zeros((count, 1, 1), dtype=np.float32),
    }
    calls = []

    def imagine(wm, actor, critic, initial, *, horizon, key):
        calls.append((np.asarray(initial.hidden).copy(), horizon))
        values = initial.hidden[:, None, :] + 100 * jnp.arange(horizon + 1)[None, :, None]
        return jnp.concatenate([values, jnp.zeros_like(values)], axis=-1)

    monkeypatch.setattr(video, "_branch_world_model", lambda *args: None)
    monkeypatch.setattr(video, "_imagine_features", imagine)
    monkeypatch.setattr(video, "_decode_features", lambda wm, features: np.asarray(features)[..., :1])
    models = {"world_model": None, "actor": None, "target_critic": None}
    before = video.prediction_panels(models, episode)
    changed = {name: values.copy() for name, values in episode.items()}
    changed["hidden"][5:] += 10_000
    after = video.prediction_panels(models, changed)
    np.testing.assert_array_equal(before["full_imagination"], after["full_imagination"])
    assert not np.array_equal(before["periodic_imagination"], after["periodic_imagination"])
    np.testing.assert_array_equal(before["full_imagination"][:5, 0], np.arange(5))
    np.testing.assert_array_equal(before["full_imagination"][5:, 0], 4 + 100 * np.arange(1, count - 4))
    assert calls[1][1] == count - 5
    np.testing.assert_array_equal(calls[1][0], [[4]])
    # Periodic prediction starts at frame 4 and resets after the next five observations.
    assert before["periodic_imagination"][19, 0] == 1504
    np.testing.assert_array_equal(before["periodic_imagination"][20:25, 0], np.arange(20, 25))

    # The full branch can receive 50 frames without changing the periodic cycle.
    fifty = video.prediction_panels(models, episode, full_context=50)
    np.testing.assert_array_equal(fifty["periodic_imagination"], before["periodic_imagination"])
    np.testing.assert_array_equal(fifty["full_imagination"][:50, 0], np.arange(50))
    np.testing.assert_array_equal(
        fifty["full_imagination"][50:, 0], 49 + 100 * np.arange(1, count - 49),
    )
    future_changed = {name: values.copy() for name, values in episode.items()}
    future_changed["hidden"][50:] += 10_000
    after_fifty = video.prediction_panels(models, future_changed, full_context=50)
    np.testing.assert_array_equal(fifty["full_imagination"], after_fifty["full_imagination"])
