"""First-update reconstruction and exact tensor comparisons for diagnostics.

Returning intermediates changes the compiled graph. These traces isolate a
reproduced discrepancy; they do not recover unsaved tensors from an old run.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import optax
from einops import rearrange
from flax import nnx

from dreamer.actor_critic import Actor, Critic
from dreamer.buffer import get_batch, refresh_model_states
from dreamer.imagine import run_imagination_rollout


@dataclass(frozen=True)
class TraceConfig:
    batch_size: int = 32
    seq_length: int = 64
    imagination_horizon: int = 15
    wm_lr: float = 1e-4
    actor_lr: float = 4e-5
    critic_lr: float = 4e-5
    adamw_eps: float = 1e-5

    @classmethod
    def from_metadata(cls, metadata):
        args = metadata["train_args"]
        fields = {name: args[name] for name in cls.__dataclass_fields__ if name in args}
        # Before 0a0a301, policy_lr was unused and the critic used actor_lr.
        fields["critic_lr"] = args.get("critic_lr", args.get("actor_lr", 4e-5))
        return cls(**fields)


STAGES = (
    "initial",
    "post_collection",
    "replay",
    "sample",
    "wm_forward",
    "wm_gradients",
    "wm_updated",
    "posterior_embedding",
    "posterior",
    "imagined_states",
    "imagined_rewards",
    "imagined_continuations",
    "bootstrap_values",
    "critic_targets",
    "critic_predictions",
    "critic_loss",
    "actor_gradients",
    "critic_gradients",
    "refreshed_replay",
    "post_step_models",
)


def model_state(models, *, parameters_only=False):
    return {
        name: nnx.state(model, nnx.Param).to_pure_dict()
        if parameters_only
        else nnx.state(model).to_pure_dict()
        for name, model in models.items()
    }


def make_optimizers(models, config):
    return {
        name: nnx.Optimizer(
            models[name],
            tx=optax.chain(
                optax.transforms.adaptive_grad_clip(0.3),
                optax.adamw(lr, eps=config.adamw_eps),
            ),
            wrt=nnx.Param,
        )
        for name, lr in (
            ("world_model", config.wm_lr),
            ("actor", config.actor_lr),
            ("critic", config.critic_lr),
        )
    }


def historical_wm_objective(wm, batch, initial_state):
    # Preserve the original objective/auxiliary outputs independently of newer
    # training optimizations. The historical learner recomputed posterior states
    # after updating the world model; that second pass also advances its RNG.
    predictions = wm(batch, initial_state)
    losses = wm.loss_fn(predictions, batch, wm.cfg)
    loss = (
        wm.cfg.kl_beta * losses.kl_loss
        + losses.continue_loss
        + losses.reward_loss
        + losses.image_loss
    )
    return loss, losses


def traced_actor_objective(actor, target_critic, wm, initial, horizon, key):
    sg = jax.lax.stop_gradient
    rollout, rewards, continues = run_imagination_rollout(
        wm, actor, target_critic, sg(initial), horizon, key
    )
    losses, targets, weights = Actor.loss_fn(
        actor.cfg, rewards, continues, rollout.value, rollout.action_dist
    )
    features = sg(rollout.state.joint_latent[:, :-1])
    trace = {
        "imagined_states": {
            "hidden": rollout.state.hidden,
            "latent": rollout.state.latent_sample,
            "prior": rollout.state.prior,
            "actions": rollout.action,
        },
        "imagined_rewards": rewards,
        "imagined_continuations": continues,
        "bootstrap_values": rollout.value,
        "critic_targets": {"targets": sg(targets), "weights": weights},
    }
    return losses.loss, ((features, sg(targets), weights), trace)


def traced_critic_objective(critic, features, targets, weights):
    values = critic(features)
    return Critic.loss_fn(values, targets, weights), values


wm_value_and_grad = nnx.value_and_grad(historical_wm_objective, has_aux=True)
actor_value_and_grad = nnx.value_and_grad(traced_actor_objective, has_aux=True)
critic_value_and_grad = nnx.value_and_grad(traced_critic_objective, has_aux=True)


@nnx.jit(static_argnames=("config", "num_envs"))
def trace_update(models, optimizers, replay, config, key, num_envs):
    wm, actor = models["world_model"], models["actor"]
    critic, target = models["critic"], models["target_critic"]
    batch_key, imagination_key = jax.random.split(key)
    batch, context = get_batch(replay, config.batch_size, config.seq_length, batch_key)
    trace = {
        "sample": {
            "batch": batch,
            "context": context,
            "batch_key": batch_key,
            "imagination_key": imagination_key,
        }
    }

    (loss, losses), gradients = wm_value_and_grad(wm, batch, context.initial_state)
    trace["wm_forward"] = {"loss": loss, "losses": losses}
    trace["wm_gradients"] = gradients.to_pure_dict()
    optimizers["world_model"].update(wm, gradients)
    trace["wm_updated"] = nnx.state(wm).to_pure_dict()

    # Same operations as wm.run_rssm, with its embedding exposed separately.
    embedding = wm.encoder(batch.image)
    states = wm.rssm(
        embedding, batch.prev_action, batch.is_first, context.initial_state
    )
    trace["posterior_embedding"] = embedding
    trace["posterior"] = {
        "states": states,
        "rngs": nnx.state(wm.rssm.rngs).to_pure_dict(),
    }
    updated_replay = refresh_model_states(replay, context, states)
    initial = jax.tree.map(
        lambda x: rearrange(x, "B T ... -> (B T) ..."), states.replace(posterior=None)
    )
    (actor_loss, (critic_data, imagined)), actor_gradients = actor_value_and_grad(
        actor, target, wm, initial, config.imagination_horizon, imagination_key
    )
    trace.update(imagined)
    trace["actor_gradients"] = {
        "loss": actor_loss,
        "gradients": actor_gradients.to_pure_dict(),
    }
    optimizers["actor"].update(actor, actor_gradients)
    features, targets, weights = critic_data
    (critic_loss, values), critic_gradients = critic_value_and_grad(
        critic, features, targets, weights
    )
    trace["critic_predictions"] = values
    trace["critic_loss"] = {"loss": critic_loss}
    trace["critic_gradients"] = critic_gradients.to_pure_dict()
    optimizers["critic"].update(critic, critic_gradients)
    nnx.update(
        target,
        optax.incremental_update(
            new_tensors=nnx.state(critic, nnx.Param),
            old_tensors=nnx.state(target, nnx.Param),
            step_size=0.02,
        ),
    )
    trace["post_step_models"] = model_state(models)
    # Retain replay refresh outputs so its scatter operations are not discarded.
    trace["refreshed_replay"] = {
        "hidden": updated_replay.model_state_hidden[:num_envs],
        "latent": updated_replay.model_state_latent[:num_envs],
    }
    return trace


def write_snapshot(directory, stage, tree):
    """Store full arrays and hashes, including typed RNG keys without pickles."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays, manifest = {}, {}
    for index, (path, value) in enumerate(
        jax.tree_util.tree_flatten_with_path(tree)[0]
    ):
        name = jax.tree_util.keystr(path) or "<root>"
        original_dtype = str(value.dtype)
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            value = jax.random.key_data(value)
        array = np.asarray(jax.device_get(value))
        array = np.array(array, copy=True, order="C")
        key = f"array_{index:04d}"
        arrays[key] = array
        manifest[name] = {
            "key": key,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "original_dtype": original_dtype,
            "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
        }
    np.savez(directory / f"{stage}.npz", **arrays)
    (directory / f"{stage}.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def array_difference(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "bitwise_equal": False,
            "reason": "shape_or_dtype",
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
        }
    a, b = left.reshape(-1), right.reshape(-1)
    different, first, max_abs = 0, None, 0.0
    close = True
    for start in range(0, a.size, 1_000_000):
        av, bv = a[start : start + 1_000_000], b[start : start + 1_000_000]
        changed = np.any(
            av.view(np.uint8).reshape(-1, av.dtype.itemsize)
            != bv.view(np.uint8).reshape(-1, bv.dtype.itemsize),
            axis=1,
        )
        different += int(changed.sum())
        if first is None and changed.any():
            first = start + int(np.flatnonzero(changed)[0])
        delta = np.abs(av.astype(np.float64) - bv.astype(np.float64))
        if delta.size:
            max_abs = max(max_abs, float(np.nanmax(delta)))
        close &= bool(np.allclose(av, bv, rtol=1e-5, atol=1e-7, equal_nan=True))
    return {
        "bitwise_equal": different == 0,
        "different_elements": different,
        "first_index": [int(i) for i in np.unravel_index(first, left.shape)]
        if first is not None
        else None,
        "left_at_first": a[first].item() if first is not None else None,
        "right_at_first": b[first].item() if first is not None else None,
        "max_abs": max_abs,
        "allclose_rtol_1e-5_atol_1e-7": close,
    }


def compare_snapshots(left, right, stages=STAGES):
    result = {"first_differing_stage": None, "stages": {}}
    for stage in stages:
        lp, rp = Path(left) / f"{stage}.json", Path(right) / f"{stage}.json"
        if not lp.exists() and not rp.exists():
            continue
        if not lp.exists() or not rp.exists():
            detail = {"bitwise_equal": False, "reason": "missing_stage"}
        else:
            lm, rm = json.loads(lp.read_text()), json.loads(rp.read_text())
            diffs = {}
            with (
                np.load(lp.with_suffix(".npz"), allow_pickle=False) as la,
                np.load(rp.with_suffix(".npz"), allow_pickle=False) as ra,
            ):
                for path in sorted(lm.keys() | rm.keys()):
                    if path not in lm or path not in rm:
                        diffs[path] = {
                            "bitwise_equal": False,
                            "reason": "missing_tensor",
                        }
                    elif lm[path] != rm[path]:
                        difference = array_difference(
                            la[lm[path]["key"]], ra[rm[path]["key"]]
                        )
                        if lm[path]["original_dtype"] != rm[path]["original_dtype"]:
                            difference.update(
                                bitwise_equal=False, reason="original_dtype"
                            )
                        if not difference["bitwise_equal"]:
                            diffs[path] = difference
            detail = {"bitwise_equal": not diffs, "different_tensors": diffs}
        result["stages"][stage] = detail
        if not detail["bitwise_equal"] and result["first_differing_stage"] is None:
            result["first_differing_stage"] = stage
    return result
