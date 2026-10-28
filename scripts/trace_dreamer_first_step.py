"""Compare real initial checkpoints and reconstructed first-update tensors.

The original replay and critic targets were not checkpointed. This reconstructs
them and repeats frozen inputs on one device; it cannot recover historical GPU
rounding or guarantee the instrumented graph matches the original executable.
"""

import argparse
import datetime
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def save_report(root, report):
    (root / "report.json").write_text(json.dumps(report, indent=2))
    rows = [
        "# Dreamer first-step trace",
        "",
        report["scope"],
        "",
        "Initial tensors come from real step-zero checkpoints. Replay and learner tensors are reconstructions.",
        "",
        "| Comparison | First differing stage |",
        "|---|---|",
    ]
    for name, comparison in report.get("comparisons", {}).items():
        rows.append(
            f"| {name} | {comparison['first_differing_stage'] or 'All compared tensors bitwise equal'} |"
        )
    for name, comparison in report.get("comparisons", {}).items():
        rows += [
            "",
            f"## {name}",
            "",
            "| Stage | Exact match | Different tensors |",
            "|---|---|---:|",
        ]
        for stage, detail in comparison["stages"].items():
            rows.append(
                f"| {stage} | {detail['bitwise_equal']} | {len(detail.get('different_tensors', {}))} |"
            )
        first = comparison["first_differing_stage"]
        if first:
            rows += [
                "",
                f"First differing tensor details ({first}):",
                "",
                "```json",
                json.dumps(comparison["stages"][first], indent=2),
                "```",
            ]
    rows += [
        "",
        "## Reconstructed losses",
        "",
        "| Case | Critic loss | Mean target |",
        "|---|---:|---:|",
    ]
    for name, case in report.get("traces", {}).items():
        rows.append(
            f"| {name} | {case['critic_loss']:.12g} | {case['target_mean']:.12g} |"
        )
    rows += [
        "",
        "Returning intermediate tensors can change XLA fusion/autotuning. A reconstruction may not reproduce the exact historical scalar loss.",
        "A frozen-input mismatch demonstrates execution variability in the traced step. Matching frozen inputs do not prove independently compiled original jobs were deterministic.",
        "The frozen control uses the stock case's post-collection models, replay and key, with fresh optimizer state and cloned model/RNG arrays.",
        "See report.json for per-tensor absolute errors and first differing indices; NPZ files contain full arrays without pickle.",
        "",
    ]
    (root / "report.md").write_text("\n".join(rows))


def run(args):
    import gymnasium as gym
    import jax
    import numpy as np
    from flax import nnx

    from dreamer.buffer import collect_new_rollouts, init_buffer
    from dreamer.checkpoints import load_checkpoint
    from dreamer.first_step_trace import (
        STAGES,
        TraceConfig,
        compare_snapshots,
        make_optimizers,
        model_state,
        trace_update,
        write_snapshot,
    )
    from wm.envs import CONTROLLER_TRAINING_ENV_SPEC

    jax.config.update("jax_platforms", args.backend)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "report.json").exists():
        raise FileExistsError(f"Choose a new output directory: {root}")
    report = {
        "status": "running",
        "scope": "Controlled first-step reconstruction, not a recovery of unsaved historical replay/targets.",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "devices": [str(device) for device in jax.devices()],
        "device_kinds": [device.device_kind for device in jax.devices()],
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "jax",
                "jaxlib",
                "flax",
                "gymnasium",
                "pygame-ce",
                "Box2D",
                "optax",
            )
        },
        "cases": {},
        "traces": {},
        "comparisons": {},
    }
    save_report(root, report)
    workspace = Path(__file__).resolve().parents[1]
    report["source_sha256"] = {
        name: hashlib.sha256((workspace / name).read_bytes()).hexdigest()
        for name in (
            "scripts/trace_dreamer_first_step.py",
            "src/dreamer/first_step_trace.py",
            "src/dreamer/buffer.py",
            "src/dreamer/actor_critic.py",
            "src/dreamer/imagine.py",
            "src/dreamer/wm/world_model.py",
            "src/dreamer/wm/rssm.py",
        )
    }
    prepared = {}
    started = perf_counter()
    try:
        # Compare actual initial arrays before spending time reconstructing collection.
        for label, checkpoint in (
            ("stock", args.stock_checkpoint),
            ("optimized", args.optimized_checkpoint),
        ):
            print(f"{label}: loading real initial checkpoint {checkpoint}", flush=True)
            models, metadata = load_checkpoint(checkpoint)
            if metadata.get("train_step") != 0:
                raise ValueError(
                    f"First-step tracing requires a step-zero checkpoint: {checkpoint}"
                )
            config = TraceConfig.from_metadata(metadata)
            seed = metadata["train_args"]["seed"]
            collection_key, key = jax.random.split(jax.random.key(seed))
            step_key, _ = jax.random.split(key)
            write_snapshot(root / label, "initial", model_state(models))
            prepared[label] = {
                "models": models,
                "metadata": metadata,
                "config": config,
                "collection_key": collection_key,
                "step_key": step_key,
            }
            report["cases"][label] = {
                "checkpoint": str(checkpoint.resolve()),
                "metadata": metadata,
                "effective_first_step": asdict(config),
                "collection_steps": args.collection_steps,
            }
        report["comparisons"]["actual_initial_checkpoints"] = compare_snapshots(
            root / "stock", root / "optimized", stages=("initial",)
        )
        save_report(root, report)
        print(
            "Initial checkpoint comparison:",
            report["comparisons"]["actual_initial_checkpoints"]["first_differing_stage"]
            or "bitwise equal",
            flush=True,
        )

        for label in ("stock", "optimized"):
            case = prepared[label]
            info, models = case["metadata"]["train_args"], case["models"]
            count = info["num_envs"]
            if args.collection_steps + 1 < case["config"].seq_length:
                raise ValueError(
                    "Collection must be at least sequence length minus one"
                )
            replay = init_buffer(
                capacity=info["buffer_size"],
                seq_length=args.collection_steps + 1,
                cfg=models["world_model"].cfg,
            )
            env_id = info["env_id"]
            spec = (
                CONTROLLER_TRAINING_ENV_SPEC
                if label == "optimized" and env_id == "CarRacing-v3"
                else env_id
            )
            print(
                f"{label}: reconstructing {count} × {args.collection_steps} transitions",
                flush=True,
            )
            envs = gym.make_vec(
                spec,
                num_envs=count,
                vectorization_mode="async",
                vector_kwargs={"context": "spawn"},
            )
            try:
                # Progress is outside all compiled model functions.
                original_step = envs.step
                steps = 0

                def step(actions, _step=original_step, _label=label):
                    nonlocal steps
                    result = _step(actions)
                    steps += 1
                    if steps % 200 == 0:
                        print(f"{_label}: collected {steps} vector steps", flush=True)
                    return result

                envs.step = step
                replay = collect_new_rollouts(
                    replay,
                    envs,
                    models["world_model"],
                    models["actor"],
                    count,
                    case["collection_key"],
                )
                jax.block_until_ready(replay)
            finally:
                envs.close()
            case["replay"] = replay
            write_snapshot(root / label, "post_collection", model_state(models))
            active = {
                name: getattr(replay, name)[:count]
                for name in (
                    "obs",
                    "acts",
                    "conts",
                    "rewards",
                    "model_state_hidden",
                    "model_state_latent",
                )
            }
            active.update(
                cursor=replay.cursor,
                full=replay.full,
                collection_key=case["collection_key"],
                step_key=case["step_key"],
            )
            write_snapshot(root / label, "replay", active)
            save_report(root, report)
            del active

        for label, source in (
            ("stock", "stock"),
            ("optimized", "optimized"),
            ("stock_repeat", "stock"),
        ):
            print(
                f"{label}: tracing one learner update (first call compiles)", flush=True
            )
            case = prepared[source]
            models = {
                name: nnx.clone(model, arrays=True)
                for name, model in case["models"].items()
            }
            optimizers = make_optimizers(models, case["config"])
            trace = trace_update(
                models,
                optimizers,
                case["replay"],
                case["config"],
                case["step_key"],
                case["metadata"]["train_args"]["num_envs"],
            )
            jax.block_until_ready(trace)
            for stage in STAGES[3:]:
                write_snapshot(root / label, stage, trace[stage])
            context = trace["sample"]["context"]
            times = np.asarray(context.time_indices[:, 1:])
            episodes = np.broadcast_to(np.asarray(context.batch_indices), times.shape)
            slots = episodes * case["replay"].episode_length + times
            _, counts = np.unique(slots, return_counts=True)
            report["traces"][label] = {
                "critic_loss": float(trace["critic_loss"]["loss"]),
                "target_mean": float(trace["critic_targets"]["targets"].mean()),
                "conflicting_replay_slots": int(np.count_nonzero(counts > 1)),
                "extra_replay_writes": int((counts - 1).sum()),
                "frozen_input_source": source,
            }
            print(label, report["traces"][label], flush=True)
            save_report(root, report)
            del trace, models, optimizers
            gc.collect()

        report["comparisons"]["stock_vs_optimized_reconstruction"] = compare_snapshots(
            root / "stock", root / "optimized"
        )
        report["comparisons"]["stock_frozen_input_repeat"] = compare_snapshots(
            root / "stock", root / "stock_repeat", stages=STAGES[3:]
        )
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["elapsed_s"] = perf_counter() - started
        save_report(root, report)
    for label, comparison in report["comparisons"].items():
        print(
            f"{label}: first difference = {comparison['first_differing_stage'] or 'none'}",
            flush=True,
        )
    print(
        f"Read {root / 'report.md'} and return report.json with the Slurm log",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-checkpoint", required=True, type=Path)
    parser.add_argument("--optimized-checkpoint", required=True, type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--collection-steps", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Control XLA GPU nondeterminism for a separate diagnostic run; may be slower or unsupported for some ops.",
    )
    arguments = parser.parse_args()
    if arguments.collection_steps < 1:
        parser.error("--collection-steps must be positive")
    for checkpoint in (arguments.stock_checkpoint, arguments.optimized_checkpoint):
        if not (checkpoint / "metadata.json").is_file():
            parser.error(f"Checkpoint metadata not found: {checkpoint}")
    if arguments.deterministic:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + " --xla_gpu_exclude_nondeterministic_ops=true"
        )
    if arguments.output_dir is None:
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        arguments.output_dir = (
            Path("experiments/dreamer/diagnostics")
            / f"first-step-{stamp}-{os.getpid()}"
        )
    run(arguments)
