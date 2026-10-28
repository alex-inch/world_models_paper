"""Diagnose ARC collection performance; see docs/dreamer_benchmark.md.

JAX/model imports stay inside parent-only functions. Environment workers use
spawn, as train_dreamer.py does. No W&B runs or checkpoints are written.
"""

import argparse
import cProfile
import datetime
import functools
import gc
import importlib.metadata
import json
import os
import platform
import pstats
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter, process_time, time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.envs.box2d.car_racing import CarRacing

RENDERERS = ("stock", "cached_font", "optimized", "no_render")


def describe(values):
    """All durations are seconds; empty series retain an explicit zero count."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "total_s": 0.0}
    return {
        "count": len(values),
        "total_s": float(values.sum()),
        "mean_ms": float(values.mean() * 1000),
        "p50_ms": float(np.percentile(values, 50) * 1000),
        "p95_ms": float(np.percentile(values, 95) * 1000),
        "p99_ms": float(np.percentile(values, 99) * 1000),
        "max_ms": float(values.max() * 1000),
    }


def affinity():
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


def scratch_root():
    # TMPDIR can point at shared storage on HPC systems; prefer the node's /tmp.
    return os.environ.get("SLURM_TMPDIR") or (
        "/tmp" if sys.platform == "linux" else tempfile.gettempdir()
    )


def timed_call(samples, name, function, finish=None):
    def call(*args, **kwargs):
        started = perf_counter()
        result = function(*args, **kwargs)
        if finish is not None:
            finish(result)
        samples.setdefault(name, []).append(perf_counter() - started)
        return result

    return call


class TimedCarRacing(gym.Wrapper):
    """Record worker timings without sending telemetry through the step pipe.

    Timings nest: render includes road/font/image; step includes render. The
    non-render remainder includes car dynamics, Box2D, and reward bookkeeping.
    """

    def __init__(self, renderer):
        if renderer == "optimized":
            from wm.envs import ControllerTrainingCarRacing

            env = ControllerTrainingCarRacing()
        else:
            env = CarRacing()
        super().__init__(env)
        self.renderer = renderer
        self.samples = {}
        self.font_cache = {}
        self.benchmark_reset()
        for method, name in (
            ("_render_road", "road"),
            ("_render_indicators", "dashboard"),
            ("_create_image_array", "image_conversion"),
        ):
            original = getattr(env, method)

            def timed(*args, _fn=original, _name=name, **kwargs):
                started = perf_counter()
                result = _fn(*args, **kwargs)
                self.record(_name, perf_counter() - started)
                return result

            setattr(env, method, timed)

        render = env._render

        def timed_render(mode):
            started = perf_counter()
            original_font = pygame.font.Font

            def font(*args, **kwargs):
                key = (args, tuple(sorted(kwargs.items())))
                if renderer == "cached_font" and key in self.font_cache:
                    return self.font_cache[key]
                before = perf_counter()
                value = original_font(*args, **kwargs)
                self.record("font_load", perf_counter() - before)
                if renderer == "cached_font":
                    self.font_cache[key] = value
                return value

            pygame.font.Font = font
            try:
                result = (
                    np.zeros((96, 96, 3), dtype=np.uint8)
                    if renderer == "no_render" and mode == "state_pixels"
                    else render(mode)
                )
            finally:
                pygame.font.Font = original_font
            self.record("render", perf_counter() - started)
            return result

        env._render = timed_render

    def record(self, name, seconds):
        self.samples.setdefault(name, []).append(seconds)
        if name == "render":
            self._render_seconds += seconds

    def step(self, action):
        before_render = self.render_seconds
        started = perf_counter()
        result = self.env.step(action)
        elapsed = perf_counter() - started
        self.record("step", elapsed)
        self.record(
            "non_render", max(0.0, elapsed - (self.render_seconds - before_render))
        )
        return result

    @property
    def render_seconds(self):
        return self._render_seconds

    def reset(self, *, seed=None, options=None):
        started = perf_counter()
        result = self.env.reset(seed=seed, options=options)
        self.record("reset", perf_counter() - started)
        return result

    def benchmark_reset(self):
        self.samples = {}
        self._render_seconds = 0.0
        self.wall_started = perf_counter()
        self.cpu_started = process_time()

    def benchmark_stats(self):
        wall = perf_counter() - self.wall_started
        cpu = process_time() - self.cpu_started
        status = Path("/proc/self/status")
        threads = None
        if status.exists():
            threads = next(
                (
                    line.split(":", 1)[1].strip()
                    for line in status.read_text().splitlines()
                    if line.startswith("Threads:")
                ),
                None,
            )
        return {
            "pid": os.getpid(),
            "affinity": affinity(),
            "threads": threads,
            "wall_s": wall,
            "cpu_s": cpu,
            "cpu_wall_ratio": cpu / wall,
            "timings": {name: describe(xs) for name, xs in self.samples.items()},
            "samples_s": self.samples,
        }


def make_env(renderer):
    return gym.wrappers.TimeLimit(TimedCarRacing(renderer), max_episode_steps=1000)


def vector_env(renderer, count):
    return gym.vector.AsyncVectorEnv(
        [functools.partial(make_env, renderer) for _ in range(count)], context="spawn"
    )


class Report:
    def __init__(self, directory, args):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "results.json").exists() or (
            directory / "results.jsonl"
        ).exists():
            raise FileExistsError(f"Choose a new output directory: {directory}")
        self.data = {
            "arguments": vars(args) | {"output_dir": str(directory)},
            "metadata": metadata(),
            "cases": [],
        }
        self.save()

    def add(self, kind, **values):
        row = {"kind": kind, "finished_unix_s": time(), **values}
        self.data["cases"].append(row)
        with (self.directory / "results.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")
        self.save()
        label = "/".join(
            str(row[k]) for k in ("renderer", "num_envs", "mode", "repeat") if k in row
        )
        print(f"{kind:12s} {label:35s} {row.get('wall_s', 0):9.3f} s", flush=True)

    def save(self):
        path = self.directory / "results.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2))
        temporary.replace(path)
        rows = [
            "# Dreamer collection benchmark",
            "",
            "Durations exclude startup/warmup unless stated. Compare equal step counts.",
            "",
        ]
        rows += comparison_rows(self.data["cases"])
        rows += [
            "",
            "## Individual cases",
            "",
            "| Case | Renderer | Envs | Mode | Repeat | Seconds | Transitions/s |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
        for case in self.data["cases"]:
            rows.append(
                f"| {case['kind']} | {case.get('renderer', '')} | {case.get('num_envs', '')} "
                f"| {case.get('mode', '')} | {case.get('repeat', '')} "
                f"| {case.get('wall_s', 0):.3f} | {case.get('transitions_per_s', 0):.2f} |"
            )
        rows += [
            "",
            "## Production collection stages",
            "",
            "Residual includes action/carry synchronization and host staging.",
            "",
            "| Renderer | Repeat | Stage | Seconds | % of collection |",
            "|---|---:|---|---:|---:|",
        ]
        for case in self.data["cases"]:
            if case["kind"] == "collection":
                stages = {
                    name: stats["total_s"] for name, stats in case["stages"].items()
                }
                stages["residual"] = case["residual_s"]
                for name, seconds in stages.items():
                    rows.append(
                        f"| {case['renderer']} | {case['repeat']} | {name} "
                        f"| {seconds:.3f} | {100 * seconds / case['wall_s']:.1f}% |"
                    )
        rows += [
            "",
            "See results.json for p50/p95/p99, worker CPU/affinity, stage timings,",
            "first-episode usable fractions, and backend metadata. Worker timings nest;",
            "do not add render to road/font/image times. See docs/dreamer_benchmark.md.",
        ]
        (self.directory / "report.md").write_text("\n".join(rows) + "\n")


def comparison_rows(cases):
    """Compare median elapsed times only within matched workloads."""
    groups = {}
    for case in cases:
        key = (
            case["kind"],
            case.get("renderer"),
            case.get("num_envs"),
            case.get("mode"),
        )
        groups.setdefault(key, []).append(case["wall_s"])
    medians = {key: float(np.median(values)) for key, values in groups.items()}
    rows = [
        "## Measured comparisons",
        "",
        "Speedup = baseline seconds / changed seconds.",
        "",
        "| Change | Scope | Speedup |",
        "|---|---|---:|",
    ]
    for (kind, renderer, count, mode), seconds in medians.items():
        baseline = None
        label = ""
        if kind in ("environment", "collection") and renderer != "stock":
            baseline = medians.get((kind, "stock", count, mode))
            label = f"stock → {renderer} ({count} envs)"
        elif kind == "overlap" and mode == "overlap":
            baseline = medians.get((kind, renderer, count, "serial"))
            label = f"serial → overlap ({renderer}, {count} envs)"
        elif kind == "font" and mode != "package_lookup":
            baseline = medians.get((kind, renderer, count, "package_lookup"))
            label = f"font lookup → {mode}"
        elif kind == "learning" and mode == "sync_at_end":
            baseline = medians.get((kind, renderer, count, "sync_each_update"))
            label = "per-update sync → final sync"
        if baseline is not None and seconds > 0:
            rows.append(f"| {label} | {kind} | {baseline / seconds:.2f}× |")
    return rows


def metadata():
    packages = {}
    for name in ("jax", "jaxlib", "flax", "gymnasium", "pygame-ce", "opencv-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "affinity": affinity(),
        "packages": packages,
        "pygame_path": pygame.__file__,
        "cwd": str(Path.cwd()),
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("SLURM_")
            or k
            in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "XLA_FLAGS",
                "JAX_PLATFORMS",
                "CUDA_VISIBLE_DEVICES",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "TMPDIR",
            )
        },
    }
    for name, command in (
        ("lscpu", ["lscpu"]),
        (
            "nvidia_smi",
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv",
            ],
        ),
        ("git_revision", ["git", "rev-parse", "HEAD"]),
        ("git_status", ["git", "status", "--short"]),
        (
            "filesystem",
            ["df", "-T", str(Path(pygame.__file__).parent), scratch_root()],
        ),
    ):
        try:
            proc = subprocess.run(
                command, capture_output=True, text=True, timeout=10, check=False
            )
            result[name] = proc.stdout.strip() or proc.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = str(error)
    return result


def warm_env(envs, args, seed):
    started = perf_counter()
    envs.reset(seed=seed)
    actions = np.tile(np.array([0, 0.5, 0], np.float32), (envs.num_envs, 1))
    for _ in range(args.warmup):
        envs.step(actions)
    return perf_counter() - started


def environment_cases(args, report, counts):
    for count in counts:
        for renderer in args.renderers:
            started = perf_counter()
            envs = vector_env(renderer, count)
            startup = perf_counter() - started
            try:
                warmup = warm_env(envs, args, args.seed)
                actions = np.tile(np.array([0, 0.5, 0], np.float32), (count, 1))
                for repeat in range(args.repeats):
                    started = perf_counter()
                    envs.reset(seed=args.seed + repeat * 1009)
                    reset_s = perf_counter() - started
                    envs.call("benchmark_reset")
                    times = []
                    started = perf_counter()
                    for _ in range(args.steps):
                        before = perf_counter()
                        envs.step(actions)
                        times.append(perf_counter() - before)
                    wall = perf_counter() - started
                    report.add(
                        "environment",
                        renderer=renderer,
                        num_envs=count,
                        repeat=repeat,
                        wall_s=wall,
                        steps=args.steps,
                        transitions_per_s=count * args.steps / wall,
                        startup_s=startup,
                        warmup_s=warmup,
                        reset_s=reset_s,
                        vector_step=describe(times),
                        vector_step_samples_s=times,
                        workers=envs.call("benchmark_stats"),
                    )
            finally:
                envs.close()


def font_cases(args, report):
    pygame.font.init()
    default = pygame.font.get_default_font()
    package_font = Path(pygame.__file__).parent / default
    if not package_font.exists():
        raise FileNotFoundError(f"Cannot find Pygame's default font: {package_font}")
    scratch = scratch_root()
    with tempfile.TemporaryDirectory(prefix="dreamer-font-", dir=scratch) as directory:
        local_font = Path(directory) / default
        shutil.copyfile(package_font, local_font)
        cached = pygame.font.Font(str(package_font), 42)
        for repeat in range(args.repeats):
            for mode, factory in (
                ("package_lookup", lambda: pygame.font.Font(default, 42)),
                ("package_absolute", lambda: pygame.font.Font(str(package_font), 42)),
                ("scratch_absolute", lambda: pygame.font.Font(str(local_font), 42)),
                ("cached", lambda: cached),
            ):
                loads, renders = [], []
                for _ in range(args.font_iterations):
                    before = perf_counter()
                    font = factory()
                    loads.append(perf_counter() - before)
                    before = perf_counter()
                    font.render("0123", True, (255, 255, 255), (0, 0, 0))
                    renders.append(perf_counter() - before)
                report.add(
                    "font",
                    mode=mode,
                    repeat=repeat,
                    wall_s=sum(loads) + sum(renders),
                    load=describe(loads),
                    render=describe(renders),
                    package_font=str(package_font),
                    scratch=str(local_font),
                )


def cpu_profile(args, report):
    env = make_env("stock")
    try:
        env.reset(seed=args.seed)
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(args.profile_steps):
            env.step(np.array([0, 0.5, 0], np.float32))
        profiler.disable()
        profiler.dump_stats(str(report.directory / "stock-render.prof"))
        with (report.directory / "stock-render.txt").open("w") as file:
            pstats.Stats(profiler, stream=file).sort_stats("cumulative").print_stats(60)
    finally:
        env.close()


def model_cases(args, report):
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from dreamer.actor_critic import Actor
    from dreamer.buffer import (
        add_episodes,
        choose_action,
        collect_new_rollouts,
        init_buffer,
    )
    from dreamer.wm import DreamerConfig, DreamerWM
    from wm.utils.obs import crop_obs, normalise_obs

    jax.config.update("jax_platforms", args.backend)
    devices = jax.devices()
    report.data["metadata"]["jax_devices"] = [str(device) for device in devices]
    report.data["metadata"]["jax_device_kind"] = [
        device.device_kind for device in devices
    ]
    cfg = DreamerConfig()
    if args.checkpoint:
        from dreamer.checkpoints import load_checkpoint

        models, info = load_checkpoint(args.checkpoint)
        wm, actor = models["world_model"], models["actor"]
        cfg = wm.cfg
        del models
        report.data["metadata"]["checkpoint"] = info
    else:
        wm = DreamerWM(cfg, rngs=nnx.Rngs(args.seed + 1))
        actor = Actor(cfg, rngs=nnx.Rngs(args.seed + 2))
    report.save()
    count = args.policy_envs
    host_image = np.zeros((count, 64, 64, 3), np.float32)
    image = jax.device_put(host_image)
    carry = wm.rssm.initialize_state(count)
    action = jnp.zeros((count, cfg.action_dim))
    first = jnp.zeros(count, dtype=bool)
    key = jax.random.key(args.seed)
    started = perf_counter()
    for _ in range(max(2, args.warmup)):
        key, subkey = jax.random.split(key)
        carry, action = choose_action(wm, actor, carry, image, action, first, subkey)
        jax.block_until_ready((carry, action))
    report.add("policy_warmup", num_envs=count, wall_s=perf_counter() - started)
    fresh = jax.jit(lambda tree: jax.tree.map(lambda x: x + 1, tree))
    for repeat in range(args.repeats):
        for mode in ("policy_compute", "image_h2d", "action_d2h", "carry_d2h"):
            times = []
            for _ in range(args.steps):
                key, subkey = jax.random.split(key)
                jax.block_until_ready(subkey)
                # New device results avoid measuring JAX's cached host copies.
                if mode in ("action_d2h", "carry_d2h"):
                    transferred = fresh(action if mode == "action_d2h" else carry)
                    jax.block_until_ready(transferred)
                before = perf_counter()
                if mode == "policy_compute":
                    carry, action = choose_action(
                        wm, actor, carry, image, action, first, subkey
                    )
                    jax.block_until_ready((carry, action))
                elif mode == "image_h2d":
                    jax.block_until_ready(jax.device_put(host_image))
                elif mode == "action_d2h":
                    np.asarray(transferred)
                else:
                    jax.device_get(transferred)
                times.append(perf_counter() - before)
            report.add(
                "policy",
                mode=mode,
                num_envs=count,
                repeat=repeat,
                wall_s=sum(times),
                timing=describe(times),
            )

    # Test scheduling before the longer full-episode collection runs, so the
    # low-complexity overlap result is available early even on a very slow node.
    if not args.skip_learning:
        learning_cases(args, report, wm, actor, cfg)

    # Call the actual production collector, with wrappers around its Python
    # stages. NumPy conversion waits remain in the residual, not policy_dispatch.
    from unittest.mock import patch

    import dreamer.buffer as buffer_module

    for renderer in (r for r in args.renderers if r != "no_render"):
        envs = vector_env(renderer, count)
        try:
            warm_env(envs, args, args.seed)
            # Warm the complete production path, including reset-key generation,
            # initial flags, CPU preprocessing, and implicit NumPy conversions.
            # Warming only choose_action would favour every renderer after stock.
            started = perf_counter()
            tiny = init_buffer(capacity=count, seq_length=2, cfg=cfg)
            tiny = collect_new_rollouts(
                tiny, envs, wm, actor, count, jax.random.key(args.seed)
            )
            jax.block_until_ready(tiny)
            del tiny
            report.add(
                "collect_warmup",
                renderer=renderer,
                num_envs=count,
                wall_s=perf_counter() - started,
            )
            for repeat in range(args.repeats):
                nnx.reseed(wm, default=args.seed + 1)
                replay = init_buffer(
                    capacity=max(args.replay_capacity, count),
                    seq_length=args.collection_steps + 1,
                    cfg=cfg,
                )
                jax.block_until_ready(replay)
                stages = {}

                # Warm the insertion compilation outside the measured call.
                length = args.collection_steps + 1
                small = np.zeros((count, length), np.float32)
                jax.block_until_ready(
                    add_episodes(
                        replay,
                        np.zeros((count, length, 64, 64, 3), np.float32),
                        np.zeros((count, length, cfg.action_dim), np.float32),
                        small,
                        small,
                        np.zeros((count, length, cfg.wm_hidden_dim), np.float32),
                        np.zeros((count, length, cfg.latent_dim), np.float32),
                    )
                )
                envs.call("benchmark_reset")
                with (
                    patch.object(
                        buffer_module,
                        "choose_action",
                        timed_call(stages, "policy_dispatch", choose_action),
                    ),
                    patch.object(
                        buffer_module,
                        "crop_obs",
                        timed_call(stages, "crop_resize", crop_obs),
                    ),
                    patch.object(
                        buffer_module,
                        "normalise_obs",
                        timed_call(stages, "normalize", normalise_obs),
                    ),
                    patch.object(
                        buffer_module,
                        "add_episodes",
                        timed_call(
                            stages, "replay_insert", add_episodes, jax.block_until_ready
                        ),
                    ),
                    patch.object(
                        envs, "step", timed_call(stages, "env_step", envs.step)
                    ),
                    patch.object(
                        envs, "reset", timed_call(stages, "env_reset", envs.reset)
                    ),
                ):
                    started = perf_counter()
                    replay = collect_new_rollouts(
                        replay,
                        envs,
                        wm,
                        actor,
                        count,
                        jax.random.key(args.seed + repeat * 1009),
                    )
                    jax.block_until_ready(replay)
                    wall = perf_counter() - started
                conts = np.asarray(replay.conts[:count])
                usable = np.cumprod(conts[:, :-1], axis=1).sum(axis=1)
                accounted = sum(sum(xs) for xs in stages.values())
                report.add(
                    "collection",
                    renderer=renderer,
                    num_envs=count,
                    repeat=repeat,
                    wall_s=wall,
                    steps=args.collection_steps,
                    transitions_per_s=count * args.collection_steps / wall,
                    first_episode_steps=usable.tolist(),
                    first_episode_usable_fraction=float(
                        usable.sum() / (count * args.collection_steps)
                    ),
                    stages={name: describe(xs) for name, xs in stages.items()},
                    residual_s=wall - accounted,
                    workers=envs.call("benchmark_stats"),
                )
                del replay
                gc.collect()
        finally:
            envs.close()


def learning_cases(args, report, wm, actor, cfg):
    """Use the real learner on synthetic replay, with frozen acting models."""
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from train_dreamer import TrainArgs, model_learning_step

    from dreamer.actor_critic import Critic
    from dreamer.buffer import choose_action, init_buffer
    from wm.utils.obs import crop_obs, normalise_obs

    learner_wm, learner_actor = (
        nnx.clone(wm, arrays=True),
        nnx.clone(actor, arrays=True),
    )
    critic = Critic(cfg, rngs=nnx.Rngs(args.seed + 3))
    target = nnx.clone(critic, arrays=True)
    train_args = TrainArgs(
        name="benchmark",
        cfg=cfg,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
    )
    optimizers = [
        nnx.Optimizer(
            model,
            tx=optax.chain(
                optax.adaptive_grad_clip(0.3), optax.adamw(lr, eps=train_args.adamw_eps)
            ),
            wrt=nnx.Param,
        )
        for model, lr in (
            (learner_wm, train_args.wm_lr),
            (learner_actor, train_args.actor_lr),
            (critic, train_args.actor_lr),
        )
    ]
    replay = init_buffer(capacity=8, seq_length=args.seq_length + 1, cfg=cfg)
    # Valid full episodes. Synthetic data isolates compute; this is not a learning-quality experiment.
    replay = replay.replace(
        full=jnp.asarray(True), conts=jnp.ones_like(replay.conts).at[:, -1].set(0)
    )
    key = jax.random.key(args.seed)

    def update():
        nonlocal key, replay
        key, subkey = jax.random.split(key)
        metrics, replay = model_learning_step(
            learner_wm,
            optimizers[0],
            learner_actor,
            optimizers[1],
            critic,
            optimizers[2],
            target,
            replay,
            train_args,
            subkey,
        )
        return metrics

    started = perf_counter()
    for _ in range(3):
        metrics = update()
        float(metrics["ac/targets"].mean())
    report.add("learn_warmup", wall_s=perf_counter() - started)
    for repeat in range(args.repeats):
        for mode in ("sync_each_update", "sync_at_end"):
            started = perf_counter()
            for _ in range(args.learning_steps):
                metrics = update()
                if mode == "sync_each_update":
                    float(metrics["ac/targets"].mean())
            jax.block_until_ready((metrics, replay))
            report.add(
                "learning",
                mode=mode,
                repeat=repeat,
                updates=args.learning_steps,
                wall_s=perf_counter() - started,
            )

    count = args.policy_envs
    for renderer in (r for r in args.renderers if r in ("stock", "optimized")):
        envs = vector_env(renderer, count)
        try:
            warm_env(envs, args, args.seed)
            for repeat in range(args.repeats):
                # Alternate order to expose drift/contention rather than always favouring the second run.
                modes = (
                    ("serial", "overlap") if repeat % 2 == 0 else ("overlap", "serial")
                )
                for mode in modes:
                    nnx.reseed(wm, default=args.seed + 1)
                    obs, _ = envs.reset(seed=args.seed + repeat * 1009)
                    carry = wm.rssm.initialize_state(count)
                    previous = jnp.zeros((count, cfg.action_dim))
                    policy_key = jax.random.key(args.seed)
                    first = jnp.ones(count, dtype=bool)
                    updates = 0
                    reset_pending = np.zeros(count, dtype=bool)
                    jax.block_until_ready((carry, previous))
                    envs.call("benchmark_reset")
                    started = perf_counter()
                    for step in range(args.overlap_steps):
                        policy_key, subkey = jax.random.split(policy_key)
                        pixels = normalise_obs(crop_obs(obs)) - 0.5
                        carry, previous = choose_action(
                            wm,
                            actor,
                            carry,
                            jnp.asarray(pixels),
                            previous,
                            first,
                            subkey,
                        )
                        actions = np.asarray(previous)
                        if mode == "overlap":
                            envs.step_async(actions)
                        else:
                            result = envs.step(actions)
                        due = int((step + 1) * args.updates_per_vector_step + 1e-9)
                        while updates < due:
                            metrics = update()
                            # Match the current learner's per-update host synchronisation.
                            float(metrics["ac/targets"].mean())
                            updates += 1
                        if mode == "overlap":
                            result = envs.step_wait()
                        obs, _, terminated, truncated, _ = result
                        # NEXT_STEP autoreset: reset the carry on the reset observation,
                        # which will arrive one iteration after the terminal observation.
                        first = jnp.asarray(reset_pending)
                        reset_pending = terminated | truncated
                    jax.block_until_ready(replay)
                    wall = perf_counter() - started
                    report.add(
                        "overlap",
                        renderer=renderer,
                        mode=mode,
                        repeat=repeat,
                        num_envs=count,
                        steps=args.overlap_steps,
                        updates=updates,
                        wall_s=wall,
                        transitions_per_s=count * args.overlap_steps / wall,
                        workers=envs.call("benchmark_stats"),
                    )
        finally:
            envs.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backend", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--env-counts", nargs="+", type=int, default=[1, 8, 12, 16])
    parser.add_argument(
        "--renderers", nargs="+", choices=RENDERERS, default=list(RENDERERS)
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Vector steps per environment-only case."
    )
    parser.add_argument("--collection-steps", type=int, default=1000)
    parser.add_argument("--policy-envs", type=int, default=8)
    parser.add_argument("--replay-capacity", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--font-iterations", type=int, default=100)
    parser.add_argument("--profile-steps", type=int, default=100)
    parser.add_argument("--learning-steps", type=int, default=50)
    parser.add_argument("--overlap-steps", type=int, default=100)
    parser.add_argument("--updates-per-vector-step", type=float, default=1.6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--skip-model", action="store_true", help="CPU environment/font tests only."
    )
    parser.add_argument("--skip-learning", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "steps",
        "collection_steps",
        "policy_envs",
        "replay_capacity",
        "repeats",
        "font_iterations",
        "profile_steps",
        "learning_steps",
        "overlap_steps",
        "batch_size",
        "seq_length",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if min(args.env_counts) < 1 or args.warmup < 0 or args.updates_per_vector_step <= 0:
        parser.error(
            "Environment counts and update ratio must be positive; warmup must be non-negative"
        )
    if args.checkpoint:
        args.checkpoint = str(args.checkpoint.resolve())
    if args.output_dir is None:
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = (
            Path("experiments/dreamer/benchmarks") / f"{stamp}-{os.getpid()}"
        )
    return args


def main():
    args = parse_args()
    directory = args.output_dir.resolve()
    report = Report(directory, args)
    print(f"Saving results to {directory}", flush=True)
    try:
        font_cases(args, report)
        environment_cases(args, report, [args.policy_envs])
        cpu_profile(args, report)
        if not args.skip_model:
            model_cases(args, report)
        environment_cases(
            args, report, [n for n in args.env_counts if n != args.policy_envs]
        )
        report.data["status"] = "complete"
    except BaseException as error:
        report.data["status"] = "failed"
        report.data["error"] = repr(error)
        raise
    finally:
        report.save()


if __name__ == "__main__":
    main()
