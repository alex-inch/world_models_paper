import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

script = Path(__file__).resolve().parents[1] / "scripts/benchmark_dreamer.py"
spec = importlib.util.spec_from_file_location("benchmark_dreamer", script)
assert spec is not None and spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_cached_font_preserves_observations_and_nested_timing_accounting():
    stock = benchmark.TimedCarRacing("stock")
    cached = benchmark.TimedCarRacing("cached_font")
    try:
        stock.reset(seed=17)
        cached.reset(seed=17)
        stock.benchmark_reset()
        cached.benchmark_reset()
        for _ in range(8):
            action = np.array([0.1, 0.5, 0], np.float32)
            reference = stock.step(action)
            result = cached.step(action)
            np.testing.assert_array_equal(reference[0], result[0])
            assert reference[1:] == result[1:]

        stats = stock.benchmark_stats()["timings"]
        assert stats["font_load"]["count"] == 8
        assert "font_load" not in cached.benchmark_stats()["timings"]
        assert stats["non_render"]["total_s"] > 0
        assert stats["render"]["total_s"] + stats["non_render"][
            "total_s"
        ] == pytest.approx(stats["step"]["total_s"])
        assert stats["road"]["total_s"] < stats["render"]["total_s"]
    finally:
        stock.close()
        cached.close()


def test_comparisons_match_scope_and_worker_count_and_use_medians():
    cases = [
        {"kind": "collection", "renderer": "stock", "num_envs": 8, "wall_s": 10},
        {"kind": "collection", "renderer": "stock", "num_envs": 8, "wall_s": 14},
        {"kind": "collection", "renderer": "optimized", "num_envs": 8, "wall_s": 4},
        {"kind": "collection", "renderer": "optimized", "num_envs": 8, "wall_s": 8},
        {"kind": "environment", "renderer": "optimized", "num_envs": 8, "wall_s": 1},
        {"kind": "collection", "renderer": "optimized", "num_envs": 16, "wall_s": 1},
        {
            "kind": "overlap",
            "renderer": "stock",
            "num_envs": 8,
            "mode": "serial",
            "wall_s": 9,
        },
        {
            "kind": "overlap",
            "renderer": "stock",
            "num_envs": 8,
            "mode": "overlap",
            "wall_s": 6,
        },
    ]
    rows = benchmark.comparison_rows(cases)
    comparisons = [row for row in rows if "×" in row]
    assert comparisons == [
        "| stock → optimized (8 envs) | collection | 2.00× |",
        "| serial → overlap (stock, 8 envs) | overlap | 1.50× |",
    ]


def test_report_keeps_completed_results_and_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "metadata", dict)
    args = SimpleNamespace(output_dir=tmp_path)
    report = benchmark.Report(tmp_path, args)
    report.add(
        "environment", renderer="stock", num_envs=1, wall_s=1, transitions_per_s=3
    )
    import json

    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["cases"][0]["finished_unix_s"] > 0
    assert saved["cases"][0]["wall_s"] == 1
    assert "stock" in (tmp_path / "report.md").read_text()
    with pytest.raises(FileExistsError):
        benchmark.Report(tmp_path, args)


def test_rejects_invalid_benchmark_sizes():
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--steps", "0"])
