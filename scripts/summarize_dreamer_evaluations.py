"""Publish mean/IQM rankings while the recent ARC checkpoint batches finish."""

import csv
import html
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT_FILES = ("r6hn3pff-latest-mps.json", "r6hn3pff.json", "f3bqo4fm.json", "tcdl3ruw.json")
INK, PAPER = "#343633", "#F7F4EB"


def write_atomic(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def collect_results(directory, inventory):
    allowed = {str(Path(row["checkpoint"]).resolve())
               for run in inventory["runs"].values() for row in run["snapshots"]}
    records = {}
    # Keep previously validated results if a worker file is briefly mid-write.
    previous = directory / "all-checkpoints.json"
    sources = []
    if previous.exists():
        sources.append(previous)
    sources.extend(directory / filename for filename in REPORT_FILES)
    for run_id in inventory["runs"]:
        sources.extend(sorted(directory.glob(f"{run_id}-resume-*.json")))
    for path in sources:
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if (report.get("num_episodes") != 64 or report.get("num_envs") != 8
                or report.get("max_steps") != 1000 or report.get("backend") != "mps"
                or report.get("episode_seeds") != list(range(500000, 500064))):
            continue
        for row in report.get("results", []):
            checkpoint = str(Path(row["checkpoint"]).resolve())
            if checkpoint not in allowed:
                continue
            values = np.asarray(row["returns"], dtype=np.float64)
            assert values.shape == (64,) and np.isfinite(values).all()
            np.testing.assert_allclose(values.mean(), row["mean"], atol=1e-8)
            np.testing.assert_allclose(np.median(values), row["median"], atol=1e-8)
            np.testing.assert_allclose(np.sort(values)[16:48].mean(), row["iqm"], atol=1e-8)
            run_id = Path(checkpoint).parent.name
            records[checkpoint] = {
                **row, "run_id": run_id, "label": inventory["runs"][run_id]["label"],
                "env_steps": row["checkpoint_metadata"]["env_steps"],
            }
    return sorted(records.values(), key=lambda row: (row["run_id"], row["train_step"]))


def ranked(records, metric):
    return sorted(records, key=lambda row: (-row[metric], row["run_id"], row["train_step"]))


def score_table(records, metric, count=10, markdown=True):
    columns = ["Rank", "Run", "Updates", "Mean", "IQM", "IQM 95% CI"]
    rows = []
    for index, row in enumerate(ranked(records, metric)[:count], 1):
        low, high = row["iqm_ci_95"]
        rows.append([str(index), row["run_id"], f"{row['train_step']:,}",
                     f"{row['mean']:.2f}", f"{row['iqm']:.2f}", f"[{low:.2f}, {high:.2f}]"])
    if markdown:
        return "\n".join(["| " + " | ".join(columns) + " |",
                          "| ---: | --- | ---: | ---: | ---: | --- |",
                          *["| " + " | ".join(row) + " |" for row in rows]])
    return ("<table><thead><tr>" + "".join(f"<th>{html.escape(value)}</th>" for value in columns)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row)
                                               + "</tr>" for row in rows) + "</tbody></table>")


def make_plot(records, inventory, destination, *, figure_size=(10, 7), ink=INK, paper=PAPER):
    plt.rcParams.update({"font.family": "Routed Gothic", "font.size": 11,
                         "figure.facecolor": paper, "axes.facecolor": paper,
                         "text.color": ink, "axes.labelcolor": ink,
                         "xtick.color": ink, "ytick.color": ink, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 1, figsize=figure_size, sharex=True)
    labels = {"r6hn3pff": "BUFFER 300 · SEED 20", "f3bqo4fm": "LOW LR · SEED 20",
              "tcdl3ruw": "LOW LR · SEED 75"}
    for run_id, linestyle, marker in zip(inventory["runs"], ("-", "--", ":"), ("o", "s", "^"), strict=True):
        values = [row for row in records if row["run_id"] == run_id]
        if not values:
            continue
        x = np.asarray([row["env_steps"] for row in values]) / 1e6
        for axis, metric in zip(axes, ("mean", "iqm"), strict=True):
            y = np.asarray([row[metric] for row in values])
            axis.plot(x, y, color=ink, linestyle=linestyle, linewidth=1.5, label=labels[run_id])
            best = int(np.argmax(y))
            axis.scatter(x[best], y[best], color=ink, edgecolor=paper, marker=marker, s=55, zorder=5)
            if metric == "iqm":
                bounds = np.asarray([row["iqm_ci_95"] for row in values])
                axis.fill_between(x, bounds[:, 0], bounds[:, 1], color=ink, alpha=0.09, linewidth=0)
    minimum = min([row["mean"] for row in records] + [-100])
    for axis, title in zip(axes, ("MEAN EPISODE RETURN", "INTERQUARTILE MEAN AND 95% CI"), strict=True):
        axis.set_title(title, pad=12)
        axis.set_ylabel("RETURN")
        axis.set_xlim(0, max(row["env_steps"] for run in inventory["runs"].values()
                             for row in run["snapshots"]) / 1e6)
        axis.set_ylim(minimum - 20, 1000)
        axis.spines[:].set_visible(False)
        axis.tick_params(length=0, pad=8)
        axis.grid(axis="y", color=ink, alpha=0.18, linestyle="--", linewidth=0.7)
        axis.set_axisbelow(True)
    axes[-1].set_xlabel("TRAINING ENVIRONMENT STEPS (MILLIONS)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=2)
    for extension in (".png", ".svg"):
        fig.savefig(destination.with_suffix(extension), dpi=250, bbox_inches="tight")
    return fig


def publish(directory, *, paused=False):
    inventory = json.loads((directory / "inventory.json").read_text())
    records = collect_results(directory, inventory)
    expected = sum(len(run["snapshots"]) for run in inventory["runs"].values())
    complete = len(records) == expected
    updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = {"num_episodes": 64, "num_envs": 8, "max_steps": 1000,
              "episode_seeds": list(range(500000, 500064)), "backend": "mps",
              "bootstrap_reps": 50000, "confidence_level": 0.95,
              "updated_at": updated, "complete": complete, "paused": paused, "expected_checkpoints": expected,
              "results": records}
    write_atomic(directory / "all-checkpoints.json", json.dumps(report, indent=2) + "\n")
    counts = {run: sum(row["run_id"] == run for row in records) for run in inventory["runs"]}
    progress = {"completed": len(records), "expected": expected, "complete": complete, "paused": paused,
                "updated_at": updated, "runs": counts}
    write_atomic(directory / "progress.json", json.dumps(progress, indent=2) + "\n")
    for metric in ("mean", "iqm"):
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["rank", "run", "variant", "train_step", "env_steps", "mean", "median", "iqm",
                         "iqm_ci_low", "iqm_ci_high", "checkpoint"])
        for index, row in enumerate(ranked(records, metric), 1):
            writer.writerow([index, row["run_id"], row["label"], row["train_step"], row["env_steps"],
                             row["mean"], row["median"], row["iqm"], *row["iqm_ci_95"], row["checkpoint"]])
        write_atomic(directory / f"leaderboard-{metric}.csv", stream.getvalue())
    settings = ["| Run | Variant | Buffer | Actor / critic LR | Seed | Complete |",
                "| --- | --- | ---: | --- | ---: | ---: |"]
    for run_id, run in inventory["runs"].items():
        settings.append(f"| {run_id} | {run['label']} | {run['buffer_size']} | "
                        f"{run['actor_lr']:g} / {run['critic_lr']:g} | {run['seed']} | "
                        f"{counts[run_id]}/{len(run['snapshots'])} |")
    status = "Complete" if complete else "Paused — ready to resume" if paused else "Provisional — evaluations still running"
    note = ("64 episodes per checkpoint; the same seeds 500000–500063, eight environments, "
            "and at most 1000 steps per episode. IQM intervals use 50,000 rliable bootstrap resamples. "
            "All three new runs use bfloat16 and the same MPS evaluation backend. "
            "Intervals describe episode uncertainty for a fixed checkpoint; ranks use point estimates.")
    body = (f"# Recent ARC checkpoint rankings\n\n{status}: **{len(records)}/{expected} checkpoints**. "
            f"Updated {updated}.\n\n{note}\n\n" + "\n".join(settings)
            + "\n\n## Highest mean returns\n\n" + score_table(records, "mean")
            + "\n\n## Highest IQM returns\n\n" + score_table(records, "iqm")
            + "\n\n![Mean and IQM learning curves](learning-curves.png)\n\n"
            "Markers show the best completed checkpoint within each run for each metric. "
            "Unfinished checkpoints are not ranked. Older CPU evaluations are kept separately.\n")
    write_atomic(directory / "README.md", body)
    if records:
        plt.close(make_plot(records, inventory, directory / "learning-curves"))
    page = ("<!doctype html><html><meta charset='utf-8'>"
            + ("<meta http-equiv='refresh' content='60'>" if not complete and not paused else "")
            + "<title>ARC checkpoint rankings</title><style>"
            "@font-face{font-family:Routed;src:url('../../policy-gifs/fonts/routed-gothic.ttf')}"
            f"body{{background:{PAPER};color:{INK};font:17px Routed,monospace;max-width:1100px;margin:32px auto;padding:0 24px}}"
            "h1,h2{font-weight:400}p{line-height:1.5}table{border-collapse:collapse;width:100%;margin:24px 0}"
            "td,th{text-align:right;padding:8px;border-bottom:1px solid #34363330}td:nth-child(2),th:nth-child(2){text-align:left}"
            "a{color:inherit}img{width:100%;height:auto}</style>"
            f"<h1>ARC CHECKPOINT RANKINGS — {len(records)}/{expected}</h1><p>{html.escape(status)}. "
            f"Updated {html.escape(updated)}.</p><p>{html.escape(note)}</p>"
            "<p><a href='README.md'>RUN SETTINGS</a> · <a href='leaderboard-mean.csv'>MEAN CSV</a> · "
            "<a href='leaderboard-iqm.csv'>IQM CSV</a> · <a href='all-checkpoints.json'>RAW RETURNS</a></p>"
            "<h2>HIGHEST MEAN RETURNS</h2>" + score_table(records, "mean", markdown=False)
            + "<h2>HIGHEST IQM RETURNS</h2>" + score_table(records, "iqm", markdown=False)
            + "<img src='learning-curves.png' alt='Mean and IQM returns over training'>"
            "<p>Markers show the best completed checkpoint within each run. The page refreshes while evaluations run.</p></html>")
    write_atomic(directory / "index.html", page)
    return progress


def main(directory: Path, watch: bool = False, interval: int = 30, paused: bool = False):
    if watch:
        (directory / "report-updater.pid").write_text(str(os.getpid()) + "\n")
    previous = None
    while True:
        progress = publish(directory, paused=paused)
        if progress["completed"] != previous:
            print(json.dumps(progress), flush=True)
            previous = progress["completed"]
        if not watch or progress["complete"] or paused:
            break
        time.sleep(interval)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
