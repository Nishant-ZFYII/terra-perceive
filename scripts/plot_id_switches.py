#!/usr/bin/env python3
"""Plot cumulative ID-switch count over frames for one or more runs.

Reads `id_switches.csv` from each run directory and overlays them on a
single matplotlib figure. The headline supplementary plot for Ablation A
(greedy vs Munkres on crossings).

Each run contributes one line in the legend; the line steps up by 1 at
every frame where an ID-switch was logged.

Usage:
    python scripts/plot_id_switches.py \\
        --run results_m4/ablation_a/greedy   --label "Greedy" \\
        --run results_m4/ablation_a/munkres  --label "Munkres" \\
        --out results_m4/ablation_a/id_switches.png

`--run` and `--label` repeat as a pair. The N-th --run pairs with the N-th
--label. If --label is omitted for a run, the directory name is used.

CSV schema (id_switches.csv):
    frame_id,gt_track_id,prev_track_id,curr_track_id
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_switches(run_dir: Path) -> Tuple[List[int], List[int]]:
    """Returns (frame_ids_with_a_switch, cumulative_count)."""
    csv_path = run_dir / "id_switches.csv"
    if not csv_path.exists():
        return [], []
    frames: List[int] = []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            frames.append(int(row["frame_id"]))
    frames.sort()
    cum = list(range(1, len(frames) + 1))
    return frames, cum


def plot_runs(runs: List[Tuple[Path, str]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    # Compute the total frame range across runs so we can extend each line
    # horizontally to the right edge.
    all_frames: List[int] = []
    series = []
    for run_dir, label in runs:
        fs, cs = load_switches(run_dir)
        series.append((label, fs, cs))
        all_frames.extend(fs)

    # Also peek at tracks.csv frame range so the x-axis covers the whole run,
    # not just up to the last switch.
    max_frame_seen = 0
    for run_dir, _ in runs:
        tracks_csv = run_dir / "tracks.csv"
        if tracks_csv.exists():
            with tracks_csv.open() as f:
                r = csv.DictReader(f)
                for row in r:
                    max_frame_seen = max(max_frame_seen, int(row["frame_id"]))
    if max_frame_seen == 0 and all_frames:
        max_frame_seen = max(all_frames)

    for label, fs, cs in series:
        if not fs:
            # Run had zero ID-switches — flat zero line across the whole run.
            ax.step([0, max_frame_seen], [0, 0], where="post", label=f"{label} (0)")
            continue
        # Step plot starting from (0, 0), with steps at each switch frame.
        xs = [0] + fs + [max_frame_seen]
        ys = [0] + cs + [cs[-1]]
        ax.step(xs, ys, where="post", label=f"{label} ({cs[-1]})")

    ax.set_xlabel("frame")
    ax.set_ylabel("cumulative ID-switches")
    ax.set_title("ID-switch count over time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_id_switches] wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run",   action="append", type=Path, required=True,
                   help="run output directory (contains id_switches.csv)")
    p.add_argument("--label", action="append", type=str,  default=[],
                   help="legend label per --run (repeat in same order)")
    p.add_argument("--out",   type=Path, required=True,
                   help="output PNG path")
    args = p.parse_args()

    if args.label and len(args.label) != len(args.run):
        sys.exit("number of --label must match number of --run")
    labels = args.label or [r.name for r in args.run]
    runs = list(zip(args.run, labels))

    plot_runs(runs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
