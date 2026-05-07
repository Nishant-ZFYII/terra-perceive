#!/usr/bin/env python3
"""Track-id-over-frame timeline for the max_misses sweep (Ablation F).

Three vertically stacked panels (one per max_misses cell). Each panel:
    - X = frame_id
    - Y = published track_id
    - Marker per (frame, track_id) row in tracks.csv
    - Gray vertical band shows the OCCLUSION GAP (where input detections
      had no rows; computed from the gap in the input CSV's frame_ids)
    - Different track ids → different colors

Story:
    max_misses=1 → 2 distinct ids: pre-gap track dies, post-gap gets new id.
    max_misses=3 → 2 distinct ids: same.
    max_misses=10 → 1 distinct id: track survives the gap.

Usage:
    python scripts/plot_max_misses_sweep.py \\
        --runs       results_m4/ablation_f/m_1 results_m4/ablation_f/m_3 results_m4/ablation_f/m_10 \\
        --labels     "max_misses=1" "max_misses=3" "max_misses=10" \\
        --detections results_m4/ablation_f/occluded.csv \\
        --out        results_m4/ablation_f/max_misses_sweep.png
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


def load_tracks(run_dir: Path) -> List[Tuple[int, int]]:
    """Return list of (frame_id, track_id) from tracks.csv."""
    out: List[Tuple[int, int]] = []
    with (run_dir / "tracks.csv").open() as f:
        r = csv.DictReader(f)
        for row in r:
            out.append((int(row["frame_id"]), int(row["track_id"])))
    return out


def detect_gap(det_csv: Path) -> Tuple[int, int]:
    """Find the longest contiguous gap of missing frame_ids in the dets CSV.
    Returns (gap_start, gap_end_inclusive). If no gap, returns (-1, -1)."""
    frames = set()
    with det_csv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            frames.add(int(row["frame_id"]))
    if not frames:
        return (-1, -1)
    lo, hi = min(frames), max(frames)
    longest = (-1, -1)
    cur_start = -1
    for i in range(lo, hi + 1):
        if i not in frames:
            if cur_start == -1:
                cur_start = i
        else:
            if cur_start != -1:
                gap = (cur_start, i - 1)
                if (gap[1] - gap[0]) > (longest[1] - longest[0]):
                    longest = gap
                cur_start = -1
    return longest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs",       nargs="+", type=Path, required=True)
    p.add_argument("--labels",     nargs="+", type=str,  required=True)
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--out",        type=Path, required=True)
    args = p.parse_args()

    if len(args.runs) != len(args.labels):
        sys.exit("--runs and --labels must have the same length")

    gap = detect_gap(args.detections)

    plt.rcParams.update({
        "font.size":       12,
        "axes.titlesize":  13,
        "axes.labelsize":  12,
        "legend.fontsize": 10,
    })
    n = len(args.runs)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    # Determine global x-range from all detections + all tracks for consistency.
    all_frames: List[int] = []
    for run in args.runs:
        all_frames.extend(fr for fr, _ in load_tracks(run))
    if gap[0] >= 0:
        all_frames.extend([gap[0] - 1, gap[1] + 1])
    x_lo = min(all_frames) if all_frames else 0
    x_hi = max(all_frames) if all_frames else 1

    for ax, run, label in zip(axes, args.runs, args.labels):
        rows = load_tracks(run)
        unique_tids = sorted({tid for _, tid in rows})

        for tid in unique_tids:
            xs = [fr for fr, t in rows if t == tid]
            ys = [tid] * len(xs)
            ax.scatter(xs, ys, s=42, edgecolor="black",
                       label=f"track {tid}",
                       color=plt.cm.tab10(tid % 10))

        if gap[0] >= 0:
            ax.axvspan(gap[0] - 0.5, gap[1] + 0.5,
                       color="lightgray", alpha=0.4, zorder=0,
                       label="occlusion gap")

        ax.set_title(f"{label}    (distinct ids = {len(unique_tids)})")
        ax.set_ylabel("track_id")
        ax.set_xlim(x_lo - 1, x_hi + 1)
        ax.set_yticks(unique_tids)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.9)

    axes[-1].set_xlabel("frame")
    fig.suptitle("max_misses sweep — track persistence through occlusion",
                 fontsize=15, y=1.0)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"[plot_max_misses_sweep] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
