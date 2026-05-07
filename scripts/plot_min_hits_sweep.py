#!/usr/bin/env python3
"""Bar chart for the min_hits confirmation-threshold sweep (Ablation E).

Two grouped bars per cell:
    - fp_count       — number of distinct track IDs that EVER appeared in
                       publishable output (`tracks.csv`) AND whose closest
                       detection on their first publish frame had gt_track_id=-1.
    - init_latency   — frames between the first gt=0 detection in the input
                       and the first time gt=0 appears in `tracks.csv`.
                       Returns -1 if the gt=0 track never reaches publishable.

Computation notes:
    The runner does NOT write fp_count / init_latency directly — they're
    post-processed from `tracks.csv` + the input detections CSV here. Keeps
    the runner generic; specialization lives at the plot layer.

Usage:
    python scripts/plot_min_hits_sweep.py \\
        --runs       results_m4/ablation_e/mh_1 results_m4/ablation_e/mh_3 results_m4/ablation_e/mh_5 \\
        --labels     "min_hits=1" "min_hits=3" "min_hits=5" \\
        --detections results_m4/ablation_e/spurious.csv \\
        --out        results_m4/ablation_e/min_hits_sweep.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_tracks(run_dir: Path) -> List[Tuple[int, int, float, float]]:
    """Return list of (frame_id, track_id, x, y) rows from tracks.csv."""
    out: List[Tuple[int, int, float, float]] = []
    with (run_dir / "tracks.csv").open() as f:
        r = csv.DictReader(f)
        for row in r:
            out.append((int(row["frame_id"]), int(row["track_id"]),
                        float(row["x"]), float(row["y"])))
    return out


def load_dets(csv_path: Path) -> Dict[int, List[Tuple[int, float, float, int]]]:
    """frame_id -> list of (det_id, x, y, gt_track_id)."""
    out: Dict[int, List[Tuple[int, float, float, int]]] = defaultdict(list)
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            out[int(row["frame_id"])].append((
                int(row["det_id"]),
                float(row["x"]), float(row["y"]),
                int(row["gt_track_id"]),
            ))
    return out


def compute_fp_count(tracks: List[Tuple[int, int, float, float]],
                     dets: Dict[int, List[Tuple[int, float, float, int]]]) -> int:
    """Number of distinct track IDs whose first publish frame is closest to
    a gt=-1 detection."""
    first_pub: Dict[int, Tuple[int, float, float]] = {}
    for fr, tid, x, y in tracks:
        if tid not in first_pub:
            first_pub[tid] = (fr, x, y)

    fp_ids = 0
    for tid, (fr, x, y) in first_pub.items():
        frame_dets = dets.get(fr, [])
        if not frame_dets:
            continue
        # Nearest detection in this frame.
        best = min(frame_dets, key=lambda d: (d[1] - x) ** 2 + (d[2] - y) ** 2)
        if best[3] < 0:    # gt_track_id < 0 → spurious
            fp_ids += 1
    return fp_ids


def compute_init_latency(tracks: List[Tuple[int, int, float, float]],
                         dets: Dict[int, List[Tuple[int, float, float, int]]]) -> int:
    """Frames between gt=0's first appearance in detections and its first
    appearance in publishable tracks.csv. -1 if never published."""
    first_gt0_det = None
    for fr in sorted(dets.keys()):
        for det_id, x, y, gt in dets[fr]:
            if gt == 0:
                first_gt0_det = (fr, x, y)
                break
        if first_gt0_det is not None:
            break
    if first_gt0_det is None:
        return -1

    # Find first track row whose nearest gt-labeled det is gt=0.
    for fr, tid, x, y in sorted(tracks, key=lambda r: r[0]):
        frame_dets = dets.get(fr, [])
        if not frame_dets:
            continue
        best = min(frame_dets, key=lambda d: (d[1] - x) ** 2 + (d[2] - y) ** 2)
        if best[3] == 0:
            return fr - first_gt0_det[0]
    return -1


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

    dets = load_dets(args.detections)

    fp_counts: List[int] = []
    init_lats: List[int] = []
    for run in args.runs:
        tracks = load_tracks(run)
        fp_counts.append(compute_fp_count(tracks, dets))
        init_lats.append(compute_init_latency(tracks, dets))

    plt.rcParams.update({
        "font.size":       12,
        "axes.titlesize":  14,
        "axes.labelsize":  12,
        "legend.fontsize": 11,
    })
    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(args.labels))
    width = 0.35
    bar_fp  = ax.bar(x - width/2, fp_counts, width, color="#d6604d",
                     edgecolor="black", label="false-positive tracks")
    bar_lat = ax.bar(x + width/2, init_lats, width, color="#4393c3",
                     edgecolor="black", label="init latency  [frames]")

    for bar_grp, vals in ((bar_fp, fp_counts), (bar_lat, init_lats)):
        for r, v in zip(bar_grp, vals):
            label = "—" if v < 0 else str(v)
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.05,
                    label, ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(args.labels)
    ax.set_xlabel("min_hits confirmation threshold")
    ax.set_ylabel("count / frames")
    ax.set_title("Effect of min_hits on false-positive suppression and init latency  (spurious scenario)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"[plot_min_hits_sweep] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
