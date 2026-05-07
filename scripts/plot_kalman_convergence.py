#!/usr/bin/env python3
"""Two-panel convergence plot for a Kalman process-noise (Q) sweep.

Reads `tracks.csv` from each `--run` directory and renders the headline
asset for the M10 blog post's Kalman section. Designed for hiring-manager
visuals: large fonts, two side-by-side panels, distinct colors per Q value,
raw detections overlaid as scatter so the smoothing-vs-jitter story is
visible at a glance.

Layout (one PNG, two axes side by side):
    [x-position over frame]                 [covariance trace over frame]
        scatter: raw detections                  log-y: cov trace per Q line
        4 colored lines: KF estimate                  legend in upper-right
        legend in upper-left

Usage:
    python scripts/plot_kalman_convergence.py \\
        --runs   results_m4/ablation_b/q_0.01 results_m4/ablation_b/q_0.1 \\
                 results_m4/ablation_b/q_1.0  results_m4/ablation_b/q_10.0 \\
        --labels "Q=0.01" "Q=0.1" "Q=1.0" "Q=10.0" \\
        --detections results_m4/ablation_b/linear.csv \\
        --out    results_m4/ablation_b/q_sweep.png

CSV schemas:
    tracks.csv       — frame_id,track_id,x,y,vx,vy,age,cov_trace
    detections.csv   — frame_id,det_id,x,y,class_id,gt_track_id
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_tracks(run_dir: Path) -> Dict[int, Tuple[float, float, float]]:
    """frame_id -> (x, y, cov_trace) for the SOLE tracked object.

    Ablation B is single-target by construction; we take the first row per
    frame. If the CSV ever has multiple track_ids per frame, this script
    would need extending — fine for now.
    """
    out: Dict[int, Tuple[float, float, float]] = {}
    csv_path = run_dir / "tracks.csv"
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            fi = int(row["frame_id"])
            if fi in out:
                continue   # keep first track per frame for single-target plot
            out[fi] = (float(row["x"]), float(row["y"]),
                       float(row["cov_trace"]))
    return out


def load_dets(csv_path: Path) -> Dict[int, Tuple[float, float]]:
    """frame_id -> (x, y) for the gt_track_id=0 (or first) detection."""
    out: Dict[int, Tuple[float, float]] = {}
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            fi = int(row["frame_id"])
            if fi in out:
                continue
            out[fi] = (float(row["x"]), float(row["y"]))
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs",       type=Path, nargs="+", required=True)
    p.add_argument("--labels",     type=str,  nargs="+", required=True)
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--out",        type=Path, required=True)
    p.add_argument("--title",      type=str,
                   default="Process-noise Q sweep — model trust vs measurement trust",
                   help="suptitle text; reused for R-sweep and other parameter sweeps")
    args = p.parse_args()

    if len(args.runs) != len(args.labels):
        sys.exit("--runs and --labels must have the same length")

    # Load every run.
    series = []
    for run_dir, label in zip(args.runs, args.labels):
        tr = load_tracks(run_dir)
        if not tr:
            print(f"[plot_kalman_convergence] WARN: no tracks in {run_dir}")
            continue
        series.append((label, tr))
    if not series:
        sys.exit("no runs had tracks.csv data")

    dets = load_dets(args.detections)

    # ---- Derive "truth" from detections via linear regression --------------
    # The detections are unbiased noisy samples around the true linear
    # trajectory; a least-squares line through them recovers truth without
    # needing the scenario's p0/v parameters hardcoded.
    d_frames_arr = np.array(sorted(dets.keys())) if dets else np.array([])
    if d_frames_arr.size >= 2:
        d_xs_arr = np.array([dets[f][0] for f in d_frames_arr])
        slope_x, intercept_x = np.polyfit(d_frames_arr, d_xs_arr, 1)
        truth_x = lambda f: slope_x * f + intercept_x
    else:
        truth_x = lambda f: 0.0   # fallback for degenerate input

    # ---- Figure -------------------------------------------------------------
    plt.rcParams.update({
        "font.size":       12,
        "axes.titlesize":  13,
        "axes.labelsize":  12,
        "legend.fontsize": 10,
    })
    fig, (ax_pos, ax_res, ax_cov) = plt.subplots(1, 3, figsize=(18, 5.5))

    # Stable per-Q color scheme.
    colors = plt.cm.viridis([i / max(1, len(series) - 1)
                             for i in range(len(series))])

    # ---- Panel 1: x-position vs frame (context) ----------------------------
    if dets:
        d_frames = sorted(dets.keys())
        d_xs = [dets[f][0] for f in d_frames]
        ax_pos.scatter(d_frames, d_xs, color="lightgray", s=18,
                       label="raw detection", zorder=1)
    for (label, tr), c in zip(series, colors):
        frames = sorted(tr.keys())
        xs = [tr[f][0] for f in frames]
        ax_pos.plot(frames, xs, "-", color=c, lw=2.0, label=label, zorder=2)
    ax_pos.set_title("Position estimate vs raw detections")
    ax_pos.set_xlabel("frame")
    ax_pos.set_ylabel("x  [m]")
    ax_pos.grid(True, alpha=0.3)
    ax_pos.legend(loc="upper left", framealpha=0.9)

    # ---- Panel 2: residual (estimate - truth_from_linear_fit) --------------
    # This is the headline panel for the smoothing-vs-jitter story. Zooming
    # into the residual scale exposes what's invisible at full position scale.
    if dets:
        d_frames = sorted(dets.keys())
        det_residuals = [dets[f][0] - truth_x(f) for f in d_frames]
        ax_res.scatter(d_frames, det_residuals, color="lightgray", s=18,
                       label="raw detection", zorder=1)
    for (label, tr), c in zip(series, colors):
        frames = sorted(tr.keys())
        residuals = [tr[f][0] - truth_x(f) for f in frames]
        ax_res.plot(frames, residuals, "-", color=c, lw=2.0, label=label, zorder=2)
    ax_res.axhline(0.0, color="black", lw=0.8, alpha=0.5, zorder=0)
    ax_res.set_title("Estimate residual (x − linear fit)")
    ax_res.set_xlabel("frame")
    ax_res.set_ylabel("residual  [m]")
    ax_res.grid(True, alpha=0.3)
    ax_res.legend(loc="upper left", framealpha=0.9)

    # ---- Panel 3: cov_trace vs frame (log-y to span dynamic range) ---------
    for (label, tr), c in zip(series, colors):
        frames = sorted(tr.keys())
        traces = [tr[f][2] for f in frames]
        ax_cov.semilogy(frames, traces, "-", color=c, lw=2.0, label=label)
    ax_cov.set_title("Posterior covariance trace (log scale)")
    ax_cov.set_xlabel("frame")
    ax_cov.set_ylabel("trace(P)")
    ax_cov.grid(True, alpha=0.3, which="both")
    ax_cov.legend(loc="upper right", framealpha=0.9)

    fig.suptitle(args.title, fontsize=15, y=1.0)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"[plot_kalman_convergence] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
