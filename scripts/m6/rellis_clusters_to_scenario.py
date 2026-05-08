#!/usr/bin/env python3
"""P2-M6 Phase 5: convert RELLIS DBSCAN cluster centroids into a single
scenario CSV that safety_runner can replay through both safety modes.

For each frame, picks the nearest cluster centroid inside the forward arc
(+/- 60 degrees, x > 0) and treats it as the "worker" position. Worker
velocity (vx, vy) is estimated from the inter-frame delta of that nearest
centroid, optionally smoothed with a moving average to suppress DBSCAN
centroid jitter (the m10 lesson).

When no cluster sits inside the forward arc for a given frame, the worker
position is emitted as a far sentinel (x=100, y=0) so the supervisor reads
"no closing geometry, no engagement."

Usage:
    python scripts/m6/rellis_clusters_to_scenario.py \\
        --clusters-dir /media/.../m4_perframe/clusters_sweetspot \\
        --out scripts/m6/scenarios/rellis_hero.csv \\
        --smoothing 5 \\
        --vehicle-v 2.0
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

DT = 0.1
ARC_HALF = math.radians(60.0)
SENTINEL_X = 100.0
SENTINEL_Y = 0.0


def cluster_centroids(csv_path: Path) -> list[tuple[float, float, float]]:
    """Return [(x_mean, y_mean, z_mean)] for each non-noise cluster."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    pts: dict[int, list[tuple[float, float, float]]] = {}
    with csv_path.open() as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip().split(",")
            if len(parts) < 4:
                continue
            try:
                cid = int(parts[3])
            except ValueError:
                continue
            if cid < 0:
                continue  # noise
            pts.setdefault(cid, []).append(
                (float(parts[0]), float(parts[1]), float(parts[2]))
            )
    out: list[tuple[float, float, float]] = []
    for cid, group in pts.items():
        a = np.asarray(group)
        out.append((float(a[:, 0].mean()), float(a[:, 1].mean()), float(a[:, 2].mean())))
    return out


def in_forward_arc(centroids):
    """Return [(x, y, range)] for centroids inside the forward arc."""
    out = []
    for cx, cy, _cz in centroids:
        if cx <= 0:
            continue
        bearing = math.atan2(cy, cx)
        if abs(bearing) > ARC_HALF:
            continue
        out.append((cx, cy, math.hypot(cx, cy)))
    return out


def nearest_to_anchor(arc_pts, anchor, max_jump=2.0):
    """Pick the centroid in arc_pts closest to `anchor` (the previous frame's
    worker position). If no centroid is within max_jump meters of anchor,
    return None. anchor=None falls back to nearest-to-ego."""
    if not arc_pts:
        return None
    if anchor is None:
        return min(arc_pts, key=lambda p: p[2])[:2]
    ax, ay = anchor
    best = None
    best_d = max_jump
    for cx, cy, _ in arc_pts:
        d = math.hypot(cx - ax, cy - ay)
        if d < best_d:
            best_d = d
            best = (cx, cy)
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clusters-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--vehicle-v", type=float, default=2.0)
    p.add_argument("--smoothing", type=int, default=5,
                   help="moving-average window applied to worker (x, y); "
                        "0 disables")
    p.add_argument("--frames", type=int, default=0,
                   help="cap on frames; 0 = all available")
    args = p.parse_args()

    cluster_files = sorted(args.clusters_dir.glob("clusters_*.csv"))
    if not cluster_files:
        raise SystemExit(f"no cluster CSVs under {args.clusters_dir}")
    if args.frames > 0:
        cluster_files = cluster_files[: args.frames]
    print(f"[adapter] reading {len(cluster_files)} cluster files")

    raw_xy: list[tuple[float, float] | None] = []
    anchor: tuple[float, float] | None = None
    coast_remaining = 0  # frames we keep coasting after losing the worker
    COAST_FRAMES = 5     # ~0.5 s grace period before declaring "no worker"
    for f in cluster_files:
        cents = cluster_centroids(f)
        arc_pts = in_forward_arc(cents)
        pick = nearest_to_anchor(arc_pts, anchor, max_jump=2.0)
        if pick is None and anchor is not None and coast_remaining > 0:
            pick = anchor
            coast_remaining -= 1
        if pick is not None:
            anchor = pick
            coast_remaining = COAST_FRAMES
        else:
            anchor = None
        raw_xy.append(pick)

    # Smooth the (x, y) sequence over valid (in-arc) frames. Stretches with
    # "no worker in arc" remain sentinels and are not smoothed.
    xs = np.full(len(raw_xy), np.nan)
    ys = np.full(len(raw_xy), np.nan)
    for i, p_xy in enumerate(raw_xy):
        if p_xy is not None:
            xs[i] = p_xy[0]
            ys[i] = p_xy[1]

    if args.smoothing >= 2:
        kernel = np.ones(args.smoothing) / args.smoothing
        # Only smooth contiguous valid runs; NaN pads break convolution.
        for arr in (xs, ys):
            valid = ~np.isnan(arr)
            if not valid.any():
                continue
            # Smooth in-place inside contiguous valid blocks only.
            i = 0
            while i < len(arr):
                if not valid[i]:
                    i += 1
                    continue
                j = i
                while j < len(arr) and valid[j]:
                    j += 1
                if j - i >= args.smoothing:
                    arr[i:j] = np.convolve(arr[i:j], kernel, mode="same")
                i = j

    # Velocities from finite differences (centered when possible).
    # Cap at +/- 2 m/s to suppress centroid-jitter / cluster-switch artifacts
    # (workers walk at < 2 m/s; anything above is noise).
    VEL_CAP = 2.0
    vxs = np.zeros(len(raw_xy))
    vys = np.zeros(len(raw_xy))
    for i in range(len(raw_xy)):
        if np.isnan(xs[i]):
            continue
        prev_i = i - 1 if i > 0 and not np.isnan(xs[i - 1]) else i
        next_i = i + 1 if i + 1 < len(raw_xy) and not np.isnan(xs[i + 1]) else i
        span = max(1, next_i - prev_i)
        vx = (xs[next_i] - xs[prev_i]) / (span * DT)
        vy = (ys[next_i] - ys[prev_i]) / (span * DT)
        vxs[i] = max(-VEL_CAP, min(VEL_CAP, vx))
        vys[i] = max(-VEL_CAP, min(VEL_CAP, vy))

    n_in_arc = int((~np.isnan(xs)).sum())
    print(f"[adapter] {n_in_arc} / {len(raw_xy)} frames have a worker in arc "
          f"({100 * n_in_arc / len(raw_xy):.1f}%)")

    # safety_runner integrates sim-ego forward at vehicle_v starting from
    # origin. Cluster centroids are in real-ego frame at each frame. To make
    # the runner's `dx = worker.x - sim_ego.x` come out as the real ego-frame
    # x, pre-add the runner's expected cumulative sim-ego x to each worker
    # row. This is exact when the supervisor stays at scale=1, and within a
    # few cm during brief engagements (sim-ego decelerates slightly slower
    # than the assumed 2 m/s while the supervisor clamps).
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "worker_id", "x", "y", "vx", "vy",
                    "vehicle_v", "vehicle_dir"])
        for i in range(len(raw_xy)):
            ego_advance_x = i * args.vehicle_v * DT
            if np.isnan(xs[i]):
                w.writerow([i, 0, SENTINEL_X + ego_advance_x, SENTINEL_Y,
                            0.0, 0.0, args.vehicle_v, 0.0])
            else:
                w.writerow([i, 0, f"{xs[i] + ego_advance_x:.4f}", f"{ys[i]:.4f}",
                            f"{vxs[i]:.4f}", f"{vys[i]:.4f}",
                            args.vehicle_v, 0.0])
    print(f"[adapter] wrote {args.out}")


if __name__ == "__main__":
    main()
