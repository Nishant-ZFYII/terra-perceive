#!/usr/bin/env python3
"""Replicates the earlier `False revivals` audit — but on the Fix-B tracks.csv.

For every track_id visible in [1750, 1830], find any internal gaps (frames
during which the track was Lost in the cascade), and measure the WORLD-FRAME
drift between the position last-seen-before-gap and first-seen-after-gap.

Pre-Fix-B (max_age=300, ego-frame anchor): 18/35 had gaps > 50 frames with
drift up to 15.7 m. Trees don't move 15 m. Those were false-merges.

Post-Fix-B (max_age=300, world-frame anchor): every "revived" track must
have small world-frame drift, because cascade now only matches re-detections
that line up with the original world position (not the stale ego coord).
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/nishant/MS_Project/terra-perceive-p2m4")
TRACKS_CSV = REPO / "results_m4/ablation_g/sort_on_rellis/tracks.csv"
POSES_CSV  = REPO / "data/poses_slam_full.csv"

WIN_LO, WIN_HI = 1750, 1830  # same stationary window as the prior audit

# ---- 1. Load SLAM poses (frame_id → SE(2): yaw + tx, ty) ----
def yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))

poses = {}  # frame_id -> (yaw, tx, ty)
with POSES_CSV.open() as f:
    r = csv.DictReader(f)
    for row in r:
        fid = int(row["frame_id"])
        tx = float(row["tx"]); ty = float(row["ty"])
        qx = float(row["qx"]); qy = float(row["qy"])
        qz = float(row["qz"]); qw = float(row["qw"])
        poses[fid] = (yaw_from_quat(qx, qy, qz, qw), tx, ty)

def ego_to_world(fid, ex, ey):
    yaw, tx, ty = poses.get(fid, (0.0, 0.0, 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * ex - s * ey + tx, s * ex + c * ey + ty)

# ---- 2. Load tracks.csv grouped by track_id ----
by_track = defaultdict(list)  # track_id -> [(frame, x_ego, y_ego, x_world, y_world), ...]
with TRACKS_CSV.open() as f:
    r = csv.DictReader(f)
    for row in r:
        fid = int(row["frame_id"]); tid = int(row["track_id"])
        x = float(row["x"]); y = float(row["y"])
        wx, wy = ego_to_world(fid, x, y)
        by_track[tid].append((fid, x, y, wx, wy))
for tid in by_track:
    by_track[tid].sort()

# ---- 3. Pick tracks visible inside the stationary window ----
visible = [tid for tid, rows in by_track.items()
           if any(WIN_LO <= r[0] <= WIN_HI for r in rows)]

# ---- 4. For each, find internal gaps (consecutive frames with frame_diff > 1) ----
revivals = []  # (track_id, last_frame_before_gap, first_frame_after_gap,
               #  gap_len, drift_world_m, drift_ego_m)
for tid in visible:
    rows = by_track[tid]
    for i in range(1, len(rows)):
        prev = rows[i - 1]; cur = rows[i]
        gap = cur[0] - prev[0]
        if gap <= 1:
            continue
        drift_world = math.hypot(cur[3] - prev[3], cur[4] - prev[4])
        drift_ego   = math.hypot(cur[1] - prev[1], cur[2] - prev[2])
        revivals.append((tid, prev[0], cur[0], gap, drift_world, drift_ego))

# ---- 5. Report ----
print(f"=== Fix B audit — stationary window [{WIN_LO}, {WIN_HI}] ===")
print(f"tracks visible in window           : {len(visible)}")
print(f"internal-gap revivals across these : {len(revivals)}")
print()

# Long-gap revivals (the original bug class)
long_gaps = [r for r in revivals if r[3] > 50]
print(f"long-gap revivals (gap > 50 frames): {len(long_gaps)}")
if long_gaps:
    print(f"{'tid':>5} {'last_f':>6} {'next_f':>6} {'gap':>4}  "
          f"{'world_drift_m':>13}  {'ego_drift_m':>11}")
    for r in sorted(long_gaps, key=lambda x: -x[4])[:25]:
        print(f"{r[0]:>5} {r[1]:>6} {r[2]:>6} {r[3]:>4}  "
              f"{r[4]:>13.2f}  {r[5]:>11.2f}")

# All-revivals histogram
print()
print("World-frame drift histogram for ALL revivals in this window:")
buckets = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 1e9)]
for lo, hi in buckets:
    n = sum(1 for r in revivals if lo <= r[4] < hi)
    label = f"[{lo:>4.1f}, {hi:>4.1f})" if hi < 1e9 else f"[{lo:>4.1f},   inf)"
    bar = "#" * n
    print(f"  {label}  n={n:>3}  {bar}")

# Worst offenders globally (not just in window)
print()
print("Worst world-frame drifts ACROSS THE WHOLE DRIVE (not just window):")
all_revivals = []
for tid, rows in by_track.items():
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        gap = cur[0] - prev[0]
        if gap <= 1:
            continue
        d_world = math.hypot(cur[3] - prev[3], cur[4] - prev[4])
        all_revivals.append((tid, prev[0], cur[0], gap, d_world))
all_revivals.sort(key=lambda x: -x[4])
print(f"total revivals across drive: {len(all_revivals)}")
for r in all_revivals[:10]:
    print(f"  tid={r[0]:>4}  gap=[{r[1]}..{r[2]}] ({r[3]:>3} frames)  "
          f"drift={r[4]:.2f} m")
