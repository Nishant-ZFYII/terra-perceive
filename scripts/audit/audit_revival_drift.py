#!/usr/bin/env python3
"""Tighter audit: gaps with BOTH endpoints inside [1750, 1830].

This is the correct definition of "stationary-window revival" — a track
that was last published inside the window, went Lost during the window,
and was revived inside the window. If the ego barely moves across those
frames (which is the whole point of calling it a stationary segment),
then a real re-acquisition of the same physical object should have
near-zero world-frame drift; a false revival shows up as drift large
compared to DBSCAN cluster noise (< 1 m).
"""
import csv, math
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/nishant/MS_Project/terra-perceive-p2m4")
TRACKS_CSV = REPO / "results_m4/ablation_g/sort_on_rellis/tracks.csv"
POSES_CSV  = REPO / "data/poses_slam_full.csv"
WIN_LO, WIN_HI = 1750, 1830

def yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))

poses = {}
with POSES_CSV.open() as f:
    for row in csv.DictReader(f):
        fid = int(row["frame_id"])
        poses[fid] = (yaw_from_quat(float(row["qx"]), float(row["qy"]),
                                    float(row["qz"]), float(row["qw"])),
                      float(row["tx"]), float(row["ty"]))

def e2w(fid, x, y):
    if fid not in poses: return None  # don't fake-Identity for missing poses
    yaw, tx, ty = poses[fid]
    c, s = math.cos(yaw), math.sin(yaw)
    return (c*x - s*y + tx, s*x + c*y + ty)

# Confirm ego barely moves across [1750, 1830]
p_lo = poses[WIN_LO]; p_hi = poses[WIN_HI]
ego_displacement = math.hypot(p_hi[1] - p_lo[1], p_hi[2] - p_lo[2])
print(f"=== Stationary window check ===")
print(f"Ego world displacement [{WIN_LO}, {WIN_HI}]: {ego_displacement:.2f} m")
print(f"  pose@{WIN_LO} = (tx={p_lo[1]:.2f}, ty={p_lo[2]:.2f})")
print(f"  pose@{WIN_HI} = (tx={p_hi[1]:.2f}, ty={p_hi[2]:.2f})")
print()

# Per-track rows
by_track = defaultdict(list)
with TRACKS_CSV.open() as f:
    for row in csv.DictReader(f):
        fid = int(row["frame_id"]); tid = int(row["track_id"])
        x = float(row["x"]); y = float(row["y"])
        wp = e2w(fid, x, y)
        if wp is None: continue
        by_track[tid].append((fid, x, y, wp[0], wp[1]))
for tid in by_track:
    by_track[tid].sort()

# Gaps strictly inside the window
in_window = []
for tid, rows in by_track.items():
    for i in range(1, len(rows)):
        prev, cur = rows[i-1], rows[i]
        gap = cur[0] - prev[0]
        if gap <= 1: continue
        if not (WIN_LO <= prev[0] <= WIN_HI and WIN_LO <= cur[0] <= WIN_HI):
            continue
        d_world = math.hypot(cur[3]-prev[3], cur[4]-prev[4])
        d_ego   = math.hypot(cur[1]-prev[1], cur[2]-prev[2])
        in_window.append((tid, prev[0], cur[0], gap, d_world, d_ego))

# Tracks that LIVE in this window (any row in window)
live_in = sorted({tid for tid, rows in by_track.items()
                  if any(WIN_LO <= r[0] <= WIN_HI for r in rows)})

print(f"distinct tracks visible in window: {len(live_in)}")
print(f"gaps with BOTH endpoints in window: {len(in_window)}")
print()

# Cascade-eligible: gap > max_misses+1 = 11 (the threshold to enter Lost)
cascade = [g for g in in_window if g[3] > 11]
print(f"cascade-revival gaps (gap > 11)  : {len(cascade)}")
print(f"long cascade-revival gaps (>50)  : {sum(1 for g in in_window if g[3] > 50)}")
print()

print("World-frame drift distribution for cascade revivals in-window:")
buckets = [(0,0.5),(0.5,1),(1,2),(2,5),(5,10),(10,1e9)]
for lo, hi in buckets:
    n = sum(1 for g in cascade if lo <= g[4] < hi)
    label = f"[{lo:>4.1f}, {hi:>4.1f})" if hi<1e9 else f"[{lo:>4.1f},   inf)"
    print(f"  {label}  n={n}")

print()
print("Worst world-drift cascade revivals INSIDE the stationary window:")
print(f"{'tid':>5} {'last_f':>6} {'next_f':>6} {'gap':>4}  "
      f"{'world_m':>8}  {'ego_m':>8}")
for g in sorted(cascade, key=lambda x:-x[4])[:20]:
    print(f"{g[0]:>5} {g[1]:>6} {g[2]:>6} {g[3]:>4}  "
          f"{g[4]:>8.2f}  {g[5]:>8.2f}")

# Compare also: what fraction of total cascade revivals across the whole
# drive happen in moving-ego vs stationary segments?
print()
print("=== Whole-drive drift sanity ===")
moving_revivals  = []
all_cascade = []
for tid, rows in by_track.items():
    for i in range(1, len(rows)):
        prev, cur = rows[i-1], rows[i]
        if cur[0] - prev[0] <= 11: continue
        d_world = math.hypot(cur[3]-prev[3], cur[4]-prev[4])
        all_cascade.append((tid, prev[0], cur[0], cur[0]-prev[0], d_world))
        # ego displacement during the gap
        if prev[0] in poses and cur[0] in poses:
            yaw1, tx1, ty1 = poses[prev[0]]
            yaw2, tx2, ty2 = poses[cur[0]]
            ego_disp = math.hypot(tx2-tx1, ty2-ty1)
            moving_revivals.append((d_world, ego_disp, cur[0]-prev[0]))

# How does world drift correlate with ego displacement in the gap?
big_drift = [(d_world, ego_disp, gap) for d_world, ego_disp, gap in moving_revivals if d_world > 5]
print(f"cascade revivals (gap>11): {len(all_cascade)}")
print(f"  with world drift > 5 m: {sum(1 for r in moving_revivals if r[0] > 5)}")
print(f"  with world drift > 10 m: {sum(1 for r in moving_revivals if r[0] > 10)}")
print(f"  with world drift > 20 m: {sum(1 for r in moving_revivals if r[0] > 20)}")
print()
print("Of the >5 m drift revivals, was ego stationary or moving?")
print(f"{'world_drift':>12}  {'ego_disp':>10}  {'gap':>4}")
for d_w, d_e, gap in sorted(big_drift, key=lambda x:-x[0])[:15]:
    note = "[STATIONARY ego]" if d_e < 0.5 else "[ego moved]"
    print(f"{d_w:>12.2f}  {d_e:>10.2f}  {gap:>4}  {note}")
