#!/usr/bin/env python3
"""K=1 DBSCAN cluster-centroid jitter audit (Phase-4 step 1).

Measures the actual frame-to-frame world-frame movement of DBSCAN
cluster centroids on stationary segments of RELLIS-3D, to test the
hypothesis that "stationary tree centroids jitter 5–15 m between
sightings as LiDAR scans different sides of the trunk." This is the
prerequisite for Phase-4 multi-frame accumulation — if the hypothesis
is wrong, accumulation won't help and we need a different diagnosis
of the Mahalanobis ceiling.

Approach:
  1. Load rellis_detections.csv (one row per DBSCAN cluster centroid
     per frame, in current-ego frame).
  2. Load poses_slam_full.csv; build SE(2) frame_id → T_world_ego.
  3. For each pair of consecutive frames in the stationary window
     [1750, 1830] — where ego world displacement is 1.55 m — compose
     every detection into world frame, greedy-match clusters by
     nearest-neighbor in world frame (no track-ID lookup, just pure
     nearest-neighbor association), and record the matched-pair
     world-frame distance.
  4. Repeat with frame gaps of {1, 5, 10, 20, 50} to measure how
     jitter compounds.
  5. Histogram + print median / p90 / p99 per gap.

PASS condition (verifies the hypothesis): median ≥ 1 m, p90 ≥ 5 m
   on adjacent-frame matches. Then K=3 accumulation has structural
   reason to help.
FAIL condition (invalidates the hypothesis): median < 0.5 m and
   p90 < 2 m. Re-diagnose the Mahalanobis ceiling — it's not DBSCAN
   noise, it's something else.

Caveats:
  - Greedy nearest-neighbor matching can mismatch when two clusters
    cross. Filter out matches with d_world > 8 m (almost certainly
    different physical objects); this filtering is biased TOWARD
    showing low jitter, so if we still see median ≥ 1 m the
    hypothesis is robust.
  - The stationary window has 80 frames; with ~20 visible clusters
    per frame, gap=1 produces ~20×80 ≈ 1600 sample matches —
    enough for a stable distribution.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/nishant/MS_Project/terra-perceive-p2m4")
DETS_CSV = REPO / "results_m4/ablation_g/rellis_detections.csv"
POSES_CSV = REPO / "data/poses_slam_full.csv"

# Stationary window — same as the revival audit. Ego world displacement
# is 1.55 m across these 80 frames (verified).
WIN_LO, WIN_HI = 1750, 1830
GAPS = [1, 5, 10, 20, 50]
MAX_PLAUSIBLE_MATCH_M = 8.0  # reject pairs farther than this — almost certainly different objects


def yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def load_poses():
    poses = {}
    with POSES_CSV.open() as f:
        for row in csv.DictReader(f):
            fid = int(row["frame_id"])
            poses[fid] = (
                yaw_from_quat(float(row["qx"]), float(row["qy"]),
                              float(row["qz"]), float(row["qw"])),
                float(row["tx"]), float(row["ty"]),
            )
    return poses


def e2w(pose, x, y):
    yaw, tx, ty = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * x - s * y + tx, s * x + c * y + ty)


def load_dets_in_world():
    """Return frame_id -> list of (wx, wy) detections in world frame."""
    poses = load_poses()
    by_frame_world = defaultdict(list)
    with DETS_CSV.open() as f:
        for row in csv.DictReader(f):
            fid = int(row["frame_id"])
            if fid not in poses:
                continue
            ex = float(row["x"]); ey = float(row["y"])
            wx, wy = e2w(poses[fid], ex, ey)
            by_frame_world[fid].append((wx, wy))
    return by_frame_world


def greedy_nn_match(set_a, set_b, max_d=MAX_PLAUSIBLE_MATCH_M):
    """Greedy nearest-neighbor 1-to-1 match.

    Each item in set_a paired with at most one item in set_b. Iterates
    in order, picking each a's nearest unused b within max_d.
    Returns list of (idx_a, idx_b, d_world).
    """
    used_b = [False] * len(set_b)
    pairs = []
    for ai, (ax, ay) in enumerate(set_a):
        best_bi = -1
        best_d = math.inf
        for bi, (bx, by) in enumerate(set_b):
            if used_b[bi]:
                continue
            d = math.hypot(ax - bx, ay - by)
            if d < best_d:
                best_d = d; best_bi = bi
        if best_bi >= 0 and best_d <= max_d:
            used_b[best_bi] = True
            pairs.append((ai, best_bi, best_d))
    return pairs


def percentile(sorted_xs, p):
    if not sorted_xs:
        return float("nan")
    k = (len(sorted_xs) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_xs[lo]
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (k - lo)


def main():
    print(f"=== K=1 DBSCAN centroid jitter — stationary window [{WIN_LO}, {WIN_HI}] ===")
    print(f"Greedy nearest-neighbor in world frame; matches > {MAX_PLAUSIBLE_MATCH_M} m discarded")
    print(f"(this filtering is BIASED TOWARD low jitter — if median is still ≥ 1 m,")
    print(f" the DBSCAN-noise hypothesis is confirmed.)")
    print()

    by_frame = load_dets_in_world()
    in_window = sorted(f for f in by_frame if WIN_LO <= f <= WIN_HI)
    print(f"frames in window with detections : {len(in_window)}")
    print(f"avg detections per frame         : "
          f"{statistics.mean(len(by_frame[f]) for f in in_window):.1f}")
    print()

    # Per-gap jitter distribution
    print(f"{'gap':>4} {'pairs':>7} {'median_m':>10} {'p90_m':>8} {'p99_m':>8} {'max_m':>8}  verdict")
    print("-" * 70)
    for gap in GAPS:
        all_d = []
        for f in in_window:
            f2 = f + gap
            if f2 not in by_frame:
                continue
            if not (WIN_LO <= f2 <= WIN_HI):
                continue
            pairs = greedy_nn_match(by_frame[f], by_frame[f2])
            all_d.extend(d for _, _, d in pairs)
        if not all_d:
            print(f"{gap:>4} {'0':>7} {'-':>10} {'-':>8} {'-':>8} {'-':>8}")
            continue
        all_d.sort()
        med = statistics.median(all_d)
        p90 = percentile(all_d, 90)
        p99 = percentile(all_d, 99)
        mx = all_d[-1]
        # Verdict per the protocol in m10-debug-log.md
        if gap == 1:
            if med >= 1.0 and p90 >= 5.0:
                v = "✅ hypothesis CONFIRMED (median ≥ 1 m, p90 ≥ 5 m)"
            elif med < 0.5 and p90 < 2.0:
                v = "❌ hypothesis FAILED (median < 0.5 m, p90 < 2 m)"
            else:
                v = "↔ INCONCLUSIVE — re-examine"
        else:
            v = ""
        print(f"{gap:>4} {len(all_d):>7} {med:>10.2f} {p90:>8.2f} {p99:>8.2f} {mx:>8.2f}  {v}")

    print()
    # Histogram for gap=1 (the headline number)
    print(f"=== Histogram for gap=1 (adjacent frames in window) ===")
    all_d = []
    for f in in_window:
        f2 = f + 1
        if f2 not in by_frame:
            continue
        if not (WIN_LO <= f2 <= WIN_HI):
            continue
        pairs = greedy_nn_match(by_frame[f], by_frame[f2])
        all_d.extend(d for _, _, d in pairs)
    bins = [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0),
            (5.0, 8.0)]
    for lo, hi in bins:
        n = sum(1 for d in all_d if lo <= d < hi)
        bar = "#" * min(n // 2, 80)
        print(f"  [{lo:>4.2f}, {hi:>4.2f})  n={n:>4}  {bar}")

    # Also report ego displacement during the window
    poses = load_poses()
    p_lo, p_hi = poses[WIN_LO], poses[WIN_HI]
    ego_disp = math.hypot(p_hi[1] - p_lo[1], p_hi[2] - p_lo[2])
    print()
    print(f"Ego world displacement across [{WIN_LO}, {WIN_HI}]: {ego_disp:.2f} m")
    print("(stationary segment confirmed — any centroid drift > ego drift")
    print(" is genuine cluster jitter, not real motion.)")


if __name__ == "__main__":
    main()
