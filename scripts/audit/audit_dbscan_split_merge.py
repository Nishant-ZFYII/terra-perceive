#!/usr/bin/env python3
"""Phase-4 step 2: split/merge confound diagnostic.

The K=1 jitter audit (step 1) measured a 3.07 m median centroid drift
between adjacent frames on greedy nearest-neighbor pairs. The hypothesis
behind that 3 m number is "DBSCAN cluster centroids on stationary trees
are intrinsically noisy because LiDAR scans different sides of the
trunk each frame." But there's a competing explanation: DBSCAN
*splits and merges* clusters between frames (a tree that's 1 cluster
at frame f, 2 sub-clusters at frame f+1 because the seam between
branches happens to fall above DBSCAN's eps in one frame and below in
another). Greedy NN matching on a split parent → child pairs against
the wrong half-cluster, producing apparent "jitter" of ~half the
parent's spatial extent. If splits/merges dominate the apparent
jitter, K-frame accumulation helps STRUCTURALLY (denser point support
eliminates the seam) and beats the √K Gaussian-noise model by a wide
margin.

Approach:
  1. For each frame f in the stationary window, record the cluster
     count N_f (one row per cluster centroid in rellis_detections.csv).
  2. For each adjacent pair (f, f+1):
       a. Greedy NN match in world frame, cap 8 m
       b. count split/merge events as count_appeared + count_disappeared
       c. record per-pair: |ΔN| = |N_{f+1} - N_f|, jitter median on
          matched pairs in this pair
  3. Bucket the pair-medians by ΔN ∈ {0, 1, 2, ≥3}, see if jitter
     correlates with cluster-count change.

PASS condition (split/merge confound dominant): jitter median for
   pairs with |ΔN| ≥ 1 is ≥ 1.5× the jitter median for stable pairs
   (ΔN = 0). Then K=3 accumulation should shrink jitter by more than
   √3, possibly close to half.
FAIL condition (independent Gaussian noise dominant): jitter median
   is roughly the same for ΔN = 0 and ΔN ≥ 1. Then √K is the right
   model and K=3's expected gain is 1.7×.

This is purely a diagnostic — no code changes downstream depend on
the result, only the EXPECTATION for K=3 does.
"""
from __future__ import annotations
import csv, math, statistics
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/nishant/MS_Project/terra-perceive-p2m4")
DETS_CSV = REPO / "results_m4/ablation_g/rellis_detections.csv"
POSES_CSV = REPO / "data/poses_slam_full.csv"
WIN_LO, WIN_HI = 1750, 1830
MAX_MATCH_M = 8.0


def yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))

def load_poses():
    poses = {}
    with POSES_CSV.open() as f:
        for row in csv.DictReader(f):
            fid = int(row["frame_id"])
            poses[fid] = (yaw_from_quat(float(row["qx"]), float(row["qy"]),
                                        float(row["qz"]), float(row["qw"])),
                          float(row["tx"]), float(row["ty"]))
    return poses

def e2w(pose, x, y):
    yaw, tx, ty = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (c*x - s*y + tx, s*x + c*y + ty)

def load_dets_world():
    poses = load_poses()
    by_frame = defaultdict(list)
    with DETS_CSV.open() as f:
        for row in csv.DictReader(f):
            fid = int(row["frame_id"])
            if fid not in poses: continue
            x = float(row["x"]); y = float(row["y"])
            wx, wy = e2w(poses[fid], x, y)
            by_frame[fid].append((wx, wy))
    return by_frame

def greedy_nn(set_a, set_b, max_d=MAX_MATCH_M):
    used = [False]*len(set_b); pairs = []
    for ai, (ax, ay) in enumerate(set_a):
        best_bi = -1; best_d = math.inf
        for bi, (bx, by) in enumerate(set_b):
            if used[bi]: continue
            d = math.hypot(ax-bx, ay-by)
            if d < best_d: best_d = d; best_bi = bi
        if best_bi >= 0 and best_d <= max_d:
            used[best_bi] = True
            pairs.append((ai, best_bi, best_d))
    return pairs

def main():
    print(f"=== Phase-4 step 2: cluster split/merge confound diagnostic ===")
    print(f"Window [{WIN_LO}, {WIN_HI}], stationary segment.")
    print()
    by_frame = load_dets_world()
    in_window = sorted(f for f in by_frame if WIN_LO <= f <= WIN_HI)

    # Per-frame cluster counts
    counts = [len(by_frame[f]) for f in in_window]
    print(f"Cluster counts per frame:")
    print(f"  min={min(counts)}  max={max(counts)}  mean={statistics.mean(counts):.1f}  "
          f"stdev={statistics.stdev(counts):.2f}")

    # Adjacent frame pair stats
    pair_stats = []  # (frame_f, N_f, N_f1, dN, n_matched, n_appeared, n_disappeared, median_jitter)
    for f in in_window:
        f1 = f+1
        if f1 not in by_frame or not (WIN_LO <= f1 <= WIN_HI): continue
        Nf, Nf1 = len(by_frame[f]), len(by_frame[f1])
        pairs = greedy_nn(by_frame[f], by_frame[f1])
        nm = len(pairs)
        n_disappeared = Nf - nm   # in f, not matched
        n_appeared    = Nf1 - nm  # in f+1, not matched
        med = statistics.median(d for _,_,d in pairs) if pairs else float("nan")
        pair_stats.append((f, Nf, Nf1, abs(Nf1-Nf), nm, n_appeared, n_disappeared, med))

    # Bucket by |ΔN|
    print()
    print("=== Pair-level cluster-count stability ===")
    buckets = {0: [], 1: [], 2: []}
    bucket_ge3 = []
    for s in pair_stats:
        dN = s[3]
        if dN == 0:   buckets[0].append(s)
        elif dN == 1: buckets[1].append(s)
        elif dN == 2: buckets[2].append(s)
        else:         bucket_ge3.append(s)
    total = len(pair_stats)
    print(f"{'|ΔN|':>5} {'pairs':>7} {'%':>6} {'med_jitter_m':>13} "
          f"{'med_appeared':>13} {'med_disappeared':>16}")
    for dN in (0, 1, 2):
        rows = buckets[dN]
        n = len(rows)
        pct = 100.0 * n / total if total else 0
        if rows:
            med_j = statistics.median(s[7] for s in rows if not math.isnan(s[7]))
            med_app = statistics.median(s[5] for s in rows)
            med_dis = statistics.median(s[6] for s in rows)
        else:
            med_j = med_app = med_dis = float("nan")
        print(f"{dN:>5} {n:>7} {pct:>5.1f}% {med_j:>13.2f} {med_app:>13.1f} {med_dis:>16.1f}")
    n_ge3 = len(bucket_ge3)
    if n_ge3:
        med_j = statistics.median(s[7] for s in bucket_ge3 if not math.isnan(s[7]))
        med_app = statistics.median(s[5] for s in bucket_ge3)
        med_dis = statistics.median(s[6] for s in bucket_ge3)
    else:
        med_j = med_app = med_dis = float("nan")
    pct = 100.0 * n_ge3 / total if total else 0
    print(f"{'≥3':>5} {n_ge3:>7} {pct:>5.1f}% {med_j:>13.2f} {med_app:>13.1f} {med_dis:>16.1f}")

    # Aggregate split/merge fraction
    print()
    total_apps = sum(s[5] for s in pair_stats)
    total_dis  = sum(s[6] for s in pair_stats)
    total_match = sum(s[4] for s in pair_stats)
    print(f"Aggregate across {total} pairs in window:")
    print(f"  matched (persisted)       : {total_match}")
    print(f"  appeared (probable splits): {total_apps}")
    print(f"  disappeared (probable merges/missed): {total_dis}")
    if total_match:
        churn_frac = (total_apps + total_dis) / (total_match + total_apps + total_dis)
        print(f"  churn fraction = (app+disapp) / (matched+app+disapp) = {churn_frac:.1%}")

    # Verdict on the confound hypothesis
    print()
    print("=== Verdict ===")
    if not buckets[0] or sum(1 for s in buckets[0] if not math.isnan(s[7])) == 0:
        print("Insufficient stable pairs to grade.")
        return
    stable_med = statistics.median(s[7] for s in buckets[0] if not math.isnan(s[7]))
    unstable_pairs = buckets[1] + buckets[2] + bucket_ge3
    if not unstable_pairs:
        print("All pairs stable — no split/merge evidence.")
        return
    unstable_med = statistics.median(s[7] for s in unstable_pairs if not math.isnan(s[7]))
    ratio = unstable_med / stable_med if stable_med > 0 else float("inf")
    print(f"Median jitter, stable pairs   (|ΔN|=0)   : {stable_med:.2f} m  ({len(buckets[0])} pairs)")
    print(f"Median jitter, unstable pairs (|ΔN|≥1)   : {unstable_med:.2f} m  ({len(unstable_pairs)} pairs)")
    print(f"Ratio = unstable / stable                : {ratio:.2f}×")
    print()
    if ratio >= 1.5:
        print("✅ HYPOTHESIS CONFIRMED — split/merge events drive ≥ 1.5× the jitter")
        print("   on stable cluster pairs. K-frame accumulation should beat the")
        print("   √K Gaussian model — likely close to halving the median.")
    elif ratio < 1.15:
        print("❌ HYPOTHESIS REJECTED — split/merge frame-pairs have similar jitter")
        print("   to stable pairs. The 3 m noise floor is intrinsic centroid wobble,")
        print("   not split/merge artifacts. K=3 expected gain ≈ √3 ≈ 1.7×.")
    else:
        print("↔ INCONCLUSIVE — split/merge contributes some jitter inflation but")
        print("   not the dominant effect. K-accumulation gain expected between √K")
        print("   and 2× √K.")

if __name__ == "__main__":
    main()
