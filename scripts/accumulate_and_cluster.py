#!/usr/bin/env python3
"""K-frame point cloud accumulation + DBSCAN, drop-in replacement for the
existing clusters_sweetspot/ pipeline (Phase-4).

Reads:
  - Per-frame obstacle CSVs ({obstacles_dir}/obstacles_NNNNNN.csv,
    schema: x,y,z in current-ego frame, post-RANSAC ground removal).
  - SLAM ego poses (data/poses_slam_full.csv).

For each target frame f:
  1. Collect obstacle points from frames [f - K + 1, f].
  2. For each non-target frame f', transform points into f's ego frame
     via T_ego_f_ego_f' = T_world_ego(f)^-1 · T_world_ego(f').
     (Identity for f' == f.)
  3. DBSCAN the union with the same (eps, min_samples) as the K=1 baseline
     (sklearn implementation). Keep all points in the output, including
     noise points (cluster_id = -1) — matches the existing schema so
     clusters_to_detections.py doesn't need changes.
  4. Write clusters_kN/clusters_NNNNNN.csv with x,y,z,cluster_id rows.

The output directory schema is identical to clusters_sweetspot/ —
clusters_to_detections.py + tracker_runner consume it unchanged.

Usage:
  python3 scripts/accumulate_and_cluster.py \
      --obstacles-dir /media/nishant/SeeGayt2/terra_perceive/m4_perframe/obstacles \
      --poses-csv     data/poses_slam_full.csv \
      --eps           0.5 \
      --min-samples   10 \
      --K             3 \
      --out-dir       /media/nishant/SeeGayt2/terra_perceive/m4_perframe/clusters_k3 \
      --frame-start   0 \
      --frame-end     2848

Caveats:
  - Uses sklearn.cluster.DBSCAN (same algorithm as the C++ dbscan_cli, but
    implemented in Python). For the same (eps, min_samples) on the same
    points, output should be identical modulo numerical floor.
  - K=1 with this script should reproduce clusters_sweetspot/ exactly
    (modulo C++ vs sklearn noise) — that's a useful sanity check.
  - K-accumulation does NOT re-run RANSAC; it accumulates the already-
    ground-removed obstacle points. Ground RANSAC is per-frame.
"""
from __future__ import annotations
import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN


def yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2 * (qw * qz + qx * qy),
                      1 - 2 * (qy * qy + qz * qz))


def load_poses(csv_path):
    """frame_id -> 3x3 SE(2) homogeneous matrix T_world_ego."""
    poses = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            fid = int(row["frame_id"])
            yaw = yaw_from_quat(float(row["qx"]), float(row["qy"]),
                                float(row["qz"]), float(row["qw"]))
            tx = float(row["tx"]); ty = float(row["ty"])
            c, s = math.cos(yaw), math.sin(yaw)
            T = np.array([[c, -s, tx],
                          [s,  c, ty],
                          [0,  0,  1]], dtype=np.float64)
            poses[fid] = T
    return poses


def load_obstacles(obstacles_dir, fid):
    """Return Nx3 array of (x, y, z) points in ego frame, or None if missing."""
    path = obstacles_dir / f"obstacles_{fid:06d}.csv"
    if not path.exists():
        return None
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 3))
    if arr.ndim == 1:        # single-row file
        arr = arr.reshape(1, -1)
    return arr[:, :3]


def transform_xy(T, points_xyz):
    """Apply 3x3 SE(2) transform to xy of an Nx3 array; pass z through."""
    if points_xyz.shape[0] == 0:
        return points_xyz
    xy = points_xyz[:, :2]                      # Nx2
    homog = np.hstack([xy, np.ones((xy.shape[0], 1))])  # Nx3
    out_xy = (T @ homog.T).T[:, :2]             # Nx2
    out = np.empty_like(points_xyz)
    out[:, :2] = out_xy
    out[:, 2]  = points_xyz[:, 2]
    return out


def write_cluster_csv(path, points_xyz, labels):
    """Write x,y,z,cluster_id rows in the same schema as clusters_sweetspot/."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("x,y,z,cluster_id\n")
        for (x, y, z), c in zip(points_xyz, labels):
            f.write(f"{x},{y},{z},{int(c)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obstacles-dir", required=True, type=Path)
    ap.add_argument("--poses-csv",     required=True, type=Path)
    ap.add_argument("--out-dir",       required=True, type=Path)
    ap.add_argument("--eps",           type=float, default=0.5)
    ap.add_argument("--min-samples",   type=int,   default=10)
    ap.add_argument("--K",             type=int,   default=3,
                    help="number of frames to accumulate (1 = baseline)")
    ap.add_argument("--frame-start",   type=int, default=0)
    ap.add_argument("--frame-end",     type=int, required=True)
    ap.add_argument("--print-every",   type=int, default=200)
    args = ap.parse_args()

    print(f"[accumulate] K={args.K} eps={args.eps} min_samples={args.min_samples}")
    print(f"             obstacles: {args.obstacles_dir}")
    print(f"             out:       {args.out_dir}")

    poses = load_poses(args.poses_csv)
    print(f"[accumulate] loaded {len(poses)} ego poses")

    written = 0
    skipped_no_target = 0
    skipped_no_pose   = 0
    n_clusters_total  = 0
    n_points_total    = 0

    for f in range(args.frame_start, args.frame_end + 1):
        # Need an obstacle file at the target frame to write output
        target_pts = load_obstacles(args.obstacles_dir, f)
        if target_pts is None:
            skipped_no_target += 1
            continue
        if f not in poses:
            # No pose for target; can only do K=1 (no transform needed),
            # but skip otherwise.
            if args.K > 1:
                skipped_no_pose += 1
                continue
            T_world_ego_f_inv = np.eye(3)
        else:
            T_world_ego_f_inv = np.linalg.inv(poses[f])

        # Accumulate
        chunks = [target_pts]
        for k in range(1, args.K):
            f_prev = f - k
            if f_prev < 0:
                break
            pts_prev = load_obstacles(args.obstacles_dir, f_prev)
            if pts_prev is None or pts_prev.shape[0] == 0:
                continue
            if f_prev not in poses:
                continue
            T_world_ego_prev = poses[f_prev]
            # Transform prev_ego → world → target_ego
            T_target_ego_from_prev_ego = T_world_ego_f_inv @ T_world_ego_prev
            pts_in_target_ego = transform_xy(T_target_ego_from_prev_ego, pts_prev)
            chunks.append(pts_in_target_ego)

        points = np.vstack(chunks) if chunks else np.empty((0, 3))

        if points.shape[0] < args.min_samples:
            # Not enough points to form any cluster — write empty file
            labels = -np.ones(points.shape[0], dtype=int)
        else:
            db = DBSCAN(eps=args.eps, min_samples=args.min_samples)
            labels = db.fit_predict(points)  # 3D — matches src/dbscan.cpp (Eigen::Vector3f)

        out_path = args.out_dir / f"clusters_{f:06d}.csv"
        write_cluster_csv(out_path, points, labels)
        written += 1
        n_clusters = labels.max() + 1 if labels.size and labels.max() >= 0 else 0
        n_clusters_total += n_clusters
        n_points_total   += points.shape[0]

        if args.print_every > 0 and (f - args.frame_start) % args.print_every == 0:
            print(f"[accumulate] frame {f:>5} / {args.frame_end}  "
                  f"points={points.shape[0]}  clusters={n_clusters}")

    print()
    print(f"[accumulate] DONE — wrote {written} files to {args.out_dir}")
    print(f"             skipped (no obstacle file): {skipped_no_target}")
    print(f"             skipped (no pose for K>1):  {skipped_no_pose}")
    if written:
        print(f"             avg clusters/frame: {n_clusters_total/written:.1f}")
        print(f"             avg points/frame:   {n_points_total/written:.0f}")


if __name__ == "__main__":
    main()
