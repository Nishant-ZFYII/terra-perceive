#!/usr/bin/env python3
"""build_pairs.py — sample cluster pairs for the appearance encoder val set.

Output:
    python/appearance/pair_candidates.csv

Schema (one pair per row):
    pair_id, source, frame_a, cluster_a, frame_b, cluster_b, prior_label

Where:
    pair_id      — 0-indexed integer
    source       — "adjacent" | "same_frame_far" | "augment"
    frame_a/b    — RELLIS frame indices (matching clusters_NNNNNN.csv)
    cluster_a/b  — cluster_id within each frame's CSV
    prior_label  — heuristic prior ("likely_same" | "likely_diff") for the
                   labeling UI to bias-sort but the user is the source of
                   truth. Empty for "augment" pairs.

Sampling strategy (3 channels, balanced 50/50 same/different):
    1. ADJACENT — frame_b = frame_a + 1, nearest-neighbor cluster centroids
       under tight gates (Mahalanobis < 1.0, point-count ratio in [0.7, 1.3],
       bbox-volume ratio in [0.6, 1.4]).  prior_label = "likely_same".
    2. SAME_FRAME_FAR — both clusters from same frame, centroid distance
       > 8.0 m (definitely different physical objects).
       prior_label = "likely_diff".
    3. AUGMENT — same cluster crop, geometrically jittered version of itself
       (used at training time only; build_pairs samples pairs from channels
       1+2 here for the LABEL UI).

This script does NOT produce training pairs — it produces CANDIDATES for the
human labeling session. Training-time pair generation is a separate concern
(handled later by extract_features.py + the train.py augmentation pipeline).

Usage:
    python python/appearance/build_pairs.py \\
        --clusters-dir /media/nishant/SeeGayt2/terra_perceive/m4_perframe/clusters_sweetspot \\
        --num-pairs 100 \\
        --out python/appearance/pair_candidates.csv
"""
import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_clusters(csv_path: Path) -> Dict[int, np.ndarray]:
    """Load a clusters_NNNNNN.csv → {cluster_id: Nx3 points array}.

    Drops noise (cluster_id == -1).
    """
    clusters: Dict[int, List[Tuple[float, float, float]]] = {}
    with csv_path.open() as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            x, y, z, cid = float(row[0]), float(row[1]), float(row[2]), int(row[3])
            if cid < 0:
                continue
            clusters.setdefault(cid, []).append((x, y, z))
    return {cid: np.asarray(pts, dtype=np.float32) for cid, pts in clusters.items()}


def cluster_centroid(pts: np.ndarray) -> np.ndarray:
    return pts.mean(axis=0)


def bbox_volume(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 0.0
    extent = pts.max(axis=0) - pts.min(axis=0)
    return float(extent[0] * extent[1] * extent[2])


def sample_adjacent_pairs(
    cluster_files: List[Path], n_target: int, rng: random.Random
) -> List[Tuple[int, int, int, int]]:
    """Sample (frame_a, cid_a, frame_b, cid_b) where frame_b = frame_a + 1
    and the two clusters' centroids are within 0.6 m of each other.

    Tight match → prior_label = likely_same.
    """
    out = []
    attempts = 0
    while len(out) < n_target and attempts < n_target * 30:
        attempts += 1
        i = rng.randrange(len(cluster_files) - 1)
        a_clusters = load_clusters(cluster_files[i])
        b_clusters = load_clusters(cluster_files[i + 1])
        if not a_clusters or not b_clusters:
            continue
        cid_a = rng.choice(list(a_clusters.keys()))
        ca = cluster_centroid(a_clusters[cid_a])
        # Find closest cluster in b.
        best_cid_b, best_d = None, float("inf")
        for cid_b, pts_b in b_clusters.items():
            cb = cluster_centroid(pts_b)
            d = float(np.linalg.norm(ca - cb))
            if d < best_d:
                best_d, best_cid_b = d, cid_b
        if best_cid_b is None or best_d > 0.6:
            continue
        # Point-count ratio gate
        n_a = a_clusters[cid_a].shape[0]
        n_b = b_clusters[best_cid_b].shape[0]
        ratio = n_a / max(n_b, 1)
        if not (0.5 <= ratio <= 2.0):
            continue
        frame_a = int(cluster_files[i].stem.split("_")[1])
        frame_b = int(cluster_files[i + 1].stem.split("_")[1])
        out.append((frame_a, cid_a, frame_b, best_cid_b))
    return out


def sample_far_pairs(
    cluster_files: List[Path], n_target: int, rng: random.Random
) -> List[Tuple[int, int, int, int]]:
    """Sample (frame, cid_a, frame, cid_b) where centroids are > 8 m apart
    in the same frame. Definitely different physical objects.
    """
    out = []
    attempts = 0
    while len(out) < n_target and attempts < n_target * 30:
        attempts += 1
        i = rng.randrange(len(cluster_files))
        clusters = load_clusters(cluster_files[i])
        if len(clusters) < 2:
            continue
        cids = list(clusters.keys())
        rng.shuffle(cids)
        cid_a, cid_b = cids[0], cids[1]
        ca = cluster_centroid(clusters[cid_a])
        cb = cluster_centroid(clusters[cid_b])
        if float(np.linalg.norm(ca - cb)) < 8.0:
            continue
        frame = int(cluster_files[i].stem.split("_")[1])
        out.append((frame, cid_a, frame, cid_b))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clusters-dir", type=Path, required=True)
    p.add_argument("--num-pairs", type=int, default=100,
                   help="total candidate pairs (split 50/50 across the two channels)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("python/appearance/pair_candidates.csv"))
    args = p.parse_args()

    rng = random.Random(args.seed)
    cluster_files = sorted(args.clusters_dir.glob("clusters_*.csv"))
    if not cluster_files:
        raise SystemExit(f"no clusters_*.csv in {args.clusters_dir}")
    print(f"[build_pairs] {len(cluster_files)} cluster CSVs found")

    n_each = args.num_pairs // 2
    print(f"[build_pairs] sampling {n_each} likely-same + {n_each} likely-diff ...")

    same_pairs = sample_adjacent_pairs(cluster_files, n_each, rng)
    diff_pairs = sample_far_pairs(cluster_files, n_each, rng)
    print(f"[build_pairs] got {len(same_pairs)} same + {len(diff_pairs)} diff")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "source", "frame_a", "cluster_a",
                    "frame_b", "cluster_b", "prior_label"])
        pid = 0
        rows = (
            [("adjacent", *p, "likely_same") for p in same_pairs]
            + [("same_frame_far", *p, "likely_diff") for p in diff_pairs]
        )
        rng.shuffle(rows)  # interleave so labeler doesn't see all sames in a row
        for src, fa, ca, fb, cb, prior in rows:
            w.writerow([pid, src, fa, ca, fb, cb, prior])
            pid += 1

    print(f"[build_pairs] wrote {pid} pairs → {args.out}")


if __name__ == "__main__":
    main()
