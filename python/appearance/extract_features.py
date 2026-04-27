#!/usr/bin/env python3
"""extract_features.py — per-cluster 8-dim hand-crafted feature dump.

Reads:
    {clusters_dir}/clusters_NNNNNN.csv   (per-frame DBSCAN output, schema:
                                          x,y,z,cluster_id; cluster_id < 0
                                          is noise and is dropped)

Writes:
    {out_dir}/features_NNNNNN.csv        (per-frame, schema:
                                          cluster_id, n_log, bbox_x, bbox_y,
                                          bbox_z, min_z, eig_r1, eig_r2,
                                          range)
    {out_dir}/corpus_stats.json          (training-time z-score: per-feature
                                          mean and std across the entire
                                          corpus, used by train.py and later
                                          baked into appearance_model_weights.hpp)

The 8 features per cluster (matches `include/appearance_encoder.hpp`):
    1. n_log        : log of point count — captures cluster size / density
    2. bbox_x       : axis-aligned bbox extent in x (meters)
    3. bbox_y       :                              y
    4. bbox_z       :                              z (height of cluster)
    5. min_z        : lowest z of the cluster — ground-relative height
                      (after RANSAC-style ground segmentation upstream)
    6. eig_r1       : λ_1 / Σλ from 3x3 cluster covariance — captures
                      "stick-like" vs "blob-like" vs "plate-like" shape
    7. eig_r2       : λ_2 / Σλ — same shape descriptor, second component
    8. range        : √(c_x² + c_y²) — distance from LiDAR sensor.
                      LiDAR cluster appearance changes with range
                      (point density drops at distance), so we encode it
                      as an explicit feature instead of letting the network
                      try to recover it.

Why hand-crafted vs PointNet:
    Average DBSCAN cluster on RELLIS has ~30 points. PointNet's max-pool
    needs 100s of points per cluster to extract per-point features that
    survive the pool. With 30 points, the max-pool collapses to a near-
    constant per-cluster vector — useless for metric learning. Hand-crafted
    geometric features sidestep this.

Reference: see Wojke 2017 §3 for the cost-matrix integration these
features feed into; see Hermans 2017 §4 for the triplet-loss training that
will consume the features dumped here.

Usage:
    python python/appearance/extract_features.py \\
        --clusters-dir /media/.../clusters_sweetspot \\
        --out-dir      /media/.../m4_perframe/appearance \\
        --workers      0   # (default; >0 enables multiprocessing)

Wall-clock: ~30 sec serial on 2849 frames (RELLIS-3D); <10 sec with
--workers 8.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


FEATURE_NAMES = [
    "n_log", "bbox_x", "bbox_y", "bbox_z",
    "min_z", "eig_r1", "eig_r2", "range",
]


# -----------------------------------------------------------------------------
# Per-cluster feature extraction. The CRITICAL function — must match the
# C++ inference path in src/appearance_encoder.cpp::encode() Step 1, which
# applies the same z-score normalization to whatever features came out of
# the upstream cluster-to-feature pipeline. Drift between the two will
# poison embeddings silently.
# -----------------------------------------------------------------------------

def cluster_features(pts: np.ndarray) -> np.ndarray:
    """Compute the 8-dim feature vector for one cluster.

    pts: (N, 3) numpy array of cluster points (x, y, z) in LiDAR frame.

    Returns: (8,) float32 array in the FEATURE_NAMES order.
    """
    n = pts.shape[0]

    # Single-point clusters are pathological for shape descriptors — bbox
    # and PCA both degenerate. Fill with sensible defaults instead of
    # NaN-ing out the row. These tiny clusters get filtered out in
    # build_pairs.py anyway (we sample pairs only from clusters with
    # ≥ 0.5x and ≤ 2.0x point-count ratio gates), so they have minimal
    # effect on training but they do show up in the per-frame features
    # files.
    if n < 2:
        if n == 0:
            return np.zeros(8, dtype=np.float32)
        return np.array([
            np.log(1.0),                       # n_log
            0.0, 0.0, 0.0,                     # bbox sizes
            float(pts[0, 2]),                  # min_z
            1.0, 0.0,                          # degenerate eigval ratios
            float(np.sqrt(pts[0, 0] ** 2 + pts[0, 1] ** 2)),
        ], dtype=np.float32)

    # 1. log point count.
    n_log = float(np.log(n))

    # 2–4. Axis-aligned bbox dimensions.
    bbox = pts.max(axis=0) - pts.min(axis=0)

    # 5. min_z — bottom of the cluster. After upstream RANSAC ground
    # segmentation this is "height above ground" (in the locally-flattened
    # frame). In the un-flattened LiDAR frame it's a noisier proxy that
    # still carries signal: tall objects (trees) have higher min_z than
    # ground patches.
    min_z = float(pts[:, 2].min())

    # 6–7. PCA eigenvalue ratios — shape descriptors.
    #
    # Cluster covariance matrix is 3x3 SPD. Its eigenvalues are non-
    # negative; sort descending and normalize by sum. The first two ratios
    # span a 2-simplex (r1 + r2 + r3 = 1, r3 = 1 - r1 - r2), which encodes
    # the cluster's elongation:
    #     r1 ≈ 1.0      → stick (one dominant direction; e.g. tree trunk)
    #     r1 ≈ r2       → plate (two dominant directions; e.g. road sign)
    #     r1 ≈ r2 ≈ r3  → blob (isotropic; e.g. small shrub)
    #
    # np.cov(pts.T) wants (3, N) input — pts.T transposes (N, 3) → (3, N).
    # eigvalsh is the symmetric variant of eigvals; faster + more stable
    # on covariance matrices than the general eig. Returns eigenvalues
    # in ASCENDING order; we reverse to descending.
    cov = np.cov(pts.T)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.maximum(eigvals, 0.0)         # clip tiny float-negatives
    total = float(eigvals.sum())
    if total > 1e-12:
        r1 = float(eigvals[0] / total)
        r2 = float(eigvals[1] / total)
    else:
        # All points coincident (cov is zero). Defensible default.
        r1, r2 = 1.0, 0.0

    # 8. centroid range (xy distance from LiDAR origin).
    centroid = pts.mean(axis=0)
    rng = float(np.sqrt(centroid[0] ** 2 + centroid[1] ** 2))

    return np.array(
        [n_log, float(bbox[0]), float(bbox[1]), float(bbox[2]),
         min_z, r1, r2, rng],
        dtype=np.float32,
    )


# -----------------------------------------------------------------------------
# Per-frame I/O.
# -----------------------------------------------------------------------------

def load_cluster_csv(path: Path) -> Dict[int, np.ndarray]:
    """{cluster_id: (N, 3) float32 array}. Drops noise (cluster_id < 0)."""
    out: Dict[int, List[Tuple[float, float, float]]] = {}
    with path.open() as f:
        r = csv.reader(f)
        next(r)                                # skip header
        for row in r:
            x, y, z, cid = float(row[0]), float(row[1]), float(row[2]), int(row[3])
            if cid < 0:
                continue
            out.setdefault(cid, []).append((x, y, z))
    return {cid: np.asarray(pts, dtype=np.float32) for cid, pts in out.items()}


def process_one_frame(args: Tuple[Path, Path]) -> Tuple[str, np.ndarray]:
    """Worker: load one clusters CSV, write the matching features CSV,
    and return a (frame_stem, all_features_for_this_frame) pair so the
    parent can build corpus stats without reloading.
    """
    in_path, out_path = args
    clusters = load_cluster_csv(in_path)

    if not clusters:
        # Write an empty features file so downstream tools see consistent
        # frame coverage.
        with out_path.open("w", newline="") as f:
            csv.writer(f).writerow(["cluster_id"] + FEATURE_NAMES)
        return in_path.stem, np.empty((0, 8), dtype=np.float32)

    rows = []
    feats = []
    for cid, pts in sorted(clusters.items()):
        feat = cluster_features(pts)
        rows.append([cid] + feat.tolist())
        feats.append(feat)

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id"] + FEATURE_NAMES)
        w.writerows(rows)

    return in_path.stem, np.asarray(feats, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--clusters-dir", type=Path, required=True,
                   help="directory of clusters_NNNNNN.csv")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="where to write features_NNNNNN.csv + corpus_stats.json")
    p.add_argument("--workers", type=int, default=0,
                   help=">0 enables multiprocessing across frames")
    p.add_argument("--frame-glob", type=str, default="clusters_*.csv",
                   help="glob to select cluster CSVs (default: clusters_*.csv)")
    args = p.parse_args()

    cluster_files = sorted(args.clusters_dir.glob(args.frame_glob))
    if not cluster_files:
        raise SystemExit(f"no {args.frame_glob} found under {args.clusters_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract_features] {len(cluster_files)} cluster CSVs found")
    print(f"[extract_features] writing to {args.out_dir}")

    # Build per-frame input/output paths.
    tasks = []
    for cf in cluster_files:
        # clusters_001078.csv → features_001078.csv
        frame_id = cf.stem.split("_", 1)[1]
        out_path = args.out_dir / f"features_{frame_id}.csv"
        tasks.append((cf, out_path))

    # Iterate. Multiprocessing is optional because the per-frame work is
    # cheap and the I/O hits the same disk; it helps modestly above
    # ~1000 frames.
    all_feats: List[np.ndarray] = []
    if args.workers > 0:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_one_frame, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                _, feats = fut.result()
                if feats.size:
                    all_feats.append(feats)
                if i % 200 == 0 or i == len(tasks):
                    print(f"  processed {i}/{len(tasks)} frames", file=sys.stderr)
    else:
        for i, t in enumerate(tasks, 1):
            _, feats = process_one_frame(t)
            if feats.size:
                all_feats.append(feats)
            if i % 200 == 0 or i == len(tasks):
                print(f"  processed {i}/{len(tasks)} frames", file=sys.stderr)

    if not all_feats:
        raise SystemExit("[extract_features] no clusters extracted — "
                         "is clusters_dir empty?")

    # ------------------------------------------------------------------
    # Corpus stats — z-score normalization parameters baked into the
    # weights header at training time. C++ encoder applies (x - mean)/std
    # in encode() Step 1; if these don't match the values the network was
    # trained against, embeddings drift silently.
    # ------------------------------------------------------------------
    corpus = np.concatenate(all_feats, axis=0)        # (N_total_clusters, 8)
    mean = corpus.mean(axis=0)
    std  = corpus.std(axis=0)
    # Guard against zero-std features (constant across the corpus —
    # shouldn't happen in practice, but cheap insurance against div-by-0
    # at training time).
    std = np.where(std < 1e-6, 1.0, std)

    stats = {
        "n_clusters_total": int(corpus.shape[0]),
        "n_frames":         len(cluster_files),
        "feature_names":    FEATURE_NAMES,
        "mean":             mean.astype(float).tolist(),
        "std":              std.astype(float).tolist(),
    }
    stats_path = args.out_dir / "corpus_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print()
    print(f"[extract_features] {corpus.shape[0]} clusters across "
          f"{len(cluster_files)} frames")
    print(f"[extract_features] feature stats:")
    print(f"  {'name':<10}  {'mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:<10}  {mean[i]:>10.3f}  {std[i]:>10.3f}  "
              f"{corpus[:, i].min():>10.3f}  {corpus[:, i].max():>10.3f}")
    print()
    print(f"[extract_features] wrote corpus_stats.json → {stats_path}")


if __name__ == "__main__":
    main()
