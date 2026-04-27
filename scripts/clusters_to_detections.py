#!/usr/bin/env python3
"""clusters_to_detections.py — convert per-frame DBSCAN cluster CSVs into the
single-CSV detections schema that tracker_runner consumes.

Per-frame input (one file per frame):
    {clusters_dir}/clusters_NNNNNN.csv
        x,y,z,cluster_id    (cluster_id = -1 = noise; ≥ 0 = cluster index)

Optional input (P3-M13 appearance pipeline; if --features-dir set):
    {features_dir}/features_NNNNNN.csv
        cluster_id, n_log, bbox_x, bbox_y, bbox_z, min_z, eig_r1, eig_r2, range
        (produced by python/appearance/extract_features.py)

Output (one CSV across all frames):
    {out_csv}
        frame_id,det_id,x,y,class_id,gt_track_id

Optional output (when --features-dir is set):
    {features_out}                                 (defaults to
                                                    {out_csv}.features.csv)
        frame_id,det_id,n_log,bbox_x,bbox_y,bbox_z,min_z,eig_r1,eig_r2,range

For each frame:
  - Group rows by cluster_id (skip noise rows where cluster_id < 0).
  - Compute the (x, y) centroid of each cluster — that's one detection.
  - det_id = local cluster index within the frame (0, 1, 2, ...).
  - class_id = 0 (no semantic labels available from RELLIS LiDAR alone).
  - gt_track_id = -1 (no ground-truth tracking labels for RELLIS).
  - If features-dir is set, the per-frame features CSV is parsed and
    joined by cluster_id; one features row per detection row, in
    matching (frame_id, det_id) order.

This decouples clustering (M4 DBSCAN) from tracking (M4 SORT) and from
appearance encoding (M13 MLP). The same tracker_runner CLI that ate
synthetic CSVs in Ablations A-F now eats real RELLIS-derived detections,
optionally enriched with appearance features for the M13 cost-matrix.

Usage:
    # M4/M12 (position-only):
    python scripts/clusters_to_detections.py \\
        --clusters-dir /media/.../m4_perframe/clusters_sweetspot \\
        --frame-start 0 --frame-end 2848 \\
        --out results_m4/ablation_g/rellis_detections.csv

    # M13 (also dumps features):
    python scripts/clusters_to_detections.py \\
        --clusters-dir   /media/.../m4_perframe/clusters_sweetspot \\
        --features-dir   /media/.../m4_perframe/appearance \\
        --frame-start 0 --frame-end 2848 \\
        --out results_m4/ablation_g/rellis_detections.csv \\
        --features-out results_m4/ablation_g/rellis_features.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


FEATURE_COLS = ["n_log", "bbox_x", "bbox_y", "bbox_z",
                "min_z", "eig_r1", "eig_r2", "range"]


def load_clusters_one_frame(path: Path
                            ) -> Dict[int, List[Tuple[float, float]]]:
    """cluster_id -> [(x, y), ...] for clusters in ONE frame. Noise dropped."""
    out: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            cid = int(row["cluster_id"])
            if cid < 0:
                continue   # skip noise
            out[cid].append((float(row["x"]), float(row["y"])))
    return out


def load_features_one_frame(path: Path) -> Dict[int, List[float]]:
    """cluster_id -> [n_log, bbox_x, bbox_y, bbox_z, min_z, eig_r1,
    eig_r2, range] for ONE frame. Returns {} if path missing."""
    if not path.exists():
        return {}
    out: Dict[int, List[float]] = {}
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            cid = int(row["cluster_id"])
            out[cid] = [float(row[c]) for c in FEATURE_COLS]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clusters-dir", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, default=None,
                   help="if set, emit a parallel features CSV per detection")
    p.add_argument("--frame-start",  type=int, required=True)
    p.add_argument("--frame-end",    type=int, required=True)
    p.add_argument("--out",          type=Path, required=True)
    p.add_argument("--features-out", type=Path, default=None,
                   help="features CSV path (defaults to <out>.features.csv "
                        "when --features-dir is set)")
    args = p.parse_args()

    emit_features = args.features_dir is not None
    if emit_features and args.features_out is None:
        args.features_out = args.out.with_suffix(".features.csv")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    feat_writer = None
    feat_handle = None
    if emit_features:
        args.features_out.parent.mkdir(parents=True, exist_ok=True)
        feat_handle = args.features_out.open("w", newline="")
        feat_writer = csv.writer(feat_handle)
        feat_writer.writerow(["frame_id", "det_id"] + FEATURE_COLS)

    n_frames = 0
    n_dets = 0
    n_feat_missing = 0
    try:
        with args.out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id", "det_id", "x", "y", "class_id", "gt_track_id"])

            for fid in range(args.frame_start, args.frame_end + 1):
                path = args.clusters_dir / f"clusters_{fid:06d}.csv"
                if not path.exists():
                    continue
                n_frames += 1

                clusters = load_clusters_one_frame(path)
                features: Dict[int, List[float]] = {}
                if emit_features:
                    features = load_features_one_frame(
                        args.features_dir / f"features_{fid:06d}.csv")

                # Sort by cluster_id for deterministic det_id assignment —
                # MUST match the same sort used in extract_features.py so
                # the join below pairs correctly.
                for det_id, cid in enumerate(sorted(clusters.keys())):
                    pts = clusters[cid]
                    if not pts:
                        continue
                    cx = sum(p[0] for p in pts) / len(pts)
                    cy = sum(p[1] for p in pts) / len(pts)
                    w.writerow([fid, det_id, f"{cx:.6f}", f"{cy:.6f}", 0, -1])
                    n_dets += 1

                    if emit_features:
                        feat = features.get(cid)
                        if feat is None:
                            # Tiny clusters can be present in clusters_*.csv
                            # but missing in features_*.csv (e.g., if
                            # extract_features.py was re-run with a stricter
                            # min-points threshold). Emit zero features so
                            # row counts stay aligned; tracker_runner's
                            # cosine-similarity term safely returns 1.0.
                            feat = [0.0] * 8
                            n_feat_missing += 1
                        feat_writer.writerow(
                            [fid, det_id] + [f"{v:.6f}" for v in feat])
    finally:
        if feat_handle is not None:
            feat_handle.close()

    print(f"[clusters_to_detections] wrote {n_dets} detections "
          f"from {n_frames} frames to {args.out}")
    if emit_features:
        print(f"[clusters_to_detections] wrote features to {args.features_out}")
        if n_feat_missing:
            print(f"[clusters_to_detections] WARN: {n_feat_missing} "
                  f"detections had no matching features (filled with zeros)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
