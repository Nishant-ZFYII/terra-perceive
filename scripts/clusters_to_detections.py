#!/usr/bin/env python3
"""clusters_to_detections.py — convert per-frame DBSCAN cluster CSVs into the
single-CSV detections schema that tracker_runner consumes.

Per-frame input (one file per frame):
    {clusters_dir}/clusters_NNNNNN.csv
        x,y,z,cluster_id    (cluster_id = -1 = noise; ≥ 0 = cluster index)

Output (one CSV across all frames):
    {out_csv}
        frame_id,det_id,x,y,class_id,gt_track_id

For each frame:
  - Group rows by cluster_id (skip noise rows where cluster_id < 0).
  - Compute the (x, y) centroid of each cluster — that's one detection.
  - det_id = local cluster index within the frame (0, 1, 2, ...).
  - class_id = 0 (no semantic labels available from RELLIS LiDAR alone).
  - gt_track_id = -1 (no ground-truth tracking labels for RELLIS).

This decouples clustering (M4 DBSCAN) from tracking (M4 SORT). The same
tracker_runner CLI that ate synthetic CSVs in Ablations A-F now eats
real RELLIS-derived detections.

Usage:
    python scripts/clusters_to_detections.py \\
        --clusters-dir /media/.../m4_perframe/clusters_sweetspot \\
        --frame-start 0 --frame-end 2848 \\
        --out results_m4/ablation_g/rellis_detections.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clusters-dir", type=Path, required=True)
    p.add_argument("--frame-start",  type=int, required=True)
    p.add_argument("--frame-end",    type=int, required=True)
    p.add_argument("--out",          type=Path, required=True)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_frames = 0
    n_dets = 0
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "det_id", "x", "y", "class_id", "gt_track_id"])

        for fid in range(args.frame_start, args.frame_end + 1):
            path = args.clusters_dir / f"clusters_{fid:06d}.csv"
            if not path.exists():
                continue
            n_frames += 1

            clusters = load_clusters_one_frame(path)
            # Sort by cluster_id for deterministic det_id assignment.
            for det_id, cid in enumerate(sorted(clusters.keys())):
                pts = clusters[cid]
                if not pts:
                    continue
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                w.writerow([fid, det_id, f"{cx:.6f}", f"{cy:.6f}", 0, -1])
                n_dets += 1

    print(f"[clusters_to_detections] wrote {n_dets} detections "
          f"from {n_frames} frames to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
