#!/usr/bin/env python3
"""Render the full-sequence DBSCAN clustering animation for the M10 blog.

Reads per-frame `clusters_NNNNNN.csv` files (x, y, z, cluster_id) produced by
`dbscan_cli` at the sweet-spot params, and renders a top-down (x-y) animation
where points are colored by cluster (gray for noise, distinct hues per
cluster). Output: one MP4 (full frame rate) and one stride-subsampled GIF
suitable for embedding inline in a blog post.

Per ablation pre-flight rule #8 — pre-compute the full-trajectory bbox over
ALL frames before starting the animation loop, so the viewport is sized to
fit the whole recording, not just frame 0.

Usage:
    python scripts/dbscan_animate.py \\
        --clusters-dir /media/.../m4_perframe/clusters_sweetspot \\
        --frame-start 0 --frame-end 2486 \\
        --eps 0.5 --min-points 10 \\
        --fps 10 \\
        --out-mp4 results_m4/ablation_g/dbscan_animation.mp4 \\
        --out-gif results_m4/ablation_g/dbscan_animation.gif \\
        --gif-stride 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np


def load_cluster_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    xy: List[Tuple[float, float]] = []
    cid: List[int] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            xy.append((float(row["x"]), float(row["y"])))
            cid.append(int(row["cluster_id"]))
    if not xy:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    return np.array(xy), np.array(cid)


def list_available_frames(clusters_dir: Path,
                          frame_start: int,
                          frame_end: int) -> List[int]:
    out = []
    for fid in range(frame_start, frame_end + 1):
        p = clusters_dir / f"clusters_{fid:06d}.csv"
        if p.exists():
            out.append(fid)
    return out


def compute_global_bbox(clusters_dir: Path, frame_ids: List[int],
                        sample_every: int = 10
                        ) -> Tuple[float, float, float, float]:
    """Pre-compute bounding box across the full sequence (rule #8).
    Subsampled to keep this fast on long recordings."""
    xs, ys = [], []
    for fid in frame_ids[::sample_every]:
        xy, _ = load_cluster_csv(clusters_dir / f"clusters_{fid:06d}.csv")
        if xy.size:
            xs.append(xy[:, 0])
            ys.append(xy[:, 1])
    if not xs:
        return (-50.0, 50.0, -50.0, 50.0)
    all_x = np.concatenate(xs); all_y = np.concatenate(ys)
    margin = 5.0
    return (float(all_x.min()) - margin, float(all_x.max()) + margin,
            float(all_y.min()) - margin, float(all_y.max()) + margin)


def render_animation(clusters_dir: Path,
                     frame_ids: List[int],
                     bbox: Tuple[float, float, float, float],
                     eps: float, mp: int,
                     fps: int,
                     out_path: Path) -> None:
    plt.rcParams.update({
        "font.size":      11,
        "axes.titlesize": 13,
    })
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.tab20

    def draw(fid: int) -> None:
        ax.clear()
        ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x  [m]"); ax.set_ylabel("y  [m]")
        xy, cid = load_cluster_csv(clusters_dir / f"clusters_{fid:06d}.csv")
        if xy.size:
            noise = cid < 0
            if noise.any():
                ax.scatter(xy[noise, 0], xy[noise, 1],
                           s=3, c="lightgray", alpha=0.5, zorder=1)
            for k in sorted(set(cid[~noise])):
                m = cid == k
                ax.scatter(xy[m, 0], xy[m, 1],
                           s=8, color=cmap(int(k) % 20), edgecolor="none",
                           zorder=2)
            K = len(set(cid[~noise]))
            N_noise = int(noise.sum())
        else:
            K = 0; N_noise = 0
        ax.set_title(
            f"DBSCAN (eps={eps}m, mp={mp})  frame {fid}   "
            f"{K} clusters, {N_noise} noise"
        )

    anim = FuncAnimation(fig, draw, frames=frame_ids, interval=1000.0 / fps)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".mp4":
        anim.save(str(out_path), writer=FFMpegWriter(fps=fps, bitrate=2400))
    elif suffix == ".gif":
        anim.save(str(out_path), writer=PillowWriter(fps=fps))
    else:
        sys.exit(f"unsupported suffix: {suffix}")
    plt.close(fig)
    print(f"[dbscan_animate] wrote {out_path}  ({len(frame_ids)} frames)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clusters-dir", type=Path, required=True)
    p.add_argument("--frame-start",  type=int, required=True)
    p.add_argument("--frame-end",    type=int, required=True)
    p.add_argument("--eps",          type=float, required=True)
    p.add_argument("--min-points",   type=int,   required=True)
    p.add_argument("--fps",          type=int,   default=10)
    p.add_argument("--out-mp4",      type=Path,  required=True)
    p.add_argument("--out-gif",      type=Path,  default=None)
    p.add_argument("--gif-stride",   type=int,   default=1,
                   help="render every Nth frame in the GIF (keep size down)")
    args = p.parse_args()

    frames = list_available_frames(args.clusters_dir,
                                   args.frame_start, args.frame_end)
    if not frames:
        sys.exit("[dbscan_animate] no cluster CSVs found in given range")
    print(f"[dbscan_animate] {len(frames)} frames available")

    bbox = compute_global_bbox(args.clusters_dir, frames)
    print(f"[dbscan_animate] viewport bbox: x=[{bbox[0]:.1f}, {bbox[1]:.1f}] "
          f"y=[{bbox[2]:.1f}, {bbox[3]:.1f}]")

    # MP4 (full frame rate).
    render_animation(args.clusters_dir, frames, bbox,
                     args.eps, args.min_points, args.fps, args.out_mp4)

    # GIF (stride-subsampled to keep filesize sane).
    if args.out_gif is not None:
        gif_frames = frames[::args.gif_stride]
        render_animation(args.clusters_dir, gif_frames, bbox,
                         args.eps, args.min_points,
                         max(1, args.fps // args.gif_stride),
                         args.out_gif)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
