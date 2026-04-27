#!/usr/bin/env python3
"""animate_tracker_3d.py — Waymo-style 3D animation of LiDAR + tracks.

Per-frame render:
  • RAW LiDAR points in faint gray/white as scene context (~90k/frame).
  • Tracked clusters colored by track_id (tab20 colormap), one consistent
    color per persistent track across the whole drive.
  • Axis-aligned 3D bounding boxes drawn around each tracked cluster
    (green wireframe, like the reference Waymo viz).
  • Ego vehicle as a black box at the origin.
  • Frame-id + distinct-track-count badge top-left.
  • Range rings on the ground plane.

Inputs:
    --lidar-dir       data/extracted_frames_full          (KITTI .bin)
    --clusters-dir    {ext_root}/clusters_sweetspot       (clusters_NNNNNN.csv)
    --tracks-csv      results_m4/blog_renders/<cfg>/tracks.csv

Output:
    --out-mp4         <out>.mp4

Camera defaults to a chase-cam-style elevated front-view (elev=20°,
azim=−85°, fixed at ego origin since LiDAR points are already in ego
frame). Override with --elev / --azim / --range.

Wall-clock: matplotlib 3D scatter at 90k points/frame is slow.
Expect ~10–20 sec per output frame at default settings, which means
~5 minutes for a 30-frame stride window or ~30 minutes for a full
580-frame stride-5 render. Use --frame-end to limit, --stride to
subsample, or switch to Open3D for production-grade output.

Usage:
    python scripts/animate_tracker_3d.py \\
        --lidar-dir    data/extracted_frames_full \\
        --clusters-dir /media/.../clusters_sweetspot \\
        --tracks-csv   results_m4/blog_renders/m13_5/tracks.csv \\
        --frame-start 1700 --frame-end 1900 --stride 5 \\
        --out-mp4      results_m4/blog_renders/m13_5/3d.mp4
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
import numpy as np


# -----------------------------------------------------------------------------
# Loaders — same shape as the 2D animator so visual language matches.
# -----------------------------------------------------------------------------

def load_lidar_bin(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 3), dtype=np.float32)
    raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return raw[:, :3]


def load_clusters_xyz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (Nx3 xyz, N cluster_id) — noise rows kept for completeness
    (cluster_id == -1)."""
    if not path.exists():
        return np.empty((0, 3)), np.empty(0, dtype=int)
    pts: List[Tuple[float, float, float]] = []
    cid: List[int] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            pts.append((float(row["x"]), float(row["y"]), float(row["z"])))
            cid.append(int(row["cluster_id"]))
    return np.array(pts, dtype=np.float32), np.array(cid, dtype=np.int32)


def load_tracks_indexed(path: Path) -> Dict[int, List[Tuple[int, float, float]]]:
    """frame_id → list of (track_id, x, y) for that frame's published tracks."""
    out: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            out[int(row["frame_id"])].append((
                int(row["track_id"]),
                float(row["x"]),
                float(row["y"]),
            ))
    return out


def assign_cluster_to_track(
    cluster_xy: np.ndarray, tracks: List[Tuple[int, float, float]],
    max_dist: float = 5.0,
) -> int:
    """Return the closest track_id to this cluster's centroid, or -1."""
    if not tracks:
        return -1
    best_id, best_d2 = -1, max_dist * max_dist
    for tid, tx, ty in tracks:
        d2 = (cluster_xy[0] - tx) ** 2 + (cluster_xy[1] - ty) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_id = tid
    return best_id


# -----------------------------------------------------------------------------
# Bounding box rendering — 12 line segments around an axis-aligned box.
# -----------------------------------------------------------------------------

def aabb_lines(pts: np.ndarray) -> List[np.ndarray]:
    """Return a list of 12 (2x3) line segments tracing the AABB of pts."""
    if pts.shape[0] < 2:
        return []
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    # 8 corners
    c = np.array([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]
    return [np.array([c[a], c[b]]) for a, b in edges]


def draw_aabb(ax, pts: np.ndarray, color: str, lw: float = 0.8) -> None:
    for seg in aabb_lines(pts):
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=lw)


def draw_ego(ax, length: float = 1.5, width: float = 1.0, height: float = 0.6) -> None:
    """Black box at origin with cyan edges — represents the vehicle."""
    half_l, half_w = length / 2, width / 2
    z_lo, z_hi = 0.0, height
    pts = np.array([
        [-half_l, -half_w, z_lo], [half_l, -half_w, z_lo],
        [half_l,  half_w, z_lo], [-half_l, half_w, z_lo],
        [-half_l, -half_w, z_hi], [half_l, -half_w, z_hi],
        [half_l,  half_w, z_hi], [-half_l, half_w, z_hi],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        ax.plot([pts[a, 0], pts[b, 0]],
                [pts[a, 1], pts[b, 1]],
                [pts[a, 2], pts[b, 2]],
                color="cyan", linewidth=1.4, zorder=20)
    # Forward direction arrow (cyan, in -x because RELLIS calib has cam→-x)
    ax.plot([0, -2.5], [0, 0], [height/2, height/2],
            color="cyan", linewidth=2.0, zorder=20)


def draw_range_rings(ax, max_r: float, height: float = 0.0) -> None:
    """Concentric range rings on the ground plane — visual scale anchor."""
    theta = np.linspace(0, 2 * np.pi, 80)
    for r in [10.0, 20.0, 30.0, 40.0]:
        if r > max_r:
            break
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.full_like(x, height)
        ax.plot(x, y, z, color="#3a3a3a", linewidth=0.6, linestyle=":",
                zorder=1)


# -----------------------------------------------------------------------------
# Per-frame draw
# -----------------------------------------------------------------------------

def setup_axes(ax, view_range: float, elev: float, azim: float) -> None:
    ax.set_facecolor("black")
    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)
    ax.set_zlim(-2.0, 8.0)
    ax.set_xlabel("x (m)", color="white")
    ax.set_ylabel("y (m)", color="white")
    ax.set_zlabel("z (m)", color="white")
    ax.tick_params(colors="white", labelsize=7)
    # Make the panes black so it looks like a true 3D scene, not a graph.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_facecolor((0.05, 0.05, 0.05, 1.0))
        axis.pane.set_edgecolor((0.2, 0.2, 0.2, 1.0))
    ax.grid(False)
    ax.view_init(elev=elev, azim=azim)


def draw_frame(ax, frame: int, args, tracks_idx, view_range: float,
               point_decim: int) -> None:
    ax.clear()
    setup_axes(ax, view_range, args.elev, args.azim)

    # 1. Raw LiDAR — faint gray. Decimate to keep render time reasonable;
    #    matplotlib 3D scatter is O(N) per frame and 90k points kills perf.
    full = load_lidar_bin(args.lidar_dir / f"{frame:06d}.bin")
    if full.size:
        full = full[::point_decim]
        ax.scatter(full[:, 0], full[:, 1], full[:, 2],
                   s=0.5, c="#808080", alpha=0.45,
                   edgecolors="none", zorder=2)

    # 2. Cluster points colored by their assigned track_id (consistent
    #    color across frames for the same physical object).
    cluster_xyz, cluster_cid = load_clusters_xyz(
        args.clusters_dir / f"clusters_{frame:06d}.csv")
    tracks_this_frame = tracks_idx.get(frame, [])

    # Group points by cluster_id, then assign each cluster to a track.
    cmap = plt.cm.tab20
    n_drawn_tracks = 0
    if cluster_xyz.size:
        for cid in np.unique(cluster_cid):
            if cid < 0:
                continue
            mask = cluster_cid == cid
            pts = cluster_xyz[mask]
            centroid = pts.mean(axis=0)[:2]
            tid = assign_cluster_to_track(centroid, tracks_this_frame,
                                          args.max_assign_dist)
            if tid < 0:
                # Cluster not associated to any published track —
                # render in muted yellow so DBSCAN-only clusters are
                # still visible without dominating the scene.
                color = "#666644"
                lw = 0.4
            else:
                color = cmap(tid % 20)
                lw = 0.8
                n_drawn_tracks += 1
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       s=4, color=color, alpha=0.9,
                       edgecolors="none", zorder=3)
            draw_aabb(ax, pts, color="#22ff44" if tid >= 0 else "#666644",
                      lw=lw)

    # 3. Range rings + ego marker.
    draw_range_rings(ax, view_range)
    draw_ego(ax)

    # 4. Frame-id badge.
    ax.text2D(0.02, 0.98,
              f"frame {frame}\n{len(tracks_this_frame)} tracks "
              f"({n_drawn_tracks} visible)",
              transform=ax.transAxes,
              color="white", fontsize=10, fontweight="bold",
              ha="left", va="top",
              bbox=dict(facecolor="black", edgecolor="white",
                        alpha=0.7, pad=4),
              zorder=99)


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--lidar-dir",     type=Path, required=True)
    p.add_argument("--clusters-dir",  type=Path, required=True)
    p.add_argument("--tracks-csv",    type=Path, required=True)
    p.add_argument("--frame-start",   type=int,  default=0)
    p.add_argument("--frame-end",     type=int,  default=2848)
    p.add_argument("--stride",        type=int,  default=5)
    p.add_argument("--fps",           type=int,  default=10)
    p.add_argument("--elev",          type=float, default=20.0)
    p.add_argument("--azim",          type=float, default=-85.0)
    p.add_argument("--view-range",    type=float, default=40.0,
                   help="x/y limits in meters (default 40)")
    p.add_argument("--max-assign-dist", type=float, default=5.0,
                   help="cluster-centroid → track-centroid match distance (m)")
    p.add_argument("--point-decim",   type=int, default=4,
                   help="raw LiDAR decimation factor (1=no decim, 4=quarter)")
    p.add_argument("--out-mp4",       type=Path, required=True)
    p.add_argument("--out-gif",       type=Path, default=None,
                   help="optional GIF output (large for 3D — skip unless needed)")
    args = p.parse_args()

    print(f"[anim-3d] indexing tracks ...")
    tracks_idx = load_tracks_indexed(args.tracks_csv)

    frames = list(range(args.frame_start,
                        min(args.frame_end, max(tracks_idx.keys()) if tracks_idx else args.frame_end) + 1,
                        args.stride))
    print(f"[anim-3d] rendering {len(frames)} frames "
          f"(stride={args.stride}, decim={args.point_decim}, "
          f"~{int(0.05 * 90000 / args.point_decim / 1000)} sec per frame) ...")

    fig = plt.figure(figsize=(12, 9), dpi=100, facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def update(i: int):
        if i % 25 == 0:
            print(f"  frame {i}/{len(frames)}  ({frames[i]})")
        draw_frame(ax, frames[i], args, tracks_idx,
                   args.view_range, args.point_decim)
        return ()

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=int(1000 / args.fps), blit=False)

    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)
    print(f"[anim-3d] writing MP4 → {args.out_mp4}")
    try:
        writer = FFMpegWriter(fps=args.fps, codec="libx264",
                              extra_args=["-pix_fmt", "yuv420p"])
        anim.save(args.out_mp4, writer=writer)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  ffmpeg writer failed ({e}); trying PillowWriter as MP4 fallback")
        anim.save(args.out_mp4.with_suffix(".gif"),
                  writer=PillowWriter(fps=args.fps))

    if args.out_gif:
        print(f"[anim-3d] writing GIF → {args.out_gif} (this is slow for 3D)")
        anim.save(args.out_gif, writer=PillowWriter(fps=args.fps))

    print(f"[anim-3d] DONE")


if __name__ == "__main__":
    main()
