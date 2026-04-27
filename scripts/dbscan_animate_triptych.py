#!/usr/bin/env python3
"""dbscan_animate_triptych.py — 3-panel perception-pipeline animation.

Per frame, three side-by-side panels:
    Left    — RGB camera image (the actual driving scene)
    Middle  — full LiDAR cloud, top-down: ground in brown, obstacles in white
    Right   — DBSCAN clusters on obstacle-only points, colored per cluster

This is the closing-hero figure for the M4 blog post: a hiring manager
glancing at one second of GIF understands the full pipeline.

Inputs (per frame fid, all on the external drive):
    {extracted_frames}/<fid:06d>.bin          LiDAR cloud (KITTI: x,y,z,intensity)
    {camera_dir}/<fid:06d>.jpg                RGB image, time-synced to fid
    {obstacles_dir}/obstacles_<fid:06d>.csv   x,y,z (obstacle subset)
    {clusters_dir}/clusters_<fid:06d>.csv     x,y,z,cluster_id (DBSCAN out)

Usage:
    python scripts/dbscan_animate_triptych.py \\
        --lidar-dir       /media/.../m4_perframe/extracted_frames \\
        --camera-dir      /media/.../m4_perframe/extracted_frames_camera \\
        --obstacles-dir   /media/.../m4_perframe/obstacles \\
        --clusters-dir    /media/.../m4_perframe/clusters_sweetspot \\
        --frame-start 0 --frame-end 2848 \\
        --eps 0.5 --min-points 10 \\
        --fps 10 \\
        --stride 5 \\
        --out-mp4 results_m4/ablation_g/triptych.mp4 \\
        --out-gif results_m4/ablation_g/triptych.gif
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_lidar_bin(path: Path) -> np.ndarray:
    """Returns Nx3 float32 (x, y, z), drops the intensity column."""
    raw = np.fromfile(path, dtype=np.float32)
    arr = raw.reshape(-1, 4)
    return arr[:, :3]


def load_obstacle_set(path: Path) -> Set[Tuple[float, float, float]]:
    """Return set of (x, y, z) tuples for the obstacle subset.
    Used to mask 'ground' in the middle panel by SET DIFFERENCE on the full cloud.
    """
    out: Set[Tuple[float, float, float]] = set()
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            out.add((float(row["x"]), float(row["y"]), float(row["z"])))
    return out


def load_obstacle_xyz(path: Path) -> np.ndarray:
    """Return obstacle points as Nx3 array (without the set wrapping above)."""
    pts: List[Tuple[float, float, float]] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            pts.append((float(row["x"]), float(row["y"]), float(row["z"])))
    return np.array(pts) if pts else np.empty((0, 3))


def load_clusters(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    xy: List[Tuple[float, float]] = []
    cid: List[int] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            xy.append((float(row["x"]), float(row["y"])))
            cid.append(int(row["cluster_id"]))
    return (np.array(xy) if xy else np.empty((0, 2)),
            np.array(cid, dtype=int))


def load_camera(path: Path) -> np.ndarray:
    """Returns HxWx3 uint8."""
    return np.array(PILImage.open(path))


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------
def render_panel_camera(ax, camera_path: Path, fid: int) -> None:
    ax.clear()
    if camera_path.exists():
        img = load_camera(camera_path)
        ax.imshow(img)
        ax.set_title(f"RGB camera  frame {fid}")
    else:
        ax.text(0.5, 0.5, "(no camera frame)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"RGB camera  frame {fid}")
    ax.axis("off")


def render_panel_lidar(ax,
                       lidar_path: Path,
                       cluster_xy: np.ndarray,
                       cluster_id: np.ndarray,
                       bbox_xy: Tuple[float, float, float, float]) -> None:
    """Top-down full LiDAR cloud (gray) + DBSCAN clusters overlaid (colored).

    Showing both layers in one panel makes the algorithm's effect obvious:
    "this dense gray cloud is the input; these colored points are what
    clustering picked out." No fragile float-key set-difference required.
    """
    ax.clear()
    ax.set_facecolor("#1a1a1a")

    # Background: full LiDAR cloud as small dim gray dots.
    if lidar_path.exists():
        full = load_lidar_bin(lidar_path)
        if full.size:
            ax.scatter(full[:, 0], full[:, 1],
                       s=1, c="#555555", alpha=0.5, zorder=1, edgecolors="none")

    # Foreground: DBSCAN clusters in distinct colors (skip noise=-1).
    cmap = plt.cm.tab20
    if cluster_xy.size:
        for k in sorted(set(cluster_id[cluster_id >= 0].tolist())):
            m = cluster_id == k
            ax.scatter(cluster_xy[m, 0], cluster_xy[m, 1],
                       s=6, color=cmap(int(k) % 20),
                       alpha=0.9, zorder=2, edgecolors="none")
        # Noise (gt_track_id < 0) shown in slightly brighter gray on top of bg.
        noise = cluster_id < 0
        if noise.any():
            ax.scatter(cluster_xy[noise, 0], cluster_xy[noise, 1],
                       s=3, c="#aaaaaa", alpha=0.6, zorder=1.5, edgecolors="none")

    ax.set_xlim(bbox_xy[0], bbox_xy[1])
    ax.set_ylim(bbox_xy[2], bbox_xy[3])
    ax.set_aspect("equal")
    ax.set_title("LiDAR (gray) + DBSCAN clusters (colored)")
    ax.set_xlabel("x  [m]"); ax.set_ylabel("y  [m]")
    ax.grid(True, alpha=0.15, color="white")


def render_panel_clusters(ax,
                          xy: np.ndarray,
                          cid: np.ndarray,
                          bbox_xy: Tuple[float, float, float, float],
                          eps: float,
                          mp: int) -> None:
    ax.clear()
    ax.set_facecolor("#1a1a1a")
    cmap = plt.cm.tab20

    if xy.size:
        noise = cid < 0
        if noise.any():
            ax.scatter(xy[noise, 0], xy[noise, 1],
                       s=2, c="#666666", alpha=0.5, zorder=1, edgecolors="none")
        cluster_ids = sorted(set(cid[~noise].tolist()))
        for k in cluster_ids:
            m = cid == k
            ax.scatter(xy[m, 0], xy[m, 1],
                       s=4, color=cmap(int(k) % 20),
                       alpha=0.9, zorder=2, edgecolors="none")
        K = len(cluster_ids)
        N_noise = int(noise.sum())
    else:
        K = 0; N_noise = 0

    ax.set_xlim(bbox_xy[0], bbox_xy[1])
    ax.set_ylim(bbox_xy[2], bbox_xy[3])
    ax.set_aspect("equal")
    ax.set_title(f"DBSCAN  eps={eps}m, mp={mp}   {K} clusters, {N_noise} noise")
    ax.set_xlabel("x  [m]"); ax.set_ylabel("y  [m]")
    ax.grid(True, alpha=0.15, color="white")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def list_available_frames(clusters_dir: Path,
                          frame_start: int,
                          frame_end: int) -> List[int]:
    out = []
    for fid in range(frame_start, frame_end + 1):
        if (clusters_dir / f"clusters_{fid:06d}.csv").exists():
            out.append(fid)
    return out


def compute_global_bbox_from_clusters(clusters_dir: Path,
                                      frame_ids: List[int],
                                      sample_every: int = 20,
                                      margin_m: float = 5.0
                                      ) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for fid in frame_ids[::sample_every]:
        xy, _ = load_clusters(clusters_dir / f"clusters_{fid:06d}.csv")
        if xy.size:
            xs.append(xy[:, 0]); ys.append(xy[:, 1])
    if not xs:
        return (-50.0, 50.0, -50.0, 50.0)
    all_x = np.concatenate(xs); all_y = np.concatenate(ys)
    return (float(all_x.min()) - margin_m, float(all_x.max()) + margin_m,
            float(all_y.min()) - margin_m, float(all_y.max()) + margin_m)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lidar-dir",     type=Path, required=True)
    p.add_argument("--camera-dir",    type=Path, required=True)
    p.add_argument("--obstacles-dir", type=Path, required=True)
    p.add_argument("--clusters-dir",  type=Path, required=True)
    p.add_argument("--frame-start",   type=int,  required=True)
    p.add_argument("--frame-end",     type=int,  required=True)
    p.add_argument("--eps",           type=float, required=True)
    p.add_argument("--min-points",    type=int,   required=True)
    p.add_argument("--fps",           type=int,   default=10)
    p.add_argument("--stride",        type=int,   default=1,
                   help="render every Nth frame (keeps wall-clock manageable)")
    p.add_argument("--out-mp4",       type=Path,  required=True)
    p.add_argument("--out-gif",       type=Path,  default=None)
    args = p.parse_args()

    frames = list_available_frames(args.clusters_dir, args.frame_start, args.frame_end)
    if not frames:
        sys.exit("[triptych] no cluster CSVs in given range")
    frames = frames[::args.stride]
    print(f"[triptych] rendering {len(frames)} frames "
          f"(stride={args.stride}, full range had {args.frame_end - args.frame_start + 1})")

    bbox = compute_global_bbox_from_clusters(args.clusters_dir, frames)
    print(f"[triptych] global bbox: x=[{bbox[0]:.1f},{bbox[1]:.1f}] "
          f"y=[{bbox[2]:.1f},{bbox[3]:.1f}]")

    plt.rcParams.update({
        "font.size":      11,
        "axes.titlesize": 12,
        "figure.titlesize": 14,
    })
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             gridspec_kw={"width_ratios": [1.3, 1.0, 1.0]})

    def draw(fid: int) -> None:
        # Panel 1 — camera
        render_panel_camera(axes[0], args.camera_dir / f"{fid:06d}.jpg", fid)

        # Panel 2 — full LiDAR cloud + DBSCAN clusters overlaid (colored)
        xy, cid = load_clusters(args.clusters_dir / f"clusters_{fid:06d}.csv")
        render_panel_lidar(axes[1], args.lidar_dir / f"{fid:06d}.bin", xy, cid, bbox)

        # Panel 3 — DBSCAN clusters only (zoomed view for detail)
        render_panel_clusters(axes[2], xy, cid, bbox, args.eps, args.min_points)

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000.0 / args.fps)

    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)
    print(f"[triptych] writing {args.out_mp4}")
    anim.save(str(args.out_mp4), writer=FFMpegWriter(fps=args.fps, bitrate=3000))

    if args.out_gif is not None:
        print(f"[triptych] writing {args.out_gif}")
        anim.save(str(args.out_gif), writer=PillowWriter(fps=args.fps))

    plt.close(fig)
    print(f"[triptych] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
