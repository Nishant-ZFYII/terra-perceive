#!/usr/bin/env python3
"""animate_tracker_vs_dbscan.py — closing-hero animation for the M4 blog.

Three panels per frame:
    Left   — RGB camera image (RELLIS Pylon)
    Middle — full LiDAR top-down (gray) + DBSCAN clusters colored by
             cluster_id   →   colors FLICKER frame-to-frame
    Right  — full LiDAR top-down (gray) + SORT tracks colored by
             track_id     →   colors are STABLE across frames, with each
             track's ID printed beside its centroid.

The middle panel is the "before tracking" state: per-frame DBSCAN gives
each tree a fresh cluster_id every scan, so colors change every frame.
The right panel is the "after tracking" state: SORT binds a Kalman-
predicted identity across frames, so the same tree keeps the same color
and id throughout.

Together: the side-by-side IS the headline argument for why a tracker
exists on top of a clusterer.

Usage:
    python scripts/animate_tracker_vs_dbscan.py \\
        --lidar-dir       /media/.../m4_perframe/extracted_frames \\
        --camera-dir      /media/.../m4_perframe/extracted_frames_camera \\
        --clusters-dir    /media/.../m4_perframe/clusters_sweetspot \\
        --tracks-csv      results_m4/ablation_g/sort_on_rellis/tracks.csv \\
        --frame-start 0 --frame-end 2848 \\
        --fps 10 --stride 5 \\
        --out-mp4 results_m4/ablation_g/sort_vs_dbscan.mp4 \\
        --out-gif results_m4/ablation_g/sort_vs_dbscan.gif

Inputs:
    {clusters_dir}/clusters_NNNNNN.csv   x,y,z,cluster_id  (per-frame DBSCAN)
    {tracks_csv}                         frame_id,track_id,x,y,vx,vy,age,cov_trace
                                          (one CSV across all frames, from
                                           tracker_runner)
    {lidar_dir}/NNNNNN.bin               KITTI x,y,z,intensity per frame
    {camera_dir}/NNNNNN.jpg              RGB camera frame
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

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
    if not path.exists():
        return np.empty((0, 3), dtype=np.float32)
    raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return raw[:, :3]


def load_clusters(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.empty((0, 2)), np.empty(0, dtype=int)
    xy: List[Tuple[float, float]] = []
    cid: List[int] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            xy.append((float(row["x"]), float(row["y"])))
            cid.append(int(row["cluster_id"]))
    return np.array(xy), np.array(cid, dtype=int)


def load_tracks_indexed(path: Path) -> Dict[int, List[Tuple[int, float, float]]]:
    """frame_id -> [(track_id, x, y), ...]"""
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


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------
def render_camera(ax, camera_path: Path, fid: int) -> None:
    ax.clear()
    if camera_path.exists():
        ax.imshow(np.array(PILImage.open(camera_path)))
        ax.set_title(f"RGB camera   frame {fid}")
    else:
        ax.text(0.5, 0.5, "(no camera frame)", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(f"RGB camera   frame {fid}")
    ax.axis("off")


def render_lidar_with_layer(ax,
                            lidar_path: Path,
                            layer_xy: np.ndarray,
                            layer_id: np.ndarray,
                            bbox_xy: Tuple[float, float, float, float],
                            title: str,
                            label_each: bool = False) -> None:
    """Top-down LiDAR cloud (gray bg) + colored layer (clusters or tracks)."""
    ax.clear()
    ax.set_facecolor("#1a1a1a")

    full = load_lidar_bin(lidar_path)
    if full.size:
        ax.scatter(full[:, 0], full[:, 1],
                   s=1, c="#555555", alpha=0.5, zorder=1, edgecolors="none")

    cmap = plt.cm.tab20
    if layer_xy.size:
        # For DBSCAN: skip noise (id < 0). For tracker: ids are always ≥ 0.
        valid = layer_id >= 0
        unique_ids = sorted(set(layer_id[valid].tolist()))
        for k in unique_ids:
            m = layer_id == k
            ax.scatter(layer_xy[m, 0], layer_xy[m, 1],
                       s=18, color=cmap(int(k) % 20),
                       alpha=0.95, zorder=3, edgecolors="black",
                       linewidths=0.4)
            if label_each:
                # Label at the centroid of this id's points.
                cx = layer_xy[m, 0].mean()
                cy = layer_xy[m, 1].mean()
                ax.text(cx + 0.4, cy + 0.4, f"#{k}",
                        color="white", fontsize=7, zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor=cmap(int(k) % 20),
                                  edgecolor="none", alpha=0.85))
        # Show noise lightly (DBSCAN only).
        noise = layer_id < 0
        if noise.any():
            ax.scatter(layer_xy[noise, 0], layer_xy[noise, 1],
                       s=3, c="#aaaaaa", alpha=0.5, zorder=2, edgecolors="none")

    ax.set_xlim(bbox_xy[0], bbox_xy[1])
    ax.set_ylim(bbox_xy[2], bbox_xy[3])
    ax.set_aspect("equal")
    ax.set_title(title)
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


def compute_bbox_from_tracks(tracks_idx,
                             frame_ids,
                             margin_m: float = 5.0
                             ) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for fid in frame_ids:
        for _, x, y in tracks_idx.get(fid, []):
            xs.append(x); ys.append(y)
    if not xs:
        return (-50.0, 50.0, -50.0, 50.0)
    return (min(xs) - margin_m, max(xs) + margin_m,
            min(ys) - margin_m, max(ys) + margin_m)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lidar-dir",     type=Path, required=True)
    p.add_argument("--camera-dir",    type=Path, required=True)
    p.add_argument("--clusters-dir",  type=Path, required=True)
    p.add_argument("--tracks-csv",    type=Path, required=True)
    p.add_argument("--frame-start",   type=int,  required=True)
    p.add_argument("--frame-end",     type=int,  required=True)
    p.add_argument("--fps",           type=int,  default=10)
    p.add_argument("--stride",        type=int,  default=1)
    p.add_argument("--out-mp4",       type=Path, required=True)
    p.add_argument("--out-gif",       type=Path, default=None)
    args = p.parse_args()

    frames = list_available_frames(args.clusters_dir, args.frame_start, args.frame_end)
    if not frames:
        sys.exit("[sort_vs_dbscan] no cluster CSVs in given range")
    frames = frames[::args.stride]
    print(f"[sort_vs_dbscan] rendering {len(frames)} frames (stride={args.stride})")

    # Load tracks once, index by frame.
    print(f"[sort_vs_dbscan] loading tracks from {args.tracks_csv} ...")
    tracks_idx = load_tracks_indexed(args.tracks_csv)
    print(f"[sort_vs_dbscan]   {sum(len(v) for v in tracks_idx.values())} track rows "
          f"across {len(tracks_idx)} frames")

    bbox = compute_bbox_from_tracks(tracks_idx, frames)
    print(f"[sort_vs_dbscan] bbox: x=[{bbox[0]:.1f},{bbox[1]:.1f}] "
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
        render_camera(axes[0], args.camera_dir / f"{fid:06d}.jpg", fid)

        # Panel 2 — DBSCAN clusters (flickering colors per cluster_id)
        cluster_xy, cluster_id = load_clusters(
            args.clusters_dir / f"clusters_{fid:06d}.csv")
        render_lidar_with_layer(
            axes[1], args.lidar_dir / f"{fid:06d}.bin",
            cluster_xy, cluster_id, bbox,
            title=f"DBSCAN  per-frame  ({len(set(cluster_id[cluster_id>=0]))} clusters; "
                  f"colors flicker)",
            label_each=False,
        )

        # Panel 3 — SORT tracks (stable colors per track_id, with labels)
        tracks_this = tracks_idx.get(fid, [])
        if tracks_this:
            tids  = np.array([t[0] for t in tracks_this], dtype=int)
            tx_ty = np.array([(t[1], t[2]) for t in tracks_this])
        else:
            tids  = np.empty(0, dtype=int)
            tx_ty = np.empty((0, 2))
        render_lidar_with_layer(
            axes[2], args.lidar_dir / f"{fid:06d}.bin",
            tx_ty, tids, bbox,
            title=f"SORT  ({len(tids)} live tracks; stable IDs)",
            label_each=True,
        )

    fig.suptitle(
        "DBSCAN per-frame clusters (middle)  vs  SORT tracker stable IDs (right)",
        y=0.995,
    )

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000.0 / args.fps)

    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)
    print(f"[sort_vs_dbscan] writing {args.out_mp4}")
    anim.save(str(args.out_mp4), writer=FFMpegWriter(fps=args.fps, bitrate=3000))

    if args.out_gif is not None:
        print(f"[sort_vs_dbscan] writing {args.out_gif}")
        anim.save(str(args.out_gif), writer=PillowWriter(fps=args.fps))

    plt.close(fig)
    print(f"[sort_vs_dbscan] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
