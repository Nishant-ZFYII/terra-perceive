#!/usr/bin/env python3
"""P2-M6: full RELLIS-3D sequence 00 LiDAR visualization, every frame.

Renders the raw point cloud at every frame as a top-down scatter, colored by
height. The output covers all 2849 frames so the reader can see what the
M6 traversability + safety stack is actually consuming. No RANSAC, no
DBSCAN, no derived layer — just the raw LiDAR.

Usage:
    python scripts/m6/animate_rellis_lidar.py \\
        --lidar-dir /media/nishant/SeeGayt2/terra_perceive/m4_perframe/extracted_frames \\
        --out       /media/nishant/SeeGayt2/terra_perceive/m6_animations/rellis_lidar_bev.mp4 \\
        --fps 30

By default the BEV view covers (-30, +50) m in x and (-30, +30) m in y;
points outside are clipped. Use --xlim/--ylim to override.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter


def load_bin(path: Path) -> np.ndarray:
    """KITTI-style float32 [x, y, z, intensity] per point."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lidar-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--xlim", nargs=2, type=float, default=[-30.0, 50.0])
    p.add_argument("--ylim", nargs=2, type=float, default=[-30.0, 30.0])
    p.add_argument("--zmin", type=float, default=-2.0)
    p.add_argument("--zmax", type=float, default=3.0)
    p.add_argument("--marker-size", type=float, default=0.4)
    p.add_argument("--max-frames", type=int, default=0,
                   help="cap on rendered frames (0 = all)")
    p.add_argument("--stride", type=int, default=1,
                   help="render every Nth frame; default 1 = every frame")
    p.add_argument("--downsample", type=int, default=1,
                   help="keep every Nth point per frame for speed; default 1 keeps all")
    args = p.parse_args()

    lidar_dir = Path(args.lidar_dir)
    paths = sorted(lidar_dir.glob("*.bin"))
    print(f"[info] {len(paths)} frames found under {lidar_dir}")
    if not paths:
        raise SystemExit("[error] no .bin files")

    indices = list(range(0, len(paths), args.stride))
    if args.max_frames > 0:
        indices = indices[: args.max_frames]
    print(f"[info] rendering {len(indices)} frames at {args.fps} fps")

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_xlim(*args.xlim)
    ax.set_ylim(*args.ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x — forward (m)")
    ax.set_ylabel("y — left (m)")
    ax.grid(True, alpha=0.25)

    # Reference range rings.
    for r in (5, 10, 20, 30, 40):
        circle = plt.Circle((0, 0), r, fill=False, color="#888",
                             linewidth=0.5, alpha=0.4)
        ax.add_patch(circle)
        ax.text(r, 0.4, f"{r}m", fontsize=7, color="#666", alpha=0.6)
    # Ego marker.
    ax.plot(0, 0, "s", color="black", markersize=8, zorder=10, label="ego")

    # Empty scatter to update each frame.
    sc = ax.scatter([], [], c=[], cmap="viridis", vmin=args.zmin, vmax=args.zmax,
                    s=args.marker_size, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax, label="height z (m)", shrink=0.85)
    title = ax.set_title("RELLIS-3D sequence 00 — frame 00000")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, bitrate=4800)
    with writer.saving(fig, str(out), dpi=110):
        for k, fi in enumerate(indices):
            cloud = load_bin(paths[fi])
            if args.downsample > 1:
                cloud = cloud[::args.downsample]
            x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
            # Clip to viewport to keep matplotlib responsive.
            mask = ((x >= args.xlim[0]) & (x <= args.xlim[1])
                    & (y >= args.ylim[0]) & (y <= args.ylim[1]))
            sc.set_offsets(np.column_stack([x[mask], y[mask]]))
            sc.set_array(z[mask])
            title.set_text(f"RELLIS-3D sequence 00 — frame {fi:05d} / {len(paths) - 1:05d}   "
                           f"({mask.sum():,} pts)")
            writer.grab_frame()
            if k % 200 == 0:
                print(f"[render] {k + 1}/{len(indices)}")
    plt.close(fig)
    print(f"[wrote] {out}  ({len(indices)} frames @ {args.fps} fps "
          f"= {len(indices)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
