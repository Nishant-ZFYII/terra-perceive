#!/usr/bin/env python3
"""P2-M6 m12 hero animation: side-by-side BEV confidence, every frame.

Reads per-frame snapshot CSVs from heuristic and probabilistic
traversability_runner runs (with --snapshot-every 1) and renders a side-by-side
animated BEV. Each cell is a colored marker, viridis-colormapped on confidence.
The visual story is the heuristic panel cliffing past r=30 m vs the
probabilistic panel showing smooth confidence past 30 m.

Usage:
    python scripts/m6/animate_confidence_bev.py \\
        --heuristic-root     /media/.../trav_heuristic_perframe \\
        --probabilistic-root /media/.../trav_probabilistic_perframe \\
        --out                /media/.../m6_animations/confidence_bev.mp4 \\
        --fps 30 --stride 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter


def list_snapshots(root: Path) -> list[Path]:
    snap = root / "snapshots"
    if not snap.exists():
        return []
    return sorted(snap.glob("frame_*.csv"))


def load_frame(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return np.genfromtxt(path, delimiter=",", names=True, dtype=None,
                             encoding="utf-8")
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heuristic-root", required=True)
    p.add_argument("--probabilistic-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--stride", type=int, default=1,
                   help="render every Nth frame; default 1 = every frame")
    p.add_argument("--max-frames", type=int, default=0,
                   help="cap on rendered frames (0 = all)")
    p.add_argument("--xlim", nargs=2, type=float, default=[-5.0, 30.0])
    p.add_argument("--ylim", nargs=2, type=float, default=[-15.0, 15.0])
    p.add_argument("--marker-size", type=float, default=32.0)
    p.add_argument("--figsize", nargs=2, type=float, default=[17.0, 9.0])
    args = p.parse_args()

    h_paths = list_snapshots(Path(args.heuristic_root))
    p_paths = list_snapshots(Path(args.probabilistic_root))
    print(f"[info] heuristic snapshots:    {len(h_paths)}")
    print(f"[info] probabilistic snapshots: {len(p_paths)}")
    if len(h_paths) == 0 or len(p_paths) == 0:
        raise SystemExit("[error] missing snapshots")

    n = min(len(h_paths), len(p_paths))
    indices = list(range(0, n, args.stride))
    if args.max_frames > 0:
        indices = indices[: args.max_frames]

    fig, axes = plt.subplots(1, 2, figsize=tuple(args.figsize))
    titles = ["heuristic", "probabilistic"]
    scats = []
    for ax, title in zip(axes, titles):
        ax.set_xlim(*args.xlim)
        ax.set_ylim(*args.ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        sc = ax.scatter([], [], c=[], cmap="viridis", vmin=0.0, vmax=1.0,
                        s=args.marker_size, edgecolors="none")
        scats.append(sc)
        plt.colorbar(sc, ax=ax, label="confidence", shrink=0.85)

    suptitle = fig.suptitle("", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, bitrate=4800)
    with writer.saving(fig, str(out), dpi=120):
        for k, fi in enumerate(indices):
            h_arr = load_frame(h_paths[fi])
            p_arr = load_frame(p_paths[fi])
            if h_arr is None or p_arr is None:
                continue
            scats[0].set_offsets(np.column_stack([h_arr["x_center"], h_arr["y_center"]]))
            scats[0].set_array(h_arr["confidence"])
            scats[1].set_offsets(np.column_stack([p_arr["x_center"], p_arr["y_center"]]))
            scats[1].set_array(p_arr["confidence"])
            suptitle.set_text(f"BEV cell confidence — frame {fi:05d} / {n - 1:05d}")
            writer.grab_frame()
            if k % 200 == 0:
                print(f"[render] {k + 1}/{len(indices)} frames")
    plt.close(fig)
    print(f"[wrote] {out}  ({len(indices)} frames @ {args.fps} fps "
          f"= {len(indices)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
