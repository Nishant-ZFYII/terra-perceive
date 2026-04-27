#!/usr/bin/env python3
"""animate_pair_context.py — per-pair temporal-context GIF for the M13
labeling UI.

Given a labeling pair (frame_a, cluster_a, frame_b, cluster_b), render
a short GIF showing 2 seconds before + 1 second after the focal frame
(at 10 Hz that's 20 + 10 = 30 frames). Side-by-side panels:

    LEFT  — DBSCAN per-frame view: every cluster's points colored by
            cluster_id (tab20 colormap), with cluster A highlighted in
            red on the frame where it lives, and cluster B highlighted
            in magenta on its frame. Cluster_id labels (#cid) printed
            next to each cluster. Frame-id badge top-left.

    RIGHT — SORT-track view: each track_id is drawn as a colored marker
            with track_id label. Stable IDs across frames show the
            tracker's persistence. Same frame-id badge.

Both panels share the M4 closing-hero animation styling: dark-gray
background, ±40m view, faint gray full-LiDAR scene context, M4
colormap for clusters/tracks.

Inputs (paths default to project layout):
    --lidar-dir       data/extracted_frames_full     (KITTI .bin per frame)
    --clusters-dir    /media/.../clusters_sweetspot   (clusters_NNNNNN.csv)
    --tracks-csv      results_m4/ablation_g/sort_on_rellis/tracks.csv

Output: a GIF at --out (default /tmp/pair_<id>_context.gif).

Typical wall-clock: ~6-10 seconds per render.

Usage:
    python scripts/animate_pair_context.py \\
        --frame-a 1078 --cluster-a 9 \\
        --frame-b 1079 --cluster-b 12 \\
        --clusters-dir /media/nishant/SeeGayt2/terra_perceive/m4_perframe/clusters_sweetspot \\
        --out /tmp/pair_42_context.gif
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless render — no display required
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


# -----------------------------------------------------------------------------
# Loaders — same shape as scripts/animate_tracker_vs_dbscan.py so the visual
# language matches.
# -----------------------------------------------------------------------------

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


def load_tracks(path: Path) -> Dict[int, List[Tuple[int, float, float]]]:
    """frame_id → list of (track_id, x, y). Returns empty dict if path
    missing — the GIF still renders, just without the SORT panel populated.
    """
    out: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    if not path.exists():
        return out
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            out[int(row["frame_id"])].append((
                int(row["track_id"]),
                float(row["x"]),
                float(row["y"]),
            ))
    return out


# -----------------------------------------------------------------------------
# Per-frame draw — called by FuncAnimation for each frame in the window.
# -----------------------------------------------------------------------------

def setup_axes(ax, title: str) -> None:
    ax.set_facecolor("#1a1a1a")
    ax.set_aspect("equal")
    ax.set_xlim(-40, 40); ax.set_ylim(-40, 40)
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlabel("x (m)", color="white", fontsize=8)
    ax.set_ylabel("y (m)", color="white", fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    ax.set_title(title, color="white", fontsize=10)


def frame_id_badge(ax, frame: int, extra: str = "") -> None:
    label = f"frame {frame}"
    if extra:
        label = f"{label}  ({extra})"
    ax.text(0.015, 0.985, label, transform=ax.transAxes,
            color="white", fontsize=10, fontweight="bold",
            ha="left", va="top",
            bbox=dict(facecolor="black", edgecolor="white",
                      alpha=0.7, pad=3),
            zorder=15)


def draw_frame(
    ax_db, ax_sort, frame: int,
    args, tracks_idx: Dict[int, List[Tuple[int, float, float]]],
) -> None:
    ax_db.clear(); ax_sort.clear()
    setup_axes(ax_db,   "DBSCAN per-frame (cluster_id)")
    setup_axes(ax_sort, "SORT tracks (stable IDs)")

    # Faint gray full LiDAR scan as scene context — shared by both panels.
    full = load_lidar_bin(args.lidar_dir / f"{frame:06d}.bin")
    if full.size:
        for ax in (ax_db, ax_sort):
            ax.scatter(full[:, 0], full[:, 1], s=1, c="#555555",
                       alpha=0.5, zorder=1, edgecolors="none")

    cmap = plt.cm.tab20

    # --- LEFT panel: DBSCAN ----------------------------------------------
    cluster_csv = args.clusters_dir / f"clusters_{frame:06d}.csv"
    xy, cids = load_clusters(cluster_csv)
    extra = ""
    if xy.size:
        for cid in np.unique(cids):
            if cid < 0:           # skip noise
                continue
            mask = cids == cid
            # Highlight A on its frame, B on its frame, in our bright pair colors.
            if frame == args.frame_a and cid == args.cluster_a:
                color = "red"
                ax_db.scatter(xy[mask, 0], xy[mask, 1], s=28, c=color,
                              alpha=0.95, zorder=5,
                              edgecolors="white", linewidths=0.6)
                cx, cy = xy[mask, 0].mean(), xy[mask, 1].mean()
                ax_db.text(cx + 1.2, cy + 1.2, f"A: #{cid}",
                           color="white", fontsize=9, fontweight="bold",
                           bbox=dict(facecolor=color, edgecolor="none",
                                     alpha=0.9, pad=2.5),
                           zorder=11)
                extra = "A frame"
            elif frame == args.frame_b and cid == args.cluster_b:
                color = "magenta"
                ax_db.scatter(xy[mask, 0], xy[mask, 1], s=28, c=color,
                              alpha=0.95, zorder=5,
                              edgecolors="white", linewidths=0.6)
                cx, cy = xy[mask, 0].mean(), xy[mask, 1].mean()
                ax_db.text(cx + 1.2, cy + 1.2, f"B: #{cid}",
                           color="white", fontsize=9, fontweight="bold",
                           bbox=dict(facecolor=color, edgecolor="none",
                                     alpha=0.9, pad=2.5),
                           zorder=11)
                extra = "B frame" if not extra else "A+B frame"
            else:
                color = cmap(cid % 20)
                ax_db.scatter(xy[mask, 0], xy[mask, 1], s=14, color=color,
                              alpha=0.85, zorder=3,
                              edgecolors="black", linewidths=0.3)
    frame_id_badge(ax_db, frame, extra)

    # --- RIGHT panel: SORT ------------------------------------------------
    for track_id, x, y in tracks_idx.get(frame, []):
        color = cmap(track_id % 20)
        ax_sort.scatter([x], [y], s=80, color=color,
                        edgecolors="black", linewidths=0.5, zorder=4)
        ax_sort.text(x + 1, y + 1, f"#{track_id}",
                     color="white", fontsize=8, fontweight="bold",
                     bbox=dict(facecolor=color, edgecolor="none",
                               alpha=0.85, pad=2),
                     zorder=5)
    frame_id_badge(ax_sort, frame, extra)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--frame-a",    type=int, required=True)
    ap.add_argument("--cluster-a",  type=int, required=True)
    ap.add_argument("--frame-b",    type=int, required=True)
    ap.add_argument("--cluster-b",  type=int, required=True)
    ap.add_argument("--lidar-dir",     type=Path,
                    default=Path("data/extracted_frames_full"))
    ap.add_argument("--clusters-dir",  type=Path, required=True)
    ap.add_argument("--tracks-csv",    type=Path,
                    default=Path("results_m4/ablation_g/sort_on_rellis/tracks.csv"))
    ap.add_argument("--frames-before", type=int, default=20,
                    help="frames to render BEFORE the focal frame (default 20 = 2.0 s @ 10Hz)")
    ap.add_argument("--frames-after",  type=int, default=10,
                    help="frames to render AFTER the focal frame (default 10 = 1.0 s @ 10Hz)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    focal_min = min(args.frame_a, args.frame_b)
    focal_max = max(args.frame_a, args.frame_b)
    f_start = max(0, focal_min - args.frames_before)
    f_end   = focal_max + args.frames_after
    frames = list(range(f_start, f_end + 1))
    print(f"[anim-pair] rendering frames [{f_start}..{f_end}]  "
          f"({len(frames)} frames @ {args.fps} fps)")

    tracks_idx = load_tracks(args.tracks_csv)
    if not tracks_idx:
        print(f"[anim-pair] WARNING: tracks.csv empty/missing at "
              f"{args.tracks_csv} — SORT panel will be empty",
              file=sys.stderr)

    fig, (ax_db, ax_sort) = plt.subplots(1, 2, figsize=(13, 6.5),
                                         facecolor="#1a1a1a")
    fig.suptitle(
        f"context: pair  (A frame {args.frame_a} cid {args.cluster_a}  ↔  "
        f"B frame {args.frame_b} cid {args.cluster_b})",
        color="white", fontsize=12,
    )

    def update(i: int):
        draw_frame(ax_db, ax_sort, frames[i], args, tracks_idx)
        return ()

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=int(1000 / args.fps), blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    print(f"[anim-pair] wrote {args.out}")


if __name__ == "__main__":
    main()
