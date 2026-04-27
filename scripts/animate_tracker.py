#!/usr/bin/env python3
"""Render an MP4/GIF of a tracker run, or a side-by-side of two runs.

Reads the per-frame `tracks.csv` produced by `tracker_runner` and animates
each track as a labeled marker that moves over time. Velocity is shown as
a short arrow. Track IDs persist across frames.

Two modes:
    single   — one run, one panel.
    compare  — two runs (e.g. greedy vs munkres) rendered side-by-side
               with synchronized frame index. Headline asset for Ablation A.

The viewport is auto-fit to the full trajectory of the inputs (M3 lesson:
fixed defaults silently truncate). Override with --viewport fixed --xmin
--xmax --ymin --ymax if you want to lock it for comparison shots.

Usage:
    # Single
    python scripts/animate_tracker.py single \\
        --tracks results_m4/ablation_a/munkres/tracks.csv \\
        --detections tests/data/crossing.csv \\
        --out results_m4/ablation_a/munkres.mp4 \\
        --fps 10

    # Side-by-side
    python scripts/animate_tracker.py compare \\
        --tracks-left  results_m4/ablation_a/greedy/tracks.csv \\
        --tracks-right results_m4/ablation_a/munkres/tracks.csv \\
        --label-left   "Greedy" \\
        --label-right  "Munkres" \\
        --detections   tests/data/crossing.csv \\
        --out          results_m4/ablation_a/greedy_vs_munkres.mp4 \\
        --fps          10

CSV schemas:
    tracks.csv:      frame_id,track_id,x,y,vx,vy,age,cov_trace
    detections.csv:  frame_id,det_id,x,y,class_id,gt_track_id
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # headless rendering for HPC / CI

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass
class TrackRow:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float


@dataclass
class DetRow:
    det_id: int
    x: float
    y: float
    gt_track_id: int   # -1 if unknown


def load_tracks_csv(path: Path) -> Dict[int, List[TrackRow]]:
    """frame_id -> [TrackRow, ...]"""
    out: Dict[int, List[TrackRow]] = defaultdict(list)
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            out[int(row["frame_id"])].append(TrackRow(
                track_id=int(row["track_id"]),
                x=float(row["x"]), y=float(row["y"]),
                vx=float(row["vx"]), vy=float(row["vy"]),
            ))
    return out


def load_detections_csv(path: Path) -> Dict[int, List[DetRow]]:
    """frame_id -> [DetRow, ...]. Returns {} if file missing."""
    if not path.exists():
        return {}
    out: Dict[int, List[DetRow]] = defaultdict(list)
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            out[int(row["frame_id"])].append(DetRow(
                det_id=int(row.get("det_id", -1)),
                x=float(row["x"]), y=float(row["y"]),
                gt_track_id=int(row.get("gt_track_id", -1)),
            ))
    return out


# -----------------------------------------------------------------------------
# Viewport helper — rule #8 (auto-bbox from full trajectory).
# -----------------------------------------------------------------------------
def compute_viewport(
    *frame_dicts: Dict[int, List],
    margin_m: float = 10.0,
) -> Tuple[float, float, float, float]:
    """Compute (xmin, xmax, ymin, ymax) over every (x, y) in every input."""
    xs: List[float] = []
    ys: List[float] = []
    for fd in frame_dicts:
        for rows in fd.values():
            for r in rows:
                xs.append(r.x)
                ys.append(r.y)
    if not xs:
        return (-10.0, 10.0, -10.0, 10.0)
    return (min(xs) - margin_m, max(xs) + margin_m,
            min(ys) - margin_m, max(ys) + margin_m)


# -----------------------------------------------------------------------------
# Color helper — stable per track_id (so colors don't flip frame-to-frame).
# -----------------------------------------------------------------------------
def color_for(track_id: int) -> str:
    cmap = plt.cm.tab20
    return cmap(track_id % 20)


# -----------------------------------------------------------------------------
# Single-panel renderer
# -----------------------------------------------------------------------------
def render_single(
    tracks_path: Path,
    detections_path: Optional[Path],
    out_path: Path,
    fps: int,
    title: str,
    viewport: Optional[Tuple[float, float, float, float]],
) -> None:
    tracks_by_frame = load_tracks_csv(tracks_path)
    dets_by_frame = (load_detections_csv(detections_path)
                     if detections_path else {})
    frames = sorted(set(tracks_by_frame) | set(dets_by_frame))
    if not frames:
        sys.exit("no frames found in inputs")

    if viewport is None:
        xmin, xmax, ymin, ymax = compute_viewport(tracks_by_frame, dets_by_frame)
    else:
        xmin, xmax, ymin, ymax = viewport

    fig, ax = plt.subplots(figsize=(8, 8))

    def draw(frame_id: int) -> None:
        ax.clear()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title(f"{title}  frame {frame_id}")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.2)

        # Detections (light gray dots)
        for d in dets_by_frame.get(frame_id, []):
            ax.plot(d.x, d.y, "o", color="gray", alpha=0.6, markersize=6)

        # Tracks (colored squares + velocity arrow + id label)
        for t in tracks_by_frame.get(frame_id, []):
            c = color_for(t.track_id)
            ax.plot(t.x, t.y, "s", color=c, markersize=10,
                    markeredgecolor="black")
            ax.annotate(str(t.track_id), (t.x, t.y),
                        xytext=(6, 6), textcoords="offset points",
                        color=c, fontsize=10, fontweight="bold")
            ax.arrow(t.x, t.y, t.vx, t.vy,
                     head_width=0.2, length_includes_head=True,
                     color=c, alpha=0.8)

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    write_animation(anim, out_path, fps)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Side-by-side renderer (compare two runs on synchronized frame index)
# -----------------------------------------------------------------------------
def render_compare(
    tracks_left: Path, tracks_right: Path,
    label_left: str, label_right: str,
    detections_path: Optional[Path],
    out_path: Path,
    fps: int,
    viewport: Optional[Tuple[float, float, float, float]],
) -> None:
    left = load_tracks_csv(tracks_left)
    right = load_tracks_csv(tracks_right)
    dets = load_detections_csv(detections_path) if detections_path else {}
    frames = sorted(set(left) | set(right) | set(dets))
    if not frames:
        sys.exit("no frames found in inputs")

    if viewport is None:
        xmin, xmax, ymin, ymax = compute_viewport(left, right, dets)
    else:
        xmin, xmax, ymin, ymax = viewport

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 7))

    def draw_panel(ax, frame_id: int, frame_tracks, label: str) -> None:
        ax.clear()
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
        ax.set_title(f"{label}  frame {frame_id}")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.2)
        for d in dets.get(frame_id, []):
            ax.plot(d.x, d.y, "o", color="gray", alpha=0.6, markersize=6)
        for t in frame_tracks:
            c = color_for(t.track_id)
            ax.plot(t.x, t.y, "s", color=c, markersize=10,
                    markeredgecolor="black")
            ax.annotate(str(t.track_id), (t.x, t.y),
                        xytext=(6, 6), textcoords="offset points",
                        color=c, fontsize=10, fontweight="bold")
            ax.arrow(t.x, t.y, t.vx, t.vy,
                     head_width=0.2, length_includes_head=True,
                     color=c, alpha=0.8)

    def draw(frame_id: int) -> None:
        draw_panel(ax_l, frame_id, left.get(frame_id, []),  label_left)
        draw_panel(ax_r, frame_id, right.get(frame_id, []), label_right)

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    write_animation(anim, out_path, fps)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Animation writer — picks ffmpeg for .mp4, Pillow for .gif.
# -----------------------------------------------------------------------------
def write_animation(anim, out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        anim.save(str(out_path), writer=PillowWriter(fps=fps))
    elif suffix in (".mp4", ".mov"):
        anim.save(str(out_path), writer=FFMpegWriter(fps=fps, bitrate=2000))
    else:
        sys.exit(f"unsupported output suffix: {suffix} (use .mp4 or .gif)")
    print(f"[animate_tracker] wrote {out_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_viewport(ns: argparse.Namespace) -> Optional[Tuple[float, float, float, float]]:
    if ns.viewport == "auto":
        return None
    if any(v is None for v in (ns.xmin, ns.xmax, ns.ymin, ns.ymax)):
        sys.exit("--viewport fixed requires --xmin --xmax --ymin --ymax")
    return (ns.xmin, ns.xmax, ns.ymin, ns.ymax)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    # single
    s = sub.add_parser("single", help="one panel, one run")
    s.add_argument("--tracks",      type=Path, required=True)
    s.add_argument("--detections",  type=Path, default=None)
    s.add_argument("--out",         type=Path, required=True)
    s.add_argument("--fps",         type=int,  default=10)
    s.add_argument("--title",       type=str,  default="")
    s.add_argument("--viewport",    choices=["auto", "fixed"], default="auto")
    s.add_argument("--xmin", type=float); s.add_argument("--xmax", type=float)
    s.add_argument("--ymin", type=float); s.add_argument("--ymax", type=float)

    # compare
    c = sub.add_parser("compare", help="two panels (e.g. greedy vs munkres)")
    c.add_argument("--tracks-left",  type=Path, required=True)
    c.add_argument("--tracks-right", type=Path, required=True)
    c.add_argument("--label-left",   type=str,  default="left")
    c.add_argument("--label-right",  type=str,  default="right")
    c.add_argument("--detections",   type=Path, default=None)
    c.add_argument("--out",          type=Path, required=True)
    c.add_argument("--fps",          type=int,  default=10)
    c.add_argument("--viewport",     choices=["auto", "fixed"], default="auto")
    c.add_argument("--xmin", type=float); c.add_argument("--xmax", type=float)
    c.add_argument("--ymin", type=float); c.add_argument("--ymax", type=float)

    args = p.parse_args()

    if args.mode == "single":
        render_single(args.tracks, args.detections, args.out,
                      args.fps, args.title or args.tracks.stem,
                      parse_viewport(args))
    else:
        render_compare(args.tracks_left, args.tracks_right,
                       args.label_left, args.label_right,
                       args.detections, args.out, args.fps,
                       parse_viewport(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
