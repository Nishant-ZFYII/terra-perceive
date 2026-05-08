#!/usr/bin/env python3
"""P2-M6 Phase 5 RELLIS hero: kinematic vs CBF replayed on real LiDAR data.

Two-panel side-by-side animation:
    Top    — kinematic and CBF v(t) traces growing in lockstep.
    Bottom — BEV scene per frame: raw LiDAR (gray), tracked-worker dot
             (cyan, with a tail showing recent positions), ego at origin
             (red square), forward-arc cone, range rings.

Reads:
    --lidar-dir            directory of *.bin LiDAR frames
    --kinematic-events     events.csv from `safety_runner --safety-mode kinematic`
    --cbf-events           events.csv from `safety_runner --safety-mode cbf`
    --scenario             rellis_hero.csv from rellis_clusters_to_scenario.py
                           (used to extract the worker positions per frame for
                           the BEV overlay; the ego-advance baked into x is
                           subtracted off so the worker plots at its real
                           ego-frame position)

Output:
    --out                  MP4 path on seagate.

Usage:
    python scripts/m6/animate_rellis_hero.py \\
        --lidar-dir /media/.../m4_perframe/extracted_frames \\
        --kinematic-events /tmp/rellis_hero/kinematic/events.csv \\
        --cbf-events       /tmp/rellis_hero/cbf/events.csv \\
        --scenario         scripts/m6/scenarios/rellis_hero.csv \\
        --out              /media/.../m6_animations/rellis_hero.mp4 \\
        --fps 30
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Wedge

DT = 0.1
ARC_HALF_DEG = 60.0


def load_lidar_bin(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 3), dtype=np.float32)
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def load_events(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None,
                         encoding="utf-8")


def load_scenario_workers(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_real, y) per frame in real ego-frame.
    The CSV stores x = x_real + i * vehicle_v * DT; subtract that off."""
    rows = []
    with path.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append((int(r["frame_id"]),
                          float(r["x"]),
                          float(r["y"]),
                          float(r["vehicle_v"])))
    rows.sort(key=lambda r: r[0])
    fids = np.array([r[0] for r in rows])
    xs = np.array([r[1] for r in rows])
    ys = np.array([r[2] for r in rows])
    vs = np.array([r[3] for r in rows])
    x_real = xs - fids * vs * DT
    return x_real, ys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lidar-dir", required=True, type=Path)
    p.add_argument("--kinematic-events", required=True, type=Path)
    p.add_argument("--cbf-events", required=True, type=Path)
    p.add_argument("--scenario", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--xlim", nargs=2, type=float, default=[-25.0, 25.0])
    p.add_argument("--ylim", nargs=2, type=float, default=[-25.0, 25.0])
    p.add_argument("--lidar-decim", type=int, default=2)
    args = p.parse_args()

    lidar_paths = sorted(args.lidar_dir.glob("*.bin"))
    print(f"[hero] {len(lidar_paths)} LiDAR frames")

    kin = load_events(args.kinematic_events)
    cbf = load_events(args.cbf_events)
    n = min(len(lidar_paths), len(kin), len(cbf))
    if args.max_frames > 0:
        n = min(n, args.max_frames)

    wx, wy = load_scenario_workers(args.scenario)
    n = min(n, len(wx))
    print(f"[hero] rendering {n} frames")

    # Mark frames where kinematic engaged but CBF did not (the comparison story).
    kin_engaged = kin["scale"][:n] < 0.999
    cbf_engaged = cbf["scale"][:n] < 0.999

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.2], hspace=0.18)
    ax_v = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # v(t) panel.
    ax_v.set_xlim(0, kin["t"][n - 1])
    ax_v.set_ylim(-0.05, 2.3)
    ax_v.set_xlabel("t (s)", fontsize=9)
    ax_v.set_ylabel("v (m/s)", fontsize=9)
    ax_v.set_title("kinematic vs CBF on RELLIS sequence 00 (full 2849 frames)",
                    fontsize=10)
    ax_v.grid(True, alpha=0.3)
    line_kin, = ax_v.plot([], [], color="#1f77b4", linewidth=1.6, label="kinematic")
    line_cbf, = ax_v.plot([], [], color="#d62728", linewidth=1.6, label="CBF")
    ax_v.legend(fontsize=8, loc="upper right")

    # Static engagement annotations on the v-axis (so the viewer can see where
    # the differences are in advance).
    for fi in np.where(kin_engaged & ~cbf_engaged)[0]:
        ax_v.axvspan(fi * DT - 0.05, fi * DT + 0.05, color="#1f77b4", alpha=0.15)

    # BEV panel.
    ax_b.set_xlim(*args.xlim)
    ax_b.set_ylim(*args.ylim)
    ax_b.set_aspect("equal")
    ax_b.set_xlabel("x — forward (m)", fontsize=9)
    ax_b.set_ylabel("y — left (m)", fontsize=9)
    ax_b.grid(True, alpha=0.25)
    for r in (5, 10, 20):
        ax_b.add_patch(plt.Circle((0, 0), r, fill=False, color="#888",
                                    linewidth=0.5, alpha=0.4))
    arc = Wedge((0, 0), 25.0, -ARC_HALF_DEG, ARC_HALF_DEG,
                facecolor="#d62728", alpha=0.05, edgecolor="none")
    ax_b.add_patch(arc)

    lidar_sc = ax_b.scatter([], [], c="#888", s=0.8, alpha=0.5)
    worker_dot, = ax_b.plot([], [], "o", color="#00bcd4", markersize=12,
                              markeredgecolor="black", markeredgewidth=1.0,
                              zorder=10)
    worker_trail, = ax_b.plot([], [], "-", color="#00bcd4", linewidth=1.5,
                                alpha=0.4)
    ax_b.plot(0, 0, "s", color="black", markersize=10, zorder=11)

    text = ax_b.text(0.02, 0.97, "", transform=ax_b.transAxes, fontsize=9,
                       verticalalignment="top", family="monospace",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                 edgecolor="gray", alpha=0.8))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, bitrate=4800)
    print(f"[hero] writing {args.out}")
    with writer.saving(fig, str(args.out), dpi=120):
        for fi in range(n):
            cloud = load_lidar_bin(lidar_paths[fi])
            if args.lidar_decim > 1:
                cloud = cloud[::args.lidar_decim]
            mask = ((cloud[:, 0] >= args.xlim[0]) & (cloud[:, 0] <= args.xlim[1])
                    & (cloud[:, 1] >= args.ylim[0]) & (cloud[:, 1] <= args.ylim[1]))
            lidar_sc.set_offsets(np.column_stack([cloud[mask, 0], cloud[mask, 1]])
                                  if mask.any() else np.zeros((0, 2)))

            x_real = wx[fi]
            if x_real < 50:  # not a sentinel
                worker_dot.set_data([x_real], [wy[fi]])
                tail_lo = max(0, fi - 30)
                worker_trail.set_data(wx[tail_lo:fi + 1], wy[tail_lo:fi + 1])
            else:
                worker_dot.set_data([], [])
                worker_trail.set_data([], [])

            line_kin.set_data(kin["t"][:fi + 1], kin["vel_after"][:fi + 1])
            line_cbf.set_data(cbf["t"][:fi + 1], cbf["vel_after"][:fi + 1])

            text.set_text(
                f"frame {fi:05d}  t={fi * DT:.1f}s\n"
                f"kin v={kin['vel_after'][fi]:.2f}  scale={kin['scale'][fi]:.2f}\n"
                f"cbf v={cbf['vel_after'][fi]:.2f}  scale={cbf['scale'][fi]:.2f}"
            )
            writer.grab_frame()
            if fi % 200 == 0:
                print(f"  rendered {fi}/{n}")
    plt.close(fig)
    print(f"[hero] wrote {args.out}")


if __name__ == "__main__":
    main()
