#!/usr/bin/env python3
"""P2-M6 m13 closing hero: replay M4 tracker output through both safety modes.

Reads M4 tracker output (frame-by-frame worker positions), runs each frame
through the kinematic supervisor and the CBF supervisor side-by-side, and
renders an MP4 with three panels:
    Left   : RELLIS camera frame (or BEV cluster overlay).
    Middle : kinematic v(t) trace.
    Right  : CBF v(t) trace.

The script invokes the safety_runner binary on a synthesised scenario CSV
constructed from the tracker output. For the qualitative demo a 5-frame
moving average is applied to v_relative (documented in m13 blog).

This script is HPC-targeted because rendering 2849 frames at 30 FPS takes
~10-15 minutes on a CPU. Use scripts/slurm/m6_rellis_hero.sh to submit.

Usage:
    python scripts/m6/animate_cbf_rellis.py \\
        --tracks results_m4/blog_renders/m13/tracks.csv \\
        --frames-dir /media/nishant/SeeGayt2/terra_perceive/m4_perframe/extracted_frames_camera \\
        --frame-stride 1 \\
        --smoothing 5 \\
        --out results_m6/figures/cbf_rellis_hero.mp4

Tracker CSV schema expected:
    frame_id, track_id, x, y, vx, vy
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def smooth_velocity(rows: list[dict], window: int) -> list[dict]:
    """Apply a centered moving average to (vx, vy) per track_id."""
    by_track: dict[int, list[dict]] = {}
    for r in rows:
        by_track.setdefault(r["track_id"], []).append(r)
    out: list[dict] = []
    for tid, track_rows in by_track.items():
        track_rows.sort(key=lambda r: r["frame_id"])
        vx = np.array([r["vx"] for r in track_rows])
        vy = np.array([r["vy"] for r in track_rows])
        kernel = np.ones(window) / window
        if len(vx) >= window:
            vx_s = np.convolve(vx, kernel, mode="same")
            vy_s = np.convolve(vy, kernel, mode="same")
        else:
            vx_s, vy_s = vx, vy
        for i, r in enumerate(track_rows):
            r2 = dict(r)
            r2["vx"] = float(vx_s[i])
            r2["vy"] = float(vy_s[i])
            out.append(r2)
    out.sort(key=lambda r: (r["frame_id"], r["track_id"]))
    return out


def write_scenario_csv(rows: list[dict], path: Path, ego_v: float = 2.0) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "worker_id", "x", "y", "vx", "vy",
                    "vehicle_v", "vehicle_dir"])
        for r in rows:
            w.writerow([r["frame_id"], r["track_id"], r["x"], r["y"],
                        r["vx"], r["vy"], ego_v, 0.0])


def run_supervisor(scenario_csv: Path, mode: str, out_dir: Path,
                   bin_path: Path, gamma: float = 1.0) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(bin_path), "--scenario", str(scenario_csv),
           "--safety-mode", mode, "--frames", "0",
           "--out", str(out_dir)]
    if mode == "cbf":
        cmd += ["--cbf-gamma", str(gamma)]
    subprocess.run(cmd, check=True)
    return out_dir / "events.csv"


def render_animation(events_kinematic: Path, events_cbf: Path,
                     out_path: Path, fps: int = 10) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    kin = np.genfromtxt(events_kinematic, delimiter=",", names=True,
                        dtype=None, encoding="utf-8")
    cbf = np.genfromtxt(events_cbf, delimiter=",", names=True,
                        dtype=None, encoding="utf-8")
    n = min(len(kin), len(cbf))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    line_k, = axes[0].plot([], [], color="#1f77b4", linewidth=1.8)
    line_c, = axes[1].plot([], [], color="#d62728", linewidth=1.8)
    for ax, title in zip(axes, ["kinematic supervisor", "CBF clamp"]):
        ax.set_xlim(0, kin["t"][n - 1])
        ax.set_ylim(0, max(kin["vel_after"].max(), cbf["vel_after"].max()) * 1.1)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("v (m/s)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.suptitle("CBF vs kinematic on RELLIS sequence 00 (M4 tracker output)")

    writer = FFMpegWriter(fps=fps, bitrate=2400)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(out_path), dpi=120):
        for i in range(n):
            line_k.set_data(kin["t"][:i + 1], kin["vel_after"][:i + 1])
            line_c.set_data(cbf["t"][:i + 1], cbf["vel_after"][:i + 1])
            writer.grab_frame()
    plt.close(fig)
    print(f"[wrote] {out_path}")


def load_tracks(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append({
                "frame_id": int(d["frame_id"]),
                "track_id": int(d["track_id"]),
                "x": float(d["x"]),
                "y": float(d["y"]),
                "vx": float(d["vx"]),
                "vy": float(d["vy"]),
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", required=True, help="M4 tracker output CSV")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--smoothing", type=int, default=5,
                   help="moving-average window on (vx, vy); set 0 to disable")
    p.add_argument("--cbf-gamma", type=float, default=1.0)
    p.add_argument("--bin", default="build/construction_perception/safety_runner")
    p.add_argument("--out", default="results_m6/figures/cbf_rellis_hero.mp4")
    args = p.parse_args()

    rows = load_tracks(Path(args.tracks))
    if args.frame_stride > 1:
        rows = [r for r in rows if r["frame_id"] % args.frame_stride == 0]
    if args.smoothing >= 2:
        rows = smooth_velocity(rows, args.smoothing)

    with tempfile.TemporaryDirectory() as tmpdir:
        scen_csv = Path(tmpdir) / "rellis_scenario.csv"
        write_scenario_csv(rows, scen_csv)

        events_kin = run_supervisor(scen_csv, "kinematic",
                                     Path(tmpdir) / "kin", Path(args.bin))
        events_cbf = run_supervisor(scen_csv, "cbf",
                                     Path(tmpdir) / "cbf", Path(args.bin),
                                     gamma=args.cbf_gamma)

        render_animation(events_kin, events_cbf, Path(args.out))


if __name__ == "__main__":
    main()
