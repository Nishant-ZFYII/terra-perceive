#!/usr/bin/env python3
"""P2-M6 m13 animation: per-scenario BEV view of vehicle vs workers vs safety state.

Reads the scenario CSV (worker positions over time) and the kinematic + CBF
events.csv (vehicle's velocity trace). Reconstructs the ego trajectory by
forward-integrating the vehicle's velocity, then renders a side-by-side
animation:
    Left  — BEV scene: ego (red dot, with forward-arc cone) and workers (blue
            dots), under kinematic mode
    Right — same scene under CBF mode
The text overlay shows v, scale_factor, and the supervisor's reason.

One MP4 per scenario.

Usage:
    python scripts/m6/animate_scenario_bev.py \\
        --scenarios-dir scripts/m6/scenarios \\
        --kinematic-root results_m6/cbf_kinematic \\
        --cbf-root results_m6/cbf_cbf \\
        --out-dir /media/nishant/SeeGayt2/terra_perceive/m6_animations/scenarios \\
        --fps 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Wedge

SCENARIOS = ["head_on", "angled_20", "occluded", "multi_worker", "far_pass", "edge_of_arc"]
DT = 0.1


def load_scenario(path: Path):
    rows = []
    with path.open() as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append({
                "frame_id": int(d["frame_id"]),
                "worker_id": int(d["worker_id"]),
                "x": float(d["x"]),
                "y": float(d["y"]),
                "vx": float(d["vx"]),
                "vy": float(d["vy"]),
                "vehicle_v": float(d["vehicle_v"]),
                "vehicle_dir": float(d["vehicle_dir"]),
            })
    return rows


def integrate_ego(events: np.ndarray, dir_rad: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """From events.csv vel_after column, integrate ego (x, y) with dt=0.1."""
    n = len(events)
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] + events["vel_after"][i] * np.cos(dir_rad) * DT
        y[i] = y[i - 1] + events["vel_after"][i] * np.sin(dir_rad) * DT
    return x, y


def render(scen_rows: list[dict], events_kin: np.ndarray, events_cbf: np.ndarray,
           out_path: Path, fps: int) -> None:
    by_frame = {}
    for r in scen_rows:
        by_frame.setdefault(r["frame_id"], []).append(r)

    n_frames = min(len(events_kin), len(events_cbf), max(by_frame) + 1)
    dir_rad = scen_rows[0]["vehicle_dir"]

    ego_kx, ego_ky = integrate_ego(events_kin, dir_rad)
    ego_cx, ego_cy = integrate_ego(events_cbf, dir_rad)

    # Pre-compute scene extent. Exclude "off-stage" worker positions
    # (anything > 40 m from the origin, used as a sentinel for "not yet visible"
    # in the occluded scenario) so the autoscaled bbox stays tight on the
    # actual scene.
    on_stage = [r for r in scen_rows
                if (r["x"] * r["x"] + r["y"] * r["y"]) ** 0.5 < 40.0]
    if on_stage:
        all_x = [r["x"] for r in on_stage] + list(ego_kx) + list(ego_cx)
        all_y = [r["y"] for r in on_stage] + list(ego_ky) + list(ego_cy)
    else:
        all_x = list(ego_kx) + list(ego_cx)
        all_y = list(ego_ky) + list(ego_cy)
    pad = 2.0
    xlim = (min(all_x) - pad, max(all_x) + pad)
    ylim = (min(all_y) - pad, max(all_y) + pad)
    # Also clip rendered worker scatter to on-stage cells (off-stage shows
    # nothing rather than a dot at (100, 100)).
    for r in scen_rows:
        r["_on_stage"] = (r["x"] * r["x"] + r["y"] * r["y"]) ** 0.5 < 40.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    titles = ["kinematic supervisor", "CBF clamp"]
    arc_half = np.deg2rad(60)

    # Per-axis artists.
    artists = []
    for ax, title in zip(axes, titles):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ego_dot, = ax.plot([], [], "o", color="#d62728", markersize=10, zorder=5)
        ego_trail, = ax.plot([], [], "-", color="#d62728", linewidth=1.5, alpha=0.4)
        worker_scat = ax.scatter([], [], c="#1f77b4", s=60, edgecolor="black",
                                  linewidth=0.8, zorder=4)
        arc = Wedge((0, 0), 12.0,
                    np.degrees(dir_rad - arc_half),
                    np.degrees(dir_rad + arc_half),
                    facecolor="#d62728", alpha=0.06, edgecolor="none")
        ax.add_patch(arc)
        text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=9,
                       verticalalignment="top", family="monospace",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                 edgecolor="gray", alpha=0.8))
        artists.append({"ax": ax, "ego": ego_dot, "trail": ego_trail,
                        "workers": worker_scat, "arc": arc, "text": text})

    fig.suptitle(f"{out_path.stem}: kinematic vs CBF", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    with writer.saving(fig, str(out_path), dpi=130):
        for fi in range(n_frames):
            workers = [w for w in by_frame.get(fi, []) if w.get("_on_stage", True)]
            wx = [w["x"] for w in workers]
            wy = [w["y"] for w in workers]

            for k, (ax_data, evx, evy, ev) in enumerate(zip(
                    artists,
                    [ego_kx, ego_cx],
                    [ego_ky, ego_cy],
                    [events_kin, events_cbf])):
                ax_data["ego"].set_data([evx[fi]], [evy[fi]])
                ax_data["trail"].set_data(evx[:fi + 1], evy[:fi + 1])
                ax_data["workers"].set_offsets(np.column_stack([wx, wy])
                                                if wx else np.zeros((0, 2)))
                ax_data["arc"].set_center((evx[fi], evy[fi]))
                rule = ev["rule"][fi] if ev["rule"].dtype.kind in ("U", "S") else ""
                if isinstance(rule, bytes):
                    rule = rule.decode()
                ax_data["text"].set_text(
                    f"t={fi*DT:.1f}s  v={ev['vel_after'][fi]:.2f}\n"
                    f"scale={ev['scale'][fi]:.2f}\n{rule}"
                )
            writer.grab_frame()
    print(f"[wrote] {out_path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios-dir", default="scripts/m6/scenarios")
    p.add_argument("--kinematic-root", default="results_m6/cbf_kinematic")
    p.add_argument("--cbf-root", default="results_m6/cbf_cbf")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()

    for scen in SCENARIOS:
        scen_csv = Path(args.scenarios_dir) / f"{scen}.csv"
        kin_events = Path(args.kinematic_root) / scen / "events.csv"
        cbf_events = Path(args.cbf_root) / scen / "events.csv"
        if not (scen_csv.exists() and kin_events.exists() and cbf_events.exists()):
            print(f"[skip] missing files for {scen}")
            continue
        rows = load_scenario(scen_csv)
        events_kin = np.genfromtxt(kin_events, delimiter=",", names=True,
                                    dtype=None, encoding="utf-8")
        events_cbf = np.genfromtxt(cbf_events, delimiter=",", names=True,
                                    dtype=None, encoding="utf-8")
        render(rows, events_kin, events_cbf,
               Path(args.out_dir) / f"{scen}.mp4", args.fps)


if __name__ == "__main__":
    main()
