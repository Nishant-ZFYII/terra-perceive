#!/usr/bin/env python3
"""P2-M6: synthetic safety scenarios for the m13 CBF ablation.

Six deterministic CSVs that exercise the supervisor across the regimes the
blog needs to cover. The vehicle starts at the origin, faces +x at 2 m/s,
and the worker positions evolve frame-by-frame at dt=0.1 s.

CSV schema (consumed by safety_runner --scenario):
    frame_id, worker_id, x, y, vx, vy, vehicle_v, vehicle_dir

vehicle_v and vehicle_dir on row 0 seed the supervisor's initial state.
The scenarios assume a 100-frame horizon (10 seconds at dt=0.1) unless
explicitly extended.

Scenarios:
    head_on        Worker stationary 12 m ahead. The cleanest CBF stop.
    angled_20      Worker drifting laterally past the forward arc.
    occluded       Worker appears at frame 30 at 4 m (late detection).
    multi_worker   Two workers, the nearer one closes faster.
    far_pass       Worker walks across the field at 25 m perpendicular.
    edge_of_arc    Worker at +/-30 deg edge of forward arc, slowly closing.

Usage:
    python scripts/m6/synth_safety_scenarios.py --out scripts/m6/scenarios/
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

DT = 0.1
N_FRAMES = 100
EGO_V = 2.0
EGO_DIR = 0.0


def write_scenario(out: Path, name: str, rows: list[tuple]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["frame_id", "worker_id", "x", "y", "vx", "vy", "vehicle_v", "vehicle_dir"]
        )
        for r in rows:
            w.writerow(r)
    print(f"[wrote] {out} ({len(rows)} rows)")


def head_on() -> list[tuple]:
    rows = []
    for f in range(N_FRAMES):
        rows.append((f, 0, 12.0, 0.0, 0.0, 0.0, EGO_V, EGO_DIR))
    return rows


def angled_20() -> list[tuple]:
    rows = []
    for f in range(N_FRAMES):
        x = 12.0 - 0.0 * f * DT
        y = 0.0 + 0.5 * f * DT
        rows.append((f, 0, x, y, 0.0, 0.5, EGO_V, EGO_DIR))
    return rows


def occluded() -> list[tuple]:
    # Ego cruises at EGO_V for 30 frames before the worker is detected, so the
    # worker spawn x must be ahead of where the ego will be at frame 30.
    # At dt=0.1, 30 frames at v=2 covers ~6 m. Spawn worker 4 m ahead of that.
    rows = []
    spawn_x = EGO_V * 30 * DT + 4.0  # = 10.0 m absolute
    for f in range(N_FRAMES):
        if f < 30:
            rows.append((f, 0, 100.0, 100.0, 0.0, 0.0, EGO_V, EGO_DIR))
        else:
            x = spawn_x - 0.5 * (f - 30) * DT
            rows.append((f, 0, x, 0.0, -0.5, 0.0, EGO_V, EGO_DIR))
    return rows


def multi_worker() -> list[tuple]:
    rows = []
    for f in range(N_FRAMES):
        rows.append((f, 0, 8.0 - 0.6 * f * DT, 0.4, -0.6, 0.0, EGO_V, EGO_DIR))
        rows.append((f, 1, 14.0, -0.3, 0.0, 0.0, EGO_V, EGO_DIR))
    return rows


def far_pass() -> list[tuple]:
    rows = []
    for f in range(N_FRAMES):
        x = 25.0
        y = -5.0 + 1.0 * f * DT
        rows.append((f, 0, x, y, 0.0, 1.0, EGO_V, EGO_DIR))
    return rows


def edge_of_arc() -> list[tuple]:
    rows = []
    for f in range(N_FRAMES):
        r = 12.0 - 0.2 * f * DT
        theta = math.radians(28.0)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        rows.append((f, 0, x, y, -0.2 * math.cos(theta), -0.2 * math.sin(theta),
                     EGO_V, EGO_DIR))
    return rows


SCENARIOS = {
    "head_on": head_on,
    "angled_20": angled_20,
    "occluded": occluded,
    "multi_worker": multi_worker,
    "far_pass": far_pass,
    "edge_of_arc": edge_of_arc,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="scripts/m6/scenarios", help="output directory")
    args = p.parse_args()

    out_dir = Path(args.out)
    for name, fn in SCENARIOS.items():
        write_scenario(out_dir / f"{name}.csv", name, fn())
    print(f"[done] wrote {len(SCENARIOS)} scenarios to {out_dir}")


if __name__ == "__main__":
    main()
