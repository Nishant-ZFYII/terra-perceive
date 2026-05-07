"""
pick_hero_cell.py — find a single (csv_row, csv_col) cell that has
interesting risk dynamics across the per-frame snapshots of a run.

The chosen cell is then read in the same way by figures #8 (rule
comparison time-series), #9 (alpha sweep), and #10 (decay sweep) so
all three plots tell consistent stories about *the same physical
location on the map*.

Selection criteria (the part the user implements):
  - Cell must be observed at least N_MIN_OBS times.
  - Cell's risk must not be "stuck" (variance > some threshold).
  - Cell should sit near the trajectory (not a one-off far-field ray).
  - Among candidates passing the filter, pick the one with the highest
    temporal variance — that's where the update-rule ablation story
    will be most visually striking.

Usage:
    python scripts/blog_m3/pick_hero_cell.py
    -> writes scripts/blog_m3/HERO_CELL.json with {"row": ..., "col": ...,
       "world_x": ..., "world_y": ..., "n_obs": ..., "var": ...}

NOTE on the WorldGrid axis convention:
  csv 'row' field = world-x index (forward)
  csv 'col' field = world-y index (left)
  origin = (-250, -250), resolution = 0.5 m
  see scripts/blog_m3/_utils.py for the load helper.
"""

from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
REPO = THIS.parents[2]

# Use the demo run as the reference for hero-cell selection.
# This run's poses are the SLAM-manifold ones, same as Ablation B / C / D
# baselines, so the chosen cell is meaningful across all three sweeps.
REF_RUN = REPO / "results" / "m3" / "slam_ema_covg2o_perframe"

OUT = THIS.parent / "HERO_CELL.json"

# Filter knobs — adjust as needed.
N_MIN_OBS = 8          # cell must be re-observed across at least this many snapshots
SAMPLE_EVERY = 20      # read every Nth snapshot (speed; 2847 snapshots is a lot)
VAR_FLOOR = 0.005      # minimum risk variance to count as "interesting"


def list_snapshots(run_dir: Path) -> list[Path]:
    snap_dir = run_dir / "snapshots"
    if not snap_dir.is_dir():
        sys.exit(f"missing snapshots/ in {run_dir}")
    return sorted(snap_dir.glob("frame_*.csv"))


def main() -> None:
    snaps = list_snapshots(REF_RUN)
    snaps = snaps[::SAMPLE_EVERY]
    print(f"reading {len(snaps)} snapshots (every {SAMPLE_EVERY}th of {len(list_snapshots(REF_RUN))})")

    # Build a dict: (row, col) -> list of (frame_idx, risk)
    history: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for path in snaps:
        m = re.search(r"frame_(\d+)\.csv$", path.name)
        if not m:
            continue
        frame_idx = int(m.group(1))
        df = pd.read_csv(path, skiprows=1)
        if df.empty:
            continue
        for r, c, risk in zip(df["row"], df["col"], df["risk"]):
            history.setdefault((int(r), int(c)), []).append((frame_idx, float(risk)))

    print(f"unique cells observed: {len(history)}")

    # Load the trajectory once so we can rank cells by closeness to the path.
    traj = pd.read_csv(REF_RUN / "trajectory.csv")
    traj_xy = np.column_stack([traj["tx"].to_numpy(), traj["ty"].to_numpy()])

    # Score each surviving cell.
    rows = []
    for (r, c), obs in history.items():
        if len(obs) < N_MIN_OBS:
            continue
        risks = np.array([v for _, v in obs], dtype=np.float32)
        var = float(risks.var())
        if var < VAR_FLOOR:
            continue
        wx = -250.0 + r * 0.5
        wy = -250.0 + c * 0.5
        # Distance from this cell to the nearest trajectory waypoint.
        d_traj = float(np.min(np.hypot(traj_xy[:, 0] - wx, traj_xy[:, 1] - wy)))
        rows.append({
            "row": r, "col": c, "world_x": wx, "world_y": wy,
            "n_obs": len(obs), "var": var, "d_traj": d_traj,
            "risk_min": float(risks.min()), "risk_max": float(risks.max()),
        })

    if not rows:
        sys.exit("no candidate cells passed the filters — relax N_MIN_OBS or VAR_FLOOR")
    cand = pd.DataFrame(rows)

    # Combined score: high temporal variance, near the trajectory.
    # d_traj scaled into a 0..1 penalty (10 m as the soft cutoff).
    cand["score"] = cand["var"] / (1.0 + cand["d_traj"] / 10.0)
    cand = cand.sort_values("score", ascending=False).reset_index(drop=True)

    print("\nTop 10 hero-cell candidates (by score = var / (1 + d_traj/10)):")
    print(cand.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    winner = cand.iloc[0]
    out_payload = {
        "row": int(winner["row"]),
        "col": int(winner["col"]),
        "world_x": float(winner["world_x"]),
        "world_y": float(winner["world_y"]),
        "n_obs": int(winner["n_obs"]),
        "var": float(winner["var"]),
        "d_traj": float(winner["d_traj"]),
        "risk_min": float(winner["risk_min"]),
        "risk_max": float(winner["risk_max"]),
        "selection_run": REF_RUN.name,
        "sample_every": SAMPLE_EVERY,
    }
    OUT.write_text(json.dumps(out_payload, indent=2))
    print(f"\nwrote hero cell to {OUT}:")
    print(json.dumps(out_payload, indent=2))


if __name__ == "__main__":
    main()
