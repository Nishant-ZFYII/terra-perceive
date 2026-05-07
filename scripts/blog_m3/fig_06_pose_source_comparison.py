"""
fig_06_pose_source_comparison.py — Ablation A side-by-side.

Four final BEV maps, one per pose source, at the same scale and crop.
Loads from final_grid.csv directly to avoid the PNG axis-swap pitfall.

Output: docs/assets/m3/ablation_a_pose_sources.png
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from _utils import load_bev_grid, trajectory_bbox  # noqa: E402

REPO = THIS.parents[2]
M3 = REPO / "results" / "m3"
OUT = REPO / "docs" / "assets" / "m3" / "ablation_a_pose_sources.png"
MARGIN_M = 25.0

RUNS = [
    ("slam_ema_perframe",  "SLAM (manifold)"),
    ("carto_ema_perframe", "Cartographer"),
    ("icp_ema_perframe",   "ICP (KISS)"),
    ("gps_ema_perframe",   "GPS only"),
]


def _read_metric(run_dir: Path, key: str) -> float | None:
    metrics = run_dir / "metrics.json"
    if not metrics.exists():
        return None
    try:
        return float(json.loads(metrics.read_text())[key])
    except Exception:
        return None


def main() -> None:
    # Shared bbox = union of all four trajectories.
    traj_csvs = [M3 / r / "trajectory.csv" for r, _ in RUNS]
    xlim, ylim = trajectory_bbox(traj_csvs, MARGIN_M)

    cmap = matplotlib.colormaps.get_cmap("viridis").copy()
    cmap.set_bad("black")

    fig, axes = plt.subplots(2, 2, figsize=(13, 13), facecolor="black")

    for ax, (run_name, pretty) in zip(axes.flat, RUNS):
        run_dir = M3 / run_name
        ax.set_facecolor("black")

        try:
            grid, extent = load_bev_grid(run_dir)
            masked = np.where(grid > 0.0, grid, np.nan)
            ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0,
                      extent=extent, origin="lower", interpolation="nearest")
        except FileNotFoundError as e:
            print(f"  skip {run_name}: {e}")

        traj_csv = run_dir / "trajectory.csv"
        if traj_csv.exists():
            traj = pd.read_csv(traj_csv)
            ax.plot(traj["tx"], traj["ty"], color="#ff6b6b",
                    linewidth=0.7, alpha=0.7)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.tick_params(colors="#bbbbbb", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#666666")
        ax.grid(True, color="#333333", linewidth=0.3, linestyle="--", alpha=0.5)

        cov = _read_metric(run_dir, "final_coverage")
        title = pretty if cov is None else f"{pretty}  (final cov = {cov:.3f})"
        ax.set_title(title, color="white", fontsize=12, loc="left")

    fig.suptitle(
        "Ablation A: same LiDAR + same accumulator, four pose sources",
        color="white", fontsize=14, y=0.995,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
