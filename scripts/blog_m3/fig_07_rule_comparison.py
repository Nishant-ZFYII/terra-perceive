"""
fig_07_rule_comparison.py — Ablation B static side-by-side.

Three final BEV maps (EMA / log-odds / overwrite) using identical SLAM poses
and identical LiDAR data, same crop and colormap. Loads from final_grid.csv
to avoid the PNG axis-swap pitfall.

Output: docs/assets/m3/ablation_b_rules_static.png
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
OUT = REPO / "docs" / "assets" / "m3" / "ablation_b_rules_static.png"
MARGIN_M = 25.0

RUNS = [
    ("slam_ema_perframe",       "EMA (a=0.3)"),
    ("slam_logodds_perframe",   "Log-odds (OctoMap-style)"),
    ("slam_overwrite_perframe", "Overwrite"),
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
    # Same poses across all three runs, so trajectory bbox from the first.
    xlim, ylim = trajectory_bbox([M3 / RUNS[0][0] / "trajectory.csv"], MARGIN_M)
    traj = pd.read_csv(M3 / RUNS[0][0] / "trajectory.csv")

    cmap = matplotlib.colormaps.get_cmap("viridis").copy()
    cmap.set_bad("black")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor="black")

    for ax, (run_name, pretty) in zip(axes, RUNS):
        run_dir = M3 / run_name
        ax.set_facecolor("black")

        try:
            grid, extent = load_bev_grid(run_dir)
            masked = np.where(grid > 0.0, grid, np.nan)
            ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0,
                      extent=extent, origin="lower", interpolation="nearest")
        except FileNotFoundError as e:
            print(f"  skip {run_name}: {e}")

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
        title = pretty if cov is None else f"{pretty}\n(final cov = {cov:.3f})"
        ax.set_title(title, color="white", fontsize=12, loc="center")

    fig.suptitle(
        "Ablation B: same SLAM poses, three update rules, "
        "near-identical scalar coverage (within 0.6%)",
        color="white", fontsize=14, y=0.995,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
