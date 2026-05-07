"""
fig_05_hero_final_bev.py — hero static BEV map for the M3 blog post.

Loads the final_grid.csv from slam_ema_covg2o_perframe (avoids the WorldGrid
PNG axis-convention pitfall), overlays the SLAM trajectory in matching world
coordinates, crops to the trajectory bbox + margin.

Output: docs/assets/m3/hero_final_bev.png
"""

from __future__ import annotations
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
RUN = REPO / "results" / "m3" / "slam_ema_covg2o_perframe"
OUT = REPO / "docs" / "assets" / "m3" / "hero_final_bev.png"
MARGIN_M = 20.0


def main() -> None:
    if not RUN.exists():
        sys.exit(f"missing {RUN}")

    grid, extent = load_bev_grid(RUN)
    traj = pd.read_csv(RUN / "trajectory.csv")
    xlim, ylim = trajectory_bbox([RUN / "trajectory.csv"], MARGIN_M)

    # Mask zero (unobserved) cells so they render as the figure background.
    masked = np.where(grid > 0.0, grid, np.nan)
    cmap = matplotlib.colormaps.get_cmap("viridis").copy()
    cmap.set_bad("black")

    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0,
              extent=extent, origin="lower", interpolation="nearest")
    ax.plot(traj["tx"], traj["ty"], color="#ff6b6b", linewidth=0.8, alpha=0.7,
            label="SLAM trajectory")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, world frame)", color="#dddddd", fontsize=10)
    ax.set_ylabel("y (m, world frame)", color="#dddddd", fontsize=10)
    ax.tick_params(colors="#bbbbbb", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#666666")
    ax.grid(True, color="#333333", linewidth=0.4, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9, frameon=True,
              facecolor="#000000", edgecolor="#555555", labelcolor="white")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
