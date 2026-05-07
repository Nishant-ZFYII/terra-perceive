"""
fig_13_trajectory_sigma_color.py — trajectory colored by pose_sigma.

XY scatter of the SLAM trajectory, with each point colored by the
pose_sigma at that frame. The "uncertainty cone widening along the
path" effect: open-loop drift accumulates, sigma climbs from ~0 at
the start to ~2.2 m at the end.

Source: slam_ema_covg2o_perframe (cov-source=g2o, the path that
produces meaningful sigma).

Output: docs/assets/m3/trajectory_sigma_color.svg
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "results" / "m3" / "slam_ema_covg2o_perframe"
OUT = REPO / "docs" / "assets" / "m3" / "trajectory_sigma_color.svg"


def main() -> None:
    traj_csv = RUN / "trajectory.csv"
    sigma_csv = RUN / "pose_sigma.csv"
    if not traj_csv.exists() or not sigma_csv.exists():
        sys.exit(f"missing one of:\n  {traj_csv}\n  {sigma_csv}")

    traj = pd.read_csv(traj_csv)
    sigma = pd.read_csv(sigma_csv)
    df = traj.merge(sigma, on="frame_id")

    # Frame 0 carries a sentinel pose_sigma = 1000 before the optimizer warms up.
    df = df[df["frame_id"] >= 1].copy()

    # Cap the colormap at a reasonable upper bound so the early near-zero
    # sigmas don't all collapse to one color.
    vmax = float(np.quantile(df["pose_sigma"], 0.99))

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(df["tx"], df["ty"], c=df["pose_sigma"], cmap="plasma",
                    s=4, vmin=0.0, vmax=vmax)
    ax.plot(df["tx"], df["ty"], color="#cccccc", linewidth=0.4, alpha=0.5,
            zorder=0)

    ax.set_xlabel("x (m, world frame)")
    ax.set_ylabel("y (m, world frame)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("pose_sigma  (m, sqrt-trace of P_trans, g2o marginal)")

    ax.set_title(
        "SLAM trajectory colored by pose uncertainty over 2847 frames\n"
        f"sigma climbs from ~0 at start to {df['pose_sigma'].max():.2f} m at the trajectory tail",
        fontsize=11, loc="left",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
