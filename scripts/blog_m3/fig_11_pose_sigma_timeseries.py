"""
fig_11_pose_sigma_timeseries.py — Ablation E hero figure.

Three lines: pose_sigma per frame for cov-source in {none, heuristic, g2o}.

The story: under open-loop motion the SLAM uncertainty grows monotonically.
The g2o marginal correctly tracks that growth (sigma climbs from 0 to ~2.2 m
over 2847 frames). The edge-information-sum heuristic ignores loop-closure
correlations and dramatically underestimates uncertainty (sigma reaches only
~0.17 m at the same frame). The 'none' baseline is identically zero.

The y-axis is log-scaled so all three lines are readable on the same plot
despite the order-of-magnitude gap.

Output: docs/assets/m3/ablation_e_pose_sigma.svg
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
M3 = REPO / "results" / "m3"
OUT = REPO / "docs" / "assets" / "m3" / "ablation_e_pose_sigma.svg"

LINES = [
    ("slam_ema_perframe",              "cov-source = none",       "#888888", "--"),
    ("slam_ema_covheuristic_perframe", "cov-source = heuristic",  "#1f77b4", "-"),
    ("slam_ema_covg2o_perframe",       "cov-source = g2o",        "#d62728", "-"),
]


def main() -> None:
    fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for run, label, color, ls in LINES:
        csv = M3 / run / "pose_sigma.csv"
        if not csv.exists():
            print(f"  skip missing: {csv}")
            continue
        df = pd.read_csv(csv)
        # Frame 0 carries a sentinel value (1000) before the optimizer warms up;
        # plot from frame 1 onward.
        df = df[df["frame_id"] >= 1].copy()
        for ax in (ax_lin, ax_log):
            ax.plot(df["frame_id"], df["pose_sigma"],
                    label=label, color=color, linestyle=ls, linewidth=1.4)

    for ax in (ax_lin, ax_log):
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_ylabel("pose_sigma  (m, sqrt-trace of P_trans)")

    ax_lin.set_title("Linear scale", fontsize=11, loc="left")
    ax_log.set_title("Log scale (same data, makes heuristic visible)",
                     fontsize=11, loc="left")
    ax_log.set_yscale("symlog", linthresh=1e-3)
    ax_log.set_xlabel("frame index")
    ax_lin.legend(loc="upper left", fontsize=10, frameon=False)

    fig.suptitle(
        "Ablation E — pose uncertainty over RELLIS seq 00\n"
        "g2o tracks the open-loop drift, heuristic underestimates by ~10x",
        fontsize=12, y=0.99,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
