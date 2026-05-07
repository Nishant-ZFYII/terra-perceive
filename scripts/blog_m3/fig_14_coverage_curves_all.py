"""
fig_14_coverage_curves_all.py — coverage vs frame for every M3 ablation
overlaid on a single plot.

The headline finding: across 13 ablations spanning 4 axes (pose source,
update rule, alpha, decay, covariance source), the scalar coverage metric
moves only ~0.8% (0.377 to 0.385). That's the "the metric is blind to
the variable" story; the per-cell value-vs-time plots in the other
figures carry the real argument.

Output: docs/assets/m3/coverage_curves_all.svg
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
OUT = REPO / "docs" / "assets" / "m3" / "coverage_curves_all.svg"

# Group runs by ablation axis so the legend tells a coherent story.
GROUPS = {
    "Pose source (Ablation A)": [
        ("slam_ema_perframe",  "slam (manifold)", "#1f77b4"),
        ("carto_ema_perframe", "carto",            "#ff7f0e"),
        ("icp_ema_perframe",   "icp",              "#2ca02c"),
        ("gps_ema_perframe",   "gps",              "#d62728"),
    ],
    "Update rule (Ablation B)": [
        ("slam_ema_perframe",        "ema",       "#9467bd"),
        ("slam_logodds_perframe",    "log-odds",  "#8c564b"),
        ("slam_overwrite_perframe",  "overwrite", "#e377c2"),
    ],
    "Alpha (Ablation C)": [
        ("slam_ema_alpha0p1_perframe", "a=0.1", "#bcbd22"),
        ("slam_ema_perframe",          "a=0.3", "#17becf"),
        ("slam_ema_alpha0p5_perframe", "a=0.5", "#aec7e8"),
        ("slam_ema_alpha0p7_perframe", "a=0.7", "#ffbb78"),
    ],
    "Decay (Ablation D)": [
        ("slam_ema_perframe",            "decay=0",       "#98df8a"),
        ("slam_ema_decay0p01_perframe",  "decay=0.01/s",  "#ff9896"),
        ("slam_ema_decay0p1_perframe",   "decay=0.1/s",   "#c5b0d5"),
    ],
    "Covariance source (Ablation E)": [
        ("slam_ema_perframe",                "cov=none",       "#7f7f7f"),
        ("slam_ema_covheuristic_perframe",   "cov=heuristic",  "#c49c94"),
        ("slam_ema_covg2o_perframe",         "cov=g2o",        "#f7b6d2"),
    ],
}


def main() -> None:
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    for ax, (title, runs) in zip(axes, GROUPS.items()):
        for run, label, color in runs:
            csv = M3 / run / "coverage.csv"
            if not csv.exists():
                print(f"  skip missing: {csv}")
                continue
            df = pd.read_csv(csv)
            ax.plot(df["frame_id"], df["coverage_pct"],
                    label=label, color=color, linewidth=1.0)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel("coverage")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.legend(loc="lower right", fontsize=8, ncol=2, frameon=False)

    axes[-1].set_xlabel("frame index (RELLIS sequence 00, 2847 frames)")
    fig.suptitle(
        "Coverage scalar barely moves across 13 ablations — "
        "the argument for per-cell time series",
        fontsize=12, y=0.995,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")

    # Also dump the final-coverage table to stdout for the blog body.
    print("\nFinal coverage per run (frame 2846):")
    for _title, runs in GROUPS.items():
        for run, label, _ in runs:
            csv = M3 / run / "coverage.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            print(f"  {run:38s} {label:18s} cov_final={df['coverage_pct'].iloc[-1]:.4f}")


if __name__ == "__main__":
    main()
