"""
fig_12_confidence_shift_hist.py — Ablation E secondary figure.

For every cell observed in BOTH the cov=none baseline and the cov=g2o
run, plot the per-cell distribution of confidence_g2o - confidence_none.
Most cells should sit at zero or slightly negative; the cells observed
near the trajectory tail (where pose_sigma is largest) form a left tail
of strongly-downweighted cells.

This is the per-cell version of the Ablation E pose_sigma timeseries
(fig_11) — together they tell the same story at two scales.

Output: docs/assets/m3/ablation_e_confidence_hist.svg
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
M3 = REPO / "results" / "m3"
OUT = REPO / "docs" / "assets" / "m3" / "ablation_e_confidence_hist.svg"


def main() -> None:
    none_csv = M3 / "slam_ema_perframe" / "final_grid.csv"
    g2o_csv  = M3 / "slam_ema_covg2o_perframe" / "final_grid.csv"
    if not none_csv.exists() or not g2o_csv.exists():
        sys.exit(f"missing one of:\n  {none_csv}\n  {g2o_csv}")

    df_none = pd.read_csv(none_csv,  skiprows=1)
    df_g2o  = pd.read_csv(g2o_csv,   skiprows=1)
    print(f"none cells: {len(df_none)}, g2o cells: {len(df_g2o)}")

    merged = df_none[["row", "col", "confidence"]].merge(
        df_g2o[["row", "col", "confidence"]],
        on=["row", "col"], suffixes=("_none", "_g2o"),
    )
    merged["delta"] = merged["confidence_g2o"] - merged["confidence_none"]
    print(f"common cells: {len(merged)}, "
          f"mean delta = {merged['delta'].mean():.4f}, "
          f"median = {merged['delta'].median():.4f}, "
          f"min = {merged['delta'].min():.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(merged["delta"], bins=80, color="#d62728", alpha=0.85,
            edgecolor="#7a1414")
    ax.axvline(0.0, color="#666", linewidth=1.0, linestyle="--",
               label="no shift")
    ax.axvline(merged["delta"].mean(), color="#1f77b4", linewidth=1.5,
               label=f"mean = {merged['delta'].mean():.3f}")
    ax.set_xlabel("per-cell shift  (confidence with g2o cov - confidence with no cov)")
    ax.set_ylabel("number of cells")
    ax.set_title(
        "Ablation E - distribution of per-cell confidence shift, g2o vs none\n"
        f"({len(merged)} cells observed in both runs; "
        f"left tail = cells the SLAM downweighted because they were observed under high pose uncertainty)",
        fontsize=11, loc="left",
    )
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
