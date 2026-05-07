"""
fig_15_rule_distributions.py — Ablation B distribution view.

Histogram of every observed cell's FINAL risk value, one panel per
update rule. Same SLAM poses, same LiDAR data, same alpha — only the
update rule differs, so the shape of the resulting distribution is
the rule talking.

Expected:
  - Log-odds: bimodal, mass piling near 0 and 1 because the OctoMap
    saturating clamp pushes confident cells to the rails.
  - EMA: smooth, approximately a beta-like distribution between 0 and 1.
  - Overwrite: jagged, reflecting the raw last-frame LiDAR risk.

Output: docs/assets/m3/ablation_b_distributions.svg
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
OUT = REPO / "docs" / "assets" / "m3" / "ablation_b_distributions.svg"

RUNS = [
    ("slam_ema_perframe",       "EMA  (a = 0.3)",            "#1f77b4"),
    ("slam_logodds_perframe",   "Log-odds  (OctoMap-style)", "#d62728"),
    ("slam_overwrite_perframe", "Overwrite  (no memory)",    "#2ca02c"),
]


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    summaries = []
    for ax, (run_name, label, color) in zip(axes, RUNS):
        csv = M3 / run_name / "final_grid.csv"
        if not csv.exists():
            print(f"  skip missing: {csv}")
            continue
        df = pd.read_csv(csv, skiprows=1)
        risks = df["risk"].to_numpy()
        ax.hist(risks, bins=60, range=(0.0, 1.0), color=color,
                edgecolor="black", linewidth=0.4, alpha=0.9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("final risk")
        ax.set_title(label, fontsize=11, loc="left")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

        # Summary stats inset for the blog body.
        summaries.append({
            "rule": label,
            "n_cells": len(risks),
            "mean": float(risks.mean()),
            "frac_lt_0p1": float((risks < 0.1).mean()),
            "frac_gt_0p9": float((risks > 0.9).mean()),
        })

    axes[0].set_ylabel("number of observed cells")
    fig.suptitle(
        "Ablation B - final-risk distribution of every observed cell, "
        "same SLAM poses, three update rules",
        fontsize=13, y=1.02,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")

    # Print summary so the blog body has the numbers to drop in.
    print("\nDistribution summary (for blog body):")
    for s in summaries:
        print(f"  {s['rule']:30s}  n={s['n_cells']:6d}  mean={s['mean']:.3f}  "
              f"frac<0.1={s['frac_lt_0p1']:.3f}  frac>0.9={s['frac_gt_0p9']:.3f}")


if __name__ == "__main__":
    main()
