#!/usr/bin/env python3
"""Bar chart for the max_dist gating sweep (Ablation D).

Two grouped bars per cell:
    - id_switches         (taken from metrics.json)
    - distinct_track_ids  (taken from metrics.json; proxy for fragmentation)

Hiring-manager visual: large fonts, single PNG, story visible in one glance.

Usage:
    python scripts/plot_max_dist_sweep.py \\
        --runs   results_m4/ablation_d/d_1  results_m4/ablation_d/d_3  results_m4/ablation_d/d_10 \\
        --labels "1m" "3m" "10m" \\
        --out    results_m4/ablation_d/max_dist_sweep.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs",   nargs="+", type=Path, required=True)
    p.add_argument("--labels", nargs="+", type=str,  required=True)
    p.add_argument("--out",    type=Path, required=True)
    args = p.parse_args()

    if len(args.runs) != len(args.labels):
        sys.exit("--runs and --labels must have the same length")

    id_sw = []
    distinct = []
    for run in args.runs:
        with (run / "metrics.json").open() as f:
            m = json.load(f)
        id_sw.append(int(m.get("id_switches", 0)))
        distinct.append(int(m.get("distinct_track_ids", 0)))

    plt.rcParams.update({
        "font.size":       12,
        "axes.titlesize":  14,
        "axes.labelsize":  12,
        "legend.fontsize": 11,
    })
    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(args.labels))
    width = 0.35
    bar_sw   = ax.bar(x - width/2, id_sw,   width, color="#d6604d",
                      edgecolor="black", label="ID switches")
    bar_dist = ax.bar(x + width/2, distinct, width, color="#4393c3",
                      edgecolor="black", label="distinct track IDs")

    for bar_grp, vals in ((bar_sw, id_sw), (bar_dist, distinct)):
        for r, v in zip(bar_grp, vals):
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.05,
                    str(v), ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(args.labels)
    ax.set_xlabel("max_dist gating threshold")
    ax.set_ylabel("count")
    ax.set_title("Effect of max_dist gating on ID stability  (crossing scenario)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"[plot_max_dist_sweep] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
