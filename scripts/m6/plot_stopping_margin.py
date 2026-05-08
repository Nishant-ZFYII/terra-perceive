#!/usr/bin/env python3
"""P2-M6 m13 figure: stopping-margin distribution per scenario per mode.

Reads metrics.json from each scenario's run under both modes; produces a
grouped bar chart of min_margin (final stopping distance beyond d_stop).

Usage:
    python scripts/m6/plot_stopping_margin.py \\
        --kinematic-root results_m6/cbf_kinematic \\
        --cbf-root       results_m6/cbf_cbf \\
        --out            results_m6/figures/stopping_margin.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCENARIOS = ["head_on", "angled_20", "occluded", "multi_worker", "far_pass", "edge_of_arc"]


def load_metric(path: Path, key: str) -> float:
    try:
        with open(path) as f:
            return float(json.load(f).get(key, np.nan))
    except FileNotFoundError:
        return np.nan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kinematic-root", default="results_m6/cbf_kinematic")
    p.add_argument("--cbf-root", default="results_m6/cbf_cbf")
    p.add_argument("--out", default="results_m6/figures/stopping_margin.png")
    args = p.parse_args()

    kin = [load_metric(Path(args.kinematic_root) / s / "metrics.json", "min_margin")
           for s in SCENARIOS]
    cbf = [load_metric(Path(args.cbf_root) / s / "metrics.json", "min_margin")
           for s in SCENARIOS]

    x = np.arange(len(SCENARIOS))
    w = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, kin, w, label="kinematic", color="#1f77b4")
    ax.bar(x + w/2, cbf, w, label="CBF", color="#d62728")
    ax.axhline(0.5, color="green", linewidth=1, linestyle="--",
               label="d_safe_min = 0.5 m")
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS, rotation=20, ha="right")
    ax.set_ylabel("min margin = min(d_worker - d_stop) over the run (m)")
    ax.set_title("Stopping margin per scenario: kinematic vs CBF")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[wrote] {out}")
    print(f"kinematic margins: {dict(zip(SCENARIOS, kin))}")
    print(f"cbf margins:       {dict(zip(SCENARIOS, cbf))}")


if __name__ == "__main__":
    main()
