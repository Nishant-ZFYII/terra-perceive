#!/usr/bin/env python3
"""P2-M6 m13 figure: velocity profile comparison kinematic vs CBF per scenario.

Reads events.csv from the kinematic and cbf runs of each scenario and writes
a 2x3 grid of v(t) plots (one panel per scenario). Hero figure for m13.

Usage:
    python scripts/m6/plot_velocity_profiles.py \\
        --kinematic-root results_m6/cbf_kinematic \\
        --cbf-root       results_m6/cbf_cbf \\
        --out            results_m6/figures/velocity_profiles.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCENARIOS = ["head_on", "angled_20", "occluded", "multi_worker", "far_pass", "edge_of_arc"]


def load_events(path: Path):
    if not path.exists():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.size == 0:
        return None
    return data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kinematic-root", default="results_m6/cbf_kinematic")
    p.add_argument("--cbf-root", default="results_m6/cbf_cbf")
    p.add_argument("--out", default="results_m6/figures/velocity_profiles.png")
    args = p.parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=False, sharey=False)
    axes = axes.flatten()
    for i, scen in enumerate(SCENARIOS):
        ax = axes[i]
        kin = load_events(Path(args.kinematic_root) / scen / "events.csv")
        cbf = load_events(Path(args.cbf_root) / scen / "events.csv")
        if kin is not None:
            ax.plot(kin["t"], kin["vel_after"], color="#1f77b4",
                    linewidth=1.6, label="kinematic")
        if cbf is not None:
            ax.plot(cbf["t"], cbf["vel_after"], color="#d62728",
                    linewidth=1.6, label="CBF")
        ax.set_title(scen, fontsize=10)
        ax.set_xlabel("t (s)", fontsize=8)
        ax.set_ylabel("v (m/s)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Velocity profile: kinematic supervisor vs CBF clamp",
                 fontsize=12)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
