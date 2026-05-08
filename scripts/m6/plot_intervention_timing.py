#!/usr/bin/env python3
"""P2-M6 m13 figure: first-intervention distance per scenario.

For each (scenario, mode) pair, find the earliest frame where scale_factor < 1
(the first time the supervisor clamps the commanded velocity) and report
d_worker at that frame. Shows how early CBF engages relative to kinematic.

Usage:
    python scripts/m6/plot_intervention_timing.py \\
        --kinematic-root results_m6/cbf_kinematic \\
        --cbf-root       results_m6/cbf_cbf \\
        --out            results_m6/figures/intervention_timing.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCENARIOS = ["head_on", "angled_20", "occluded", "multi_worker", "far_pass", "edge_of_arc"]


def first_intervention_distance(events_path: Path) -> float:
    if not events_path.exists():
        return np.nan
    data = np.genfromtxt(events_path, delimiter=",", names=True, dtype=None,
                         encoding="utf-8")
    if data.size == 0:
        return np.nan
    mask = data["scale"] < 0.999
    if not mask.any():
        return np.nan
    return float(data["d_worker"][mask][0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kinematic-root", default="results_m6/cbf_kinematic")
    p.add_argument("--cbf-root", default="results_m6/cbf_cbf")
    p.add_argument("--out", default="results_m6/figures/intervention_timing.png")
    args = p.parse_args()

    kin = [first_intervention_distance(Path(args.kinematic_root) / s / "events.csv")
           for s in SCENARIOS]
    cbf = [first_intervention_distance(Path(args.cbf_root) / s / "events.csv")
           for s in SCENARIOS]

    x = np.arange(len(SCENARIOS))
    w = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, kin, w, label="kinematic", color="#1f77b4")
    ax.bar(x + w/2, cbf, w, label="CBF", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS, rotation=20, ha="right")
    ax.set_ylabel("d_worker at first intervention (m)")
    ax.set_title("First-intervention distance: kinematic vs CBF")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[wrote] {out}")
    print(f"kinematic first-intervention distances: {dict(zip(SCENARIOS, kin))}")
    print(f"cbf first-intervention distances:       {dict(zip(SCENARIOS, cbf))}")


if __name__ == "__main__":
    main()
