#!/usr/bin/env python3
"""P2-M6 m12 calibration figure: sigma(r) overlaid with Ouster OS1-64 anchors.

Pure analytic plot. No data needed. Ships with the m12 blog as the
near-cover figure for the noise-model derivation.

Usage:
    python scripts/m6/plot_sigma_r.py --out results_m6/figures/sigma_r.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sigma(r: np.ndarray, sigma_0: float, k: float) -> np.ndarray:
    return sigma_0 + k * r * r


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sigma-0", type=float, default=0.01)
    p.add_argument("--sigma-k", type=float, default=0.0001)
    p.add_argument("--out", default="results_m6/figures/sigma_r.png")
    args = p.parse_args()

    r = np.linspace(0.5, 60.0, 400)
    s = sigma(r, args.sigma_0, args.sigma_k)

    # Datasheet-anchored expectations (qualitative).
    anchors_r = np.array([2.0, 25.0, 50.0])
    anchors_s = np.array([0.015, 0.06, 0.25])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(r, s * 100, label=r"$\sigma(r) = \sigma_0 + k\,r^2$", color="#d62728",
            linewidth=2.0)
    ax.scatter(anchors_r, anchors_s * 100, color="black", s=40, zorder=5,
               label="Ouster OS1-64 datasheet anchors (cm)")
    ax.set_xlabel("range r (m)")
    ax.set_ylabel(r"$\sigma$ (cm)")
    ax.set_title(rf"Range-noise model with $\sigma_0$={args.sigma_0:.3g} m, "
                 rf"$k$={args.sigma_k:.4g} m / m$^2$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
