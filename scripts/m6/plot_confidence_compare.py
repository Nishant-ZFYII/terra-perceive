#!/usr/bin/env python3
"""P2-M6 m12 hero figures: confidence-vs-range curves and integrated AUC.

Reads snapshots from heuristic and probabilistic traversability_runner runs
on the same RELLIS sequence. Produces:
    - confidence_vs_range.png : curves of mean confidence vs range bin, both modes.
    - confidence_compare_frame.png : side-by-side BEV scatter of confidence at
      a chosen frame.

Headline scalar (printed and written to confidence_compare.json):
    integrated AUC of |c_prob - c_heur| over r in [5, 30] m.

Usage:
    python scripts/m6/plot_confidence_compare.py \\
        --heuristic-root     results_m6/trav_heuristic \\
        --probabilistic-root results_m6/trav_probabilistic \\
        --out-dir            results_m6/figures \\
        --hero-frame         01500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_snapshots(root: Path) -> np.ndarray:
    """Load all snapshot CSVs under root/snapshots/ into one structured array."""
    rows = []
    snap_dir = root / "snapshots"
    if not snap_dir.exists():
        return np.array([])
    for csv in sorted(snap_dir.glob("frame_*.csv")):
        d = np.genfromtxt(csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
        if d.size > 0:
            rows.append(d)
    if not rows:
        return np.array([])
    return np.concatenate(rows)


def confidence_vs_range_curve(arr: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean confidence per range bin."""
    out = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        m = (arr["range"] >= edges[i]) & (arr["range"] < edges[i + 1])
        if m.any():
            out[i] = arr["confidence"][m].mean()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heuristic-root", required=True)
    p.add_argument("--probabilistic-root", required=True)
    p.add_argument("--out-dir", default="results_m6/figures")
    p.add_argument("--hero-frame", default="01500")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h_arr = load_snapshots(Path(args.heuristic_root))
    p_arr = load_snapshots(Path(args.probabilistic_root))
    if h_arr.size == 0 or p_arr.size == 0:
        print("[error] no snapshots found in one or both roots")
        return

    edges = np.linspace(0, 35, 36)  # 1 m bins
    centers = 0.5 * (edges[:-1] + edges[1:])
    h_curve = confidence_vs_range_curve(h_arr, edges)
    p_curve = confidence_vs_range_curve(p_arr, edges)

    # Integrated AUC of |c_prob - c_heur| over r in [5, 30] m. Trapezoidal.
    mask = (centers >= 5.0) & (centers <= 30.0)
    diff = np.abs(p_curve - h_curve)
    diff_clean = np.where(np.isnan(diff), 0.0, diff)
    auc = np.trapz(diff_clean[mask], centers[mask])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(centers, h_curve, color="#1f77b4", linewidth=2.0, label="heuristic")
    ax.plot(centers, p_curve, color="#d62728", linewidth=2.0, label="probabilistic")
    ax.fill_between(centers, h_curve, p_curve,
                    where=~np.isnan(h_curve) & ~np.isnan(p_curve),
                    alpha=0.15, color="gray",
                    label=f"|Δ| AUC over [5,30]m = {auc:.3f}")
    ax.set_xlabel("range r (m)")
    ax.set_ylabel("mean cell confidence")
    ax.set_title("Mean confidence vs range: heuristic vs probabilistic")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_vs_range.png", dpi=140, bbox_inches="tight")
    print(f"[wrote] {out_dir / 'confidence_vs_range.png'}")

    # Hero side-by-side BEV scatter at one frame.
    h_frame = np.genfromtxt(
        Path(args.heuristic_root) / "snapshots" / f"frame_{args.hero_frame}.csv",
        delimiter=",", names=True, dtype=None, encoding="utf-8")
    p_frame = np.genfromtxt(
        Path(args.probabilistic_root) / "snapshots" / f"frame_{args.hero_frame}.csv",
        delimiter=",", names=True, dtype=None, encoding="utf-8")
    if h_frame.size > 0 and p_frame.size > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        for ax, arr, title in [(axes[0], h_frame, "heuristic"),
                                (axes[1], p_frame, "probabilistic")]:
            sc = ax.scatter(arr["x_center"], arr["y_center"],
                            c=arr["confidence"], cmap="viridis",
                            s=14, vmin=0, vmax=1)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(f"{title}: frame {args.hero_frame}")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            plt.colorbar(sc, ax=ax, label="confidence", shrink=0.85)
        fig.tight_layout()
        fig.savefig(out_dir / f"confidence_compare_frame_{args.hero_frame}.png",
                    dpi=140, bbox_inches="tight")
        print(f"[wrote] {out_dir / f'confidence_compare_frame_{args.hero_frame}.png'}")

    # Headline scalar.
    summary = {
        "auc_diff_5_30": float(auc),
        "h_mean_confidence_5_30": float(np.nanmean(h_curve[mask])),
        "p_mean_confidence_5_30": float(np.nanmean(p_curve[mask])),
        "h_max_range_with_data": float(centers[~np.isnan(h_curve)][-1])
            if (~np.isnan(h_curve)).any() else None,
        "p_max_range_with_data": float(centers[~np.isnan(p_curve)][-1])
            if (~np.isnan(p_curve)).any() else None,
    }
    with open(out_dir / "confidence_compare.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] {summary}")


if __name__ == "__main__":
    main()
