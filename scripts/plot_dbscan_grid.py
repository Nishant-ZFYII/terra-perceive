#!/usr/bin/env python3
"""3×3 thumbnail grid showing DBSCAN clustering on RELLIS obstacle clouds.

Reads the per-cell cluster CSVs produced by `dbscan_cli` during Ablation G
and renders one matplotlib panel per (eps, min_points) cell. Each panel:
    - Top-down (x-y) scatter of the obstacle points
    - Color = cluster_id (gray for noise, distinct hue per cluster)
    - Title: "eps=X  mp=Y   K clusters, M noise"

Headline blog asset for the DBSCAN section of M10. Designed for
hiring-manager-grade visuals — wide figure, large fonts, the
parameter-sensitivity story visible at one glance.

Usage:
    python scripts/plot_dbscan_grid.py \\
        --root        results_m4/ablation_g \\
        --eps         0.3 0.5 1.0 \\
        --min-points  5 10 20 \\
        --frame       50 \\
        --out         results_m4/ablation_g/dbscan_grid.png

Cluster CSV schema (one per cell, written by dbscan_cli):
    x,y,z,cluster_id    (cluster_id = -1 for noise)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_cluster_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (XY: Nx2, cluster_id: N) arrays."""
    xy: List[Tuple[float, float]] = []
    cid: List[int] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            xy.append((float(row["x"]), float(row["y"])))
            cid.append(int(row["cluster_id"]))
    return np.array(xy), np.array(cid)


def plot_one_cell(ax, xy: np.ndarray, cid: np.ndarray,
                  eps: float, mp: int,
                  global_xlim: Tuple[float, float],
                  global_ylim: Tuple[float, float]) -> None:
    """Render one panel of the 3×3 grid."""
    # Separate noise from clusters.
    noise_mask = cid < 0
    cluster_mask = ~noise_mask
    cluster_ids = sorted(set(cid[cluster_mask]))

    # Plot noise as small gray dots (in background).
    if noise_mask.any():
        ax.scatter(xy[noise_mask, 0], xy[noise_mask, 1],
                   s=4, c="lightgray", alpha=0.5, zorder=1)

    # Plot each cluster with a distinct color.
    cmap = plt.cm.tab20
    for k, c in enumerate(cluster_ids):
        m = cid == c
        ax.scatter(xy[m, 0], xy[m, 1],
                   s=8, color=cmap(k % 20),
                   edgecolor="none", zorder=2)

    K = len(cluster_ids)
    N_noise = int(noise_mask.sum())
    ax.set_title(f"eps={eps}m  mp={mp}\n{K} clusters, {N_noise} noise",
                 fontsize=11)
    ax.set_xlim(*global_xlim)
    ax.set_ylim(*global_ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_xticklabels([])    # save vertical real-estate; keep ticks for sense of scale
    ax.set_yticklabels([])


def compute_global_extent(panels: List[np.ndarray],
                          margin_m: float = 5.0
                          ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute a bbox covering ALL points across ALL cells (rule #8)."""
    all_xy = np.vstack([xy for xy in panels if xy.size > 0])
    if all_xy.size == 0:
        return (-50.0, 50.0), (-50.0, 50.0)
    x_lo, x_hi = all_xy[:, 0].min(), all_xy[:, 0].max()
    y_lo, y_hi = all_xy[:, 1].min(), all_xy[:, 1].max()
    return ((x_lo - margin_m, x_hi + margin_m),
            (y_lo - margin_m, y_hi + margin_m))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root",       type=Path, required=True,
                   help="results_m4/ablation_g directory")
    p.add_argument("--eps",        type=float, nargs="+", required=True)
    p.add_argument("--min-points", type=int,   nargs="+", required=True)
    p.add_argument("--frame",      type=int,   required=True)
    p.add_argument("--out",        type=Path,  required=True)
    args = p.parse_args()

    eps_values = list(args.eps)
    mp_values  = list(args.min_points)
    rows = len(eps_values)   # eps along rows (top → bottom: tightest → loosest)
    cols = len(mp_values)    # min_points along columns (left → right: low → high)

    # Load every cell first so we can compute a shared extent (rule #8).
    cell_data = {}   # (eps, mp) → (xy, cid)
    panels_xy = []
    for eps in eps_values:
        for mp in mp_values:
            csv_path = (args.root /
                        f"eps_{eps}_mp_{mp}" /
                        f"clusters_{args.frame:06d}.csv")
            if not csv_path.exists():
                print(f"[plot_dbscan_grid] WARN: missing {csv_path}", file=sys.stderr)
                cell_data[(eps, mp)] = (np.empty((0, 2)), np.empty(0, dtype=int))
                continue
            xy, cid = load_cluster_csv(csv_path)
            cell_data[(eps, mp)] = (xy, cid)
            panels_xy.append(xy)

    xlim, ylim = compute_global_extent(panels_xy, margin_m=5.0)

    # Render.
    plt.rcParams.update({
        "font.size":       11,
        "axes.titlesize":  11,
        "figure.titlesize": 15,
    })
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.0 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for r, eps in enumerate(eps_values):
        for c, mp in enumerate(mp_values):
            xy, cid = cell_data[(eps, mp)]
            plot_one_cell(axes[r, c], xy, cid, eps, mp, xlim, ylim)

    fig.suptitle(
        f"DBSCAN parameter sweep on RELLIS frame {args.frame}  "
        f"(top-down view; gray = noise)",
        y=0.995,
    )
    fig.text(0.5, 0.01, "min_points →", ha="center", fontsize=12)
    fig.text(0.01, 0.5, "eps →", va="center", rotation="vertical", fontsize=12)
    fig.tight_layout(rect=(0.025, 0.025, 1.0, 0.97))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"[plot_dbscan_grid] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
