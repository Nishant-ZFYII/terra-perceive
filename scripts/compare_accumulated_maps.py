#!/usr/bin/env python3
"""
compare_accumulated_maps.py — Ablation A visualizer for P2-M3.

Given four accumulator_runner output directories (one per pose source), build a
2×2 panel figure with each accumulated world map rendered in the SAME world
coordinate system, with the trajectory overlaid and coverage number labeled.

The point of sharing axes is direct visual comparison: the reader sees that the
same physical trajectory was estimated differently by each pose source, and
therefore the same LiDAR data was deposited in different world-frame locations.
Map fuzziness / corridor sharpness becomes interpretable.

Usage:
    python scripts/compare_accumulated_maps.py \\
        --slam  results/m3/slam_ema_full/ \\
        --icp   results/m3/icp_ema_full/ \\
        --gps   results/m3/gps_ema_full/ \\
        --carto results/m3/carto_ema_full/ \\
        --out   results/m3/comparison.png

Notes:
  * final_grid.csv is expected to have the metadata line
    "# rows = N, cols = M, resolution = R" followed by a column header and data
    rows. Only observed cells (obs_count > 0) are written by saveSnapshot().
  * trajectory.csv expected schema: frame_id, tx, ty, tz, qx, qy, qz, qw.
  * metrics.json expected keys: final_coverage, n_frames, update_rule, etc.

Requires: numpy, matplotlib, pandas (std Python + conda 'terra-perceive' env).
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Umeyama SE(2) alignment (translation + rotation, scale fixed at 1)
# -----------------------------------------------------------------------------
# Umeyama (1991), "Least-Squares Estimation of Transformation Parameters
# Between Two Point Patterns." Closed-form solution via SVD.
#
# Scale is fixed to 1 because all four pose sources estimate the same physical
# trajectory in metric units — differences are rigid (rotation + translation),
# NOT scale-related. If you ever compare a mono-VO trajectory to a metric one,
# set allow_scale=True and return the scale factor too.
def umeyama_se2(src: np.ndarray, ref: np.ndarray):
    """
    Find (R 2×2, t (2,)) such that R @ src.T + t ≈ ref.T on average,
    minimizing sum of squared distances. Frame-to-frame correspondence is
    assumed (row i of src ↔ row i of ref).

    Returns (R, t) or (None, None) if inputs are too small / degenerate.
    """
    if src.shape != ref.shape or src.shape[0] < 2:
        return None, None

    mu_src = src.mean(axis=0)
    mu_ref = ref.mean(axis=0)
    src_c = src - mu_src
    ref_c = ref - mu_ref

    # Cross-covariance
    H = src_c.T @ ref_c
    U, _S, Vt = np.linalg.svd(H)

    # Handle reflection: enforce det(R)=+1.
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    t = mu_ref - R @ mu_src
    return R, t


def apply_se2(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply (R, t) to Nx2 points."""
    return (R @ points.T).T + t


# -----------------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------------
META_RE = re.compile(
    r"#\s*rows\s*=\s*(\d+)\s*,\s*cols\s*=\s*(\d+)\s*,\s*resolution\s*=\s*([0-9.eE+\-]+)"
)


def load_final_grid(run_dir: Path):
    """
    Returns (rows, cols, resolution, world_x_min, world_y_min, cells_df).

    cells_df columns: row, col, risk, confidence, mean_z, pose_sigma_at_last_obs, ...

    world_x_min / world_y_min are NOT in final_grid.csv — they are the WorldGrid
    config used in accumulator_runner. We hardcode the values used for the
    ablation-A runs (x_min=y_min=-250.0, aligned with the cfg in main()).
    If a future run uses different extents, pass --world-origin explicitly.
    """
    csv_path = run_dir / "final_grid.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    with open(csv_path) as f:
        meta_line = f.readline()
    m = META_RE.search(meta_line)
    if not m:
        raise ValueError(f"malformed metadata line in {csv_path}: {meta_line!r}")
    rows = int(m.group(1))
    cols = int(m.group(2))
    res = float(m.group(3))

    # Column header is on line 2; data starts line 3.
    df = pd.read_csv(csv_path, skiprows=1)
    return rows, cols, res, df


def load_trajectory(run_dir: Path):
    """Returns Nx2 numpy array of (tx, ty) in world meters."""
    traj_path = run_dir / "trajectory.csv"
    df = pd.read_csv(traj_path)
    return df[["tx", "ty"]].to_numpy()


def load_metrics(run_dir: Path):
    """Returns the metrics.json dict, or an empty dict if missing/malformed."""
    m_path = run_dir / "metrics.json"
    if not m_path.exists():
        return {}
    try:
        with open(m_path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


# -----------------------------------------------------------------------------
# Rasterization helpers
# -----------------------------------------------------------------------------
def cells_to_raster(rows: int, cols: int, cells_df: pd.DataFrame, field: str = "risk"):
    """
    Build a dense (rows, cols) ndarray of float values from the sparse observed-cell
    DataFrame. Unobserved cells are NaN so matplotlib can render them as background.
    """
    raster = np.full((rows, cols), np.nan, dtype=np.float32)
    r = cells_df["row"].to_numpy()
    c = cells_df["col"].to_numpy()
    v = cells_df[field].to_numpy()
    raster[r, c] = v
    return raster


def world_extent(rows: int, cols: int, res: float, x_min: float, y_min: float):
    """
    Returns (left, right, bottom, top) for matplotlib imshow.
    Convention: row↔x-axis (forward/north), col↔y-axis (left/east). Displayed
    with north=up by setting imshow origin='lower'.
    """
    x_max = x_min + rows * res
    y_max = y_min + cols * res
    return (y_min, y_max, x_min, x_max)


# -----------------------------------------------------------------------------
# Plot one panel
# -----------------------------------------------------------------------------
def plot_panel(ax, run_dir: Path, label: str, world_origin, shared_xlim, shared_ylim,
               R=None, t=None, point_size=0.8):
    """
    Render a single panel. If R and t are provided, they are applied as an
    SE(2) transform to BOTH the trajectory and the accumulated cells —
    aligning this source's frame to a reference source's frame (Umeyama).

    Uses scatter (point cloud) rather than imshow because after an SE(2)
    rotation, cells no longer lie on an axis-aligned grid. Scatter handles
    arbitrary point clouds cleanly at the cost of being slower for very
    dense maps (OK up to ~500K points per panel).
    """
    rows, cols, res, cells_df = load_final_grid(run_dir)
    traj = load_trajectory(run_dir)
    metrics = load_metrics(run_dir)

    x_min, y_min = world_origin

    # Compute per-cell world coordinates from (row, col) — vectorized.
    r = cells_df["row"].to_numpy()
    c = cells_df["col"].to_numpy()
    risk = cells_df["risk"].to_numpy()
    cell_x_world = x_min + (r + 0.5) * res   # forward axis
    cell_y_world = y_min + (c + 0.5) * res   # left axis
    cell_xy = np.column_stack([cell_x_world, cell_y_world])

    # Optional SE(2) alignment.
    if R is not None and t is not None:
        cell_xy = apply_se2(cell_xy, R, t)
        traj = apply_se2(traj, R, t)

    # Background — for scatter plots there's no "bad" cell semantic, so just
    # use a mid-gray axes facecolor behind the cell points.
    ax.set_facecolor((0.4, 0.4, 0.4))

    # Scatter: x-axis = world-y (east/left), y-axis = world-x (north/forward).
    ax.scatter(
        cell_xy[:, 1],
        cell_xy[:, 0],
        c=risk,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=point_size,
        marker="s",      # square, tiles better than circles
        linewidths=0,
        alpha=0.9,
    )

    # Trajectory overlay.
    ax.plot(
        traj[:, 1],
        traj[:, 0],
        color="red",
        linewidth=1.2,
        alpha=0.85,
        label="trajectory",
    )
    ax.plot(traj[0, 1],  traj[0, 0],  marker="o", color="white",  markersize=5, markeredgecolor="black")
    ax.plot(traj[-1, 1], traj[-1, 0], marker="X", color="yellow", markersize=7, markeredgecolor="black")

    # Title + coverage annotation
    coverage = metrics.get("final_coverage", float("nan"))
    n_frames = metrics.get("n_frames", "?")
    aligned_note = "" if R is None else "  [Umeyama-aligned]"
    ax.set_title(f"{label}{aligned_note}\ncoverage = {coverage:.3f} (n={n_frames})", fontsize=11)
    ax.set_xlabel("y world (m)")
    ax.set_ylabel("x world (m)")

    ax.set_xlim(shared_xlim)
    ax.set_ylim(shared_ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def compute_shared_limits_from_trajs(trajs, margin=10.0):
    """
    Union bbox over a list of trajectories (post-alignment if applicable).
    matplotlib xlim = world y, ylim = world x.
    """
    if not trajs:
        return (-50, 50), (-50, 50)
    stacked = np.vstack(trajs)
    x_world_min, x_world_max = stacked[:, 0].min(), stacked[:, 0].max()
    y_world_min, y_world_max = stacked[:, 1].min(), stacked[:, 1].max()
    xlim = (y_world_min - margin, y_world_max + margin)
    ylim = (x_world_min - margin, x_world_max + margin)
    return xlim, ylim


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slam",  type=str, required=True, help="SLAM run dir")
    parser.add_argument("--icp",   type=str, required=True, help="ICP run dir")
    parser.add_argument("--gps",   type=str, required=True, help="GPS run dir")
    parser.add_argument("--carto", type=str, required=True, help="Cartographer run dir")
    parser.add_argument("--out",   type=str, default="results/m3/comparison.png",
                        help="Output PNG path (default: results/m3/comparison.png)")
    parser.add_argument("--world-origin", type=float, nargs=2, default=(-250.0, -250.0),
                        metavar=("X_MIN", "Y_MIN"),
                        help="WorldGrid (x_min, y_min) from accumulator_runner cfg (default: -250 -250)")
    parser.add_argument("--margin", type=float, default=15.0,
                        help="Margin in meters added to the trajectory bbox for axes (default: 15.0)")
    parser.add_argument("--title", type=str, default="P2-M3 Ablation A — Accumulated BEV per pose source",
                        help="Figure super-title")
    parser.add_argument("--align-to", type=str, default="carto",
                        choices=["slam", "icp", "gps", "carto", "none"],
                        help="Umeyama-align all other sources to this reference (default: carto). "
                             "'none' disables alignment and renders each source in its own raw frame.")
    parser.add_argument("--point-size", type=float, default=0.8,
                        help="Scatter point size for cell rendering (default: 0.8).")
    args = parser.parse_args()

    sources = [
        ("slam",  "SLAM (P2-M2 manifold)", args.slam),
        ("carto", "Cartographer",          args.carto),
        ("icp",   "ICP (KISS-ICP)",        args.icp),
        ("gps",   "GPS",                   args.gps),
    ]
    # Keep the 2×2 render order in the original layout (SLAM, Carto, ICP, GPS).

    # --- Compute Umeyama transforms to the reference source ---
    transforms = {}   # key -> (R, t) or (None, None)
    if args.align_to != "none":
        try:
            ref_traj = load_trajectory(Path(dict((k, p) for k, _, p in sources)[args.align_to]))
        except FileNotFoundError:
            print(f"[warn] alignment reference '{args.align_to}' not found; skipping alignment")
            args.align_to = "none"

    for key, label, run_dir in sources:
        if args.align_to == "none" or key == args.align_to:
            transforms[key] = (None, None)   # reference (or no-align): identity
            continue
        try:
            src_traj = load_trajectory(Path(run_dir))
            n = min(len(src_traj), len(ref_traj))
            R, t = umeyama_se2(src_traj[:n], ref_traj[:n])
            transforms[key] = (R, t)
            angle_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0])) if R is not None else float("nan")
            print(f"[info] Umeyama {key}→{args.align_to}: rot={angle_deg:+.2f}°, trans=({t[0]:+.2f}, {t[1]:+.2f})")
        except Exception as e:
            print(f"[warn] Umeyama failed for {key}: {e}")
            transforms[key] = (None, None)

    # --- Compute shared axis limits from POST-ALIGNMENT trajectories ---
    aligned_trajs = []
    for key, _, run_dir in sources:
        try:
            traj = load_trajectory(Path(run_dir))
            R, t = transforms[key]
            if R is not None:
                traj = apply_se2(traj, R, t)
            aligned_trajs.append(traj)
        except FileNotFoundError:
            pass
    xlim, ylim = compute_shared_limits_from_trajs(aligned_trajs, margin=args.margin)
    print(f"[info] shared xlim={xlim}, ylim={ylim}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (key, label, run_dir) in zip(axes.flatten(), sources):
        try:
            R, t = transforms[key]
            plot_panel(ax, Path(run_dir), label, tuple(args.world_origin), xlim, ylim,
                       R=R, t=t, point_size=args.point_size)
            print(f"[info] plotted {label} from {run_dir}")
        except Exception as e:
            ax.set_title(f"{label}\n(ERROR: {e})", fontsize=10, color="red")
            ax.set_xticks([])
            ax.set_yticks([])
            print(f"[warn] failed to plot {label}: {e}")

    fig.suptitle(args.title, fontsize=14, y=0.995)

    # Shared colorbar for risk [0, 1]
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02, aspect=30)
    cbar.set_label("risk [0 = safe (purple), 1 = hazard (yellow)]", fontsize=10)

    # Legend on one panel
    axes[0, 0].legend(
        handles=[
            plt.Line2D([], [], color="red", linewidth=1.2, label="trajectory"),
            plt.Line2D([], [], marker="o", color="white", linestyle="", markeredgecolor="black", label="start"),
            plt.Line2D([], [], marker="X", color="yellow", linestyle="", markeredgecolor="black", label="end"),
        ],
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[info] saved {out_path}")


if __name__ == "__main__":
    main()
