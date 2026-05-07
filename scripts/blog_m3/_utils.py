"""
_utils.py — shared helpers for the M3 blog figure scripts.

Key fact: WorldGrid uses the convention
    row index   = world-x (forward) direction
    col index   = world-y (left) direction
which is the OPPOSITE of standard image-display convention. Loading the
final_grid.png with matplotlib imshow then plotting trajectory in (tx, ty)
will misalign because imshow treats the first array index as row=y.

This module loads the sparse final_grid.csv and rebuilds the dense risk
grid with the standard [y_idx, x_idx] layout so imshow + extent +
origin='lower' renders correctly without further surgery.
"""

from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_bev_grid(run_dir: Path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Read final_grid.csv from a perframe run dir, return (risk_grid, extent).

    The returned grid has shape (n_rows_image, n_cols_image) where
    image_row = world-y index, image_col = world-x index — the standard
    matplotlib imshow layout. extent is (x_min, x_max, y_min, y_max).
    """
    csv_path = run_dir / "final_grid.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # First line is the metadata header: "# rows = 1000, cols = 1000, resolution = 0.5"
    with csv_path.open() as f:
        meta_line = f.readline()
    m = re.search(r"rows\s*=\s*(\d+),\s*cols\s*=\s*(\d+),\s*resolution\s*=\s*([\d.]+)", meta_line)
    if not m:
        raise ValueError(f"could not parse meta header in {csv_path}: {meta_line!r}")
    n_rows = int(m.group(1))
    n_cols = int(m.group(2))
    res = float(m.group(3))

    df = pd.read_csv(csv_path, skiprows=1)
    if df.empty:
        return np.zeros((n_rows, n_cols), dtype=np.float32), \
               (-res * n_cols / 2.0, res * n_cols / 2.0,
                -res * n_rows / 2.0, res * n_rows / 2.0)

    # WorldGrid convention: csv 'row' = world-x index, csv 'col' = world-y index.
    # For matplotlib imshow we want grid[y_idx, x_idx], so:
    grid = np.zeros((n_rows, n_cols), dtype=np.float32)
    grid[df["col"].to_numpy(), df["row"].to_numpy()] = df["risk"].to_numpy(dtype=np.float32)

    origin_x = -res * n_cols / 2.0
    origin_y = -res * n_rows / 2.0
    extent = (origin_x, origin_x + n_cols * res,
              origin_y, origin_y + n_rows * res)
    return grid, extent


_FRAME_ID_RE = re.compile(r"frame_(\d+)\.csv$")


def list_snapshots(run_dir: Path, every: int = 1) -> list[tuple[int, Path]]:
    """Return [(frame_id, path), ...] of snapshots, sub-sampled by `every`."""
    snap_dir = run_dir / "snapshots"
    if not snap_dir.is_dir():
        raise FileNotFoundError(snap_dir)
    out: list[tuple[int, Path]] = []
    for p in sorted(snap_dir.glob("frame_*.csv")):
        m = _FRAME_ID_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return out[::every]


def snapshot_to_dense(snap_path: Path, rows: int = 1000, cols: int = 1000
                      ) -> np.ndarray:
    """Read one snapshot CSV and return a dense (rows, cols) risk array.
    Honors the WorldGrid axis convention (csv 'col' -> y_idx, csv 'row' -> x_idx)."""
    df = pd.read_csv(snap_path, skiprows=1)
    grid = np.zeros((rows, cols), dtype=np.float32)
    if not df.empty:
        grid[df["col"].to_numpy(), df["row"].to_numpy()] = df["risk"].to_numpy(dtype=np.float32)
    return grid


def auto_xlim(dfs: list[pd.DataFrame], margin: int = 50) -> tuple[int, int]:
    """Given several cell_history DataFrames, return an x-axis range that
    spans the first-to-last non-NaN observation across all of them.
    Useful for zooming in on cells whose action sits in a narrow frame
    window, without misleadingly cropping NaN gaps that mean 'not yet
    observed'."""
    first, last = None, None
    for df in dfs:
        valid = df.dropna(subset=["risk"])
        if valid.empty:
            continue
        f0 = int(valid["frame_id"].iloc[0])
        f1 = int(valid["frame_id"].iloc[-1])
        first = f0 if first is None else min(first, f0)
        last = f1 if last is None else max(last, f1)
    if first is None or last is None:
        return (0, 2847)
    return (max(0, first - margin), last + margin)


def cell_history(run_dir: Path, csv_row: int, csv_col: int,
                 every: int = 10) -> pd.DataFrame:
    """Trace one cell's risk across the per-frame snapshots of `run_dir`.

    csv_row / csv_col are the WorldGrid CSV indices (csv_row -> world_x_idx,
    csv_col -> world_y_idx; see the axis-convention memory). Sub-sampling by
    `every` keeps reads manageable: 2847 snapshots / 10 = 285 points, plenty
    smooth for a line plot.

    Returns DataFrame(frame_id, risk). Rows where the cell has not yet been
    observed are filled with NaN so matplotlib draws a gap (rather than a
    misleading 0-line).
    """
    snap_dir = run_dir / "snapshots"
    if not snap_dir.is_dir():
        raise FileNotFoundError(snap_dir)
    paths = sorted(snap_dir.glob("frame_*.csv"))[::every]
    rows: list[tuple[int, float]] = []
    for path in paths:
        m = _FRAME_ID_RE.search(path.name)
        if not m:
            continue
        fid = int(m.group(1))
        df = pd.read_csv(path, skiprows=1)
        hit = df[(df["row"] == csv_row) & (df["col"] == csv_col)]
        if hit.empty:
            rows.append((fid, np.nan))
        else:
            rows.append((fid, float(hit["risk"].iloc[0])))
    return pd.DataFrame(rows, columns=["frame_id", "risk"])


def trajectory_bbox(traj_csvs: list[Path], margin_m: float = 25.0
                    ) -> tuple[tuple[float, float], tuple[float, float]]:
    """Union bbox of one or more trajectory.csv files, padded by margin.
    Returns ((xlim_lo, xlim_hi), (ylim_lo, ylim_hi))."""
    xs: list[float] = []
    ys: list[float] = []
    for p in traj_csvs:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        xs.extend(df["tx"].tolist())
        ys.extend(df["ty"].tolist())
    if not xs:
        raise FileNotFoundError(f"no trajectory rows in {traj_csvs}")
    span = max(max(xs) - min(xs), max(ys) - min(ys)) + 2 * margin_m
    cx = (max(xs) + min(xs)) / 2.0
    cy = (max(ys) + min(ys)) / 2.0
    return ((cx - span / 2.0, cx + span / 2.0),
            (cy - span / 2.0, cy + span / 2.0))
