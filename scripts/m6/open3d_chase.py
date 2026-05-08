#!/usr/bin/env python3
"""P2-M6: Open3D 3D chase-camera animations of RELLIS sequence 00.

Three modes share the same rendering loop and chase-camera framing:

    --mode raw         every-frame raw LiDAR, height-colored.
    --mode ransac      every-frame raw LiDAR, ground (gray) vs obstacle (yellow)
                       split by a single-plane RANSAC.
    --mode confidence  every-frame raw LiDAR, points colored by the M6
                       per-cell probabilistic confidence (looked up from the
                       perframe snapshot CSV at the point's (x, y) cell).

Camera defaults to the locked-in M10 chase-cam preset A (wider, drone-style).
Pattern, geometry helpers, and ffmpeg stitch reused from
`scripts/animate_tracker_3d_open3d.py`.

Usage examples:

    python scripts/m6/open3d_chase.py --mode raw \\
        --lidar-dir /media/nishant/SeeGayt2/terra_perceive/m4_perframe/extracted_frames \\
        --out /media/nishant/SeeGayt2/terra_perceive/m6_animations/open3d_raw.mp4

    python scripts/m6/open3d_chase.py --mode confidence \\
        --lidar-dir /media/nishant/SeeGayt2/terra_perceive/m4_perframe/extracted_frames \\
        --conf-snapshots /media/nishant/SeeGayt2/terra_perceive/m6_perframe/trav_probabilistic_perframe/snapshots \\
        --out /media/nishant/SeeGayt2/terra_perceive/m6_animations/open3d_confidence.mp4
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
    import open3d.visualization.rendering as rendering
except ImportError:
    print("ERROR: open3d is not installed in this env.", file=sys.stderr)
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_lidar_bin(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 3), dtype=np.float32)
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def height_to_rgb(z: np.ndarray, zmin: float = -2.0, zmax: float = 3.0) -> np.ndarray:
    """Map z to viridis."""
    norm = np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0)
    rgba = plt.cm.viridis(norm)
    return rgba[:, :3]


def confidence_to_rgb(c: np.ndarray) -> np.ndarray:
    norm = np.clip(c, 0.0, 1.0)
    rgba = plt.cm.viridis(norm)
    return rgba[:, :3]


def ransac_ground(points: np.ndarray, *, distance_thresh: float = 0.20,
                  iterations: int = 200) -> np.ndarray:
    """Open3D's built-in plane RANSAC. Returns inlier (ground) mask."""
    if len(points) < 50:
        return np.zeros(len(points), dtype=bool)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane_model, inlier_idx = pcd.segment_plane(
        distance_threshold=distance_thresh, ransac_n=3,
        num_iterations=iterations)
    # Reject if the fitted plane's normal is far from vertical (sometimes a
    # building wall would dominate the plane fit on RELLIS); fall back to a
    # smaller threshold and a height filter.
    a, b, c, _ = plane_model
    if abs(c) < 0.5:
        # Plane is too vertical; fall back to a near-ground heuristic.
        z_med = float(np.median(points[:, 2]))
        return points[:, 2] < z_med + 0.3
    mask = np.zeros(len(points), dtype=bool)
    mask[inlier_idx] = True
    return mask


def load_confidence_grid(path: Path) -> dict[tuple[int, int], float]:
    """Snapshot CSV -> {(ix, iy): confidence}."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    out: dict[tuple[int, int], float] = {}
    with path.open() as f:
        next(f)  # header
        for line in f:
            parts = line.split(",")
            if len(parts) < 8:
                continue
            ix = int(parts[0]); iy = int(parts[1])
            try:
                c = float(parts[7])
            except ValueError:
                continue
            out[(ix, iy)] = c
    return out


def color_by_confidence(points: np.ndarray, conf_grid: dict[tuple[int, int], float],
                        *, x_min: float = -5.0, y_min: float = -15.0,
                        resolution: float = 0.5) -> np.ndarray:
    """For each point: if its cell is in the snapshot, color by confidence
    (viridis). Otherwise (obstacle point above the ground plane, or off-grid)
    fade to a darker gray so the ground confidence story pops visually."""
    n = len(points)
    # Vectorized cell-key lookup using numpy hashing.
    ix = np.floor((points[:, 0] - x_min) / resolution).astype(np.int32)
    iy = np.floor((points[:, 1] - y_min) / resolution).astype(np.int32)
    rgb = np.full((n, 3), 0.20, dtype=np.float64)  # dim background
    if not conf_grid:
        return rgb
    confs = np.zeros(n, dtype=float)
    has_conf = np.zeros(n, dtype=bool)
    # Bulk dict.get over python dict — still O(n) but fast in C since ix/iy
    # are int32 numpy arrays. Use a packed key.
    cells = list(zip(ix.tolist(), iy.tolist()))
    for i, k in enumerate(cells):
        v = conf_grid.get(k)
        if v is not None:
            confs[i] = v
            has_conf[i] = True
    rgb[has_conf] = confidence_to_rgb(confs[has_conf])
    return rgb


# --- static reference geometry (lifted from animate_tracker_3d_open3d.py) ---

def ground_grid_lineset(extent=40.0, step=5.0,
                        color=(0.18, 0.18, 0.22)) -> o3d.geometry.LineSet:
    ticks = np.arange(-extent, extent + 1e-3, step)
    pts, edges = [], []
    idx = 0
    for t in ticks:
        pts.append([t, -extent, 0]); pts.append([t, extent, 0])
        edges.append([idx, idx + 1]); idx += 2
        pts.append([-extent, t, 0]); pts.append([extent, t, 0])
        edges.append([idx, idx + 1]); idx += 2
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(edges, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(edges), 1)))
    return ls


def range_ring_lineset(radius: float, n_segments: int = 80,
                       color=(0.25, 0.25, 0.25)) -> o3d.geometry.LineSet:
    theta = np.linspace(0, 2 * np.pi, n_segments + 1)
    pts = np.stack([radius * np.cos(theta), radius * np.sin(theta),
                    np.zeros_like(theta)], axis=1)
    edges = np.array([[i, i + 1] for i in range(n_segments)])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (n_segments, 1)))
    return ls


def ego_lineset(length=2.0, width=1.2, height=0.8) -> o3d.geometry.LineSet:
    pts = np.array([
        [-length/2, -width/2, 0.0], [length/2, -width/2, 0.0],
        [length/2,  width/2, 0.0], [-length/2,  width/2, 0.0],
        [-length/2, -width/2, height], [length/2, -width/2, height],
        [length/2,  width/2, height], [-length/2,  width/2, height],
    ], dtype=np.float64)
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector(np.tile([0.0, 1.0, 1.0], (12, 1)))
    return ls


def render_frame(renderer, frame_idx, lidar_paths, snap_paths, args,
                 mat_pcd, mat_line) -> np.ndarray:
    renderer.scene.clear_geometry()

    cloud = load_lidar_bin(lidar_paths[frame_idx])
    if cloud.size == 0:
        return np.zeros((args.height, args.width, 3), dtype=np.uint8)
    if args.point_decim > 1:
        cloud = cloud[::args.point_decim]

    if args.mode == "raw":
        rgb = height_to_rgb(cloud[:, 2])
    elif args.mode == "ransac":
        ground_mask = ransac_ground(cloud)
        rgb = np.empty((len(cloud), 3))
        rgb[ground_mask] = (0.50, 0.50, 0.55)            # gray ground
        rgb[~ground_mask] = (0.95, 0.78, 0.18)            # warm yellow obstacle
    elif args.mode == "confidence":
        if frame_idx >= len(snap_paths):
            rgb = np.full((len(cloud), 3), 0.30)
        else:
            cg = load_confidence_grid(snap_paths[frame_idx])
            rgb = color_by_confidence(cloud, cg)
    else:
        raise SystemExit(f"unknown mode {args.mode}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    renderer.scene.add_geometry("pcd", pcd, mat_pcd)

    if args.ground_grid:
        renderer.scene.add_geometry("grid",
                                    ground_grid_lineset(extent=args.view_range),
                                    mat_line)
    for r in (10.0, 20.0, 30.0, 40.0):
        if r <= args.view_range:
            renderer.scene.add_geometry(f"ring_{int(r)}",
                                        range_ring_lineset(r), mat_line)
    renderer.scene.add_geometry("ego", ego_lineset(), mat_line)

    eye = [args.cam_x, args.cam_y, args.cam_z]
    lookat = [args.look_x, args.look_y, args.look_z]
    up = [0.0, 0.0, 1.0]
    renderer.setup_camera(args.fov, lookat, eye, up)
    img = renderer.render_to_image()
    return np.asarray(img)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["raw", "ransac", "confidence"], required=True)
    p.add_argument("--lidar-dir", type=Path, required=True)
    p.add_argument("--conf-snapshots", type=Path,
                   help="Required for --mode confidence: directory of frame_*.csv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-end", type=int, default=-1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--point-size", type=float, default=2.5)
    p.add_argument("--point-decim", type=int, default=2)
    p.add_argument("--fov", type=float, default=50.0)
    # Locked-in chase-cam preset A (wider) from project_open3d_chase_camera.md.
    p.add_argument("--cam-x", type=float, default=50.0)
    p.add_argument("--cam-y", type=float, default=-2.0)
    p.add_argument("--cam-z", type=float, default=12.0)
    p.add_argument("--look-x", type=float, default=-25.0)
    p.add_argument("--look-y", type=float, default=0.0)
    p.add_argument("--look-z", type=float, default=0.0)
    p.add_argument("--view-range", type=float, default=40.0)
    p.add_argument("--no-ground-grid", dest="ground_grid", action="store_false")
    p.set_defaults(ground_grid=True)
    p.add_argument("--keep-frames", action="store_true")
    args = p.parse_args()

    if args.mode == "confidence" and args.conf_snapshots is None:
        sys.exit("--mode confidence requires --conf-snapshots")

    lidar_paths = sorted(args.lidar_dir.glob("*.bin"))
    if not lidar_paths:
        sys.exit(f"no .bin files in {args.lidar_dir}")
    end = args.frame_end if args.frame_end > 0 else len(lidar_paths) - 1
    indices = list(range(args.frame_start, end + 1, args.stride))
    print(f"[open3d] {len(indices)} frames at {args.width}x{args.height}, "
          f"mode={args.mode}, fps={args.fps}")

    snap_paths: list[Path] = []
    if args.mode == "confidence":
        snap_paths = sorted(args.conf_snapshots.glob("frame_*.csv"))
        print(f"[open3d] {len(snap_paths)} confidence snapshots")

    renderer = rendering.OffscreenRenderer(args.width, args.height)
    renderer.scene.set_background([0.05, 0.05, 0.07, 1.0])

    mat_pcd = rendering.MaterialRecord()
    mat_pcd.shader = "defaultUnlit"
    mat_pcd.point_size = args.point_size

    mat_line = rendering.MaterialRecord()
    mat_line.shader = "unlitLine"
    mat_line.line_width = 2.0

    tmpdir = Path(tempfile.mkdtemp(prefix="m6_o3d_"))
    print(f"[open3d] staging frames in {tmpdir}")
    try:
        for k, fi in enumerate(indices):
            img = render_frame(renderer, fi, lidar_paths, snap_paths, args,
                               mat_pcd, mat_line)
            o3d.io.write_image(str(tmpdir / f"f_{k:06d}.png"),
                               o3d.geometry.Image(img))
            if k % 100 == 0:
                print(f"  rendered {k + 1}/{len(indices)}  (frame {fi})")
        print(f"[open3d] stitching with ffmpeg")
        if shutil.which("ffmpeg") is None:
            sys.exit("ffmpeg missing")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(args.fps),
            "-i", str(tmpdir / "f_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(args.out),
        ]
        subprocess.run(cmd, check=True)
        print(f"[open3d] wrote {args.out}")
    finally:
        if not args.keep_frames:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
