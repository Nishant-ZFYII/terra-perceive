#!/usr/bin/env python3
"""animate_tracker_3d_open3d.py — Waymo-style 3D animation via Open3D
offscreen rendering.

Produces the polished look of the user's reference image: dark background,
LiDAR points sized properly with depth, bounded by green wireframe AABBs,
ego vehicle visible as a black-with-cyan-edges box at the origin, smooth
3D perspective camera.

Pipeline per frame:
    1. Load raw LiDAR (KITTI .bin, ~90k points). Decimated for speed.
    2. Load this-frame's clusters (clusters_NNNNNN.csv) and bind each
       cluster to the closest published track_id from tracks.csv.
    3. Build a single Open3D PointCloud with per-point colors:
         - raw scene points     : faint gray (#888888)
         - cluster, untracked   : mute yellow
         - cluster, tracked     : tab20 color of track_id (consistent
                                  across frames for the same physical
                                  object)
    4. Build LineSet wireframes for each cluster's AABB. Tracked clusters
       get green edges; untracked get dim yellow.
    5. Add an ego-vehicle box at the origin (black faces, cyan edges).
    6. Render to a PNG via Open3D's OffscreenRenderer.
    7. ffmpeg stitches PNGs → MP4 at the requested fps.

Inputs (paths default to project layout):
    --lidar-dir       data/extracted_frames_full
    --clusters-dir    {ext_root}/clusters_sweetspot
    --tracks-csv      results_m4/blog_renders/<cfg>/tracks.csv
    --out-mp4         results_m4/blog_renders/<cfg>/3d_open3d.mp4

Wall-clock per frame: ~0.5–2 s on a CPU (no GPU required for offscreen).
A full stride-5 render of the 2849-frame drive takes ~5–10 minutes —
much faster than the matplotlib 3D version.

Dependencies:
    pip install open3d
    (Open3D ≥ 0.13 has the OffscreenRenderer API used here.)

Usage:
    python scripts/animate_tracker_3d_open3d.py \\
        --lidar-dir    data/extracted_frames_full \\
        --clusters-dir /media/.../clusters_sweetspot \\
        --tracks-csv   results_m4/blog_renders/m13_5/tracks.csv \\
        --frame-start 1700 --frame-end 1900 --stride 5 \\
        --out-mp4      results_m4/blog_renders/m13_5/3d_open3d.mp4
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import open3d as o3d
    import open3d.visualization.rendering as rendering
except ImportError:
    print("ERROR: open3d is not installed. Install with:\n"
          "    pip install open3d\n"
          "  (or use scripts/animate_tracker_3d.py for the matplotlib fallback)",
          file=sys.stderr)
    sys.exit(1)

# Tab20 colormap stolen from matplotlib so colors match the 2D animator.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Loaders — same shape as the 2D and matplotlib-3D animators.
# -----------------------------------------------------------------------------

def load_lidar_bin(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 3), dtype=np.float32)
    raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return raw[:, :3]


def load_clusters_xyz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.empty((0, 3)), np.empty(0, dtype=int)
    pts: List[Tuple[float, float, float]] = []
    cid: List[int] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            pts.append((float(row["x"]), float(row["y"]), float(row["z"])))
            cid.append(int(row["cluster_id"]))
    return np.array(pts, dtype=np.float32), np.array(cid, dtype=np.int32)


def load_tracks_indexed(path: Path) -> Dict[int, List[Tuple[int, float, float]]]:
    out: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            out[int(row["frame_id"])].append((
                int(row["track_id"]),
                float(row["x"]),
                float(row["y"]),
            ))
    return out


def assign_cluster_to_track(centroid_xy, tracks, max_dist=5.0) -> int:
    if not tracks:
        return -1
    best, best_d2 = -1, max_dist * max_dist
    for tid, tx, ty in tracks:
        d2 = (centroid_xy[0] - tx) ** 2 + (centroid_xy[1] - ty) ** 2
        if d2 < best_d2:
            best_d2, best = d2, tid
    return best


# -----------------------------------------------------------------------------
# Scene-building helpers.
# -----------------------------------------------------------------------------

def aabb_lineset(pts: np.ndarray, color: Tuple[float, float, float]) -> o3d.geometry.LineSet:
    """12-edge wireframe AABB for `pts`. Returns an Open3D LineSet."""
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    corners = np.array([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ], dtype=np.float64)
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (12, 1)))
    return ls


def ground_grid_lineset(extent=40.0, step=5.0,
                        color=(0.18, 0.18, 0.22)) -> o3d.geometry.LineSet:
    """Square grid on the z=0 plane — gives a strong static reference so
    the eye can see when the world is moving past the ego (LiDAR points
    change but ground grid stays put because grid is rendered in ego
    frame too)."""
    ticks = np.arange(-extent, extent + 1e-3, step)
    pts = []
    edges = []
    idx = 0
    for t in ticks:
        # Lines parallel to y axis
        pts.append([t, -extent, 0]);  pts.append([t, extent, 0])
        edges.append([idx, idx + 1]); idx += 2
        # Lines parallel to x axis
        pts.append([-extent, t, 0]);  pts.append([extent, t, 0])
        edges.append([idx, idx + 1]); idx += 2
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(edges, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(edges), 1)))
    return ls


def ego_lineset(length=2.0, width=1.2, height=0.8) -> o3d.geometry.LineSet:
    """Cyan wireframe box at the origin representing the vehicle."""
    pts = np.array([
        [-length/2, -width/2, 0.0], [length/2, -width/2, 0.0],
        [length/2,  width/2, 0.0], [-length/2, width/2, 0.0],
        [-length/2, -width/2, height], [length/2, -width/2, height],
        [length/2,  width/2, height], [-length/2, width/2, height],
    ], dtype=np.float64)
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 1.0, 1.0], (12, 1)))
    return ls


def forward_arrow_lineset(length=8.0, height=0.6) -> o3d.geometry.LineSet:
    """Cyan arrow pointing in the camera-forward direction (-x in this
    RELLIS calib). Drawn LONG and bright with arrowhead barbs so the
    viewer can immediately see which way the ego is facing.

    Geometry:
        shaft           : (0,0,h) → (-length, 0, h)
        arrowhead barbs : two short diagonal lines forming a chevron
                          at the shaft's tip (-length).
    """
    tip_x = -length
    barb_dx = -length * 0.15
    barb_dy = length * 0.10
    pts = np.array([
        [0,        0,         height],   # shaft start at ego
        [tip_x,    0,         height],   # shaft end (forward)
        [tip_x - barb_dx,  barb_dy, height],
        [tip_x - barb_dx, -barb_dy, height],
    ], dtype=np.float64)
    edges = np.array([
        [0, 1],   # shaft
        [1, 2],   # arrowhead barb (left)
        [1, 3],   # arrowhead barb (right)
    ])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 1.0, 1.0], (3, 1)))
    return ls


def range_ring_lineset(radius: float, n_segments: int = 80,
                       color=(0.25, 0.25, 0.25)) -> o3d.geometry.LineSet:
    """Concentric distance ring on the ground plane (z=0)."""
    theta = np.linspace(0, 2 * np.pi, n_segments + 1)
    pts = np.stack([radius * np.cos(theta),
                    radius * np.sin(theta),
                    np.zeros_like(theta)], axis=1)
    edges = np.array([[i, i + 1] for i in range(n_segments)])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (n_segments, 1)))
    return ls


# -----------------------------------------------------------------------------
# Main render loop.
# -----------------------------------------------------------------------------

def render_frame(renderer: rendering.OffscreenRenderer,
                 frame: int, args, tracks_idx,
                 cmap, mat_pcd, mat_line) -> np.ndarray:
    """Build the scene for `frame` and render to a numpy image."""
    renderer.scene.clear_geometry()

    # 1. Raw LiDAR — faint gray context.
    full = load_lidar_bin(args.lidar_dir / f"{frame:06d}.bin")
    if full.size:
        full = full[::args.point_decim]

    # 2. Clusters → track assignment.
    cluster_xyz, cluster_cid = load_clusters_xyz(
        args.clusters_dir / f"clusters_{frame:06d}.csv")
    tracks_this_frame = tracks_idx.get(frame, [])

    # Build a single PointCloud with per-point RGB.
    all_pts = []
    all_cols = []

    if full.size:
        all_pts.append(full)
        all_cols.append(np.tile([0.45, 0.45, 0.45], (full.shape[0], 1)))

    # Iterate clusters; color points; draw bounding boxes only if requested.
    aabb_geoms: List[o3d.geometry.LineSet] = []
    if cluster_xyz.size:
        for cid in np.unique(cluster_cid):
            if cid < 0:
                continue
            mask = cluster_cid == cid
            pts = cluster_xyz[mask]
            centroid = pts.mean(axis=0)[:2]
            tid = assign_cluster_to_track(centroid, tracks_this_frame,
                                          args.max_assign_dist)
            if tid < 0:
                col = np.array([0.55, 0.55, 0.30])   # mute yellow
                box_col = (0.45, 0.45, 0.20)
            else:
                rgba = cmap(tid % 20)
                col = np.array([rgba[0], rgba[1], rgba[2]])
                box_col = (0.13, 1.0, 0.27)            # bright green
            all_pts.append(pts)
            all_cols.append(np.tile(col, (pts.shape[0], 1)))
            if args.bboxes and pts.shape[0] >= 2:
                aabb_geoms.append(aabb_lineset(pts, box_col))

    if all_pts:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
        pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_cols))
        renderer.scene.add_geometry("pcd", pcd, mat_pcd)

    for i, geom in enumerate(aabb_geoms):
        renderer.scene.add_geometry(f"aabb_{i}", geom, mat_line)

    # Static reference geometry — gives the viewer a strong directional
    # anchor since the ego frame is fixed (the WORLD moves past).
    if args.ground_grid:
        renderer.scene.add_geometry("grid",
                                    ground_grid_lineset(extent=args.view_range),
                                    mat_line)
    for r in [10.0, 20.0, 30.0, 40.0]:
        if r > args.view_range:
            break
        renderer.scene.add_geometry(f"ring_{int(r)}",
                                    range_ring_lineset(r), mat_line)

    # Ego marker + LONG forward arrow (longer + brighter than before so
    # forward direction is unmistakable, addresses "can't tell which way
    # the bot is facing" feedback).
    renderer.scene.add_geometry("ego", ego_lineset(), mat_line)
    renderer.scene.add_geometry("ego_fwd",
                                forward_arrow_lineset(length=8.0),
                                mat_line)

    # Camera. Chase-cam style: behind ego (positive x in this frame's
    # convention is "behind"), elevated, looking forward into -x.
    eye    = [args.cam_dist * np.cos(np.radians(args.azim)) * np.cos(np.radians(args.elev)),
              args.cam_dist * np.sin(np.radians(args.azim)) * np.cos(np.radians(args.elev)),
              args.cam_dist * np.sin(np.radians(args.elev))]
    lookat = [-args.lookat_offset, 0.0, 0.5]
    up     = [0.0, 0.0, 1.0]
    renderer.setup_camera(args.fov, lookat, eye, up)

    img = renderer.render_to_image()
    return np.asarray(img)


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--lidar-dir",     type=Path, required=True)
    p.add_argument("--clusters-dir",  type=Path, required=True)
    p.add_argument("--tracks-csv",    type=Path, required=True)
    p.add_argument("--frame-start",   type=int,  default=0)
    p.add_argument("--frame-end",     type=int,  default=2848)
    p.add_argument("--stride",        type=int,  default=5)
    p.add_argument("--fps",           type=int,  default=10)
    p.add_argument("--width",         type=int,  default=1920)
    p.add_argument("--height",        type=int,  default=1080)
    p.add_argument("--point-size",    type=float, default=4.0,
                   help="rendered LiDAR point size in pixels")
    p.add_argument("--fov",           type=float, default=60.0)
    p.add_argument("--cam-dist",      type=float, default=35.0)
    p.add_argument("--elev",          type=float, default=22.0,
                   help="camera elevation in degrees (above horizon)")
    p.add_argument("--azim",          type=float, default=180.0,
                   help="camera azimuth in degrees; 180 = behind ego (+x), "
                        "0 = in front of ego (-x), 90 = right side")
    p.add_argument("--lookat-offset", type=float, default=8.0,
                   help="how far in front of ego (along -x) the camera looks")
    p.add_argument("--view-range",    type=float, default=40.0)
    p.add_argument("--max-assign-dist", type=float, default=5.0)
    p.add_argument("--point-decim",   type=int, default=2,
                   help="raw LiDAR decimation factor (1=no decim, 2=half, 4=quarter)")
    p.add_argument("--bboxes",        action="store_true",
                   help="draw green wireframe AABBs around each tracked "
                        "cluster (off by default — was visual clutter)")
    p.add_argument("--no-ground-grid", dest="ground_grid",
                   action="store_false",
                   help="disable the ground-plane reference grid (on by default)")
    p.set_defaults(ground_grid=True)
    p.add_argument("--out-mp4",       type=Path, default=None,
                   help="MP4 output path (required unless --save-pngs is set)")
    p.add_argument("--save-pngs",     type=Path, default=None,
                   help="frame-by-frame inspection mode: save each rendered "
                        "frame as a PNG into this directory and skip MP4 "
                        "stitching. Useful for inspecting individual frames "
                        "(e.g. the stationary segment 1750-1830).")
    p.add_argument("--keep-frames",   action="store_true",
                   help="don't delete the per-frame PNGs after stitching")
    args = p.parse_args()

    if args.out_mp4 is None and args.save_pngs is None:
        sys.exit("ERROR: must pass either --out-mp4 (animation) or "
                 "--save-pngs DIR (frame-by-frame inspection)")

    print(f"[anim-3d-o3d] indexing tracks ...")
    tracks_idx = load_tracks_indexed(args.tracks_csv)

    frames = list(range(args.frame_start,
                        min(args.frame_end,
                            max(tracks_idx.keys()) if tracks_idx else args.frame_end) + 1,
                        args.stride))
    print(f"[anim-3d-o3d] rendering {len(frames)} frames "
          f"({args.width}x{args.height}, decim={args.point_decim}, "
          f"point_size={args.point_size}, "
          f"bboxes={'on' if args.bboxes else 'off'})")

    cmap = plt.cm.tab20

    # Set up offscreen renderer + materials.
    renderer = rendering.OffscreenRenderer(args.width, args.height)
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

    mat_pcd = rendering.MaterialRecord()
    mat_pcd.shader = "defaultUnlit"
    mat_pcd.point_size = args.point_size

    mat_line = rendering.MaterialRecord()
    mat_line.shader = "unlitLine"
    mat_line.line_width = 2.0

    # ─── PNG inspection mode ─────────────────────────────────────────────
    # User passed --save-pngs DIR: render each frame to a stable filename
    # in that dir (e.g. frame_001750.png) and skip the MP4 stitch. Useful
    # to scrub the stationary segment frame-by-frame.
    if args.save_pngs is not None:
        args.save_pngs.mkdir(parents=True, exist_ok=True)
        print(f"[anim-3d-o3d] save-pngs mode → {args.save_pngs}")
        for i, frame_id in enumerate(frames):
            img = render_frame(renderer, frame_id, args, tracks_idx,
                               cmap, mat_pcd, mat_line)
            png_path = args.save_pngs / f"frame_{frame_id:06d}.png"
            o3d.io.write_image(str(png_path), o3d.geometry.Image(img))
            if i % 10 == 0 or i == len(frames) - 1:
                print(f"  rendered {i+1}/{len(frames)}  → {png_path.name}")
        print(f"[anim-3d-o3d] DONE — {len(frames)} PNGs in {args.save_pngs}")
        return

    # ─── MP4 animation mode ─────────────────────────────────────────────
    # Stage frames in a temp dir, then ffmpeg to MP4.
    tmpdir = Path(tempfile.mkdtemp(prefix="anim3d_"))
    print(f"[anim-3d-o3d] staging frames in {tmpdir}")
    try:
        for i, frame_id in enumerate(frames):
            img = render_frame(renderer, frame_id, args, tracks_idx,
                               cmap, mat_pcd, mat_line)
            o3d.io.write_image(str(tmpdir / f"f_{i:06d}.png"),
                               o3d.geometry.Image(img))
            if i % 25 == 0:
                print(f"  frame {i}/{len(frames)}  ({frame_id})")

        print(f"[anim-3d-o3d] stitching {len(frames)} frames → MP4 via ffmpeg")
        if shutil.which("ffmpeg") is None:
            raise SystemExit("ERROR: ffmpeg not found in PATH; install ffmpeg first")
        args.out_mp4.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(args.fps),
            "-i", str(tmpdir / "f_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # ensure even dims
            str(args.out_mp4),
        ]
        subprocess.run(cmd, check=True)
        print(f"[anim-3d-o3d] wrote {args.out_mp4}")
    finally:
        if not args.keep_frames:
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            print(f"[anim-3d-o3d] kept frame PNGs at {tmpdir}")

    print(f"[anim-3d-o3d] DONE")


if __name__ == "__main__":
    main()
