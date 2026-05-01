#!/usr/bin/env python3
"""Open3D chase-camera 3D animation.

Renders one PNG per frame, then ffmpeg compiles to MP4. Camera sits at
a fixed offset behind and above the ego (chase camera). Points are
rendered in ego frame so the camera follows the ego automatically.
LiDAR points use a height gradient (mute palette) for scene context;
tracked clusters get saturated track-ID-keyed colors with slightly
larger point size so they pop visually against the scene.

Usage:
    python scripts/animate_tracker_3d_chase.py \\
        --lidar-dir     data/extracted_frames_full \\
        --clusters-dir  /scratch/np3129/m4_perframe/clusters_sweetspot \\
        --tracks-csv    results_m4/.../sort_on_rellis/tracks_k3_eps07.csv \\
        --frame-start   0 \\
        --frame-end     2848 \\
        --stride        1 \\
        --width         3840 \\
        --height        2160 \\
        --fps           10 \\
        --out-mp4       open3d_k3_eps07_4k.mp4 \\
        --out-frames-dir /scratch/np3129/render_frames_open3d
"""
from __future__ import annotations
import argparse
import csv
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d


def load_kitti_bin(path: Path) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return arr[:, :3]


def load_cluster_points(path: Path):
    """Return (points Nx3, cluster_ids N) including noise (-1)."""
    if not path.exists():
        return np.empty((0, 3)), np.empty(0, dtype=int)
    pts = []
    ids = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append((float(row["x"]), float(row["y"]), float(row["z"])))
            ids.append(int(row["cluster_id"]))
    return np.asarray(pts), np.asarray(ids, dtype=int)


def load_tracks_per_frame(path: Path) -> dict:
    """Return dict frame_id -> list of (track_id, x, y, z=0) tuples."""
    out = defaultdict(list)
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row["frame_id"])
            tid = int(row["track_id"])
            x = float(row["x"])
            y = float(row["y"])
            out[fid].append((tid, x, y, 0.0))
    return out


def height_color(z: np.ndarray, z_floor: float, z_ceil: float) -> np.ndarray:
    """Mute palette: dark blue at low z, soft yellow at high z."""
    z_norm = np.clip((z - z_floor) / (z_ceil - z_floor + 1e-6), 0.0, 1.0)
    out = np.zeros((len(z), 3))
    out[:, 0] = 0.20 + 0.55 * z_norm
    out[:, 1] = 0.30 + 0.50 * z_norm
    out[:, 2] = 0.55 - 0.35 * z_norm
    return out


def track_palette(track_id: int):
    palette = np.array([
        [0.121, 0.466, 0.705], [1.000, 0.498, 0.054],
        [0.172, 0.627, 0.172], [0.839, 0.152, 0.156],
        [0.580, 0.403, 0.741], [0.549, 0.337, 0.294],
        [0.890, 0.466, 0.760], [0.498, 0.498, 0.498],
        [0.737, 0.741, 0.133], [0.090, 0.745, 0.811],
        [0.682, 0.780, 0.909], [1.000, 0.733, 0.470],
        [0.596, 0.875, 0.541], [1.000, 0.596, 0.588],
        [0.772, 0.690, 0.835], [0.768, 0.611, 0.580],
        [0.968, 0.713, 0.823], [0.858, 0.858, 0.552],
        [0.619, 0.854, 0.898], [0.780, 0.780, 0.780],
    ])
    return tuple(palette[track_id % len(palette)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lidar-dir",       required=True, type=Path)
    ap.add_argument("--clusters-dir",    required=True, type=Path)
    ap.add_argument("--tracks-csv",      required=True, type=Path)
    ap.add_argument("--frame-start",     type=int, default=0)
    ap.add_argument("--frame-end",       type=int, required=True)
    ap.add_argument("--stride",          type=int, default=1)
    ap.add_argument("--width",           type=int, default=3840)
    ap.add_argument("--height",          type=int, default=2160)
    ap.add_argument("--fps",             type=int, default=10)
    ap.add_argument("--out-mp4",         required=True, type=Path)
    ap.add_argument("--out-frames-dir",  required=True, type=Path,
                    help="Per-frame PNGs land here. Re-used between runs if present.")
    # Camera position in ego frame. Default values picked interactively
    # via scripts/pick_chase_view.py. Camera sits AHEAD of ego at +X,
    # slightly up, looking further forward (drone-flyalong, not chase-from-behind).
    ap.add_argument("--cam-x",        type=float, default=28.43,
                    help="Camera x in ego frame (forward of ego, m). Farther pick.")
    ap.add_argument("--cam-y",        type=float, default=0.0)
    ap.add_argument("--cam-z",        type=float, default=11.77,
                    help="Camera height in ego frame (m).")
    ap.add_argument("--look-x",       type=float, default=37.75)
    ap.add_argument("--look-y",       type=float, default=0.0)
    ap.add_argument("--look-z",       type=float, default=11.77,
                    help="Equal to cam-z = flat gaze.")
    ap.add_argument("--up-x",         type=float, default=0.0)
    ap.add_argument("--up-y",         type=float, default=0.0)
    ap.add_argument("--up-z",         type=float, default=1.0,
                    help="World up direction. (0,0,1) = LiDAR-frame up.")
    ap.add_argument("--fov-deg",      type=float, default=75.0,
                    help="Vertical field-of-view (deg). Default was 60 implicitly; 75 widens the scene.")
    ap.add_argument("--bg-rgba",         type=float, nargs=4,
                    default=[0.05, 0.06, 0.08, 1.0])
    args = ap.parse_args()

    args.out_frames_dir.mkdir(parents=True, exist_ok=True)

    tracks_per_frame = load_tracks_per_frame(args.tracks_csv)

    rend = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    rend.scene.set_background(list(args.bg_rgba))

    eye = np.array([args.cam_x, args.cam_y, args.cam_z])
    look_at = np.array([args.look_x, args.look_y, args.look_z])
    up = np.array([args.up_x, args.up_y, args.up_z])

    pts_mat = o3d.visualization.rendering.MaterialRecord()
    pts_mat.shader = "defaultUnlit"
    pts_mat.point_size = 2.5

    track_mat = o3d.visualization.rendering.MaterialRecord()
    track_mat.shader = "defaultUnlit"
    track_mat.point_size = 6.0

    frame_ids = list(range(args.frame_start, args.frame_end + 1, args.stride))
    n_total = len(frame_ids)
    print(f"[chase] rendering {n_total} frames at {args.width}x{args.height}, "
          f"stride={args.stride}", flush=True)

    for idx, fid in enumerate(frame_ids):
        out_png = args.out_frames_dir / f"frame_{fid:06d}.png"
        if out_png.exists() and out_png.stat().st_size > 1000:
            continue

        bin_path = args.lidar_dir / f"{fid:06d}.bin"
        if not bin_path.exists():
            print(f"[chase] skip frame {fid}: missing {bin_path}", flush=True)
            continue
        cluster_path = args.clusters_dir / f"clusters_{fid:06d}.csv"

        rend.scene.clear_geometry()

        raw = load_kitti_bin(bin_path)
        r = np.linalg.norm(raw[:, :2], axis=1)
        mask = (r < 70.0) & (raw[:, 2] > -3.0) & (raw[:, 2] < 12.0)
        raw = raw[mask]
        if raw.size:
            colors = height_color(raw[:, 2], raw[:, 2].min(), raw[:, 2].max())
            pcd_raw = o3d.geometry.PointCloud()
            pcd_raw.points = o3d.utility.Vector3dVector(raw.astype(np.float64))
            pcd_raw.colors = o3d.utility.Vector3dVector(colors)
            rend.scene.add_geometry("raw", pcd_raw, pts_mat)

        cl_pts, cl_ids = load_cluster_points(cluster_path)
        if cl_pts.size:
            tracks = tracks_per_frame.get(fid, [])
            cluster_to_track = {}
            if tracks and cl_ids.max() >= 0:
                centroids = {}
                for cid in np.unique(cl_ids[cl_ids >= 0]):
                    centroids[int(cid)] = cl_pts[cl_ids == cid, :2].mean(axis=0)
                for tid, tx, ty, _ in tracks:
                    best_cid = -1
                    best_d = math.inf
                    for cid, c in centroids.items():
                        d = math.hypot(tx - c[0], ty - c[1])
                        if d < best_d:
                            best_d = d
                            best_cid = cid
                    if best_cid >= 0 and best_d < 3.0:
                        cluster_to_track[best_cid] = tid

            colors = np.full((len(cl_pts), 3), 0.45)
            for cid, tid in cluster_to_track.items():
                m = cl_ids == cid
                colors[m] = track_palette(tid)
            pcd_tracked = o3d.geometry.PointCloud()
            pcd_tracked.points = o3d.utility.Vector3dVector(cl_pts.astype(np.float64))
            pcd_tracked.colors = o3d.utility.Vector3dVector(colors)
            rend.scene.add_geometry("tracked", pcd_tracked, track_mat)

        rend.setup_camera(args.fov_deg, look_at.astype(np.float64),
                          eye.astype(np.float64), up)

        img = rend.render_to_image()
        o3d.io.write_image(str(out_png), img, 8)

        if idx % 50 == 0:
            print(f"[chase] frame {fid} ({idx+1}/{n_total})", flush=True)

    print(f"[chase] frame render done; encoding to {args.out_mp4}", flush=True)
    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-pattern_type", "glob",
        "-i", str(args.out_frames_dir / "frame_*.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-vf", f"fps={args.fps}",
        str(args.out_mp4),
    ]
    print("[chase] " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    print(f"[chase] DONE  {args.out_mp4}", flush=True)


if __name__ == "__main__":
    main()
