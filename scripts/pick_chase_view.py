#!/usr/bin/env python3
"""Interactive Open3D viewer for picking the chase-camera angle.

Loads ONE LiDAR frame plus its tracked clusters, opens the standard
Open3D viewer. Navigate the camera to the angle you want, then press:
  - 'P' (capture) to dump the current camera params to the terminal
  - 'Q' to quit

The dumped params (eye, look_at, up, fov) plug straight into
animate_tracker_3d_chase.py via --cam-back / --cam-up / --cam-look-fwd
or as raw values you can edit into the script.

Usage:
  /home/nishant/anaconda3/envs/foundation_stereo/bin/python \\
      scripts/pick_chase_view.py \\
      --frame 1780 \\
      --lidar-dir   data/extracted_frames_full \\
      --clusters-dir /media/nishant/SeeGayt2/terra_perceive/m4_perframe/clusters_k3_eps07
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d


def load_kitti_bin(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def height_color(z: np.ndarray) -> np.ndarray:
    z_norm = np.clip((z - z.min()) / (z.max() - z.min() + 1e-6), 0.0, 1.0)
    out = np.zeros((len(z), 3))
    out[:, 0] = 0.20 + 0.55 * z_norm
    out[:, 1] = 0.30 + 0.50 * z_norm
    out[:, 2] = 0.55 - 0.35 * z_norm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame",        type=int, required=True)
    ap.add_argument("--lidar-dir",    type=Path, required=True)
    ap.add_argument("--clusters-dir", type=Path, required=True)
    ap.add_argument("--width",        type=int, default=1600)
    ap.add_argument("--height",       type=int, default=1000)
    args = ap.parse_args()

    fid = args.frame
    raw = load_kitti_bin(args.lidar_dir / f"{fid:06d}.bin")
    r = np.linalg.norm(raw[:, :2], axis=1)
    raw = raw[(r < 70.0) & (raw[:, 2] > -3.0) & (raw[:, 2] < 12.0)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(height_color(raw[:, 2]))

    # Add a forward-axis arrow so you can see which way is +X (ego forward)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.15, cone_radius=0.35,
        cylinder_height=3.0, cone_height=0.8,
    )
    # The default arrow points along +Z; rotate it to point along +X
    R = arrow.get_rotation_matrix_from_axis_angle([0, np.pi / 2, 0])
    arrow.rotate(R, center=[0, 0, 0])
    arrow.paint_uniform_color([1.0, 0.2, 0.2])

    # Coordinate axes at origin (red=X, green=Y, blue=Z)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.5)

    geometries = [pcd, arrow, axes]

    # Standard interactive viewer (keys: drag rotate, scroll zoom, shift+drag pan)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"frame {fid}  —  press P to dump camera, Q to quit",
                      width=args.width, height=args.height)
    for g in geometries:
        vis.add_geometry(g)

    # Background dark
    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.06, 0.08])
    opt.point_size = 2.0

    def dump_camera(v):
        ctrl = v.get_view_control()
        params = ctrl.convert_to_pinhole_camera_parameters()
        ext = np.asarray(params.extrinsic)
        intr = params.intrinsic.intrinsic_matrix
        # Camera-to-world: invert extrinsic
        T_cw = np.linalg.inv(ext)
        eye = T_cw[:3, 3]
        # Open3D camera looks down -Z in camera frame; world look-at = eye + R * (0,0,-1) * d
        forward_world = -T_cw[:3, 2]
        # Open3D camera Y axis points DOWN in the image (CV convention).
        # The "up" passed to OffscreenRenderer.setup_camera should be the
        # world direction that maps to image-up — so we negate.
        up_world = -T_cw[:3, 1]
        look_at = eye + forward_world * 10.0  # arbitrary 10m ahead

        print()
        print("=" * 60)
        print(f"frame {fid}  camera dump")
        print("=" * 60)
        print(f"eye      = [{eye[0]:.3f}, {eye[1]:.3f}, {eye[2]:.3f}]")
        print(f"look_at  = [{look_at[0]:.3f}, {look_at[1]:.3f}, {look_at[2]:.3f}]")
        print(f"up       = [{up_world[0]:.3f}, {up_world[1]:.3f}, {up_world[2]:.3f}]")
        print(f"intrinsic fx={intr[0,0]:.1f}  fy={intr[1,1]:.1f}")
        # In ego frame (raw is in ego frame already), the chase params are simply:
        print()
        print("→ animate_tracker_3d_chase.py args (copy-paste, single line):")
        print(f"   --cam-x {eye[0]:.2f} --cam-y {eye[1]:.2f} --cam-z {eye[2]:.2f} "
              f"--look-x {look_at[0]:.2f} --look-y {look_at[1]:.2f} --look-z {look_at[2]:.2f} "
              f"--up-x {up_world[0]:.3f} --up-y {up_world[1]:.3f} --up-z {up_world[2]:.3f}")
        print("=" * 60)
        return False

    vis.register_key_callback(ord("P"), dump_camera)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
