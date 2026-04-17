#!/usr/bin/env python3
"""
viz_polyscope_trajectory.py — 3D Polyscope visualization of SLAM trajectory
with LiDAR point clouds at key frames.

Usage:
  python3 scripts/viz_polyscope_trajectory.py

Then in the Polyscope window:
  - Rotate/zoom to explore the 3D scene
  - Use OBS or ffmpeg to screen-record for the blog
  - Screenshot key views for the blog post

Output: Interactive Polyscope window (manual screenshot/recording)
"""

import numpy as np
import polyscope as ps
import os
import sys

# Paths
SLAM_PATH = "data/poses_slam_manifold.csv"
ICP_PATH = "data/poses_icp.csv"
CARTO_PATH = "data/poses_carto.csv"
GPS_PATH = "data/poses_gps.csv"
LIDAR_DIR = "data/RELLIS-3D/Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/00000/os1_cloud_node_kitti_bin"

# Key frames to load point clouds at (spaced evenly, not too many for performance)
KEY_FRAMES = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700]

def umeyama_align(src, tgt):
    n = min(len(src), len(tgt))
    src, tgt = src[:n], tgt[:n]
    ms, mt = src.mean(0), tgt.mean(0)
    sc, tc = src - ms, tgt - mt
    cov = tc.T @ sc / n
    U, S, Vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2, 2] = -1
    R = U @ d @ Vt
    s = np.trace(np.diag(S) @ d) / np.var(sc, axis=0).sum()
    t = mt - s * R @ ms
    return s, R, t

def load_poses(path):
    """Load poses, return Nx3 positions and Nx4x4 transforms."""
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    positions = data[:, 1:4]  # x, y, z
    # Reconstruct rotation from quaternion if available
    if data.shape[1] >= 8:
        from scipy.spatial.transform import Rotation
        quats = data[:, 4:8]  # qw, qx, qy, qz
        # Convert to scipy format (x, y, z, w)
        quats_scipy = np.column_stack([quats[:, 1], quats[:, 2], quats[:, 3], quats[:, 0]])
        rotations = Rotation.from_quat(quats_scipy).as_matrix()
        return positions, rotations
    return positions, None

def load_bin(path):
    """Load KITTI-format .bin point cloud."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]  # x, y, z only

def main():
    print("Loading trajectories...")
    carto_pos = np.loadtxt(CARTO_PATH, delimiter=',', skiprows=1, usecols=[1, 2, 3])

    slam_pos, slam_rot = load_poses(SLAM_PATH)
    icp_pos, icp_rot = load_poses(ICP_PATH)
    gps_pos = np.loadtxt(GPS_PATH, delimiter=',', skiprows=1, usecols=[1, 2, 3])

    n = min(len(carto_pos), len(slam_pos), len(icp_pos), len(gps_pos))

    # Align SLAM and ICP to Cartographer frame
    print("Aligning trajectories...")
    s_slam, R_slam, t_slam = umeyama_align(slam_pos[:n], carto_pos[:n])
    slam_aligned = (s_slam * (R_slam @ slam_pos[:n].T).T + t_slam)

    s_icp, R_icp, t_icp = umeyama_align(icp_pos[:n], carto_pos[:n])
    icp_aligned = (s_icp * (R_icp @ icp_pos[:n].T).T + t_icp)

    s_gps, R_gps, t_gps = umeyama_align(gps_pos[:n], carto_pos[:n])
    gps_aligned = (s_gps * (R_gps @ gps_pos[:n].T).T + t_gps)

    # Initialize Polyscope
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    # Register trajectories as curve networks
    def make_edges(n_pts):
        return np.array([[i, i + 1] for i in range(n_pts - 1)])

    carto_net = ps.register_curve_network("Cartographer (ref)",
                                           carto_pos[:n], make_edges(n))
    carto_net.set_color([0.0, 0.0, 0.0])
    carto_net.set_radius(0.003)

    slam_net = ps.register_curve_network("Our SLAM",
                                          slam_aligned, make_edges(n))
    slam_net.set_color([0.0, 0.8, 0.0])
    slam_net.set_radius(0.005)

    icp_net = ps.register_curve_network("KISS-ICP",
                                         icp_aligned, make_edges(n))
    icp_net.set_color([0.0, 0.0, 1.0])
    icp_net.set_radius(0.003)

    gps_cloud = ps.register_point_cloud("GPS", gps_aligned[::10])  # subsample
    gps_cloud.set_color([1.0, 0.0, 0.0])
    gps_cloud.set_radius(0.003)
    gps_cloud.set_enabled(False)  # hidden by default, toggle on in GUI

    # Load point clouds at key frames
    print("Loading LiDAR scans at key frames...")
    for frame_id in KEY_FRAMES:
        if frame_id >= n:
            continue
        bin_path = os.path.join(LIDAR_DIR, f"{frame_id:06d}.bin")
        if not os.path.exists(bin_path):
            print(f"  {bin_path} not found, skipping")
            continue

        pts_local = load_bin(bin_path)

        # Transform to world frame using SLAM pose (aligned)
        if slam_rot is not None and frame_id < len(slam_rot):
            R_local = s_slam * R_slam @ slam_rot[frame_id]
            t_local = slam_aligned[frame_id]
            pts_world = (R_local @ pts_local.T).T + t_local
        else:
            pts_world = pts_local + slam_aligned[frame_id]

        # Subsample for performance (keep every 5th point)
        pts_world = pts_world[::5]

        cloud = ps.register_point_cloud(f"scan_{frame_id:06d}", pts_world)
        cloud.set_point_render_mode("quad")
        cloud.set_radius(0.01)

        # Color by height
        heights = pts_world[:, 2]
        cloud.add_scalar_quantity("height", heights, enabled=True, cmap='viridis')

        print(f"  Loaded frame {frame_id}: {len(pts_world)} points")

    print("\nPolyscope ready. Use the GUI to explore the 3D scene.")
    print("Tips:")
    print("  - Toggle trajectories on/off in the left panel")
    print("  - Screenshot: Ctrl+Shift+S (or use OBS for video)")
    print("  - Rotate: left-click drag, Zoom: scroll, Pan: right-click drag")

    ps.show()

if __name__ == "__main__":
    main()
