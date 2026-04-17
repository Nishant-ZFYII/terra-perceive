#!/usr/bin/env python3
"""
viz_polyscope_playback.py — Frame-by-frame SLAM playback in Polyscope.
Point clouds ACCUMULATE over time (persist, don't disappear).
Trajectory grows. GPS markers shown.

Record with OBS or: ffmpeg -f x11grab -framerate 30 -i :0 output.mp4
"""

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import os
import time

# Paths
SLAM_PATH = "data/poses_slam_manifold.csv"
CARTO_PATH = "data/poses_carto.csv"
GPS_PATH = "data/poses_gps.csv"
LIDAR_DIR = "data/RELLIS-3D/Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/00000/os1_cloud_node_kitti_bin"

# Playback settings
FRAME_STEP = 15        # show every Nth LiDAR frame
POINT_SUBSAMPLE = 8    # keep every Nth point per scan (performance)
POINT_RADIUS = 0.006   # small square-ish points
POINT_ALPHA = 0.15     # transparent so trajectory shows through

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

def load_poses_with_rotations(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    positions = data[:, 1:4]
    quats = data[:, 4:8]  # qw, qx, qy, qz
    rotations = []
    for q in quats:
        qw, qx, qy, qz = q
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        rotations.append(R)
    return positions, np.array(rotations)

def load_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]

def main():
    print("Loading data...")
    carto = np.loadtxt(CARTO_PATH, delimiter=',', skiprows=1, usecols=[1, 2, 3])
    gps_raw = np.loadtxt(GPS_PATH, delimiter=',', skiprows=1, usecols=[1, 2, 3])
    slam_pos, slam_rot = load_poses_with_rotations(SLAM_PATH)

    n = min(len(carto), len(slam_pos), len(gps_raw))
    s, R_align, t_align = umeyama_align(slam_pos[:n], carto[:n])
    slam_aligned = (s * (R_align @ slam_pos[:n].T).T + t_align)

    s_gps, R_gps, t_gps = umeyama_align(gps_raw[:n], carto[:n])
    gps_aligned = (s_gps * (R_gps @ gps_raw[:n].T).T + t_gps)

    frame_ids = list(range(0, n, FRAME_STEP))
    total_playback_frames = len(frame_ids)

    # State
    current_idx = [0]
    playing = [False]
    last_time = [time.time()]
    loaded_scans = set()

    # Accumulated world points (grows over time)
    all_pts_world = []
    all_pts_heights = []

    # Init Polyscope
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    # Cartographer reference (full, always visible, faded)
    edges = np.array([[i, i + 1] for i in range(n - 1)])
    carto_net = ps.register_curve_network("Cartographer (ref)", carto[:n], edges)
    carto_net.set_color([0.4, 0.4, 0.4])
    carto_net.set_radius(0.002)
    carto_net.set_transparency(0.25)

    # GPS markers (full trajectory, small red dots)
    gps_sub = gps_aligned[::5]  # every 5th for performance
    gps_cloud = ps.register_point_cloud("GPS markers", gps_sub)
    gps_cloud.set_color([1.0, 0.15, 0.15])
    gps_cloud.set_radius(0.004)
    gps_cloud.set_transparency(0.4)
    gps_cloud.set_point_render_mode("quad")

    # SLAM trajectory (will be re-registered each frame as it grows)
    slam_net = None

    # Accumulated point cloud (re-registered as it grows)
    accum_cloud = None

    def update_frame(idx):
        nonlocal slam_net, accum_cloud
        frame_id = frame_ids[idx]
        traj_end = min(frame_id + 1, n)

        # --- Growing SLAM trajectory ---
        if traj_end > 1:
            traj_edges = np.array([[i, i + 1] for i in range(traj_end - 1)])
            if slam_net is not None:
                ps.remove_curve_network("SLAM trajectory")
            slam_net = ps.register_curve_network(
                "SLAM trajectory", slam_aligned[:traj_end], traj_edges)
            slam_net.set_color([0.15, 0.3, 1.0])  # BLUE trajectory (stands out against green cloud)
            slam_net.set_radius(0.008)             # thicker for visibility

        # --- Keyframe markers along trajectory (every 50 frames) ---
        ps.remove_point_cloud("keyframes", error_if_absent=False)
        kf_step = 50
        kf_indices = list(range(0, traj_end, kf_step))
        if len(kf_indices) > 0:
            kf_pts = slam_aligned[kf_indices]
            kf_cloud = ps.register_point_cloud("keyframes", kf_pts)
            kf_cloud.set_color([0.8, 0.1, 0.1])   # RED keyframe markers
            kf_cloud.set_radius(0.018)
            kf_cloud.set_point_render_mode("quad")

        # --- Current position marker ---
        ps.remove_point_cloud("current_pos", error_if_absent=False)
        pos = slam_aligned[frame_id]
        marker = ps.register_point_cloud("current_pos", pos.reshape(1, 3))
        marker.set_color([1.0, 1.0, 0.0])  # bright yellow
        marker.set_radius(0.03)
        marker.set_point_render_mode("quad")

        # --- Load and ACCUMULATE LiDAR scan ---
        bin_path = os.path.join(LIDAR_DIR, f"{frame_id:06d}.bin")
        if os.path.exists(bin_path) and frame_id not in loaded_scans:
            pts_local = load_bin(bin_path)[::POINT_SUBSAMPLE]

            # Transform to aligned world frame
            R_world = s * R_align @ slam_rot[frame_id]
            t_world = slam_aligned[frame_id]
            pts_world = (R_world @ pts_local.T).T + t_world
            heights = pts_world[:, 2]

            all_pts_world.append(pts_world)
            all_pts_heights.append(heights)
            loaded_scans.add(frame_id)

            # Re-register the accumulated cloud
            if accum_cloud is not None:
                ps.remove_point_cloud("accumulated_map")

            combined_pts = np.vstack(all_pts_world)
            combined_h = np.concatenate(all_pts_heights)

            accum_cloud = ps.register_point_cloud("accumulated_map", combined_pts)
            accum_cloud.set_radius(POINT_RADIUS)
            accum_cloud.set_point_render_mode("quad")
            accum_cloud.set_transparency(POINT_ALPHA)
            accum_cloud.add_scalar_quantity("height", combined_h,
                                            enabled=True, cmap='viridis')

    # Initial frame
    update_frame(0)

    def callback():
        psim.TextUnformatted(f"Frame: {frame_ids[current_idx[0]]}/{n}  "
                             f"({current_idx[0]+1}/{total_playback_frames})  "
                             f"Scans loaded: {len(loaded_scans)}")

        changed, playing[0] = psim.Checkbox("Play", playing[0])

        if psim.Button("< Prev"):
            current_idx[0] = max(0, current_idx[0] - 1)
            update_frame(current_idx[0])
        psim.SameLine()
        if psim.Button("Next >"):
            current_idx[0] = min(total_playback_frames - 1, current_idx[0] + 1)
            update_frame(current_idx[0])
        psim.SameLine()
        if psim.Button("Reset"):
            current_idx[0] = 0
            all_pts_world.clear()
            all_pts_heights.clear()
            loaded_scans.clear()
            ps.remove_point_cloud("accumulated_map", error_if_absent=False)
            update_frame(0)

        _, FRAME_STEP_local = psim.InputInt("Frame step", FRAME_STEP)

        # Auto-play
        if playing[0]:
            now = time.time()
            if now - last_time[0] > 0.15:  # ~7 fps playback
                last_time[0] = now
                if current_idx[0] < total_playback_frames - 1:
                    current_idx[0] += 1
                    update_frame(current_idx[0])
                else:
                    playing[0] = False

    ps.set_user_callback(callback)
    print(f"\nReady: {total_playback_frames} frames, step={FRAME_STEP}")
    print("Controls:")
    print("  - Click 'Play' to auto-advance")
    print("  - Use Prev/Next for manual stepping")
    print("  - Record with OBS or ffmpeg")
    ps.show()

if __name__ == "__main__":
    main()
