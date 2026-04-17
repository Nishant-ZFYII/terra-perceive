#!/usr/bin/env python3
"""
animate_forster_style.py — Forster-style SLAM visualization.

Layout:
  ┌──────────────┬──────────────────────────┐
  │              │                          │
  │  RGB Camera  │  3D Point Cloud Map      │
  │              │  + Blue trajectory line  │
  │              │  + Red keyframe markers  │
  │              │  + Green LiDAR points    │
  │              │  (accumulating)          │
  ├──────────────┴──────────────────────────┤
  │  ●──────────── progress ──────────────  │
  │  LiDAR-Inertial SLAM (Manifold)        │
  └─────────────────────────────────────────┘

Output: results/slam_forster_style.mp4
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import glob
import os

os.makedirs("results", exist_ok=True)

# Paths
SLAM_PATH = "data/poses_slam_manifold.csv"
CARTO_PATH = "data/poses_carto.csv"
LIDAR_DIR = "data/RELLIS-3D/Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/00000/os1_cloud_node_kitti_bin"
IMAGE_DIR = "data/RELLIS-3D/Rellis_3D_pylon_camera_node/Rellis-3D/00000/pylon_camera_node"

# Settings
FRAME_STEP = 20         # every Nth frame
POINT_SUBSAMPLE = 30    # keep every Nth LiDAR point (aggressive for 3D render speed)
MAX_ACCUMULATED = 80000 # cap total accumulated points for render speed

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
    quats = data[:, 4:8]
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

def get_image_path(frame_id):
    pattern = os.path.join(IMAGE_DIR, f"frame{frame_id:06d}-*.jpg")
    matches = glob.glob(pattern)
    return matches[0] if matches else None

# Load data
print("Loading trajectories...")
carto = np.loadtxt(CARTO_PATH, delimiter=',', skiprows=1, usecols=[1, 2, 3])
icp_raw = np.loadtxt("data/poses_icp.csv", delimiter=',', skiprows=1, usecols=[1, 2, 3])
gps_raw = np.loadtxt("data/poses_gps.csv", delimiter=',', skiprows=1, usecols=[1, 2, 3])
slam_pos, slam_rot = load_poses_with_rotations(SLAM_PATH)

n = min(len(carto), len(slam_pos), len(icp_raw), len(gps_raw))
s_align, R_align, t_align = umeyama_align(slam_pos[:n], carto[:n])
slam_aligned = (s_align * (R_align @ slam_pos[:n].T).T + t_align)

# Align ICP and GPS to same frame
s_i, R_i, t_i = umeyama_align(icp_raw[:n], carto[:n])
icp_a = (s_i * (R_i @ icp_raw[:n].T).T + t_i)

s_g, R_g, t_g = umeyama_align(gps_raw[:n], carto[:n])
gps_aligned = (s_g * (R_g @ gps_raw[:n].T).T + t_g)

frame_ids = list(range(0, n, FRAME_STEP))
total_frames = len(frame_ids)

# Determine 3D view limits from full trajectory
pad = 20
x_center = (slam_aligned[:, 0].min() + slam_aligned[:, 0].max()) / 2
y_center = (slam_aligned[:, 1].min() + slam_aligned[:, 1].max()) / 2
z_center = (slam_aligned[:, 2].min() + slam_aligned[:, 2].max()) / 2
span = max(np.ptp(slam_aligned[:, 0]), np.ptp(slam_aligned[:, 1]), np.ptp(slam_aligned[:, 2])) / 2 + pad 

# Accumulated point cloud data
accum_pts = []
accum_colors = []

print(f"Rendering {total_frames} frames...")

# Create figure
fig = plt.figure(figsize=(16, 9), facecolor='white')
gs = GridSpec(2, 2, width_ratios=[0.8, 1.4], height_ratios=[1, 0.06],
             hspace=0.02, wspace=0.02)

ax_rgb = fig.add_subplot(gs[0, 0])
ax_3d = fig.add_subplot(gs[0, 1], projection='3d')
ax_progress = fig.add_subplot(gs[1, :])

# Style the 3D axes
ax_3d.set_facecolor('white')
ax_3d.xaxis.pane.fill = False
ax_3d.yaxis.pane.fill = False
ax_3d.zaxis.pane.fill = False
ax_3d.xaxis.pane.set_edgecolor('lightgray')
ax_3d.yaxis.pane.set_edgecolor('lightgray')
ax_3d.zaxis.pane.set_edgecolor('lightgray')
ax_3d.grid(True, alpha=0.15)
ax_3d.set_xlabel('x (m)', fontsize=8, labelpad=-2)
ax_3d.set_ylabel('y (m)', fontsize=8, labelpad=-2)
ax_3d.set_zlabel('z (m)', fontsize=8, labelpad=-2)
ax_3d.tick_params(labelsize=6)

# Set fixed camera angle (slightly elevated, like looking down stairs)
ax_3d.view_init(elev=55, azim=-60)
ax_3d.set_xlim(x_center - span, x_center + span)
ax_3d.set_ylim(y_center - span, y_center + span)
ax_3d.set_zlim(z_center - span/3, z_center + span/3)

# RGB panel setup
ax_rgb.axis('off')
rgb_img_display = ax_rgb.imshow(np.zeros((600, 960, 3), dtype=np.uint8))

# Progress bar setup
ax_progress.set_xlim(0, 1)
ax_progress.set_ylim(0, 1)
ax_progress.axis('off')
progress_line, = ax_progress.plot([0, 0], [0.5, 0.5], 'r-', linewidth=3)
progress_dot, = ax_progress.plot([0], [0.5], 'ro', markersize=8)
ax_progress.plot([0, 1], [0.5, 0.5], 'lightgray', linewidth=2, zorder=0)
progress_text = ax_progress.text(0.02, 0.1, '', fontsize=10, fontweight='bold',
                                  color='darkred')
ax_progress.text(0.5, -0.5, 'Terra Perceive — LiDAR-Inertial SLAM (On-Manifold)',
                 fontsize=11, ha='center', fontweight='bold', color='#333333',
                 transform=ax_progress.transAxes)

def animate(frame_num):
    fid = frame_ids[frame_num]
    progress = frame_num / max(total_frames - 1, 1)

    # --- RGB camera (left panel) ---
    img_path = get_image_path(fid)
    if img_path and os.path.exists(img_path):
        img = np.array(Image.open(img_path).resize((960, 600)))
        rgb_img_display.set_data(img)
    ax_rgb.set_title(f'RGB Camera — Frame {fid}', fontsize=11,
                      fontweight='bold', pad=5)

    # --- 3D accumulated map (right panel) ---
    ax_3d.clear()

    # Styling (must re-apply after clear)
    ax_3d.set_facecolor('white')
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.grid(True, alpha=0.15)
    ax_3d.tick_params(labelsize=6)
    ax_3d.view_init(elev=55, azim=-60 + frame_num * 0.3)  # slow rotation
    ax_3d.set_xlim(x_center - span, x_center + span)
    ax_3d.set_ylim(y_center - span, y_center + span)
    ax_3d.set_zlim(z_center - span/3, z_center + span/3)

    # Load and accumulate LiDAR scan
    bin_path = os.path.join(LIDAR_DIR, f"{fid:06d}.bin")
    if os.path.exists(bin_path):
        pts_local = load_bin(bin_path)[::POINT_SUBSAMPLE]
        R_world = s_align * R_align @ slam_rot[fid]
        t_world = slam_aligned[fid]
        pts_world = (R_world @ pts_local.T).T + t_world
        accum_pts.append(pts_world)

    # Plot accumulated 3D points (HEIGHT-COLORED, light/transparent)
    if accum_pts:
        all_pts = np.vstack(accum_pts)
        if len(all_pts) > MAX_ACCUMULATED:
            idx = np.random.choice(len(all_pts), MAX_ACCUMULATED, replace=False)
            all_pts = all_pts[idx]

        ax_3d.scatter(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2],
                       c=all_pts[:, 2], cmap='YlGn', s=0.15, alpha=0.2,
                       edgecolors='none', vmin=-3, vmax=8)

    # Plot GPS measurements (RED dashed, shows raw noisy GPS path)
    gps_end = min(fid + 1, len(gps_aligned))
    if gps_end > 1:
        ax_3d.plot(gps_aligned[:gps_end, 0], gps_aligned[:gps_end, 1],
                    gps_aligned[:gps_end, 2],
                    color='#e74c3c', linewidth=0.8, alpha=0.5, linestyle='--',
                    label='GPS (raw)')

    # Plot ICP trajectory (faded blue, the "before" optimization)
    icp_end = min(fid + 1, len(icp_a))
    if icp_end > 1:
        ax_3d.plot(icp_a[:icp_end, 0], icp_a[:icp_end, 1], icp_a[:icp_end, 2],
                    color='#3498db', linewidth=1.0, alpha=0.4,
                    label='ICP (before)')

    # Plot SLAM trajectory (dark blue, the "after" optimization)
    traj = slam_aligned[:fid+1]
    if len(traj) > 1:
        ax_3d.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color='#1a237e', linewidth=2.0, alpha=0.95,
                    label='SLAM (optimized)')

    # Plot keyframes (red squares = keyframe markers along trajectory)
    kf_step = 100
    kf_indices = list(range(0, fid, kf_step))
    if kf_indices:
        kf_pts = slam_aligned[kf_indices]
        ax_3d.scatter(kf_pts[:, 0], kf_pts[:, 1], kf_pts[:, 2],
                       c='#e74c3c', s=20, marker='s', alpha=0.8,
                       edgecolors='darkred', linewidths=0.3,
                       label='Keyframes')

    # Current position (yellow dot)
    ax_3d.scatter([slam_aligned[fid, 0]], [slam_aligned[fid, 1]],
                   [slam_aligned[fid, 2]],
                   c='yellow', s=80, marker='o', edgecolors='black',
                   linewidths=1.5, zorder=10, label='Current')

    # Legend (only on first few frames to avoid flicker)
    if frame_num < 3:
        ax_3d.legend(loc='upper right', fontsize=7, framealpha=0.8,
                      markerscale=3, handlelength=1.5)

    ax_3d.set_title(f'3D Map — {len(accum_pts)} scans | '
                     f'GPS pulls → optimizer corrects',
                     fontsize=10, fontweight='bold', pad=5)

    # --- Progress bar ---
    progress_line.set_data([0, progress], [0.5, 0.5])
    progress_dot.set_data([progress], [0.5])
    progress_text.set_text(f'Frame {fid}/{n}')

    if frame_num % 5 == 0:
        print(f"  {frame_num}/{total_frames} (frame {fid})")

    return rgb_img_display, progress_line, progress_dot, progress_text

print("Creating animation...")
anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                interval=150, blit=False)

try:
    anim.save('results/slam_forster_style.mp4', writer='ffmpeg', fps=8, dpi=120)
    print('Saved results/slam_forster_style.mp4')
except Exception as e:
    print(f"ffmpeg failed ({e}), trying GIF...")
    anim.save('results/slam_forster_style.gif', writer='pillow', fps=6, dpi=90)
    print('Saved results/slam_forster_style.gif')

plt.close()
print("Done!")
