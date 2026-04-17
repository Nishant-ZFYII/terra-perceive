#!/usr/bin/env python3
"""
animate_splitview.py — Split-view SLAM demo video.

Layout:
  ┌──────────────┬─────────────────────┐
  │  RGB Camera  │                     │
  │   (top-left) │   Accumulated Map   │
  ├──────────────┤   + Trajectory      │
  │  LiDAR BEV   │   (right panel)     │
  │ (bottom-left)│                     │
  └──────────────┴─────────────────────┘

Output: results/slam_splitview.mp4 (or .gif)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from PIL import Image
import os
import glob

os.makedirs("results", exist_ok=True)

# Paths
SLAM_PATH = "data/poses_slam_manifold.csv"
CARTO_PATH = "data/poses_carto.csv"
ICP_PATH = "data/poses_icp.csv"
LIDAR_DIR = "data/RELLIS-3D/Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/00000/os1_cloud_node_kitti_bin"
IMAGE_DIR = "data/RELLIS-3D/Rellis_3D_pylon_camera_node/Rellis-3D/00000/pylon_camera_node"

# Settings
FRAME_STEP = 20       # show every Nth frame (20 = every 2 seconds)
OUTPUT_FORMAT = "mp4"  # "mp4" or "gif"

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
    return (s * (R @ src.T).T + t)

def load_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]

def get_image_path(frame_id):
    """Find the RGB image for a given frame ID."""
    pattern = os.path.join(IMAGE_DIR, f"frame{frame_id:06d}-*.jpg")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None

print("Loading trajectories...")
carto = np.loadtxt(CARTO_PATH, delimiter=',', skiprows=1, usecols=[1, 2, 3])
slam = np.loadtxt(SLAM_PATH, delimiter=',', skiprows=1, usecols=[1, 2, 3])
icp = np.loadtxt(ICP_PATH, delimiter=',', skiprows=1, usecols=[1, 2, 3])

n = min(len(carto), len(slam), len(icp))
slam_a = umeyama_align(slam[:n], carto[:n])
icp_a = umeyama_align(icp[:n], carto[:n])

frame_ids = list(range(0, n, FRAME_STEP))
total_frames = len(frame_ids)

# Precompute trajectory bounds for consistent axis limits
margin = 20
x_min, x_max = carto[:n, 0].min() - margin, carto[:n, 0].max() + margin
y_min, y_max = carto[:n, 1].min() - margin, carto[:n, 1].max() + margin

print(f"Rendering {total_frames} frames (step={FRAME_STEP})...")

# Create figure with GridSpec layout
fig = plt.figure(figsize=(18, 9))
gs = GridSpec(2, 2, width_ratios=[1.2, 1.5], hspace=0.08, wspace=0.08)

ax_rgb = fig.add_subplot(gs[0, 0])       # top-left: RGB
ax_lidar = fig.add_subplot(gs[1, 0])     # bottom-left: LiDAR BEV
ax_map = fig.add_subplot(gs[:, 1])       # right: accumulated map

# Static elements on map
ax_map.plot(carto[:n, 0], carto[:n, 1], 'k--', linewidth=0.8, alpha=0.3,
            label='Cartographer')
ax_map.set_xlim(x_min, x_max)
ax_map.set_ylim(y_min, y_max)
ax_map.set_aspect('equal')
ax_map.grid(True, alpha=0.15)
ax_map.set_xlabel('x (m)')
ax_map.set_ylabel('y (m)')

# Dynamic elements
slam_line, = ax_map.plot([], [], 'g-', linewidth=2, label='Our SLAM')
icp_line, = ax_map.plot([], [], 'b-', linewidth=0.8, alpha=0.5, label='KISS-ICP')
pos_marker, = ax_map.plot([], [], 'yo', markersize=10, zorder=5,
                           markeredgecolor='black', markeredgewidth=1)
scan_scatter = ax_map.scatter([], [], s=0.1, c=[], cmap='viridis', alpha=0.4)
ax_map.legend(loc='upper left', fontsize=9)

# Initialize image panels
rgb_img = ax_rgb.imshow(np.zeros((100, 100, 3), dtype=np.uint8))
ax_rgb.set_title('RGB Camera', fontsize=11, fontweight='bold')
ax_rgb.axis('off')

lidar_scatter = ax_lidar.scatter([], [], s=0.3, c='cyan', alpha=0.6)
ax_lidar.set_xlim(-40, 40)
ax_lidar.set_ylim(-40, 40)
ax_lidar.set_aspect('equal')
ax_lidar.set_facecolor('black')
ax_lidar.set_title('LiDAR BEV (sensor frame)', fontsize=11, fontweight='bold')
ax_lidar.tick_params(colors='gray', labelsize=7)

fig.suptitle('Terra Perceive — LiDAR-Inertial SLAM', fontsize=14, fontweight='bold')

# Accumulated map points for the right panel
accumulated_pts = []

def animate(frame_num):
    fid = frame_ids[frame_num]

    # --- Top-left: RGB camera ---
    img_path = get_image_path(fid)
    if img_path and os.path.exists(img_path):
        img = np.array(Image.open(img_path))
        rgb_img.set_data(img)
    ax_rgb.set_title(f'RGB Camera — Frame {fid}', fontsize=11, fontweight='bold')

    # --- Bottom-left: LiDAR BEV (sensor frame) ---
    bin_path = os.path.join(LIDAR_DIR, f"{fid:06d}.bin")
    if os.path.exists(bin_path):
        pts = load_bin(bin_path)
        pts_sub = pts[::3]  # subsample
        ax_lidar.clear()
        # BEV: x=forward (up on screen), y=left (right on screen)
        ax_lidar.scatter(pts_sub[:, 0], pts_sub[:, 1], s=0.3, c=pts_sub[:, 2],
                         cmap='plasma', alpha=0.7, vmin=-2, vmax=5)
        ax_lidar.set_xlim(-60, 60)
        ax_lidar.set_ylim(-60, 60)
        ax_lidar.set_aspect('equal')
        ax_lidar.set_facecolor('black')
        ax_lidar.set_title(f'LiDAR BEV — Frame {fid}', fontsize=11,
                            fontweight='bold', color='cyan')
        ax_lidar.tick_params(colors='gray', labelsize=7)
        # Draw sensor origin (center)
        ax_lidar.plot(0, 0, 'r+', markersize=15, markeredgewidth=2)
        ax_lidar.plot(0, 0, 'ro', markersize=4)

    # --- Right: accumulated trajectory + map ---
    slam_line.set_data(slam_a[:fid, 0], slam_a[:fid, 1])
    icp_line.set_data(icp_a[:fid, 0], icp_a[:fid, 1])
    pos_marker.set_data([slam_a[fid, 0]], [slam_a[fid, 1]])

    ax_map.set_title(f'Accumulated Map — Frame {fid}/{n}', fontsize=12)

    if frame_num % 10 == 0:
        print(f"  Frame {frame_num}/{total_frames} (id={fid})")

    return rgb_img, slam_line, icp_line, pos_marker

print("Creating animation...")
anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                interval=100, blit=False)

if OUTPUT_FORMAT == "mp4":
    try:
        anim.save('results/slam_splitview.mp4', writer='ffmpeg', fps=10, dpi=120)
        print('Saved results/slam_splitview.mp4')
    except Exception as e:
        print(f"ffmpeg failed ({e}), falling back to GIF...")
        anim.save('results/slam_splitview.gif', writer='pillow', fps=10, dpi=100)
        print('Saved results/slam_splitview.gif')
else:
    anim.save('results/slam_splitview.gif', writer='pillow', fps=10, dpi=100)
    print('Saved results/slam_splitview.gif')

plt.close()
