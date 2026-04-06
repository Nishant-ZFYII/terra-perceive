"""Generate camera + BEV accumulation GIF for blog post."""
import numpy as np
import os
from PIL import Image
import imageio.v2 as imageio

# Avoid Matplotlib cache warnings on environments where ~/.config is not writable.
os.environ.setdefault("MPLCONFIGDIR", os.path.join("output", ".mplconfig"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

poses = np.load("results/2026-03-31_13-26-44/00000_poses.npy")
BIN_DIR = "data/RELLIS-3D/Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/00000/os1_cloud_node_kitti_bin"
IMG_DIR = "data/RELLIS-3D/Rellis_3D_pylon_camera_node/Rellis-3D/00000/pylon_camera_node"
bins = sorted(os.listdir(BIN_DIR))
imgs = sorted(os.listdir(IMG_DIR))

os.makedirs('output', exist_ok=True)
accumulated = np.empty((0, 3), dtype=np.float32)
traj_so_far = []

# Memory safety controls:
# - Sample each transformed cloud before accumulating.
# - Cap global accumulated points to avoid OOM on long sequences.
FRAME_STEP = 10
MAX_POINTS_PER_FRAME = 20000
MAX_ACCUMULATED_POINTS = 400000
rng = np.random.default_rng(42)

n_frames = min(len(poses), len(imgs), len(bins))
gif_path = 'output/kiss_icp_bev_accumulation.gif'
written_frames = 0

with imageio.get_writer(gif_path, mode='I', duration=0.2, loop=0) as writer:
    for frame_idx in range(0, n_frames, FRAME_STEP):
        path = os.path.join(BIN_DIR, bins[frame_idx])
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]

        T = poses[frame_idx]
        pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
        pts_world = (T @ pts_h.T).T[:, :3].astype(np.float32, copy=False)

        if len(pts_world) > MAX_POINTS_PER_FRAME:
            keep_idx = rng.choice(len(pts_world), size=MAX_POINTS_PER_FRAME, replace=False)
            pts_world = pts_world[keep_idx]

        if len(accumulated) == 0:
            accumulated = pts_world
        else:
            accumulated = np.vstack([accumulated, pts_world])

        if len(accumulated) > MAX_ACCUMULATED_POINTS:
            keep_idx = rng.choice(len(accumulated), size=MAX_ACCUMULATED_POINTS, replace=False)
            accumulated = accumulated[keep_idx]

        traj_so_far.append(poses[frame_idx, :3, 3])

        all_pts = accumulated
        traj_arr = np.array(traj_so_far)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=80)
        z_min, z_max = -2, 4
        colors = np.clip((all_pts[:, 2] - z_min) / (z_max - z_min + 1e-6), 0, 1)
        ax.scatter(all_pts[:, 0], all_pts[:, 1], c=plt.cm.viridis(colors), s=0.02, alpha=0.3)
        ax.plot(traj_arr[:, 0], traj_arr[:, 1], 'r-', linewidth=1.5)
        ax.scatter(traj_arr[0, 0], traj_arr[0, 1], c='lime', s=80, zorder=5, edgecolors='black')
        ax.scatter(traj_arr[-1, 0], traj_arr[-1, 1], c='red', s=80, zorder=5, marker='o')
        ax.set_xlim(-170, 20)
        ax.set_ylim(-20, 240)
        ax.set_aspect('equal')
        ax.set_title(f'BEV Map — Frame {frame_idx}', fontsize=10)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.canvas.draw()
        bev_img = Image.frombytes('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert('RGB')
        plt.close(fig)

        cam_img = Image.open(os.path.join(IMG_DIR, imgs[frame_idx]))
        cam_img = cam_img.resize((480, 300))
        bev_img = bev_img.resize((400, 300))

        combined = Image.new('RGB', (880, 300), (255, 255, 255))
        combined.paste(cam_img, (0, 0))
        combined.paste(bev_img, (480, 0))
        writer.append_data(np.asarray(combined))
        written_frames += 1

        if frame_idx % FRAME_STEP == 0:
            print(f"Frame {frame_idx}/{n_frames}")

print(f"Saved: {gif_path} ({written_frames} frames)")
