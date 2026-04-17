#!/usr/bin/env python3
"""
render_camera_feed.py — Render RGB camera images with projected LiDAR points.
Creates a video of the camera feed with depth-colored LiDAR overlay.

This is the LEFT PANEL of the Forster-style split view.
The RIGHT PANEL comes from recording the Polyscope playback.
Composite them with:
  ffmpeg -i results/camera_feed.mp4 -i results/polyscope_recording.mp4 \
         -filter_complex "[0:v]scale=640:480[left];[1:v]scale=960:480[right];[left][right]hstack" \
         results/slam_demo.mp4

Output: results/camera_feed.mp4
"""

import numpy as np
import cv2
import glob
import os

# Paths
IMAGE_DIR = "data/RELLIS-3D/Rellis_3D_pylon_camera_node/Rellis-3D/00000/pylon_camera_node"
LIDAR_DIR = "data/RELLIS-3D/Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/00000/os1_cloud_node_kitti_bin"

# Camera intrinsics (from RELLIS-3D calibration)
# If you have exact values from data/RELLIS-3D/Rellis_3D_cam_intrinsic/, use those
# These are approximate for the Basler camera on the Warthog
FX, FY = 905.0, 905.0
CX, CY = 960.0, 600.0
IMG_W, IMG_H = 1920, 1200

# LiDAR-to-camera extrinsic (approximate — check your calibration)
# From Phase 1 cam_lidar_projection work
# If you have the exact T_cam_lidar, replace this
T_cam_lidar = np.eye(4)  # PLACEHOLDER — load from calibration

# Output settings
FRAME_STEP = 2        # process every Nth frame
OUTPUT_FPS = 15
OUTPUT_SIZE = (960, 600)  # half resolution for video

os.makedirs("results", exist_ok=True)

def load_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]

def project_lidar_to_image(pts_lidar, T_cam_lidar, fx, fy, cx, cy, w, h):
    """Project 3D LiDAR points to 2D image coordinates."""
    # Transform to camera frame
    pts_hom = np.hstack([pts_lidar, np.ones((len(pts_lidar), 1))])
    pts_cam = (T_cam_lidar @ pts_hom.T).T[:, :3]

    # Filter points behind camera
    mask = pts_cam[:, 2] > 0.5  # at least 0.5m in front
    pts_cam = pts_cam[mask]

    if len(pts_cam) == 0:
        return np.array([]), np.array([]), np.array([])

    # Project to image
    u = (fx * pts_cam[:, 0] / pts_cam[:, 2] + cx).astype(int)
    v = (fy * pts_cam[:, 1] / pts_cam[:, 2] + cy).astype(int)
    depths = pts_cam[:, 2]

    # Filter to image bounds
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u[valid], v[valid], depths[valid]

def depth_to_color(depths, min_d=1.0, max_d=40.0):
    """Convert depth values to BGR colors (jet colormap)."""
    normalized = np.clip((depths - min_d) / (max_d - min_d), 0, 1)
    colors = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return colors.reshape(-1, 3)

def get_sorted_images():
    """Get all image paths sorted by frame number."""
    pattern = os.path.join(IMAGE_DIR, "frame*.jpg")
    paths = sorted(glob.glob(pattern))
    return paths

def try_load_calibration():
    """Try to load camera-lidar calibration from RELLIS-3D data."""
    global T_cam_lidar

    # Check for calibration files
    calib_dir = "data/RELLIS-3D/Rellis_3D_cam2lidar_20210224"
    if os.path.exists(calib_dir):
        calib_files = glob.glob(os.path.join(calib_dir, "**/*.yaml"), recursive=True)
        if calib_files:
            print(f"  Found calibration: {calib_files[0]}")
            # Would need yaml parsing — use approximate for now

    # Approximate extrinsic for RELLIS-3D Warthog setup
    # LiDAR (Ouster OS1) is mounted on top, camera looks forward
    # This is an approximation — replace with actual calibration
    T_cam_lidar = np.array([
        [ 0.0, -1.0,  0.0,  0.0],
        [ 0.0,  0.0, -1.0, -0.1],
        [ 1.0,  0.0,  0.0, -0.3],
        [ 0.0,  0.0,  0.0,  1.0]
    ])
    print("  Using approximate LiDAR-camera extrinsic")

def main():
    print("Rendering camera feed with LiDAR overlay...")

    try_load_calibration()
    image_paths = get_sorted_images()
    print(f"  Found {len(image_paths)} images")

    if not image_paths:
        print("ERROR: No images found!")
        return

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('results/camera_feed.mp4', fourcc, OUTPUT_FPS, OUTPUT_SIZE)

    total_frames = len(image_paths) // FRAME_STEP
    print(f"  Rendering {total_frames} frames (step={FRAME_STEP})...")

    for idx, img_path in enumerate(image_paths):
        if idx % FRAME_STEP != 0:
            continue

        frame_id = idx

        # Load RGB image
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Load and project LiDAR points
        bin_path = os.path.join(LIDAR_DIR, f"{frame_id:06d}.bin")
        if os.path.exists(bin_path):
            pts = load_bin(bin_path)
            pts_sub = pts[::3]  # subsample for speed

            u, v, depths = project_lidar_to_image(
                pts_sub, T_cam_lidar, FX, FY, CX, CY, IMG_W, IMG_H)

            if len(u) > 0:
                colors = depth_to_color(depths)
                for i in range(len(u)):
                    cv2.circle(img, (u[i], v[i]), 2, colors[i].tolist(), -1)

        # Add frame info overlay
        cv2.putText(img, f"Frame {frame_id}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Resize and write
        img_resized = cv2.resize(img, OUTPUT_SIZE)
        out.write(img_resized)

        if (idx // FRAME_STEP) % 50 == 0:
            print(f"    {idx // FRAME_STEP}/{total_frames}")

    out.release()
    print(f"Saved results/camera_feed.mp4 ({total_frames} frames at {OUTPUT_FPS}fps)")
    print("\nTo composite with Polyscope recording:")
    print("  ffmpeg -i results/camera_feed.mp4 -i results/polyscope_recording.mp4 \\")
    print("         -filter_complex \"[0:v]scale=640:480[left];[1:v]scale=960:480[right];[left][right]hstack\" \\")
    print("         results/slam_demo.mp4")

if __name__ == "__main__":
    main()
