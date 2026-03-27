"""
visualize_projection.py — LiDAR-on-camera overlay for visual verification.
P1-M4.3: Verify camera-LiDAR projection is correct.

Usage:
    python scripts/visualize_projection.py
    python scripts/visualize_projection.py <image.png> <pointcloud.bin> <calib.yaml>

Default paths use the bundled RELLIS-3D example frame (000104).
Output: docs/assets/projection_overlay.png
"""

import sys
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

# ---------------------------------------------------------------------------
# Defaults — RELLIS-3D example frame
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_IMAGE  = os.path.join(REPO_ROOT, "third_party/RELLIS-3D/utils/example/frame000104-1581624663_149.jpg")
DEFAULT_BIN    = os.path.join(REPO_ROOT, "third_party/RELLIS-3D/utils/example/000104.bin")
DEFAULT_CALIB  = os.path.join(REPO_ROOT, "config/camera_lidar_calib.yaml")
OUTPUT_PATH    = os.path.join(REPO_ROOT, "docs/assets/projection_overlay.png")

# ---------------------------------------------------------------------------
# Load calibration
# ---------------------------------------------------------------------------
def load_calib(calib_path: str):
    with open(calib_path, "r") as f:
        cfg = yaml.safe_load(f)

    intr = cfg["camera_intrinsics"]
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]

    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)

    rows = cfg["extrinsic_T_cam_lidar"]
    T_cam_lidar = np.array(rows, dtype=np.float64)   # 4×4

    width  = intr["width"]
    height = intr["height"]

    return K, T_cam_lidar, width, height

# ---------------------------------------------------------------------------
# Load point cloud  (.bin — KITTI format: x, y, z, intensity as float32)
# ---------------------------------------------------------------------------
def load_bin(bin_path: str) -> np.ndarray:
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # drop zero-origin points (padding in Ouster binary format)
    valid = ~((pts[:, 0] == 0) & (pts[:, 1] == 0) & (pts[:, 2] == 0))
    return pts[valid, :3]   # (N, 3) — x, y, z only

# ---------------------------------------------------------------------------
# Project LiDAR points → pixel coordinates
# Mirrors the C++ project_point() logic in src/cam_lidar_projection.cpp
# ---------------------------------------------------------------------------
def project_points(pts_lidar: np.ndarray, T_cam_lidar: np.ndarray,
                   K: np.ndarray, width: int, height: int):
    """
    Returns:
        pixels : (M, 2) float array of (u, v) for visible points
        depths : (M,)   float array of camera-frame Z values
    """
    N = len(pts_lidar)

    # Step 1: homogeneous transform  P_cam = T_cam_lidar * [x, y, z, 1]^T
    ones = np.ones((N, 1), dtype=np.float64)
    pts_h = np.hstack([pts_lidar.astype(np.float64), ones])   # (N, 4)
    P_cam = (T_cam_lidar @ pts_h.T).T                          # (N, 4)

    # Step 2: depth check — keep only points in front of camera
    depths = P_cam[:, 2]
    front  = depths > 0
    P_cam  = P_cam[front]
    depths = depths[front]

    # Step 3: project  p = K * P_cam[:3],  then perspective divide
    p = (K @ P_cam[:, :3].T).T   # (M, 3)
    u = p[:, 0] / p[:, 2]
    v = p[:, 1] / p[:, 2]

    # Step 4: bounds check
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u      = u[in_bounds]
    v      = v[in_bounds]
    depths = depths[in_bounds]

    pixels = np.stack([u, v], axis=1)
    return pixels, depths

# ---------------------------------------------------------------------------
# Draw overlay
# ---------------------------------------------------------------------------
def draw_overlay(image: np.ndarray, pixels: np.ndarray, depths: np.ndarray,
                 dot_radius: int = 2) -> np.ndarray:
    """
    Draws depth-coloured dots on a copy of image.
    Near = warm (yellow/red), far = cool (blue/purple).
    """
    overlay = image.copy()

    # normalise depth to [0, 1] for colormap
    d_min, d_max = np.percentile(depths, 2), np.percentile(depths, 98)
    d_norm = np.clip((depths - d_min) / (d_max - d_min + 1e-6), 0, 1)

    colormap = cm.get_cmap("plasma")
    colors = (colormap(d_norm)[:, :3] * 255).astype(np.uint8)   # (M, 3) RGB

    us = pixels[:, 0].astype(int)
    vs = pixels[:, 1].astype(int)

    h, w = overlay.shape[:2]
    for u, v, color in zip(us, vs, colors):
        # draw a small filled square (avoids cv2 dependency)
        u0, u1 = max(0, u - dot_radius), min(w, u + dot_radius + 1)
        v0, v1 = max(0, v - dot_radius), min(h, v + dot_radius + 1)
        overlay[v0:v1, u0:u1] = color

    return overlay

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    bin_path   = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BIN
    calib_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_CALIB

    print(f"Image  : {image_path}")
    print(f"LiDAR  : {bin_path}")
    print(f"Calib  : {calib_path}")

    # load
    image       = np.array(Image.open(image_path).convert("RGB"))
    pts_lidar   = load_bin(bin_path)
    K, T_cam_lidar, width, height = load_calib(calib_path)

    print(f"Image size    : {image.shape[1]}×{image.shape[0]}")
    print(f"LiDAR points  : {len(pts_lidar):,}")

    # project
    pixels, depths = project_points(pts_lidar, T_cam_lidar, K, width, height)
    print(f"Projected (visible): {len(pixels):,} / {len(pts_lidar):,} points")
    print(f"Depth range : {depths.min():.2f}m – {depths.max():.2f}m")

    # draw
    overlay = draw_overlay(image, pixels, depths, dot_radius=2)

    # save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    Image.fromarray(overlay).save(OUTPUT_PATH)
    print(f"\nSaved → {OUTPUT_PATH}")

    # also show a quick matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"LiDAR overlay — {len(pixels):,} projected points")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH.replace(".png", "_comparison.png"), dpi=150)
    print(f"Saved → {OUTPUT_PATH.replace('.png', '_comparison.png')}")
    plt.show()


if __name__ == "__main__":
    main()
