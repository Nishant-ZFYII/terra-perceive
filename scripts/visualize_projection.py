"""
visualize_projection.py — LiDAR-on-camera overlay for visual verification.
P1-M4.3: Verify camera-LiDAR projection is correct.

Usage:
    python scripts/visualize_projection.py <image.png> <pointcloud.bin> <calib.yaml>

YOUR TASK:
  1. Load camera image
  2. Load point cloud
  3. Load calibration (T_cam_lidar, K)
  4. Project each LiDAR point to pixel coordinates
  5. Draw colored dots on image (color by depth)
  6. Save overlay image as PNG
  7. Points should land on correct objects (trees on trees, ground on ground)
"""
