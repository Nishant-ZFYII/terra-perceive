"""
visualize_pointcloud.py — Open3D point cloud viewer for RELLIS-3D .bin files.
P1-M1.3: Verify data loading is correct.

Usage:
    python scripts/visualize_pointcloud.py data/RELLIS-3D/path/to/frame.bin

TASK:
  1. Load .bin file: packed float32 [x, y, z, intensity] per point
  2. Extract xyz (first 3 floats per point, skip intensity)
  3. Color by height (z-value) using a colormap
  4. Render in Open3D window
"""

import numpy as np
import argparse

parser = argparse.ArgumentParser(prog = "visualize_pointcloud",
                                 description="Visualize RELLIS-3D .bin point cloud",
                                 epilog="Example: python visualize_pointcloud.py data/RELLIS-3D/path/to/frame.bin")
parser.add_argument("bin_path", type=str, help="Path to .bin point cloud file")
args = parser.parse_args()

def load_bin_pointcloud(bin_path: str) -> np.ndarray:
    """Load RELLIS-3D .bin point cloud. Returns Nx3 float32 array."""
    #return only the x,y,z and remove the intensity. 
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3] 

def visualize(points: np.ndarray) -> None:
    """Render point cloud in Open3D, colored by z-value."""
    import open3d as o3d
    import matplotlib.pyplot as plt
    pcd = o3d.geometry.PointCloud()
    #Convert float64 numpy array of shape (n, 3) to Open3D format.
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    #Color by height (z-value) using a colormap
    z = points[:, 2]

    #Normalize z values to [0,1] for the colormap, to prevent division by zero add a small epsilon to the denominator.
    eps = 1e-8
    z_normalized = (z - z.min()) / (z.max() - z.min() + eps)

    colors = plt.cm.viridis(z_normalized)[:, :3]  # Get RGB colors from the colormap
    pcd.colors = o3d.utility.Vector3dVector(colors) 
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    pts = load_bin_pointcloud(args.bin_path)
    if pts is not None:
        print(f"Loaded {len(pts)} points")
        print(f"X: [{pts[:, 0].min():.1f}, {pts[:, 0].max():.1f}]")
        print(f"Y: [{pts[:, 1].min():.1f}, {pts[:, 1].max():.1f}]")
        print(f"Z: [{pts[:, 2].min():.1f}, {pts[:, 2].max():.1f}]")
        visualize(pts)
