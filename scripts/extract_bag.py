"""
extract_bag.py — Extract LiDAR frames from RELLIS-3D ROS1 bag.
Saves frames as KITTI-format .bin files (x, y, z, intensity float32).

Usage:
    python scripts/extract_bag.py <bag_path> <out_dir> [--every N] [--max M]

Example:
    python scripts/extract_bag.py data/RELLIS-3D/00000_00.bag data/extracted_frames --every 6 --max 100
"""

import argparse
import os
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

LIDAR_TOPIC = "/os1_cloud_node/points"


def extract_frames(bag_path, out_dir, every=1, max_frames=None):
    os.makedirs(out_dir, exist_ok=True)
    typestore = get_typestore(Stores.ROS1_NOETIC)

    with Reader(bag_path) as bag:
        connections = [c for c in bag.connections if c.topic == LIDAR_TOPIC]
        if not connections:
            print(f"Topic {LIDAR_TOPIC} not found in bag.")
            return 0

        saved = 0
        seen  = 0

        for conn, timestamp, data in bag.messages(connections=connections):
            seen += 1
            if seen % every != 0:
                continue

            msg = typestore.deserialize_ros1(data, conn.msgtype)

            # PointCloud2 layout: x@0, y@4, z@8, intensity@16 (float32), step=48
            raw   = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            step  = msg.point_step
            npts  = msg.width * msg.height
            # Vectorised extraction — reshape to (npts, step) then view as float32
            raw_2d = raw.reshape(npts, step)
            x = raw_2d[:,  0: 4].copy().view(np.float32).reshape(-1)
            y = raw_2d[:,  4: 8].copy().view(np.float32).reshape(-1)
            z = raw_2d[:,  8:12].copy().view(np.float32).reshape(-1)
            intensity = raw_2d[:, 16:20].copy().view(np.float32).reshape(-1)
            xyzI = np.stack([x, y, z, intensity], axis=1)

            # Remove NaN / zero-range points
            valid = np.isfinite(xyzI).all(axis=1) & (np.linalg.norm(xyzI[:, :3], axis=1) > 0.1)
            xyzI  = xyzI[valid]

            out_path = os.path.join(out_dir, f"{saved:06d}.bin")
            xyzI.tofile(out_path)
            saved += 1
            print(f"[{saved}] {out_path}  ({len(xyzI)} points)")

            if max_frames and saved >= max_frames:
                break

    print(f"\nDone — {saved} frames saved to {out_dir}/")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract LiDAR frames from ROS1 bag")
    parser.add_argument("bag_path", help="Path to .bag file")
    parser.add_argument("out_dir",  help="Output directory for .bin frames")
    parser.add_argument("--every",  type=int, default=1,
                        help="Save every Nth frame (default: 1 = all)")
    parser.add_argument("--max",    type=int, default=None,
                        help="Maximum number of frames to save")
    args = parser.parse_args()

    extract_frames(args.bag_path, args.out_dir, every=args.every, max_frames=args.max)
