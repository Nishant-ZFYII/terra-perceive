#!/usr/bin/env python3
"""extract_camera.py — Extract Pylon RGB frames from a RELLIS-3D bag.

Saves JPEGs numbered to match LiDAR frame indices (so frame N in
extracted_frames/000000.bin pairs with extracted_frames_camera/000000.jpg).

Sync strategy: for each LiDAR frame, find the camera frame whose
publication timestamp is closest. Both LiDAR and Pylon RGB run at ~10 Hz
in RELLIS-3D, so this lands within a few ms.

Usage:
    python scripts/extract_camera.py \\
        data/RELLIS-3D/00000_00.bag \\
        /media/.../m4_perframe/extracted_frames_camera \\
        --start-id 0 \\
        --lidar-dir /media/.../m4_perframe/extracted_frames

The --lidar-dir argument is needed so we know how many lidar frames this
bag produced (and therefore how many camera frames to output to keep IDs
matching). If omitted, every camera frame is saved.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

CAM_TOPIC   = "/pylon_camera_node/image_raw"
LIDAR_TOPIC = "/os1_cloud_node/points"


def collect_message_timestamps(bag_path: str, topic: str) -> List[Tuple[int, int]]:
    """Return list of (msg_index, ros_time_ns) for all messages on `topic`."""
    typestore = get_typestore(Stores.ROS1_NOETIC)
    out: List[Tuple[int, int]] = []
    with Reader(bag_path) as bag:
        connections = [c for c in bag.connections if c.topic == topic]
        if not connections:
            return out
        idx = 0
        for conn, ts, data in bag.messages(connections=connections):
            out.append((idx, ts))
            idx += 1
    return out


def decode_pylon_image(msg) -> np.ndarray:
    """Decode a sensor_msgs/Image from the Pylon camera into HxWx3 BGR uint8."""
    h, w = msg.height, msg.width
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    enc = (msg.encoding or "").lower()
    if enc in ("rgb8",):
        img = raw.reshape(h, w, 3)
    elif enc in ("bgr8",):
        img = raw.reshape(h, w, 3)
        img = img[:, :, ::-1]   # → RGB for matplotlib/Pillow
    elif enc in ("bayer_rggb8", "bayer_grbg8", "bayer_gbrg8", "bayer_bggr8"):
        # Bayer demosaic — RELLIS's Pylon publishes bayer_rggb8.
        # Note: OpenCV's Bayer flag naming is OFFSET vs ROS's pattern naming
        # (OpenCV names by 2nd-row 2nd-pixel; ROS names by 1st-row 1st-pixel).
        # Empirically verified on RELLIS frame 100: BayerBG2RGB gives correct
        # blue sky; BayerRG2RGB gives an orange-sky artifact.
        try:
            import cv2
            patterns = {
                "bayer_rggb8": cv2.COLOR_BayerBG2RGB,   # ROS RGGB ↔ OpenCV "BG"
                "bayer_grbg8": cv2.COLOR_BayerGB2RGB,
                "bayer_gbrg8": cv2.COLOR_BayerGR2RGB,
                "bayer_bggr8": cv2.COLOR_BayerRG2RGB,
            }
            img = cv2.cvtColor(raw.reshape(h, w), patterns[enc])
        except ImportError:
            # Fallback: naive 2x2 average (ugly but works without OpenCV).
            arr = raw.reshape(h, w)
            r = arr[0::2, 0::2]
            g = (arr[0::2, 1::2].astype(np.uint16) + arr[1::2, 0::2]) // 2
            b = arr[1::2, 1::2]
            small = np.stack([r, g.astype(np.uint8), b], axis=-1)
            # Upsample to original h×w via repeat.
            img = np.kron(small, np.ones((2, 2, 1), dtype=np.uint8))[:h, :w]
    elif enc == "mono8":
        img = np.repeat(raw.reshape(h, w, 1), 3, axis=2)
    else:
        raise ValueError(f"unsupported encoding: {msg.encoding!r}")
    return img


def extract_camera_synced(
    bag_path: str,
    out_dir: str,
    start_id: int,
    lidar_count_hint: int = -1,
    jpeg_quality: int = 80,
    every: int = 1,
) -> int:
    """Extract camera frames synced to LiDAR timestamps.

    Process:
      1. Walk the LiDAR topic to collect (lidar_idx, ts).
      2. Walk the camera topic and for each LiDAR ts find the camera msg
         with nearest ts. Save it as `<out>/<start_id+lidar_idx:06d>.jpg`.
      3. Returns the number of frames saved.
    """
    from PIL import Image as PILImage

    os.makedirs(out_dir, exist_ok=True)
    typestore = get_typestore(Stores.ROS1_NOETIC)

    print(f"[extract_camera] indexing {bag_path} ...", flush=True)
    with Reader(bag_path) as bag:
        lidar_conns = [c for c in bag.connections if c.topic == LIDAR_TOPIC]
        cam_conns   = [c for c in bag.connections if c.topic == CAM_TOPIC]
        if not lidar_conns:
            print(f"[extract_camera] {LIDAR_TOPIC} not in bag")
            return 0
        if not cam_conns:
            print(f"[extract_camera] {CAM_TOPIC} not in bag")
            return 0

        lidar_ts: List[int] = []
        for conn, ts, _ in bag.messages(connections=lidar_conns):
            lidar_ts.append(ts)
        if lidar_count_hint > 0 and len(lidar_ts) != lidar_count_hint:
            print(f"[extract_camera] WARN: lidar count mismatch "
                  f"(got {len(lidar_ts)}, expected {lidar_count_hint})")

        # Collect ALL camera frames (with timestamps + decoded images).
        # On a 6 GB bag that's ~600 camera msgs at maybe 5 MB each decoded
        # → up to 3 GB held in RAM. Acceptable on the 16 GB laptop and
        # avoids re-walking the bag for nearest-neighbor lookup.
        cam_ts: List[int] = []
        cam_msgs = []
        for conn, ts, data in bag.messages(connections=cam_conns):
            cam_ts.append(ts)
            cam_msgs.append((conn, data))
        print(f"[extract_camera]   lidar_ts={len(lidar_ts)}  cam_ts={len(cam_ts)}",
              flush=True)
        if not cam_ts:
            return 0

        cam_ts_arr = np.array(cam_ts, dtype=np.int64)

        saved = 0
        for li, lts in enumerate(lidar_ts):
            if li % every != 0:
                continue
            # Nearest cam frame by timestamp.
            ci = int(np.argmin(np.abs(cam_ts_arr - lts)))
            conn, data = cam_msgs[ci]
            try:
                msg = typestore.deserialize_ros1(data, conn.msgtype)
                img = decode_pylon_image(msg)
            except Exception as e:
                print(f"[extract_camera] frame {li}: decode failed ({e})")
                continue

            out_path = os.path.join(out_dir, f"{li + start_id:06d}.jpg")
            PILImage.fromarray(img).save(out_path, quality=jpeg_quality)
            saved += 1
            if saved % 100 == 0:
                print(f"[extract_camera]   {saved} frames saved (last: {out_path})",
                      flush=True)

    print(f"[extract_camera] done — {saved} frames saved to {out_dir}/", flush=True)
    return saved


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag_path",       type=str)
    p.add_argument("out_dir",        type=str)
    p.add_argument("--start-id",     type=int, default=0,
                   help="Offset for output frame numbering")
    p.add_argument("--lidar-count",  type=int, default=-1,
                   help="Expected LiDAR frame count for this bag (sanity check)")
    p.add_argument("--every",        type=int, default=1,
                   help="Save every Nth lidar-aligned camera frame")
    p.add_argument("--quality",      type=int, default=80,
                   help="JPEG quality (1-100)")
    args = p.parse_args()

    n = extract_camera_synced(
        args.bag_path, args.out_dir,
        start_id=args.start_id,
        lidar_count_hint=args.lidar_count,
        jpeg_quality=args.quality,
        every=args.every,
    )
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
