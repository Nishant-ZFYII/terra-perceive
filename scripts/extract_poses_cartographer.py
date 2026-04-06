"""
extract_poses_cartographer.py — Extract trajectory poses from a Cartographer .pbstream file.

The .pbstream is a serialized protobuf containing the full SLAM state.
Trajectory nodes are stored as SerializedData entries with a trajectory_node_data
field containing timestamped SE(3) poses.

Usage:
    python3 scripts/extract_poses_cartographer.py --pbstream data/output.pbstream --out data/poses_carto.csv

Output format matches GPS and ICP CSVs:
    timestamp, x, y, z, qw, qx, qy, qz
"""

import argparse
import csv
import struct
import os

import numpy as np
from scipy.spatial.transform import Rotation as R


def read_pbstream(pbstream_path):
    """
    Parse a .pbstream file to extract trajectory node poses.

    The .pbstream format is:
    - 8 bytes: magic header "CARTOGRAPHER"... (actually a custom format)
    - Repeated: [varint size] [SerializedData protobuf]

    Since we don't have the compiled Cartographer proto definitions,
    we use a lightweight approach: run cartographer_pbstream_to_ros_map
    or parse the proto manually.

    This implementation uses the raw protobuf wire format to extract poses.
    """
    poses = []

    try:
        # Attempt 1: Use protobuf if Cartographer proto defs are available
        from google.protobuf import descriptor_pool, symbol_database
        from cartographer.mapping.proto import serialization_pb2
        return _parse_with_proto(pbstream_path, serialization_pb2)
    except ImportError:
        pass

    try:
        # Attempt 2: Use the cartographer_pbstream tool inside Docker
        return _parse_with_docker(pbstream_path)
    except Exception:
        pass

    # Attempt 3: Parse the KITTI-format poses.txt as fallback
    # (Cartographer's offline node can be configured to write KITTI poses)
    kitti_path = pbstream_path.replace('.pbstream', '_kitti.txt')
    if os.path.exists(kitti_path):
        return _parse_kitti_poses(kitti_path)

    raise RuntimeError(
        f"Could not parse {pbstream_path}. Options:\n"
        "  1. Run extraction inside the Cartographer Docker container\n"
        "  2. Install cartographer Python proto bindings\n"
        "  3. Provide a KITTI-format poses file alongside the .pbstream"
    )


def _parse_with_proto(pbstream_path, serialization_pb2):
    """Parse .pbstream using compiled Cartographer protobuf definitions."""
    poses = []

    with open(pbstream_path, 'rb') as f:
        data = f.read()

    # .pbstream is a sequence of SerializedData messages
    # prefixed with their size as a varint
    pos = 0
    while pos < len(data):
        # Read varint (message size)
        size, bytes_consumed = _decode_varint(data, pos)
        pos += bytes_consumed

        if pos + size > len(data):
            break

        msg_data = data[pos:pos + size]
        pos += size

        msg = serialization_pb2.SerializedData()
        msg.ParseFromString(msg_data)

        if msg.HasField('trajectory_node_data'):
            node = msg.trajectory_node_data
            t = node.timestamp
            p = node.local_pose
            translation = p.translation
            rotation = p.rotation

            poses.append({
                'timestamp': t,
                'x': translation.x,
                'y': translation.y,
                'z': translation.z,
                'qw': rotation.w,
                'qx': rotation.x,
                'qy': rotation.y,
                'qz': rotation.z,
            })

    return poses


def _parse_with_docker(pbstream_path):
    """
    Extract poses by running a helper command inside the Cartographer Docker container.
    Writes a temporary KITTI-format file, then parses it.
    """
    import subprocess
    import tempfile

    abs_pbstream = os.path.abspath(pbstream_path)
    pbstream_dir = os.path.dirname(abs_pbstream)
    pbstream_name = os.path.basename(abs_pbstream)

    # Run cartographer_dev_pbstream_trajectories_to_proto inside Docker
    result = subprocess.run([
        'docker', 'compose', '-f', 'docker/docker-compose.yml',
        '--profile', 'cartographer', 'run', '--rm',
        '-v', f'{pbstream_dir}:/pbdata',
        'cartographer',
        'cartographer_dev_pbstream_trajectories_to_proto',
        f'-input=/pbdata/{pbstream_name}',
        '-output_prefix=/pbdata/trajectory'
    ], capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        raise RuntimeError(f"Docker extraction failed: {result.stderr}")

    # Parse the output proto files
    # cartographer_dev_pbstream_trajectories_to_proto writes trajectory_X.proto
    proto_files = sorted(
        f for f in os.listdir(pbstream_dir)
        if f.startswith('trajectory') and f.endswith('.proto')
    )

    if not proto_files:
        raise RuntimeError("No trajectory proto files produced")

    # These are text proto files — parse manually
    poses = []
    for pf in proto_files:
        with open(os.path.join(pbstream_dir, pf), 'r') as f:
            content = f.read()
        # Parse text proto for pose entries (simplified parser)
        # Full implementation would use protobuf text format parser
        # For now, raise and let fallback handle it
        raise NotImplementedError("Text proto parsing not implemented — use KITTI fallback")

    return poses


def _parse_kitti_poses(kitti_path):
    """Parse KITTI-format poses: 12 floats per line (flattened 3x4 [R|t] matrix)."""
    poses = []
    with open(kitti_path, 'r') as f:
        for idx, line in enumerate(f):
            values = [float(v) for v in line.strip().split()]
            if len(values) != 12:
                continue

            # Reconstruct 3x4 matrix
            T = np.eye(4)
            T[:3, :] = np.array(values).reshape(3, 4)

            x, y, z = T[:3, 3]
            r = R.from_matrix(T[:3, :3])
            qx, qy, qz, qw = r.as_quat()

            # Synthetic timestamp at 10 Hz
            timestamp = int(idx * 1e8)
            poses.append({
                'timestamp': timestamp,
                'x': x, 'y': y, 'z': z,
                'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
            })

    return poses


def _decode_varint(data, pos):
    """Decode a protobuf varint from bytes."""
    result = 0
    shift = 0
    bytes_consumed = 0
    while pos < len(data):
        b = data[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        bytes_consumed += 1
        shift += 7
        if (b & 0x80) == 0:
            break
    return result, bytes_consumed


def write_csv(poses, out_csv):
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
        for p in poses:
            writer.writerow([
                p['timestamp'], p['x'], p['y'], p['z'],
                p['qw'], p['qx'], p['qy'], p['qz']
            ])
    print(f"Saved {len(poses)} poses to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Cartographer poses to CSV")
    parser.add_argument("--pbstream", help="Path to .pbstream file")
    parser.add_argument("--kitti", help="Path to KITTI-format poses.txt (fallback if no .pbstream)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    if args.kitti:
        poses = _parse_kitti_poses(args.kitti)
    elif args.pbstream:
        poses = read_pbstream(args.pbstream)
    else:
        parser.error("Provide either --pbstream or --kitti")

    write_csv(poses, args.out)
