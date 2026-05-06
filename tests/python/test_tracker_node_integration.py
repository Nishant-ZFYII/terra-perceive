"""
Integration test for the C++ tracker_node binary (P2-M5.A2 step 3).

Spawns the binary directly (no NATS), pipes 5 length-prefixed
DetectionList frames into stdin, reads 5 length-prefixed TrackList
frames from stdout, and asserts that the same physical person carries
a stable track ID across the sequence.

Why this exists: it's the wondrous-crane-plan A2 exit gate ("5
synthetic Detection2D frames -> tracker maintains IDs across frames
-> published TrackList ID stability matches expected"). It also
exercises the full I/O frame in C++ (proto parse, LiDAR load, projection,
SORT update, proto write) so any compile-time-OK / runtime-broken
issue surfaces here, not in the live NATS demo.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from transport.frame_io import read_frame, write_frame  # noqa: E402
from transport.proto import perception_pb2  # noqa: E402

BINARY = REPO / "build/construction_perception/tracker_node"
LIDAR_DIR = (
    REPO
    / "data/RELLIS-3D/Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/00000"
    / "os1_cloud_node_kitti_bin"
)
CALIB = REPO / "config/camera_lidar_calib.yaml"

# Frames 84..88 sit inside the 0b person-window (frames 83..240); detection_node
# already showed person hits at conf 0.72-0.85 across this slice. The bbox
# below is the typical person-on-left-edge silhouette seen in the smoke test:
#   x in [0, 144], y in [332, 982], on a 1853x1025 image.
PERSON_BBOX = {"x_min": 0.0, "y_min": 332.0, "x_max": 144.0, "y_max": 982.0}
TEST_FRAMES = [84, 85, 86, 87, 88]


def _build_detection_list_bytes(frame_idx: int) -> bytes:
    msg = perception_pb2.DetectionList()
    msg.header.frame_id = f"frame_{frame_idx:06d}"
    msg.header.timestamp = float(frame_idx) * 0.1
    msg.header.schema_version = 1
    msg.header.source = "integration-test"
    d = msg.detections.add()
    d.x_min = PERSON_BBOX["x_min"]
    d.y_min = PERSON_BBOX["y_min"]
    d.x_max = PERSON_BBOX["x_max"]
    d.y_max = PERSON_BBOX["y_max"]
    d.class_id = 0  # person
    d.confidence = 0.85
    d.depth_3d = 0.0
    return msg.SerializeToString()


@pytest.fixture(scope="module")
def binary_present():
    if not BINARY.is_file() or not os.access(BINARY, os.X_OK):
        pytest.skip(
            f"tracker_node binary missing at {BINARY}. "
            "Build first: colcon build --packages-select construction_perception",
        )
    if not LIDAR_DIR.is_dir():
        pytest.skip(f"RELLIS LiDAR dir missing: {LIDAR_DIR}")


def test_tracker_node_stable_ids_over_5_frames(binary_present):
    """Pipe 5 person-bbox frames in; expect 5 TrackLists out + ID stability."""
    proc = subprocess.Popen(
        [
            str(BINARY),
            "--calib", str(CALIB),
            "--lidar-dir", str(LIDAR_DIR),
            "--max-dist", "5.0",
            "--max-misses", "10",
            "--min-hits", "1",
            "--dt", "0.1",
            "--process-noise", "2.0",
            "--meas-noise", "0.3",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert proc.stdin is not None and proc.stdout is not None

    # Write all 5 frames synchronously, then close stdin so the child sees EOF.
    for fi in TEST_FRAMES:
        write_frame(proc.stdin, _build_detection_list_bytes(fi))
    proc.stdin.close()

    # Read back 5 frames.
    out_frames = []
    for _ in TEST_FRAMES:
        payload = read_frame(proc.stdout)
        if payload is None:
            break
        msg = perception_pb2.TrackList()
        msg.ParseFromString(payload)
        out_frames.append(msg)

    proc.wait(timeout=15)
    stderr_tail = proc.stderr.read().decode("utf-8", errors="replace")[-2000:]

    assert len(out_frames) == len(TEST_FRAMES), (
        f"got {len(out_frames)} TrackList frames; expected {len(TEST_FRAMES)}.\n"
        f"stderr tail:\n{stderr_tail}"
    )

    n_with_tracks = sum(1 for m in out_frames if len(m.tracks) > 0)
    assert n_with_tracks >= 4, (
        f"expected >= 4 of 5 frames to produce >= 1 track; got {n_with_tracks}.\n"
        f"This usually means LiDAR projection inside the bbox is dropping all "
        f"points (option-i drop) — check the bbox size + person distance.\n"
        f"stderr tail:\n{stderr_tail}"
    )

    id_counts: Counter[int] = Counter()
    for m in out_frames:
        for t in m.tracks:
            id_counts[t.track_id] += 1

    most_common_id, most_common_count = id_counts.most_common(1)[0]
    assert most_common_count >= 4, (
        f"expected the same track_id to recur in >= 4 of 5 frames; "
        f"the most common id={most_common_id} appeared {most_common_count}x. "
        f"Full id histogram: {dict(id_counts)}.\n"
        f"stderr tail:\n{stderr_tail}"
    )
