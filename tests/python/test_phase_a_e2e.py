"""
Phase A end-to-end integration test (P2-M5 exit gate).

Wires together everything Phase A produced — NATS broker, tracker_bridge,
the C++ tracker_node child, and the proto codec — and asserts the
detection-bytes-in -> track-bytes-out contract holds across NATS.

Steps:
  1. Spawn nats-server -js on a free port.
  2. Spawn `python -m transport.tracker_bridge` (which spawns the C++
     binary as its own grandchild).
  3. Subscribe to `perception.objects.tracks`.
  4. Publish 5 DetectionList frames on `perception.objects.detections`
     (the same frame_ids 84-88 the unit test uses).
  5. Wait until 5 TrackList messages arrive (or timeout).
  6. Assert stable track IDs across them.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import nats  # noqa: E402

from transport.proto import perception_pb2  # noqa: E402

NATS_SERVER = shutil.which("nats-server") or os.path.expanduser("~/.local/bin/nats-server")
TRACKER_NODE = REPO / "build/construction_perception/tracker_node"
LIDAR_DIR = (
    REPO
    / "data/RELLIS-3D/Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/00000"
    / "os1_cloud_node_kitti_bin"
)
CALIB = REPO / "config/camera_lidar_calib.yaml"
TEST_FRAMES = [84, 85, 86, 87, 88]
PERSON_BBOX = {"x_min": 0.0, "y_min": 332.0, "x_max": 144.0, "y_max": 982.0}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_detection_list_bytes(frame_idx: int) -> bytes:
    msg = perception_pb2.DetectionList()
    msg.header.frame_id = f"frame_{frame_idx:06d}"
    msg.header.timestamp = float(frame_idx) * 0.1
    msg.header.schema_version = 1
    msg.header.source = "phase-a-e2e"
    d = msg.detections.add()
    d.x_min = PERSON_BBOX["x_min"]
    d.y_min = PERSON_BBOX["y_min"]
    d.x_max = PERSON_BBOX["x_max"]
    d.y_max = PERSON_BBOX["y_max"]
    d.class_id = 0
    d.confidence = 0.85
    return msg.SerializeToString()


@pytest.fixture
def transient_stack():
    if not Path(NATS_SERVER).is_file() or not os.access(NATS_SERVER, os.X_OK):
        pytest.skip(f"nats-server missing at {NATS_SERVER}")
    if not TRACKER_NODE.is_file() or not os.access(TRACKER_NODE, os.X_OK):
        pytest.skip(
            f"tracker_node missing at {TRACKER_NODE}. "
            "Build with: colcon build --packages-select construction_perception",
        )

    port = _free_port()
    store = tempfile.mkdtemp(prefix="natsjs_e2e_")
    nats_proc = subprocess.Popen(
        [NATS_SERVER, "-js", "-p", str(port), "-sd", store],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        nats_proc.terminate()
        nats_proc.wait(timeout=3)
        pytest.fail("nats-server did not come up")

    bridge_proc = subprocess.Popen(
        [
            sys.executable, "-m", "transport.tracker_bridge",
            "--binary", str(TRACKER_NODE),
            "--nats-url", f"nats://127.0.0.1:{port}",
            "--calib", str(CALIB),
            "--lidar-dir", str(LIDAR_DIR),
        ],
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Bridge needs a moment to connect to NATS and spawn the C++ child.
    time.sleep(1.0)

    try:
        yield f"nats://127.0.0.1:{port}"
    finally:
        bridge_proc.terminate()
        try:
            bridge_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            bridge_proc.kill()
            bridge_proc.wait()
        nats_proc.terminate()
        try:
            nats_proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            nats_proc.kill()
            nats_proc.wait()
        shutil.rmtree(store, ignore_errors=True)


def test_phase_a_end_to_end(transient_stack):
    url = transient_stack

    async def run() -> Counter:
        nc = await nats.connect(url)
        tracks_received: list[perception_pb2.TrackList] = []
        done = asyncio.Event()

        async def handler(msg) -> None:
            tl = perception_pb2.TrackList()
            tl.ParseFromString(msg.data)
            tracks_received.append(tl)
            if len(tracks_received) >= len(TEST_FRAMES):
                done.set()

        try:
            await nc.subscribe("perception.objects.tracks", cb=handler)
            await asyncio.sleep(0.2)  # let subscription register

            for fi in TEST_FRAMES:
                await nc.publish(
                    "perception.objects.detections",
                    _build_detection_list_bytes(fi),
                )
                await asyncio.sleep(0.05)  # pace gently for the bridge's drain

            try:
                await asyncio.wait_for(done.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                pass
        finally:
            await nc.drain()

        ids: Counter[int] = Counter()
        for tl in tracks_received:
            for t in tl.tracks:
                ids[t.track_id] += 1
        return ids, len(tracks_received)

    ids, n_received = asyncio.run(run())

    assert n_received >= len(TEST_FRAMES) - 1, (
        f"received {n_received} TrackLists; expected ~{len(TEST_FRAMES)}. "
        f"Bridge likely failed to forward through C++ child."
    )
    most_common, count = ids.most_common(1)[0] if ids else (None, 0)
    assert count >= 4, (
        f"Expected the same track_id to recur in ≥4 frames over the bridge; "
        f"got id={most_common} count={count}, full histogram={dict(ids)}"
    )
