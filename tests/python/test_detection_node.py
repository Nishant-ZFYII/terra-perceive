"""
Tests for detection_node.py (P2-M5.A1).

Two tests, matching the wondrous-crane plan A1 spec:
  test_detection_on_known_image — load one RELLIS frame from the dense
    person window (frame 84, conf 0.85 in our Phase 0 audit), run YOLO,
    assert exactly one Detection2D class_id=0 (person) with conf >= 0.5.
  test_detection_list_round_trip — construct a DetectionList programmatically,
    serialize, parse, assert field-by-field equality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from python.detection_node import build_detection_list  # noqa: E402
from transport.proto import perception_pb2  # noqa: E402

# Frame 84 in the RELLIS seq 00 person window (frames 83..240).
# Phase 0b found YOLOv8n fires "person" at conf 0.845 on this frame.
# Glob the timestamp suffix so the test isn't tied to one exact ms offset.
RELLIS_PYLON = (
    REPO
    / "data/RELLIS-3D/Rellis_3D_pylon_camera_node/Rellis-3D/00000/pylon_camera_node"
)
_matches = sorted(RELLIS_PYLON.glob("frame000084-*.jpg")) if RELLIS_PYLON.exists() else []
KNOWN_PERSON_FRAME = _matches[0] if _matches else RELLIS_PYLON / "missing.jpg"


@pytest.fixture(scope="module")
def yolo_model():
    from ultralytics import YOLO

    weights = "/media/nishant/SeeGayt2/terra_perceive/models/yolov8n.pt"
    return YOLO(weights)


def test_detection_on_known_image(yolo_model):
    if not KNOWN_PERSON_FRAME.exists():
        pytest.skip(f"RELLIS frame missing: {KNOWN_PERSON_FRAME}")
    img = cv2.imread(str(KNOWN_PERSON_FRAME))
    assert img is not None, "cv2.imread returned None"

    msg, ndjson = build_detection_list(
        img, yolo_model, frame_id="frame_000084", source_tag="test",
    )

    assert isinstance(msg, perception_pb2.DetectionList)
    assert msg.header.frame_id == "frame_000084"
    assert msg.header.schema_version == 1
    # On this frame Phase 0 saw exactly 1 person hit; assert at least 1, mostly 1.
    assert len(msg.detections) >= 1, "expected >= 1 person detection on frame 84"

    person_dets = [d for d in msg.detections if d.class_id == 0]
    assert len(person_dets) >= 1
    p = person_dets[0]
    assert p.confidence >= 0.5
    assert p.x_max > p.x_min
    assert p.y_max > p.y_min
    assert p.depth_3d == 0.0  # detection node does NOT fill depth (boundary contract)
    assert ndjson, "ndjson list should be non-empty when detections exist"
    assert ndjson[0]["class_name"] == "person"


def test_detection_list_round_trip():
    msg = perception_pb2.DetectionList()
    msg.header.timestamp = 1234.5
    msg.header.frame_id = "frame_000042"
    msg.header.schema_version = 1
    msg.header.source = "unit-test"
    d = msg.detections.add()
    d.x_min, d.y_min, d.x_max, d.y_max = 10.0, 20.0, 110.0, 220.0
    d.class_id = 0
    d.confidence = 0.8765
    d.depth_3d = 0.0

    payload = msg.SerializeToString()
    parsed = perception_pb2.DetectionList()
    parsed.ParseFromString(payload)

    assert parsed.header.timestamp == pytest.approx(1234.5)
    assert parsed.header.frame_id == "frame_000042"
    assert parsed.header.schema_version == 1
    assert parsed.header.source == "unit-test"
    assert len(parsed.detections) == 1
    pd = parsed.detections[0]
    assert pd.x_min == pytest.approx(10.0)
    assert pd.y_min == pytest.approx(20.0)
    assert pd.x_max == pytest.approx(110.0)
    assert pd.y_max == pytest.approx(220.0)
    assert pd.class_id == 0
    assert pd.confidence == pytest.approx(0.8765, abs=1e-6)
    assert pd.depth_3d == pytest.approx(0.0)
