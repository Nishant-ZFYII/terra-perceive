"""
Round-trip tests for transport/proto_codec.py (P2-M5.A3).

Three tests, one per message type. Each:
  1. constructs a typed dict
  2. packs to bytes
  3. unpacks back
  4. asserts field-by-field equality
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from transport.proto_codec import (  # noqa: E402
    Header,
    pack_detection_list,
    unpack_detection_list,
    pack_track_list,
    unpack_track_list,
    pack_safety_event,
    unpack_safety_event,
)


def test_detection_list_round_trip():
    header = Header(frame_id="frame_000007", source="unit-test", timestamp=42.0)
    dets = [
        {
            "x_min": 0.5, "y_min": 1.5, "x_max": 100.5, "y_max": 200.5,
            "class_id": 0, "confidence": 0.92, "depth_3d": 7.25,
        },
        {
            "x_min": 200.0, "y_min": 50.0, "x_max": 320.0, "y_max": 250.0,
            "class_id": 2, "confidence": 0.71, "depth_3d": 0.0,
        },
    ]
    payload = pack_detection_list(dets, header)
    h, parsed = unpack_detection_list(payload)
    assert h.frame_id == "frame_000007"
    assert h.source == "unit-test"
    assert h.timestamp == pytest.approx(42.0)
    assert h.schema_version == 1
    assert len(parsed) == 2
    for orig, got in zip(dets, parsed):
        for k, v in orig.items():
            assert got[k] == pytest.approx(v, abs=1e-6), f"field={k}"


def test_track_list_round_trip():
    header = Header(frame_id="frame_000018", source="tracker_node:test")
    tracks = [
        {
            "track_id": 3, "class_id": 0, "x": 5.5, "y": -2.3,
            "vx": 0.4, "vy": 0.1, "z_3d": 0.0, "hits": 12,
        },
        {
            "track_id": 4, "class_id": 2, "x": 10.0, "y": 1.5,
            "vx": -1.2, "vy": 0.0, "z_3d": 0.5, "hits": 3,
        },
    ]
    payload = pack_track_list(tracks, header)
    h, parsed = unpack_track_list(payload)
    assert h.frame_id == "frame_000018"
    assert h.source == "tracker_node:test"
    assert len(parsed) == 2
    for orig, got in zip(tracks, parsed):
        for k, v in orig.items():
            assert got[k] == pytest.approx(v, abs=1e-6), f"field={k}"


def test_safety_event_round_trip():
    event = {
        "timestamp": 1234.5, "rule": "TTC < 2s", "trigger_value": 1.7,
        "vel_before": 2.0, "vel_after": 0.2,
        "details": "worker at 3.4 m, mu=0.6",
    }
    payload = pack_safety_event(event)
    parsed = unpack_safety_event(payload)
    for k, v in event.items():
        if isinstance(v, float):
            assert parsed[k] == pytest.approx(v, abs=1e-6), f"field={k}"
        else:
            assert parsed[k] == v, f"field={k}"


def test_header_default_timestamp_is_recent():
    """When Header.timestamp is None, the packed message should pick time.time()."""
    import time as _t
    header = Header(frame_id="f", source="s", timestamp=None)
    t0 = _t.time()
    payload = pack_detection_list([], header)
    h, _ = unpack_detection_list(payload)
    assert abs(h.timestamp - t0) < 1.0, "default timestamp should be ~now"
