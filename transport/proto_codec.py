"""
proto_codec.py — typed pack/unpack helpers for the M5 NATS message types.

The M3 nats_adapter.Adapter ships raw bytes; serialization is the caller's job.
Without a shared codec, each consumer (tracker_node, safety_supervisor, replay
harness, dashboard) ends up duplicating SerializeToString / ParseFromString
plus header-population boilerplate. This module is the single place that owns
that boilerplate so the rest of M5 reads cleanly.

Scope: just three message types we actually use across M5 — DetectionList,
TrackList, SafetyEvent. Add more as needed; do NOT add codecs for messages
the project doesn't transmit.

Header schema (perception.proto:10-15) is populated automatically:
  timestamp = caller-supplied OR time.time()
  frame_id = caller-supplied
  schema_version = 1 (bump on breaking change)
  source = caller-supplied module name
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

from transport.proto import perception_pb2, safety_pb2

SCHEMA_VERSION = 1


@dataclass
class Header:
    """Plain dataclass mirror of perception_pb2.Header for ergonomic call sites."""
    frame_id: str
    source: str
    timestamp: float | None = None     # None → time.time()
    schema_version: int = SCHEMA_VERSION


def _fill_header(target, h: Header) -> None:
    target.timestamp = h.timestamp if h.timestamp is not None else time.time()
    target.frame_id = h.frame_id
    target.schema_version = h.schema_version
    target.source = h.source


# ----------------------------- DetectionList ----------------------------- #

def pack_detection_list(detections: Iterable[dict], header: Header) -> bytes:
    """detections: iterable of {x_min,y_min,x_max,y_max,class_id,confidence,depth_3d}."""
    msg = perception_pb2.DetectionList()
    _fill_header(msg.header, header)
    for d in detections:
        det = msg.detections.add()
        det.x_min = float(d["x_min"])
        det.y_min = float(d["y_min"])
        det.x_max = float(d["x_max"])
        det.y_max = float(d["y_max"])
        det.class_id = int(d["class_id"])
        det.confidence = float(d["confidence"])
        det.depth_3d = float(d.get("depth_3d", 0.0))
    return msg.SerializeToString()


def unpack_detection_list(payload: bytes) -> tuple[Header, list[dict]]:
    msg = perception_pb2.DetectionList()
    msg.ParseFromString(payload)
    h = Header(
        frame_id=msg.header.frame_id,
        source=msg.header.source,
        timestamp=msg.header.timestamp,
        schema_version=msg.header.schema_version,
    )
    dets = [
        {
            "x_min": d.x_min, "y_min": d.y_min, "x_max": d.x_max, "y_max": d.y_max,
            "class_id": d.class_id, "confidence": d.confidence,
            "depth_3d": d.depth_3d,
        }
        for d in msg.detections
    ]
    return h, dets


# ------------------------------- TrackList ------------------------------- #

def pack_track_list(tracks: Iterable[dict], header: Header) -> bytes:
    """tracks: iterable of {track_id,class_id,x,y,vx,vy,z_3d,hits}."""
    msg = perception_pb2.TrackList()
    _fill_header(msg.header, header)
    for t in tracks:
        tr = msg.tracks.add()
        tr.track_id = int(t["track_id"])
        tr.class_id = int(t.get("class_id", 0))
        tr.x = float(t["x"])
        tr.y = float(t["y"])
        tr.vx = float(t.get("vx", 0.0))
        tr.vy = float(t.get("vy", 0.0))
        tr.z_3d = float(t.get("z_3d", 0.0))
        tr.hits = int(t.get("hits", 0))
    return msg.SerializeToString()


def unpack_track_list(payload: bytes) -> tuple[Header, list[dict]]:
    msg = perception_pb2.TrackList()
    msg.ParseFromString(payload)
    h = Header(
        frame_id=msg.header.frame_id,
        source=msg.header.source,
        timestamp=msg.header.timestamp,
        schema_version=msg.header.schema_version,
    )
    tracks = [
        {
            "track_id": t.track_id, "class_id": t.class_id,
            "x": t.x, "y": t.y, "vx": t.vx, "vy": t.vy,
            "z_3d": t.z_3d, "hits": t.hits,
        }
        for t in msg.tracks
    ]
    return h, tracks


# ------------------------------ SafetyEvent ------------------------------ #
# safety.proto's SafetyEvent has its own flat fields (no embedded Header).
# We treat the SafetyEvent.timestamp + rule + details as the audit-trail tuple.

def pack_safety_event(event: dict) -> bytes:
    """event: {timestamp, rule, trigger_value, vel_before, vel_after, details}."""
    msg = safety_pb2.SafetyEvent()
    msg.timestamp = float(event.get("timestamp", time.time()))
    msg.rule = str(event["rule"])
    msg.trigger_value = float(event.get("trigger_value", 0.0))
    msg.vel_before = float(event.get("vel_before", 0.0))
    msg.vel_after = float(event.get("vel_after", 0.0))
    msg.details = str(event.get("details", ""))
    return msg.SerializeToString()


def unpack_safety_event(payload: bytes) -> dict:
    msg = safety_pb2.SafetyEvent()
    msg.ParseFromString(payload)
    return {
        "timestamp": msg.timestamp, "rule": msg.rule,
        "trigger_value": msg.trigger_value,
        "vel_before": msg.vel_before, "vel_after": msg.vel_after,
        "details": msg.details,
    }
