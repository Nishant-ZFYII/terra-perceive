"""
run_phase_b_rellis.py — P2-M5 Phase B RELLIS edge-case demo runner.

Drives the full M5 pipeline on a contiguous RELLIS frame window:

    YOLOv8n (Python)
       ↓ DetectionList (length-prefixed protobuf, on a pipe)
    build/.../tracker_node  (C++)
       ↓ TrackList  (length-prefixed protobuf, on a pipe)
    safety_replay.evaluate (Python port of supervisor rules)
       ↓ SafetyEvent

Outputs (default `results_m5/phase_b/`):
    tracks.ndjson         per-frame track rows
    safety_events.ndjson  per-frame safety event rows (rule, ttc, d_stop, ...)
    metrics.json          distinct_track_ids, lifetime stats, n_safety_events

By design this script is offline (NO live NATS) — Phase A's e2e test already
proved the NATS path works; Phase B measures the algorithm. NATS-replay
verification is a separate step (verify_phase_b_replay.py).

Decisions captured at runtime in the JSON header so they're never lost:
  - frame range used
  - SORT tuning (max_dist, max_misses, min_hits, dt, process_noise, meas_noise)
  - terrain traversability score used by the safety port (default 0.5 = mixed)
  - lateral arc threshold for "forward-arc tracks" (default 5 m)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import struct
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from transport.frame_io import read_frame, write_frame  # noqa: E402
from transport.proto import perception_pb2  # noqa: E402
from transport.safety_replay import evaluate as safety_evaluate, SafetyConfig  # noqa: E402

log = logging.getLogger("phase_b")

TARGET_CLASSES = {0, 1, 2, 3, 7}  # person, bicycle, car, motorcycle, truck
HEADLINE_CONF = 0.5
SCAN_CONF = 0.25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-frame", type=int, default=83)
    p.add_argument("--end-frame", type=int, default=241,
                   help="exclusive upper bound; default 241 = 158 frames (83..240)")
    p.add_argument(
        "--frames-dir",
        default="data/RELLIS-3D/Rellis_3D_pylon_camera_node/Rellis-3D/00000/pylon_camera_node",
    )
    p.add_argument(
        "--lidar-dir",
        default="data/RELLIS-3D/Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/00000/os1_cloud_node_kitti_bin",
    )
    p.add_argument("--calib", default="config/camera_lidar_calib.yaml")
    p.add_argument("--poses-csv", default="data/poses_carto.csv")
    p.add_argument("--weights", default="/media/nishant/SeeGayt2/terra_perceive/models/yolov8n.pt")
    p.add_argument("--tracker-binary", default="build/construction_perception/tracker_node")
    p.add_argument("--out-dir", default="results_m5/phase_b")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")

    # SORT defaults match M13.5 production config + tracker_node defaults.
    p.add_argument("--max-dist", type=float, default=5.0)
    p.add_argument("--max-misses", type=int, default=10)
    p.add_argument("--min-hits", type=int, default=1)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--process-noise", type=float, default=2.0)
    p.add_argument("--meas-noise", type=float, default=0.3)

    # Safety-port knobs.
    p.add_argument("--trav-score", type=float, default=0.5,
                   help="terrain traversability score (0..1); 0.5 = mixed default")
    p.add_argument("--lateral-arc-m", type=float, default=5.0,
                   help="forward-arc filter: |track.y| <= this for safety eval")
    return p.parse_args()


def precompute_ego_speed(poses_csv: Path) -> list[float]:
    """Returns a list speed[frame_idx] = ||Δ(x,y)|| / Δt in m/s.
    speed[0] is set to 0; index N-1 is the final frame."""
    import pandas as pd
    df = pd.read_csv(poses_csv)
    ts = df["timestamp"].to_numpy(dtype=np.float64) / 1e9   # s
    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    speeds = [0.0]
    for k in range(1, len(df)):
        dt = ts[k] - ts[k - 1]
        if dt <= 0:
            speeds.append(speeds[-1])
            continue
        d = math.hypot(x[k] - x[k - 1], y[k] - y[k - 1])
        speeds.append(d / dt)
    return speeds


def build_detection_list_payload(
    img: np.ndarray, model, frame_id: str, imgsz: int, device: str,
) -> bytes:
    msg = perception_pb2.DetectionList()
    msg.header.timestamp = time.time()
    msg.header.frame_id = frame_id
    msg.header.schema_version = 1
    msg.header.source = "phase_b_runner"

    result = model.predict(
        img, conf=SCAN_CONF, imgsz=imgsz, device=device, verbose=False,
    )[0]
    if result.boxes is not None and result.boxes.shape[0] > 0:
        cls = result.boxes.cls.cpu().numpy().astype(int)
        conf = result.boxes.conf.cpu().numpy()
        xyxy = result.boxes.xyxy.cpu().numpy()
        for cid, cf, box in zip(cls, conf, xyxy):
            if int(cid) not in TARGET_CLASSES or float(cf) < HEADLINE_CONF:
                continue
            d = msg.detections.add()
            d.x_min, d.y_min = float(box[0]), float(box[1])
            d.x_max, d.y_max = float(box[2]), float(box[3])
            d.class_id = int(cid)
            d.confidence = float(cf)
            d.depth_3d = 0.0
    return msg.SerializeToString()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = out_dir / "tracks.ndjson"
    events_path = out_dir / "safety_events.ndjson"
    metrics_path = out_dir / "metrics.json"

    frames_dir = Path(args.frames_dir).resolve()
    all_frames = sorted(frames_dir.glob("frame*.jpg"))
    if not all_frames:
        log.error("no jpgs in %s", frames_dir); return 1

    # Map frame index -> jpg path. Filenames are frameNNNNNN-<ts>_<ms>.jpg.
    by_idx: dict[int, Path] = {}
    for fp in all_frames:
        try:
            idx = int(fp.name[5:11])  # "frameNNNNNN"
            by_idx[idx] = fp
        except Exception:
            continue

    log.info("frame window: [%d, %d) (n=%d)",
             args.start_frame, args.end_frame, args.end_frame - args.start_frame)

    ego_speeds = precompute_ego_speed(Path(args.poses_csv))
    log.info("ego speed in window: mean=%.3f m/s max=%.3f",
             float(np.mean(ego_speeds[args.start_frame:args.end_frame])),
             float(np.max(ego_speeds[args.start_frame:args.end_frame])))

    # Spawn the C++ tracker_node child.
    tracker_argv = [
        args.tracker_binary,
        "--calib", args.calib,
        "--lidar-dir", args.lidar_dir,
        "--max-dist", str(args.max_dist),
        "--max-misses", str(args.max_misses),
        "--min-hits", str(args.min_hits),
        "--dt", str(args.dt),
        "--process-noise", str(args.process_noise),
        "--meas-noise", str(args.meas_noise),
    ]
    log.info("spawning %s", " ".join(tracker_argv))
    proc = subprocess.Popen(
        tracker_argv,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    # Lazy YOLO import (~1s) after the tracker spawned.
    from ultralytics import YOLO
    Path(args.weights).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    track_lifetimes: dict[int, int] = defaultdict(int)        # id -> frames seen
    track_first_frame: dict[int, int] = {}
    track_last_frame: dict[int, int] = {}
    n_safety_events = 0
    n_frames_with_track = 0

    cfg = SafetyConfig()

    tracks_fh = tracks_path.open("w")
    events_fh = events_path.open("w")

    t_start = time.time()
    n_processed = 0
    n_dets_total = 0
    for fi in range(args.start_frame, args.end_frame):
        if fi not in by_idx:
            log.warning("missing camera frame for idx %d; skipping", fi); continue
        img = cv2.imread(str(by_idx[fi]))
        if img is None:
            log.warning("imread failed: %s", by_idx[fi]); continue

        payload = build_detection_list_payload(
            img, model, f"frame_{fi:06d}", args.imgsz, args.device,
        )
        write_frame(proc.stdin, payload)

        # Read exactly one TrackList back (blocking; matches the C++ side
        # which always emits one TrackList per DetectionList).
        out = read_frame(proc.stdout)
        if out is None:
            log.error("tracker_node EOF mid-run; stderr tail:")
            sys.stderr.write(proc.stderr.read().decode(errors="replace"))
            return 2
        tl = perception_pb2.TrackList()
        tl.ParseFromString(out)
        n_processed += 1

        # Estimate det count by stepping into the inbound proto. We could
        # parse the outgoing DetectionList for a perfectly accurate count,
        # but the tracker stderr surface is already metric-rich; skip.
        # n_dets_total stays best-effort.

        if len(tl.tracks) > 0:
            n_frames_with_track += 1

        # Persist tracks.
        for t in tl.tracks:
            track_lifetimes[t.track_id] += 1
            track_first_frame.setdefault(t.track_id, fi)
            track_last_frame[t.track_id] = fi
            tracks_fh.write(json.dumps({
                "frame_idx": fi,
                "track_id": int(t.track_id),
                "class_id": int(t.class_id),
                "x": float(t.x), "y": float(t.y),
                "vx": float(t.vx), "vy": float(t.vy),
                "z_3d": float(t.z_3d),
                "hits": int(t.hits),
            }) + "\n")

        # Safety pass — pick the nearest track in lateral arc and run the
        # python-port supervisor rules.
        #
        # FRAME-CONVENTION NOTE: per docs/m4-fusion.md, the RELLIS Warthog's
        # camera optical axis maps to -X in the LiDAR frame ("the camera's
        # optical axis points toward -X in LiDAR frame"). The SORT tracker
        # consumes positions in LiDAR frame, so a person visible to the
        # camera lands at NEGATIVE track.x. Use a sign-agnostic formulation:
        #   d_worker      = sqrt(x^2 + y^2)                       (Euclidean)
        #   v_radial_out  = (x*vx + y*vy) / d_worker  (positive = receding)
        # Both line up with the supervisor's "worker_approach_speed positive
        # = worker moving away" sign convention (m5-safety.md TTC table).
        v_ego = float(ego_speeds[fi]) if fi < len(ego_speeds) else 0.0
        nearest = None
        nearest_d = math.inf
        for t in tl.tracks:
            if abs(t.y) > args.lateral_arc_m:
                continue                            # outside lateral arc
            d = math.hypot(t.x, t.y)
            if d < nearest_d:
                nearest_d = d
                nearest = t
        if nearest is not None and nearest_d > 1e-6:
            v_worker = (nearest.x * nearest.vx + nearest.y * nearest.vy) / nearest_d
            rule, scale = safety_evaluate(
                v_vehicle=v_ego, d_worker=nearest_d,
                v_worker=v_worker, trav_score=args.trav_score, cfg=cfg,
            )
            if rule != "TTC >= 5s":
                n_safety_events += 1
                events_fh.write(json.dumps({
                    "frame_idx": fi,
                    "timestamp": time.time(),
                    "rule": rule,
                    "trigger_value": nearest_d,
                    "vel_before": v_ego,
                    "vel_after": v_ego * scale,
                    "track_id": int(nearest.track_id),
                    "details": json.dumps({
                        "v": v_ego, "d": nearest_d, "v_w": v_worker,
                        "trav": args.trav_score,
                    }),
                }) + "\n")

    # Close child cleanly.
    try:
        proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    stderr_tail = proc.stderr.read().decode(errors="replace")[-2000:]

    tracks_fh.close()
    events_fh.close()

    wall = time.time() - t_start
    lifetimes = list(track_lifetimes.values())
    longest_lifetime = max(lifetimes) if lifetimes else 0
    mean_lifetime = float(np.mean(lifetimes)) if lifetimes else 0.0
    metrics = {
        "frame_window": [args.start_frame, args.end_frame],
        "n_frames_processed": n_processed,
        "n_frames_with_track": n_frames_with_track,
        "distinct_track_ids": len(track_lifetimes),
        "track_lifetimes": dict(sorted(track_lifetimes.items())),
        "longest_lifetime": int(longest_lifetime),
        "mean_lifetime": mean_lifetime,
        "n_safety_events": n_safety_events,
        "wall_seconds": wall,
        "fps": (n_processed / wall) if wall > 0 else 0.0,
        "config": {
            "max_dist": args.max_dist, "max_misses": args.max_misses,
            "min_hits": args.min_hits, "dt": args.dt,
            "process_noise": args.process_noise, "meas_noise": args.meas_noise,
            "trav_score": args.trav_score, "lateral_arc_m": args.lateral_arc_m,
        },
        "tracker_stderr_tail": stderr_tail,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print()
    print(f"=== Phase B metrics ({args.start_frame}..{args.end_frame}) ===")
    print(f"frames processed       : {n_processed}")
    print(f"frames with >=1 track  : {n_frames_with_track}")
    print(f"distinct track IDs     : {len(track_lifetimes)}")
    print(f"longest track lifetime : {longest_lifetime} frames")
    print(f"mean track lifetime    : {mean_lifetime:.2f} frames")
    print(f"safety events fired    : {n_safety_events}")
    print(f"wall                   : {wall:.2f} s ({metrics['fps']:.2f} fps)")
    print(f"outputs                : {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
