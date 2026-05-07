"""
publish_synced_stream.py — single combined publisher for lockstep BEV + camera.

Replaces the pair (publish_grid_stream.py, publish_camera_stream.py) for the
M3 dashboard demo recording. The two-process pair drifted out of sync over
the 9.5 min run because the camera publisher (~1 MB messages) hits the
requested 5 Hz easily while the BEV publisher (~9 MB messages) tops out at
roughly 2.4 Hz on this hardware. With independent pacers, the camera ran
ahead, the dashboard's 200-frame camera buffer eventually no longer covered
the BEV's frame index, and the rendered camera panel showed a frame much
later than the BEV panel.

This script enforces sync at the producer side: one frame index, two publishes
per tick, then sleep. The slowest publish (BEV) paces the loop; the camera
side waits its turn. Both subjects always carry the same Header.frame_id.

Usage:
    python scripts/publish_synced_stream.py \
        --run results/m3/slam_ema_covg2o_perframe \
        --camera-dir data/RELLIS-3D/.../pylon_camera_node \
        --hz 2.5 \
        --start-frame 0
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from transport.nats_adapter import Adapter
from transport.proto import perception_pb2 as pb

log = logging.getLogger("publish_synced_stream")

_HEADER_RE = re.compile(r"#\s*rows\s*=\s*(\d+),\s*cols\s*=\s*(\d+),\s*resolution\s*=\s*([\d.]+)")
_BEV_FRAME_RE = re.compile(r"frame_(\d+)\.csv$")
_CAM_FRAME_RE = re.compile(r"frame(\d+)[-_].*\.(?:jpg|jpeg|png)$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def list_bev_snapshots(run_dir: Path) -> dict[int, Path]:
    snap_dir = run_dir / "snapshots"
    if not snap_dir.is_dir():
        raise FileNotFoundError(snap_dir)
    out: dict[int, Path] = {}
    for p in sorted(snap_dir.glob("frame_*.csv")):
        m = _BEV_FRAME_RE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    if not out:
        raise FileNotFoundError(f"no frame_*.csv files in {snap_dir}")
    return out


def list_camera_frames(camera_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in camera_dir.glob("frame*.jpg"):
        m = _CAM_FRAME_RE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    if not out:
        raise FileNotFoundError(f"no camera frames in {camera_dir}")
    return out


def load_pose_sigma(run_dir: Path) -> dict[int, float]:
    p = run_dir / "pose_sigma.csv"
    if not p.exists():
        return {}
    out: dict[int, float] = {}
    with p.open() as f:
        for row in csv.DictReader(f):
            try:
                out[int(row["frame_id"])] = float(row["pose_sigma"])
            except (KeyError, ValueError):
                continue
    return out


# --------------------------------------------------------------------------- #
# BEV proto build (sparse-cell CSV -> dense BEVGrid proto)
# --------------------------------------------------------------------------- #
def parse_bev_snapshot(path: Path) -> tuple[int, int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open() as f:
        meta = f.readline()
        m = _HEADER_RE.match(meta)
        if not m:
            raise ValueError(f"bad header in {path}")
        rows, cols = int(m.group(1)), int(m.group(2))
        resolution = float(m.group(3))

        risk = np.zeros((rows, cols), dtype=np.float32)
        confidence = np.zeros((rows, cols), dtype=np.float32)
        obs_count = np.zeros((rows, cols), dtype=np.uint32)

        for row in csv.DictReader(f):
            r = int(row["row"]); c = int(row["col"])
            risk[r, c] = float(row["risk"])
            confidence[r, c] = float(row["confidence"])
            obs_count[r, c] = int(row["obs_count"])

    mask = (obs_count > 0).astype(np.uint8)
    return rows, cols, resolution, risk, confidence, obs_count, mask


def build_bev_proto(*, frame_id: int, resolution: float, rows: int, cols: int,
                    origin_x: float, origin_y: float,
                    risk: np.ndarray, confidence: np.ndarray,
                    obs_count: np.ndarray, mask: np.ndarray,
                    pose_sigma: float) -> pb.BEVGrid:
    msg = pb.BEVGrid()
    msg.header.timestamp = time.time()
    msg.header.frame_id = f"frame_{frame_id:06d}"
    msg.header.schema_version = 1
    msg.header.source = "publish_synced_stream"
    msg.resolution = resolution
    msg.width = cols
    msg.height = rows
    msg.origin_x = origin_x
    msg.origin_y = origin_y
    msg.scores.extend(risk.ravel().tolist())
    msg.confidence.extend(confidence.ravel().tolist())
    msg.observation_count.extend(obs_count.ravel().tolist())
    msg.coverage_mask = np.packbits(mask.ravel()).tobytes()
    msg.pose_sigma_at_snapshot = float(pose_sigma)
    return msg


def build_camera_proto(*, frame_id: int, jpeg_bytes: bytes,
                       width: int, height: int) -> pb.CameraFrame:
    msg = pb.CameraFrame()
    msg.header.timestamp = time.time()
    msg.header.frame_id = f"frame_{frame_id:06d}"
    msg.header.schema_version = 1
    msg.header.source = "publish_synced_stream"
    msg.width = width
    msg.height = height
    msg.encoding = "jpeg"
    msg.data = jpeg_bytes
    return msg


def jpeg_dimensions(buf: bytes) -> tuple[int, int]:
    if len(buf) < 4 or buf[:2] != b"\xff\xd8":
        return (0, 0)
    i = 2
    n = len(buf)
    while i < n - 9:
        if buf[i] != 0xFF:
            i += 1
            continue
        marker = buf[i + 1]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height = (buf[i + 5] << 8) | buf[i + 6]
            width = (buf[i + 7] << 8) | buf[i + 8]
            return (width, height)
        seg_len = (buf[i + 2] << 8) | buf[i + 3]
        i += 2 + seg_len
    return (0, 0)


# --------------------------------------------------------------------------- #
# Main lockstep loop
# --------------------------------------------------------------------------- #
async def stream(*, run_dir: Path, camera_dir: Path, nats_url: str,
                 grid_subject: str, camera_subject: str,
                 hz: float, loop_forever: bool,
                 start_frame: int, max_frames: Optional[int],
                 origin_x: Optional[float], origin_y: Optional[float]) -> None:
    bev_paths = list_bev_snapshots(run_dir)
    cam_paths = list_camera_frames(camera_dir)
    sigmas = load_pose_sigma(run_dir)

    # Frame indices that exist in BOTH sources (intersection). The sync is
    # at the frame-index level: tick N publishes BEV[N] and camera[N], or
    # nothing for that tick if either side is missing.
    common = sorted(set(bev_paths.keys()) & set(cam_paths.keys()))
    common = [f for f in common if f >= start_frame]
    log.info("BEV snapshots:    %d", len(bev_paths))
    log.info("camera frames:    %d", len(cam_paths))
    log.info("common indices:   %d  (intersection)", len(common))
    log.info("publishing on %s and %s at %.2f Hz", grid_subject, camera_subject, hz)

    if not common:
        raise SystemExit("no overlapping frame indices between BEV snapshots and camera frames")

    adapter = Adapter(nats_url)
    await adapter.connect()

    period = 1.0 / hz
    n_published = 0
    try:
        while True:
            for fid in common:
                t0 = time.perf_counter()

                # -- BEV --
                rows, cols, resolution, risk, conf, obs, mask = parse_bev_snapshot(bev_paths[fid])
                ox = origin_x if origin_x is not None else -resolution * cols / 2.0
                oy = origin_y if origin_y is not None else -resolution * rows / 2.0
                bev_msg = build_bev_proto(
                    frame_id=fid, resolution=resolution, rows=rows, cols=cols,
                    origin_x=ox, origin_y=oy,
                    risk=risk, confidence=conf, obs_count=obs, mask=mask,
                    pose_sigma=sigmas.get(fid, 0.0),
                )
                await adapter.publish(grid_subject, bev_msg.SerializeToString())

                # -- camera (immediately after BEV; same Header.frame_id) --
                jpeg_bytes = cam_paths[fid].read_bytes()
                w, h = jpeg_dimensions(jpeg_bytes)
                cam_msg = build_camera_proto(
                    frame_id=fid, jpeg_bytes=jpeg_bytes, width=w, height=h,
                )
                await adapter.publish(camera_subject, cam_msg.SerializeToString())

                n_published += 1
                if max_frames is not None and n_published >= max_frames:
                    log.info("reached --max-frames=%d, stopping", max_frames)
                    return

                elapsed = time.perf_counter() - t0
                if elapsed < period:
                    await asyncio.sleep(period - elapsed)
                else:
                    log.warning("frame %d took %.0f ms (>%.0f ms target)",
                                fid, elapsed * 1000, period * 1000)

            if not loop_forever:
                log.info("synced sequence exhausted, exiting")
                return
            log.info("looping back to first frame")
    finally:
        await adapter.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True,
                    help="results/m3/<run-dir>/  (must contain snapshots/ and pose_sigma.csv)")
    ap.add_argument("--camera-dir", type=Path, required=True)
    ap.add_argument("--hz", type=float, default=2.5,
                    help="lockstep rate (default 2.5; BEV serialize+publish bottlenecks at ~2.4 Hz)")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--grid-subject", default="perception.traversability.grid")
    ap.add_argument("--camera-subject", default="sensor.camera.rgb")
    ap.add_argument("--nats-url", default="nats://localhost:4222")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--origin-x", type=float, default=None)
    ap.add_argument("--origin-y", type=float, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    asyncio.run(stream(
        run_dir=args.run,
        camera_dir=args.camera_dir,
        nats_url=args.nats_url,
        grid_subject=args.grid_subject,
        camera_subject=args.camera_subject,
        hz=args.hz,
        loop_forever=args.loop,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        origin_x=args.origin_x,
        origin_y=args.origin_y,
    ))


if __name__ == "__main__":
    main()
