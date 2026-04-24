"""
publish_camera_stream.py — replay RELLIS camera frames through NATS as a fake
live RGB stream.

Mirrors scripts/publish_grid_stream.py in shape: walks a directory of JPEGs in
filename-index order, wraps each in a CameraFrame protobuf, publishes on
sensor.camera.rgb at the requested rate.

Pair with publish_grid_stream.py at the same --hz so the dashboard's two
panels stay roughly aligned. Time-sync is best-effort (publishers paced
independently); a real consumer would buffer-and-match by header.timestamp.

Usage:
    python scripts/publish_camera_stream.py \\
        --camera-dir data/RELLIS-3D/Rellis_3D_pylon_camera_node/Rellis-3D/00000/pylon_camera_node \\
        --hz 5 --loop

CLI flags:
    --camera-dir   Directory containing frame*.jpg files.
    --hz           Publish rate (default 5; match the grid publisher).
    --loop         Loop the sequence forever.
    --subject      NATS subject (default sensor.camera.rgb).
    --nats-url     NATS server URL (default nats://localhost:4222).
    --start-frame  Skip the first N frames in filename-sorted order.
    --max-frames   Stop after this many publishes.
    --pattern      Glob pattern for camera files (default 'frame*.jpg').
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Optional

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from transport.nats_adapter import Adapter
from transport.proto import perception_pb2 as pb

log = logging.getLogger("publish_camera_stream")


# RELLIS filename: "frame001375-1581624790_249.jpg"
#                       ^^^^^^ 6-digit frame index
_FRAME_RE = re.compile(r"frame(\d+)[-_].*\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def list_frames(camera_dir: Path, pattern: str) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in camera_dir.glob(pattern):
        m = _FRAME_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    if not out:
        raise FileNotFoundError(
            f"no camera frames matching {pattern!r} in {camera_dir}"
        )
    out.sort(key=lambda t: t[0])
    return out


def build_camera_frame(*, frame_idx: int, jpeg_bytes: bytes,
                       width: int, height: int) -> pb.CameraFrame:
    msg = pb.CameraFrame()
    msg.header.timestamp = time.time()
    msg.header.frame_id = f"frame_{frame_idx:06d}"
    msg.header.schema_version = 1
    msg.header.source = "publish_camera_stream"
    msg.width = width
    msg.height = height
    msg.encoding = "jpeg"
    msg.data = jpeg_bytes
    return msg


# Optional: cheap dimension probe so the proto fields are accurate. Reads JPEG
# SOF marker without decoding pixels. Falls back to (0, 0) if unrecognized.
def jpeg_dimensions(buf: bytes) -> tuple[int, int]:
    i = 0
    n = len(buf)
    if n < 4 or buf[:2] != b"\xff\xd8":
        return (0, 0)
    i = 2
    while i < n - 9:
        if buf[i] != 0xFF:
            i += 1
            continue
        marker = buf[i + 1]
        # SOF0..SOF15 except DHT/DAC/DNL.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height = (buf[i + 5] << 8) | buf[i + 6]
            width = (buf[i + 7] << 8) | buf[i + 8]
            return (width, height)
        seg_len = (buf[i + 2] << 8) | buf[i + 3]
        i += 2 + seg_len
    return (0, 0)


async def stream(*, camera_dir: Path, subject: str, hz: float,
                 loop_forever: bool, nats_url: str,
                 start_frame: int, max_frames: Optional[int],
                 pattern: str) -> None:
    frames = [(idx, p) for idx, p in list_frames(camera_dir, pattern) if idx >= start_frame]
    log.info("loaded %d camera frames from %s", len(frames), camera_dir)
    log.info("publishing on subject %s at %.2f Hz", subject, hz)

    adapter = Adapter(nats_url)
    await adapter.connect()

    period = 1.0 / hz
    n_published = 0
    try:
        while True:
            for idx, path in frames:
                t0 = time.perf_counter()
                jpeg_bytes = path.read_bytes()
                w, h = jpeg_dimensions(jpeg_bytes)
                msg = build_camera_frame(frame_idx=idx, jpeg_bytes=jpeg_bytes,
                                         width=w, height=h)
                payload = msg.SerializeToString()
                await adapter.publish(subject, payload)
                n_published += 1
                if max_frames is not None and n_published >= max_frames:
                    log.info("reached --max-frames=%d, stopping", max_frames)
                    return
                elapsed = time.perf_counter() - t0
                if elapsed < period:
                    await asyncio.sleep(period - elapsed)
            if not loop_forever:
                log.info("camera sequence exhausted, exiting")
                return
            log.info("looping back to first camera frame")
    finally:
        await adapter.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-dir", type=Path, required=True)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--subject", default="sensor.camera.rgb")
    ap.add_argument("--nats-url", default="nats://localhost:4222")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--pattern", default="frame*.jpg")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    asyncio.run(stream(
        camera_dir=args.camera_dir,
        subject=args.subject,
        hz=args.hz,
        loop_forever=args.loop,
        nats_url=args.nats_url,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        pattern=args.pattern,
    ))


if __name__ == "__main__":
    main()
