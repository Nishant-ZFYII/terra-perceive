"""
tracker_bridge.py — Python NATS sidecar for the C++ tracker_node binary
(P2-M5.A2 step 3).

Architecture (debug log 2026-05-01):

    perception.objects.detections (NATS)
                  │
                  ▼
        +---------------------+        write 4B length + DetectionList
        | tracker_bridge.py   | ─────────────────────────────────────►
        |  (this module)      |                                stdin
        |                     |        read 4B length + TrackList
        |                     | ◄─────────────────────────────────────
        +---------------------+                                stdout
                  │
                  ▼
    perception.objects.tracks (NATS)

The bridge spawns the `tracker_node` C++ binary as a long-running child,
shoves serialized DetectionList bytes into its stdin (length-prefixed),
reads serialized TrackList bytes from its stdout (length-prefixed),
and re-publishes those on NATS. The C++ side is unchanged from how it
was run from the smoke test — just stdin/stdout.

Mentor-mode tag: pure glue. No algorithmic content the user is expected
to fill in.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import struct
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from transport.nats_adapter import Adapter  # noqa: E402

log = logging.getLogger("tracker_bridge")

DEFAULT_BINARY = "build/construction_perception/tracker_node"
DEFAULT_SUBJECT_IN = "perception.objects.detections"
DEFAULT_SUBJECT_OUT = "perception.objects.tracks"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--binary", default=DEFAULT_BINARY,
                   help="Path to compiled tracker_node executable.")
    p.add_argument("--nats-url", default="nats://localhost:4222")
    p.add_argument("--subject-in", default=DEFAULT_SUBJECT_IN)
    p.add_argument("--subject-out", default=DEFAULT_SUBJECT_OUT)
    p.add_argument("--calib", default="config/camera_lidar_calib.yaml")
    p.add_argument(
        "--lidar-dir",
        default="data/RELLIS-3D/Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/00000/os1_cloud_node_kitti_bin",
    )
    p.add_argument("--max-dist", type=float, default=5.0)
    p.add_argument("--max-misses", type=int, default=10)
    p.add_argument("--min-hits", type=int, default=1)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--process-noise", type=float, default=2.0)
    p.add_argument("--meas-noise", type=float, default=0.3)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def build_subprocess_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        args.binary,
        "--calib", args.calib,
        "--lidar-dir", args.lidar_dir,
        "--max-dist", str(args.max_dist),
        "--max-misses", str(args.max_misses),
        "--min-hits", str(args.min_hits),
        "--dt", str(args.dt),
        "--process-noise", str(args.process_noise),
        "--meas-noise", str(args.meas_noise),
    ]
    if args.verbose:
        argv.append("--verbose")
    return argv


async def reader_loop(proc: asyncio.subprocess.Process, pub: Adapter,
                      subject_out: str) -> None:
    """Read length-prefixed TrackList frames from the child's stdout and
    republish them on NATS until the child closes its stdout."""
    stdout = proc.stdout
    assert stdout is not None
    while True:
        try:
            hdr = await stdout.readexactly(4)
        except asyncio.IncompleteReadError:
            log.info("child stdout closed (EOF)")
            return
        (n,) = struct.unpack(">I", hdr)
        try:
            payload = await stdout.readexactly(n)
        except asyncio.IncompleteReadError:
            log.warning("partial TrackList payload at EOF; dropping")
            return
        await pub.publish(subject_out, payload)


async def stderr_drain(proc: asyncio.subprocess.Process) -> None:
    """Mirror the child's stderr to ours so its --verbose metrics surface."""
    stderr = proc.stderr
    assert stderr is not None
    while True:
        line = await stderr.readline()
        if not line:
            return
        sys.stderr.write("[tracker_node] " + line.decode(errors="replace"))
        sys.stderr.flush()


async def run(args: argparse.Namespace) -> int:
    if not Path(args.binary).is_file() or not os.access(args.binary, os.X_OK):
        log.error("binary not found or not executable: %s", args.binary)
        log.error("Build it first:  colcon build --packages-select construction_perception")
        return 2

    subscriber = Adapter(args.nats_url)
    publisher = Adapter(args.nats_url)
    await subscriber.connect()
    await publisher.connect()

    log.info("spawning %s", " ".join(build_subprocess_argv(args)))
    proc = await asyncio.create_subprocess_exec(
        *build_subprocess_argv(args),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    reader_task = asyncio.create_task(reader_loop(proc, publisher, args.subject_out))
    stderr_task = asyncio.create_task(stderr_drain(proc))

    async def on_detections(payload: bytes) -> None:
        # Forward verbatim into the C++ binary's stdin. The bridge does
        # NOT inspect the DetectionList payload; the C++ side parses it.
        if proc.stdin is None or proc.returncode is not None:
            log.warning("child stdin closed; dropping detection frame")
            return
        proc.stdin.write(struct.pack(">I", len(payload)))
        proc.stdin.write(payload)
        await proc.stdin.drain()

    await subscriber.subscribe(args.subject_in, on_detections)
    log.info("bridge live: in=%s  out=%s  child_pid=%d",
             args.subject_in, args.subject_out, proc.pid)

    # Run until cancelled (Ctrl-C) or the child exits unexpectedly.
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    try:
        await stop.wait()
    finally:
        log.info("shutting down")
        if proc.stdin is not None and proc.returncode is None:
            try:
                proc.stdin.close()
            except Exception:
                pass
        await reader_task
        await stderr_task
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        await subscriber.close()
        await publisher.close()

    return proc.returncode or 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    args = parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
