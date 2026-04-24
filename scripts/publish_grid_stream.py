"""
publish_grid_stream.py — replay accumulator snapshots through NATS as a fake
live BEVGrid stream.

Reads the sparse-cell snapshot CSVs produced by accumulator_runner
(`results/m3/<run>/snapshots/frame_*.csv`), rehydrates each into a dense grid,
wraps it in a BEVGrid protobuf, and publishes it on
`perception.traversability.grid` at the requested rate.

Used for the M3 "transport bootstrap" demo: pair this with
scripts/dashboard_subscriber.py to record a live screen capture of the BEV
map filling in over NATS while the offline run is fully done. The map you see
on screen IS the offline run, replayed.

Usage:
    python scripts/publish_grid_stream.py --run results/m3/slam_ema_full/
    python scripts/publish_grid_stream.py --run results/m3/slam_ema_covg2o_full/ \\
        --hz 5 --loop --nats-url nats://localhost:4222

CLI flags:
    --run             Directory containing snapshots/ and pose_sigma.csv.
    --hz              Publish rate (default 5).
    --loop            Loop the snapshot sequence forever.
    --subject         NATS subject (default perception.traversability.grid).
    --nats-url        NATS server URL (default nats://localhost:4222).
    --start-frame     Start from this frame index (default 0).
    --max-frames      Stop after this many publishes (default unlimited).
    --origin-x,
    --origin-y        World origin in metres for the BEVGrid.origin_{x,y}
                      fields. Defaults to centred at -resolution * cols/2.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import re
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Local imports — adjust path so `transport.` is importable when running from
# the repo root.
import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from transport.nats_adapter import Adapter
from transport.proto import perception_pb2 as pb

log = logging.getLogger("publish_grid_stream")


# --------------------------------------------------------------------------- #
# Snapshot-CSV parsing
# --------------------------------------------------------------------------- #
_HEADER_RE = re.compile(r"#\s*rows\s*=\s*(\d+),\s*cols\s*=\s*(\d+),\s*resolution\s*=\s*([\d.]+)")
_FRAME_RE = re.compile(r"frame_(\d+)\.csv$")


def parse_snapshot(path: Path) -> tuple[int, int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (rows, cols, resolution, risk, confidence, obs_count, mask) as
    dense arrays of shape (rows, cols).
    Cells absent from the sparse CSV stay zero (and unobserved in mask)."""
    with path.open() as f:
        header = f.readline()
        m = _HEADER_RE.match(header)
        if not m:
            raise ValueError(f"{path}: missing dimension header on line 1")
        rows, cols = int(m.group(1)), int(m.group(2))
        resolution = float(m.group(3))

        risk = np.zeros((rows, cols), dtype=np.float32)
        confidence = np.zeros((rows, cols), dtype=np.float32)
        obs_count = np.zeros((rows, cols), dtype=np.uint32)

        reader = csv.DictReader(f)
        for row in reader:
            r = int(row["row"])
            c = int(row["col"])
            risk[r, c] = float(row["risk"])
            confidence[r, c] = float(row["confidence"])
            obs_count[r, c] = int(row["obs_count"])

    mask = (obs_count > 0).astype(np.uint8)
    return rows, cols, resolution, risk, confidence, obs_count, mask


def load_pose_sigma(run_dir: Path) -> dict[int, float]:
    """Return frame_id -> pose_sigma. Empty dict if file missing."""
    p = run_dir / "pose_sigma.csv"
    if not p.exists():
        log.warning("no pose_sigma.csv in %s — pose_sigma_at_snapshot will be 0", run_dir)
        return {}
    out: dict[int, float] = {}
    with p.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out[int(row["frame_id"])] = float(row["pose_sigma"])
            except (KeyError, ValueError):
                continue
    return out


def list_snapshots(run_dir: Path) -> list[tuple[int, Path]]:
    snap_dir = run_dir / "snapshots"
    if not snap_dir.is_dir():
        raise FileNotFoundError(f"no snapshots/ subdir in {run_dir}")
    out = []
    for p in sorted(snap_dir.glob("frame_*.csv")):
        m = _FRAME_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    if not out:
        raise FileNotFoundError(f"no frame_*.csv files in {snap_dir}")
    return out


# --------------------------------------------------------------------------- #
# Build BEVGrid proto from numpy arrays
# --------------------------------------------------------------------------- #
def build_bevgrid(
    *,
    frame_id: int,
    resolution: float,
    rows: int,
    cols: int,
    origin_x: float,
    origin_y: float,
    risk: np.ndarray,
    confidence: np.ndarray,
    obs_count: np.ndarray,
    mask: np.ndarray,
    pose_sigma: float,
) -> pb.BEVGrid:
    msg = pb.BEVGrid()
    msg.header.timestamp = time.time()
    msg.header.frame_id = f"frame_{frame_id:06d}"
    msg.header.schema_version = 1
    msg.header.source = "publish_grid_stream"

    msg.resolution = resolution
    msg.width = cols
    msg.height = rows
    msg.origin_x = origin_x
    msg.origin_y = origin_y

    # Row-major flatten, matching the convention WorldGrid uses.
    msg.scores.extend(risk.ravel().tolist())
    msg.confidence.extend(confidence.ravel().tolist())
    msg.observation_count.extend(obs_count.ravel().tolist())
    msg.coverage_mask = np.packbits(mask.ravel()).tobytes()
    msg.pose_sigma_at_snapshot = float(pose_sigma)
    return msg


# --------------------------------------------------------------------------- #
# Main publish loop
# --------------------------------------------------------------------------- #
async def stream(
    *,
    run_dir: Path,
    subject: str,
    hz: float,
    loop_forever: bool,
    nats_url: str,
    start_frame: int,
    max_frames: Optional[int],
    origin_x: Optional[float],
    origin_y: Optional[float],
) -> None:
    snaps = [(fid, p) for fid, p in list_snapshots(run_dir) if fid >= start_frame]
    sigmas = load_pose_sigma(run_dir)

    log.info("loaded %d snapshots from %s", len(snaps), run_dir / "snapshots")
    log.info("publishing on subject %s at %.2f Hz", subject, hz)

    adapter = Adapter(nats_url)
    await adapter.connect()

    period = 1.0 / hz
    n_published = 0
    try:
        while True:
            for fid, path in snaps:
                t0 = time.perf_counter()

                rows, cols, resolution, risk, conf, obs, mask = parse_snapshot(path)
                ox = origin_x if origin_x is not None else -resolution * cols / 2.0
                oy = origin_y if origin_y is not None else -resolution * rows / 2.0
                pose_sigma = sigmas.get(fid, 0.0)

                msg = build_bevgrid(
                    frame_id=fid,
                    resolution=resolution,
                    rows=rows,
                    cols=cols,
                    origin_x=ox,
                    origin_y=oy,
                    risk=risk,
                    confidence=conf,
                    obs_count=obs,
                    mask=mask,
                    pose_sigma=pose_sigma,
                )
                payload = msg.SerializeToString()
                await adapter.publish(subject, payload)
                n_published += 1

                if max_frames is not None and n_published >= max_frames:
                    log.info("reached --max-frames=%d, stopping", max_frames)
                    return

                # pace to maintain --hz
                elapsed = time.perf_counter() - t0
                if elapsed < period:
                    await asyncio.sleep(period - elapsed)

            if not loop_forever:
                log.info("snapshot sequence exhausted, exiting")
                return
            log.info("looping back to first snapshot")
    finally:
        await adapter.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="results/m3/<run-dir>/")
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--loop", action="store_true", help="loop forever")
    ap.add_argument("--subject", default="perception.traversability.grid")
    ap.add_argument("--nats-url", default="nats://localhost:4222")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--origin-x", type=float, default=None)
    ap.add_argument("--origin-y", type=float, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    asyncio.run(stream(
        run_dir=args.run,
        subject=args.subject,
        hz=args.hz,
        loop_forever=args.loop,
        nats_url=args.nats_url,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        origin_x=args.origin_x,
        origin_y=args.origin_y,
    ))


if __name__ == "__main__":
    main()
