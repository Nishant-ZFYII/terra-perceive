#!/usr/bin/env python3
"""Synthetic detection-CSV generator for SORT tracker tests and ablations.

Usage:
    python scripts/synth_detections.py linear   --frames 30 --out tests/data/linear.csv
    python scripts/synth_detections.py crossing --frames 30 --out tests/data/crossing.csv
    python scripts/synth_detections.py occluded --frames 30 --out tests/data/occluded.csv
    python scripts/synth_detections.py spurious --frames 30 --out tests/data/spurious.csv
    python scripts/synth_detections.py dense    --frames 30 --out tests/data/dense.csv

CSV schema (consumed by tracker_runner --detections):
    frame_id,det_id,x,y,class_id,gt_track_id

`gt_track_id` is the ground-truth track label for MOT evaluation. Spurious
detections and noise points have gt_track_id = -1.

Mentor-mode split:
    - argparse, dispatch, CSV writer = pure glue (this file's bottom half).
    - Each scenario's geometry = algorithmic; YOUR CODE blocks below.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List


# -----------------------------------------------------------------------------
# Detection record (rows of the output CSV).
# -----------------------------------------------------------------------------
@dataclass
class Det:
    frame_id: int
    det_id: int
    x: float
    y: float
    class_id: int = 0
    gt_track_id: int = -1   # -1 = no ground-truth label (spurious/noise)


# =============================================================================
# Scenarios — YOUR CODE
# =============================================================================
# Each generator returns a flat list of Det records across all frames.
# Use a deterministic random seed (rng) so ablations are reproducible.
#
# Suggested coordinate units: meters (matches RELLIS/LiDAR scale).

def gen_linear(frames: int, rng: random.Random) -> List[Det]:
    """One target moving in a straight line, with isotropic measurement noise.

    Used by Ablations B (process-noise Q sweep) and C (measurement-noise R sweep).

    Geometry:
      Start  p0 = (1.0, 2.0)
      Vel    v  = (1.0, 0.6) per frame
      Noise sigma = 0.15 m on each axis (deliberately VISIBLE so the Q sweep
                    can show low-Q smoothing vs high-Q jitter on the plot —
                    smaller noise would compress the visual story)

    Output: one detection per frame, gt_track_id = 0.
    """
    p0 = (1.0, 2.0)
    v = (1.0, 0.6)
    # sigma deliberately large enough to be VISIBLY noisy on the Q-sweep
    # plot — small sigma compresses the smoothing-vs-jitter visual story.
    # 0.5m on a 1m/frame trajectory makes dets cluster around the line but
    # individually visible.
    sigma = 0.5
    rows: List[Det] = []
    for i in range(frames):
        x = p0[0] + v[0] * i + rng.gauss(0.0, sigma)
        y = p0[1] + v[1] * i + rng.gauss(0.0, sigma)
        rows.append(Det(frame_id=i, det_id=0, x=x, y=y,
                        class_id=0, gt_track_id=0))
    return rows


def gen_crossing(frames: int, rng: random.Random) -> List[Det]:
    """Two targets crossing at right angles. Headline scenario for Ablation A.

    Geometry:
      Track 0: starts at (0.0, 5.0), velocity (+1.0, 0.0) m/frame.
      Track 1: starts at (5.0, 0.0), velocity ( 0.0, +1.0) m/frame.
      Both pass through (5.0, 5.0) on frame 5.
      Small Gaussian noise (~0.05 m std) on each detection so the cost matrix
      at the crossing has near-ties — the regime greedy is most likely to bite.

    Per frame i in [0, frames):
      Det(frame_id=i, det_id=0, x, y, class_id=0, gt_track_id=0)
      Det(frame_id=i, det_id=1, x, y, class_id=0, gt_track_id=1)

    The interview-defensible justification: this is the textbook two-target
    crossing scenario from Bewley 2016 Section 3, used to demonstrate that
    a tracker preserving identity through ambiguous association is doing real
    work. Noise level chosen so neither solver hits a matching gating cutoff
    at max_dist=3.0; the differentiator is purely the assignment algorithm.
    """
    sigma = 0.05  # detection noise (m)
    # Track 1 is offset by 0.5 m in x so the trajectories pass near each other
    # but never coincide. Without the offset, the cost matrix at the "crossing"
    # frame is all-zeros and tie-breaking inside the evaluator's Munkres
    # association produces phantom ID switches that aren't real tracker swaps.
    track1_x_offset = 0.5
    rows: List[Det] = []
    for i in range(frames):
        # Track 0 — moving +x along y=5
        x0 = 0.0 + 1.0 * i              + rng.gauss(0.0, sigma)
        y0 = 5.0                        + rng.gauss(0.0, sigma)
        rows.append(Det(frame_id=i, det_id=0, x=x0, y=y0,
                        class_id=0, gt_track_id=0))
        # Track 1 — moving +y along x = 5 + offset
        x1 = 5.0 + track1_x_offset      + rng.gauss(0.0, sigma)
        y1 = 0.0 + 1.0 * i              + rng.gauss(0.0, sigma)
        rows.append(Det(frame_id=i, det_id=1, x=x1, y=y1,
                        class_id=0, gt_track_id=1))
    return rows


def gen_occluded(frames: int, rng: random.Random) -> List[Det]:
    """One linear target with a contiguous detection gap (simulated occlusion).

    Used by Ablation F (max_misses sweep) — demonstrates how track persistence
    through occlusion depends on max_misses. Low max_misses prunes the track
    and a NEW id is assigned on resume; high max_misses keeps the same id.

    Geometry:
      Start  p0 = (0.0, 5.0)
      Vel    v  = (1.0, 0.0) per frame
      Noise sigma = 0.10 m
      Occlusion: contiguous gap [occ_start, occ_end). Default 8 frames at the
        middle of the run — long enough to challenge max_misses=3 and short
        enough that max_misses=10 should still recover.

    Output: detection rows ONLY on non-occluded frames, gt_track_id = 0.
    The runner sees no detections at all on frames inside the gap.
    """
    p0 = (0.0, 5.0)
    v = (1.0, 0.0)
    sigma = 0.10
    occ_start = max(1, frames // 3)              # ~ frame 10 for frames=30
    occ_end   = min(frames, occ_start + 8)        # 8-frame gap

    rows: List[Det] = []
    for i in range(frames):
        if occ_start <= i < occ_end:
            continue   # no detection emitted -> tracker sees an empty frame
        x = p0[0] + v[0] * i + rng.gauss(0.0, sigma)
        y = p0[1] + v[1] * i + rng.gauss(0.0, sigma)
        rows.append(Det(frame_id=i, det_id=0, x=x, y=y,
                        class_id=0, gt_track_id=0))
    return rows


def gen_spurious(frames: int, rng: random.Random) -> List[Det]:
    """One persistent linear target plus a few single-frame false positives.

    Used by Ablation E (min_hits sweep) — demonstrates how min_hits suppresses
    flickering false positives at the cost of init latency for true tracks.
    Low min_hits → publishes spurious tracks. High min_hits → clean output but
    slow to confirm new real tracks.

    Geometry:
      Persistent target: p0 = (0.0, 5.0), v = (1.0, 0.0), sigma = 0.10
      Spurious detections at frames {5, 12, 20}, each at a faraway position
      so they CANNOT be associated to the persistent track (max_dist gates
      them off into NEW track candidates).

    Output: one persistent gt_track_id=0 detection per frame; three additional
    rows with gt_track_id=-1 at the spurious frames. Spurious dets carry
    det_id=1 (within-frame numbering, not a track label).
    """
    p0 = (0.0, 5.0)
    v = (1.0, 0.0)
    sigma = 0.10
    spurious_frames = {5, 12, 20}
    spurious_offsets = [(15.0, 15.0), (-10.0, 8.0), (8.0, -12.0)]

    rows: List[Det] = []
    spur_idx = 0
    for i in range(frames):
        # Persistent target (always emitted).
        x = p0[0] + v[0] * i + rng.gauss(0.0, sigma)
        y = p0[1] + v[1] * i + rng.gauss(0.0, sigma)
        rows.append(Det(frame_id=i, det_id=0, x=x, y=y,
                        class_id=0, gt_track_id=0))
        # Optional spurious false-positive at this frame.
        if i in spurious_frames and spur_idx < len(spurious_offsets):
            ox, oy = spurious_offsets[spur_idx]
            spur_idx += 1
            rows.append(Det(frame_id=i, det_id=1,
                            x=ox + rng.gauss(0.0, sigma),
                            y=oy + rng.gauss(0.0, sigma),
                            class_id=0, gt_track_id=-1))
    return rows


def gen_maneuver(frames: int, rng: random.Random) -> List[Det]:
    """One linear target that REVERSES velocity at the midpoint.

    Used by Ablation H — the integration-level visualization of the
    predict/update ordering bug exposed by the unit test
    KalmanFilter.UpdateOrderMatters.

    Geometry:
      p0    = (1.0, 2.0)
      Phase 1 (frames 0..N/2):     v1 = (+1.0, +0.6) per frame
      Phase 2 (frames N/2..N):     v2 = (-1.0, -0.6) per frame  ← sharp reversal
      sigma = 0.5 m noise per axis

    The reversal forces the filter to update its velocity estimate at the
    turn frame. UpdateThenPredict's stale-prediction problem becomes
    visible because the predict at the END of the turn frame uses the
    pre-turn velocity, producing a residual spike that PredictThenUpdate
    avoids.
    """
    p0 = (1.0, 2.0)
    v1 = (1.0, 0.6)
    v2 = (-1.0, -0.6)
    sigma = 0.5
    turn_frame = frames // 2

    rows: List[Det] = []
    pos = list(p0)
    for i in range(frames):
        v = v1 if i < turn_frame else v2
        pos[0] += v[0]
        pos[1] += v[1]
        x = pos[0] + rng.gauss(0.0, sigma)
        y = pos[1] + rng.gauss(0.0, sigma)
        rows.append(Det(frame_id=i, det_id=0, x=x, y=y,
                        class_id=0, gt_track_id=0))
    return rows


def gen_dense(frames: int, rng: random.Random) -> List[Det]:
    """K targets moving through a shared region. Stress test for the matcher.

    Used by the deferred dense-scene Ablation A variant where greedy is more
    likely to ID-swap than on simple 2-target crossings (more near-tied costs
    in the matrix). K=10 by default.

    Geometry:
      K = 10 targets, each with a random start in [0, 20]^2 and a random
      velocity in [-1, 1]^2 m/frame. Trajectories are linear; some pairs
      will pass close together over the run, producing the near-ties.

      sigma = 0.15 m measurement noise on each target.

      drop_prob = 0.05: each detection has a 5% chance of being dropped per
      frame, simulating detector flicker. Tracks must rely on Kalman predict
      to bridge.

    Output: up to K detections per frame, gt_track_id = target index.
    """
    K = 10
    sigma = 0.15
    drop_prob = 0.05

    starts = [(rng.uniform(0.0, 20.0), rng.uniform(0.0, 20.0)) for _ in range(K)]
    vels   = [(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)) for _ in range(K)]

    rows: List[Det] = []
    for i in range(frames):
        next_det_id = 0
        for k in range(K):
            if rng.random() < drop_prob:
                continue
            x = starts[k][0] + vels[k][0] * i + rng.gauss(0.0, sigma)
            y = starts[k][1] + vels[k][1] * i + rng.gauss(0.0, sigma)
            rows.append(Det(frame_id=i, det_id=next_det_id,
                            x=x, y=y, class_id=0, gt_track_id=k))
            next_det_id += 1
    return rows


# =============================================================================
# Glue — argparse, CSV writer, dispatch (mine)
# =============================================================================

SCENARIOS = {
    "linear":   gen_linear,
    "crossing": gen_crossing,
    "occluded": gen_occluded,
    "spurious": gen_spurious,
    "dense":    gen_dense,
    "maneuver": gen_maneuver,
}


def write_csv(rows: List[Det], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "det_id", "x", "y", "class_id", "gt_track_id"])
        for r in rows:
            w.writerow([
                r.frame_id, r.det_id,
                f"{r.x:.6f}", f"{r.y:.6f}",
                r.class_id, r.gt_track_id,
            ])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", choices=sorted(SCENARIOS.keys()),
                   help="which synthetic scene to generate")
    p.add_argument("--frames", type=int, default=30,
                   help="number of frames (default: 30)")
    p.add_argument("--seed",   type=int, default=42,
                   help="rng seed; same scenario+seed → identical CSV")
    p.add_argument("--out",    type=Path, required=True,
                   help="output CSV path")
    args = p.parse_args()

    rng = random.Random(args.seed)
    rows = SCENARIOS[args.scenario](args.frames, rng)

    if not rows:
        print(f"[synth_detections] WARN: scenario '{args.scenario}' "
              f"returned 0 rows — did you fill in the YOUR CODE block?")
        return 1

    write_csv(rows, args.out)
    print(f"[synth_detections] wrote {len(rows)} detections "
          f"across {args.frames} frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
