"""
P2-M5.0d -- End-to-end latency budget.

Profiles YOLOv8n CPU inference time per frame on the actual RELLIS
camera image size. We do NOT benchmark cam-LiDAR projection or the
SORTTracker update here -- both are microsecond-scale C++ on the
detection counts we care about (a few persons per frame), and
benchmarking them properly would require building the C++ runners
which is Phase A territory. YOLO is the dominant CPU cost; document
that in the summary.

Pass/fail (per plan):
  GREEN  : >= 5 Hz sustained on laptop CPU.
  YELLOW : 2-5 Hz (stride demo: every other frame).
  RED    : < 2 Hz (offline pre-compute YOLO).

Outputs (default /media/nishant/SeeGayt2/terra_perceive/m5_preflight/0d_latency/):
  per_frame_ms.csv    inference time per frame (cold + warm)
  summary.json        mean / median / p95 latency, projected fps, verdict
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--frames-dir",
        default="data/RELLIS-3D/Rellis_3D_pylon_camera_node/Rellis-3D/00000/pylon_camera_node",
    )
    p.add_argument("--n-frames", type=int, default=30)
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument(
        "--weights",
        default="/media/nishant/SeeGayt2/terra_perceive/models/yolov8n.pt",
    )
    p.add_argument(
        "--out-dir",
        default="/media/nishant/SeeGayt2/terra_perceive/m5_preflight/0d_latency",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(Path(args.frames_dir).resolve().glob("frame*.jpg"))
    if not frames:
        print(f"No frames in {args.frames_dir}", file=sys.stderr)
        return 1

    sample = frames[: args.n_frames + args.n_warmup]
    imgs = [cv2.imread(str(p)) for p in sample]
    h, w = imgs[0].shape[:2]
    print(f"[0d] image size {w}x{h}, n_frames={args.n_frames}, imgsz={args.imgsz}")

    model = YOLO(args.weights)

    rows = []
    for k, img in enumerate(imgs):
        t0 = time.perf_counter()
        _ = model.predict(
            img, conf=0.25, imgsz=args.imgsz, device=args.device, verbose=False,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        rows.append(
            {"k": k, "is_warmup": k < args.n_warmup, "ms": dt_ms},
        )
        if k < 5 or k % 10 == 0:
            tag = "warm" if k < args.n_warmup else "    "
            print(f"  [{tag}][{k:3d}] {dt_ms:7.2f} ms")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_frame_ms.csv", index=False)

    warm = df[~df["is_warmup"]]["ms"].values
    mean_ms = float(np.mean(warm))
    median_ms = float(np.median(warm))
    p95_ms = float(np.percentile(warm, 95))
    fps_mean = 1000.0 / mean_ms

    if fps_mean >= 5.0:
        verdict = "GREEN"
    elif fps_mean >= 2.0:
        verdict = "YELLOW"
    else:
        verdict = "RED"

    summary = {
        "image_size_wxh": [int(w), int(h)],
        "imgsz_inference": args.imgsz,
        "device": args.device,
        "n_warmup": args.n_warmup,
        "n_measured": int(len(warm)),
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "projected_fps_from_mean": fps_mean,
        "verdict": verdict,
        "note": (
            "YOLOv8n inference only. Cam-LiDAR projection and SORTTracker "
            "update are C++ microsecond-scale on the per-frame detection "
            "counts here; not benchmarked separately."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(f"[0d] mean {mean_ms:.2f} ms, median {median_ms:.2f}, p95 {p95_ms:.2f}")
    print(f"[0d] projected fps: {fps_mean:.2f}  -> verdict: {verdict}")
    print(f"[0d] outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
