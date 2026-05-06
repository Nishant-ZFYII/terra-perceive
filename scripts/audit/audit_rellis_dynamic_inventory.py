"""
P2-M5.0b — RELLIS dynamic-object inventory.

Note on method substitution from the wondrous-crane plan:
The plan's original 0b criterion was "M13.5 tracker velocity > 1 m/s
for >= 10 frames". That tracker output (per-track CSV) was never
persisted to disk in this repo, only summary metrics.json files. We
do NOT re-run M13.5 here -- the plan's anti-celebration guardrails
forbid touching M13.5 territory. Substitute method:

  1) Dense YOLOv8n scan over a configurable frame window (default
     0..500, where 0a clustered the target hits).
  2) Find the longest contiguous run of frames with at least one
     person detection at conf >= 0.5.
  3) Cross-check ego speed from poses_carto.csv in that run.
  4) Save annotated thumbnails for visual confirmation that the
     person is real (not a tree / signpost false fire).

Pass/fail (per plan):
  GREEN: contiguous run of >= 50 frames with confirmed person
         visible -- use as RELLIS dynamic-filter B3 demo segment.
  RED  : no such run -> dynamic-filter sub-goal moves entirely to
         Roboflow (Phase C3).

Outputs (default /media/nishant/SeeGayt2/terra_perceive/m5_preflight/0b_dynamic/):
  per_frame_dense.csv      detections per frame in the scan window
  contiguous_runs.csv      every run of consecutive person-firing frames
  ego_speed_in_run.csv     ego speed (carto) per frame in the longest run
  thumbnails/              annotated jpgs of representative frames
  summary.json             aggregate + verdict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

PERSON_CLS = 0
VEHICLE_CLSES = {1, 2, 3, 7}  # bicycle, car, motorcycle, truck
HEADLINE_CONF = 0.5
SCAN_CONF = 0.25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--frames-dir",
        default="data/RELLIS-3D/Rellis_3D_pylon_camera_node/Rellis-3D/00000/pylon_camera_node",
    )
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=500)
    p.add_argument(
        "--weights",
        default="/media/nishant/SeeGayt2/terra_perceive/models/yolov8n.pt",
    )
    p.add_argument("--poses-csv", default="data/poses_carto.csv")
    p.add_argument(
        "--out-dir",
        default="/media/nishant/SeeGayt2/terra_perceive/m5_preflight/0b_dynamic",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def longest_contiguous(mask: np.ndarray) -> tuple[int, int, int]:
    """Return (start, end_exclusive, length) of longest True run in mask."""
    best = (0, 0, 0)
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i > best[2]:
            best = (i, j, j - i)
        i = j
    return best


def all_runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        runs.append((i, j, j - i))
        i = j
    return runs


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "thumbnails").mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir).resolve()
    all_frames = sorted(frames_dir.glob("frame*.jpg"))
    if not all_frames:
        print(f"No frames in {frames_dir}", file=sys.stderr)
        return 1

    end = min(args.end_frame, len(all_frames))
    window = all_frames[args.start_frame : end]
    print(
        f"[0b] dense scan frames {args.start_frame}..{end} "
        f"(n={len(window)}) weights={args.weights}",
    )

    model = YOLO(args.weights)
    rows = []
    for k, fp in enumerate(window):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        result = model.predict(
            img, conf=SCAN_CONF, imgsz=args.imgsz, device=args.device, verbose=False,
        )[0]

        person_conf_max = 0.0
        person_centers: list[tuple[float, float]] = []
        vehicle_count = 0
        if result.boxes is not None and result.boxes.shape[0] > 0:
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            xywh = result.boxes.xywh.cpu().numpy()
            for cid, cf, box in zip(cls_ids, confs, xywh):
                if cid == PERSON_CLS and cf >= HEADLINE_CONF:
                    person_conf_max = max(person_conf_max, float(cf))
                    person_centers.append((float(box[0]), float(box[1])))
                if cid in VEHICLE_CLSES and cf >= HEADLINE_CONF:
                    vehicle_count += 1

        rows.append(
            {
                "frame_idx": args.start_frame + k,
                "frame_name": fp.name,
                "n_person_at_50": len(person_centers),
                "person_conf_max": person_conf_max,
                "n_vehicle_at_50": vehicle_count,
                "person_cx": person_centers[0][0] if person_centers else np.nan,
                "person_cy": person_centers[0][1] if person_centers else np.nan,
            },
        )

        if (k + 1) % 50 == 0:
            print(f"  [{k + 1}/{len(window)}]")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_frame_dense.csv", index=False)

    person_mask = (df["n_person_at_50"] > 0).to_numpy()
    runs = all_runs(person_mask)
    runs_df = pd.DataFrame(
        [{"start_idx": s, "end_idx": e, "length": L} for s, e, L in runs],
    )
    runs_df.to_csv(out_dir / "contiguous_runs.csv", index=False)

    longest = longest_contiguous(person_mask)
    print(
        f"[0b] longest person-run: idx_in_window {longest[0]}..{longest[1]} "
        f"(length {longest[2]}); maps to frame "
        f"{args.start_frame + longest[0]}..{args.start_frame + longest[1]}",
    )

    # Ego-speed cross-check across the longest run.
    ego_summary = {"available": False}
    if longest[2] > 0 and Path(args.poses_csv).exists():
        poses = pd.read_csv(args.poses_csv)
        # poses use a 0..N row index = lidar frame index per P2-M1.
        run_start = args.start_frame + longest[0]
        run_end = args.start_frame + longest[1]
        if run_end <= len(poses):
            sub = poses.iloc[run_start:run_end].copy()
            sub["dt_s"] = np.r_[
                np.nan, np.diff(sub["timestamp"].values) / 1e9,
            ]
            dx = np.r_[np.nan, np.diff(sub["x"].values)]
            dy = np.r_[np.nan, np.diff(sub["y"].values)]
            sub["speed_mps"] = np.sqrt(dx**2 + dy**2) / sub["dt_s"]
            sub.to_csv(out_dir / "ego_speed_in_run.csv", index=False)
            ego_summary = {
                "available": True,
                "frame_start": int(run_start),
                "frame_end": int(run_end),
                "speed_mps_mean": float(np.nanmean(sub["speed_mps"])),
                "speed_mps_max": float(np.nanmax(sub["speed_mps"])),
                "speed_mps_min": float(np.nanmin(sub["speed_mps"])),
            }
            print(
                f"[0b] ego speed in run: mean "
                f"{ego_summary['speed_mps_mean']:.3f} m/s, "
                f"max {ego_summary['speed_mps_max']:.3f}",
            )

    # Save 6 annotated thumbnails spaced across the longest run.
    if longest[2] > 0:
        idx_in_window = np.linspace(
            longest[0], longest[1] - 1, num=min(6, longest[2]), dtype=int,
        )
        for k in idx_in_window:
            fp = window[k]
            img = cv2.imread(str(fp))
            result = model.predict(
                img, conf=SCAN_CONF, imgsz=args.imgsz, device=args.device, verbose=False,
            )[0]
            ann = result.plot()
            out_path = out_dir / "thumbnails" / f"{args.start_frame + k:06d}_{fp.stem}.jpg"
            cv2.imwrite(str(out_path), ann)

    verdict = "GREEN" if longest[2] >= 50 else "RED"
    summary = {
        "scan_start": args.start_frame,
        "scan_end": end,
        "n_frames_scanned": int(len(df)),
        "n_frames_with_person": int(person_mask.sum()),
        "n_frames_with_vehicle": int((df["n_vehicle_at_50"] > 0).sum()),
        "longest_person_run_length": int(longest[2]),
        "longest_person_run_frame_range": [
            int(args.start_frame + longest[0]),
            int(args.start_frame + longest[1]),
        ],
        "ego_speed": ego_summary,
        "verdict": verdict,
        "verdict_threshold_frames": 50,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[0b] verdict: {verdict}")
    print(f"[0b] outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
