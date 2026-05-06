"""
P2-M5.0a — YOLO firing-rate audit on RELLIS.

Sample N evenly-spaced camera frames, run YOLOv8n at a loose
confidence threshold, and report:
  - histogram of detections-per-frame
  - top-K COCO classes by total fires
  - fraction of frames with at least one
    person/car/truck/bicycle/motorcycle at conf >= 0.5

Pass/fail (decided in the wondrous-crane plan, Phase 0a):
  GREEN   : fraction >= 0.30  -> RELLIS-primary headline
  YELLOW  : 0.05 <= frac < 0.30 -> RELLIS edge-case clip; Roboflow headline
  RED     : fraction < 0.05  -> RELLIS pipeline-only, no detection headline

Outputs (default /media/nishant/SeeGayt2/terra_perceive/m5_preflight/0a_yolo_firing/):
  per_frame.csv          row per sampled frame
  class_totals.csv       total detections per COCO class id
  histogram.png          detections-per-frame histogram
  summary.json           aggregate metrics + verdict
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# COCO class ids of interest for the M5 safety/perception story.
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    7: "truck",
}
HEADLINE_CONF = 0.5
SCAN_CONF = 0.25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--frames-dir",
        default="data/RELLIS-3D/Rellis_3D_pylon_camera_node/Rellis-3D/00000/pylon_camera_node",
        help="Directory of RELLIS camera jpgs.",
    )
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument(
        "--weights",
        default="/media/nishant/SeeGayt2/terra_perceive/models/yolov8n.pt",
        help="YOLOv8n weights path. Auto-downloaded by ultralytics if missing.",
    )
    p.add_argument(
        "--out-dir",
        default="/media/nishant/SeeGayt2/terra_perceive/m5_preflight/0a_yolo_firing",
    )
    p.add_argument("--device", default="cpu", help="cpu or 0 for first GPU.")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    frames_dir = Path(args.frames_dir).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames = sorted(frames_dir.glob("frame*.jpg"))
    if not all_frames:
        print(f"No frames found in {frames_dir}", file=sys.stderr)
        return 1

    n = min(args.n_samples, len(all_frames))
    idx = np.linspace(0, len(all_frames) - 1, num=n, dtype=int)
    sampled = [all_frames[i] for i in idx]
    print(
        f"[0a] frames_dir={frames_dir} total={len(all_frames)} sampled={n} "
        f"weights={args.weights}",
    )

    Path(args.weights).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    rows = []
    class_total: Counter[int] = Counter()
    target_hits_count = 0

    for i, frame_path in enumerate(sampled):
        # Load via cv2 to avoid ultralytics' silent path quirks.
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"  [skip] could not read {frame_path}", file=sys.stderr)
            continue

        result = model.predict(
            img,
            conf=SCAN_CONF,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        boxes = result.boxes
        per_frame_total = int(boxes.shape[0]) if boxes is not None else 0

        target_hit_in_frame = False
        if per_frame_total > 0:
            cls_ids = boxes.cls.cpu().numpy().astype(int).tolist()
            confs = boxes.conf.cpu().numpy().tolist()
            for c in cls_ids:
                class_total[c] += 1
            for c, conf in zip(cls_ids, confs):
                if c in TARGET_CLASSES and conf >= HEADLINE_CONF:
                    target_hit_in_frame = True
                    break

        if target_hit_in_frame:
            target_hits_count += 1

        rows.append(
            {
                "sample_idx": int(idx[i]),
                "frame_name": frame_path.name,
                "n_detections": per_frame_total,
                "target_hit_at_50": int(target_hit_in_frame),
            },
        )

        if (i + 1) % 25 == 0 or i + 1 == n:
            print(
                f"  [{i + 1}/{n}] last_dets={per_frame_total} "
                f"running_hit_frac={target_hits_count / (i + 1):.3f}",
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_frame.csv", index=False)

    # COCO-80 class names from ultralytics' bundled metadata.
    class_names = model.model.names if hasattr(model, "model") else {}
    cls_rows = [
        {"class_id": cid, "class_name": class_names.get(cid, str(cid)), "total": cnt}
        for cid, cnt in class_total.most_common()
    ]
    pd.DataFrame(cls_rows).to_csv(out_dir / "class_totals.csv", index=False)

    fraction = target_hits_count / max(len(df), 1)
    if fraction >= 0.30:
        verdict = "GREEN"
    elif fraction >= 0.05:
        verdict = "YELLOW"
    else:
        verdict = "RED"

    summary = {
        "n_sampled": int(len(df)),
        "n_total_frames": len(all_frames),
        "scan_conf": SCAN_CONF,
        "headline_conf": HEADLINE_CONF,
        "target_classes": TARGET_CLASSES,
        "frames_with_target_hit_at_50": int(target_hits_count),
        "target_hit_fraction": float(fraction),
        "verdict": verdict,
        "top_classes": cls_rows[:10],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Histogram.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["n_detections"], bins=range(0, max(df["n_detections"].max() + 2, 6)))
        ax.set_xlabel("Detections per frame (conf >= 0.25)")
        ax.set_ylabel("Frame count")
        ax.set_title(
            f"RELLIS YOLOv8n firing rate — {verdict} "
            f"(target-hit frac at 0.5 = {fraction:.2%})",
        )
        fig.tight_layout()
        fig.savefig(out_dir / "histogram.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # plotting is non-critical
        print(f"  [warn] histogram render failed: {exc}", file=sys.stderr)

    print()
    print(f"[0a] verdict: {verdict}")
    print(f"[0a] target-hit fraction (conf>=0.5): {fraction:.4f}")
    print(f"[0a] top 5 classes: {cls_rows[:5]}")
    print(f"[0a] outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
