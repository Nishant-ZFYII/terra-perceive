"""
run_b_ext_gt_eval.py — P2-M5 B-ext: YOLOv8n vs RELLIS-3D GT detection eval.

Evaluates YOLOv8n-COCO's person-class detection performance on the M5
demo segment against RELLIS-3D's pixel-wise ground-truth annotations.

Why this exists:
  Phase B's headline result was "1 stable track over 157 frames; 0 safety
  events because the dynamics were benign." That doesn't tell us how good
  the *detector* was. B-ext quantifies that: precision/recall/mAP@0.5 of
  YOLOv8n-COCO's person predictions vs RELLIS GT person silhouettes,
  evaluated on the 79 even-numbered annotated frames in window 84..240.

Method:
  1. For each annotated frame:
     - Load PNG mask (1920x1200 uint8 per-pixel class id).
     - Threshold to class 17 (person), drop noise (component area < 200 px).
     - cv2.connectedComponentsWithStats → tight bboxes per person instance.
  2. For each annotated frame:
     - Run YOLOv8n at conf 0.25 (Phase B's default scan threshold).
     - Keep class 0 (person) detections at conf >= 0.5 (Phase B headline conf).
  3. Pair YOLO predictions with GT bboxes via greedy IoU matching at threshold 0.5.
     - matched -> TP, unmatched YOLO -> FP, unmatched GT -> FN.
  4. Compute precision = TP/(TP+FP), recall = TP/(TP+FN), F1.
  5. Compute mAP@0.5 by ranking YOLO predictions by confidence and integrating
     precision-recall curve at IoU threshold 0.5.
  6. Write results_m5/phase_b/b_ext_metrics.json + per-frame CSV.

Stays purely on RELLIS data already on disk. No new datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

log = logging.getLogger("b_ext")

PERSON_CLASS = 17       # RELLIS class id for person
COCO_PERSON = 0         # YOLO COCO class id for person
HEADLINE_CONF = 0.5
SCAN_CONF = 0.25
MIN_GT_AREA = 200       # pixels; drop GT components smaller than this


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-frame", type=int, default=84)
    p.add_argument("--end-frame", type=int, default=241)
    p.add_argument(
        "--label-dir",
        default="data/RELLIS-3D/Rellis_3D_pylon_camera_node_label_id/Rellis-3D/00000/pylon_camera_node_label_id",
    )
    p.add_argument(
        "--frames-dir",
        default="data/RELLIS-3D/Rellis_3D_pylon_camera_node/Rellis-3D/00000/pylon_camera_node",
    )
    p.add_argument("--weights", default="/media/nishant/SeeGayt2/terra_perceive/models/yolov8n.pt")
    p.add_argument("--out-dir", default="results_m5/phase_b")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    return p.parse_args()


def gt_person_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Return list of (x1, y1, x2, y2, area) for person instances."""
    bin_mask = (mask == PERSON_CLASS).astype(np.uint8)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    out = []
    for i in range(1, n):  # skip background
        x, y, w, h, area = stats[i]
        if area < MIN_GT_AREA:
            continue
        out.append((int(x), int(y), int(x + w), int(y + h), int(area)))
    return out


def yolo_person_bboxes(model, img: np.ndarray, args) -> list[tuple[float, float, float, float, float]]:
    """Return list of (x1, y1, x2, y2, conf)."""
    result = model.predict(
        img, conf=SCAN_CONF, imgsz=args.imgsz, device=args.device, verbose=False,
    )[0]
    out = []
    if result.boxes is None or result.boxes.shape[0] == 0:
        return out
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    xyxy = result.boxes.xyxy.cpu().numpy()
    for c, cf, box in zip(cls, conf, xyxy):
        if int(c) != COCO_PERSON or float(cf) < HEADLINE_CONF:
            continue
        out.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(cf)))
    return out


def iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ar_a = (ax2 - ax1) * (ay2 - ay1)
    ar_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(ar_a + ar_b - inter, 1e-9)


def match_greedy(preds: list, gts: list, iou_thr: float):
    """Greedy IoU matching: highest-confidence preds first, take best-IoU GT.
    Returns (tp_indices_into_preds, fp_indices, unmatched_gt_indices)."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][4])
    used_gt = set()
    tp, fp = [], []
    for i in order:
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gts):
            if j in used_gt:
                continue
            v = iou(preds[i], gt)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            tp.append(i); used_gt.add(best_j)
        else:
            fp.append(i)
    fn = [j for j in range(len(gts)) if j not in used_gt]
    return tp, fp, fn


def average_precision_at_iou(per_frame_pred_gt: list[tuple[list, list]], iou_thr: float) -> float:
    """Compute AP using Pascal-VOC-style PR-curve integration at fixed IoU."""
    all_records = []   # (conf, is_tp)
    n_gt_total = 0
    for preds, gts in per_frame_pred_gt:
        n_gt_total += len(gts)
        order = sorted(range(len(preds)), key=lambda i: -preds[i][4])
        used = set()
        for i in order:
            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(gts):
                if j in used:
                    continue
                v = iou(preds[i], g)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0 and best_iou >= iou_thr:
                used.add(best_j)
                all_records.append((preds[i][4], True))
            else:
                all_records.append((preds[i][4], False))
    if not all_records or n_gt_total == 0:
        return 0.0
    all_records.sort(key=lambda r: -r[0])
    tp = 0; fp = 0
    precisions, recalls = [], []
    for _, is_tp in all_records:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / max(n_gt_total, 1))
    # 11-point interpolation (PASCAL VOC style)
    ap = 0.0
    for r in [i / 10 for i in range(11)]:
        ps = [p for p, rc in zip(precisions, recalls) if rc >= r]
        ap += (max(ps) if ps else 0.0) / 11.0
    return ap


def find_camera_jpg(frames_dir: Path, frame_idx: int) -> Path | None:
    matches = sorted(frames_dir.glob(f"frame{frame_idx:06d}-*.jpg"))
    return matches[0] if matches else None


def find_label_png(label_dir: Path, frame_idx: int) -> Path | None:
    matches = sorted(label_dir.glob(f"frame{frame_idx:06d}-*.png"))
    return matches[0] if matches else None


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    label_dir = Path(args.label_dir).resolve()
    frames_dir = Path(args.frames_dir).resolve()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    rows = []
    per_frame_pred_gt = []
    n_eval = 0
    n_tp = n_fp = n_fn = 0
    t0 = time.time()
    for fi in range(args.start_frame, args.end_frame):
        lbl_path = find_label_png(label_dir, fi)
        if lbl_path is None:
            continue   # frame not annotated (odd frames + a few even gaps)
        cam_path = find_camera_jpg(frames_dir, fi)
        if cam_path is None:
            log.warning("annotation present for f%d but no jpg; skipping", fi); continue

        mask = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(cam_path))
        if mask is None or img is None:
            log.warning("read failure on f%d; skipping", fi); continue

        gts = gt_person_bboxes(mask)
        preds = yolo_person_bboxes(model, img, args)
        tp_i, fp_i, fn_i = match_greedy(preds, gts, args.iou_threshold)

        n_eval += 1
        n_tp += len(tp_i); n_fp += len(fp_i); n_fn += len(fn_i)
        per_frame_pred_gt.append((preds, gts))

        rows.append({
            "frame_idx": fi,
            "n_gt_persons": len(gts),
            "n_pred_persons": len(preds),
            "n_tp": len(tp_i),
            "n_fp": len(fp_i),
            "n_fn": len(fn_i),
            "max_pred_conf": float(max((p[4] for p in preds), default=0.0)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "b_ext_per_frame.csv", index=False)

    precision = n_tp / max(n_tp + n_fp, 1)
    recall = n_tp / max(n_tp + n_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    map50 = average_precision_at_iou(per_frame_pred_gt, args.iou_threshold)

    metrics = {
        "evaluator": "YOLOv8n-COCO vs RELLIS-3D GT",
        "frame_window": [args.start_frame, args.end_frame],
        "n_frames_evaluated": int(n_eval),
        "iou_threshold": args.iou_threshold,
        "min_gt_component_area_px": MIN_GT_AREA,
        "yolo_conf_threshold": HEADLINE_CONF,
        "totals": {"tp": int(n_tp), "fp": int(n_fp), "fn": int(n_fn)},
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "ap@0.5_pascal11pt": float(map50),
        "wall_seconds": time.time() - t0,
    }
    (out_dir / "b_ext_metrics.json").write_text(json.dumps(metrics, indent=2))

    print()
    print(f"=== B-ext: YOLOv8n-COCO vs RELLIS-3D GT (frames {args.start_frame}..{args.end_frame}) ===")
    print(f"  frames evaluated     : {n_eval}")
    print(f"  TP / FP / FN         : {n_tp} / {n_fp} / {n_fn}")
    print(f"  precision            : {precision:.3f}")
    print(f"  recall               : {recall:.3f}")
    print(f"  F1                   : {f1:.3f}")
    print(f"  AP@0.5 (Pascal 11pt) : {map50:.3f}")
    print(f"  outputs              : {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
