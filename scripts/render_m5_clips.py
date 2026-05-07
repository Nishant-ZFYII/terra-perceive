"""
render_m5_clips.py — produce the two M5 blog clips (P2-M5 Phase D).

Gated behind the TP_M5_RENDER environment variable per the wondrous-crane
plan's anti-celebration guardrail #3 / decision F: animations only at the
end of the milestone, after the metrics are honest.

Two clips:
  clip1_tracker_overlay.mp4  158 frames (window 83..240) with YOLO bbox,
                             track ID, and d_worker distance overlaid on
                             the camera image.
  clip2_b_ext_side_by_side.mp4  79 even-numbered annotated frames in window
                                showing YOLO prediction (left) vs RELLIS-3D
                                GT silhouette (right) side-by-side. Misses
                                are highlighted with a red border.

Output: results_m5/phase_b/clips/*.mp4 + a thumbnail PNG per clip into
docs/assets/m5/ for the blog.

Usage:
  TP_M5_RENDER=1 python scripts/render_m5_clips.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

log = logging.getLogger("m5_render")

START_FRAME = 83
END_FRAME = 241
FPS = 10
WINDOW_LATERAL_M = 5.0
PERSON_CLASS = 17
COCO_PERSON = 0
HEADLINE_CONF = 0.5
MIN_GT_AREA = 200

FRAMES_DIR = REPO / "data/RELLIS-3D/Rellis_3D_pylon_camera_node/Rellis-3D/00000/pylon_camera_node"
LABEL_DIR = REPO / "data/RELLIS-3D/Rellis_3D_pylon_camera_node_label_id/Rellis-3D/00000/pylon_camera_node_label_id"
TRACKS_NDJSON = REPO / "results_m5/phase_b/tracks.ndjson"
WEIGHTS = "/media/nishant/SeeGayt2/terra_perceive/models/yolov8n.pt"
OUT_DIR = REPO / "results_m5/phase_b/clips"
THUMB_DIR = REPO / "docs/assets/m5"


def load_tracks_by_frame() -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = {}
    with TRACKS_NDJSON.open() as fh:
        for line in fh:
            r = json.loads(line)
            by_frame.setdefault(r["frame_idx"], []).append(r)
    return by_frame


def find_jpg(frame_idx: int) -> Path | None:
    matches = sorted(FRAMES_DIR.glob(f"frame{frame_idx:06d}-*.jpg"))
    return matches[0] if matches else None


def find_label(frame_idx: int) -> Path | None:
    matches = sorted(LABEL_DIR.glob(f"frame{frame_idx:06d}-*.png"))
    return matches[0] if matches else None


def yolo_person_bboxes(model, img: np.ndarray):
    res = model.predict(img, conf=0.25, imgsz=640, device="cpu", verbose=False)[0]
    out = []
    if res.boxes is None or res.boxes.shape[0] == 0:
        return out
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    xyxy = res.boxes.xyxy.cpu().numpy()
    for c, cf, box in zip(cls, conf, xyxy):
        if int(c) != COCO_PERSON or float(cf) < HEADLINE_CONF:
            continue
        out.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(cf)))
    return out


def gt_person_bboxes(mask: np.ndarray):
    bin_mask = (mask == PERSON_CLASS).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_GT_AREA:
            continue
        out.append((int(x), int(y), int(x + w), int(y + h), int(area)))
    return out


def overlay_track_panel(img: np.ndarray, bboxes_yolo, tracks, frame_idx: int) -> np.ndarray:
    out = img.copy()
    for x1, y1, x2, y2, conf in bboxes_yolo:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(out, f"YOLO p={conf:.2f}", (int(x1), max(int(y1) - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    panel_lines = [f"frame {frame_idx}", f"tracks: {len(tracks)}"]
    for t in tracks[:3]:
        d = float(np.hypot(t["x"], t["y"]))
        panel_lines.append(f"id={t['track_id']}  d={d:5.2f}m  hits={t['hits']}")
    y0 = 40
    for i, line in enumerate(panel_lines):
        cv2.putText(out, line, (20, y0 + i * 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(out, line, (20, y0 + i * 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
    return out


def render_clip1(model, tracks_by_frame: dict[int, list[dict]]) -> Path:
    out_path = OUT_DIR / "clip1_tracker_overlay.mp4"
    log.info("clip1: rendering frames %d..%d -> %s", START_FRAME, END_FRAME, out_path)
    writer = imageio.get_writer(
        out_path, fps=FPS, codec="libx264", quality=8,
        macro_block_size=1, ffmpeg_log_level="error",
    )
    thumb = None
    for fi in range(START_FRAME, END_FRAME):
        jpg = find_jpg(fi)
        if jpg is None:
            continue
        img = cv2.imread(str(jpg))
        if img is None:
            continue
        bboxes = yolo_person_bboxes(model, img)
        tracks = tracks_by_frame.get(fi, [])
        ann = overlay_track_panel(img, bboxes, tracks, fi)
        # Downscale to keep the mp4 small.
        ann_small = cv2.resize(ann, (1280, 800))
        writer.append_data(cv2.cvtColor(ann_small, cv2.COLOR_BGR2RGB))
        if thumb is None and tracks:
            thumb = ann_small
    writer.close()
    if thumb is not None:
        cv2.imwrite(str(THUMB_DIR / "clip1_tracker_overlay_thumb.png"), thumb)
    return out_path


def overlay_b_ext_panel(
    img: np.ndarray, mask: np.ndarray, yolo_boxes, gt_boxes,
    frame_idx: int, status: str,
) -> np.ndarray:
    H, W = img.shape[:2]
    left = img.copy()
    for x1, y1, x2, y2, conf in yolo_boxes:
        cv2.rectangle(left, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(left, f"YOLO {conf:.2f}", (int(x1), max(int(y1) - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(left, "YOLOv8n-COCO", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(left, "YOLOv8n-COCO", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

    right = img.copy()
    overlay = right.copy()
    person_mask_bool = (mask == PERSON_CLASS)
    overlay[person_mask_bool] = (0, 0, 255)
    right = cv2.addWeighted(right, 0.55, overlay, 0.45, 0)
    for x1, y1, x2, y2, area in gt_boxes:
        cv2.rectangle(right, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(right, f"GT area={area}", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(right, "RELLIS-3D GT (person class)", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(right, "RELLIS-3D GT (person class)", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

    pair = np.hstack([left, right])
    if status == "FN":
        cv2.rectangle(pair, (0, 0), (pair.shape[1] - 1, pair.shape[0] - 1), (0, 0, 255), 14)
    pair = cv2.resize(pair, (1920, 600))

    cv2.putText(pair, f"frame {frame_idx}  status={status}", (20, pair.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(pair, f"frame {frame_idx}  status={status}", (20, pair.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return pair


def render_clip2(model) -> Path:
    """Render every frame in the demo window (continuous playback at 10 FPS).
    GT mask is forward-filled from the most recent annotated frame between
    annotation samples (RELLIS annotates every other frame). The 'status'
    label says ANNOT vs FILL so the viewer can tell which frames carry real
    GT vs forward-filled GT.
    """
    out_path = OUT_DIR / "clip2_b_ext_side_by_side.mp4"
    log.info("clip2: rendering CONTINUOUS frames in %d..%d -> %s",
             START_FRAME, END_FRAME, out_path)
    writer = imageio.get_writer(
        out_path, fps=10, codec="libx264", quality=8,
        macro_block_size=1, ffmpeg_log_level="error",
    )
    thumb = None
    cached_mask = None
    cached_gt = []
    cached_annot_frame = None
    for fi in range(START_FRAME, END_FRAME):
        jpg = find_jpg(fi)
        if jpg is None:
            continue
        img = cv2.imread(str(jpg))
        if img is None:
            continue
        lbl = find_label(fi)
        if lbl is not None:
            mask_load = cv2.imread(str(lbl), cv2.IMREAD_GRAYSCALE)
            if mask_load is not None:
                cached_mask = mask_load
                cached_gt = gt_person_bboxes(mask_load)
                cached_annot_frame = fi
        if cached_mask is None:
            # First frame may be unannotated; render with empty GT.
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            gt = []
        else:
            mask = cached_mask
            gt = cached_gt
        yolo = yolo_person_bboxes(model, img)
        # Determine status: PASS if at least one IoU>=0.5 match, else MISS / FN.
        status = "PASS"
        if not yolo and gt:
            status = "FN"
        elif gt:
            best_iou = 0.0
            for y in yolo:
                for g in gt:
                    ix1 = max(y[0], g[0]); iy1 = max(y[1], g[1])
                    ix2 = min(y[2], g[2]); iy2 = min(y[3], g[3])
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        ar_y = (y[2] - y[0]) * (y[3] - y[1])
                        ar_g = (g[2] - g[0]) * (g[3] - g[1])
                        u = ar_y + ar_g - inter
                        v = inter / max(u, 1e-9)
                        if v > best_iou:
                            best_iou = v
            if best_iou < 0.5:
                status = "FN"
        # If this frame had no fresh annotation, mark as forward-filled.
        if cached_annot_frame != fi:
            status = f"{status}/FILL"
        pair = overlay_b_ext_panel(img, mask, yolo, gt, fi, status)
        writer.append_data(cv2.cvtColor(pair, cv2.COLOR_BGR2RGB))
        if thumb is None:
            thumb = pair
    writer.close()
    if thumb is not None:
        cv2.imwrite(str(THUMB_DIR / "clip2_b_ext_side_by_side_thumb.png"), thumb)
    return out_path


def main() -> int:
    if os.environ.get("TP_M5_RENDER") != "1":
        print("TP_M5_RENDER not set; skipping (the wondrous-crane plan gates "
              "rendering behind this flag).")
        return 0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(WEIGHTS)

    tracks_by_frame = load_tracks_by_frame()
    log.info("tracks loaded: %d frames", len(tracks_by_frame))

    t0 = time.time()
    p1 = render_clip1(model, tracks_by_frame)
    log.info("clip1 done in %.1fs -> %s", time.time() - t0, p1)
    t1 = time.time()
    p2 = render_clip2(model)
    log.info("clip2 done in %.1fs -> %s", time.time() - t1, p2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
