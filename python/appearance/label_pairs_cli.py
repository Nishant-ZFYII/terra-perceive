#!/usr/bin/env python3
"""label_pairs_cli.py — Tk UI for hand-labeling cluster pairs.

This is the human-in-the-loop step that produces the held-out validation
set for the P3-M13 appearance encoder. Outputs go to
`python/appearance/labels.csv` and feed `tests/data/appearance_reference.csv`
indirectly (by anchoring training-time data quality decisions).

Why hand labeling matters:
    The training data has THREE candidate sources (per the wondrous-crane
    plan, Decision D):
       1. Geometric augmentation (jitter the same cluster).
       2. Adjacent-frame nearest-neighbor with tight gates.
       3. M4 SORT tracks with lifetime ≥ 30 etc.
    All three risk circularity — they encode "same object" via assumptions
    we built into the data, not via real ground truth. Hand-labels are the
    ONE source that breaks circularity. They're held out of training and
    used purely to detect whether the encoder learned appearance vs. learned
    "the augmentation pattern."

UI:
    Each pair shows:
        Left panel:  cluster A — BEV scatter (3D points in x-y), top-down
        Right panel: cluster B — BEV scatter (same view)
        Inset (each panel): the camera frame at that timestamp
                            with a red dot at the projected cluster centroid
    Buttons:  [SAME]  [DIFFERENT]  [SKIP]  [QUIT]
    Keyboard: s → SAME, d → DIFFERENT, space → SKIP, q → QUIT

Output schema (labels.csv):
    pair_id, label, frame_a, cluster_a, frame_b, cluster_b, source

    label ∈ {"same", "different", "skip"}.

Usage:
    python python/appearance/label_pairs_cli.py \\
        --pair-csv python/appearance/pair_candidates.csv \\
        --clusters-dir /media/nishant/SeeGayt2/terra_perceive/m4_perframe/clusters_sweetspot \\
        --camera-dir  /media/nishant/SeeGayt2/terra_perceive/m4_perframe/extracted_frames_camera \\
        --calib       config/camera_lidar_calib.yaml \\
        --out         python/appearance/labels.csv

Resume support: if labels.csv exists, the UI skips already-labeled pair_ids.
"""
import argparse
import csv
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import patches
from PIL import Image


# -----------------------------------------------------------------------------
# Calibration + projection
# -----------------------------------------------------------------------------

def load_calib(yaml_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (K, T_cam_lidar) from the project's camera_lidar_calib.yaml.

    Expected schema (RELLIS-3D, see config/camera_lidar_calib.yaml):

        camera_intrinsics:
            fx: <float>
            fy: <float>
            cx: <float>
            cy: <float>
            (width, height optional)

        extrinsic_T_cam_lidar:
            - [r00, r01, r02, tx]
            - [r10, r11, r12, ty]
            - [r20, r21, r22, tz]
            - [0,   0,   0,   1]

    Returns (K: 3x3, T_cam_lidar: 4x4) as float64 numpy arrays.
    """
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f)

    intr = cfg.get("camera_intrinsics") or cfg.get("intrinsic_K")
    if intr is None:
        raise KeyError(
            f"{yaml_path}: expected top-level key 'camera_intrinsics' or "
            f"'intrinsic_K'; got {list(cfg.keys())}"
        )
    K = np.array([
        [intr["fx"], 0,          intr["cx"]],
        [0,          intr["fy"], intr["cy"]],
        [0,          0,          1],
    ], dtype=np.float64)

    T_raw = cfg["extrinsic_T_cam_lidar"]
    # YAML list-of-rows form (RELLIS) → already 4x4 nested. Flat 16-float
    # form falls back to .reshape().
    T_arr = np.array(T_raw, dtype=np.float64)
    if T_arr.shape == (4, 4):
        T = T_arr
    else:
        T = T_arr.reshape(4, 4)
    return K, T


def project_centroid_to_camera(
    centroid_lidar: np.ndarray, K: np.ndarray, T_cam_lidar: np.ndarray
) -> Optional[Tuple[float, float]]:
    """Project a LiDAR-frame point to image-frame (u, v). Returns None if
    behind the camera.
    """
    p_h = np.array([centroid_lidar[0], centroid_lidar[1], centroid_lidar[2], 1.0])
    p_cam = T_cam_lidar @ p_h
    if p_cam[2] <= 0:
        return None
    p_img = K @ p_cam[:3]
    return float(p_img[0] / p_img[2]), float(p_img[1] / p_img[2])


# -----------------------------------------------------------------------------
# Cluster CSV loader (mirrors build_pairs.py for consistency)
# -----------------------------------------------------------------------------

def load_clusters(csv_path: Path) -> Dict[int, np.ndarray]:
    out: Dict[int, list] = {}
    with csv_path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            x, y, z, cid = float(row[0]), float(row[1]), float(row[2]), int(row[3])
            if cid < 0:
                continue
            out.setdefault(cid, []).append((x, y, z))
    return {cid: np.asarray(pts, dtype=np.float32) for cid, pts in out.items()}


# -----------------------------------------------------------------------------
# Labels persistence
# -----------------------------------------------------------------------------

def load_existing_labels(out_csv: Path) -> set:
    if not out_csv.exists():
        return set()
    done = set()
    with out_csv.open() as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            done.add(int(row[0]))
    return done


def append_label(
    out_csv: Path,
    pair_id: int,
    label: str,
    frame_a: int,
    cluster_a: int,
    frame_b: int,
    cluster_b: int,
    source: str,
) -> None:
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["pair_id", "label", "frame_a", "cluster_a",
                        "frame_b", "cluster_b", "source"])
        w.writerow([pair_id, label, frame_a, cluster_a, frame_b, cluster_b, source])


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

class LabelApp:
    def __init__(
        self,
        pairs: List[Dict],
        clusters_dir: Path,
        camera_dir: Path,
        lidar_dir: Path,
        K: np.ndarray,
        T_cam_lidar: np.ndarray,
        out_csv: Path,
    ):
        self.pairs = pairs
        self.clusters_dir = clusters_dir
        self.camera_dir = camera_dir
        self.lidar_dir = lidar_dir
        self.K = K
        self.T_cam_lidar = T_cam_lidar
        self.out_csv = out_csv
        self.idx = 0

        self.root = tk.Tk()
        self.root.title("M13 — appearance pair labeler")
        # 2-row layout (camera-POV row dropped; was unhelpful per user):
        #   row 0 (h=2):   A BEV (zoomed) | B BEV (zoomed)
        #   row 1 (h=2):   ego-centered radar BEV (M4-animation style:
        #                  black bg, white centroids, concentric distance
        #                  circles, all clusters from the frame visible) |
        #                  3D rotatable view.
        # Total ~9 inches tall → fits on a 1080p screen with the buttons.
        self.fig = Figure(figsize=(12, 9), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Button(btn_frame, text="SAME (s)", width=14, bg="#a3e4a3",
                  command=lambda: self.record("same")).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(btn_frame, text="DIFFERENT (d)", width=14, bg="#e4a3a3",
                  command=lambda: self.record("different")).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(btn_frame, text="SKIP (space)", width=14,
                  command=lambda: self.record("skip")).pack(side=tk.LEFT, padx=4, pady=4)
        # Render the temporal-context GIF for the current pair
        # (~6-10 sec wall-clock; UI freezes during render, then opens
        # the GIF in the system viewer). Useful for tough adjacent pairs
        # where the static side-by-side radars don't disambiguate.
        tk.Button(btn_frame, text="Context GIF (g)", width=18,
                  bg="#a3c3e4",
                  command=self.render_context_gif).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(btn_frame, text="QUIT (q)", width=14,
                  command=self.root.destroy).pack(side=tk.RIGHT, padx=4, pady=4)

        self.status = tk.Label(self.root, text="", anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.bind("<KeyPress-s>", lambda e: self.record("same"))
        self.root.bind("<KeyPress-d>", lambda e: self.record("different"))
        self.root.bind("<KeyPress-space>", lambda e: self.record("skip"))
        self.root.bind("<KeyPress-q>", lambda e: self.root.destroy())
        self.root.bind("<KeyPress-g>", lambda e: self.render_context_gif())

        self.show_pair()

    # Color scheme: red for cluster A, magenta for cluster B. Both rare in
    # RELLIS foliage scenes, both stand out against grass/sky/dirt.
    COLOR_A = "#ff0000"
    COLOR_B = "#ff00ff"

    def _load_panel_data(self, side: str, pair):
        """Helper: returns (frame, cid, pts, centroid) for side ∈ {'a', 'b'}.
        Returns None values if the cluster can't be loaded.
        """
        frame = pair[f"frame_{side}"]
        cid   = pair[f"cluster_{side}"]
        cluster_csv = self.clusters_dir / f"clusters_{frame:06d}.csv"
        if not cluster_csv.exists():
            return frame, cid, None, None
        clusters = load_clusters(cluster_csv)
        if cid not in clusters:
            return frame, cid, None, None
        pts = clusters[cid]
        centroid = pts.mean(axis=0) if pts.shape[0] > 0 else None
        return frame, cid, pts, centroid

    def _draw_camera_inset(self, ax, frame: int, dots):
        """Draw the camera image as an inset, with `dots` overlaid.

        dots: list of (centroid_xyz, color, label, size). For each dot, we
        project to image coords and either render a colored circle or, if
        projection fails, append "<label> behind camera" to the inset title.
        """
        cam_path = self.camera_dir / f"{frame:06d}.jpg"
        ax_in = ax.inset_axes([0.55, 0.62, 0.42, 0.36])
        ax_in.set_xticks([]); ax_in.set_yticks([])
        if not cam_path.exists():
            ax_in.text(0.5, 0.5, f"camera frame\n{frame:06d}.jpg\nmissing",
                       ha="center", va="center", transform=ax_in.transAxes,
                       fontsize=8, color="gray")
            return
        try:
            img = np.array(Image.open(cam_path).convert("RGB"))
        except Exception as e:
            ax_in.text(0.5, 0.5, f"image load error:\n{e}",
                       ha="center", va="center", transform=ax_in.transAxes,
                       fontsize=8, color="red")
            return

        ax_in.imshow(img)
        H, W = img.shape[0], img.shape[1]

        annotations = []
        for centroid, color, label, size in dots:
            if centroid is None:
                annotations.append(f"{label}: no cluster")
                continue
            uv = project_centroid_to_camera(centroid, self.K, self.T_cam_lidar)
            if uv is None:
                annotations.append(f"{label} behind camera")
                continue
            u, v = uv
            if not (0 <= u < W and 0 <= v < H):
                annotations.append(f"{label} off-frame ({int(u)}, {int(v)})")
                continue
            ax_in.scatter([u], [v], s=size, c=color,
                          edgecolors="white", linewidths=1.8, zorder=10)

        if annotations:
            # Stack annotations top-left of the inset, white-on-black for
            # readability against any image background.
            ax_in.text(0.02, 0.98, "\n".join(annotations),
                       transform=ax_in.transAxes,
                       ha="left", va="top",
                       fontsize=8, color="white",
                       bbox=dict(facecolor="black", alpha=0.6,
                                 edgecolor="none", pad=2))

    def _draw_camera_overlay(self, ax, frame: int,
                             highlight_a: int = None,
                             highlight_b: int = None,
                             side_label: str = "") -> None:
        """Vehicle-POV view: load the frame's camera image, project EVERY
        cluster's points onto it (in muted gray), then overlay highlight_a
        in red and highlight_b in magenta on top.

        Decimates each cluster to ≤ 80 points to keep render time fast.
        """
        ax.set_xticks([]); ax.set_yticks([])
        cam_path = self.camera_dir / f"{frame:06d}.jpg"
        if not cam_path.exists():
            ax.text(0.5, 0.5, f"camera frame {frame:06d}.jpg missing",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray")
            return
        try:
            img = np.array(Image.open(cam_path).convert("RGB"))
        except Exception as e:
            ax.text(0.5, 0.5, f"image load error: {e}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="red")
            return

        ax.imshow(img)
        H, W = img.shape[0], img.shape[1]

        cluster_csv = self.clusters_dir / f"clusters_{frame:06d}.csv"
        if not cluster_csv.exists():
            ax.set_title(f"frame {frame} (no cluster CSV)", fontsize=10)
            return
        clusters = load_clusters(cluster_csv)

        # Project every cluster, decimating large ones for speed.
        n_other_in_frame = 0
        for cid, pts in clusters.items():
            # Decimate large clusters — uniform sample down to 80 pts max.
            if pts.shape[0] > 80:
                idx = np.linspace(0, pts.shape[0] - 1, 80, dtype=int)
                pts_dec = pts[idx]
            else:
                pts_dec = pts

            # Per-point projection (vectorize for speed).
            ones = np.ones((pts_dec.shape[0], 1), dtype=np.float64)
            pts_h = np.hstack([pts_dec.astype(np.float64), ones])
            pts_cam = (self.T_cam_lidar @ pts_h.T).T          # (N, 4)
            in_front = pts_cam[:, 2] > 0
            pts_cam = pts_cam[in_front, :3]
            if pts_cam.shape[0] == 0:
                continue
            uv_h = (self.K @ pts_cam.T).T                     # (N, 3)
            uv = uv_h[:, :2] / uv_h[:, 2:3]
            in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < W) & \
                       (uv[:, 1] >= 0) & (uv[:, 1] < H)
            uv = uv[in_frame]
            if uv.shape[0] == 0:
                continue

            if cid == highlight_a:
                ax.scatter(uv[:, 0], uv[:, 1], s=20, c=self.COLOR_A,
                           edgecolors="white", linewidths=0.5,
                           alpha=0.95, zorder=10)
            elif cid == highlight_b:
                ax.scatter(uv[:, 0], uv[:, 1], s=20, c=self.COLOR_B,
                           edgecolors="white", linewidths=0.5,
                           alpha=0.95, zorder=10)
            else:
                ax.scatter(uv[:, 0], uv[:, 1], s=4, c="gray",
                           alpha=0.45, zorder=5)
                n_other_in_frame += 1

        title = f"camera POV — frame {frame}  ({len(clusters)} clusters total"
        if n_other_in_frame:
            title += f", {n_other_in_frame} visible)"
        else:
            title += ")"
        if side_label:
            title = f"{side_label}: {title}"
        ax.set_title(title, fontsize=10)

    def _draw_combined_2d(self, ax, pts_a, c_a, pts_b, c_b):
        """Combined-BEV 2D plot. Pads the smaller axis so far-apart pairs
        don't render as a useless ultra-wide-thin strip."""
        plotted_anything = False
        all_x, all_y = [0.0], [0.0]   # always include ego origin

        if pts_a is not None:
            ax.scatter(pts_a[:, 0], pts_a[:, 1], s=3, c=self.COLOR_A,
                       alpha=0.5, label="A points")
            ax.scatter([c_a[0]], [c_a[1]], marker="*", s=260, c=self.COLOR_A,
                       edgecolors="black", linewidths=1.2, zorder=10,
                       label="A centroid")
            all_x.extend(pts_a[:, 0].tolist())
            all_y.extend(pts_a[:, 1].tolist())
            plotted_anything = True

        if pts_b is not None:
            ax.scatter(pts_b[:, 0], pts_b[:, 1], s=3, c=self.COLOR_B,
                       alpha=0.5, label="B points")
            ax.scatter([c_b[0]], [c_b[1]], marker="*", s=260, c=self.COLOR_B,
                       edgecolors="black", linewidths=1.2, zorder=10,
                       label="B centroid")
            all_x.extend(pts_b[:, 0].tolist())
            all_y.extend(pts_b[:, 1].tolist())
            plotted_anything = True

        ax.scatter([0], [0], marker="s", s=80, c="black",
                   edgecolors="white", linewidths=1.5, zorder=11,
                   label="ego origin")

        if plotted_anything and c_a is not None and c_b is not None:
            distance = float(np.linalg.norm(c_a - c_b))
            ax.set_title(f"2D BEV (top-down) — A↔B 3D dist = {distance:.2f} m",
                         fontsize=10)
        else:
            ax.set_title("2D BEV (top-down)", fontsize=10)

        # Equal aspect, but pad whichever axis has the smaller range so the
        # plot doesn't degenerate into a thin strip when clusters are far
        # apart on one axis only. Target: smaller range ≥ 35% of larger.
        x_arr, y_arr = np.array(all_x), np.array(all_y)
        x_min, x_max = float(x_arr.min()), float(x_arr.max())
        y_min, y_max = float(y_arr.min()), float(y_arr.max())
        x_range = max(x_max - x_min, 1.0)   # at least 1 m to avoid zero-pad
        y_range = max(y_max - y_min, 1.0)
        target = 0.35 * max(x_range, y_range)
        if x_range < target:
            mid = 0.5 * (x_min + x_max)
            x_min, x_max = mid - target / 2, mid + target / 2
        if y_range < target:
            mid = 0.5 * (y_min + y_max)
            y_min, y_max = mid - target / 2, mid + target / 2
        # Add a 10% margin all around for visual breathing room.
        margin = 0.10
        ax.set_xlim(x_min - margin * (x_max - x_min),
                    x_max + margin * (x_max - x_min))
        ax.set_ylim(y_min - margin * (y_max - y_min),
                    y_max + margin * (y_max - y_min))
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)   [-x = camera forward]")
        ax.set_ylabel("y (m)   [+y = right]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    def _load_full_lidar(self, frame: int) -> np.ndarray:
        """Load the raw LiDAR scan for `frame` from KITTI-format .bin
        (x, y, z, intensity per row, float32). Returns (N, 3) xyz or empty
        if the file's missing.
        """
        path = self.lidar_dir / f"{frame:06d}.bin"
        if not path.exists():
            return np.empty((0, 3), dtype=np.float32)
        raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return raw[:, :3]

    def _draw_one_radar(self, ax, frame: int,
                        clusters: List[Tuple[int, str, np.ndarray, np.ndarray, str]],
                        title: str, max_r: float = 40.0) -> None:
        """Render ONE M4-style radar in `ax`.

        Args:
            frame:    which frame's full LiDAR scan to render as the
                      background (faint gray).
            clusters: list of (cid, color, pts, centroid, label) — every
                      tuple in this list gets rendered as filled colored
                      circles + white centroid dot + colored ID badge.
                      For same-frame pairs, pass both A and B here.
                      For adjacent pairs, use two side-by-side radars
                      and pass one cluster to each.
            title:    text to put in the radar's title (top of axes).
        """
        ax.set_facecolor("#1a1a1a")
        ax.set_aspect("equal")

        # Background: full LiDAR scan as faint gray scene context.
        full = self._load_full_lidar(frame)
        if full.size:
            ax.scatter(full[:, 0], full[:, 1], s=1, c="#555555",
                       alpha=0.5, zorder=1, edgecolors="none")

        # Each cluster: filled colored circles + black edges (M4 style),
        # white centroid dot, colored ID badge offset from the dot.
        for cid, color, pts, centroid, label in clusters:
            if pts is not None:
                ax.scatter(pts[:, 0], pts[:, 1], s=22, c=color,
                           alpha=0.95, zorder=4,
                           edgecolors="black", linewidths=0.4)
            if centroid is not None:
                ax.scatter([centroid[0]], [centroid[1]], marker="o",
                           s=42, c="white",
                           edgecolors="black", linewidths=0.8, zorder=10)
                ax.text(centroid[0] + 1.2, centroid[1] + 1.2,
                        f"{label}: #{cid}",
                        color="white", fontsize=9, fontweight="bold",
                        bbox=dict(facecolor=color, edgecolor="none",
                                  alpha=0.9, pad=2.5),
                        zorder=11)

        # Ego marker is implied by the concentric distance rings centered
        # at (0, 0) — no explicit square needed (and it was just visual
        # noise in the middle of the scene).

        # Auto-extend ring radius to include any far centroid in this
        # subplot.
        if clusters:
            for _, _, _, c, _ in clusters:
                if c is not None:
                    max_r = max(max_r, float(np.linalg.norm(c[:2])) + 5.0)
        for r in range(5, int(max_r) + 1, 5):
            circle = patches.Circle((0, 0), r, fill=False,
                                    edgecolor="#3a3a3a", linewidth=0.6,
                                    linestyle=":", zorder=2)
            ax.add_patch(circle)
            ax.text(r, 0.2, f"{r}m", color="#5a5a5a", fontsize=7,
                    ha="left", va="bottom", zorder=2)

        # Frame-id badge top-left.
        ax.text(0.015, 0.985, f"frame {frame}", transform=ax.transAxes,
                color="white", fontsize=10, fontweight="bold",
                ha="left", va="top",
                bbox=dict(facecolor="black", edgecolor="white",
                          alpha=0.7, pad=3),
                zorder=15)

        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.set_xlabel("x (m)", color="white")
        ax.set_ylabel("y (m)", color="white")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.set_title(title, color="white", fontsize=10)

    def _draw_combined_3d(self, ax, pts_a, c_a, pts_b, c_b):
        """Combined 3D scatter of A, B, and the ego/LiDAR origin. Drag with
        mouse to rotate. Ego is the white-edged cyan square at (0, 0, 0)
        — the LiDAR mount position, which is the reference frame all the
        cluster coordinates are in.
        """
        if pts_a is not None:
            ax.scatter(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2],
                       s=3, c=self.COLOR_A, alpha=0.5, label="A")
            ax.scatter([c_a[0]], [c_a[1]], [c_a[2]],
                       marker="*", s=180, c="white",
                       edgecolors=self.COLOR_A, linewidths=1.4)
        if pts_b is not None:
            ax.scatter(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2],
                       s=3, c=self.COLOR_B, alpha=0.5, label="B")
            ax.scatter([c_b[0]], [c_b[1]], [c_b[2]],
                       marker="*", s=180, c="white",
                       edgecolors=self.COLOR_B, linewidths=1.4)

        # Ego at origin. Same color treatment as the radar (white fill,
        # cyan edge) so the eye links the two views together.
        ax.scatter([0], [0], [0], marker="s", s=120, c="white",
                   edgecolors="cyan", linewidths=1.4, label="ego")

        ax.set_title("3D view (drag to rotate)", fontsize=10)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        # Lower elevation, side-on view — clusters appear at "ground level"
        # in the visual rather than viewed from above. elev=12, azim=-90
        # is "looking from the right side, slightly above horizon."
        ax.view_init(elev=12, azim=-90)
        if pts_a is not None or pts_b is not None:
            ax.legend(loc="upper left", fontsize=7, framealpha=0.85)

    def show_pair(self):
        if self.idx >= len(self.pairs):
            messagebox.showinfo("Done", f"All {len(self.pairs)} pairs labeled.")
            self.root.destroy()
            return

        pair = self.pairs[self.idx]
        self.fig.clear()

        # Pre-load both clusters so we can cross-reference (e.g. show A's
        # centroid on B's camera image when both are from the same frame).
        frame_a, cid_a, pts_a, c_a = self._load_panel_data("a", pair)
        frame_b, cid_b, pts_b, c_b = self._load_panel_data("b", pair)
        same_frame = (frame_a == frame_b)

        # 2x2 gridspec — top row primary, bottom row context.
        gs = self.fig.add_gridspec(2, 2, height_ratios=[2, 2])

        # Render the two BEV panels (top row).
        for col, (side, frame, cid, pts, centroid, color) in enumerate([
            ("a", frame_a, cid_a, pts_a, c_a, self.COLOR_A),
            ("b", frame_b, cid_b, pts_b, c_b, self.COLOR_B),
        ]):
            ax = self.fig.add_subplot(gs[0, col])
            if pts is None:
                ax.set_title(f"frame {frame}  cid {cid}  — MISSING")
                ax.text(0.5, 0.5, "cluster CSV row missing\n(skip this pair)",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            # BEV scatter — color the points to match A/B convention so the
            # left/right correspondence with the camera dots is unambiguous.
            ax.scatter(pts[:, 0], pts[:, 1], s=4, c=color, alpha=0.7)
            ax.set_aspect("equal")

            # Title now includes (x, y, z) of centroid — the numeric backstop
            # for cases where the camera inset can't show a useful dot.
            cx, cy, cz = (float(centroid[0]), float(centroid[1]), float(centroid[2]))
            label_letter = side.upper()
            ax.set_title(
                f"{label_letter}: frame {frame}  cid {cid}  N={pts.shape[0]}\n"
                f"centroid ({cx:+.2f}, {cy:+.2f}, {cz:+.2f}) m",
                fontsize=10,
            )
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
            # Per-panel camera inset removed (was obstructing the BEV
            # scatter at the top-right corner of each axes). The middle
            # row's full vehicle-POV view supersedes it — same camera
            # image, larger, with A and B highlighted in red/magenta on
            # top of every other cluster's projection.

        # ─── Bottom-left: M4-animation-style radar(s) ──────────────────
        # SAME-FRAME pair → ONE big radar with both A and B on the same
        # scene context.
        # ADJACENT-FRAME pair → TWO side-by-side mini-radars at matching
        # scale. Each shows its own frame's full LiDAR scan + its own
        # cluster (A on the left, B on the right). User scans left-right
        # to compare positions: same world-spot in both = SAME object.
        if same_frame:
            ax_radar = self.fig.add_subplot(gs[1, 0])
            self._draw_one_radar(
                ax_radar, frame_a,
                [(cid_a, self.COLOR_A, pts_a, c_a, "A"),
                 (cid_b, self.COLOR_B, pts_b, c_b, "B")],
                title=(f"radar BEV — A↔B 3D dist = "
                       f"{float(np.linalg.norm(c_a - c_b)):.2f} m"
                       if c_a is not None and c_b is not None
                       else "radar BEV"),
            )
        else:
            sub = gs[1, 0].subgridspec(1, 2, wspace=0.05)
            ax_left = self.fig.add_subplot(sub[0])
            ax_right = self.fig.add_subplot(sub[1])
            self._draw_one_radar(
                ax_left, frame_a,
                [(cid_a, self.COLOR_A, pts_a, c_a, "A")],
                title=f"frame {frame_a} (A)",
            )
            self._draw_one_radar(
                ax_right, frame_b,
                [(cid_b, self.COLOR_B, pts_b, c_b, "B")],
                title=f"frame {frame_b} (B)",
            )
            # Show the inter-frame displacement of the two centroids in
            # the suptitle margin so the user knows how far apart they
            # are in the SAME world frame (since both frames are <0.1s
            # apart, the world frame ≈ shared).
            if c_a is not None and c_b is not None:
                d = float(np.linalg.norm(c_a - c_b))
                ax_left.text(
                    0.015, 0.91, f"A↔B Δ = {d:.2f} m",
                    transform=ax_left.transAxes,
                    color="yellow", fontsize=9, fontweight="bold",
                    ha="left", va="top",
                    bbox=dict(facecolor="black", edgecolor="yellow",
                              alpha=0.7, pad=3),
                    zorder=15,
                )

        # ─── Bottom-right: combined 3D ──────────────────────────────────
        ax_3d = self.fig.add_subplot(gs[1, 1], projection="3d")
        self._draw_combined_3d(ax_3d, pts_a, c_a, pts_b, c_b)

        self.fig.suptitle(
            f"pair {pair['pair_id']}  ({self.idx+1}/{len(self.pairs)})  "
            f"prior: {pair['prior_label']}  source: {pair['source']}"
            + ("  [same frame — insets share image]" if same_frame else ""),
            fontsize=11,
        )
        # constrained_layout (set in __init__) handles spacing; skip
        # tight_layout() because it warns on inset_axes children.
        self.canvas.draw()
        self.status.config(
            text=f"{self.idx+1}/{len(self.pairs)} — "
                 "red=A, magenta=B — keys: s=SAME, d=DIFFERENT, space=SKIP, q=QUIT"
        )

    def render_context_gif(self):
        """Render the temporal-context GIF for the current pair via
        scripts/animate_pair_context.py and open it in the system viewer.

        Blocks the UI for the duration of the render (~6-10 s). Cached
        per-pair under /tmp/pair_<id>_context.gif so re-clicking on the
        same pair just reopens the cached file.
        """
        import subprocess
        import shutil

        if self.idx >= len(self.pairs):
            return
        pair = self.pairs[self.idx]
        out = Path(f"/tmp/pair_{pair['pair_id']:04d}_context.gif")

        if not out.exists():
            self.status.config(text=f"rendering context GIF for pair "
                                    f"{pair['pair_id']} ... (~10 s)")
            self.root.update_idletasks()
            cmd = [
                sys.executable,
                "scripts/animate_pair_context.py",
                "--frame-a",   str(pair["frame_a"]),
                "--cluster-a", str(pair["cluster_a"]),
                "--frame-b",   str(pair["frame_b"]),
                "--cluster-b", str(pair["cluster_b"]),
                "--clusters-dir", str(self.clusters_dir),
                "--lidar-dir",    str(self.lidar_dir),
                "--out",          str(out),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                self.status.config(
                    text=f"GIF render failed: {e.stderr.strip()[:120]}")
                return
            except FileNotFoundError:
                self.status.config(text="GIF render: animate_pair_context.py not found")
                return

        # Open in default viewer (xdg-open on Linux, open on macOS).
        opener = "xdg-open" if shutil.which("xdg-open") else (
            "open" if shutil.which("open") else None)
        if opener:
            subprocess.Popen([opener, str(out)])
            self.status.config(text=f"opened {out}")
        else:
            self.status.config(text=f"GIF saved to {out} (open manually)")

    def record(self, label: str):
        pair = self.pairs[self.idx]
        append_label(
            self.out_csv,
            pair["pair_id"], label,
            pair["frame_a"], pair["cluster_a"],
            pair["frame_b"], pair["cluster_b"],
            pair["source"],
        )
        self.idx += 1
        self.show_pair()

    def run(self):
        self.root.mainloop()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pair-csv", type=Path, required=True)
    p.add_argument("--clusters-dir", type=Path, required=True)
    p.add_argument("--camera-dir", type=Path, required=True)
    # Raw LiDAR scans (KITTI .bin, x,y,z,intensity per row). Used as the
    # faint-gray scene context in the radar view, mirroring the M4
    # closing-hero animation's look. If missing, radar still works but
    # without scene background.
    p.add_argument("--lidar-dir", type=Path,
                   default=Path("data/extracted_frames_full"))
    p.add_argument("--calib", type=Path, default=Path("config/camera_lidar_calib.yaml"))
    p.add_argument("--out", type=Path, default=Path("python/appearance/labels.csv"))
    args = p.parse_args()

    K, T = load_calib(args.calib)

    with args.pair_csv.open() as f:
        pairs = list(csv.DictReader(f))
    for r in pairs:
        for k in ("pair_id", "frame_a", "cluster_a", "frame_b", "cluster_b"):
            r[k] = int(r[k])

    done = load_existing_labels(args.out)
    if done:
        print(f"[label_ui] resuming — skipping {len(done)} already-labeled pairs")
        pairs = [p for p in pairs if p["pair_id"] not in done]
    if not pairs:
        print("[label_ui] nothing to label — labels.csv is complete.")
        return

    LabelApp(pairs, args.clusters_dir, args.camera_dir,
             args.lidar_dir, K, T, args.out).run()


if __name__ == "__main__":
    main()
