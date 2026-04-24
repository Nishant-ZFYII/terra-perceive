"""
dashboard_subscriber.py — live perception dashboard subscribing to NATS.

Two-panel layout:
  * LEFT — sensor.camera.rgb: most-recent JPEG from the RELLIS pylon camera.
  * RIGHT — perception.traversability.grid: most-recent BEVGrid, risk-coloured,
    auto-zoomed to observed-cell bbox, with scalebar.

Each panel is fed by an independent subscription. If only the BEV publisher is
running, the camera panel stays on the placeholder and vice versa — that's
pub/sub working as intended (consumers compose; producers are unaware).

Usage:
    python scripts/dashboard_subscriber.py
    python scripts/dashboard_subscriber.py --subject perception.traversability.grid \\
        --nats-url nats://localhost:4222

CLI flags:
    --grid-subject    NATS subject for BEVGrid (default perception.traversability.grid).
    --camera-subject  NATS subject for CameraFrame (default sensor.camera.rgb).
    --nats-url        NATS server URL (default nats://localhost:4222).
    --refresh-ms      GUI refresh interval in ms (default 100).
    --colormap        matplotlib colormap (default viridis).
    --save-frames     Optional dir; if set, every received BEVGrid is saved
                      as a .npy for offline montage / video assembly.
    --no-autozoom     Disable auto-zoom-to-observed-bbox.
    --margin-m        Margin (metres) around observed-cell bbox when autozooming.
    --bg              Background colour (default 'black').
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import time
from collections import deque
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")  # works without DISPLAY tricks in most envs
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties
import numpy as np

# Pillow for JPEG → numpy decode. Lighter than OpenCV; ships with most envs.
try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from transport.nats_adapter import Adapter
from transport.proto import perception_pb2 as pb

log = logging.getLogger("dashboard_subscriber")


# --------------------------------------------------------------------------- #
# Shared state between the NATS callback (asyncio task) and the matplotlib
# refresh callback (Tk timer). Updates are atomic-ish under the GIL; the
# dashboard reads-and-blits whenever it wakes up.
# --------------------------------------------------------------------------- #
_FRAME_ID_RE = re.compile(r"frame_(\d+)")


def _extract_frame_idx(frame_id: str) -> Optional[int]:
    """'frame_001750' → 1750.  None if not parseable."""
    if not frame_id:
        return None
    m = _FRAME_ID_RE.search(frame_id)
    return int(m.group(1)) if m else None


class LatestFrame:
    """Snapshot of the most-recently-received BEVGrid plus a buffer of recent
    camera frames. Time/frame sync is done at render time by picking the
    camera entry whose `frame_id` integer is closest to the current BEV
    `frame_id` — this is the consumer-side equivalent of ROS's TimeSynchronizer.
    Producers remain independent (the whole point of pub/sub)."""

    # How many recent camera frames to keep buffered. Big enough to span the
    # gap when the camera has many frames per BEV snapshot stride; small enough
    # to bound memory at ~150 KB × N JPEGs ≈ tens of MB.
    CAMERA_BUFFER_SIZE = 200

    def __init__(self) -> None:
        # ----- BEVGrid (always the most recent) -----
        self.grid: Optional[np.ndarray] = None       # shape (rows, cols), risk in [0,1]
        self.obs_mask: Optional[np.ndarray] = None   # bool (rows, cols), True = observed
        self.resolution: float = 1.0                 # metres per cell
        self.origin_x: float = 0.0                   # world coords of grid (0,0)
        self.origin_y: float = 0.0
        self.grid_frame_id: str = "(none)"
        self.grid_frame_idx: Optional[int] = None    # parsed integer
        self.pose_sigma: float = 0.0
        self.n_grid: int = 0
        self.grid_latencies: deque[float] = deque(maxlen=30)
        # Cumulative observed-cell bbox in WORLD coordinates (metres). Grows
        # monotonically across received frames so the auto-zoom doesn't jitter.
        self.bbox_world: Optional[tuple[float, float, float, float]] = None

        # ----- CameraFrame buffer (recent N, sorted by frame_idx for matching) -----
        # Each entry: (frame_idx, image_array, frame_id_str)
        self.camera_buffer: deque[tuple[int, np.ndarray, str]] = deque(
            maxlen=self.CAMERA_BUFFER_SIZE)
        self.n_camera: int = 0
        self.camera_latencies: deque[float] = deque(maxlen=30)

    def find_camera_match(self, target_idx: Optional[int]
                          ) -> Optional[tuple[int, np.ndarray, str]]:
        """Pick the buffered camera frame whose frame_idx is closest to the
        target. Returns (frame_idx, image, frame_id_str) or None if buffer
        empty / target is None."""
        if target_idx is None or not self.camera_buffer:
            # No BEV reference yet — fall back to most recent camera so the
            # panel isn't blank during startup.
            if self.camera_buffer:
                return self.camera_buffer[-1]
            return None
        # Linear scan is fine for buffer of 200; deque doesn't support binary
        # search anyway.
        best = min(self.camera_buffer, key=lambda e: abs(e[0] - target_idx))
        return best


def _grid_extent(width: int, height: int, resolution: float,
                 origin_x: float, origin_y: float) -> tuple[float, float, float, float]:
    """imshow extent (left, right, bottom, top) for world-coordinate axes."""
    return (origin_x, origin_x + width * resolution,
            origin_y, origin_y + height * resolution)


def _observed_bbox_world(obs_mask: np.ndarray, resolution: float,
                         origin_x: float, origin_y: float
                         ) -> Optional[tuple[float, float, float, float]]:
    """Tight bbox of observed cells in world (metres). None if nothing observed yet."""
    if not obs_mask.any():
        return None
    rows = np.where(obs_mask.any(axis=1))[0]
    cols = np.where(obs_mask.any(axis=0))[0]
    r_min, r_max = int(rows[0]), int(rows[-1])
    c_min, c_max = int(cols[0]), int(cols[-1])
    # imshow with origin='lower' has row 0 at the bottom; convert via the
    # extent y-axis directly.
    x_min = origin_x + c_min * resolution
    x_max = origin_x + (c_max + 1) * resolution
    y_min = origin_y + r_min * resolution
    y_max = origin_y + (r_max + 1) * resolution
    return (x_min, x_max, y_min, y_max)


def _union_bbox(a, b):
    """Union of two (xmin, xmax, ymin, ymax) bboxes; either may be None."""
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), max(a[1], b[1]),
            min(a[2], b[2]), max(a[3], b[3]))


# --------------------------------------------------------------------------- #
# Async NATS subscriber — registers handlers for BOTH subjects
# --------------------------------------------------------------------------- #
async def subscribe(latest: LatestFrame, *, nats_url: str,
                    grid_subject: str, camera_subject: str,
                    save_frames_dir: Optional[Path]) -> None:
    adapter = Adapter(nats_url)
    await adapter.connect()

    async def grid_handler(payload: bytes) -> None:
        t_recv = time.time()
        try:
            msg = pb.BEVGrid()
            msg.ParseFromString(payload)
            grid = np.asarray(msg.scores, dtype=np.float32).reshape(msg.height, msg.width)
            if len(msg.observation_count) == grid.size:
                obs = (np.asarray(msg.observation_count, dtype=np.uint32)
                       .reshape(msg.height, msg.width) > 0)
            else:
                obs = (grid > 0.0)
        except Exception:
            log.exception("failed to deserialize BEVGrid")
            return

        latency_ms = (t_recv - msg.header.timestamp) * 1000.0
        latest.grid = grid
        latest.obs_mask = obs
        latest.resolution = float(msg.resolution) or 1.0
        latest.origin_x = float(msg.origin_x)
        latest.origin_y = float(msg.origin_y)
        latest.grid_frame_id = msg.header.frame_id
        latest.grid_frame_idx = _extract_frame_idx(msg.header.frame_id)
        latest.pose_sigma = float(msg.pose_sigma_at_snapshot)
        latest.n_grid += 1
        latest.grid_latencies.append(latency_ms)

        new_bbox = _observed_bbox_world(obs, latest.resolution,
                                        latest.origin_x, latest.origin_y)
        latest.bbox_world = _union_bbox(latest.bbox_world, new_bbox)

        if save_frames_dir is not None:
            save_frames_dir.mkdir(parents=True, exist_ok=True)
            out = save_frames_dir / f"{msg.header.frame_id}.npy"
            np.save(out, grid)

    async def camera_handler(payload: bytes) -> None:
        t_recv = time.time()
        try:
            msg = pb.CameraFrame()
            msg.ParseFromString(payload)
            if Image is None:
                log.error("Pillow not installed — camera panel will not render")
                return
            from io import BytesIO
            if msg.encoding.lower() == "jpeg":
                img = Image.open(BytesIO(msg.data)).convert("RGB")
                arr = np.asarray(img, dtype=np.uint8)
            elif msg.encoding.lower() == "rgb8":
                arr = (np.frombuffer(msg.data, dtype=np.uint8)
                       .reshape(msg.height, msg.width, 3))
            else:
                log.warning("unsupported camera encoding %r — skipping", msg.encoding)
                return
        except Exception:
            log.exception("failed to deserialize CameraFrame")
            return

        latency_ms = (t_recv - msg.header.timestamp) * 1000.0
        idx = _extract_frame_idx(msg.header.frame_id)
        if idx is not None:
            latest.camera_buffer.append((idx, arr, msg.header.frame_id))
        latest.n_camera += 1
        latest.camera_latencies.append(latency_ms)

    await adapter.subscribe(grid_subject, grid_handler)
    log.info("subscribed to %s  (BEVGrid)", grid_subject)
    await adapter.subscribe(camera_subject, camera_handler)
    log.info("subscribed to %s  (CameraFrame)", camera_subject)

    # Idle forever — the matplotlib loop drives the program; this coroutine
    # just keeps the NATS client alive.
    while True:
        await asyncio.sleep(1.0)


# --------------------------------------------------------------------------- #
# Matplotlib live update — periodic timer redraws from `latest`
# --------------------------------------------------------------------------- #
def _nice_scalebar_length(span_m: float) -> tuple[float, str]:
    """Pick a 'nice' round metric scalebar length (~10-25% of span)."""
    target = span_m * 0.18
    candidates = [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000]
    for c in candidates:
        if c >= target:
            return float(c), f"{c} m" if c < 1000 else f"{c/1000:g} km"
    return float(candidates[-1]), f"{candidates[-1]} m"


def make_dashboard(latest: LatestFrame, *, refresh_ms: int, colormap: str,
                   shape: tuple[int, int], autozoom: bool, margin_m: float,
                   bg: str) -> None:
    # ---- Dark theme ----------------------------------------------------------
    plt.rcParams.update({
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "axes.edgecolor": "#cccccc",
        "axes.labelcolor": "#dddddd",
        "xtick.color": "#bbbbbb",
        "ytick.color": "#bbbbbb",
        "text.color": "#eeeeee",
        "savefig.facecolor": bg,
    })

    # Two panels side-by-side: camera (left) + BEV (right).
    fig, (ax_cam, ax_bev) = plt.subplots(
        1, 2, figsize=(15, 7.5),
        gridspec_kw={"width_ratios": [1.05, 1.0]},
    )
    fig.canvas.manager.set_window_title("Terra Perceive — live perception (NATS)")

    # ---- LEFT panel: camera --------------------------------------------------
    cam_blank = np.zeros((10, 10, 3), dtype=np.uint8)
    cam_img = ax_cam.imshow(cam_blank, interpolation="bilinear")
    ax_cam.set_xticks([])
    ax_cam.set_yticks([])
    ax_cam.set_title("waiting for camera…", fontsize=11, color="#cccccc",
                     loc="left", pad=8)
    cam_label = ax_cam.text(
        0.01, 0.99, "sensor.camera.rgb",
        transform=ax_cam.transAxes, va="top", ha="left",
        fontsize=10, color="#ffffff", fontweight="bold",
        bbox=dict(facecolor="#000000aa", edgecolor="none", pad=4),
    )

    # ---- RIGHT panel: BEV grid ----------------------------------------------
    cmap = matplotlib.colormaps.get_cmap(colormap).copy()
    cmap.set_bad(bg)
    bev_blank = np.zeros(shape, dtype=np.float32)
    bev_img = ax_bev.imshow(
        bev_blank, cmap=cmap, vmin=0.0, vmax=1.0, origin="lower",
        interpolation="nearest",
        extent=(-shape[1] / 2.0, shape[1] / 2.0,
                -shape[0] / 2.0, shape[0] / 2.0),
    )
    ax_bev.set_xlabel("x  (m, world frame)", fontsize=10)
    ax_bev.set_ylabel("y  (m, world frame)", fontsize=10)
    ax_bev.set_aspect("equal")
    ax_bev.grid(True, color="#333333", linewidth=0.4, linestyle="--", alpha=0.6)

    bev_title = ax_bev.set_title("waiting for BEVGrid…",
                                 fontsize=11, color="#cccccc", loc="left", pad=8)
    bev_label = ax_bev.text(
        0.01, 0.99, "perception.traversability.grid",
        transform=ax_bev.transAxes, va="top", ha="left",
        fontsize=10, color="#ffffff", fontweight="bold",
        bbox=dict(facecolor="#000000aa", edgecolor="none", pad=4),
    )

    cbar = fig.colorbar(bev_img, ax=ax_bev, fraction=0.046, pad=0.04)
    cbar.set_label("risk  (0 = safe · 1 = hazard)", fontsize=9, color="#dddddd")
    cbar.ax.yaxis.set_tick_params(color="#bbbbbb")
    cbar.outline.set_edgecolor("#666666")

    # ---- Header + footer -----------------------------------------------------
    fig.suptitle(
        "Terra Perceive   ·   live over NATS   ·   "
        "two independent producers, one consumer",
        fontsize=13, color="#ffffff", y=0.97, fontweight="bold",
    )
    footer = fig.text(
        0.01, 0.01,
        "broker=nats://localhost:4222   schema={BEVGrid, CameraFrame} v1",
        fontsize=8, color="#888888",
    )

    scalebar_holder: list = [None]

    fig.tight_layout(rect=(0, 0.03, 1, 0.93))

    def update(_frame_idx):
        artists = []

        # ---- Camera panel: pick the buffered frame closest to the BEV ------
        match = latest.find_camera_match(latest.grid_frame_idx)
        if match is not None:
            cam_idx, cam_arr, cam_fid = match
            cam_img.set_data(cam_arr)
            cam_img.set_extent((0, cam_arr.shape[1], cam_arr.shape[0], 0))
            cam_lat = (sum(latest.camera_latencies) / len(latest.camera_latencies)
                       if latest.camera_latencies else 0.0)
            # Sync indicator: how many frames of slop between the rendered
            # camera and the rendered BEV. Goal is small / zero-ish.
            if latest.grid_frame_idx is not None:
                delta = cam_idx - latest.grid_frame_idx
                sync = f"Δframe={delta:+d}"
            else:
                sync = "Δframe=?"
            ax_cam.set_title(
                f"frame_id={cam_fid}   {sync}   "
                f"buffered={len(latest.camera_buffer)}/{latest.CAMERA_BUFFER_SIZE}   "
                f"latency ≈ {cam_lat:.0f} ms",
                fontsize=11, color="#cccccc", loc="left", pad=8,
            )
        artists.extend([cam_img])

        # ---- BEV panel -------------------------------------------------------
        if latest.grid is not None and latest.obs_mask is not None:
            masked = np.where(latest.obs_mask, latest.grid, np.nan)
            bev_img.set_data(masked)
            bev_img.set_extent(_grid_extent(
                latest.grid.shape[1], latest.grid.shape[0],
                latest.resolution, latest.origin_x, latest.origin_y))

            if autozoom and latest.bbox_world is not None:
                x0, x1, y0, y1 = latest.bbox_world
                span = max(x1 - x0, y1 - y0) + 2 * margin_m
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                ax_bev.set_xlim(cx - span / 2.0, cx + span / 2.0)
                ax_bev.set_ylim(cy - span / 2.0, cy + span / 2.0)

                if scalebar_holder[0] is not None:
                    scalebar_holder[0].remove()
                length_m, label = _nice_scalebar_length(span)
                sb = AnchoredSizeBar(
                    ax_bev.transData, length_m, label, loc="lower right",
                    pad=0.5, color="#ffffff", frameon=True,
                    size_vertical=span * 0.005,
                    fontproperties=FontProperties(size=10, weight="bold"),
                )
                sb.patch.set_facecolor("#000000")
                sb.patch.set_alpha(0.55)
                sb.patch.set_edgecolor("#ffffff")
                ax_bev.add_artist(sb)
                scalebar_holder[0] = sb

            bev_lat = (sum(latest.grid_latencies) / len(latest.grid_latencies)
                       if latest.grid_latencies else 0.0)
            bev_title.set_text(
                f"frame_id={latest.grid_frame_id}   "
                f"pose_sigma={latest.pose_sigma:.3f} m   "
                f"received={latest.n_grid}   "
                f"latency ≈ {bev_lat:.0f} ms"
            )
        artists.extend([bev_img, bev_title])
        return artists

    anim = FuncAnimation(
        fig, update,
        interval=refresh_ms,
        cache_frame_data=False,
        blit=False,
    )
    fig._terra_anim = anim  # type: ignore[attr-defined]
    plt.show()


# --------------------------------------------------------------------------- #
# Entrypoint — runs asyncio loop on a background thread so matplotlib can own
# the main thread (most GUI backends require this).
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-subject", default="perception.traversability.grid")
    ap.add_argument("--camera-subject", default="sensor.camera.rgb")
    ap.add_argument("--nats-url", default="nats://localhost:4222")
    ap.add_argument("--refresh-ms", type=int, default=100)
    ap.add_argument("--colormap", default="viridis")
    ap.add_argument("--save-frames", type=Path, default=None)
    ap.add_argument("--shape", type=int, nargs=2, default=(1000, 1000),
                    help="initial BEV canvas shape; resized when first message arrives")
    ap.add_argument("--no-autozoom", action="store_true",
                    help="disable auto-zoom; show full world grid")
    ap.add_argument("--margin-m", type=float, default=20.0,
                    help="metres of margin around observed-cell bbox when autozooming")
    ap.add_argument("--bg", default="black",
                    help="background colour (default 'black'; try 'darkgray')")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    if Image is None:
        log.warning("Pillow not installed — camera panel will be empty. "
                    "Run: pip install pillow")

    latest = LatestFrame()

    # Run the NATS subscriber on a background thread so matplotlib's mainloop
    # owns the main thread.
    import threading
    def run_subscriber() -> None:
        asyncio.run(subscribe(
            latest,
            nats_url=args.nats_url,
            grid_subject=args.grid_subject,
            camera_subject=args.camera_subject,
            save_frames_dir=args.save_frames,
        ))
    t = threading.Thread(target=run_subscriber, daemon=True)
    t.start()

    make_dashboard(
        latest,
        refresh_ms=args.refresh_ms,
        colormap=args.colormap,
        shape=tuple(args.shape),
        autozoom=not args.no_autozoom,
        margin_m=args.margin_m,
        bg=args.bg,
    )


if __name__ == "__main__":
    main()
