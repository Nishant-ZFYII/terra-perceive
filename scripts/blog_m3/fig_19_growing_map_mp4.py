"""
fig_19_growing_map_mp4.py — single growing-map MP4.

Builds a 30-ish second clip of the slam_ema_covg2o run filling in over
all 2847 frames. Single panel, viridis colormap, fixed world bbox so
the map appears to grow within a stable frame.

Output: docs/assets/m3/single_map_growing.mp4
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from _utils import list_snapshots, snapshot_to_dense, trajectory_bbox  # noqa: E402

REPO = THIS.parents[2]
RUN = REPO / "results" / "m3" / "slam_ema_covg2o_perframe"
OUT = REPO / "docs" / "assets" / "m3" / "single_map_growing.mp4"

EVERY = 4              # 2847 / 4 ~= 711 frames -> 23 s at 30 fps
FPS = 30
RES_M = 0.5
N = 1000
ORIGIN_X = -RES_M * N / 2.0
ORIGIN_Y = -RES_M * N / 2.0
EXTENT = (ORIGIN_X, ORIGIN_X + N * RES_M, ORIGIN_Y, ORIGIN_Y + N * RES_M)
MARGIN_M = 25.0


def main() -> None:
    snaps = list_snapshots(RUN, EVERY)
    print(f"animating {len(snaps)} frames at {FPS} fps "
          f"-> {len(snaps) / FPS:.1f} s clip")

    traj = pd.read_csv(RUN / "trajectory.csv")
    xlim, ylim = trajectory_bbox([RUN / "trajectory.csv"], MARGIN_M)

    cmap = matplotlib.colormaps.get_cmap("viridis").copy()
    cmap.set_bad("black")

    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="black")
    ax.set_facecolor("black")
    blank = np.full((N, N), np.nan, dtype=np.float32)
    img = ax.imshow(blank, cmap=cmap, vmin=0.0, vmax=1.0,
                    extent=EXTENT, origin="lower", interpolation="nearest")
    traj_line, = ax.plot([], [], color="#ff6b6b", linewidth=0.8, alpha=0.6)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, world frame)", color="#dddddd")
    ax.set_ylabel("y (m, world frame)", color="#dddddd")
    ax.tick_params(colors="#bbbbbb", labelsize=9)
    for sp in ax.spines.values(): sp.set_color("#666666")
    ax.grid(True, color="#333333", linewidth=0.4, linestyle="--", alpha=0.5)

    title = ax.set_title("frame 0", color="white", fontsize=12, loc="left")
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("risk  (0 = safe, 1 = hazard)", color="#dddddd")
    cbar.ax.yaxis.set_tick_params(color="#bbbbbb")

    def update(i):
        frame_id, path = snaps[i]
        grid = snapshot_to_dense(path)
        masked = np.where(grid > 0.0, grid, np.nan)
        img.set_data(masked)
        traj_line.set_data(traj["tx"][:frame_id + 1], traj["ty"][:frame_id + 1])
        title.set_text(f"slam_ema_covg2o  -  frame {frame_id} / 2847")
        if i % 25 == 0:
            print(f"  frame {i}/{len(snaps)} (sequence frame {frame_id})")
        return [img, traj_line, title]

    anim = FuncAnimation(fig, update, frames=len(snaps),
                         interval=1000 // FPS, blit=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT, writer=FFMpegWriter(fps=FPS, codec="libx264",
                                       bitrate=4000, extra_args=["-pix_fmt", "yuv420p"]))
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
