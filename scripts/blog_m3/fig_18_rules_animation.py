"""
fig_18_rules_animation.py — Ablation B animation: 3 update rules in lockstep.

Three panels (EMA / log-odds / overwrite) animated together at the same
frame index. Same SLAM poses, same LiDAR, only the update rule differs;
the panels diverge in the temporal-smoothing characteristics that scalar
coverage cannot show.

Output: docs/assets/m3/ablation_b_animation.mp4
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
M3 = REPO / "results" / "m3"
OUT = REPO / "docs" / "assets" / "m3" / "ablation_b_animation.mp4"

EVERY = 12             # 2847/12 ~= 237 frames -> 8 s at 30 fps
FPS = 30
RES_M = 0.5
N = 1000
ORIGIN_X = -RES_M * N / 2.0
ORIGIN_Y = -RES_M * N / 2.0
EXTENT = (ORIGIN_X, ORIGIN_X + N * RES_M, ORIGIN_Y, ORIGIN_Y + N * RES_M)
MARGIN_M = 25.0

RUNS = [
    ("slam_ema_perframe",       "EMA  (a = 0.3)"),
    ("slam_logodds_perframe",   "Log-odds  (OctoMap-style)"),
    ("slam_overwrite_perframe", "Overwrite  (no memory)"),
]


def main() -> None:
    # All three runs use the same SLAM poses, so list snapshots from one and
    # mirror across the three (snapshot frame_id is identical across them).
    snaps_ref = list_snapshots(M3 / RUNS[0][0], EVERY)
    print(f"animating {len(snaps_ref)} frames -> {len(snaps_ref) / FPS:.1f} s clip")

    traj = pd.read_csv(M3 / RUNS[0][0] / "trajectory.csv")
    xlim, ylim = trajectory_bbox([M3 / RUNS[0][0] / "trajectory.csv"], MARGIN_M)

    cmap = matplotlib.colormaps.get_cmap("viridis").copy()
    cmap.set_bad("black")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), facecolor="black")
    blank = np.full((N, N), np.nan, dtype=np.float32)
    imgs = []
    traj_lines = []
    titles = []
    for ax, (run, pretty) in zip(axes, RUNS):
        ax.set_facecolor("black")
        im = ax.imshow(blank, cmap=cmap, vmin=0.0, vmax=1.0,
                       extent=EXTENT, origin="lower", interpolation="nearest")
        line, = ax.plot([], [], color="#ff6b6b", linewidth=0.6, alpha=0.55)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.tick_params(colors="#bbbbbb", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#666666")
        ax.grid(True, color="#333333", linewidth=0.3, linestyle="--", alpha=0.5)
        t = ax.set_title(pretty, color="white", fontsize=12, loc="left")
        imgs.append(im); traj_lines.append(line); titles.append(t)

    suptitle = fig.suptitle("Ablation B - frame 0 / 2847",
                            color="white", fontsize=14, y=0.99)

    def update(i):
        frame_id, _ = snaps_ref[i]
        for (run, _), im, line, t_ in zip(RUNS, imgs, traj_lines, titles):
            snap_path = M3 / run / "snapshots" / f"frame_{frame_id:05d}.csv"
            if not snap_path.exists():
                continue
            grid = snapshot_to_dense(snap_path)
            im.set_data(np.where(grid > 0.0, grid, np.nan))
        for line in traj_lines:
            line.set_data(traj["tx"][:frame_id + 1], traj["ty"][:frame_id + 1])
        suptitle.set_text(f"Ablation B - frame {frame_id} / 2847")
        if i % 20 == 0:
            print(f"  frame {i}/{len(snaps_ref)} (seq {frame_id})")
        return [*imgs, *traj_lines, *titles, suptitle]

    anim = FuncAnimation(fig, update, frames=len(snaps_ref),
                         interval=1000 // FPS, blit=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT, writer=FFMpegWriter(fps=FPS, codec="libx264",
                                       bitrate=4000, extra_args=["-pix_fmt", "yuv420p"]))
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
