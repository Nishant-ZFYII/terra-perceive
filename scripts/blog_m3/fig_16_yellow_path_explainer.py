"""
fig_16_yellow_path_explainer.py — debugging-story side-by-side.

Why the trajectory is painted bright yellow in every BEV map: the per-frame
TraversabilityGrid measures local height variance, and grass at a Polaris-
sized vehicle's wheel height looks identical to a bush in pure LiDAR
geometry. The story isn't a bug; it's the limit of geometric ground
segmentation in unstructured off-road environments (Jiang et al. 2020).

Two-panel figure:
  Left  - the RELLIS pylon camera frame at index FRAME_IDX
  Right - the BEV snapshot at the same frame, cropped to the vehicle's
          local 60 m x 60 m window, with the trajectory wake annotated

Output: docs/assets/m3/yellow_path_explainer.png
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from _utils import load_bev_grid  # noqa: E402

REPO = THIS.parents[2]
RUN = REPO / "results" / "m3" / "slam_ema_covg2o_perframe"
CAM_DIR = REPO / "data" / "RELLIS-3D" / "Rellis_3D_pylon_camera_node" / \
          "Rellis-3D" / "00000" / "pylon_camera_node"
OUT = REPO / "docs" / "assets" / "m3" / "yellow_path_explainer.png"

FRAME_IDX = 1500   # mid-sequence — vehicle is well into the grassy trail
LOCAL_HALF_M = 30.0  # +/- 30 m around vehicle on the BEV crop


def find_camera_frame(idx: int) -> Path:
    pattern = re.compile(rf"^frame{idx:06d}-")
    for p in sorted(CAM_DIR.iterdir()):
        if pattern.match(p.name):
            return p
    raise FileNotFoundError(f"no camera frame matching index {idx} in {CAM_DIR}")


def main() -> None:
    cam_path = find_camera_frame(FRAME_IDX)
    cam_img = np.asarray(Image.open(cam_path).convert("RGB"))

    # BEV: use the FINAL grid (more cells visible) but crop to the vehicle
    # position at the chosen frame. The story is about what the LiDAR sees
    # along the path, and the final grid contains the same yellow wake the
    # blog will reference.
    grid, extent = load_bev_grid(RUN)
    masked = np.where(grid > 0.0, grid, np.nan)

    traj = pd.read_csv(RUN / "trajectory.csv")
    pose = traj[traj["frame_id"] == FRAME_IDX].iloc[0]
    vx, vy = float(pose["tx"]), float(pose["ty"])

    cmap = matplotlib.colormaps.get_cmap("viridis").copy()
    cmap.set_bad("black")

    fig, (ax_cam, ax_bev) = plt.subplots(1, 2, figsize=(15, 7),
                                         gridspec_kw={"width_ratios": [1.6, 1.0]})

    # Left: camera
    ax_cam.imshow(cam_img)
    ax_cam.set_xticks([])
    ax_cam.set_yticks([])
    ax_cam.set_title(
        f"RELLIS pylon camera, frame {FRAME_IDX}\n"
        "(grass at wheel height; the vehicle drove right through this)",
        fontsize=12, loc="left",
    )

    # Right: BEV crop
    ax_bev.set_facecolor("black")
    ax_bev.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0,
                  extent=extent, origin="lower", interpolation="nearest")
    ax_bev.plot(traj["tx"], traj["ty"], color="#ff6b6b", linewidth=0.8,
                alpha=0.7, label="full trajectory")
    ax_bev.plot([vx], [vy], marker="o", markersize=10, color="#ff3b3b",
                markeredgecolor="white", markeredgewidth=1.5, label="vehicle here")
    ax_bev.set_xlim(vx - LOCAL_HALF_M, vx + LOCAL_HALF_M)
    ax_bev.set_ylim(vy - LOCAL_HALF_M, vy + LOCAL_HALF_M)
    ax_bev.set_aspect("equal")
    ax_bev.set_xlabel("x (m, world frame)")
    ax_bev.set_ylabel("y (m, world frame)")
    ax_bev.tick_params(colors="#bbbbbb", labelsize=9)
    for spine in ax_bev.spines.values():
        spine.set_color("#666666")
    ax_bev.grid(True, color="#333333", linewidth=0.3, linestyle="--", alpha=0.5)
    ax_bev.legend(loc="lower right", fontsize=9, frameon=True,
                  facecolor="#000000", edgecolor="#555555", labelcolor="white")
    ax_bev.set_title(
        f"BEV at the same world location\n"
        "(yellow ribbon = high local height variance = 'looks like obstacle')",
        fontsize=12, loc="left", color="black",
    )

    fig.suptitle(
        "Why the path is bright yellow",
        fontsize=15, y=1.02, fontweight="bold",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    print(f"camera source: {cam_path.name}")
    print(f"BEV crop center: world ({vx:.1f}, {vy:.1f}) m")


if __name__ == "__main__":
    main()
