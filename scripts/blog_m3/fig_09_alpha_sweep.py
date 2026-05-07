"""
fig_09_alpha_sweep.py — Ablation C, alpha sweep on the hero cell.

Same world cell as fig_08, but now four EMA runs differing only in the
smoothing parameter alpha. Visual story: low alpha (0.1) is sluggish,
high alpha (0.7) tracks every observation but jitters, alpha=0.3 is the
shipped default.

Output: docs/assets/m3/ablation_c_alpha_sweep.svg
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from _utils import auto_xlim, cell_history  # noqa: E402

REPO = THIS.parents[2]
M3 = REPO / "results" / "m3"
HERO = THIS.parent / "HERO_CELL.json"
OUT = REPO / "docs" / "assets" / "m3" / "ablation_c_alpha_sweep.svg"
SAMPLE_EVERY = 10

LINES = [
    ("slam_ema_alpha0p1_perframe", "a = 0.1  (inert)",        "#1f77b4"),
    ("slam_ema_perframe",          "a = 0.3  (default)",      "#d62728"),
    ("slam_ema_alpha0p5_perframe", "a = 0.5",                 "#2ca02c"),
    ("slam_ema_alpha0p7_perframe", "a = 0.7  (reactive)",     "#ff7f0e"),
]


def main() -> None:
    if not HERO.exists():
        sys.exit(f"missing {HERO} — run pick_hero_cell.py first")
    hero = json.loads(HERO.read_text())

    fig, ax = plt.subplots(figsize=(10, 5.5))

    histories: list = []
    for run, label, color in LINES:
        run_dir = M3 / run
        if not run_dir.exists():
            print(f"  skip missing: {run_dir}")
            continue
        df = cell_history(run_dir, hero["row"], hero["col"], every=SAMPLE_EVERY)
        histories.append(df)
        ax.plot(df["frame_id"], df["risk"], label=label, color=color,
                linewidth=1.6, marker=".", markersize=4)

    xlim = auto_xlim(histories, margin=50)
    ax.set_xlabel(f"frame index  (cell first observed at frame ~{xlim[0] + 50})")
    ax.set_ylabel("cell risk")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(*xlim)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax.set_title(
        f"Ablation C - same cell, alpha sweep on EMA "
        f"(world ({hero['world_x']:.0f}, {hero['world_y']:.0f}) m)",
        fontsize=12, loc="left",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
