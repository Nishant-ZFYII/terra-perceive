"""
fig_10_decay_sweep.py — Ablation D, decay sweep on the hero cell.

Three EMA runs differing only in the temporal decay rate (0, 0.01/s,
0.1/s). Honest expected story: on a 285-second RELLIS sequence, decay
barely moves the curves at all — and that *is* the takeaway. Decay is
infrastructure for multi-minute construction-site scenarios where cells
should age out, not a knob you'd tune for short driving sequences.

Output: docs/assets/m3/ablation_d_decay.svg
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
OUT = REPO / "docs" / "assets" / "m3" / "ablation_d_decay.svg"
SAMPLE_EVERY = 10

LINES = [
    ("slam_ema_perframe",            "decay = 0      (default)",   "#1f77b4", "-"),
    ("slam_ema_decay0p01_perframe",  "decay = 0.01 / s",            "#2ca02c", "-"),
    ("slam_ema_decay0p1_perframe",   "decay = 0.1  / s",            "#d62728", "-"),
]


def main() -> None:
    if not HERO.exists():
        sys.exit(f"missing {HERO} — run pick_hero_cell.py first")
    hero = json.loads(HERO.read_text())

    fig, ax = plt.subplots(figsize=(10, 5.5))

    histories: list = []
    for run, label, color, ls in LINES:
        run_dir = M3 / run
        if not run_dir.exists():
            print(f"  skip missing: {run_dir}")
            continue
        df = cell_history(run_dir, hero["row"], hero["col"], every=SAMPLE_EVERY)
        histories.append(df)
        ax.plot(df["frame_id"], df["risk"], label=label, color=color,
                linestyle=ls, linewidth=1.6, marker=".", markersize=4)

    xlim = auto_xlim(histories, margin=50)
    ax.set_xlabel(f"frame index  (cell first observed at frame ~{xlim[0] + 50})")
    ax.set_ylabel("cell risk")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(*xlim)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax.set_title(
        f"Ablation D - decay barely moves on a 285 s sequence "
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
