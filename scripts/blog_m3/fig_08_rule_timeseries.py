"""
fig_08_rule_timeseries.py — Ablation B's hero figure.

Single world cell traced over time across three update rules. The visual
story the scalar coverage couldn't tell:
  - EMA fades smoothly with each new observation (1-pole IIR).
  - Log-odds snaps toward the saturating clamp on consecutive hits.
  - Overwrite jumps to whatever the latest LiDAR sweep said, no memory.

The hero cell is selected once by pick_hero_cell.py and reused here, in
fig_09 (alpha sweep) and fig_10 (decay) so all three figures point at the
same physical location.

Output: docs/assets/m3/ablation_b_cell_timeseries.svg
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
OUT = REPO / "docs" / "assets" / "m3" / "ablation_b_cell_timeseries.svg"
SAMPLE_EVERY = 10  # every 10th snapshot

LINES = [
    ("slam_ema_perframe",       "EMA (a=0.3)",              "#1f77b4", "-"),
    ("slam_logodds_perframe",   "Log-odds (OctoMap-style)", "#d62728", "-"),
    ("slam_overwrite_perframe", "Overwrite (no memory)",    "#2ca02c", "--"),
]


def main() -> None:
    if not HERO.exists():
        sys.exit(f"missing {HERO} — run pick_hero_cell.py first")
    hero = json.loads(HERO.read_text())
    print(f"hero cell: csv_row={hero['row']}, csv_col={hero['col']}, "
          f"world=({hero['world_x']:.1f}, {hero['world_y']:.1f})")

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
    ax.set_ylabel("cell risk  (0 = safe, 1 = hazard)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(*xlim)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax.set_title(
        f"Ablation B - same cell at world ({hero['world_x']:.0f}, "
        f"{hero['world_y']:.0f}) m, three update rules over 2847 frames",
        fontsize=12, loc="left",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
