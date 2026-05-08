#!/usr/bin/env python3
"""P2-M6 m13 animation: 6-panel growing v(t) curves, kinematic vs CBF.

For each of the six scenarios, animate kinematic and CBF v(t) in lockstep,
one frame per supervisor step. The output is a single MP4 with a 2x3 panel
grid; useful for the m13 blog.

Usage:
    python scripts/m6/animate_velocity_profiles.py \\
        --kinematic-root results_m6/cbf_kinematic \\
        --cbf-root       results_m6/cbf_cbf \\
        --out            /media/nishant/SeeGayt2/terra_perceive/m6_animations/velocity_profiles.mp4 \\
        --fps 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

SCENARIOS = ["head_on", "angled_20", "occluded", "multi_worker", "far_pass", "edge_of_arc"]


def load_events(path: Path):
    if not path.exists():
        return None
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kinematic-root", default="results_m6/cbf_kinematic")
    p.add_argument("--cbf-root", default="results_m6/cbf_cbf")
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()

    kins = [load_events(Path(args.kinematic_root) / s / "events.csv") for s in SCENARIOS]
    cbfs = [load_events(Path(args.cbf_root) / s / "events.csv") for s in SCENARIOS]
    n_frames = max(
        max(len(k) if k is not None else 0 for k in kins),
        max(len(c) if c is not None else 0 for c in cbfs),
    )

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    lines_k, lines_c = [], []
    for i, scen in enumerate(SCENARIOS):
        ax = axes[i]
        ax.set_title(scen, fontsize=10)
        ax.set_xlabel("t (s)", fontsize=8)
        ax.set_ylabel("v (m/s)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        # Pre-size axes from full data
        all_t = np.concatenate([kins[i]["t"] if kins[i] is not None else np.array([0.0, 1.0]),
                                cbfs[i]["t"] if cbfs[i] is not None else np.array([0.0, 1.0])])
        all_v = np.concatenate([kins[i]["vel_after"] if kins[i] is not None else np.array([0.0, 2.0]),
                                cbfs[i]["vel_after"] if cbfs[i] is not None else np.array([0.0, 2.0])])
        ax.set_xlim(0, max(0.1, all_t.max()))
        ax.set_ylim(-0.05, max(0.5, all_v.max() * 1.1))
        lk, = ax.plot([], [], color="#1f77b4", linewidth=1.8, label="kinematic")
        lc, = ax.plot([], [], color="#d62728", linewidth=1.8, label="CBF")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
        lines_k.append(lk)
        lines_c.append(lc)

    fig.suptitle("Velocity profile: kinematic supervisor vs CBF clamp (animated)",
                 fontsize=12)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, bitrate=2400)
    with writer.saving(fig, str(out), dpi=130):
        for fi in range(n_frames):
            for i in range(len(SCENARIOS)):
                if kins[i] is not None:
                    j = min(fi, len(kins[i]) - 1) + 1
                    lines_k[i].set_data(kins[i]["t"][:j], kins[i]["vel_after"][:j])
                if cbfs[i] is not None:
                    j = min(fi, len(cbfs[i]) - 1) + 1
                    lines_c[i].set_data(cbfs[i]["t"][:j], cbfs[i]["vel_after"][:j])
            writer.grab_frame()
    print(f"[wrote] {out}  ({n_frames} frames, {n_frames / args.fps:.1f}s @ {args.fps} fps)")


if __name__ == "__main__":
    main()
