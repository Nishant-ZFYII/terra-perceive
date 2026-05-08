#!/usr/bin/env python3
"""P2-M6: block diagrams for the m12 and m13 blog posts.

Generates SVG/PNG block diagrams using matplotlib. No external diagram tool.
Outputs land under docs/assets/m12/ and docs/assets/m13/.

Diagrams produced:
    m12/pipeline_diagram.svg    — full per-frame traversability pipeline
    m12/confidence_factors.svg  — three multiplicative confidence factors
    m13/supervisor_flow.svg     — kinematic vs CBF branch in evaluate()
    m13/cbf_math_blocks.svg     — CBF math dataflow (h -> hdot -> a_safe -> scale)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, xy, w, h, text, *, fc="#dde6f0", ec="#1f5c8b", fontsize=9,
            wrap=True, fontweight="normal"):
    box = FancyBboxPatch(xy, w, h,
                          boxstyle="round,pad=0.03,rounding_size=0.06",
                          facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, wrap=wrap)


def add_arrow(ax, p0, p1, *, color="#444", style="->", lw=1.4, label=None,
              label_offset=(0.0, 0.18)):
    arr = FancyArrowPatch(p0, p1, arrowstyle=style, color=color,
                          mutation_scale=12, lw=lw,
                          connectionstyle="arc3,rad=0.0")
    ax.add_patch(arr)
    if label:
        mx = 0.5 * (p0[0] + p1[0]) + label_offset[0]
        my = 0.5 * (p0[1] + p1[1]) + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=8,
                color="#666", style="italic")


def setup_axes(width=12, height=4, xlim=(0, 12), ylim=(0, 4)):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# -- m12 pipeline diagram -----------------------------------------------------

def make_m12_pipeline(out_path: Path) -> None:
    fig, ax = setup_axes(width=14, height=3.4, xlim=(0, 14), ylim=(0, 3.4))

    # Row of boxes left to right.
    boxes = [
        (0.2, 1.4, 1.7, 0.9, "LiDAR\nframe", "#fce4ec", "#9c1f4d"),
        (2.2, 1.4, 1.6, 0.9, "RANSAC\nground seg", "#dde6f0", "#1f5c8b"),
        (4.1, 1.4, 1.6, 0.9, "0.5 m\ncell binning", "#dde6f0", "#1f5c8b"),
        (6.0, 1.4, 1.7, 0.9, "PCA per cell\n(C → λ, n)", "#dde6f0", "#1f5c8b"),
        (8.0, 1.4, 1.7, 0.9, "σ(r)\nnoise model", "#fff3d6", "#a06700"),
    ]
    for (x, y, w, h, t, fc, ec) in boxes:
        add_box(ax, (x, y), w, h, t, fc=fc, ec=ec, fontsize=9)
    # Arrows along the row.
    add_arrow(ax, (1.9, 1.85), (2.2, 1.85))
    add_arrow(ax, (3.8, 1.85), (4.1, 1.85))
    add_arrow(ax, (5.7, 1.85), (6.0, 1.85))
    add_arrow(ax, (7.7, 1.85), (8.0, 1.85))

    # Branch into heuristic / probabilistic.
    add_box(ax, (10.1, 2.45), 3.5, 0.7,
            "heuristic:  c = min(1, N/20) × max(0, 1 - r/30)",
            fc="#f0e8e8", ec="#7a3a3a", fontsize=8.5)
    add_box(ax, (10.1, 0.55), 3.5, 0.7,
            "probabilistic:  c = planarity × sample × range",
            fc="#e7f4e8", ec="#3a7a3f", fontsize=8.5)
    add_box(ax, (10.1, 1.4), 3.5, 0.9,
            "branch on\nconfidence_mode",
            fc="#fff3d6", ec="#a06700", fontsize=9, fontweight="bold")
    add_arrow(ax, (9.7, 1.85), (10.1, 1.85))
    add_arrow(ax, (11.85, 2.3), (11.85, 2.45), label="--mode heuristic",
              label_offset=(1.5, 0.0))
    add_arrow(ax, (11.85, 1.4), (11.85, 1.25), label="--mode probabilistic",
              label_offset=(1.6, 0.0))

    # Caption.
    ax.text(7.0, 0.1,
            "P2-M6 traversability pipeline — eigenvalues already computed inline are now reused for probabilistic confidence",
            ha="center", va="bottom", fontsize=8.5, color="#444",
            style="italic")
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out_path}")


# -- m12 confidence factors diagram -------------------------------------------

def make_m12_confidence_factors(out_path: Path) -> None:
    fig, ax = setup_axes(width=12, height=3.6, xlim=(0, 12), ylim=(0, 3.6))

    # Three input boxes feeding into three formula boxes, then one output.
    inputs = [
        (0.5, 2.5, "λ_min, λ_max"),
        (0.5, 1.7, "point count N"),
        (0.5, 0.9, "σ(r) noise"),
    ]
    for (x, y, t) in inputs:
        add_box(ax, (x, y - 0.25), 1.6, 0.55, t, fc="#fce4ec", ec="#9c1f4d",
                fontsize=8.5)

    factors = [
        (3.6, 2.5, "planarity\n= 1 - (λ_min - σ²) / λ_max"),
        (3.6, 1.7, "sample_factor\n= 1 - exp(-N / 10)"),
        (3.6, 0.9, "range_factor\n= λ_max / (λ_max + σ²)"),
    ]
    for (x, y, t) in factors:
        add_box(ax, (x, y - 0.35), 4.2, 0.7, t, fc="#dde6f0", ec="#1f5c8b",
                fontsize=8.5)
        add_arrow(ax, (2.1, y), (3.6, y))

    # Final product.
    add_box(ax, (9.2, 1.45), 2.4, 0.6,
            "c_prob ∈ [0, 1]\n(product)",
            fc="#e7f4e8", ec="#3a7a3f", fontsize=9, fontweight="bold")
    for y in (2.5, 1.7, 0.9):
        add_arrow(ax, (7.8, y), (9.2, 1.75), color="#3a7a3f")

    ax.text(6.0, 0.2,
            "Three multiplicative factors. All bounded in [0, 1]; product is in [0, 1].",
            ha="center", va="bottom", fontsize=8.5, color="#444",
            style="italic")
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out_path}")


# -- m13 supervisor flow ------------------------------------------------------

def make_m13_supervisor_flow(out_path: Path) -> None:
    fig, ax = setup_axes(width=14, height=4.2, xlim=(0, 14), ylim=(0, 4.2))

    add_box(ax, (0.3, 1.7), 2.3, 0.9,
            "evaluate(v, d_w,\nv_w, μ, t)",
            fc="#fce4ec", ec="#9c1f4d", fontsize=9, fontweight="bold")
    add_box(ax, (3.2, 1.7), 2.4, 0.9,
            "pre_evaluate_health()\nLiDAR timeout?",
            fc="#fff3d6", ec="#a06700", fontsize=9)
    add_arrow(ax, (2.6, 2.15), (3.2, 2.15))

    # E-stop early-out.
    add_box(ax, (3.2, 0.4), 2.4, 0.7,
            "EMERGENCY_STOP\nscale = 0",
            fc="#f0e8e8", ec="#7a3a3a", fontsize=8.5)
    add_arrow(ax, (4.4, 1.7), (4.4, 1.1), label="timeout", label_offset=(0.7, 0.0))

    # Branch.
    add_box(ax, (6.2, 1.7), 2.4, 0.9,
            "branch on\nsafety_mode",
            fc="#fff3d6", ec="#a06700", fontsize=9, fontweight="bold")
    add_arrow(ax, (5.6, 2.15), (6.2, 2.15), label="ok", label_offset=(0.0, 0.18))

    # Kinematic path.
    add_box(ax, (9.2, 2.7), 4.0, 0.85,
            "kinematic path:\nTTC threshold tree",
            fc="#dde6f0", ec="#1f5c8b", fontsize=8.5)
    # CBF path.
    add_box(ax, (9.2, 0.9), 4.0, 0.85,
            "evaluate_cbf:\nh, ḣ, a_safe, scale",
            fc="#e7f4e8", ec="#3a7a3f", fontsize=8.5)

    add_arrow(ax, (8.6, 2.4), (9.2, 3.1), label="kinematic",
              label_offset=(0.6, 0.05))
    add_arrow(ax, (8.6, 1.9), (9.2, 1.3), label="cbf",
              label_offset=(0.5, -0.1))

    # Output funnel.
    add_box(ax, (5.5, 0.05), 4.0, 0.55,
            "SafetyIntervention(level, scale, reason)  →  events.csv",
            fc="#fce4ec", ec="#9c1f4d", fontsize=8.5, fontweight="bold")
    add_arrow(ax, (10.7, 2.7), (7.5, 0.6), color="#666")
    add_arrow(ax, (10.7, 0.9), (7.5, 0.6), color="#666")
    add_arrow(ax, (4.4, 0.4), (6.2, 0.35), color="#666")

    ax.text(7.0, 3.95,
            "SafetySupervisor::evaluate — kinematic path bit-for-bit unchanged from P1-M5; CBF path is the new branch",
            ha="center", va="top", fontsize=9, fontweight="bold", color="#222")
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out_path}")


# -- m13 CBF math blocks ------------------------------------------------------

def make_m13_cbf_math(out_path: Path) -> None:
    fig, ax = setup_axes(width=13, height=3.6, xlim=(0, 13), ylim=(0, 3.6))

    # Inputs at left.
    inputs = [
        (0.3, 2.5, "v, d_w, v_rel"),
        (0.3, 1.7, "μ, g, t_react"),
        (0.3, 0.9, "γ, d_safe_min, Δt"),
    ]
    for (x, y, t) in inputs:
        add_box(ax, (x, y - 0.25), 2.0, 0.5, t, fc="#fce4ec", ec="#9c1f4d",
                fontsize=8.5)

    # Step 1: stopping distance.
    add_box(ax, (3.0, 1.7), 2.2, 0.9,
            "d_stop\n= v² / (2μg) + v·t_react",
            fc="#dde6f0", ec="#1f5c8b", fontsize=8.5)
    for y in (2.5, 1.7):
        add_arrow(ax, (2.3, y), (3.0, 2.15))

    # Step 2: barrier.
    add_box(ax, (5.6, 1.7), 2.2, 0.9,
            "h(v, d_w)\n= d_w − d_stop − d_safe_min",
            fc="#dde6f0", ec="#1f5c8b", fontsize=8.5)
    add_arrow(ax, (5.2, 2.15), (5.6, 2.15))

    # Step 3: CBF clamp.
    add_box(ax, (8.2, 1.7), 2.2, 0.9,
            "a_safe\n= (γ·h − v_rel) / A",
            fc="#dde6f0", ec="#1f5c8b", fontsize=8.5)
    add_arrow(ax, (7.8, 2.15), (8.2, 2.15))
    add_arrow(ax, (1.3, 1.0), (8.2, 1.85), color="#a06700")
    ax.text(4.7, 1.18, "γ feeds CBF clamp", fontsize=7.5, color="#a06700",
            style="italic")

    # Step 4: scale.
    add_box(ax, (10.8, 1.7), 1.9, 0.9,
            "scale\n= max(0, v + a_safe·Δt) / v",
            fc="#e7f4e8", ec="#3a7a3f", fontsize=8.5, fontweight="bold")
    add_arrow(ax, (10.4, 2.15), (10.8, 2.15))

    ax.text(6.5, 0.2,
            "CBF dataflow — every block is a single closed-form expression; no QP, no iteration",
            ha="center", va="bottom", fontsize=8.5, color="#444",
            style="italic")
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default="docs/assets")
    args = p.parse_args()
    out_root = Path(args.out_root)
    (out_root / "m12").mkdir(parents=True, exist_ok=True)
    (out_root / "m13").mkdir(parents=True, exist_ok=True)

    make_m12_pipeline(out_root / "m12" / "pipeline_diagram.svg")
    make_m12_confidence_factors(out_root / "m12" / "confidence_factors.svg")
    make_m13_supervisor_flow(out_root / "m13" / "supervisor_flow.svg")
    make_m13_cbf_math(out_root / "m13" / "cbf_math_blocks.svg")


if __name__ == "__main__":
    main()
