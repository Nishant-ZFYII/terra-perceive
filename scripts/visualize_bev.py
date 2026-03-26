"""
visualize_bev.py — BEV traversability grid visualization.
P1-M3.6: The "wow" image for your README and demo.

Usage:
    # Single frame via pipeline_runner:
    python scripts/visualize_bev.py --bin data/RELLIS-3D/.../000000.bin --out results/frame0

    # Pre-exported grid (skip C++ step):
    python scripts/visualize_bev.py --grid output_grid.bin --out results/frame0

    # Animated GIF from all frames in a directory:
    python scripts/visualize_bev.py --gif data/RELLIS-3D/.../os1_cloud_node_kitti_bin --out bev_animation.gif

    # Synthetic demo (no data needed):
    python scripts/visualize_bev.py --synthetic --out results/synthetic

Outputs (per single frame):
    <out>_2d.png    — Figure 4 style: green/yellow/red/gray confidence-gated BEV
    <out>_6panel.png — Figure 3 style: 6-panel feature heatmaps
    <out>_3d.png    — 3D surface colored by risk
"""

import argparse
import os
import subprocess
import tempfile

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Grid dimensions — must match pipeline_config.yaml
ROWS, COLS = 70, 60
FIELDS = 6          # order: risk, confidence, slope_deg, roughness, step_height, mean_z
X_MIN, X_MAX = -5.0, 30.0
Y_MIN, Y_MAX = -15.0, 15.0

PIPELINE_RUNNER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build/construction_perception/pipeline_runner"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_real_grid(path):
    """Load grid exported by pipeline_runner.
    File layout: rows*cols*6 float32 values, written row-major.
    Returns 6 arrays of shape (ROWS, COLS): risk, confidence, slope_deg,
    roughness, step_height, mean_z.
    """
    data = np.fromfile(path, dtype=np.float32).reshape(ROWS, COLS, FIELDS)
    return (data[:, :, 0], data[:, :, 1], data[:, :, 2],
            data[:, :, 3], data[:, :, 4], data[:, :, 5])


def make_synthetic_grid():
    """Synthetic demo: safe center, two hazard patches, unknown border."""
    risk = np.zeros((ROWS, COLS), dtype=np.float32)
    risk[20:30, 10:20] = 0.9   # hazard patch 1
    risk[40:50, 40:50] = 0.9   # hazard patch 2

    confidence = np.ones((ROWS, COLS), dtype=np.float32)
    confidence[:5, :]  = 0.0   # unknown border
    confidence[-5:, :] = 0.0
    confidence[:, :5]  = 0.0
    confidence[:, -5:] = 0.0

    mean_z = np.zeros((ROWS, COLS), dtype=np.float32)
    mean_z[20:30, 10:20] = 0.40   # raised bump — hazard 1
    mean_z[40:50, 40:50] = 0.35   # raised bump — hazard 2

    slope_deg   = risk * 25.0    # synthetic: hazard patches are steep
    roughness   = risk * 0.20
    step_height = risk * 0.30

    return risk, confidence, slope_deg, roughness, step_height, mean_z


def run_pipeline(bin_path, grid_path):
    """Run C++ pipeline_runner on a .bin frame, produce grid binary."""
    result = subprocess.run(
        [PIPELINE_RUNNER, bin_path, grid_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"pipeline_runner failed for {bin_path}:\n{result.stderr}")
        return False
    print(result.stdout.strip())
    return True


# ---------------------------------------------------------------------------
# Color mapping
# ---------------------------------------------------------------------------

def build_color_image(risk, confidence):
    """Map risk + confidence → RGB image (ROWS, COLS, 3).
    Thresholds from docstring:
        safe    = risk < 0.3  and confidence > 0.5  → green
        caution = 0.3<=risk<=0.7 and confidence>0.5 → yellow
        hazard  = risk > 0.7  and confidence > 0.5  → red
        unknown = confidence <= 0.5                 → gray
    """
    img = np.full((ROWS, COLS, 3), 0.5, dtype=np.float32)
    conf_ok = confidence > 0.5
    img[(risk < 0.3)  & conf_ok] = [0.0, 0.8, 0.0]   # green
    img[(risk >= 0.3) & (risk <= 0.7) & conf_ok] = [1.0, 0.8, 0.0]  # yellow
    img[(risk > 0.7)  & conf_ok] = [0.9, 0.1, 0.1]   # red
    img[~conf_ok]                = [0.5, 0.5, 0.5]   # gray
    return img


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_2d_bev(img, out_path, title_suffix=""):
    """Figure 4 style — confidence-gated green/yellow/red/gray BEV."""
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img, origin="upper",
              extent=[Y_MIN, Y_MAX, X_MIN, X_MAX], aspect="equal")
    ax.scatter([0], [0], c="blue", s=120, marker="^", zorder=5)
    ax.set_xlabel("Y — lateral (m)  ←left | right→")
    ax.set_ylabel("X — forward (m)  near → far")
    ax.set_title(f"BEV Traversability Grid — P1-M3{title_suffix}")
    legend_handles = [
        mpatches.Patch(color=(0.0, 0.8, 0.0), label="Safe  (risk < 0.3)"),
        mpatches.Patch(color=(1.0, 0.8, 0.0), label="Caution (0.3–0.7)"),
        mpatches.Patch(color=(0.9, 0.1, 0.1), label="Hazard (risk > 0.7)"),
        mpatches.Patch(color=(0.5, 0.5, 0.5), label="Unknown (low confidence)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_6panel(mean_z, risk, slope_deg, roughness, step_height, confidence,
                out_path, title_suffix=""):
    """Figure 3 style — 6-panel feature heatmaps."""
    panels = [
        (mean_z,      "Elevation (mean_z)",   "terrain",   "m"),
        (risk,        "Risk (danger value)",   "RdYlGn_r",  "[0–1]"),
        (slope_deg,   "Slope",                "hot",        "deg"),
        (roughness,   "Roughness",            "hot",        "λ₀/Σλ"),
        (step_height, "Step Height",          "hot",        "m"),
        (confidence,  "Confidence",           "RdYlGn",     "[0–1]"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Traversability Features — P1-M3{title_suffix}", fontsize=13)
    for ax, (data, title, cmap, unit) in zip(axes.flat, panels):
        im = ax.imshow(data, origin="upper",
                       extent=[Y_MIN, Y_MAX, X_MIN, X_MAX],
                       cmap=cmap, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("X (m)")
        plt.colorbar(im, ax=ax, label=unit, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_3d_bev(mean_z, risk, out_path, title_suffix=""):
    """3D surface elevation colored by risk — similar to paper Figure 3."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    x = np.linspace(X_MIN, X_MAX, ROWS)
    y = np.linspace(Y_MIN, Y_MAX, COLS)
    Y_grid, X_grid = np.meshgrid(y, x)
    cmap = plt.cm.RdYlGn_r
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X_grid, Y_grid, mean_z,
                    facecolors=cmap(risk),
                    rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.scatter([0], [0], [0], c="blue", s=120, marker="^", zorder=5)
    ax.view_init(elev=35, azim=-60)   # look from above-left: X forward, Y lateral visible
    ax.set_xlabel("X — forward (m)")
    ax.set_ylabel("Y — lateral (m)")
    ax.set_zlabel("Z — elevation (m)")
    ax.set_title(f"3D Traversability BEV — P1-M3{title_suffix}")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, label="Risk [0=safe, 1=hazard]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# GIF
# ---------------------------------------------------------------------------

def make_gif(bin_dir, out_gif, fps=2):
    """Process every .bin frame in bin_dir → animated GIF of 2D BEV."""
    try:
        import imageio
    except ImportError:
        print("imageio not found — install with:  pip install imageio")
        return

    bin_files = sorted([
        os.path.join(bin_dir, f)
        for f in os.listdir(bin_dir) if f.endswith(".bin")
    ])
    if not bin_files:
        print(f"No .bin files found in {bin_dir}")
        return

    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, bin_path in enumerate(bin_files):
            grid_path = os.path.join(tmpdir, f"grid_{i:03d}.bin")
            png_path  = os.path.join(tmpdir, f"frame_{i:03d}.png")
            print(f"[{i+1}/{len(bin_files)}] {os.path.basename(bin_path)}")
            if not run_pipeline(bin_path, grid_path):
                continue
            risk, confidence, *_ = load_real_grid(grid_path)
            img = build_color_image(risk, confidence)
            plot_2d_bev(img, out_path=png_path,
                        title_suffix=f" — frame {i:03d}")
            frames.append(imageio.imread(png_path))

    if frames:
        imageio.mimsave(out_gif, frames, fps=fps, loop=0)
        print(f"\nGIF saved: {out_gif}  ({len(frames)} frames @ {fps} fps)")
    else:
        print("No frames rendered.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEV Traversability Visualizer")
    parser.add_argument("--bin",       help="Single .bin LiDAR frame (runs pipeline_runner)")
    parser.add_argument("--grid",      help="Pre-exported grid .bin (skips pipeline_runner)")
    parser.add_argument("--gif",       help="Directory of .bin frames → animated GIF")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic demo data")
    parser.add_argument("--out",       default="bev", help="Output prefix / GIF path")
    parser.add_argument("--fps",       type=int, default=2, help="GIF frames per second")
    args = parser.parse_args()

    # --- GIF mode ---
    if args.gif:
        gif_path = args.out if args.out.endswith(".gif") else args.out + ".gif"
        make_gif(args.gif, out_gif=gif_path, fps=args.fps)
        exit(0)

    # --- Single frame modes ---
    if args.synthetic:
        risk, confidence, slope_deg, roughness, step_height, mean_z = make_synthetic_grid()

    elif args.grid:
        risk, confidence, slope_deg, roughness, step_height, mean_z = load_real_grid(args.grid)

    elif args.bin:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            grid_path = tmp.name
        if not run_pipeline(args.bin, grid_path):
            exit(1)
        risk, confidence, slope_deg, roughness, step_height, mean_z = load_real_grid(grid_path)
        os.unlink(grid_path)

    else:
        parser.print_help()
        exit(1)

    img = build_color_image(risk, confidence)
    plot_2d_bev(img,
                out_path=args.out + "_2d.png")
    plot_6panel(mean_z, risk, slope_deg, roughness, step_height, confidence,
                out_path=args.out + "_6panel.png")
    plot_3d_bev(mean_z, risk,
                out_path=args.out + "_3d.png")
