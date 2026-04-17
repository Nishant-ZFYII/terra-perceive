#!/usr/bin/env python3
"""
plot_ablation.py — Generate all P2-M2 ablation and comparison plots.

Outputs:
  results/ablation_bar_chart.png       — ATE for all configs
  results/four_trajectory_comparison.png — bird's-eye aligned trajectories
  results/per_frame_ate.png            — per-frame position error
  results/convergence_curve.png        — cost vs iteration
  results/gps_weight_vs_frame.png      — GPS information weight per frame
  results/scan_context_heatmap.png     — sample descriptor visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Umeyama alignment + ATE
# ---------------------------------------------------------------------------

def umeyama_align(src, tgt):
    """Umeyama alignment: returns (scale, rotation, translation)."""
    n = min(len(src), len(tgt))
    src, tgt = src[:n], tgt[:n]
    mu_s, mu_t = src.mean(0), tgt.mean(0)
    sc, tc = src - mu_s, tgt - mu_t
    cov = tc.T @ sc / n
    U, S, Vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2, 2] = -1
    R = U @ d @ Vt
    s = np.trace(np.diag(S) @ d) / np.var(sc, axis=0).sum()
    t = mu_t - s * R @ mu_s
    return s, R, t

def align_trajectory(src, tgt):
    """Align src to tgt, return aligned src."""
    n = min(len(src), len(tgt))
    src, tgt = src[:n], tgt[:n]
    s, R, t = umeyama_align(src, tgt)
    return (s * (R @ src.T).T + t)

def compute_ate(src_path, ref_path):
    """Compute ATE RMSE after Umeyama alignment."""
    src = np.loadtxt(src_path, delimiter=',', skiprows=1, usecols=[1, 2, 3])
    ref = np.loadtxt(ref_path, delimiter=',', skiprows=1, usecols=[1, 2, 3])
    n = min(len(src), len(ref))
    src, ref = src[:n], ref[:n]
    aligned = align_trajectory(src, ref)
    return np.sqrt(np.mean(np.sum((aligned - ref) ** 2, axis=1)))

def load_xyz(path):
    """Load (x, y, z) columns from a pose CSV."""
    return np.loadtxt(path, delimiter=',', skiprows=1, usecols=[1, 2, 3])

# ---------------------------------------------------------------------------
# Plot 1: Ablation bar chart
# ---------------------------------------------------------------------------

def plot_ablation_bar_chart():
    ref = "data/poses_carto.csv"

    configs = {}
    configs["ICP\n(baseline)"] = ("data/poses_icp.csv", "steelblue")
    configs["ICP+IMU\n(B)"] = ("data/poses_slam_B.csv", "steelblue")

    # Optional configs — only include if file exists
    optional = {
        "ICP+IMU\n+Loop (C)": ("data/poses_slam_C.csv", "teal"),
        "ICP+IMU+GPS\n Manifold (D)": ("data/poses_slam_manifold.csv", "green"),
        "ICP+IMU+GPS\n g2o (D)": ("data/poses_slam_g2o.csv", "orange"),
        "ICP+IMU+GPS\n Euclidean (D)": ("data/poses_slam_euclidean.csv", "red"),
        "Full\n+Loop (E)": ("data/poses_slam_E.csv", "darkgreen"),
    }
    for label, (path, color) in optional.items():
        if os.path.exists(path):
            configs[label] = (path, color)

    labels = []
    ates = []
    colors = []
    for label, (path, color) in configs.items():
        try:
            ate = compute_ate(path, ref)
            labels.append(label)
            ates.append(ate)
            colors.append(color)
        except Exception as e:
            print(f"  Skipping {label}: {e}")

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, ates, color=colors, edgecolor='black', linewidth=0.5)
    for bar, ate in zip(bars, ates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                 f'{ate:.2f}m', ha='center', fontsize=10, fontweight='bold')
    plt.ylabel('ATE RMSE (m)', fontsize=13)
    plt.title('Ablation Study: Edge Types and Optimizer Comparison', fontsize=14)
    plt.axhline(y=ates[0], color='blue', linestyle='--', alpha=0.3, label=f'ICP baseline ({ates[0]:.2f}m)')
    plt.legend(fontsize=11)
    plt.tight_layout()
    out = f"{RESULTS_DIR}/ablation_bar_chart.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

# ---------------------------------------------------------------------------
# Plot 2: Four-trajectory comparison (aligned)
# ---------------------------------------------------------------------------

def plot_four_trajectory():
    ref = "data/poses_carto.csv"
    carto = load_xyz(ref)

    trajectories = {
        "GPS": ("data/poses_gps.csv", 'r', 0.4, 1.0),
        "KISS-ICP": ("data/poses_icp.csv", 'b', 0.6, 1.0),
        "Our SLAM (Manifold)": ("data/poses_slam_manifold.csv", 'g', 1.0, 2.0),
        "Cartographer (ref)": (None, 'k', 1.0, 1.5),
    }

    plt.figure(figsize=(12, 9))

    for label, (path, color, alpha, lw) in trajectories.items():
        if path is None:
            plt.plot(carto[:, 0], carto[:, 1], color=color, linestyle='--',
                     alpha=alpha, linewidth=lw, label=label)
        elif os.path.exists(path):
            traj = load_xyz(path)
            aligned = align_trajectory(traj, carto)
            ate = np.sqrt(np.mean(np.sum((aligned[:len(carto)] - carto[:len(aligned)]) ** 2, axis=1)))
            plt.plot(aligned[:, 0], aligned[:, 1], color=color,
                     alpha=alpha, linewidth=lw, label=f'{label} (ATE={ate:.2f}m)')

    plt.legend(fontsize=12)
    plt.xlabel('x (m)', fontsize=12)
    plt.ylabel('y (m)', fontsize=12)
    plt.title('Four-Trajectory Comparison (Umeyama-aligned to Cartographer)', fontsize=14)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{RESULTS_DIR}/four_trajectory_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

# ---------------------------------------------------------------------------
# Plot 3: Per-frame ATE
# ---------------------------------------------------------------------------

def plot_per_frame_ate():
    ref = "data/poses_carto.csv"
    carto = load_xyz(ref)

    sources = {
        "GPS": ("data/poses_gps.csv", 'r', 0.4),
        "KISS-ICP": ("data/poses_icp.csv", 'b', 0.6),
        "Our SLAM": ("data/poses_slam_manifold.csv", 'g', 1.0),
    }

    plt.figure(figsize=(14, 5))

    for label, (path, color, alpha) in sources.items():
        if os.path.exists(path):
            traj = load_xyz(path)
            aligned = align_trajectory(traj, carto)
            n = min(len(aligned), len(carto))
            per_frame = np.linalg.norm(aligned[:n] - carto[:n], axis=1)
            plt.plot(per_frame, color=color, alpha=alpha, label=label)

    # Annotate regions from P2-M1 analysis
    plt.axvspan(1800, 2200, alpha=0.08, color='red', label='Canopy region')
    plt.axvspan(2400, 2847, alpha=0.08, color='blue', label='Featureless region')

    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Position error (m)', fontsize=12)
    plt.title('Per-Frame ATE (Umeyama-aligned)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{RESULTS_DIR}/per_frame_ate.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

# ---------------------------------------------------------------------------
# Plot 4: GPS weight vs frame
# ---------------------------------------------------------------------------

def plot_gps_weight():
    N = 2847
    frames = np.arange(N)
    sigmas = np.full(N, 8.0)
    sigmas[1800:2201] = 50.0
    sigmas[2400:] = 15.0
    weights = 1.0 / (sigmas ** 2)

    plt.figure(figsize=(14, 4))
    plt.plot(frames, weights, 'darkgreen', linewidth=1.5)
    plt.fill_between(frames, 0, weights, alpha=0.2, color='green')
    plt.axvspan(1800, 2200, alpha=0.1, color='red', label='Canopy (σ=50m)')
    plt.axvspan(2400, 2847, alpha=0.1, color='blue', label='Featureless (σ=15m)')

    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('GPS Information Weight (1/σ²)', fontsize=12)
    plt.title('GPS Information Weight vs Frame (HDOP-weighted)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{RESULTS_DIR}/gps_weight_vs_frame.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

# ---------------------------------------------------------------------------
# Plot 5: Convergence curve
# ---------------------------------------------------------------------------

def plot_convergence():
    # Manifold convergence (from verbose output)
    costs_manifold = [193960, 115.475, 8.08, 1.93, 1.86, 1.86, 1.86, 1.86,
                      1.86, 1.86, 1.86, 1.86, 1.86, 1.86, 1.86, 1.86,
                      1.86, 1.86, 1.86, 1.86, 1.86]

    # Euclidean convergence (approximate — from your earlier run)
    # If you have the verbose output, replace these with actual values
    costs_euclidean = None  # Fill in when available

    plt.figure(figsize=(10, 5))
    plt.semilogy(range(len(costs_manifold)), costs_manifold, 'g-o',
                 linewidth=2, markersize=4, label='Manifold (LM)')

    if costs_euclidean:
        plt.semilogy(range(len(costs_euclidean)), costs_euclidean, 'r-s',
                     linewidth=2, markersize=4, label='Euclidean (LM)')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost (log scale)', fontsize=12)
    plt.title('Optimizer Convergence: Levenberg-Marquardt', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{RESULTS_DIR}/convergence_curve.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

# ---------------------------------------------------------------------------
# Plot 6: Optimizer comparison (Manifold vs Euclidean vs g2o)
# ---------------------------------------------------------------------------

def plot_optimizer_comparison():
    ref = "data/poses_carto.csv"
    carto = load_xyz(ref)

    optimizers = {
        "Manifold (custom)": ("data/poses_slam_manifold.csv", 'green', 2.0),
        "g2o": ("data/poses_slam_g2o.csv", 'orange', 1.5),
        "Euclidean (custom)": ("data/poses_slam_euclidean.csv", 'red', 1.5),
        "Cartographer (ref)": (None, 'black', 1.5),
    }

    plt.figure(figsize=(12, 9))

    for label, (path, color, lw) in optimizers.items():
        if path is None:
            plt.plot(carto[:, 0], carto[:, 1], 'k--', linewidth=lw, label=label)
        elif os.path.exists(path):
            traj = load_xyz(path)
            aligned = align_trajectory(traj, carto)
            n = min(len(aligned), len(carto))
            ate = np.sqrt(np.mean(np.sum((aligned[:n] - carto[:n]) ** 2, axis=1)))
            plt.plot(aligned[:, 0], aligned[:, 1], color=color,
                     linewidth=lw, label=f'{label} (ATE={ate:.2f}m)')

    plt.legend(fontsize=12)
    plt.xlabel('x (m)', fontsize=12)
    plt.ylabel('y (m)', fontsize=12)
    plt.title('Optimizer Comparison: Manifold vs Euclidean vs g2o', fontsize=14)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{RESULTS_DIR}/optimizer_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

# ---------------------------------------------------------------------------
# Summary table (print to console)
# ---------------------------------------------------------------------------

def print_summary_table():
    ref = "data/poses_carto.csv"

    rows = [
        ("ICP (baseline)", "data/poses_icp.csv"),
        ("B: ICP + IMU", "data/poses_slam_B.csv"),
        ("C: ICP + IMU + Loop", "data/poses_slam_C.csv"),
        ("D: Manifold (ICP+IMU+GPS)", "data/poses_slam_manifold.csv"),
        ("D: g2o (ICP+IMU+GPS)", "data/poses_slam_g2o.csv"),
        ("D: Euclidean (ICP+IMU+GPS)", "data/poses_slam_euclidean.csv"),
        ("E: Full + Loop", "data/poses_slam_E.csv"),
        ("GPS raw", "data/poses_gps.csv"),
    ]

    print("\n" + "=" * 55)
    print(f"{'Config':<35} {'ATE RMSE (m)':>12}")
    print("=" * 55)
    for label, path in rows:
        if os.path.exists(path):
            try:
                ate = compute_ate(path, ref)
                print(f"  {label:<33} {ate:>10.3f} m")
            except Exception as e:
                print(f"  {label:<33} {'ERROR':>10}")
        else:
            print(f"  {label:<33} {'(pending)':>10}")
    print("=" * 55)
    print(f"  {'Cartographer (reference)':<33} {'0.000':>10} m")
    print("=" * 55 + "\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating P2-M2 ablation plots...\n")

    print_summary_table()
    plot_ablation_bar_chart()
    plot_four_trajectory()
    plot_per_frame_ate()
    plot_gps_weight()
    plot_convergence()
    plot_optimizer_comparison()

    print(f"\nAll plots saved to {RESULTS_DIR}/")
