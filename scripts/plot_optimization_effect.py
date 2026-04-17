#!/usr/bin/env python3
"""
plot_optimization_effect.py — Show what the optimizer changed.

Plots:
  1. Per-frame displacement: how much each pose moved during optimization
  2. Before/after ATE comparison bar chart
  3. Correction vectors on the trajectory (arrows showing the shift)

Output: results/optimization_effect.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

os.makedirs("results", exist_ok=True)

def umeyama_align(src, tgt):
    n = min(len(src), len(tgt))
    src, tgt = src[:n], tgt[:n]
    ms, mt = src.mean(0), tgt.mean(0)
    sc, tc = src - ms, tgt - mt
    cov = tc.T @ sc / n
    U, S, Vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2, 2] = -1
    R = U @ d @ Vt
    s = np.trace(np.diag(S) @ d) / np.var(sc, axis=0).sum()
    t = mt - s * R @ ms
    return (s * (R @ src.T).T + t)

def compute_ate(aligned, ref):
    n = min(len(aligned), len(ref))
    return np.sqrt(np.mean(np.sum((aligned[:n] - ref[:n]) ** 2, axis=1)))

# Load trajectories
print("Loading trajectories...")
carto = np.loadtxt('data/poses_carto.csv', delimiter=',', skiprows=1, usecols=[1, 2, 3])
icp = np.loadtxt('data/poses_icp.csv', delimiter=',', skiprows=1, usecols=[1, 2, 3])
slam = np.loadtxt('data/poses_slam_manifold.csv', delimiter=',', skiprows=1, usecols=[1, 2, 3])
gps = np.loadtxt('data/poses_gps.csv', delimiter=',', skiprows=1, usecols=[1, 2, 3])

n = min(len(carto), len(icp), len(slam), len(gps))
carto = carto[:n]

icp_a = umeyama_align(icp[:n], carto)
slam_a = umeyama_align(slam[:n], carto)
gps_a = umeyama_align(gps[:n], carto)

# Per-frame errors
icp_err = np.linalg.norm(icp_a - carto, axis=1)
slam_err = np.linalg.norm(slam_a - carto, axis=1)
gps_err = np.linalg.norm(gps_a - carto, axis=1)

# Per-frame displacement (how much optimizer moved each pose)
displacement = np.linalg.norm(slam_a - icp_a, axis=1)

# Create figure
fig = plt.figure(figsize=(16, 12), facecolor='white')
gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)

# --- Plot 1: Per-frame ATE (before vs after optimization) ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(icp_err, 'b-', alpha=0.5, linewidth=0.8, label=f'ICP (ATE={np.sqrt(np.mean(icp_err**2)):.2f}m)')
ax1.plot(slam_err, 'g-', linewidth=1.2, label=f'SLAM (ATE={np.sqrt(np.mean(slam_err**2)):.2f}m)')
ax1.plot(gps_err, 'r-', alpha=0.3, linewidth=0.5, label=f'GPS (ATE={np.sqrt(np.mean(gps_err**2)):.2f}m)')
ax1.axvspan(1800, 2200, alpha=0.08, color='red')
ax1.axvspan(2400, n, alpha=0.08, color='blue')
ax1.text(1900, ax1.get_ylim()[1] * 0.9, 'canopy', fontsize=8, color='red', ha='center')
ax1.text(2600, ax1.get_ylim()[1] * 0.9, 'featureless', fontsize=8, color='blue', ha='center')
ax1.set_xlabel('Frame', fontsize=11)
ax1.set_ylabel('Position Error (m)', fontsize=11)
ax1.set_title('Per-Frame Error vs Cartographer', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.2)

# --- Plot 2: Optimizer displacement (how much each pose moved) ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.fill_between(range(n), displacement, alpha=0.3, color='purple')
ax2.plot(displacement, 'purple', linewidth=0.8, label='Pose displacement')
ax2.axvspan(1800, 2200, alpha=0.08, color='red')
ax2.axvspan(2400, n, alpha=0.08, color='blue')
ax2.set_xlabel('Frame', fontsize=11)
ax2.set_ylabel('Displacement (m)', fontsize=11)
ax2.set_title('How Much Optimizer Moved Each Pose', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)

# --- Plot 3: Correction vectors on bird's-eye trajectory ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(carto[:, 0], carto[:, 1], 'k--', linewidth=1, alpha=0.3, label='Cartographer')
ax3.plot(icp_a[:, 0], icp_a[:, 1], 'b-', linewidth=0.8, alpha=0.5, label='ICP (before)')
ax3.plot(slam_a[:, 0], slam_a[:, 1], 'g-', linewidth=1.5, label='SLAM (after)')

# Draw correction arrows every 100 frames
arrow_step = 100
for i in range(0, n, arrow_step):
    dx = slam_a[i, 0] - icp_a[i, 0]
    dy = slam_a[i, 1] - icp_a[i, 1]
    mag = np.sqrt(dx**2 + dy**2)
    if mag > 0.05:  # only show visible corrections
        ax3.annotate('', xy=(slam_a[i, 0], slam_a[i, 1]),
                     xytext=(icp_a[i, 0], icp_a[i, 1]),
                     arrowprops=dict(arrowstyle='->', color='red',
                                    lw=1.5, alpha=0.7))

ax3.set_xlabel('x (m)', fontsize=11)
ax3.set_ylabel('y (m)', fontsize=11)
ax3.set_title('Correction Vectors (ICP → SLAM)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.2)

# --- Plot 4: Summary table ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

table_data = [
    ['Metric', 'ICP\n(before)', 'SLAM\n(after)', 'Change'],
    ['ATE RMSE', f'{np.sqrt(np.mean(icp_err**2)):.3f} m',
     f'{np.sqrt(np.mean(slam_err**2)):.3f} m',
     f'{np.sqrt(np.mean(slam_err**2)) - np.sqrt(np.mean(icp_err**2)):+.3f} m'],
    ['Max Error', f'{icp_err.max():.3f} m', f'{slam_err.max():.3f} m',
     f'{slam_err.max() - icp_err.max():+.3f} m'],
    ['Mean Displacement', '', f'{displacement.mean():.3f} m', ''],
    ['Max Displacement', '', f'{displacement.max():.3f} m', ''],
    ['Optimizer Iters', '', '20', ''],
    ['Final Cost', '', '1.86', ''],
    ['Poses', f'{n}', f'{n}', ''],
    ['ICP Edges', '2846', '2846', ''],
    ['IMU Edges', '2846', '2846', ''],
    ['GPS Edges', '2847', '2847', ''],
]

table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                   colWidths=[0.3, 0.23, 0.23, 0.24])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)

# Style header row
for j in range(4):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Alternate row colors
for i in range(1, len(table_data)):
    color = '#f8f9fa' if i % 2 == 0 else 'white'
    for j in range(4):
        table[i, j].set_facecolor(color)

ax4.set_title('Optimization Summary', fontsize=12, fontweight='bold', pad=20)

fig.suptitle('Pose Graph Optimization: Before vs After',
             fontsize=15, fontweight='bold', y=0.98)

plt.savefig('results/optimization_effect.png', dpi=150, bbox_inches='tight',
            facecolor='white')
print('Saved results/optimization_effect.png')
plt.close()
