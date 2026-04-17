#!/usr/bin/env python3
"""
animate_trajectory.py — Generate trajectory animation GIF.
Shows ICP, SLAM, and Cartographer trajectories growing frame-by-frame.

Output: results/trajectory_animation.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

print("Loading trajectories...")
carto = np.loadtxt('data/poses_carto.csv', delimiter=',', skiprows=1, usecols=[1, 2, 3])
icp = np.loadtxt('data/poses_icp.csv', delimiter=',', skiprows=1, usecols=[1, 2, 3])
slam = np.loadtxt('data/poses_slam_manifold.csv', delimiter=',', skiprows=1, usecols=[1, 2, 3])
gps = np.loadtxt('data/poses_gps.csv', delimiter=',', skiprows=1, usecols=[1, 2, 3])

n = min(len(carto), len(icp), len(slam), len(gps))
carto, icp, slam, gps = carto[:n], icp[:n], slam[:n], gps[:n]

print("Aligning trajectories...")
icp_a = umeyama_align(icp, carto)
slam_a = umeyama_align(slam, carto)
gps_a = umeyama_align(gps, carto)

# Compute final ATEs for legend
ate_icp = np.sqrt(np.mean(np.sum((icp_a - carto) ** 2, axis=1)))
ate_slam = np.sqrt(np.mean(np.sum((slam_a - carto) ** 2, axis=1)))
ate_gps = np.sqrt(np.mean(np.sum((gps_a - carto) ** 2, axis=1)))

print(f"ATEs — GPS: {ate_gps:.2f}m, ICP: {ate_icp:.2f}m, SLAM: {ate_slam:.2f}m")

# Animation
fig, ax = plt.subplots(figsize=(11, 9))
margin = 15
ax.set_xlim(carto[:, 0].min() - margin, carto[:, 0].max() + margin)
ax.set_ylim(carto[:, 1].min() - margin, carto[:, 1].max() + margin)
ax.set_aspect('equal')
ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.grid(True, alpha=0.2)

line_carto, = ax.plot([], [], 'k--', linewidth=1.5, label='Cartographer (ref)')
line_gps, = ax.plot([], [], 'r-', alpha=0.4, linewidth=0.8, label=f'GPS (ATE={ate_gps:.2f}m)')
line_icp, = ax.plot([], [], 'b-', alpha=0.6, linewidth=1.0, label=f'KISS-ICP (ATE={ate_icp:.2f}m)')
line_slam, = ax.plot([], [], 'g-', linewidth=2.0, label=f'Our SLAM (ATE={ate_slam:.2f}m)')

# Current position markers
dot_carto, = ax.plot([], [], 'ko', markersize=5)
dot_slam, = ax.plot([], [], 'go', markersize=7, zorder=5)

ax.legend(fontsize=11, loc='upper left')
title = ax.set_title('Trajectory Comparison — Frame 0', fontsize=14)

step = 5  # frames per animation step
total_frames = n // step

print(f"Rendering {total_frames} animation frames...")

def animate(frame):
    i = min(frame * step, n - 1)
    line_carto.set_data(carto[:i, 0], carto[:i, 1])
    line_gps.set_data(gps_a[:i, 0], gps_a[:i, 1])
    line_icp.set_data(icp_a[:i, 0], icp_a[:i, 1])
    line_slam.set_data(slam_a[:i, 0], slam_a[:i, 1])

    dot_carto.set_data([carto[i, 0]], [carto[i, 1]])
    dot_slam.set_data([slam_a[i, 0]], [slam_a[i, 1]])

    title.set_text(f'Trajectory Comparison — Frame {i}/{n}')

    if frame % 50 == 0:
        print(f"  Frame {frame}/{total_frames}")

    return line_carto, line_gps, line_icp, line_slam, dot_carto, dot_slam, title

anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=40, blit=True)
anim.save('results/trajectory_animation.gif', writer='pillow', fps=25, dpi=100)
print('Saved results/trajectory_animation.gif')
plt.close()
