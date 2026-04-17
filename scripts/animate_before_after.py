#!/usr/bin/env python3
"""
animate_before_after.py — Before/after optimization toggle GIF.
Alternates between ICP (drifted) and SLAM (optimized) trajectories.

Output: results/before_after_animation.gif
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

n = min(len(carto), len(icp), len(slam))
carto, icp, slam = carto[:n], icp[:n], slam[:n]

icp_a = umeyama_align(icp, carto)
slam_a = umeyama_align(slam, carto)

ate_icp = np.sqrt(np.mean(np.sum((icp_a - carto) ** 2, axis=1)))
ate_slam = np.sqrt(np.mean(np.sum((slam_a - carto) ** 2, axis=1)))

# Create before/after toggle animation
fig, ax = plt.subplots(figsize=(11, 9))
margin = 15
ax.set_xlim(carto[:, 0].min() - margin, carto[:, 0].max() + margin)
ax.set_ylim(carto[:, 1].min() - margin, carto[:, 1].max() + margin)
ax.set_aspect('equal')
ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.grid(True, alpha=0.2)

# Cartographer always visible
ax.plot(carto[:, 0], carto[:, 1], 'k--', linewidth=1.5, alpha=0.5, label='Cartographer (ref)')

line_traj, = ax.plot([], [], linewidth=2.5)
title = ax.set_title('', fontsize=16, fontweight='bold')

# Toggle every 40 frames (~1.6 seconds at 25fps)
toggle_period = 40
total_frames = toggle_period * 8  # 4 full cycles

def animate(frame):
    showing_icp = (frame // toggle_period) % 2 == 0

    if showing_icp:
        line_traj.set_data(icp_a[:, 0], icp_a[:, 1])
        line_traj.set_color('blue')
        title.set_text(f'BEFORE: KISS-ICP (ATE = {ate_icp:.2f}m)')
        title.set_color('blue')
    else:
        line_traj.set_data(slam_a[:, 0], slam_a[:, 1])
        line_traj.set_color('green')
        title.set_text(f'AFTER: Our SLAM (ATE = {ate_slam:.2f}m)')
        title.set_color('green')

    return line_traj, title

print("Rendering before/after animation...")
anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=40, blit=True)
anim.save('results/before_after_animation.gif', writer='pillow', fps=25, dpi=100)
print('Saved results/before_after_animation.gif')
plt.close()
