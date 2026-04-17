#!/usr/bin/env python3
"""
plot_factor_graph.py — Clean factor graph diagram.
Simplified: just nodes + 4 edge types. No LiDAR diamonds or IMU dashes.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 5), facecolor='white')
ax.set_xlim(-0.5, 15.5)
ax.set_ylim(-3, 3.5)
ax.set_aspect('equal')
ax.axis('off')

# Colors
C_NODE = '#2980b9'
C_ICP = '#27ae60'
C_IMU = '#e67e22'
C_GPS = '#f39c12'
C_LOOP = '#2c3e50'
C_ANCHOR = '#c0392b'

# 6 nodes
nx = [0, 2.5, 5, 7.5, 10, 13]
labels = ['$x_0$', '$x_1$', '$x_2$', '$x_3$', '$x_4$', '$x_N$']

# ICP edges (green lines between consecutive)
for i in range(len(nx) - 1):
    if i == 4:  # gap before x_N
        ax.text((nx[4] + nx[5]) / 2, 0, '$\\cdots$', fontsize=18,
                ha='center', va='center', color='gray')
        continue
    ax.plot([nx[i] + 0.4, nx[i+1] - 0.4], [0, 0], '-',
            color=C_ICP, linewidth=4, zorder=2, solid_capstyle='round')

# IMU arcs (orange, above)
for i in range(len(nx) - 1):
    if i == 4:
        continue
    x1, x2 = nx[i], nx[i+1]
    t = np.linspace(0, np.pi, 40)
    arc_x = x1 + 0.4 + (x2 - x1 - 0.8) * t / np.pi
    arc_y = 1.3 * np.sin(t)
    ax.plot(arc_x, arc_y, '-', color=C_IMU, linewidth=2.5, zorder=2)

# GPS (yellow, vertical below select nodes)
for i in [1, 3, 5]:
    x = nx[i]
    ax.plot([x, x], [-0.45, -1.6], '-', color=C_GPS, linewidth=2.5, zorder=2)
    circle = plt.Circle((x, -2.0), 0.3, fill=True, facecolor='#fef9e7',
                          edgecolor=C_GPS, linewidth=2, zorder=4)
    ax.add_patch(circle)
    ax.text(x, -2.0, '📡', fontsize=10, ha='center', va='center')

# Loop closure (dark arc connecting x1 to x4)
x1, x2 = nx[1], nx[4]
t = np.linspace(0, np.pi, 50)
arc_x = x1 + (x2 - x1) * t / np.pi
arc_y = 2.2 + 0.8 * np.sin(t)
ax.plot(arc_x, arc_y, '-', color=C_LOOP, linewidth=2.5, zorder=2)
ax.plot((x1 + x2) / 2, 3.0, 'o', color=C_LOOP, markersize=7, zorder=3)

# Nodes (on top)
for i, (x, label) in enumerate(zip(nx, labels)):
    color = C_ANCHOR if i == 0 else C_NODE
    circle = plt.Circle((x, 0), 0.38, fill=True, facecolor=color,
                          edgecolor='white', linewidth=2.5, zorder=5)
    ax.add_patch(circle)
    ax.text(x, 0, label, fontsize=13, ha='center', va='center',
            color='white', fontweight='bold', zorder=6)

# Legend (clean, horizontal at bottom)
ly = -2.8
items = [
    (C_ICP, '━━', 'ICP Odometry'),
    (C_IMU, '⌢', 'IMU Preintegration'),
    (C_GPS, '│', 'GPS (unary)'),
    (C_LOOP, '⌢', 'Loop Closure'),
    (C_ANCHOR, '●', 'Fixed Anchor'),
    (C_NODE, '●', 'SE(3) Pose'),
]
for idx, (color, sym, label) in enumerate(items):
    lx = 0.5 + idx * 2.5
    ax.plot(lx, ly, 'o', color=color, markersize=8)
    ax.text(lx + 0.25, ly, label, fontsize=8, va='center', color='#333')

plt.tight_layout()
plt.savefig('results/factor_graph_architecture.png', dpi=200,
            bbox_inches='tight', facecolor='white')
print('Saved results/factor_graph_architecture.png')
plt.close()
