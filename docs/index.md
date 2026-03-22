---
layout: home
title: Terra Perceive
---

# Terra Perceive

*A production-grade C++ perception pipeline for construction site autonomy.*

Built from scratch using C++17, Eigen3, and ROS2 on the [RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D) off-road LiDAR dataset.
Targeting real deployment constraints: < 5ms per frame, runs on Jetson, ships via Docker.

---

## Why this project exists

Most autonomous perception demos work on highway driving datasets (KITTI, nuScenes). Construction sites are different — uneven terrain, steep grades, mud, standing water, and no lane markings. The algorithms that work on roads fail here.

This series documents building a perception stack that handles the hard cases: sloped ground, occluded obstacles, and safety decisions grounded in vehicle kinematics rather than fixed thresholds.

---

## Milestones

| # | Topic | Status |
|---|-------|--------|
| [M1: Taming Raw LiDAR Data](m1-data) | Loading RELLIS-3D `.bin` files in C++, Open3D visualization | ✅ Done |
| [M2: Sector RANSAC](m2-ransac) | Ground segmentation on sloped terrain | 🔧 In progress |
| M3: BEV Traversability Grid | Risk + confidence layers, vehicle-aware scoring | ⏳ Pending |
| M4: Camera-LiDAR Fusion | Projecting LiDAR onto image, SegFormer integration | ⏳ Pending |
| M5: Kinematic Safety Supervisor | Stopping distance, TTC, forward-arc lookahead | ⏳ Pending |

---

## Stack

- **Language**: C++17, Eigen3 (no OpenCV for core algorithms)
- **Build**: colcon / CMake, ROS2 Humble
- **Data**: RELLIS-3D (Ouster OS1-64, off-road terrain)
- **Visualization**: Open3D, Python
- **Deployment**: Docker, Ubuntu 22.04

---

*NYU MS Mechatronics & Robotics — Nishant Pushparaju*
