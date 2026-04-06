---
layout: home
title: Project Overview
---

*A production-grade C++ perception pipeline for construction site autonomy.*

Autonomous perception often breaks when it leaves the asphalt. While highway datasets like KITTI are the industry standard, they don't capture the chaotic, deforming, and unstructured terrain of a construction site. This project, **Terra Perceive**, is an engineering deep-dive into building a perception stack from the ground up to handle the "unstructured frontier."

Built using **C++17**, **Eigen3**, and **ROS2** on the **[RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D)** dataset, this stack prioritizes mathematical rigor and real-time performance. Every core component—from sector-based RANSAC to kinematic safety filters—is implemented from scratch.

---

## Phase 1: Core Perception & Safety (Complete)

Phase 1 focused on the perception-to-safety loop: taking raw LiDAR data and producing actionable safety interventions grounded in kinematics.
With P1-M6 (Docker + Integration) and P1-M7 (README, demo, PI ship) finished, the repo now ships a smoke-testable container that outputs BEV visuals, safety logs, and timing reports in under two minutes.

| Milestone | Implementation | Status |
|-----------|----------------|--------|
| [M1: Data Ingestion](m1-data) | $O(N)$ binary loader for RELLIS-3D and Open3D visualization | Completed |
| [M2: Sector RANSAC](m2-ransac) | Ground segmentation for sloped and graded terrain | Completed |
| [M3: Traversability Grid](m3-traversability) | Risk/confidence maps using PCA surface normals | Completed |
| [M4: Camera-LiDAR Fusion](m4-fusion) | Homogeneous transforms and semantic segmentation (SegFormer) | Completed |
| [M5: Kinematic Safety](m5-safety) | Stopping distance, TTC, terrain-aware friction, priority interventions | Completed |
| [M6: Integration](m6-docker) | Docker image, smoke test, end-to-end pipeline | Completed |
| [M7: README + Demo](m7-ship) | Technical README, in-repo demo, fresh-clone verification | Completed |

## Phase 2: Perception Depth + SLAM + Production Infrastructure (In Progress)

Phase 2 proves the system can go beyond single-frame perception: multi-source odometry, temporal world models, multi-object tracking, multi-dataset generalization, and production transport — all integrated into a single codebase.

### Odometry & SLAM (Weeks 1–4)

| Milestone | Implementation | Status |
|-----------|----------------|--------|
| [M7: Triple Odometry](m7-odometry) | GPS/IMU extraction, KISS-ICP, Cartographer benchmark, ATE/RPE comparison | Completed |
| M8: LiDAR-Inertial SLAM | Scan Context loop closure, IMU pre-integration, GPS factors, g2o pose graph | Planned |

### Mapping & Tracking (Weeks 5–7)

| Milestone | Implementation | Status |
|-----------|----------------|--------|
| M9: Accumulated BEV Map | World map from multi-source odometry, NATS bootstrap | Planned |
| M10: SORT Tracker | Kalman filter + Hungarian assignment (C++/Eigen, from scratch) | Planned |
| M11: Tracker-Safety Loop | YOLO + tracked object TTC, NATS wiring | Planned |

### Evaluation & Generalization (Weeks 8–13)

| Milestone | Implementation | Status |
|-----------|----------------|--------|
| M12: Ablation Study | Probabilistic traversability vs heuristic, CBF vs kinematic TTC | Planned |
| M13: nuScenes | Unified calibration adapter, second domain validation | Planned |
| M14: MOTA Evaluation | Tracking metrics on nuScenes | Planned |
| M15: 3D Viz + Demo | Four-pane visualization, gRPC control plane | Planned |
| M16: ROS2 Live Pipeline | Real-time rosbag replay | Planned |
| M17: Ship | Docker compose, evaluation harness, 95+ tests | Planned |

## Phase 3: Deployment & Operations (Stretch)

Final hardening and operational tools.

- **Munkres Hungarian**, SegFormer fine-tuning, multi-obstacle QP
- **Sensor fusion EKF**, Gazebo simulation, Jetson deployment
- **Streamlit Ops Dashboard**, Nav2 costmap plugin, TensorRT optimization

---

**Technical Stack**
- **Core Logic**: C++17, Eigen3 (No high-level CV libraries for math)
- **Infrastructure**: ROS2 Humble, Colcon, CMake
- **Data Source**: RELLIS-3D (Ouster OS1-64)
- **Deployment**: Docker, Ubuntu 22.04

---

*NYU MS Mechatronics & Robotics — Nishant Pushparaju*
