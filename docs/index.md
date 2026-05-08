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

## Phase 2: Odometry, SLAM, and Tracking (In Progress)

Phase 1 handles single frames. Phase 2 adds the temporal dimension — where is the robot over time, what moved, and how does the map accumulate across a full traversal.

### Odometry & SLAM

| Milestone | Implementation | Status |
|-----------|----------------|--------|
| [M7: Triple Odometry](m7-odometry) | GPS/IMU extraction, KISS-ICP, Cartographer benchmark, ATE/RPE comparison | Completed |
| [M8: LiDAR-Inertial SLAM](m8-slam) | From-scratch pose graph optimizer, IMU preintegration, Scan Context, manifold vs Euclidean ablation | Completed |

### Mapping & Tracking

| Milestone | Implementation | Status |
|-----------|----------------|--------|
| [M9: Accumulated BEV Map](m9-bev-map) | World map from multi-source odometry, NATS transport | Completed |
| [M10: SORT Tracker](m10-sort-tracker) | Kalman + Hungarian + DBSCAN from scratch; IMM + Deep SORT cascade + Mahalanobis gate; Phase-4 K-frame DBSCAN sweep mapping the structural ceiling | Completed |
| [M11: Tracker-Safety Loop](m5-pipeline) | YOLO + cam-LiDAR projection + SORT + safety supervisor + NATS + JetStream audit trail | Completed |

### Perception & Safety Refinement

The Phase-1 traversability and safety supervisor both shipped with formulas chosen for simplicity, not physical justification. M12 and M13 replace each with a derivation-grounded version, both behind config switches so the legacy paths stay bit-for-bit unchanged.

| Milestone | Implementation | Status |
|-----------|----------------|--------|
| [M12: Probabilistic Traversability](m12-probabilistic-traversability) | Range-dependent LiDAR noise σ(r) propagated through per-cell PCA → calibrated confidence with no artifact cliff. Full-sequence ablation on RELLIS-3D (2849 frames): integrated AUC of \|c_prob − c_heur\| over r ∈ [5, 30] m = 5.51, matching prediction within 2%. Ships with raw-LiDAR + per-cell + Open3D chase-cam visualizations. | Completed |
| [M13: CBF Safety](m13-cbf-safety) | 1D scalar Control Barrier Function clamp on commanded acceleration; 6-scenario ablation against the kinematic TTC step rule. Bang-bang elimination on occluded / multi-worker (max \|dv/dt\| 9-12× lower); tight d_safe_min stops on head-on (0.51 m vs 1.76 m kinematic); zero false positives on lateral passes. | Completed |

### Cross-domain Evaluation & Generalization

| Milestone | Implementation | Status |
|-----------|----------------|--------|
| M14: nuScenes | Unified calibration adapter, second domain validation | Planned |
| M15–M17 | MOTA eval, 3D viz, ROS2 live pipeline, final ship | Planned |

## Phase 3: Deployment (Stretch)

Jetson deployment, Gazebo simulation, Nav2 costmap plugin, TensorRT optimization.

---

**Technical Stack**
- **Core Logic**: C++17, Eigen3 (No high-level CV libraries for math)
- **Infrastructure**: ROS2 Humble, Colcon, CMake
- **Data Source**: RELLIS-3D (Ouster OS1-64)
- **Deployment**: Docker, Ubuntu 22.04

---

*NYU MS Mechatronics & Robotics — Nishant Pushparaju*
