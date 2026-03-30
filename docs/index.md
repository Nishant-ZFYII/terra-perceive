---
layout: home
title: Project Overview
---

*A production-grade C++ perception pipeline for construction site autonomy.*

Autonomous perception often breaks when it leaves the asphalt. While highway datasets like KITTI are the industry standard, they don't capture the chaotic, deforming, and unstructured terrain of a construction site. This project, **Terra Perceive**, is an engineering deep-dive into building a perception stack from the ground up to handle the "unstructured frontier."

Built using **C++17**, **Eigen3**, and **ROS2** on the **[RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D)** dataset, this stack prioritizes mathematical rigor and real-time performance. Every core component—from sector-based RANSAC to kinematic safety filters—is implemented from scratch.

---

## Phase 1: Core Perception & Safety (Current Focus)

Phase 1 focuses on the primary perception-to-safety loop: taking raw LiDAR data and producing actionable safety interventions based on terrain geometry.
With P1-M5 (Kinematic Safety Supervisor) complete, we are now wrapping P1-M6 (Docker + Integration) so the full pipeline can run from a single smoke-test command.

| Milestone | Implementation | Status |
|-----------|----------------|--------|
| [M1: Data Ingestion](m1-data) | $O(N)$ binary loader for RELLIS-3D and Open3D visualization | Completed |
| [M2: Sector RANSAC](m2-ransac) | Ground segmentation for sloped and graded terrain | Completed |
| [M3: Traversability Grid](m3-traversability) | Risk/confidence maps using PCA surface normals | Completed |
| [M4: Camera-LiDAR Fusion](m4-fusion) | Homogeneous transforms and semantic segmentation (SegFormer) | Completed |
| [M5: Kinematic Safety](m5-safety) | Stopping distance, TTC, terrain-aware friction, priority interventions | Completed |
| [M6: Integration](m6-docker) | Docker image, smoke test, end-to-end pipeline | Completed |

## Phase 2: Tracking & Production Transport

The second phase transitions the pipeline into a distributed system with persistent object awareness.

- **NATS JetStream**: Production-grade messaging for safety audit replay and low-latency transport.
- **Protobuf Schemas**: Standardized perception and telemetry messages.
- **Multi-Object Tracking**: Kalman filters and the SORT algorithm for persistent worker/vehicle tracking.
- **Probabilistic Scoring**: Uncertainty propagation through the traversability grid.

## Phase 3: Deployment & Operations

Final hardening and operational tools for fleet-scale management.

- **Streamlit Ops Dashboard**: Real-time telemetry visualization.
- **Nav2 Costmap Plugin**: Exporting custom traversability layers to the ROS2 navigation stack.
- **TensorRT Optimization**: FP16 export for deployment on NVIDIA Jetson Orin.

---

**Technical Stack**
- **Core Logic**: C++17, Eigen3 (No high-level CV libraries for math)
- **Infrastructure**: ROS2 Humble, Colcon, CMake
- **Data Source**: RELLIS-3D (Ouster OS1-64)
- **Deployment**: Docker, Ubuntu 22.04

---

*NYU MS Mechatronics & Robotics — Nishant Pushparaju*
