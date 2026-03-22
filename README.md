# Terra Perceive — Construction Site Perception Stack

**LiDAR + Camera Fusion | C++ Algorithms from Scratch | ROS2 Nav2 Pipeline**

Nishant Pushparaju | NYU MS Mechatronics & Robotics | March 2026

---

## What This Is

A complete, Dockerized autonomy perception pipeline that takes raw LiDAR + camera data from off-road terrain, computes traversability, detects and tracks workers, feeds into ROS2/Nav2 for path planning, and enforces safety constraints.

Core perception algorithms are written **from scratch in C++ with Eigen**. ML inference (YOLO, SegFormer) uses production libraries.

## Architecture

```
Camera (RGB) -> [YOLO Detection] -> [SORT Tracker (C++)] -> /tracked_objects
             -> [SegFormer]      -> /semantic_img
                                          |
LiDAR (PC2)  -> [Ground RANSAC (C++)] -> [Traversability (C++)]
                                          |
              [Cam-LiDAR Projection (C++)] -> [Fused Traversability]
                                                      |
                                    Traversability Costmap Layer (C++)
                                       Dynamic Obstacle Layer
                                        Nav2 Local Costmap
                                              |
              Odom / TF  ---------> MPPI Controller (Nav2)
                                              |
                                    Safety Supervisor (C++)
                                              |
                                     /cmd_vel_safe -> Robot
```

## Quick Start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate terra-perceive

# 2. Source ROS2
source /opt/ros/humble/setup.bash

# 3. Build
make build

# 4. Test
make test

# 5. Docker (full stack)
make docker-up
```

## Repository Structure

```
include/          C++ headers (algorithm interfaces)
src/              C++ implementations (from scratch)
ros2_nodes/       ROS2 wrappers
python/           ML inference nodes
transport/        NATS/gRPC production transport layer
  proto/          Protobuf message schemas
dashboard/        Streamlit ops dashboard
config/           All tunable parameters
docker/           Dockerfiles + compose
tests/            C++ (gtest) + Python (pytest)
launch/           ROS2 launch files
scripts/          Utility scripts
logs/             Development log
```

## From-Scratch Implementations (C++)

| Component | Lines | Key Algorithm |
|-----------|-------|---------------|
| Ground RANSAC | - | Plane fitting + SVD refinement |
| Traversability Grid | - | PCA surface normals + scoring |
| Cam-LiDAR Projection | - | SE(3) transforms + pinhole model |
| Kalman Filter | - | Constant-velocity state estimation |
| Hungarian Algorithm | - | Optimal assignment O(n^3) |
| SORT Tracker | - | Multi-object tracking |
| Safety Supervisor | - | Priority-ordered cmd_vel filter |

## Philosophy

Use AI tools for code review, not code generation. Understand every line. The resources in the spec point to primary sources so you learn the algorithms, not just the APIs.

## License

MIT
