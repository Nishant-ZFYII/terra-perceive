# Phase 2 Milestone 2: LiDAR-Inertial SLAM — Theoretical Foundation

## The Challenge: Why Local Odometry is Insufficient

Autonomous navigation in unstructured, off-road environments (like construction sites or forest trails) presents a fundamental challenge to localization. While Milestone 1 established **Triple Odometry** (GPS/IMU, KISS-ICP, and Cartographer), these sources are primarily "local" or "raw" in nature:

1.  **Odometry Drifts**: Exteroceptive methods like KISS-ICP are incrementally consistent but globally divergent. Without a global reference or loop closure, small errors in scan matching accumulate into significant "drift" over kilometers.
2.  **GPS Multipath and Dropouts**: Under heavy tree canopy or near large construction equipment, GNSS signals suffer from multipath interference or complete loss. Relying solely on GPS leads to "jumps" in the trajectory that can violate the vehicle's kinematic constraints.
3.  **Featureless Terrain**: On flat, featureless grass fields, LiDAR scan matching (ICP) often fails to find unique geometric constraints, leading to "slippage" along the direction of motion.

For **long-duration autonomy**, we need a **persistent world model** that can recognize previously visited locations and use them to correct accumulated drift. This is the role of SLAM (Simultaneous Localization and Mapping).

## Theoretical Pillars of Phase 2 SLAM

The Terra Perceive Phase 2 SLAM system is built on a **Factor Graph** architecture, bridging the gap between high-frequency proprioceptive sensing and globally-aware exteroceptive perception.

### 1. Factor Graph Optimization (The Global Brain)
Instead of a recursive filter (like an EKF) which discards past measurements, we employ **Non-linear Least Squares Optimization on Manifolds** (via g2o).
-   **Nodes**: Represent the vehicle's $SE(3)$ pose at discrete LiDAR timestamps.
-   **Edges (Factors)**: Represent probabilistic constraints.
    -   **Relative Edges**: Derived from KISS-ICP (scan-to-scan) and IMU Preintegration.
    -   **Unary Edges**: Absolute position anchors from GPS, weighted by the real-time dilution of precision (DOP).
    -   **Loop Closure Edges**: Long-range constraints that link the current pose to a non-sequential past pose.

### 2. Scan Context: The "Geometric Fingerprint"
To detect when the vehicle has returned to a known location, we need a descriptor that is both **computationally efficient** and **rotation-invariant**.
-   **Polar Encoding**: We partition the 3D LiDAR sweep into $N_r$ rings and $N_s$ sectors.
-   **Context Generation**: Each bin stores the **maximum height** of the points within it, capturing the unique "skyline" of the local environment.
-   **Rotational Invariance**: By performing a circular column-shift search on the descriptor matrix, we can match scans regardless of the vehicle's heading, simultaneously estimating the relative yaw offset.

### 3. IMU Preintegration on Manifold
Fusing 100Hz IMU data into a 10Hz pose graph would typically require re-integrating all high-frequency samples every time the optimizer adjusts a past state. We implement the **Forster et al. formulation**, which integrates relative motion in a local frame. This allows us to summarize hundreds of IMU measurements into a single "Preintegrated Factor" that remains valid even as the global trajectory is optimized.

## Implementation Architecture

The system is implemented as a **g2o-based Optimizer** in C++.

| Component | g2o Representation | Implementation Detail |
|-----------|--------------------|-----------------------|
| **Pose Node** | `VertexSE3` | $SE(3)$ transformation in world frame. |
| **Odometry Edge** | `EdgeSE3` | Relative constraint from KISS-ICP scan matching. |
| **IMU Edge** | `EdgeNavState` | Forster pre-integration summary (Pose, Velocity, Bias). |
| **GPS Factor** | Unary `EdgePosition` | Absolute XYZ position from VectorNav, weighted by HDOP. |
| **Loop Closure** | `EdgeSE3` | Relative constraint between non-sequential nodes via Scan Context. |

## The Ablation Study: Measuring Marginal Gains

A core goal of P2-M2 is to quantify the value of each sensor and algorithm through an ablation study. We compare five configurations against the **Google Cartographer** baseline:

1.  **ICP Only**: Pure scan-to-scan matching. Expected to drift significantly.
2.  **ICP + IMU**: Adding high-frequency motion priors to smooth motion and handle featureless areas.
3.  **ICP + IMU + GPS**: Anchor the trajectory to global coordinates.
4.  **ICP + IMU + Loop**: Test global consistency without absolute anchors.
5.  **Full SLAM**: All factors enabled.

By measuring the **Absolute Trajectory Error (ATE)** for each, we can empirically prove which components are critical for off-road navigation.

## Conclusion: Towards Global Consistency

By unifying these components, Terra Perceive transitions from a "reactive" perception stack to a "self-aware" mapping system. The result is a trajectory that is smooth at the frame-to-frame level (thanks to IMU/ICP) and globally accurate over time (thanks to GPS/Loop Closure). This global consistency is the prerequisite for building the **Accumulated BEV World Map** in Milestone 3.
