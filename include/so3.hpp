// so3.hpp
// SO(3) Lie group primitives — implement from scratch.
// Used by: imu_preintegration.cpp, pose_graph_slam.cpp
//
// YOUR TASK (P2-M2.0, foundation):
//   P2-M2.0.1: hat / vee — vector ↔ skew-symmetric matrix
//   P2-M2.0.2: Exp — tangent space ℝ³ → rotation matrix SO(3) (Rodrigues)
//   P2-M2.0.3: Log — rotation matrix SO(3) → tangent space ℝ³
//   P2-M2.0.4: Jr — right Jacobian of SO(3)
//
// Key insight: additive updates δφ in ℝ³ map to multiplicative updates on SO(3)
// via R ← R * Exp(δφ). This is the "on-manifold" update — the whole reason
// Forster et al. (2015) wrote the IMU preintegration paper.
//
// References:
//   Forster et al., "IMU Preintegration on Manifold" (RSS 2015) — Section II
//   Solà et al., "A micro Lie theory for state estimation" (arXiv 1812.01537) — §4
//   Eade, "Lie Groups for Computer Vision" — §3
//
// Numerical gotcha: small-angle cases (‖φ‖ → 0) need Taylor expansion, not
// direct sin/cos, or you'll get 0/0. Similarly for Log when R ≈ identity and
// when the rotation angle ≈ π.

#pragma once
#include <Eigen/Dense>

namespace so3 {

// hat operator — Forster Eq. (1)
// Maps ω = [ω₁, ω₂, ω₃]ᵀ to the 3×3 skew-symmetric matrix:
//     [  0   -ω₃   ω₂ ]
//     [  ω₃   0   -ω₁ ]
//     [ -ω₂   ω₁   0  ]
// Property: ω^ * b == ω × b (skew-symmetric acts as cross product).
Eigen::Matrix3d hat(const Eigen::Vector3d& omega);

// vee operator — inverse of hat.
// Extracts the ℝ³ vector from a skew-symmetric 3×3 matrix.
// Does NOT validate skew-symmetry — caller's responsibility to pass a valid
// skew-symmetric input (i.e., result of hat() or R - Rᵀ).
Eigen::Vector3d vee(const Eigen::Matrix3d& Omega);

// Exponential map on SO(3) — Forster Eq. (3), Rodrigues' formula.
// Maps a rotation vector φ ∈ ℝ³ (tangent space at identity) to a rotation
// matrix R ∈ SO(3). Axis = φ / ‖φ‖, angle = ‖φ‖.
//
//   Exp(φ) = I + (sin ‖φ‖ / ‖φ‖) φ^ + ((1 - cos ‖φ‖) / ‖φ‖²) (φ^)²
//
// Small-angle case (‖φ‖ < ε): use first-order approximation Exp(φ) ≈ I + φ^
// (Forster Eq. 4). Decide your ε threshold — typical values: 1e-8 to 1e-5.
Eigen::Matrix3d Exp(const Eigen::Vector3d& phi);

// Logarithm map on SO(3) — Forster Eq. (5).
// Inverse of Exp. Maps R ∈ SO(3) to φ ∈ ℝ³ such that Exp(φ) == R.
//
//   φ = (θ / (2 sin θ)) · vee(R - Rᵀ),  θ = acos((tr(R) - 1) / 2)
//
// Numerical gotchas:
//   - θ → 0: use Taylor expansion or return vee(R - Rᵀ) / 2 directly
//   - θ → π: (R - Rᵀ) vanishes; need alternative branch (see Solà §4 or Eade §3)
//   - acos argument must be clamped to [-1, 1] — floating-point error can push
//     tr(R) outside valid range even for valid rotations
Eigen::Vector3d Log(const Eigen::Matrix3d& R);

// Right Jacobian of SO(3) — Forster Eq. (8).
// Relates additive perturbation δφ in tangent space to multiplicative
// perturbation on the manifold (Forster Eq. 7):
//     Exp(φ + δφ) ≈ Exp(φ) · Exp(Jr(φ) · δφ)
//
//   Jr(φ) = I − ((1 − cos ‖φ‖) / ‖φ‖²) φ^ + ((‖φ‖ − sin ‖φ‖) / ‖φ‖³) (φ^)²
//
// Needed for:
//   - IMU preintegration noise propagation (Forster Eq. 32–35)
//   - Analytical Jacobians of residuals in pose graph optimization
//
// Small-angle case: Jr(φ) → I as ‖φ‖ → 0. Use Taylor expansion below threshold.
Eigen::Matrix3d Jr(const Eigen::Vector3d& phi);

Eigen::Matrix3d Jr_inv(const Eigen::Vector3d& phi);

}  // namespace so3
