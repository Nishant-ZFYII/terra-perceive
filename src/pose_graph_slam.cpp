// pose_graph_slam.cpp
// Implementations for pose graph SLAM declared in pose_graph_slam.hpp.
// Build incrementally: layer 1 graph assembly → layer 2 residuals →
// layer 3 on-manifold optimizer → layer 4 Euclidean variant.
//
// References:
//   Grisetti et al. 2010 — tutorial on pose graph SLAM
//   Forster et al. 2015 — IMU factor residual (Eq. 37)
//   Solà micro Lie theory — right Jacobians for SE(3) residuals
/*// pose_graph_slam.hpp
// On-manifold pose graph SLAM — from scratch, Eigen sparse as linear algebra backend.
// Ties together ICP odometry + IMU preintegration + GPS + loop closure into a
// unified factor graph, then runs Gauss-Newton / Levenberg-Marquardt on SE(3).
//
// Depends on: so3.hpp (Exp, Log, Jr, hat), imu_preintegration.hpp (IMUFactor)
// Used by:    scripts/run_slam_pipeline.py (via C++ binary)
//
// YOUR TASK (P2-M2.3 + P2-M2.4):
//   P2-M2.3.1: Pose struct + edge structs (data-in/out of the graph)
//   P2-M2.3.2: PoseGraphConfig with toggles for ablation study
//   P2-M2.3.3: Graph assembly (add* methods, setFixed, setInitialPoses)
//   P2-M2.4.1: Factor residuals for 4 edge types (Grisetti tutorial + Forster Eq. 37)
//   P2-M2.4.2: Analytical Jacobians (needed for sparse Hessian assembly)
//   P2-M2.4.3: On-manifold Gauss-Newton / Levenberg-Marquardt optimizer
//   P2-M2.4.4: Euclidean variant (for manifold-vs-Euclidean ablation)
//   P2-M2.4.5: Output trajectory to CSV (same format as poses_icp.csv)
//
// Key insight (on-manifold optimization):
//   Poses live in SE(3) = SO(3) × ℝ³.
//   Updates δx = [δφ, δt] ∈ ℝ⁶ live in the tangent space (se(3)).
//   Build sparse 6N × 6N Hessian H and 6N × 1 vector b from residuals+Jacobians.
//   Solve: δx = SimplicialLDLT(H).solve(-b) via Eigen sparse.
//   Retract: R ← R * Exp(δφ), t ← t + δt.
//   LM damping: H ← H + λI before solve; adjust λ based on cost reduction.
//
// Architecture (Grisetti et al., "Tutorial on Graph-Based SLAM", 2010):
//   Nodes: SE(3) poses at each LiDAR keyframe (2847 nodes for RELLIS-3D seq 00).
//   Edges: 4 types — ICP, IMU, GPS (unary), loop closure — each with its own
//          residual function, Jacobian, and information matrix.
//   Sparsity: each binary edge contributes a 12×12 block to the 6N×6N Hessian.
//             H is extremely sparse (~6 nonzeros per row on average).
//   Solver: Eigen::SimplicialLDLT — sparse Cholesky. Reimplementing this
//           from scratch is numerical engineering with no learning payoff;
//           Eigen is the right tool.
//
// References:
//   Grisetti et al., "A Tutorial on Graph-Based SLAM" (IEEE ITS Mag 2010)
//   Forster et al., "IMU Preintegration on Manifold" (RSS 2015) — Eq. 37 for
//     IMU factor residual that uses our preintegrated (ΔR, Δv, Δp)
//   Shan et al., "LIO-SAM" (IROS 2020) — architecture reference for ICP+IMU+GPS+loop
//   Kummerle et al., "g2o: A General Framework for Graph Optimization" (ICRA 2011)

#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <string>
#include <vector>
#include "so3.hpp"
#include "imu_preintegration.hpp"

namespace pose_graph {

// Gravity in world frame (z-up, ENU convention). Used by IMU factor residual.
const Eigen::Vector3d kGravityWorld(0.0, 0.0, -9.81);

// -----------------------------------------------------------------------------
// Pose: a single SE(3) node in the graph
// -----------------------------------------------------------------------------

struct Pose {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

// -----------------------------------------------------------------------------
// Edge types — one struct per factor type
// -----------------------------------------------------------------------------

// ICP odometry: binary edge from frame i to frame j (typically j = i+1).
// Measurement: relative transform computed by KISS-ICP scan matching.
// Residual (6D): [Log(ΔR_meas^T · R_i^T · R_j), R_i^T (t_j - t_i) - Δp_meas]
struct ICPEdge {
    int from, to;
    Eigen::Matrix3d dR_meas;                   // ΔR measurement
    Eigen::Vector3d dp_meas;                   // Δp measurement
    Eigen::Matrix<double, 6, 6> information;   // Ω = Σ^(-1)
};

// IMU preintegration: binary edge from frame i to frame j.
// Wraps the IMUFactor from imu_preintegration. Gravity enters the residual
// here (NOT during preintegration — see Forster Eq. 37).
struct IMUEdge {
    int from, to;
    imu::IMUFactor factor;                     // ΔR, Δv, Δp, covariance, dt
    // Velocity states v_i, v_j: for a minimal implementation, you may treat
    // velocities as derived (v ≈ (t_j - t_i) / dt) or add them as extra graph
    // state. RECOMMENDATION: for first pass, treat velocity implicitly via
    // the Δp residual and skip the Δv residual. Revisit if ablation shows
    // IMU not contributing enough.
};

// GPS position: UNARY edge on frame i (constrains position only, not rotation).
// HDOP-weighted information: frames under canopy get high σ_gps → low weight.
struct GPSEdge {
    int frame_id;
    Eigen::Vector3d position_meas;             // (x, y, z) in ENU
    Eigen::Matrix3d information;               // diag(1/σ², 1/σ², 1/σ²)
};

// Loop closure: binary edge from frame i to frame j (typically |i-j| >> 1).
// Same structure as ICPEdge but produced by Scan Context + ICP alignment.
// Information matrix typically higher (loop closures are strong constraints).
struct LoopEdge {
    int from, to;
    Eigen::Matrix3d dR_meas;
    Eigen::Vector3d dp_meas;
    Eigen::Matrix<double, 6, 6> information;
};

// -----------------------------------------------------------------------------
// Config — controls ablation axes (edge toggles, optimizer choice, LM params)
// -----------------------------------------------------------------------------

struct PoseGraphConfig {
    // Ablation 1: edge type toggles (produces configs A–E of the edge ablation)
    bool use_icp_edges  = true;
    bool use_imu_edges  = true;
    bool use_gps_edges  = true;
    bool use_loop_edges = true;

    // Ablation 2: optimizer choice
    enum class OptimizerType {
        Manifold,    // primary: SO(3)/SE(3) on-manifold updates via Exp/Log
        Euclidean,   // comparison: additive R update, SVD-reproject to SO(3)
        // G2O      // external library (wired in via separate wrapper)
    };
    OptimizerType optimizer = OptimizerType::Manifold;

    // Optimizer parameters (Levenberg-Marquardt)
    int max_iterations = 100;
    double convergence_tol = 1e-6;     // ‖δx‖ threshold
    double lambda_init = 1e-4;         // initial LM damping
    double lambda_up   = 10.0;         // damping increase factor on bad step
    double lambda_down = 0.1;          // damping decrease factor on good step
    bool verbose = false;              // print per-iteration cost + damping
};

// -----------------------------------------------------------------------------
// PoseGraphSLAM — the optimizer class
// -----------------------------------------------------------------------------

class PoseGraphSLAM {
   public:
    explicit PoseGraphSLAM(const PoseGraphConfig& config);

    // Set initial pose estimates (typically from KISS-ICP trajectory).
    // MUST be called before adding edges — edge indices reference pose IDs.
    void setInitialPoses(const std::vector<Pose>& poses);

    // Fix a node's pose (typically frame 0 as the world anchor).
    // Fixed nodes are excluded from the optimization variables.
    void setFixed(int node_id, bool fixed);

    // Add edges. Config flags determine whether each edge type is actually
    // used during optimization — but add them all; toggling happens in optimize().
    void addICPEdge(const ICPEdge& edge);
    void addIMUEdge(const IMUEdge& edge);
    void addGPSEdge(const GPSEdge& edge);
    void addLoopEdge(const LoopEdge& edge);

    // Run the optimizer. Returns the optimized poses.
    // Iterates Gauss-Newton / LM until convergence or max_iterations.
    std::vector<Pose> optimize();

    // Accessors for debugging / ablation reporting.
    double lastCost() const { return last_cost_; }
    int lastIterations() const { return last_iterations_; }

   private:
    PoseGraphConfig config_;
    std::vector<Pose> poses_;                 // current estimates
    std::vector<bool> fixed_;                 // fixed[i] = true → exclude from solve

    std::vector<ICPEdge> icp_edges_;
    std::vector<IMUEdge> imu_edges_;
    std::vector<GPSEdge> gps_edges_;
    std::vector<LoopEdge> loop_edges_;

    double last_cost_ = 0.0;
    int last_iterations_ = 0;

    // -------------------------------------------------------------------------
    // Residual + Jacobian functions (one pair per edge type)
    // -------------------------------------------------------------------------

    // Each residual function returns the error vector; each Jacobian function
    // returns the (d_residual / d_pose_parameters) in the tangent space.
    // For binary edges: Jacobians are 6 (residual dim) × 6 (pose dim) blocks,
    //                   one for each endpoint.
    // For unary GPS edge: Jacobian is 3 × 6.

    // ICP residual (6D = 3 rot + 3 trans)
    Eigen::Matrix<double, 6, 1> residualICP(const ICPEdge& e) const;
    void jacobianICP(const ICPEdge& e,
                     Eigen::Matrix<double, 6, 6>& J_from,
                     Eigen::Matrix<double, 6, 6>& J_to) const;

    // IMU residual (Forster Eq. 37; simplified to 6D rot+pos, skipping velocity
    // state as recommended in the IMUEdge comments). If you add velocity state
    // later, residual becomes 9D and Jacobians grow accordingly.
    Eigen::Matrix<double, 6, 1> residualIMU(const IMUEdge& e) const;
    void jacobianIMU(const IMUEdge& e,
                     Eigen::Matrix<double, 6, 6>& J_from,
                     Eigen::Matrix<double, 6, 6>& J_to) const;

    // GPS residual (unary, 3D position error)
    Eigen::Matrix<double, 3, 1> residualGPS(const GPSEdge& e) const;
    void jacobianGPS(const GPSEdge& e,
                     Eigen::Matrix<double, 3, 6>& J) const;

    // Loop closure residual (same structure as ICP)
    Eigen::Matrix<double, 6, 1> residualLoop(const LoopEdge& e) const;
    void jacobianLoop(const LoopEdge& e,
                      Eigen::Matrix<double, 6, 6>& J_from,
                      Eigen::Matrix<double, 6, 6>& J_to) const;

    // -------------------------------------------------------------------------
    // Sparse system assembly
    // -------------------------------------------------------------------------

    // Walk all active edges (respecting config_ flags), accumulate into the
    // sparse 6N × 6N Hessian H and 6N × 1 vector b.
    //   H = Σ J^T Ω J
    //   b = Σ J^T Ω r
    // where N = poses_.size(). Fixed poses contribute zero blocks.
    void buildLinearSystem(Eigen::SparseMatrix<double>& H,
                           Eigen::VectorXd& b,
                           double& cost) const;

    // -------------------------------------------------------------------------
    // Retraction (on-manifold vs Euclidean)
    // -------------------------------------------------------------------------

    // On-manifold retraction: R ← R * Exp(δφ), t ← t + δt.
    // Euclidean: R ← R + hat(δφ) * R (Euler-like), then SVD-reproject to SO(3).
    //            This is what the paper calls "wrong" — whole point of the ablation.
    void retract(const Eigen::VectorXd& dx);
};

// -----------------------------------------------------------------------------
// CSV I/O helpers
// -----------------------------------------------------------------------------

// Load initial poses from KISS-ICP output (data/poses_icp.csv).
// Format: t, tx, ty, tz, qx, qy, qz, qw  (quaternion convention)
std::vector<Pose> loadPosesFromCSV(const std::string& path);

// Save optimized trajectory to CSV. Same format as poses_icp.csv for
// downstream comparison scripts (ATE/RPE via scripts/compare_odometry.py).
void savePosesToCSV(const std::vector<Pose>& poses,
                    const std::string& path);

}  // namespace pose_graph
*/

#include "pose_graph_slam.hpp"

#include <Eigen/SparseCholesky>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <g2o/core/sparse_optimizer.h>                                                                                                                                  
#include <g2o/core/sparse_block_matrix.h> 
#include <unordered_map>
namespace pose_graph {

// -----------------------------------------------------------------------------
// Layer 1: Graph assembly (start here — no math yet)
// -----------------------------------------------------------------------------

PoseGraphSLAM::PoseGraphSLAM(const PoseGraphConfig& config) : config_(config) {}

void PoseGraphSLAM::setInitialPoses(const std::vector<Pose>& poses)  {
    // YOUR CODE (simple):
    //   poses_ = poses
    //   fixed_.resize(poses.size(), false)   // default: nothing fixed
    poses_ = poses;
    fixed_.resize(poses.size(), false);

}

void PoseGraphSLAM::setFixed(int node_id, bool fixed) {
    // YOUR CODE (simple):
    //   bounds-check node_id
    //   fixed_[node_id] = fixed
    if (node_id < 0 || node_id >= fixed_.size()) {
        std::cerr << "Error: node_id out of bounds in setFixed: " << node_id << std::endl;
        return;
    }
    fixed_[node_id] = fixed;
}

void PoseGraphSLAM::addICPEdge(const ICPEdge& edge) {
    // YOUR CODE: icp_edges_.push_back(edge);
    icp_edges_.push_back(edge);

}

void PoseGraphSLAM::addIMUEdge(const IMUEdge& edge) {
    // YOUR CODE: imu_edges_.push_back(edge);
    imu_edges_.push_back(edge);

}

void PoseGraphSLAM::addGPSEdge(const GPSEdge& edge) {
    // YOUR CODE: gps_edges_.push_back(edge);
    gps_edges_.push_back(edge);
}

void PoseGraphSLAM::addLoopEdge(const LoopEdge& edge) {
    // YOUR CODE: loop_edges_.push_back(edge);
    loop_edges_.push_back(edge);
}

// -----------------------------------------------------------------------------
// Layer 2: Residuals + Jacobians (the math core)
// -----------------------------------------------------------------------------
//
// IMPLEMENTATION ORDER:
//   1. ICP residual + Jacobian  ← start here, simplest binary edge
//   2. GPS residual + Jacobian  ← simplest, unary
//   3. Loop residual + Jacobian ← identical structure to ICP
//   4. IMU residual + Jacobian  ← Forster Eq. 37, most involved
//
// Test each edge type's residual in isolation before moving to the next.
// -----------------------------------------------------------------------------

Eigen::Matrix<double, 6, 1> PoseGraphSLAM::residualICP(const ICPEdge& e) const {
    // YOUR CODE — Grisetti tutorial Section II.C:
    //   Let R_i, t_i = poses_[e.from].R, .t
    //   Let R_j, t_j = poses_[e.to].R, .t
    //   Predicted relative transform:
    //     dR_pred = R_i.transpose() * R_j
    //     dp_pred = R_i.transpose() * (t_j - t_i)
    //   Residual (6D, first 3 = rotation, last 3 = translation):
    //     r_rot   = so3::Log(e.dR_meas.transpose() * dR_pred)
    //     r_trans = dp_pred - e.dp_meas
    //   Return r stacked: [r_rot; r_trans]
    Eigen::Matrix<double, 6, 1> r;
    const auto& R_i = poses_[e.from].R;
    const auto& t_i = poses_[e.from].t;

    const auto& R_j = poses_[e.to].R;
    const auto& t_j = poses_[e.to].t;

    Eigen::Vector3d dp_pred = R_i.transpose() * (t_j - t_i);
    Eigen::Matrix3d dR_pred = R_i.transpose() * R_j;

    //errors
    r.head<3>() = so3::Log(e.dR_meas.transpose() * dR_pred);
    r.tail<3>() = dp_pred - e.dp_meas;

    return r;
}

void PoseGraphSLAM::jacobianICP(const ICPEdge& e,
                                Eigen::Matrix<double, 6, 6>& J_from,
                                Eigen::Matrix<double, 6, 6>& J_to) const {
    // YOUR CODE — analytical Jacobians, from Grisetti Section II.D:
    //   J_from = ∂r / ∂[δφ_i, δt_i]   (6x6)
    //   J_to   = ∂r / ∂[δφ_j, δt_j]   (6x6)
    //
    // Derivation hint: linearize r around current (R_i, t_i, R_j, t_j) using
    // the SE(3) right perturbation convention:
    //   R_i ← R_i · Exp(δφ_i),  t_i ← t_i + R_i · δt_i  (or just t_i + δt_i)
    // Work out ∂r_rot / ∂δφ, etc. Use so3::Jr for the chain rule.
    //
    // If you prefer, implement numerical Jacobians first (central differences)
    // as a sanity check, then derive analytical versions and compare.
    /*  For ICP residual r = [Log(ΔR_meas^T · R_i^T · R_j); R_i^T(t_j - t_i) - Δp_meas]:

    Using the SE(3) right perturbation convention (R ← R · Exp(δφ), t ← t + δt):

    J_from (∂r/∂[δφ_i, δt_i]): 6×6 block, with sub-blocks:
    ∂r_rot   / ∂δφ_i  =  -Jr⁻¹(r_rot) · ΔR_pred^T          (3×3)
    ∂r_rot   / ∂δt_i  =   0                                 (3×3)
    ∂r_trans / ∂δφ_i  =   hat(R_i^T (t_j - t_i))            (3×3)
    ∂r_trans / ∂δt_i  =  -I                                 (3×3)

    J_to (∂r/∂[δφ_j, δt_j]):
    ∂r_rot   / ∂δφ_j  =   Jr⁻¹(r_rot)                       (3×3)
    ∂r_rot   / ∂δt_j  =   0                                 (3×3)
    ∂r_trans / ∂δφ_j  =   0                                 (3×3)
    ∂r_trans / ∂δt_j  =   R_i^T · R_j                       (3×3)

    Where Jr⁻¹ is the inverse of the right Jacobian. For practical purposes near the optimum, you can approximate Jr⁻¹(r_rot) ≈ I since r_rot is small at convergence — the
    optimizer still works (it's slightly slower to converge but Gauss-Newton is forgiving).

    Layout in the 6×6 J_from matrix:
    J_from = [ ∂r_rot/∂δφ_i    ∂r_rot/∂δt_i   ]   ← rows 0-2
            [ ∂r_trans/∂δφ_i  ∂r_trans/∂δt_i ]   ← rows 3-5
                cols 0-2          cols 3-5
    */
    J_from.setZero();
    //approximate Jr⁻¹(r_rot) ≈ I for simplicity, since r_rot is small near convergence
    //∂r_rot   / ∂δφ_i  =  -Jr⁻¹(r_rot) · ΔR_pred^T          (3×3)
    const auto& R_i = poses_[e.from].R;
    const auto& t_i = poses_[e.from].t;
    const auto& R_j = poses_[e.to].R;
    const auto& t_j = poses_[e.to].t;
    Eigen::Vector3d dp_pred = R_i.transpose() * (t_j - t_i);
    Eigen::Matrix3d dR_pred = R_i.transpose() * R_j;
    Eigen::Vector3d r_rot = so3::Log(e.dR_meas.transpose() * dR_pred);
    Eigen::Matrix3d Jr_inv_rot = so3::Jr_inv(r_rot);
    //call Jr inverse from Jr_inv
    J_from.block<3, 3>(0, 0) = -Jr_inv_rot * dR_pred.transpose();
    //∂r_rot   / ∂δt_i  =   0                                 (3×3)
    J_from.block<3, 3>(0, 3).setZero(); 

    //∂r_trans / ∂δφ_i  =   hat(R_i^T (t_j - t_i))            (3×3)
    J_from.block<3, 3>(3, 0) = so3::hat(dp_pred);
    //∂r_trans / ∂δt_i  =  -I                                 (3×3)
    J_from.block<3, 3>(3, 3) = -R_i.transpose();

    //J_to
    J_to.setZero();
    //∂r_rot   / ∂δφ_j  =   Jr⁻¹(r_rot)                       (3×3)
    J_to.block<3, 3>(0, 0) = Jr_inv_rot;
    //∂r_rot   / ∂δt_j  =   0                                 (3×3)
    J_to.block<3, 3>(0, 3).setZero();
    //∂r_trans / ∂δφ_j  =   0                                 (3×3)
    J_to.block<3, 3>(3, 0).setZero();
    //∂r_trans / ∂δt_j  =   R_i^T · R_j                       (3×3)
    J_to.block<3, 3>(3, 3) = R_i.transpose() ;


}

Eigen::Matrix<double, 3, 1> PoseGraphSLAM::residualGPS(const GPSEdge& e) const {
    // YOUR CODE — simplest residual:
    //   r = poses_[e.frame_id].t - e.position_meas
    Eigen::Matrix<double, 3, 1> r;
    const auto& t = poses_[e.frame_id].t;
    r = t - e.position_meas;
    return r;

}

void PoseGraphSLAM::jacobianGPS(const GPSEdge& e,
                                Eigen::Matrix<double, 3, 6>& J) const {
    // YOUR CODE:
    //   Residual depends only on t. Rotation block = 0, translation block = I.
    //   J = [0₃ | I₃]  (3×6, columns 0-2 are rotation Jacobian, 3-5 translation)
    J.setZero();
    J.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    
}

Eigen::Matrix<double, 6, 1> PoseGraphSLAM::residualLoop(const LoopEdge& e) const {
    // YOUR CODE: same as residualICP but with e.dR_meas, e.dp_meas from loop.
    // You can literally call residualICP with an ICPEdge built from this LoopEdge
    // to avoid duplication — up to you.
    Eigen::Matrix<double, 6, 1> r;
    //call residual icp from loop edge
    ICPEdge icp_edge;
    icp_edge.from = e.from;
    icp_edge.to = e.to;
    icp_edge.dR_meas = e.dR_meas;
    icp_edge.dp_meas = e.dp_meas;
    r = residualICP(icp_edge);

    return r;
}

void PoseGraphSLAM::jacobianLoop(const LoopEdge& e,
                                 Eigen::Matrix<double, 6, 6>& J_from,
                                 Eigen::Matrix<double, 6, 6>& J_to) const {
    // YOUR CODE: same structure as jacobianICP
    J_from.setZero();
    J_to.setZero();
    //call jacobian icp from loop edge
    ICPEdge icp_edge;
    icp_edge.from = e.from;
    icp_edge.to = e.to; 
    icp_edge.dR_meas = e.dR_meas;
    icp_edge.dp_meas = e.dp_meas;
    jacobianICP(icp_edge, J_from, J_to);

}

Eigen::Matrix<double, 6, 1> PoseGraphSLAM::residualIMU(const IMUEdge& e) const {
    // YOUR CODE — Forster Eq. 37 (simplified — 6D residual, skipping velocity):
    //   Let R_i, t_i = poses_[e.from].R, .t
    //   Let R_j, t_j = poses_[e.to].R, .t
    //   Let dt = e.factor.dt
    //
    //   Predicted rotation:  dR_pred = R_i.transpose() * R_j
    //   Predicted position:  dp_pred = R_i.transpose() * (t_j - t_i - 0.5 * kGravityWorld * dt²)
    //                        (subtract gravity's predicted contribution — see Forster Eq. 31)
    //
    //   Residual (6D):
    //     r_rot   = so3::Log(e.factor.dR.transpose() * dR_pred)
    //     r_trans = dp_pred - e.factor.dp
    //
    // NOTE: if you add velocity state later, expand to 9D residual including
    //       r_vel = R_i.transpose() * (v_j - v_i - kGravityWorld * dt) - e.factor.dv
    Eigen::Matrix<double, 6, 1> r;
    const auto& R_i = poses_[e.from].R;
    const auto& t_i = poses_[e.from].t;

    const auto& R_j = poses_[e.to].R;
    const auto& t_j = poses_[e.to].t;

    double dt = e.factor.dt;
    Eigen::Vector3d dp_pred = R_i.transpose() * (t_j - t_i - 0.5 * kGravityWorld * dt * dt);
    Eigen::Matrix3d dR_pred = R_i.transpose() * R_j;

    r.head<3>() = so3::Log(e.factor.dR.transpose() * dR_pred);
    r.tail<3>() = dp_pred - e.factor.dp;
    return r;
}

void PoseGraphSLAM::jacobianIMU(const IMUEdge& e,
                                Eigen::Matrix<double, 6, 6>& J_from,
                                Eigen::Matrix<double, 6, 6>& J_to) const {
    // YOUR CODE: similar structure to jacobianICP; gravity adds a term to
    // the translation Jacobian. Start with numerical Jacobian for validation,
    // then derive analytical.
    J_from.setZero();
    J_to.setZero();
    //Gravity has 0 residual.
    const auto& R_i = poses_[e.from].R;
    const auto& t_i = poses_[e.from].t;

    const auto& R_j = poses_[e.to].R;
    const auto& t_j = poses_[e.to].t;

    double dt = e.factor.dt;
    Eigen::Vector3d dp_pred = R_i.transpose() * (t_j - t_i - 0.5 * kGravityWorld * dt * dt);
    Eigen::Matrix3d dR_pred = R_i.transpose() * R_j;
    Eigen::Vector3d r_rot = so3::Log(e.factor.dR.transpose() * dR_pred);
    Eigen::Matrix3d Jr_inv_rot = so3::Jr_inv(r_rot);

    //∂r_rot   / ∂δφ_i  =  -Jr⁻¹(r_rot) · ΔR_pred^T          (3×3)
    J_from.block<3, 3>(0, 0) = -Jr_inv_rot * dR_pred.transpose(); // ∂r_rot / ∂δφ_i
    //∂r_rot   / ∂δt_i  =   0                                 (3×3)
    J_from.block<3, 3>(0, 3).setZero();
    //∂r_trans / ∂δφ_i  =  -R_i.transpose() * (t_j - t_i - 0.5 * kGravityWorld * dt²) ×   (3×3)
    J_from.block<3, 3>(3, 0) = so3::hat(dp_pred);
    //∂r_trans / ∂δt_i  =   -R_i.transpose()                                 (3×3)
    J_from.block<3, 3>(3, 3) = -R_i.transpose();

    //J_to
    //∂r_rot   / ∂δφ_j  =   Jr⁻¹(r_rot)                       (3×3)
    J_to.block<3, 3>(0, 0) = Jr_inv_rot;
    //∂r_rot   / ∂δt_j  =   0                                 (3×3)
    J_to.block<3, 3>(0, 3).setZero();
    //∂r_trans / ∂δφ_j  =   0                                 (3×3)
    J_to.block<3, 3>(3, 0).setZero();
    //∂r_trans / ∂δt_j  =   R_i.transpose()                       (3×3)
    J_to.block<3, 3>(3, 3) = R_i.transpose();

}

// -----------------------------------------------------------------------------
// Layer 3: Sparse system assembly + on-manifold optimizer
// -----------------------------------------------------------------------------

void PoseGraphSLAM::buildLinearSystem(Eigen::SparseMatrix<double>& H,
                                      Eigen::VectorXd& b,
                                      double& cost) const {
    // YOUR CODE:
    //   Walk all active edges (check config_.use_*_edges flags).
    //   For each edge:
    //     - compute residual r and information Ω (or = covariance.inverse())
    //     - compute Jacobian(s) J
    //     - accumulate into H: H += J^T * Ω * J  at the appropriate 6×6 blocks
    //     - accumulate into b: b += J^T * Ω * r  at the appropriate 6×1 blocks
    //     - accumulate cost: cost += r^T * Ω * r
    //     - SKIP contributions to fixed poses (zero out those blocks)
    //
    // Use Eigen::Triplet<double> and setFromTriplets for efficient sparse
    // Hessian assembly. H should be 6N × 6N where N = poses_.size().
    //
    // IMPORTANT: fixed poses have no variables in the solve. Either:
    //   (b) Build a mapping from "free pose index" to row/col in H, so H is
    //       6 * (N - n_fixed) × 6 * (N - n_fixed). Cleaner but more bookkeeping.
    // Start with (a) for simplicity.
    /*  Walk through ICP edges                                                                                                                                                    
                                                                                                                                                                          
  For each ICP edge e between nodes i and j:                                                                                                                                
                                                                                                                                                                            
  1. Compute residual: r = residualICP(e) (6×1)                                                                                                                             
  2. Compute Jacobians: jacobianICP(e, J_from, J_to) (each 6×6)                                                                                                             
  4. Accumulate cost: cost += r.transpose() * Ω * r                                                                                                                         
  5. Compute the contribution to b:                                                                                                                                         
    - b.segment<6>(6*i) += J_from.transpose() * Ω * r                                                                                                                     
    - b.segment<6>(6*j) += J_to.transpose() * Ω * r                                                                                                                         
  6. Compute the four 6×6 contributions to H:                                                                                                                               
    - H_ii block += J_from^T · Ω · J_from  (diagonal block of node i)                                                                                                       
    - H_jj block += J_to^T · Ω · J_to     (diagonal block of node j)                                                                                                        
    - H_ij block += J_from^T · Ω · J_to   (off-diagonal block)                                                                                                              
    - H_ji block += J_to^T · Ω · J_from   (transpose of above)                                                                                                              
                                                                                                                                                                            
  Adding a 6×6 block to triplets                                                                                                                                            
                                                                                                                                                                          
  Helper lambda:                                                                                                                                                            
  auto addBlock = [&](int row_start, int col_start, const Eigen::Matrix<double,6,6>& M) {                                                                                   
      for (int r = 0; r < 6; ++r)                                                                                                                                           
          for (int c = 0; c < 6; ++c)                                                                                                                                       
              triplets.emplace_back(row_start + r, col_start + c, M(r, c));                                                                                               
  };                                                                                                                                                                        
                                                                                                                                                                            
  Skipping fixed poses
                                                                                                                                                                            
  If fixed_[i] is true, don't add b contributions for node i, and don't add any H blocks involving node i. Then add a large value to the diagonal of fixed poses to "fix" 
  them:                                                                                                                                                                     
                                                                                                                                                                          
  if (fixed_[i]) {                                                                                                                                                          
      // Add huge identity to diagonal to anchor (or skip and handle separately)
      for (int k = 0; k < 6; ++k)                                                                                                                                           
          triplets.emplace_back(6*i + k, 6*i + k, 1e12);                                                                                                                  
  }           */

    int N = poses_.size();
    H.resize(6 * N, 6 * N);
    b.resize(6 * N);
    H.setZero();
    b.setZero();

    cost = 0.0;
    std::vector<Eigen::Triplet<double>> triplets;
    //estimate = number of edges * 4 blocks per edge * 36 entries per block + GPS edges * 36 + fixed pose blocks
    size_t est = (icp_edges_.size() + imu_edges_.size() +  loop_edges_.size()) * 4 * 36 + gps_edges_.size() *36 + 6 * N; 
    triplets.reserve(est); // reserve space to avoid reallocations

    auto addBlock = [&](int row_start, int col_start, const Eigen::Matrix<double, 6, 6>& M) {
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 6; ++c)
                triplets.emplace_back(row_start + r, col_start + c, M(r, c)); 
    };

    //build list of triplets


     // Walk through ICP edges
    if (config_.use_icp_edges)  {
        for (const auto& e : icp_edges_){// check config flag
            Eigen::Matrix<double, 6, 1> r = residualICP(e);
            Eigen::Matrix<double, 6, 6> J_from, J_to;
            jacobianICP(e, J_from, J_to);
            const auto& Info = e.information;

            // Accumulate cost
            cost += r.transpose() * Info * r;

            // Accumulate b
            if (!fixed_[e.from]) {
                b.segment<6>(6 * e.from) += J_from.transpose() * Info * r;
            }
            if (!fixed_[e.to]) {
                b.segment<6>(6 * e.to) += J_to.transpose() * Info * r;
            }

            // Accumulate H
            if (!fixed_[e.from]) {
                Eigen::Matrix<double, 6, 6> H_ii = J_from.transpose() * Info * J_from;
                addBlock(6 * e.from, 6 * e.from, H_ii);
            }
            if (!fixed_[e.to] ) {
                Eigen::Matrix<double, 6, 6> H_jj = J_to.transpose() * Info * J_to;
                addBlock(6 * e.to, 6 * e.to, H_jj);
            }
            if (!fixed_[e.from] && !fixed_[e.to]) {
                Eigen::Matrix<double, 6, 6> H_ij = J_from.transpose() * Info * J_to;
                addBlock(6 * e.from, 6 * e.to, H_ij);
                addBlock(6 * e.to, 6 * e.from, H_ij.transpose());
            }
            
        }
    }

    //IMU edges
    if(config_.use_imu_edges){
        for(const auto& e : imu_edges_){
            Eigen::Matrix<double, 6, 1> r = residualIMU(e);
            Eigen::Matrix<double, 6, 6> J_from, J_to;
            jacobianIMU(e, J_from, J_to);
            //Eigen::Matrix<double, 6, 6> Info = Eigen::Matrix<double, 6, 6>::Identity() * 100.0;
            
            if (e.factor.covariance.block<3,3>(0,0).determinant() < 1e-20 ||                                                                                                          
                e.factor.covariance.block<3,3>(6,6).determinant() < 1e-20) {                                                                                                          
                std::cerr << "WARNING: near-zero IMU covariance at edge "
                            << e.from << "->" << e.to << ", skipping\n";                                                                                                                
                continue;  // skip this edge entirely  
            }
            Eigen::Matrix<double, 6, 6> Info;
            Info.setZero();
            Info.block<3,3>(0,0) = e.factor.covariance.block<3,3>(0,0).inverse();
            Info.block<3,3>(3,3) = e.factor.covariance.block<3,3>(6,6).inverse();                                                                                                                               
            
            
            

            // Accumulate cost
            cost += r.transpose() * Info * r;

            // Accumulate b
            if (!fixed_[e.from]) {
                b.segment<6>(6 * e.from) += J_from.transpose() * Info * r;
            }
            if (!fixed_[e.to]) {
                b.segment<6>(6 * e.to) += J_to.transpose() * Info * r;
            }

            // Accumulate H
            if (!fixed_[e.from]) {
                Eigen::Matrix<double, 6, 6> H_ii = J_from.transpose() * Info * J_from;
                addBlock(6 * e.from, 6 * e.from, H_ii);
            }
            if (!fixed_[e.to] ) {
                Eigen::Matrix<double, 6, 6> H_jj = J_to.transpose() * Info * J_to;
                addBlock(6 * e.to, 6 * e.to, H_jj);
            }
            if (!fixed_[e.from] && !fixed_[e.to]) {
                Eigen::Matrix<double, 6, 6> H_ij = J_from.transpose() * Info * J_to;
                addBlock(6 * e.from, 6 * e.to, H_ij);
                addBlock(6 * e.to, 6 * e.from, H_ij.transpose());
            }

        }
    }

    //GPS edges : Unary
    if(config_.use_gps_edges){
        for(const auto& e : gps_edges_){
            Eigen::Matrix<double, 3, 1> r = residualGPS(e);
            Eigen::Matrix<double, 3, 6> J;
            jacobianGPS(e, J);
            const auto Info = e.information;

            // Accumulate cost
            cost += r.transpose() * Info * r;

            if(!fixed_[e.frame_id]){
                b.segment<6>(6 * e.frame_id) += J.transpose() * Info * r;
                Eigen::Matrix<double, 6, 6> H_ii = J.transpose() * Info * J;
                addBlock(6 * e.frame_id, 6 * e.frame_id, H_ii);
            }
        }
    }

    //loop edges similar to ICP
    if(config_.use_loop_edges){
        for(const auto& u: loop_edges_){
            Eigen::Matrix<double, 6, 1> r = residualLoop(u);
            Eigen::Matrix<double, 6, 6> J_from, J_to;
            jacobianLoop(u, J_from, J_to);
            const auto Info = u.information;

            // Accumulate cost
            cost += r.transpose() * Info * r;

            // Accumulate b
            if (!fixed_[u.from]) {
                b.segment<6>(6 * u.from) += J_from.transpose() * Info * r;
            }
            if (!fixed_[u.to]) {
                b.segment<6>(6 * u.to) += J_to.transpose() * Info * r;
            }

            // Accumulate H
            if (!fixed_[u.from]) {
                Eigen::Matrix<double, 6, 6> H_ii = J_from.transpose() * Info * J_from;
                addBlock(6 * u.from, 6 * u.from, H_ii);
            }
            if (!fixed_[u.to] ) {
                Eigen::Matrix<double, 6, 6> H_jj = J_to.transpose() * Info * J_to;
                addBlock(6 * u.to, 6 * u.to, H_jj);
            }
            if (!fixed_[u.from] && !fixed_[u.to]) {
                Eigen::Matrix<double, 6, 6> H_ij = J_from.transpose() * Info * J_to;
                addBlock(6 * u.from, 6 * u.to, H_ij);
                addBlock(6 * u.to, 6 * u.from, H_ij.transpose());
            }

        }
    }

    //Anchor fixed poses
    for(int i = 0; i<N; ++i){
        if(fixed_[i]){
            for (int k = 0; k < 6; ++k) {
                triplets.emplace_back(6 * i + k, 6 * i + k, 1e12);
            }
        }
    }
    H.setFromTriplets(triplets.begin(), triplets.end());

}

std::vector<Pose> PoseGraphSLAM::optimize() {
    // YOUR CODE — Levenberg-Marquardt loop:
    //   double lambda = config_.lambda_init
    //   double prev_cost = infinity
    //
    //   for iter = 0 .. config_.max_iterations:
    //       Eigen::SparseMatrix<double> H; Eigen::VectorXd b; double cost;
    //       buildLinearSystem(H, b, cost)
    //
    //       if config_.verbose: print(iter, cost, lambda)
    //
    //       // Apply LM damping: H + λI
    //       H_damped = H + lambda * SparseMatrix::Identity(H.rows(), H.cols())
    //
    //       // Solve via sparse Cholesky
    //       Eigen::SimplicialLDLT<SparseMatrix<double>> solver(H_damped);
    //       Eigen::VectorXd dx = solver.solve(-b)
    //
    //       // Trial: apply step, recompute cost
    //       snapshot poses → trial poses
    //       retract(dx) on trial
    //       build linear system on trial → trial_cost
    //
    //       if trial_cost < cost:
    //           accept step: poses_ = trial, lambda *= config_.lambda_down
    //       else:
    //           reject step: lambda *= config_.lambda_up
    //
    //       if dx.norm() < config_.convergence_tol: break
    //
    //   last_cost_ = cost; last_iterations_ = iter
    //   return poses_
    //
    // Simpler alternative for first pass: plain Gauss-Newton (no damping).
    // Always accept the step. Remove the trial-retract-compare logic. This
    // works on well-conditioned problems but can diverge on poor initial guesses.
    // Once GN works, layer LM on top.
    \
    /* With LM damping*/
    double lambda = config_.lambda_init;
    double prev_cost = std::numeric_limits<double>::infinity();
    int iter = 0;
    for (; iter < config_.max_iterations; iter++) {
        Eigen::SparseMatrix<double> H;
        Eigen::VectorXd b;
        double cost;
        buildLinearSystem(H, b, cost);

        if (config_.verbose) {
            std::cout << "Iter: " << iter << ", Cost: " << cost << ", Lambda: " << lambda << std::endl;
        }

        // Apply LM damping
        Eigen::SparseMatrix<double> I(H.rows(), H.cols());
        I.setIdentity();
        Eigen::SparseMatrix<double> H_damped = H + lambda * I;

        // Solve via sparse Cholesky
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(H_damped);
        Eigen::VectorXd dx = solver.solve(-b);

        // Trial step
        std::vector<Pose> snapshot = poses_;
        retract(dx);
        Eigen::SparseMatrix<double> H_trial;
        Eigen::VectorXd b_trial;
        double trial_cost;
        buildLinearSystem(H_trial, b_trial, trial_cost);

        if (trial_cost < cost) {
            lambda *= config_.lambda_down;
            prev_cost = cost;
            cost = trial_cost; // update current cost to trial cost after accepting step
        } else {
            poses_ = snapshot;
            lambda *= config_.lambda_up;
        }

        if (dx.norm() < config_.convergence_tol) {
            break; // Convergence criterion met
        }
    }
    last_cost_ = prev_cost; // Store the cost before the last accepted step
    last_iterations_ = iter; // Store the number of iterations
    return poses_;
    

    /*Without LM Damping Only GN
    int iter = 0;

    double cost = 0.0;
    for (; iter < config_.max_iterations; iter++) {
        Eigen::SparseMatrix<double> H;
        Eigen::VectorXd b;
        buildLinearSystem(H, b, cost);
        if (config_.verbose) {
            std::cout << "Iter: " << iter << ", Cost: " << cost << std::endl;
        }
        // Solve via sparse Cholesky
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(H);                                                                                                         
        if (solver.info() != Eigen::Success) {                
            std::cerr << "Cholesky failed\n";                                                                                                                                 
            break;                          
        }
        Eigen::VectorXd dx = solver.solve(-b);
        retract(dx);
        if(dx.norm() < config_.convergence_tol){
            last_cost_ = cost;
            last_iterations_ = iter+1;
            return poses_;
        }
    }
    last_cost_ = cost;
    last_iterations_ = iter;
    return poses_;
    */
}

// -----------------------------------------------------------------------------
// Layer 4: Retraction — on-manifold vs Euclidean
// -----------------------------------------------------------------------------

void PoseGraphSLAM::retract(const Eigen::VectorXd& dx) {
    // YOUR CODE:
    //   For i = 0 .. poses_.size() - 1:
    //     if fixed_[i]: continue
    //     δφ = dx.segment<3>(6*i + 0)
    //     δt = dx.segment<3>(6*i + 3)
    //
    //     if config_.optimizer == OptimizerType::Manifold:
    //         poses_[i].R = poses_[i].R * so3::Exp(δφ)   ← on-manifold
    //         poses_[i].t += δt
    //
    //     elif config_.optimizer == OptimizerType::Euclidean:
    //         poses_[i].R += so3::hat(δφ) * poses_[i].R  ← additive (WRONG but deliberate)
    //         // Reproject to SO(3) via SVD to keep R a rotation matrix
    //         Eigen::JacobiSVD<Matrix3d> svd(poses_[i].R, ComputeFullU | ComputeFullV)
    //         poses_[i].R = svd.matrixU() * svd.matrixV().transpose()
    //         poses_[i].t += δt
    //
    // This is THE KEY DIFFERENCE in the optimizer ablation. Manifold keeps
    // rotations valid by construction; Euclidean drifts off SO(3) and needs
    // constant SVD reprojection, losing accuracy.
    for(int i = 0; i < poses_.size(); ++i) {
        if (fixed_[i]) continue;
        Eigen::Vector3d delta_phi = dx.segment<3>(6 * i + 0); //body frame
        Eigen::Vector3d delta_t = dx.segment<3>(6 * i + 3); //world frame

        if (config_.optimizer == PoseGraphConfig::OptimizerType::Manifold) {
            poses_[i].R = poses_[i].R * so3::Exp(delta_phi);
            poses_[i].t += delta_t;
        } else if (config_.optimizer == PoseGraphConfig::OptimizerType::Euclidean) {
            poses_[i].R += so3::hat(delta_phi) * poses_[i].R;
            // Reproject to SO(3) via SVD
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(poses_[i].R, Eigen::ComputeFullU | Eigen::ComputeFullV);
            poses_[i].R = svd.matrixU() * svd.matrixV().transpose();
            poses_[i].t += delta_t;
        }
    }
}

// -----------------------------------------------------------------------------
// CSV I/O
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Header-driven CSV loader (P2-M3 step 0)
// -----------------------------------------------------------------------------
// Supports all four P2-M3 pose sources without caller-side changes:
//
//   poses_icp.csv            timestamp, x,  y,  z,  qw, qx, qy, qz     ← W-first
//   poses_gps.csv            timestamp, x,  y,  z,  qw, qx, qy, qz     ← W-first (epoch ns)
//   poses_carto.csv          timestamp, x,  y,  z,  qw, qx, qy, qz     ← W-first
//   poses_slam_manifold.csv  frame_id,  tx, ty, tz, qx, qy, qz, qw     ← W-last
//
// Strategy:
//   1. Read header line → map column-name → column-index.
//   2. Resolve aliases:
//        time column:   "timestamp" | "frame_id" | "t"   (exactly one required)
//        x column:      "x" | "tx"                       (exactly one required)
//        y column:      "y" | "ty"                       (exactly one required)
//        z column:      "z" | "tz"                       (exactly one required)
//        quaternion:    "qw","qx","qy","qz"              (all four required by name)
//   3. Per-row parse indexes BY NAME — column order in the file is irrelevant.
//   4. Always call Eigen::Quaterniond(qw, qx, qy, qz) — Eigen's constructor IS
//      W-first. The header-column order doesn't matter because we pull by name.
//
// Error handling: on any header malformation, log the specific missing/
// ambiguous column and return empty vectors. Callers MUST check for empty().
// -----------------------------------------------------------------------------

namespace {

// Helper: find first matching alias in the column map; -1 if none present.
// Returns -2 if MORE than one alias is present (ambiguous — file is malformed).
static int findAlias(const std::unordered_map<std::string, int>& cols,
                     std::initializer_list<const char*> aliases) {
    // YOUR CODE:
    //   int found = -1;
    //   for (const char* name : aliases) {
    //       auto it = cols.find(name);
    //       if (it != cols.end()) {
    //           if (found != -1) return -2;            // ambiguous: two aliases in same file
    //           found = it->second;
    //       }
    //   }
    //   return found;
    int found = -1;
    for (const char* name : aliases) {
        auto it = cols.find(name);
        if (it != cols.end()) {
            if (found != -1) return -2; // ambiguous: two aliases in same file
            found = it->second;
        }
    }
    return found;
}

// Helper: tokenize a comma-separated line into a vector<string>.
// Leading/trailing whitespace is trimmed per token (robust to "tx, ty, tz"
// vs "tx,ty,tz" header styles). Empty trailing fields ARE preserved as "".
static std::vector<std::string> splitCSV(const std::string& line) {
    // YOUR CODE:
    //   std::vector<std::string> out;
    //   std::istringstream ss(line);
    //   std::string tok;
    //   while (std::getline(ss, tok, ',')) {
    //       // Trim whitespace from both ends.
    //       auto l = tok.find_first_not_of(" \t\r\n");
    //       auto r = tok.find_last_not_of(" \t\r\n");
    //       out.push_back((l == std::string::npos) ? "" : tok.substr(l, r - l + 1));
    //   }
    //   return out;
    std::vector<std::string> out;
    std::istringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // Trim whitespace from both ends.
        auto l = tok.find_first_not_of(" \t\r\n");
        auto r = tok.find_last_not_of(" \t\r\n");
        out.push_back((l == std::string::npos) ? "" : tok.substr(l, r - l + 1));
    }
    return out;
}

}  // anonymous namespace

std::pair<std::vector<Pose>, std::vector<double>> loadPosesFromCSV(const std::string& path) {
    std::vector<Pose>   poses;
    std::vector<double> timestamps;

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[loadPosesFromCSV] failed to open: " << path << std::endl;
        return {poses, timestamps};
    }

    // -------------------------------------------------------------------------
    // 1) Read and parse the header line → column-name → index map.
    // -------------------------------------------------------------------------
    std::string header_line;
    if (!std::getline(file, header_line)) {
        std::cerr << "[loadPosesFromCSV] empty file: " << path << std::endl;
        return {poses, timestamps};
    }

    // YOUR CODE:
    //   std::vector<std::string> header = splitCSV(header_line);
    //   std::unordered_map<std::string, int> col;
    //   for (int i = 0; i < (int)header.size(); ++i) col[header[i]] = i;
    //
    //   Debug check: every column name should be non-empty. If not, the header
    //   is malformed — log and return empty.
    std::vector<std::string> header = splitCSV(header_line);
    std::unordered_map<std::string, int> col;
    for (int i = 0; i < (int)header.size(); ++i) {
        if (header[i].empty()) {
            std::cerr << "[loadPosesFromCSV] malformed header (empty column name): " << path << "\n  header was: " << header_line << "\n";
            return {poses, timestamps};
        }
        col[header[i]] = i;
    }

    // -------------------------------------------------------------------------
    // 2) Resolve aliases. If ANY required column is missing, bail with a clear
    //    error message naming the missing field.
    // -------------------------------------------------------------------------
    // YOUR CODE:
    //   int it_time = findAlias(col, {"timestamp", "frame_id", "t"});
    //   int it_x    = findAlias(col, {"x", "tx"});
    //   int it_y    = findAlias(col, {"y", "ty"});
    //   int it_z    = findAlias(col, {"z", "tz"});
    //   int it_qw   = findAlias(col, {"qw"});
    //   int it_qx   = findAlias(col, {"qx"});
    //   int it_qy   = findAlias(col, {"qy"});
    //   int it_qz   = findAlias(col, {"qz"});
    //
    //   auto require = [&](int idx, const char* label) {
    //       if (idx == -1) {
    //           std::cerr << "[loadPosesFromCSV] missing column: " << label
    //                     << " in " << path << "\n"
    //                     << "  header was: " << header_line << "\n";
    //           return false;
    //       }
    //       if (idx == -2) {
    //           std::cerr << "[loadPosesFromCSV] ambiguous alias for " << label
    //                     << " in " << path << " (more than one match)\n";
    //           return false;
    //       }
    //       return true;
    //   };
    //   if (!require(it_time, "time") || !require(it_x, "x") || !require(it_y, "y")
    //       || !require(it_z, "z")    || !require(it_qw, "qw") || !require(it_qx, "qx")
    //       || !require(it_qy, "qy") || !require(it_qz, "qz")) {
    //       return {poses, timestamps};
    //   }
    //
    //   Optional: detect which quaternion convention this file uses and log it.
    //   This is useful for the blog's "debugging story" sidebar. Pseudocode:
    //     const bool w_first = col["qw"] < col["qx"];
    //     std::cerr << "[loadPosesFromCSV] " << path
    //               << " detected " << (w_first ? "W-first" : "W-last") << "\n";

    int it_time = findAlias(col, {"timestamp", "frame_id", "t"});
    int it_x    = findAlias(col, {"x", "tx"});
    int it_y    = findAlias(col, {"y", "ty"});
    int it_z    = findAlias(col, {"z", "tz"});
    int it_qw   = findAlias(col, {"qw"});
    int it_qx   = findAlias(col, {"qx"});
    int it_qy   = findAlias(col, {"qy"});
    int it_qz   = findAlias(col, {"qz"});   

    auto require = [&](int idx, const char* label) {
        if (idx == -1) {
            std::cerr << "[loadPosesFromCSV] missing column: " << label
                      << " in " << path << "\n"
                      << "  header was: " << header_line << "\n";
            return false;
        }
        if (idx == -2) {
            std::cerr << "[loadPosesFromCSV] ambiguous alias for " << label
                      << " in " << path << " (more than one match)\n";
            return false;
        }
        return true;
    };
    if (!require(it_time, "time") || !require(it_x, "x") || !require(it_y, "y")
        || !require(it_z, "z")    || !require(it_qw, "qw") || !require(it_qx, "qx")
        || !require(it_qy, "qy") || !require(it_qz, "qz")) {
        return {poses, timestamps};
    }

    const bool w_first = col["qw"] < col["qx"];
    std::cerr << "[loadPosesFromCSV] " << path
              << " detected " << (w_first ? "W-first" : "W-last") << "\n";



    // -------------------------------------------------------------------------
    // 3) Parse data rows. Index by NAME (using it_* indices), never by position.
    // -------------------------------------------------------------------------
    std::string line;
    int line_no = 1;                              // header was line 1
    while (std::getline(file, line)) {
        ++line_no;
        if (line.empty()) continue;

        // YOUR CODE:
        //   std::vector<std::string> toks = splitCSV(line);
        //   if ((int)toks.size() < (int)header.size()) {
        //       std::cerr << "[loadPosesFromCSV] short row at " << path
        //                 << ":" << line_no << " (" << toks.size()
        //                 << " cols, expected " << header.size() << ")\n";
        //       continue;
        //   }
        //
        //   try {
        //       double t  = std::stod(toks[it_time]);
        //       double tx = std::stod(toks[it_x]);
        //       double ty = std::stod(toks[it_y]);
        //       double tz = std::stod(toks[it_z]);
        //       double qw = std::stod(toks[it_qw]);
        //       double qx = std::stod(toks[it_qx]);
        //       double qy = std::stod(toks[it_qy]);
        //       double qz = std::stod(toks[it_qz]);
        //
        //       // Eigen's ctor IS (w, x, y, z). Column order in the file is
        //       // irrelevant because we pulled by name above.
        //       Eigen::Quaterniond q(qw, qx, qy, qz);
        //       q.normalize();                              // guard against drift in source CSVs
        //
        //       Pose p;
        //       p.R = q.toRotationMatrix();
        //       p.t = Eigen::Vector3d(tx, ty, tz);
        //       poses.push_back(p);
        //       timestamps.push_back(t);
        //   } catch (const std::exception& e) {
        //       std::cerr << "[loadPosesFromCSV] parse error at " << path
        //                 << ":" << line_no << " — " << e.what() << "\n";
        //       // Continue on bad row rather than abort — RELLIS has some trailing
        //       // whitespace rows in practice.
        //   }
        std::vector<std::string> toks = splitCSV(line);
        if ((int)toks.size() < (int)header.size()) {
            std::cerr << "[loadPosesFromCSV] short row at " << path
                      << ":" << line_no << " (" << toks.size()
                      << " cols, expected " << header.size() << ")\n";
            continue;
        }
        try {
            double t  = std::stod(toks[it_time]);
            double tx = std::stod(toks[it_x]);
            double ty = std::stod(toks[it_y]);
            double tz = std::stod(toks[it_z]);
            double qw = std::stod(toks[it_qw]);
            double qx = std::stod(toks[it_qx]);
            double qy = std::stod(toks[it_qy]);
            double qz = std::stod(toks[it_qz]);

            Eigen::Quaterniond q(qw, qx, qy, qz);
            q.normalize();

            Pose p;
            p.R = q.toRotationMatrix();
            p.t = Eigen::Vector3d(tx, ty, tz);
            poses.push_back(p);
            timestamps.push_back(t);
        } catch (const std::exception& e) {
            std::cerr << "[loadPosesFromCSV] parse error at " << path
                      << ":" << line_no << " — " << e.what() << "\n";
        }
    }

    file.close();

    // Sanity: if we read the header but zero rows, the file was header-only.
    if (poses.empty()) {
        std::cerr << "[loadPosesFromCSV] no rows parsed from " << path
                  << " (header-only or all rows malformed)\n";
    }

    return {poses, timestamps};
}

void savePosesToCSV(const std::vector<Pose>& poses,
                    const std::string& path) {
    // YOUR CODE:
    //   Open file, write header: "frame_id,tx,ty,tz,qx,qy,qz,qw"
    //   For each pose:
    //     Eigen::Quaterniond q(pose.R);
    //     write row with q.x(), q.y(), q.z(), q.w()

    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << path << std::endl;
        return; 
    }
    // Write header
    file << "frame_id,tx,ty,tz,qx,qy,qz,qw\n";
    for (size_t i = 0; i < poses.size(); ++i) {
        const auto& pose = poses[i];
        Eigen::Quaterniond q(pose.R);
        file << i << "," << pose.t.x() << "," << pose.t.y() << "," << pose.t.z() << ","
             << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\n";
    }
    file.close();
}

//evaluate Costs
double PoseGraphSLAM::evaluateCost() const {                                                                                                                              
    double cost = 0.0;                                                                                                                                                  
    if (config_.use_icp_edges) {                                                                                                                                          
        for (const auto& e : icp_edges_) {  
            auto r = residualICP(e);                                                                                                                                      
            cost += r.transpose() * e.information * r;                                                                                                                  
        }                                                                                                                                                                 
    }
    if (config_.use_gps_edges) {                                                                                                                                          
        for (const auto& e : gps_edges_) {                                                                                                                              
            auto r = residualGPS(e);                                                                                                                                      
            cost += r.transpose() * e.information * r;                                                                                                                  
        }                                                                                                                                                                 
    }                                   
    if (config_.use_loop_edges) {                                                                                                                                         
        for (const auto& e : loop_edges_) {                                                                                                                               
            auto r = residualLoop(e);                                                                                                                                   
            cost += r.transpose() * e.information * r;                                                                                                                    
        }                                                                                                                                                                 
    }
    if (config_.use_imu_edges) {                                                                                                                                        
        for (const auto& e : imu_edges_) {
            auto r = residualIMU(e);                                                                                                                                      
            // For IMU, information matrix from preintegration covariance — first 6 dims
            //Eigen::Matrix<double,6,6> info = e.factor.covariance.block<6,6>(0,0).inverse();                                                                               
            //Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Identity() * 100.0;
            if (e.factor.covariance.block<3,3>(0,0).determinant() < 1e-20 ||                                                                                                          
                e.factor.covariance.block<3,3>(6,6).determinant() < 1e-20) {                                                                                                             
                continue;  // skip this edge entirely  
            }
            Eigen::Matrix<double, 6, 6> Info;
            Info.setZero();
            Info.block<3,3>(0,0) = e.factor.covariance.block<3,3>(0,0).inverse();
            Info.block<3,3>(3,3) = e.factor.covariance.block<3,3>(6,6).inverse(); 
            cost += r.transpose() * Info * r;                                                                                                                           
        }                                                                                                                                                               
    }                                                                                                                                                                     
    return cost;                                                                                                                                                          
}  


// =============================================================================                                                                                          
// Marginal covariance extraction (P2-M3)                 
// =============================================================================                                                                                          

std::vector<Eigen::Matrix<double, 6, 6>> computeMarginalsG2O(                                                                                                             
    g2o::SparseOptimizer& optimizer, int num_poses) {                                                                                                                                    
    // YOUR CODE:
    //                                                                                                                                                                    
    // 1) Build the vertex-pair list for the diagonal blocks of H^-1:
    //      std::vector<std::pair<int,int>> pairs;                                                                                                                        
    //      for (int i = 0; i < num_poses; ++i)
    //        pairs.emplace_back(i, i);                                                                                                                                   
    //                                                    
    // 2) Call computeMarginals:                                                                                                                                          
    //      g2o::SparseBlockMatrix<Eigen::MatrixXd> spinv;                                                                                                                
    //      bool ok = optimizer.computeMarginals(spinv, pairs);                                                                                                           
    //      if (!ok) {  /* log + return vector filled with Identity*1e9 */  }                                                                                             
    //                                                                                                                                                                    
    // 3) Extract each diagonal block:                    
    //      std::vector<Eigen::Matrix<double,6,6>> out(num_poses);                                                                                                        
    //      for (int i = 0; i < num_poses; ++i) {                                                                                                                         
    //        auto* block = spinv.block(i, i);
    //        if (!block || block->rows() != 6 || block->cols() != 6) {                                                                                                   
    //          out[i] = Eigen::Matrix<double,6,6>::Identity() * 1e9;  // "no info"                                                                                       
    //          continue;                                                                                                                                                 
    //        }                                                                                                                                                           
    //        out[i] = *block;   // dynamic → 6x6 copy                                                                                                                    
    //                                                                                                                                                                    
    //        // Gotcha: g2o SE3 may order blocks [trans | rot]. If your convention
    //        // is [rot | trans] (Forster-style), permute here:                                                                                                          
    //        //   Eigen::PermutationMatrix<6> P; P.setIdentity();                                                                                                        
    //        //   // swap top 3 with bottom 3 rows/cols                                                                                                                  
    //        //   out[i] = P * out[i] * P.transpose();                                                                                                                   
    //      }                                                                                                                                                             
    //      return out;                     
    std::vector<std::pair<int,int>> pairs;
    std::vector<int> hidx_to_pose_id(num_poses, -1); 
    for (int i = 0; i < num_poses; ++i) {                                                                                                                                     
        auto* v = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i));                                                                                          
        if (!v || v->fixed()) continue;                                                                                                                                       
        const int hidx = v->hessianIndex();                                                                                                                                 
        if (hidx < 0) continue;   // not in the active variable set                                                                                                           
        pairs.emplace_back(hidx, hidx);                                                                                                                                       
        if (hidx < num_poses) hidx_to_pose_id[hidx] = i;                                                                                                                      
    }
    
    g2o::SparseBlockMatrix<Eigen::MatrixXd> spinv;
    bool ok = optimizer.computeMarginals(spinv, pairs);
    std::vector<Eigen::Matrix<double, 6, 6>> out(num_poses, Eigen::Matrix<double, 6, 6>::Identity() * 1e9);  // default sentinel
    if (!ok) {
        std::cerr << "Failed to compute marginals\n";
        return out;
    }

    // Walk the result using hessian indices; map each one back to its pose ID.                                                                                               
    for (size_t h = 0; h < pairs.size(); ++h) {                                                                                                                             
        const int hidx = pairs[h].first;                                                                                                                                      
        auto* block = spinv.block(hidx, hidx);                                                                                                                                
        if (!block || block->rows() != 6 || block->cols() != 6) continue;                                                                                                     
                                                                                                                                                                            
        const int pose_id = hidx_to_pose_id[hidx];                                                                                                                            
        if (pose_id < 0 || pose_id >= num_poses) continue;                                                                                                                    
                                                                                                                                                                            
        out[pose_id] = *block;   // dynamic MatrixXd → fixed 6x6 (dimensions already verified)                                                                                
                                                                                                                                                                            
        // Permute g2o's [trans | rot] to our [rot | trans]. Verified by the                                                                                                  
        // G2OMarginalTranslationBlockIsBottomRight test.                                                                                                                   
        Eigen::PermutationMatrix<6> P;                                                                                                                                        
        P.setIdentity();                                                                                                                                                      
        P.indices() << 3, 4, 5, 0, 1, 2;                                                                                                                                      
        out[pose_id] = P * out[pose_id] * P.transpose();                                                                                                                      
    }                                                                                                                                                                         
                                                                                                                                                                            
    return out;
}                                                                                                                                                                         
                                                        
std::vector<Eigen::Matrix<double, 6, 6>> computeEdgeInformationSum(                                                                                                       
    const PoseGraphSLAM& graph, int num_poses) {                                                                                                                                        
    // YOUR CODE:                                         
    //
    // std::vector<Eigen::Matrix<double,6,6>> H(num_poses, Eigen::Matrix<double,6,6>::Zero());
    //                                                                                                                                                                    
    // for (const auto& e : graph.icp_edges)  { H[e.i] += e.information; H[e.j] += e.information; }
    // for (const auto& e : graph.imu_edges)  { H[e.i] += e.information; H[e.j] += e.information; }                                                                       
    // for (const auto& e : graph.loop_edges) { H[e.i] += e.information; H[e.j] += e.information; }                                                                       
    //                                                                                                                                                                    
    // // GPS is unary 3x3 translation-only — embed as:                                                                                                                   
    // //   H_6 = [ 0_3   0    ]                                                                                                                                          
    // //         [ 0     H_3  ]                                                                                                                                          
    // for (const auto& e : graph.gps_edges) {                                                                                                                            
    //   Eigen::Matrix<double,6,6> H6 = Eigen::Matrix<double,6,6>::Zero();                                                                                                
    //   H6.block<3,3>(3,3) = e.information;   // translation block                                                                                                       
    //   H[e.i] += H6;                                                                                                                                                    
    // }                                                                                                                                                                  
    //                                                                                                                                                                    
    // // Also add pose-0 anchor stiffness so H[0] is non-singular:
    // H[0] += Eigen::Matrix<double,6,6>::Identity() * 1e12;                                                                                                              
    //                                                                                                                                                                    
    // // Invert to get covariance; fall back to huge-sigma on singular.                                                                                                  
    // std::vector<Eigen::Matrix<double,6,6>> P(num_poses);                                                                                                               
    // for (int i = 0; i < num_poses; ++i) {                                                                                                                              
    //   Eigen::FullPivLU<Eigen::Matrix<double,6,6>> lu(H[i]);                                                                                                            
    //   P[i] = lu.isInvertible()                                                                                                                                         
    //          ? H[i].inverse()                          
    //          : Eigen::Matrix<double,6,6>::Identity() * 1e9;                                                                                                            
    // }                                                  
    // return P;                                                                                                                                                          
    //                                                                                                                                                                    
    // Honest ablation footnote for the blog: this heuristic ignores off-diagonal
    // blocks of H — it cannot see that loop closures couple distant poses.    

    std::vector<Eigen::Matrix<double, 6, 6>> H(num_poses, Eigen::Matrix<double, 6, 6>::Zero());
    for (const auto& e: graph.getICPEdges()) {
        H[e.from] += e.information;
        H[e.to] += e.information;
    }
    for (const auto& e: graph.getIMUEdges()) {
        if(e.factor.covariance.block<3,3>(0,0).determinant() < 1e-20 ||                                                                                                          
           e.factor.covariance.block<3,3>(6,6).determinant() < 1e-20) {                                                                                                          
            std::cerr << "WARNING: near-zero IMU covariance at edge "
                        << e.from << "->" << e.to << ", skipping\n";                                                                                                                
            continue;  // skip this edge entirely  
        }
        Eigen::Matrix<double, 9,9> info9 = e.factor.covariance.inverse();
        Eigen::Matrix<double, 6, 6> info6 = Eigen::Matrix<double, 6, 6>::Zero();
        info6.block<3,3>(0,0) = info9.block<3,3>(0,0); //  rotation block
        info6.block<3,3>(3,3) = info9.block<3,3>(6,6); //  translation block
        info6.block<3,3>(0,3) = info9.block<3,3>(0,6); // cross terms (translation-rotation)
        info6.block<3,3>(3,0) = info9.block<3,3>(6,0);
        H[e.from] += info6;
        H[e.to] += info6;
    }
    for (const auto& e: graph.getLoopEdges()) {
        H[e.from] += e.information;
        H[e.to] += e.information;
    }
    for (const auto& e: graph.getGPSEdges()) {
        Eigen::Matrix<double, 6, 6> H6 = Eigen::Matrix<double, 6, 6>::Zero();
        H6.block<3,3>(3,3) = e.information;   // translation block
        H[e.frame_id] += H6;
    }
    H[0] += Eigen::Matrix<double, 6, 6>::Identity() * 1e12;

    std::vector<Eigen::Matrix<double, 6, 6>> P(num_poses);
    for (int i = 0; i < num_poses; ++i) {
        Eigen::FullPivLU<Eigen::Matrix<double, 6, 6>> lu(H[i]);
        if(lu.isInvertible())
            P[i] = lu.inverse();
        else
            P[i] = Eigen::Matrix<double, 6, 6>::Identity() * 1e9;
    }
    return P;                                                                                                                                                    
}                                                                                                                                                                         
                                                        
double poseSigmaFromCovariance(const Eigen::Matrix<double, 6, 6>& P) {                                                                                                    
    // YOUR CODE:                                         
    //   Translation block location depends on your convention:
    //     [rot | trans] → P.block<3,3>(3,3)                                                                                                                              
    //     [trans | rot] → P.block<3,3>(0,0)                                                                                                                              
    //   const auto Pt = P.block<3,3>(3,3);     // pick one and stay with it                                                                                              
    //   double s = std::sqrt(std::max(0.0, Pt.trace()));                                                                                                                 
    //   return std::min(s, 1e3);               // clamp for downstream sanity          
    const auto Pt = P.block<3,3>(3,3);
    double s = std::sqrt(std::max(0.0, Pt.trace()));
    return std::min(s, 1e3);                                                                                                                                            
}                                                                                                                                                                         
                                                        
bool saveCovariancesToCSV(const std::vector<Eigen::Matrix<double, 6, 6>>& covs,                                                                                           
                        const std::string& path) {
    // YOUR CODE:
    //   std::ofstream f(path); if (!f) return false;                                                                                                                     
    //   f << "frame_id";
    //   for (int r = 0; r < 6; ++r)                                                                                                                                      
    //     for (int c = 0; c < 6; ++c)                    
    //       f << ",P" << r << c;                                                                                                                                         
    //   f << "\n";                                                                                                                                                       
    //   for (size_t i = 0; i < covs.size(); ++i) {
    //     f << i;                                                                                                                                                        
    //     for (int r = 0; r < 6; ++r)                    
    //       for (int c = 0; c < 6; ++c)                                                                                                                                  
    //         f << "," << covs[i](r, c);                 
    //     f << "\n";
    //   }
    //   return true;
    std::ofstream f(path);
    if (!f) {
        return false;
    }
    f << "frame_id";
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            f << ",P" << r << c;
        }
    }
    f << "\n";
    for (size_t i = 0; i < covs.size(); ++i) {
        f << i;
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) {
                f << "," << covs[i](r, c);
            }
        }        f << "\n";
    }
    return true;
} 


}  // namespace pose_graph
