// pose_graph_slam.hpp
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

// Forward declaration — avoids pulling <g2o/core/sparse_optimizer.h> into every
// translation unit that includes this header. The full include is needed only
// in src/pose_graph_slam.cpp where computeMarginalsG2O is defined.
namespace g2o { class SparseOptimizer; }

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

    //evaluvate costs , returns cost over all edges
    double evaluateCost() const;
    Pose& mutablePose(int i) { return poses_[i]; }
    const Pose& pose(int i) const { return poses_[i]; }
    int numPoses() const { return static_cast<int>(poses_.size()); }

    // Edge accessors (for g2o wrapper comparison)
    const std::vector<ICPEdge>& getICPEdges() const { return icp_edges_; }
    const std::vector<IMUEdge>& getIMUEdges() const { return imu_edges_; }
    const std::vector<GPSEdge>& getGPSEdges() const { return gps_edges_; }
    const std::vector<LoopEdge>& getLoopEdges() const { return loop_edges_; }

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
    // Retraction (on-manifold vs Euclidean)
    // -------------------------------------------------------------------------

    // On-manifold retraction: R ← R * Exp(δφ), t ← t + δt.
    // Euclidean: R ← R + hat(δφ) * R (Euler-like), then SVD-reproject to SO(3).
    //            This is what the paper calls "wrong" — whole point of the ablation.
    void retract(const Eigen::VectorXd& dx);

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


};

// -----------------------------------------------------------------------------
// CSV I/O helpers
// -----------------------------------------------------------------------------

// Load initial poses from KISS-ICP output (data/poses_icp.csv).
// Format: t, tx, ty, tz, qx, qy, qz, qw  (quaternion convention)
std::pair<std::vector<Pose>, std::vector<double>> loadPosesFromCSV(const std::string& path);

// Save optimized trajectory to CSV. Same format as poses_icp.csv for
// downstream comparison scripts (ATE/RPE via scripts/compare_odometry.py).
void savePosesToCSV(const std::vector<Pose>& poses,
                    const std::string& path);

// -----------------------------------------------------------------------------                                                                                          
// Per-pose marginal covariance extraction (added for P2-M3)
// -----------------------------------------------------------------------------                                                                                          
//
// Layout convention (document and stay consistent everywhere):                                                                                                           
//   6x6 block = [ P_rotation_3x3     0         ]                                                                                                                         
//               [     0          P_translation_3x3 ]                                                                                                                     
//   — OR —                                                                                                                                                               
//   g2o SE3Quat uses [trans | rot]; permute on extraction if you keep rot-first.                                                                                         
//                                                                                                                                                                        
// Used by P2-M3 world-grid confidence scaling:                                                                                                                           
//   pose_sigma = sqrt(trace(P_translation))                                                                                                                              
//   confidence_adjusted = confidence_raw * exp(-k * pose_sigma)                                                                                                          
// -----------------------------------------------------------------------------                                                                                          
                                                                                                                                                                        
// Production path — true marginal via g2o::SparseOptimizer::computeMarginals                                                                                             
// (Schur-complement inverse of the block-diagonal of H^-1).
// Requires: graph has been optimized; optimizer still holds the factorization.                                                                                           
std::vector<Eigen::Matrix<double, 6, 6>> computeMarginalsG2O(                                                                                                             
    g2o::SparseOptimizer& optimizer, int num_poses);                                                                                                                                     
                                                                                                                                                                        
// Heuristic path — sum of incident-edge information matrices, then invert.                                                                                               
// Biased: ignores correlations induced by non-adjacent (loop-closure) edges.                                                                                             
// Ablation-only baseline against computeMarginalsG2O.                                                                                                                    
std::vector<Eigen::Matrix<double, 6, 6>> computeEdgeInformationSum(
    const PoseGraphSLAM& graph, int num_poses);                                                                                                                           
                                                                                                                                                                        
// Utility: pose_sigma = sqrt(trace(P_translation_block)).                                                                                                                
double poseSigmaFromCovariance(const Eigen::Matrix<double, 6, 6>& P);
                                                                                                                                                                        
// Convenience: dump covariances to CSV for accumulator_runner.                                                                                                           
//   Schema: frame_id, P00, P01, ..., P55   (36 doubles per row, row-major)                                                                                               
bool saveCovariancesToCSV(const std::vector<Eigen::Matrix<double, 6, 6>>& covs,                                                                                           
                        const std::string& path);    
}  // namespace pose_graph
