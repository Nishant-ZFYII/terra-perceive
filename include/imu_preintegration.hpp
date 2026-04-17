// imu_preintegration.hpp
// IMU pre-integration on the SO(3) manifold — Forster et al. (RSS 2015).
// Summarizes many raw IMU measurements between two LiDAR frames into a single
// relative motion constraint (ΔR, Δv, Δp) with covariance, independent of the
// global pose at the starting frame.
//
// Depends on: so3.hpp (Exp, Log, Jr, hat)
// Used by:    pose_graph_slam.cpp (as IMU edges in the factor graph)
//
// YOUR TASK (P2-M2.2):
//   P2-M2.2.1: Bias struct + static initialization from stationary period
//   P2-M2.2.2: Incremental ΔR/Δv/Δp accumulation (Forster Eq. 26)
//   P2-M2.2.3: Covariance propagation (Forster Eq. 32–35)
//   P2-M2.2.4: CSV I/O for IMU raw data + bias file
//   P2-M2.2.5 (stretch): Bias Jacobians + first-order bias correction (Eq. 36)
//
// Key insight (Forster's contribution): the three quantities
//   ΔR_ij = Π Exp((ω̄ - b^g) Δt)
//   Δv_ij = Σ ΔR_ik · (ā - b^a) · Δt
//   Δp_ij = Σ [Δv_ik · Δt + 0.5 · ΔR_ik · (ā - b^a) · Δt²]
// are INDEPENDENT of the global pose at time i. You compute them ONCE between
// each pair of LiDAR frames, then feed them to the pose graph as a single
// relative motion factor. No re-integration when the optimizer updates poses.
//
// References:
//   Forster et al., "IMU Preintegration on Manifold" (RSS 2015) — Sections IV, V
//   Forster supplementary (covariance derivation)
//
// Numerical gotchas:
//   - Gravity: Forster integrates in WORLD frame with +g on accelerometer
//     (Eq. 21: ā = Rᵀ (w_a - w_g)); double-check sign convention matches your data
//   - dt from timestamps: if two IMU samples arrive at very close times, dt ≈ 0
//     → skip to avoid division by zero and numerical noise
//   - RELLIS-3D IMU is 50 Hz (not 100 Hz as quoted in some docs) — verify with
//     rosbag info before implementing

#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>
#include "so3.hpp"

namespace imu {

// Gravity magnitude in m/s² (WGS84 standard, close enough for most latitudes).
// Sign convention: gravity acts in +z if z points up in the world frame.
const Eigen::Vector3d kGravityWorld(0.0, 0.0, -9.81);

// -----------------------------------------------------------------------------
// Data structures
// -----------------------------------------------------------------------------

// Single raw IMU measurement from the rosbag.
// Timestamp in seconds (double — epoch seconds from ROS header).
// omega: angular velocity in rad/s (body frame, as reported by VN-300)
// accel: linear acceleration in m/s² (body frame, INCLUDES gravity)
struct IMUMeasurement {
    double t;
    Eigen::Vector3d omega;
    Eigen::Vector3d accel;
};

// IMU bias — gyroscope and accelerometer, constant across the trajectory for
// Option B (static initialization). For Option C (graph-state bias), this
// becomes per-keyframe graph state.
struct Bias {
    Eigen::Vector3d gyro  = Eigen::Vector3d::Zero();   // b^g in rad/s
    Eigen::Vector3d accel = Eigen::Vector3d::Zero();   // b^a in m/s²
};

// Pre-integrated IMU factor between two LiDAR keyframes (i, j).
// This is the single relative-motion constraint that goes into the pose graph.
struct IMUFactor {
    int frame_from;
    int frame_to;
    double dt;                             // total elapsed time, seconds
    Eigen::Matrix3d dR;                    // ΔR_ij ∈ SO(3)
    Eigen::Vector3d dv;                    // Δv_ij in m/s (in frame i)
    Eigen::Vector3d dp;                    // Δp_ij in m (in frame i)
    Eigen::Matrix<double, 9, 9> covariance;  // Σ for [δφ, δv, δp] residuals
    // Stretch (Option C): bias Jacobians for first-order correction (Eq. 36)
    // Eigen::Matrix3d dR_dbg;   // ∂ΔR/∂b^g
    // Eigen::Matrix3d dv_dbg;   // ∂Δv/∂b^g
    // Eigen::Matrix3d dv_dba;   // ∂Δv/∂b^a
    // Eigen::Matrix3d dp_dbg;   // ∂Δp/∂b^g
    // Eigen::Matrix3d dp_dba;   // ∂Δp/∂b^a
};

// -----------------------------------------------------------------------------
// Bias estimation (Option B — static initialization)
// -----------------------------------------------------------------------------

// Estimate bias from a stationary segment at the start of the dataset.
// Assumes the first `num_samples` IMU readings are taken while the vehicle
// is stationary with +z pointing up.
//
//   b^g = mean(ω)                    — all rotation rate is bias if stationary
//   b^a = mean(a) - [0, 0, kGravity] — residual after subtracting gravity
//
// Caller's responsibility: confirm stationary (e.g., check ‖ω‖ < threshold
// across all samples). Sanity checks on returned bias magnitude belong here.
Bias estimateBiasFromStatic(const std::vector<IMUMeasurement>& imu_data,
                            int num_samples = 100);

// -----------------------------------------------------------------------------
// Pre-integration (Forster Eq. 26)
// -----------------------------------------------------------------------------

// Accumulates ΔR/Δv/Δp and covariance incrementally as IMU samples are fed in.
// Create one PreIntegrator per keyframe pair, call integrate() for each IMU
// sample in the interval, then extract the final IMUFactor via result().
class PreIntegrator {
   public:
    // Construct with known (constant) bias. Option B passes the static-init
    // bias here. Option C would reset/rebuild the integrator whenever the
    // optimizer changes the bias estimate.
    explicit PreIntegrator(const Bias& bias);

    // Fold one IMU measurement into the accumulator.
    //   omega, accel: raw IMU readings (bias NOT yet subtracted)
    //   dt:           time since previous sample, seconds
    // Updates dR_, dv_, dp_, and Sigma_ in place.
    void integrate(const Eigen::Vector3d& omega,
                   const Eigen::Vector3d& accel,
                   double dt);

    // Reset to identity (dR = I, dv = dp = 0, Sigma = 0). Call between
    // keyframe pairs if reusing a single integrator instance.
    void reset();

    // Snapshot the current accumulated state as an IMUFactor.
    IMUFactor result(int frame_from, int frame_to) const;

   private:
    Bias bias_;
    Eigen::Matrix3d dR_;                      // ΔR_ij
    Eigen::Vector3d dv_;                      // Δv_ij
    Eigen::Vector3d dp_;                      // Δp_ij
    double dt_total_;                         // accumulated elapsed time
    Eigen::Matrix<double, 9, 9> Sigma_;       // covariance (Forster Eq. 35)

    // Per-step noise covariance — gyroscope and accelerometer white noise,
    // from VN-300 datasheet or empirical. Typical values:
    //   σ_gyro  ≈ 1.7e-4 rad/s/√Hz
    //   σ_accel ≈ 2.0e-3 m/s²/√Hz
    Eigen::Matrix<double, 6, 6> Q_noise_;     // block-diag(σ_g² I, σ_a² I)
};

// -----------------------------------------------------------------------------
// Batch helper — build all IMUFactors for a trajectory
// -----------------------------------------------------------------------------

// Given raw IMU samples and LiDAR keyframe timestamps, produce one IMUFactor
// per consecutive keyframe pair [i, i+1]. IMU samples falling in the half-open
// interval [t_i, t_{i+1}) are folded into the corresponding factor.
//
//   imu_data:     raw IMU samples, sorted by timestamp
//   lidar_times:  LiDAR keyframe timestamps (sorted), length N
//   bias:         constant bias (Option B)
// Returns: N-1 IMUFactors.
std::vector<IMUFactor> preintegrateTrajectory(
    const std::vector<IMUMeasurement>& imu_data,
    const std::vector<double>& lidar_times,
    const Bias& bias);

// -----------------------------------------------------------------------------
// CSV I/O
// -----------------------------------------------------------------------------

// Load IMU samples from a CSV written by scripts/extract_imu.py.
// Expected columns: t, wx, wy, wz, ax, ay, az (header row optional)
std::vector<IMUMeasurement> loadIMUFromCSV(const std::string& path);

// Load bias from a small YAML/CSV written by the static-init step.
// Expected format: b_gx, b_gy, b_gz, b_ax, b_ay, b_az on one line.
Bias loadBiasFromCSV(const std::string& path);

// Save IMUFactors to CSV for debugging / handoff to the pose graph script.
// Columns: frame_from, frame_to, dt, dRx, dRy, dRz (as Log(dR)),
//          dvx, dvy, dvz, dpx, dpy, dpz, cov_trace
void saveFactorsToCSV(const std::vector<IMUFactor>& factors,
                      const std::string& path);

}  // namespace imu
