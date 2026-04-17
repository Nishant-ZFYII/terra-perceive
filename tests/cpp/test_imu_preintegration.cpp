// test_imu_preintegration.cpp — Tests for IMU pre-integration.
// Verifies bias estimation, ΔR/Δv/Δp accumulation, and batch trajectory helper.
//
// P2-M2.2 checkpoints:
//   M2.2.1: Bias estimation from stationary data
//   M2.2.2: ΔR/Δv/Δp accumulation under known inputs (Forster Eq. 26)
//   M2.2.3: Covariance propagation grows with integration time
//   M2.2.4: CSV round-trip (save then load, verify match)
//   M2.2.5: Batch trajectory helper produces correct number of factors
//
// References:
//   Forster et al., "IMU Preintegration on Manifold" (RSS 2015), Sections IV–V

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "imu_preintegration.hpp"
#include "so3.hpp"
#include <fstream>
#include <sstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr double kTolStrict = 1e-10;
static constexpr double kTolNumeric = 1e-6;

// Helper: create N stationary IMU samples with known bias + gravity
// omega = b_gyro + small noise (zero here for determinism)
// accel = b_accel + gravity_in_body (assuming body aligned with world)
static std::vector<imu::IMUMeasurement> makeStationarySamples(
    int N, double dt, double t0 = 0.0) {
    // YOUR HELPER:
    //   Stationary vehicle, body aligned with world frame:
    //   - omega = (0.001, -0.002, 0.0005)   ← this IS the bias
    //   - accel = (-0.1, 0.05, -9.81)       ← bias + gravity in body
    //   Return N samples spaced dt apart starting at t0.
    //   These values should match kTestBiasGyro / kTestBiasAccel below.
    std::vector<imu::IMUMeasurement> samples;
    Eigen::Vector3d omega_bias(0.001, -0.002, 0.0005);
    Eigen::Vector3d accel_bias(-0.1, 0.05, 0.0);
    for (int i = 0; i < N; i++) {
        imu::IMUMeasurement meas;
        meas.t = t0 + i * dt;
        meas.omega = omega_bias; // no noise for determinism
        meas.accel = accel_bias + imu::kGravityWorld; // body frame, gravity included
        samples.push_back(meas);
    }
    return samples;
}

// Known test biases (must match what makeStationarySamples produces)
static const Eigen::Vector3d kTestBiasGyro(0.001, -0.002, 0.0005);
static const Eigen::Vector3d kTestBiasAccel(-0.1, 0.05, 0.0);
// With kGravityWorld = (0,0,-9.81), stationary accel reading = bias + gravity = (-0.1, 0.05, -9.81)

// Helper: create N constant-rotation IMU samples (vehicle spinning at known rate)
static std::vector<imu::IMUMeasurement> makeConstantRotationSamples(
    const Eigen::Vector3d& omega_true, int N, double dt, double t0 = 0.0) {
    // YOUR HELPER:
    //   - omega = omega_true (no bias for simplicity, or add kTestBiasGyro)
    //   - accel = kGravityWorld (no linear acceleration, just gravity in body —
    //     approximation valid for small rotations)
    //   Return N samples spaced dt apart starting at t0.
    std::vector<imu::IMUMeasurement> samples;
    for (int i = 0; i < N; i++) {
        imu::IMUMeasurement meas;
        meas.t = t0 + i * dt;
        meas.omega = omega_true; // no bias for simplicity
        meas.accel = imu::kGravityWorld; // body frame, gravity only
        samples.push_back(meas);
    }
    return samples;
}

// Helper: create N constant-acceleration IMU samples (vehicle accelerating linearly)
static std::vector<imu::IMUMeasurement> makeConstantAccelSamples(
    const Eigen::Vector3d& accel_body, int N, double dt, double t0 = 0.0) {
    // YOUR HELPER:
    //   - omega = zero (no rotation)
    //   - accel = accel_body + kGravityWorld (body frame, gravity included)
    //   Return N samples spaced dt apart starting at t0.
    std::vector<imu::IMUMeasurement> samples;
    for (int i = 0; i < N; i++) {
        imu::IMUMeasurement meas;
        meas.t = t0 + i * dt;
        meas.omega = Eigen::Vector3d::Zero(); // no rotation
        meas.accel = accel_body + imu::kGravityWorld; // body frame, gravity included
        samples.push_back(meas);
    }
    return samples;
}

// -----------------------------------------------------------------------------
// M2.2.1 — Bias estimation
// -----------------------------------------------------------------------------

TEST(IMUPreintegration, BiasFromStationaryMatchesKnown) {
    // Estimate bias from stationary samples, compare to known values.
    // YOUR TEST:
    //   - auto samples = makeStationarySamples(200, 0.02)
    //   - auto bias = imu::estimateBiasFromStatic(samples, 200)
    //   - assert (bias.gyro - kTestBiasGyro).norm() < kTolStrict
    //   - assert (bias.accel - kTestBiasAccel).norm() < kTolStrict
    auto samples = makeStationarySamples(200, 0.02);
    auto bias = imu::estimateBiasFromStatic(samples, 200);
    EXPECT_NEAR((bias.gyro - kTestBiasGyro).norm(), 0.0, kTolStrict);
    EXPECT_NEAR((bias.accel - kTestBiasAccel).norm(), 0.0, kTolStrict);
}

TEST(IMUPreintegration, BiasGyroIsSmall) {
    // Sanity: estimated gyro bias magnitude should be < 0.01 rad/s for
    // any reasonable IMU. Catches sign errors or gravity leaking into gyro.
    // YOUR TEST:
    //   - auto samples = makeStationarySamples(100, 0.02)
    //   - auto bias = imu::estimateBiasFromStatic(samples, 100)
    //   - assert bias.gyro.norm() < 0.01

    auto samples = makeStationarySamples(100, 0.02);
    auto bias = imu::estimateBiasFromStatic(samples, 100);
    EXPECT_LT(bias.gyro.norm(), 0.01);
}

// -----------------------------------------------------------------------------
// M2.2.2 — ΔR/Δv/Δp accumulation
// -----------------------------------------------------------------------------

TEST(IMUPreintegration, ZeroInputGivesIdentity) {
    // No rotation, no acceleration (after bias subtraction) → dR = I, dv = 0, dp = 0.
    // YOUR TEST:
    //   - create PreIntegrator with bias = (0,0,0,0,0,0)
    //   - integrate 10 samples: omega = 0, accel = 0, dt = 0.02
    //   - assert dR ≈ Identity
    //   - assert dv ≈ zero
    //   - assert dp ≈ zero
    //   - assert dt ≈ 0.2
    imu::Bias zero_bias;
    zero_bias.gyro.setZero();
    zero_bias.accel.setZero();
    imu::PreIntegrator integrator(zero_bias);
    for (int i = 0; i < 10; i++) {
        integrator.integrate(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.02);
    }
    auto factor = integrator.result(0, 1);
    EXPECT_NEAR((factor.dR - Eigen::Matrix3d::Identity()).norm(), 0.0, kTolStrict);
    EXPECT_NEAR(factor.dv.norm(), 0.0, kTolStrict);
    EXPECT_NEAR(factor.dp.norm(), 0.0, kTolStrict);
    EXPECT_NEAR(factor.dt, 0.2, kTolStrict);
}

TEST(IMUPreintegration, ConstantRotationGivesExpectedDR) {
    // Known constant angular velocity → dR should match Exp(omega * total_time).
    // YOUR TEST:
    //   - omega_true = Vector3d(0.0, 0.0, 0.1)  (slow yaw rotation, 0.1 rad/s)
    //   - bias = zero (simplify)
    //   - integrate 50 samples at dt = 0.02 → total = 1.0 sec
    //   - expected dR = so3::Exp(omega_true * 1.0)
    //   - factor = integrator.result(0, 1)
    //   - assert (factor.dR - expected).norm() < kTolNumeric
    //   Note: small error is expected (discrete vs continuous integration).
    //   If error > 1e-3, something is wrong.
    imu::Bias zero_bias;
    zero_bias.gyro.setZero();
    zero_bias.accel.setZero();
    imu::PreIntegrator integrator(zero_bias);
    Eigen::Vector3d omega_true(0.0, 0.0, 0.1);
    for (int i = 0; i < 50; i++) {
        integrator.integrate(omega_true, Eigen::Vector3d::Zero(), 0.02);
    }
    auto factor = integrator.result(0, 1);
    Eigen::Matrix3d expected_dR = so3::Exp(omega_true * 1.0);
    EXPECT_NEAR((factor.dR - expected_dR).norm(), 0.0, kTolNumeric);
}

TEST(IMUPreintegration, ConstantAccelGivesExpectedDP) {
    // Known constant acceleration → dp should match 0.5 * a * t².
    // YOUR TEST:
    //   - accel_body = Vector3d(1.0, 0.0, 0.0)  (1 m/s² forward)
    //   - omega = zero, bias = zero
    //   - integrate 50 samples: accel = accel_body + kGravityWorld, dt = 0.02
    //     (PreIntegrator doesn't subtract gravity — accel_body IS the raw reading
    //      minus gravity, so feed accel_body directly as the bias-corrected value)
    //   - total_time = 1.0s
    //   - expected dp ≈ (0.5 * 1.0 * 1.0², 0, 0) = (0.5, 0, 0)
    //   - expected dv ≈ (1.0, 0, 0)
    //   - assert (factor.dp - expected_dp).norm() < kTolNumeric
    //   - assert (factor.dv - expected_dv).norm() < kTolNumeric
    //
    //   CAREFUL: PreIntegrator does NOT subtract gravity from accel.
    //   The raw IMU reading is accel_body + gravity_in_body.
    //   But gravity is subtracted from the RESIDUAL later (in the pose graph).
    //   So for this test, pass (accel_body) as the raw measurement and set
    //   bias.accel = 0. Then dp = 0.5*a*t² = (0.5, 0, 0) exactly.
    //   OR: pass (accel_body + kGravityWorld) and set bias.accel = kGravityWorld.
    //   Either approach works — pick one and be consistent.

    imu::Bias zero_bias;
    zero_bias.gyro.setZero();
    zero_bias.accel.setZero(); // accel bias = 0, so feed accel_body directly as raw reading
    imu::PreIntegrator integrator(zero_bias);
    Eigen::Vector3d accel_body(1.0, 0.0, 0.0);
    for (int i = 0; i < 50; i++) {
        integrator.integrate(Eigen::Vector3d::Zero(), accel_body, 0.02);
    }
    auto factor = integrator.result(0, 1);
    Eigen::Vector3d expected_dp(0.5, 0.0, 0.0);
    Eigen::Vector3d expected_dv(1.0, 0.0, 0.0);
    EXPECT_NEAR((factor.dp - expected_dp).norm(), 0.0, kTolNumeric);
    EXPECT_NEAR((factor.dv - expected_dv).norm(), 0.0, kTolNumeric);
}

TEST(IMUPreintegration, DRIsOrthogonal) {
    // After any integration, dR must remain a valid rotation matrix.
    // YOUR TEST:
    //   - integrate some samples (e.g., constant rotation)
    //   - assert (dR * dR.transpose() - I).norm() < kTolStrict
    //   - assert abs(dR.determinant() - 1.0) < kTolStrict
    imu::Bias zero_bias;
    zero_bias.gyro.setZero();
    zero_bias.accel.setZero();
    imu::PreIntegrator integrator(zero_bias);
    Eigen::Vector3d omega_true(0.0, 0.0, 0.1);
    for (int i = 0; i < 50; i++) {
        integrator.integrate(omega_true, Eigen::Vector3d::Zero(), 0.02);
    }
    auto factor = integrator.result(0, 1);
    EXPECT_NEAR((factor.dR * factor.dR.transpose() - Eigen::Matrix3d::Identity()).norm(), 0.0, kTolStrict);
    EXPECT_NEAR(std::abs(factor.dR.determinant() - 1.0), 0.0, kTolStrict);
}

TEST(IMUPreintegration, UpdateOrderMatters) {
    // Verify that dp uses OLD dv, not updated dv.
    // YOUR TEST:
    //   - create PreIntegrator with zero bias
    //   - integrate ONE sample: omega = 0, accel = (1, 0, 0), dt = 1.0
    //   - After one step:
    //       dp should be 0.5 * I * (1,0,0) * 1² = (0.5, 0, 0)
    //       dv should be I * (1,0,0) * 1 = (1, 0, 0)
    //     If dp were updated AFTER dv, dp would wrongly be (1.5, 0, 0)
    //   - assert factor.dp ≈ (0.5, 0, 0)
    //   - assert factor.dv ≈ (1.0, 0, 0)
    imu::Bias zero_bias;
    zero_bias.gyro.setZero();
    zero_bias.accel.setZero();
    imu::PreIntegrator integrator(zero_bias);
    Eigen::Vector3d omega_true(0.0, 0.0, 0.0);
    Eigen::Vector3d accel_true(1.0, 0.0, 0.0);
    integrator.integrate(omega_true, accel_true, 1.0);
    auto factor = integrator.result(0, 1);
    Eigen::Vector3d expected_dp(0.5, 0.0, 0.0);
    Eigen::Vector3d expected_dv(1.0, 0.0, 0.0);
    EXPECT_NEAR((factor.dp - expected_dp).norm(), 0.0, kTolNumeric);
    EXPECT_NEAR((factor.dv - expected_dv).norm(), 0.0, kTolNumeric);

}

// -----------------------------------------------------------------------------
// M2.2.3 — Covariance (placeholder — fill in when propagation is implemented)
// -----------------------------------------------------------------------------

TEST(IMUPreintegration, CovarianceStartsZero) {
    // Before any integration, covariance should be zero.
    // YOUR TEST:
    //   - create PreIntegrator
    //   - factor = result(0, 1)
    //   - assert factor.covariance.norm() < kTolStrict
    imu::Bias zero_bias;
    zero_bias.gyro.setZero();
    zero_bias.accel.setZero();
    imu::PreIntegrator integrator(zero_bias);
    auto factor = integrator.result(0, 1);
    EXPECT_NEAR(factor.covariance.norm(), 0.0, kTolStrict);
}

TEST(IMUPreintegration, CovarianceGrowsWithTime) {
    // (DISABLED until covariance propagation is implemented)
    // After N integration steps, covariance trace should be > 0 and grow
    // monotonically with number of steps.
    // YOUR TEST:
    //   - integrate 10 steps, record trace_10
    //   - integrate 10 more, record trace_20
    //   - assert trace_20 > trace_10 > 0

    imu::Bias zero_bias;
    zero_bias.gyro.setZero();
    zero_bias.accel.setZero();
    imu::PreIntegrator integrator(zero_bias);
    for (int i = 0; i < 10; i++) {
        integrator.integrate(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.02);
    }
    auto factor_10 = integrator.result(0, 1);
    double trace_10 = factor_10.covariance.trace();
    for (int i = 0; i < 10; i++) {
        integrator.integrate(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.02);
    }
    auto factor_20 = integrator.result(0, 1);
    double trace_20 = factor_20.covariance.trace();
    EXPECT_GT(trace_10, 0.0);
    EXPECT_GT(trace_20, trace_10);  
}

// -----------------------------------------------------------------------------
// M2.2.4 — Batch helper
// -----------------------------------------------------------------------------

TEST(IMUPreintegration, BatchProducesCorrectCount) {
    // N keyframe timestamps → N-1 factors.
    // YOUR TEST:
    //   - create 100 IMU samples at 50 Hz (2 sec total)
    //   - create 5 keyframe timestamps at 0.5s intervals: [0, 0.5, 1.0, 1.5, 2.0]
    //   - factors = preintegrateTrajectory(imu_data, lidar_times, bias)
    //   - assert factors.size() == 4
    std::vector<imu::IMUMeasurement> imu_data = makeStationarySamples(100, 0.02);
    std::vector<double> lidar_times = {0.0, 0.5, 1.0, 1.5, 2.0};
    imu::Bias bias;
    bias.gyro.setZero();
    bias.accel.setZero();
    auto factors = imu::preintegrateTrajectory(imu_data, lidar_times, bias);
    EXPECT_EQ(factors.size(), 4);   
}

TEST(IMUPreintegration, BatchFactorDtMatchesInterval) {
    // Each factor's dt should approximately equal the keyframe interval.
    // YOUR TEST:
    //   - same setup as above
    //   - for each factor: assert abs(factor.dt - 0.5) < 0.05
    //     (not exact because IMU samples don't land exactly on keyframe boundaries)

    std::vector<imu::IMUMeasurement> imu_data = makeStationarySamples(100, 0.02);
    std::vector<double> lidar_times = {0.0, 0.5, 1.0, 1.5, 2.0};
    imu::Bias bias;
    bias.gyro.setZero();
    bias.accel.setZero();
    auto factors = imu::preintegrateTrajectory(imu_data, lidar_times, bias);
    for (const auto& factor : factors) {
        EXPECT_NEAR(factor.dt, 0.5, 0.05);
    }

}

// -----------------------------------------------------------------------------
// M2.2.5 — CSV round-trip (optional, but catches serialization bugs)
// -----------------------------------------------------------------------------

TEST(IMUPreintegration, SaveFactorsWritesCorrectLineCount) {                                                                                                              
      // Verify CSV output has header + N data lines.                                                                                                                       
      //   - create 3 synthetic factors via PreIntegrator                                                                                                                   
      //   - saveFactorsToCSV(factors, "/tmp/test_factors.csv")                                                                                                             
      //   - open file, count lines                         
      //   - assert line count == 4 (1 header + 3 data)  
      
    imu::Bias zero_bias;
    zero_bias.gyro.setZero();
    zero_bias.accel.setZero();
    imu::PreIntegrator integrator(zero_bias);       
    for (int i = 0; i < 3; i++) {
        integrator.integrate(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.02);
    }
    auto factors = std::vector<imu::IMUFactor>{integrator.result(0, 1), integrator.result(1, 2), integrator.result(2, 3)};
    imu::saveFactorsToCSV(factors, "/tmp/test_factors.csv");
    std::ifstream file("/tmp/test_factors.csv");
    int line_count = 0;
    std::string line;
    while (std::getline(file, line)) {
        line_count++;
    }
    EXPECT_EQ(line_count, 4);

  }  