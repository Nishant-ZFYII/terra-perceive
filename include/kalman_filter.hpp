// kalman_filter.hpp
// 2D constant-velocity Kalman filter — implement from scratch.
//
// PDF reference: Part 5.1 (pages 15-16)
// Read: Thrun, "Probabilistic Robotics", Chapter 3 (Gaussian Filters)
// Watch: Cyrill Stachniss, "Bayes Filter and Kalman Filter" (YouTube)
//
// YOUR TASK:
//   1. State vector x = [x, y, vx, vy]^T
//   2. Implement predict(): x_pred = F*x, P_pred = F*P*F^T + Q
//   3. Implement update(): Kalman gain K, state + covariance update
//   4. Use S.llt().solve() instead of S.inverse() for numerical stability
//
// This is the most foundational algorithm. Know it cold for interviews.

#pragma once
#include <Eigen/Dense>

#include "i_filter.hpp"

namespace tracker {

// Phase-3 note: now derives IFilter so SORTTracker's Track can hold a
// std::unique_ptr<IFilter> and swap between CV and IMM at runtime. Existing
// callers that hold a concrete KalmanFilter2D continue to compile unchanged.
class KalmanFilter2D : public IFilter {
   public:
    KalmanFilter2D(float dt, float process_noise, float meas_noise);

    // Second constructor: lets IMM's internal sub-filters override the
    // transition matrix (e.g. CV for one mode, constant-position for the other)
    // without polluting the public API. Q, R, H stay as constructed.
    KalmanFilter2D(float dt,
                   float process_noise,
                   float meas_noise,
                   const Eigen::Matrix4f& F_override);

    void init(float x, float y) override;
    void predict() override;
    void update(float z_x, float z_y) override;

    Eigen::Vector4f state() const override;
    Eigen::Vector2f position() const override;
    Eigen::Vector2f velocity() const override;
    Eigen::Matrix4f covariance() const override;
    float covariance_trace() const override;

    // For a single-model CV filter the "most confident sub-model" is just
    // itself, so the gating cov equals the standard top-left 2x2 block.
    Eigen::Matrix2f gating_position_covariance_2x2() const override {
        return P_.topLeftCorner<2, 2>();
    }

    std::unique_ptr<IFilter> clone() const override {
        return std::make_unique<KalmanFilter2D>(*this);
    }

    // IMM mixing needs to overwrite (x_, P_) with a mixed initial condition
    // before delegating predict()/update() to this sub-filter. Public so
    // IMMFilter can drive it; not intended for general callers.
    void set_state(const Eigen::Vector4f& x, const Eigen::Matrix4f& P);

    // Innovation + innovation covariance from the most recent update(). IMM
    // needs both for its mode-likelihood computation. Valid only after at
    // least one update() since construction; undefined before that.
    Eigen::Vector2f last_innovation() const { return last_y_; }
    Eigen::Matrix2f last_innovation_cov() const { return last_S_; }

   private:
    Eigen::Vector4f x_;
    Eigen::Matrix4f P_;
    Eigen::Matrix4f F_;
    Eigen::Matrix4f Q_;
    Eigen::Matrix<float, 2, 4> H_;
    Eigen::Matrix2f R_;

    // Cached from update() for IMM likelihood. Zero-initialized; meaningful
    // only after first update().
    Eigen::Vector2f last_y_ = Eigen::Vector2f::Zero();
    Eigen::Matrix2f last_S_ = Eigen::Matrix2f::Identity();
};

}  // namespace tracker

