// kalman_filter.cpp
// TODO: Implement 2D constant-velocity Kalman filter from scratch.
// See include/kalman_filter.hpp for interface and instructions.

#include "kalman_filter.hpp"

KalmanFilter2D::KalmanFilter2D(float dt, float process_noise, float meas_noise) {
    // TODO: initialize F_, H_, Q_, R_, P_
    // F = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1]
    // H = [1 0 0 0; 0 1 0 0]
    // Q = process_noise * I
    // R = meas_noise * I
}

void KalmanFilter2D::init(float x, float y) {
    // TODO: set initial state, zero velocity, large initial covariance
}

void KalmanFilter2D::predict() {
    // TODO: x_pred = F * x, P_pred = F * P * F^T + Q
}

void KalmanFilter2D::update(float z_x, float z_y) {
    // TODO: compute S, K, update x and P
    // Use S.llt().solve() for numerical stability
}

Eigen::Vector4f KalmanFilter2D::state() const { return x_; }
Eigen::Vector2f KalmanFilter2D::position() const { return x_.head<2>(); }
Eigen::Vector2f KalmanFilter2D::velocity() const { return x_.tail<2>(); }
