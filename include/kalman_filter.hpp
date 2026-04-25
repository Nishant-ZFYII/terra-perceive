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
namespace tracker {
class KalmanFilter2D {
   public:
    KalmanFilter2D(float dt, float process_noise, float meas_noise);
    void init(float x, float y);
    void predict();
    void update(float z_x, float z_y);
    Eigen::Vector4f state() const;
    Eigen::Vector2f position() const;
    Eigen::Vector2f velocity() const;
    float covariance_trace() const;

   private:
    Eigen::Vector4f x_;
    Eigen::Matrix4f P_;
    Eigen::Matrix4f F_;
    Eigen::Matrix4f Q_;
    Eigen::Matrix<float, 2, 4> H_;
    Eigen::Matrix2f R_;
};
}
