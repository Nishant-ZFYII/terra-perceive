// test_kalman.cpp — Tests for Kalman filter.
//
// YOUR TASK:
//   1. Synthetic constant-velocity target: x(t) = x0 + v*t
//   2. Add measurement noise
//   3. Run predict/update loop for 50 steps
//   4. ASSERT: state converges to true velocity
//   5. ASSERT: covariance shrinks after each update
//   6. Plot predicted vs actual trajectory (Python visualization script)
//
// PDF reference: Part 5.1

#include <gtest/gtest.h>

#include "kalman_filter.hpp"

TEST(KalmanFilter, ConstantVelocity) {
    // TODO: implement
}

TEST(KalmanFilter, CovarianceShrinks) {
    // TODO: implement
}
