// kalman_filter.cpp
// 2D constant-velocity Kalman filter — implementation.
//
// References:
//   Thrun, Burgard, Fox, "Probabilistic Robotics", Ch. 3 (Gaussian Filters)
//   Welch & Bishop, "An Introduction to the Kalman Filter" (TR 95-041, 2006)
//
// State:        x = [x, y, vx, vy]^T           (4 x 1)
// Process:      x_{k+1} = F x_k + w,   w ~ N(0, Q)
// Measurement:  z_k     = H x_k + v,   v ~ N(0, R)
//
// F (constant-velocity):     H (position-only):
//   [1 0 dt  0]                 [1 0 0 0]
//   [0 1  0 dt]                 [0 1 0 0]
//   [0 0  1  0]
//   [0 0  0  1]
//
// Numerical note: for the Kalman-gain computation K = P H^T S^{-1}, do NOT
// call S.inverse() directly. S is symmetric positive-definite by construction
// (H P H^T + R with R > 0), so use a Cholesky solve:
//     K = (S.llt().solve(H * P_pred)).transpose();      // one form
// Calling S.inverse() amplifies floating-point error when S becomes near-
// singular (small Q, confident state) and is one of this milestone's
// load-bearing debugging stories. See tests/cpp/test_kalman.cpp:
//   CholeskyStableWhenInverseBlowsUp.

#include "kalman_filter.hpp"
namespace tracker {

// -----------------------------------------------------------------------------
// Constructor — build the time-invariant matrices F, H, Q, R once.
// Everything that does NOT depend on a measurement goes here.
// -----------------------------------------------------------------------------
KalmanFilter2D::KalmanFilter2D(float dt, float process_noise, float meas_noise) {
    // YOUR CODE: build F_ as the 4x4 constant-velocity transition matrix above.
    // Identity on the diagonal, dt in the two position-velocity coupling slots.
    // Hint: F_.setIdentity(); then F_(0, 2) = dt; F_(1, 3) = dt;

    // YOUR CODE: build H_ as the 2x4 position-observation matrix above.
    // Hint: H_.setZero(); H_(0, 0) = 1.0f; H_(1, 1) = 1.0f;

    // YOUR CODE: build Q_ as process_noise * I_4.
    // Interpretation: isotropic process noise on every state dimension.
    // (Plenty of real systems use a block-structured Q that only injects noise
    // through velocity; keep it simple for now — the Q sweep ablation (B) will
    // vary this scalar.)

    // YOUR CODE: build R_ as meas_noise * I_2.
    // Interpretation: isotropic measurement noise on (x, y).

    // Note: x_ and P_ are left in an undefined state here. init() sets them.
    // If predict()/update() is called before init(), behavior is undefined —
    // that's a deliberate choice mirroring Eigen's "matrices are not
    // zero-initialized" convention. Covered by the InitializationIdentity test.
    F_ = Eigen::Matrix4f::Identity();
    F_(0, 2) = dt;
    F_(1, 3) = dt;
    H_ = Eigen::Matrix<float, 2, 4>::Zero();
    H_(0, 0) = 1.0f;
    H_(1, 1) = 1.0f;
    Q_ = process_noise * Eigen::Matrix4f::Identity();
    R_ = meas_noise * Eigen::Matrix2f::Identity();

}

// Phase-3 IMM constructor: same as above but lets the caller override F.
// Used by IMMFilter to spawn a constant-position sub-filter (F = I_4) while
// keeping H, Q, R in lockstep with the CV sub-filter.
KalmanFilter2D::KalmanFilter2D(float dt,
                               float process_noise,
                               float meas_noise,
                               const Eigen::Matrix4f& F_override)
    : KalmanFilter2D(dt, process_noise, meas_noise) {
    F_ = F_override;
}
// -----------------------------------------------------------------------------
// init — seed the filter with an initial position. Zero velocity prior, large
// initial covariance so the first few updates dominate.
// -----------------------------------------------------------------------------
void KalmanFilter2D::init(float x, float y) {
    // YOUR CODE: set x_ = [x, y, 0, 0]^T.

    // YOUR CODE: set P_ to a large-diagonal matrix. A common choice is
    //   P_ = 1000.0f * Matrix4f::Identity();
    // The large prior means "I have no idea where this thing is or how fast
    // it's moving — please overwrite me from the first few measurements."
    // The CovarianceShrinksMonotonically test relies on this: if P_ starts
    // small, the shrinkage is not monotone during the first couple of updates.
    P_ = 1000.0f * Eigen::Matrix4f::Identity();
    x_ << x, y, 0.0f, 0.0f;
}

// -----------------------------------------------------------------------------
// predict — propagate state forward by one time step using the motion model.
// Uncertainty grows by Q.
// -----------------------------------------------------------------------------
void KalmanFilter2D::predict() {
    // YOUR CODE: x_ = F_ * x_;
    x_ = F_ * x_;

    // YOUR CODE: P_ = F_ * P_ * F_.transpose() + Q_;
    // Symmetry of P_ is preserved in exact arithmetic but can drift in
    // floating-point over thousands of iterations. For M4 we don't need an
    // explicit symmetrize step; if it ever bites, add:
    //   P_ = 0.5f * (P_ + P_.transpose());
    P_ = F_ * P_ * F_.transpose() + Q_;
    P_ = 0.5f * (P_ + P_.transpose());
}

// -----------------------------------------------------------------------------
// update — fuse a new position measurement z = [z_x, z_y]^T into the state.
//
// Walk-through (match variable names to Thrun Ch. 3 / Welch & Bishop):
//   y = z - H x_pred             innovation (2 x 1)
//   S = H P_pred H^T + R         innovation covariance (2 x 2, SPD)
//   K = P_pred H^T S^{-1}        Kalman gain (4 x 2)     <-- use Cholesky
//   x_new = x_pred + K y                                  posterior mean
//   P_new = (I - K H) P_pred                              posterior covariance
// -----------------------------------------------------------------------------
void KalmanFilter2D::update(float z_x, float z_y) {
    // YOUR CODE: build the measurement vector z as an Eigen::Vector2f.
    Eigen::Vector2f z;
    z << z_x, z_y;

    // YOUR CODE: compute the innovation y = z - H_ * x_.
    Eigen::Vector2f y = z - H_ * x_;
    // YOUR CODE: compute S = H_ * P_ * H_.transpose() + R_.
    Eigen::Matrix2f S = H_ * P_ * H_.transpose() + R_;

    // Cache (y, S) for IMM mode-likelihood computation. Cheap and only matters
    // when this KF is being driven as a sub-filter inside IMMFilter.
    last_y_ = y;
    last_S_ = S;

    // YOUR CODE: compute K via Cholesky solve, NOT S.inverse().
    // The cleanest form is:
    //   Eigen::Matrix<float, 4, 2> K = S.llt().solve((H_ * P_).eval()).transpose();
    // Equivalent algebra, but avoids forming S^{-1} explicitly.
    // If the test CholeskyStableWhenInverseBlowsUp passes with this form and
    // fails with the naive S.inverse() form, that's the story working.
    Eigen::Matrix<float, 4, 2> K = S.llt().solve((H_ * P_).eval()).transpose();

    // YOUR CODE: x_ = x_ + K * y;
    x_ = x_ + K * y;

    // YOUR CODE: P_ = (Matrix4f::Identity() - K * H_) * P_;
    // This is the "Joseph form minus one" update. The full Joseph form,
    //   P = (I - K H) P (I - K H)^T + K R K^T,
    // is more numerically stable but heavier; we use the simpler form and
    // rely on symmetrization if drift ever shows up. Mention this in the blog.


    //full joseph form for reference:
    //   P = (I - K H) P (I - K H)^T + K R K^T
    P_ = (Eigen::Matrix4f::Identity() - K * H_) * P_ * (Eigen::Matrix4f::Identity() - K * H_).transpose() + K * R_ * K.transpose();

}

// -----------------------------------------------------------------------------
// Accessors — trivial, provided already. No work to do here.
// -----------------------------------------------------------------------------
Eigen::Vector4f KalmanFilter2D::state() const { return x_; }
Eigen::Vector2f KalmanFilter2D::position() const { return x_.head<2>(); }
Eigen::Vector2f KalmanFilter2D::velocity() const { return x_.tail<2>(); }
Eigen::Matrix4f KalmanFilter2D::covariance() const { return P_; }
float KalmanFilter2D::covariance_trace() const { return P_.trace(); }

void KalmanFilter2D::set_state(const Eigen::Vector4f& x, const Eigen::Matrix4f& P) {
    x_ = x;
    P_ = P;
}

}  // namespace tracker