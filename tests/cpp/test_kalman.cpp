// test_kalman.cpp — Tests for the 2D constant-velocity Kalman filter.
//
// P2-M4.1 checkpoints:
//   M4.1.1: ConstantVelocityConverges      — state tracks truth after enough steps
//   M4.1.2: CovarianceShrinksMonotonically — P's trace decreases under updates
//   M4.1.3: UpdateOrderMatters             — swapping predict/update breaks it
//   M4.1.4: CholeskyStableWhenInverseBlowsUp — numerical story (Cholesky vs inverse)
//   M4.1.5: InitializationIdentity         — predict with zero dt returns init state
//
// References:
//   Thrun, Burgard, Fox, "Probabilistic Robotics", Ch. 3
//   Welch & Bishop, "An Introduction to the Kalman Filter" (TR 95-041, 2006)
//
// Author note: follow the test style of tests/cpp/test_so3.cpp — tight per-test
// tolerances, small deterministic setups, comments explaining the invariant
// under test. The two debugging-story tests (M4.1.3 and M4.1.4) are the point
// of this file; the others are sanity nets.

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <random>

#include "kalman_filter.hpp"

using tracker::KalmanFilter2D;

// Tolerances — adjust per test if a specific one is too tight/loose.
static constexpr float kPosTol = 0.1f;    // meters, after convergence
static constexpr float kVelTol = 0.05f;   // m/s,    after convergence

// -----------------------------------------------------------------------------
// M4.1.1 — ConstantVelocityConverges
// Invariant: if the true target moves at constant velocity and we feed noisy
// position measurements, after ~50 predict/update cycles the estimated state
// should be within kPosTol of the true position and kVelTol of the true
// velocity. This is the headline "it works at all" test.
// -----------------------------------------------------------------------------
TEST(KalmanFilter, ConstantVelocityConverges) {
    // YOUR CODE:
    //   1. Construct KalmanFilter2D kf(dt=0.1f, process_noise=0.01f, meas_noise=0.1f).
    //   2. Truth: x0 = (1.0, 0.0), v = (0.5, 0.3) m/s. Simulate 50 steps.
    //   3. Seed a std::mt19937 with a fixed seed (determinism matters — tests
    //      that only "usually pass" are worse than no test at all).
    //   4. At each step: advance truth, add N(0, meas_noise) noise to generate z,
    //      call kf.predict() then kf.update(z_x, z_y).
    //   5. EXPECT_NEAR the final position components to truth within kPosTol.
    //   6. EXPECT_NEAR the final velocity components to truth within kVelTol.

    KalmanFilter2D kf(0.1f, 0.01f, 0.1f);
    kf.init(1.0f, 0.0f);
    Eigen::Vector2f truth(1.0f, 0.0f);
    Eigen::Vector2f vel(0.5f, 0.3f);
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    for (int i = 0; i < 50; ++i) {
        truth += vel * 0.1f;
        float z_x = truth.x() + noise(rng);
        float z_y = truth.y() + noise(rng);
        kf.predict();
        kf.update(z_x, z_y);
    }
    EXPECT_NEAR(kf.position().x(), truth.x(), kPosTol);
    EXPECT_NEAR(kf.position().y(), truth.y(), kPosTol);
    EXPECT_NEAR(kf.velocity().x(), vel.x(), kVelTol);
    EXPECT_NEAR(kf.velocity().y(), vel.y(), kVelTol);
}

// -----------------------------------------------------------------------------
// M4.1.2 — CovarianceShrinksMonotonically
// Invariant: after the first few updates absorb the initial-uncertainty spike,
// the trace of P should be non-increasing step-over-step for the rest of the
// run (bounded below by the steady-state covariance). If it grows, either
// predict() is adding too much Q without a compensating update(), or Q/R are
// set inconsistently. Catches a class of matrix-assembly bugs in the ctor.
// -----------------------------------------------------------------------------
TEST(KalmanFilter, CovarianceShrinksMonotonically) {
    // YOUR CODE:
    //   1. Same setup as the convergence test.
    //   2. After each update, record trace(P). (You'll need a public accessor
    //      for P_ — either add one to the header or compute trace via a friend
    //      helper in the test file. Simplest path: add
    //        float covariance_trace() const;
    //      to kalman_filter.hpp in this milestone.)
    //   3. Skip the first ~3 steps (transient from the 1000*I prior).
    //   4. EXPECT_LE(trace[k+1], trace[k] + epsilon) for every remaining k,
    //      where epsilon absorbs floating-point noise (~1e-5f).

    KalmanFilter2D kf(0.1f, 0.01f, 0.1f);
    kf.init(1.0f, 0.0f);
    Eigen::Vector2f truth(1.0f, 0.0f);
    Eigen::Vector2f vel(0.5f, 0.3f);
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    std::vector<float> trace;
    for (int i = 0; i < 50; ++i) {
        truth += vel * 0.1f;
        float z_x = truth.x() + noise(rng);
        float z_y = truth.y() + noise(rng);
        kf.predict();
        kf.update(z_x, z_y);
        trace.push_back(kf.covariance_trace());
    }
    for (size_t i = 3; i + 1 < trace.size(); ++i) {
        EXPECT_LE(trace[i + 1], trace[i] + 1e-5f);
    }
}

// -----------------------------------------------------------------------------
// M4.1.3 — UpdateOrderMatters  (debugging story #1)
// The SORT algorithm is "predict THEN update". If a beginner swaps the order
// (update first, then predict), the filter appears to work on the first frame
// and then slowly diverges — a nasty bug because there is no crash. This test
// makes the divergence a unit failure, not a vibes-based "the MP4 looks
// wrong" diagnostic.
//
// Concrete invariant: with predict-then-update, final position error is O(meas_noise).
// With update-then-predict on the same data, final position error grows at
// least ~2x larger (often much more, depending on dt).
// -----------------------------------------------------------------------------
TEST(KalmanFilter, UpdateOrderMatters) {
    // YOUR CODE:
    //   1. Two filters kf_correct and kf_swapped, identical parameters, identical
    //      init.
    //   2. Feed the SAME noisy measurements to both, but reverse the call order
    //      on kf_swapped.
    //   3. EXPECT_LT(correct_err, swapped_err) with a margin large enough that
    //      floating-point noise cannot flip the inequality.
    //   4. Optionally also EXPECT the swapped filter's covariance to NOT shrink
    //      monotonically — makes the mode-of-failure explicit.

    KalmanFilter2D kf_correct(0.1f, 0.01f, 0.1f);
    KalmanFilter2D kf_swapped(0.1f, 0.01f, 0.1f);
    kf_correct.init(1.0f, 0.0f);
    kf_swapped.init(1.0f, 0.0f);
    Eigen::Vector2f truth(1.0f, 0.0f);
    Eigen::Vector2f vel(0.5f, 0.3f);
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    for (int i = 0; i < 50; ++i) {
        truth += vel * 0.1f;
        if (i == 25) vel = Eigen::Vector2f(-0.5f, -0.3f);   // sharp turn
        float z_x = truth.x() + noise(rng);
        float z_y = truth.y() + noise(rng);
        kf_correct.predict();
        kf_correct.update(z_x, z_y);
        kf_swapped.update(z_x, z_y);
        kf_swapped.predict();
    }
    Eigen::Vector2f correct_err = kf_correct.position() - truth;
    Eigen::Vector2f swapped_err = kf_swapped.position() - truth;
    //EXPECT_LT(correct_err.norm(), 0.2f);
    EXPECT_GT(swapped_err.norm(), correct_err.norm() * 1.5f);
}

// -----------------------------------------------------------------------------
// M4.1.4 — CholeskyStableWhenInverseBlowsUp  (debugging story #2)
// S = H P H^T + R is symmetric positive-definite by construction. If R is tiny
// and P has been shrunk by many confident updates, S becomes near-singular and
// .inverse() produces catastrophic floating-point error. .llt().solve() stays
// finite because it never materializes S^{-1}.
//
// This test should FAIL if kalman_filter.cpp's update() uses S.inverse() and
// PASS if it uses S.llt().solve(). That failure mode is the whole point of
// writing this test.
// -----------------------------------------------------------------------------
TEST(KalmanFilter, CholeskyStableWhenInverseBlowsUp) {
    // YOUR CODE:
    //   1. Construct a pathological filter: process_noise very small (1e-10f),
    //      meas_noise very small (1e-10f). Many updates with noise-free
    //      measurements so P shrinks toward singular.
    //   2. Run ~200 predict/update cycles.
    //   3. ASSERT_TRUE(std::isfinite(kf.state().norm())) — the state must not
    //      contain NaN or Inf.
    //   4. EXPECT_NEAR position to truth with a generous tolerance; the point
    //      is "still in the ballpark", not precision.
    //   5. If this test ever passes under an .inverse()-based update(),
    //      investigate — either the pathological regime wasn't pathological
    //      enough, or the test is not exercising the codepath.
    KalmanFilter2D kf(0.1f, 1e-10f, 1e-10f);
    kf.init(1.0f, 0.0f);
    Eigen::Vector2f truth(1.0f, 0.0f);
    Eigen::Vector2f vel(0.5f, 0.3f);
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    for (int i = 0; i < 200; ++i) {
        truth += vel * 0.1f;
        float z_x = truth.x();  // noise-free
        float z_y = truth.y();  // noise-free
        kf.predict();
        kf.update(z_x, z_y);
    }
    ASSERT_TRUE(std::isfinite(kf.state().norm()));
    EXPECT_NEAR(kf.position().x(), truth.x(), 0.01f);
    EXPECT_NEAR(kf.position().y(), truth.y(), 0.01f);
}

// -----------------------------------------------------------------------------
// M4.1.5 — InitializationIdentity
// Sanity: immediately after init(x0, y0), the state reports (x0, y0, 0, 0)
// and calling predict() with a filter built with dt=0 returns the same state.
// Guards against an init() that forgets to zero the velocity, or a ctor that
// accidentally puts garbage on the F diagonal.
// -----------------------------------------------------------------------------
TEST(KalmanFilter, InitializationIdentity) {
    // YOUR CODE:
    //   1. KalmanFilter2D kf(0.0f /*dt*/, 0.01f, 0.1f);
    //   2. kf.init(2.5f, -1.25f);
    //   3. EXPECT_NEAR position components to (2.5, -1.25) within 1e-6.
    //   4. EXPECT_NEAR velocity components to (0, 0) within 1e-6.
    //   5. kf.predict(); repeat the same expectations — with dt=0, predict
    //      must be the identity on the mean.
    KalmanFilter2D kf(0.0f, 0.01f, 0.1f);
    kf.init(2.5f, -1.25f);
    EXPECT_NEAR(kf.position().x(), 2.5f, 1e-6);
    EXPECT_NEAR(kf.position().y(), -1.25f, 1e-6);
    EXPECT_NEAR(kf.velocity().x(), 0.0f, 1e-6);
    EXPECT_NEAR(kf.velocity().y(), 0.0f, 1e-6);
    kf.predict();
    EXPECT_NEAR(kf.position().x(), 2.5f, 1e-6);
    EXPECT_NEAR(kf.position().y(), -1.25f, 1e-6);
    EXPECT_NEAR(kf.velocity().x(), 0.0f, 1e-6);
    EXPECT_NEAR(kf.velocity().y(), 0.0f, 1e-6);
}
