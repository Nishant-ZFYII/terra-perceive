// imm_filter.hpp
// Interacting Multiple Model (IMM) Kalman filter — 2 modes (CV + CP).
//
// Reference:
//   Bar-Shalom, Li, Kirubarajan, "Estimation with Applications to Tracking and
//   Navigation" (Wiley 2001), §11.6.1 (one-cycle IMM algorithm) and §11.6.6
//   (numerical considerations). 30 pages — read before filling implementation.
//
// Why IMM here:
//   The M4 constant-velocity SORT tracker produced 979 distinct IDs on RELLIS
//   (~50 physical objects). The dominant failure mode was during ego
//   deceleration: the filter predicts trees still drifting at the old apparent
//   velocity; predictions miss the now-stationary cluster centroids by enough
//   that the Hungarian gating drops the match and a new ID is born. A second
//   mode that models a stationary object (constant-position, F = I_4) lets
//   the filter switch to "this thing isn't moving" within a couple of frames
//   and re-acquire the same cluster.
//
// Mode set:
//   1. CV — constant velocity, F as in KalmanFilter2D's default ctor.
//   2. CP — constant position, F = I_4. Velocity drifts via Q only.
//
// Markov transition matrix:
//   Π = [[0.95, 0.05],   row i, col j = P(mode_{k+1} = j | mode_k = i)
//        [0.05, 0.95]]   diagonal-heavy = "modes are sticky" (standard).
//
// Initial mode probabilities:
//   μ_0 = [0.5, 0.5] — uninformative.
//
// State convention:
//   IMMFilter.state()/.position()/.velocity()/.covariance() returns the COMBINED
//   posterior (Σ_j μ_j x_j, Σ_j μ_j (P_j + spread)). This is what SORTTracker
//   uses for matching. Per-mode state is internal.

#pragma once
#include <Eigen/Dense>
#include <array>

#include "i_filter.hpp"
#include "kalman_filter.hpp"

namespace tracker {

class IMMFilter : public IFilter {
   public:
    // dt, process_noise, meas_noise — same semantics as KalmanFilter2D. The
    // CP sub-filter shares meas_noise and the same H; its Q is internally
    // scaled (see imm_filter.cpp). Π and μ_0 default to the values above; the
    // ctor lets tests override them for the degenerate-collapse-to-CV check.
    IMMFilter(float dt,
              float process_noise,
              float meas_noise,
              const Eigen::Matrix2f& transition = default_transition(),
              const Eigen::Vector2f& mu_initial = default_mu_initial());

    void init(float x, float y) override;
    void predict() override;
    void update(float z_x, float z_y) override;

    Eigen::Vector4f state() const override { return x_combined_; }
    Eigen::Vector2f position() const override { return x_combined_.head<2>(); }
    Eigen::Vector2f velocity() const override { return x_combined_.tail<2>(); }
    Eigen::Matrix4f covariance() const override { return P_combined_; }
    float covariance_trace() const override { return P_combined_.trace(); }

    // For Mahalanobis cascade gating, return the MORE CONFIDENT sub-model's
    // 2×2 position covariance — not the combined P. The combined P includes
    // the inter-mode spread term `(x_j - x_combined)(x_j - x_combined)^T`
    // which inflates dramatically during pre-Lost misses when CV and CP
    // disagree on position (CV predicts forward at velocity, CP stays put);
    // a track with v ≈ 5 m/s after 10 misses has σ_pos ≈ 5–7 m in the
    // combined cov, large enough to admit a 22 m world-drift revival as
    // Mahalanobis-acceptable. That's the right answer for estimation
    // (marginal posterior under model uncertainty) but the wrong answer
    // for gating ("is this physically the same object?"). Picking the
    // more confident sub-model gives the gate a tighter, principled shape.
    //
    // "More confident" = lower trace of the 2×2 position block. Ties
    // break to CV (mode 0) — irrelevant in practice.
    Eigen::Matrix2f gating_position_covariance_2x2() const override {
        const auto P_cv = filters_[0].covariance().topLeftCorner<2, 2>();
        const auto P_cp = filters_[1].covariance().topLeftCorner<2, 2>();
        return (P_cp.trace() < P_cv.trace()) ? P_cp : P_cv;
    }

    std::unique_ptr<IFilter> clone() const override {
        return std::make_unique<IMMFilter>(*this);
    }

    // Test + diagnostic accessors — log per-frame to tracks.csv during
    // ablation to detect mode-probability lock-in.
    Eigen::Vector2f mode_probabilities() const { return mu_; }

    // Defaults exposed for tests.
    static Eigen::Matrix2f default_transition() {
        Eigen::Matrix2f Pi;
        Pi << 0.95f, 0.05f,
              0.05f, 0.95f;
        return Pi;
    }
    static Eigen::Vector2f default_mu_initial() {
        return Eigen::Vector2f(0.5f, 0.5f);
    }

   private:
    // Two sub-filters. Index 0 = CV, index 1 = CP. Held by value (not pointer)
    // because they're trivially constructible and we want stack-resident
    // contiguous state — IMM allocates nothing per frame.
    std::array<KalmanFilter2D, 2> filters_;

    // Mode probability vector μ. Always sums to 1 (modulo float epsilon),
    // every entry ∈ [μ_min, μ_max] post-clamp. See update() for the clamp.
    Eigen::Vector2f mu_;

    // Markov transition matrix Π. Π(i, j) = P(mode_{k+1} = j | mode_k = i).
    Eigen::Matrix2f Pi_;

    // Combined posterior — recomputed at the end of every predict()/update().
    // Returned by state()/covariance() so callers see one Gaussian, not two.
    Eigen::Vector4f x_combined_ = Eigen::Vector4f::Zero();
    Eigen::Matrix4f P_combined_ = Eigen::Matrix4f::Identity();

    // μ clamp bounds. Prevents the lock-in failure mode where one mode's
    // likelihood underflows to zero and the filter can never switch back.
    static constexpr float kMuMin = 0.01f;
    static constexpr float kMuMax = 0.99f;
};

}  // namespace tracker
