// imm_filter.cpp
// Interacting Multiple Model Kalman filter — implementation skeleton.
//
// Reference: Bar-Shalom et al. (2001), §11.6.1 — one-cycle IMM algorithm.
// Read this section BEFORE filling the YOUR CODE blocks. The variable names
// below match the textbook so you can map equation-by-equation.
//
// One IMM cycle (predict + update for one frame):
//
//   ── Step 1: MIXING (a-priori, before predict) ─────────────────────────────
//   For each pair (i, j):
//     c_j        = Σ_i Π(i, j) · μ_i                      (normalizing const)
//     μ_{i|j}    = (Π(i, j) · μ_i) / c_j                   (mixing weight)
//     x_0j       = Σ_i μ_{i|j} · x_i                       (mixed mean for j)
//     P_0j       = Σ_i μ_{i|j} · (P_i + (x_i - x_0j)(x_i - x_0j)^T)
//
//   ── Step 2: MODE-CONDITIONED PREDICT + UPDATE ─────────────────────────────
//   For each j: filters_[j].set_state(x_0j, P_0j); filters_[j].predict();
//                filters_[j].update(z_x, z_y);
//   After update(), each sub-filter has cached its (y_j, S_j) — pull via
//   last_innovation() / last_innovation_cov().
//
//   ── Step 3: MODE PROBABILITY UPDATE ────────────────────────────────────────
//   Per mode j:
//     log Λ_j = -0.5 · ( log(2π · |S_j|) + y_j^T S_j^{-1} y_j )
//   Numerical stability — work in log space, subtract the max before exp:
//     log_w_j  = log Λ_j + log c_j
//     m        = max_k log_w_k
//     μ_j      = exp(log_w_j - m) / Σ_k exp(log_w_k - m)
//   Clamp μ_j to [kMuMin, kMuMax] and renormalize.
//
//   ── Step 4: COMBINED OUTPUT ────────────────────────────────────────────────
//   x_combined = Σ_j μ_j · x_j
//   P_combined = Σ_j μ_j · ( P_j + (x_j - x_combined)(x_j - x_combined)^T )
//
// The mixing-then-predict-then-update sequence applies once per measurement.
// On a frame with NO measurement (track unmatched, tracker-internal predict
// only), Step 2 reduces to predict() alone and Step 3's likelihood update is
// skipped — handled separately in IMMFilter::predict().

#include "imm_filter.hpp"

#include <algorithm>
#include <cmath>

namespace tracker {

namespace {
constexpr float kTwoPi = 6.28318530717958647692f;
}  // namespace

// -----------------------------------------------------------------------------
// Constructor — build the two sub-filters and seed mode probabilities.
//
// CV sub-filter: default KalmanFilter2D (F has dt couplings).
// CP sub-filter: same KF but F = I_4 (constant position; velocity drifts via
//                Q only). Q for the CP sub-filter is internally scaled by 4×
//                on the velocity sub-block to model "I expect velocity to
//                drift, not be commanded by F" — see Bar-Shalom §11.6.6.
// -----------------------------------------------------------------------------
IMMFilter::IMMFilter(float dt,
                     float process_noise,
                     float meas_noise,
                     const Eigen::Matrix2f& transition,
                     const Eigen::Vector2f& mu_initial)
    : filters_{
          // CV sub-filter — default constructor (F has dt couplings).
          KalmanFilter2D(dt, process_noise, meas_noise),
          // CP sub-filter — F = I_4. Same Q magnitude is fine for day-1
          // scaffolding; tune later if test #5 (ModeSwitchOnDeceleration)
          // demands more aggressive velocity-drift Q on the CP mode.
          KalmanFilter2D(dt, process_noise, meas_noise, Eigen::Matrix4f::Identity()),
      },
      mu_(mu_initial),
      Pi_(transition) {
}

// -----------------------------------------------------------------------------
// init — seed both sub-filters at the same initial position.
// -----------------------------------------------------------------------------
void IMMFilter::init(float x, float y) {
    filters_[0].init(x, y);   // CV: P = 1000·I — uninformed about velocity, will learn it.
    filters_[1].init(x, y);

    // CP MODELS a stationary object: "we are confident velocity is zero."
    // KalmanFilter2D::init() seeds both filters with P_vv = 1000 (uninformed).
    // For CP that's wrong — it makes CP's innovation cov S artificially small
    // (F = I doesn't propagate vel uncertainty into pos), biasing Bayes
    // toward CP from the very first frame. Override CP's velocity sub-block
    // to express "v ≈ 0 ± 0.1 m/s" instead.
    Eigen::Matrix4f P_cp = filters_[1].covariance();
    P_cp(2, 2) = 0.01f;
    P_cp(3, 3) = 0.01f;
    filters_[1].set_state(filters_[1].state(), P_cp);

    x_combined_ = filters_[0].state();
    P_combined_ = filters_[0].covariance();
}

// -----------------------------------------------------------------------------
// predict — measurement-less propagation. No mixing or likelihood update;
// just step both sub-filters forward and recombine.
//
// Why no mixing here: mixing is the a-priori interaction step that depends on
// the previous posterior. With no measurement to update from, mixing followed
// by prediction is mathematically equivalent to predicting each sub-filter
// independently and recombining. Simpler and faster.
// -----------------------------------------------------------------------------
void IMMFilter::predict() {
    // YOUR CODE: call predict() on each sub-filter.
    //   for (auto& f : filters_) f.predict();

    // YOUR CODE: recombine. x_combined_ = Σ_j μ_j · x_j.
    //   x_combined_.setZero();
    //   for (size_t j = 0; j < filters_.size(); ++j) {
    //       x_combined_ += mu_(j) * filters_[j].state();
    //   }

    // YOUR CODE: P_combined_ = Σ_j μ_j · ( P_j + spread_j spread_j^T )
    // where spread_j = filters_[j].state() - x_combined_.
    for (auto& f : filters_) f.predict(); 
    x_combined_.setZero();
    P_combined_.setZero();
    for (size_t j = 0; j < filters_.size(); ++j) {
        x_combined_ += mu_(j) * filters_[j].state();
    }
    for (size_t j = 0; j < filters_.size(); ++j) {
        Eigen::Vector4f spread = filters_[j].state() - x_combined_;
        P_combined_ += mu_(j) * (filters_[j].covariance() + spread * spread.transpose());
    }

    

}

// -----------------------------------------------------------------------------
// update — full IMM cycle for one frame WITH a measurement.
//
// What this function must do, in order:
//
//   STEP 1  MIXING. Blend the previous-step posteriors of both sub-filters
//           into mode-conditioned initial conditions (x_0j, P_0j) — one pair
//           per mode j. This is the "interacting" in Interacting Multiple
//           Model: information leaks between modes BEFORE prediction.
//
//   STEP 2  MODE-CONDITIONED PREDICT + UPDATE. Push (x_0j, P_0j) into the j-th
//           sub-filter via set_state(), then call predict() and update(z) on
//           it. After this, each sub-filter has its (y_j, S_j) cached and
//           ready for the likelihood computation.
//
//   STEP 3  MODE PROBABILITY UPDATE. Compute the per-mode measurement
//           likelihood Λ_j in LOG SPACE (single-precision floats underflow
//           fast — that's the lock-in failure mode), combine with c_j, then
//           normalize via log-sum-exp. Clamp and renormalize.
//
//   STEP 4  COMBINED OUTPUT. Collapse the two posteriors back into one
//           (x_combined_, P_combined_) for SORTTracker to consume.
//
// Dimensions cheat-sheet (so you can sanity-check matrix shapes):
//   mu_           (2,)        mode probs after this call
//   Pi_           (2, 2)      Markov transition
//   c             (2,)        Σ_i Π(i,j) μ_i — STEP 1 normalizing const
//   mu_mix        (2, 2)      μ_{i|j} — row i, col j
//   filters_[j].state()       (4,)
//   filters_[j].covariance()  (4, 4)
//   x_0j, x_combined_         (4,)
//   P_0j, P_combined_         (4, 4)
//   y = last_innovation()     (2,)
//   S = last_innovation_cov() (2, 2)
// -----------------------------------------------------------------------------
void IMMFilter::update(float z_x, float z_y) {
    constexpr int N = 2;  // mode count — keeping local in case we ever extend.

    // =========================================================================
    // STEP 1 — MIXING
    // =========================================================================
    // Equations (for each pair i, j ∈ {0, 1}):
    //   c_j        = Σ_i Π(i, j) · μ_i
    //   μ_{i|j}    = Π(i, j) · μ_i / c_j
    //   x_0j       = Σ_i μ_{i|j} · x_i
    //   P_0j       = Σ_i μ_{i|j} · ( P_i + (x_i - x_0j)(x_i - x_0j)^T )
    //
    // Read carefully: the spread term in P_0j uses the MIXED mean x_0j, not
    // the i-th sub-filter's own mean. That's intentional — the spread captures
    // disagreement between modes about where the object is.

    // 1a. c — column-wise normalizer for the mixing weights.
    Eigen::Vector2f c = Eigen::Vector2f::Zero();
    c = Pi_.transpose() * mu_;
    // YOUR CODE: c = Pi_.transpose() * mu_;
    //   Recall Pi_(i, j) = P(next mode = j | current mode = i), so summing
    //   "from-i-to-j" over i gives the prior probability of being in mode j
    //   at the next step.

    // 1b. mu_mix — element (i, j) is the prob the previous mode was i given
    //     the current mode is j. Written as a 2×2 so column j of mu_mix is
    //     the weight vector for sub-filter j's mixed initial condition.
    Eigen::Matrix2f mu_mix = Eigen::Matrix2f::Zero();
    for (int j = 0; j < N; ++j) {                                                                                                                                                               
        if (c(j) < 1e-12f) {                                                                                                                                                                      
            // Mode j has zero prior. Don't divide; pick the identity mix
            // (sub-filter j keeps its own state). It will receive zero weight                                                                                                                    
            // in the recombine, so the contents don't really matter — but they                                                                                                                   
            // mustn't be NaN.                                                                                                                                                                    
            mu_mix(j, j) = 1.0f;                                                                                                                                                                  
            continue;                                                                                                                                                                             
        }                                                                                                                                                                                       
        for (int i = 0; i < N; ++i) {
            mu_mix(i, j) = Pi_(i, j) * mu_(i) / c(j);
        }                                                                                                                                                                                         
    }

    
    // YOUR CODE: for each (i, j), mu_mix(i, j) = Pi_(i, j) * mu_(i) / c(j);
    //   Loop is fine; vectorize later if you want. Watch the divide-by-c(j) —
    //   if c(j) ever underflows to zero you have bigger problems (μ has
    //   collapsed). Asserting c.minCoeff() > 0 is reasonable.

    // 1c. Mixed initial conditions per mode.
    //     IMPORTANT: compute x_0j FIRST for all j, then P_0j using the freshly
    //     computed x_0j. Do not interleave — P_0j depends on x_0j.
    // Step 1c: two nested loops — first all x_0[j], then all P_0[j] (don't interleave — P_0[j] reads x_0[j])  
    std::array<Eigen::Vector4f, 2> x_0;
    std::array<Eigen::Matrix4f, 2> P_0;
    
    for(int j = 0; j < N; ++j) {
        x_0[j].setZero();
        for (int i = 0; i < N; ++i) {
            x_0[j] += mu_mix(i, j) * filters_[i].state();
        }
    }

    // YOUR CODE: compute x_0[j] for j = 0, 1.
    //   for (int j = 0; j < N; ++j) {
    //       for (int i = 0; i < N; ++i) {
    //           x_0[j] += mu_mix(i, j) * filters_[i].state();
    //       }
    //   }

    // YOUR CODE: compute P_0[j] for j = 0, 1.
    //   for (int j = 0; j < N; ++j) {
    //       for (int i = 0; i < N; ++i) {
    //           Eigen::Vector4f spread = filters_[i].state() - x_0[j];
    //           P_0[j] += mu_mix(i, j) * (filters_[i].covariance()
    //                                      + spread * spread.transpose());
    //       }
    //   }
    for (int j = 0; j < N; ++j) {
        P_0[j].setZero();
        for (int i = 0; i < N; ++i) {
            Eigen::Vector4f spread = filters_[i].state() - x_0[j];
            P_0[j] += mu_mix(i, j) * (filters_[i].covariance() + spread * spread.transpose());
        }
    }

    // 1d. Push the mixed initial conditions into each sub-filter so STEP 2's
    //     predict()/update() runs from the mixed prior, not the post-update
    //     state from the previous frame.
    // YOUR CODE:
    //   for (int j = 0; j < N; ++j) filters_[j].set_state(x_0[j], P_0[j]);

    for (int j = 0; j < N; ++j) filters_[j].set_state(x_0[j], P_0[j]);

    // =========================================================================
    // STEP 2 — MODE-CONDITIONED PREDICT + UPDATE
    // =========================================================================
    // Each sub-filter is now seeded with its mixed prior. Run a standard KF
    // cycle on it. Inside KalmanFilter2D::update we cache (y, S) — Step 3
    // reads them via last_innovation() / last_innovation_cov().

    // YOUR CODE:
    //   for (int j = 0; j < N; ++j) {
    //       filters_[j].predict();
    //       filters_[j].update(z_x, z_y);
    //   }
    for (int j = 0; j < N; ++j) {
        filters_[j].predict();
        filters_[j].update(z_x, z_y);
    }

    // =========================================================================
    // STEP 3 — MODE PROBABILITY UPDATE  (Bayes in log-space)
    // =========================================================================
    // Likelihood that measurement z came from mode j (Gaussian in innovation):
    //
    //   Λ_j = N(y_j; 0, S_j)
    //       = (2π)^{-d/2} |S_j|^{-1/2} exp( -0.5 · y_j^T S_j^{-1} y_j )
    //
    // d = 2 (we measure (x, y)). Working in log space:
    //
    //   log Λ_j = -0.5 · ( d · log(2π) + log|S_j| + y_j^T S_j^{-1} y_j )
    //
    // For d = 2 this collapses to:
    //
    //   log Λ_j = -0.5 · ( log(kTwoPi^2 · |S_j|) + quad_j )
    //
    // where kTwoPi^2 ≈ 39.478. We use log(kTwoPi · |S_j|) below (Bar-Shalom's
    // d=2 form drops one factor of (2π) since it cancels in normalization;
    // either form is fine because Step 3 normalizes via log-sum-exp).
    //
    // Quadratic form: y_j^T S_j^{-1} y_j. Use Cholesky solve, NOT S.inverse().
    // Same reason as KalmanFilter2D::update — S is SPD by construction.
    //
    //   Eigen::Vector2f rhs = S.llt().solve(y);
    //   float quad = y.dot(rhs);

    Eigen::Vector2f log_lambda = Eigen::Vector2f::Zero();
    for (int j = 0; j < N; ++j) {
        const Eigen::Vector2f y_j = filters_[j].last_innovation();
        const Eigen::Matrix2f S_j = filters_[j].last_innovation_cov();
        // YOUR CODE: compute det_S, quad, and log_lambda(j).
        //   const float det_S = S_j.determinant();
        //   const float quad  = y_j.dot(S_j.llt().solve(y_j));
        //   log_lambda(j)     = -0.5f * (std::log(kTwoPi * det_S) + quad);
        // Reference for the silenced-unused-variable pattern only — delete
        // these two lines once you fill the YOUR CODE block above.
        const float det_S = S_j.determinant();
        const float quad = y_j.dot(S_j.llt().solve(y_j));
        log_lambda(j) = -0.5f * (std::log(kTwoPi * det_S) + quad);
    }

    // 3b. Combine likelihood with the prior c (computed in Step 1a):
    //
    //   log w_j = log Λ_j + log c_j
    //
    // Then normalize via log-sum-exp:
    //
    //   m       = max_k log w_k
    //   p_j     = exp(log w_j - m)
    //   μ_j     = p_j / Σ_k p_k
    //
    // Subtracting m before exp prevents overflow and is the standard trick.
    Eigen::Vector2f log_w = Eigen::Vector2f::Zero();
    // YOUR CODE: for j = 0, 1: log_w(j) = log_lambda(j) + std::log(c(j));
    for (int j = 0; j < N; ++j) log_w(j) = log_lambda(j) + std::log(c(j));



    Eigen::Vector2f new_mu = Eigen::Vector2f::Zero();
    // YOUR CODE: log-sum-exp normalize log_w into new_mu.
    //   const float m = log_w.maxCoeff();
    //   for (int j = 0; j < N; ++j) new_mu(j) = std::exp(log_w(j) - m);
    //   new_mu /= new_mu.sum();
    const float m = log_w.maxCoeff();
    for (int j = 0; j < N; ++j) new_mu(j) = std::exp(log_w(j) - m);
    new_mu /= new_mu.sum();

    // 3c. Clamp + renormalize. Prevents the lock-in failure mode where one
    //     mode's prob underflows to zero and the filter can never switch back.
    // YOUR CODE:
    //   for (int j = 0; j < N; ++j) {
    //       new_mu(j) = std::clamp(new_mu(j), kMuMin, kMuMax);
    //   }
    //   new_mu /= new_mu.sum();
    //   mu_ = new_mu;
    for (int j = 0; j < N; ++j) {
        new_mu(j) = std::clamp(new_mu(j), kMuMin, kMuMax);
    }
    new_mu /= new_mu.sum();
    mu_ = new_mu;

    // =========================================================================
    // STEP 4 — COMBINED OUTPUT
    // =========================================================================
    // Same shape as predict()'s recombine block (already filled). Collapse the
    // two posteriors into one Gaussian (x_combined_, P_combined_) using the
    // freshly updated mu_.
    //
    //   x_combined_ = Σ_j μ_j · filters_[j].state()
    //   P_combined_ = Σ_j μ_j · ( P_j + (x_j - x_combined_)(x_j - x_combined_)^T )

    x_combined_.setZero();
    P_combined_.setZero();
    // YOUR CODE: copy the two-loop pattern from predict() above.
    //   for (int j = 0; j < N; ++j) {
    //       x_combined_ += mu_(j) * filters_[j].state();
    //   }
    //   for (int j = 0; j < N; ++j) {
    //       Eigen::Vector4f spread = filters_[j].state() - x_combined_;
    //       P_combined_ += mu_(j) * (filters_[j].covariance()
    //                                + spread * spread.transpose());
    //   }
    for (int j = 0; j < N; ++j) {
        x_combined_ += mu_(j) * filters_[j].state();
    }
    for (int j = 0; j < N; ++j) {
        Eigen::Vector4f spread = filters_[j].state() - x_combined_;
        P_combined_ += mu_(j) * (filters_[j].covariance() + spread * spread.transpose());
    }

    // Silence unused-parameter warnings until the YOUR CODE blocks are filled.
    // Delete these two lines once update() is non-trivial.

}

}  // namespace tracker
