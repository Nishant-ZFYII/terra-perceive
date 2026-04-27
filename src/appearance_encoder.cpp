// appearance_encoder.cpp
// Eigen forward pass for the Deep SORT-style appearance encoder.
//
// What this file does:
//   1. Constructor reads compiled-in weights from
//      `include/appearance_model_weights.hpp` into Eigen matrices.
//   2. encode() runs the 3-layer MLP forward pass:
//        z-score → Linear(8→64) → ReLU
//                → Linear(64→32) → ReLU
//                → Linear(32→32) → L2-normalize
//      and returns a 32-dim unit-norm Embedding.
//
// Reference:
//   Wojke et al. (ICIP 2017), "Simple Online and Realtime Tracking with a
//   Deep Association Metric" §3 — cost-matrix combination.
//
// Mentor-mode note for the user (per feedback_mentor_mode.md):
//   The constructor (compiled-in weight loading) is mine — pure glue.
//   The encode() forward pass is YOURS. The layer-by-layer math is a
//   straightforward exercise in Eigen matrix arithmetic; the YOUR CODE
//   blocks below tell you exactly which Eigen calls to make at each step.
//   Don't crib structure from PyTorch or libtorch — write it yourself,
//   then verify against the PyTorch reference via
//   `tests/cpp/test_appearance_encoder.cpp:EncoderMatchesPyTorchReference`
//   once real weights have been trained.

#include "appearance_encoder.hpp"

#include "appearance_model_weights.hpp"

namespace tracker {

// =============================================================================
// Constructor — copy compiled-in weights into Eigen storage.
// =============================================================================
//
// The placeholder weights in appearance_model_weights.hpp let this file
// compile and link before training has run. Once `python/appearance/train.py`
// dumps real weights, the same code paths run unchanged — just with non-
// identity matrices.
//
// Eigen storage convention: Matrix<rows, cols> stored column-major by
// default. Our raw arrays are row-major (`kW1[out][in]`), so we copy
// element-by-element rather than memcpy. The cost is ~2.5k float copies at
// startup per encoder instance — negligible.
AppearanceEncoder::AppearanceEncoder() {
    using namespace appearance_weights;

    // ---- Z-score stats ---------------------------------------------------
    for (int i = 0; i < 8; ++i) {
        feat_mean_(i) = kFeatMean[i];
        feat_std_(i)  = kFeatStd[i];
    }

    // ---- Layer 1 (8 → 64) ------------------------------------------------
    // Weight arrays are flat row-major: kW1[out * cols_in + in].
    for (int r = 0; r < 64; ++r) {
        b1_(r) = kB1[r];
        for (int c = 0; c < 8; ++c) {
            W1_(r, c) = kW1[r * 8 + c];
        }
    }

    // ---- Layer 2 (64 → 32) -----------------------------------------------
    for (int r = 0; r < 32; ++r) {
        b2_(r) = kB2[r];
        for (int c = 0; c < 64; ++c) {
            W2_(r, c) = kW2[r * 64 + c];
        }
    }

    // ---- Layer 3 (32 → 32) -----------------------------------------------
    for (int r = 0; r < 32; ++r) {
        b3_(r) = kB3[r];
        for (int c = 0; c < 32; ++c) {
            W3_(r, c) = kW3[r * 32 + c];
        }
    }
}

// =============================================================================
// encode — forward pass. YOUR CODE.
// =============================================================================
//
// Five steps. Each is one or two Eigen calls. Don't import anything new —
// Eigen::Dense (already pulled in via appearance_encoder.hpp) has every
// operation you need.
//
// Reading guide:
//   - Eigen vector dot product:     `a.dot(b)` (returns float)
//   - Eigen matrix-vector multiply: `M * v` (when shapes line up)
//   - Eigen elementwise max:        `v.cwiseMax(0.0f)` ← this is ReLU
//   - Eigen L2 norm:                `v.norm()` (returns float)
//   - Eigen normalized copy:        `v.normalized()` (returns the unit vector)
//   - Eigen elementwise divide:     `(x - m).cwiseQuotient(s)` ← z-score
//
// Reference test:
//   tests/cpp/test_appearance_encoder.cpp:EncoderMatchesPyTorchReference
//   compares this function's output to a PyTorch reference (5 fixed input
//   vectors → 5 expected embeddings) at <1e-5 tolerance. Skipped until
//   real trained weights land — but writing the code now lets the
//   EncoderProducesUnitVectors test (which uses random inputs against
//   the placeholder weights) run from day 1.
//
// =============================================================================
Embedding AppearanceEncoder::encode(const FeatureVector& features) const {
    // Step 1 — Z-score normalize.
    //
    // YOUR CODE: compute (features - feat_mean_) elementwise-divided by
    // feat_std_. Use cwiseQuotient to avoid Eigen's default matrix-style
    // division (which would expect a square matrix).
    //
    //   FeatureVector x = (features - feat_mean_).cwiseQuotient(feat_std_);
    FeatureVector x = (features - feat_mean_).cwiseQuotient(feat_std_);

    // Step 2 — Layer 1: linear + ReLU.
    //
    // YOUR CODE:
    //   Eigen::Matrix<float, 64, 1> h1 = (W1_ * x + b1_).cwiseMax(0.0f);
    //
    // Note: cwiseMax(0.0f) is ReLU. Standard naming, no surprises.
    Eigen::Matrix<float, 64, 1> h1 = (W1_ * x + b1_).cwiseMax(0.0f);


    // Step 3 — Layer 2: linear + ReLU.
    //
    // YOUR CODE:
    //   Eigen::Matrix<float, 32, 1> h2 = (W2_ * h1 + b2_).cwiseMax(0.0f);
    Eigen::Matrix<float, 32, 1> h2 = (W2_ * h1 + b2_).cwiseMax(0.0f);

    // Step 4 — Layer 3: linear, NO activation.
    //
    // YOUR CODE:
    //   Eigen::Matrix<float, 32, 1> h3 = W3_ * h2 + b3_;
    Eigen::Matrix<float, 32, 1> h3 = W3_ * h2 + b3_;

    // Step 5 — L2 normalize.
    //
    // YOUR CODE:
    //   const float n = h3.norm();
    //   if (n > 1e-12f) h3 /= n;
    //
    // The if-guard handles the placeholder-weights edge case where every
    // input maps to the zero vector. Real trained weights almost never
    // produce h3.norm() == 0, but the guard is cheap insurance against
    // NaN propagation. Once real weights land you can drop the guard if
    // you prefer — the existing tests will tell you whether it's load-
    // bearing.
    const float n = h3.norm();
    if (n > 1e-12f) h3 /= n;



    return Embedding(h3);
}

}  // namespace tracker
