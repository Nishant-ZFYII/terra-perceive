// appearance_encoder.hpp
// Hand-crafted-feature MLP for Deep SORT-style appearance embeddings.
//
// Reference:
//   Wojke, Bewley, Paulus, "Simple Online and Realtime Tracking with a Deep
//   Association Metric" (ICIP 2017) §3 (cost matrix combination), §4
//   (appearance descriptor).
//   Hermans, Beyer, Leibe, "In Defense of the Triplet Loss for Person
//   Re-Identification" (arXiv 2017) §4 (batch-hard mining for training).
//
// Why hand-crafted features instead of PointNet:
//   Average DBSCAN cluster on RELLIS has ~30 points. PointNet's max-pool
//   needs O(100s) of points per cluster to learn meaningful per-point
//   features. With ~30 points, the max-pool collapses to a near-constant
//   per-cluster vector. Eight hand-crafted geometric features (bbox dims,
//   density, PCA eigenvalue ratios, height, range) capture the meaningful
//   shape signal at this point density.
//
// Architecture (per the wondrous-crane plan, Decision B):
//
//   input:  8-dim hand-crafted feature vector, z-score normalized
//   layer1: Linear(8 → 64)  → ReLU
//   layer2: Linear(64 → 32) → ReLU
//   layer3: Linear(32 → 32) → identity, then L2-normalize
//   output: 32-dim unit vector — the appearance embedding
//
// Trained in Python (PyTorch) with batch-hard triplet loss (Hermans 2017).
// Trained weights compiled into `include/appearance_model_weights.hpp` as
// `constexpr float` arrays — no ONNX runtime, no libtorch dependency at
// inference time. Round-trip test
// `tests/cpp/test_appearance_encoder.cpp:EncoderMatchesPyTorchReference`
// asserts < 1e-5 max abs diff between PyTorch and Eigen forward passes.

#pragma once
#include <Eigen/Dense>

namespace tracker {

// A 32-dim unit-norm vector. Type alias documents intent: anything that
// expects an Embedding got it from `AppearanceEncoder::encode()` and is
// L2-normalized.
using Embedding = Eigen::Matrix<float, 32, 1>;

// Hand-crafted geometric feature vector. Built per-cluster by
// `clusters_to_detections.py` (extended to also dump features per frame).
// Layout matches what `extract_features.py` writes to `features_NNNNNN.csv`
// — see that file for the exact 8-element ordering.
using FeatureVector = Eigen::Matrix<float, 8, 1>;

class AppearanceEncoder {
   public:
    // Default constructor — weights are loaded from
    // `appearance_model_weights.hpp` (compiled-in `constexpr` arrays).
    // Produced by `python/appearance/train.py` after training, exported via
    // `python/appearance/torch_to_eigen_check.py`.
    AppearanceEncoder();

    // Forward pass: feature vector → 32-dim unit-norm embedding.
    //
    // Pipeline inside encode():
    //   1. Z-score normalize using compiled-in (mean, std) stats.
    //   2. Linear(8→64) + ReLU.
    //   3. Linear(64→32) + ReLU.
    //   4. Linear(32→32) (no activation).
    //   5. L2-normalize.
    //
    // Output is unit-norm by construction, so cosine similarity reduces to
    // a dot product: cos_sim = a.dot(b). The cost-matrix integration in
    // sort_tracker.cpp uses this property to compute `1 - a.dot(b)` as the
    // appearance distance term.
    Embedding encode(const FeatureVector& features) const;

   private:
    // Z-score normalization: (x - feat_mean_) / feat_std_, applied before
    // layer 1. Stats are corpus-wide; computed in train.py and dumped into
    // the weights header alongside the layer matrices.
    Eigen::Matrix<float, 8, 1> feat_mean_;
    Eigen::Matrix<float, 8, 1> feat_std_;

    // Layer 1: 8 → 64
    Eigen::Matrix<float, 64, 8>  W1_;
    Eigen::Matrix<float, 64, 1>  b1_;

    // Layer 2: 64 → 32
    Eigen::Matrix<float, 32, 64> W2_;
    Eigen::Matrix<float, 32, 1>  b2_;

    // Layer 3: 32 → 32 (no activation, then L2-norm in encode())
    Eigen::Matrix<float, 32, 32> W3_;
    Eigen::Matrix<float, 32, 1>  b3_;
};

}  // namespace tracker
