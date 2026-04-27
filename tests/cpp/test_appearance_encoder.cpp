// =============================================================================
// test_appearance_encoder.cpp — hand-crafted MLP appearance encoder (P3-M13)
// =============================================================================
//
// Test plan:
//
//   Structural (run against placeholder weights, day 1)
//     - EncoderProducesUnitVectors
//         For 100 random feature vectors, ‖encode(f)‖ within 1e-5 of 1.0.
//         Pass against placeholder identity weights once Step 5 (L2-norm)
//         in encode() is implemented.
//
//     - IdenticalInputProducesIdenticalEmbedding
//         Same input → same output across two encoder instances. Catches
//         the easy bug where someone accidentally introduces randomness
//         (dropout, batchnorm in training mode, etc.) into the C++
//         forward pass.
//
//   Reference (gated on trained weights — STAYS SKIPPED until train.py runs)
//     - EncoderMatchesPyTorchReference
//         5 (input, expected_embedding) pairs from
//         tests/data/appearance_reference.csv.
//         Max abs diff < 1e-5. Catches: forgot to apply z-score, used
//         wrong activation, transposed a weight matrix, ReLU placed in
//         the wrong layer.
//
//   Semantic (gated on trained weights — STAYS SKIPPED until train.py runs)
//     - EmbeddingDistanceMonotoneOnAugmentation
//         For 100 sampled triplets (cluster A, jittered-A, random-other),
//         the embedding-space distance d(A, jittered-A) is smaller than
//         d(A, random-other) on ≥ 95% of triplets. The actual semantic
//         goal — "the encoder learned to be invariant to small jitter
//         while distinguishing different clusters."
//
// Day-by-day mapping (per let-us-get-the-wondrous-crane.md M13 §):
//   Mon 05-04: skeleton lands; user implements encode() Steps 1-5.
//              EncoderProducesUnitVectors + IdenticalInputProducesIdenticalEmbedding
//              activate against placeholder weights.
//   Fri 05-08: HPC training completes; weights land in
//              appearance_model_weights.hpp (overwriting placeholder).
//              tests/data/appearance_reference.csv lands.
//              EncoderMatchesPyTorchReference + EmbeddingDistanceMonotoneOnAugmentation
//              activate.
//
// DO NOT fill in:
//   - The PyTorch reference data — that gets dumped by
//     `python/appearance/torch_to_eigen_check.py` after training.
//     Don't hand-write the expected embeddings.
//   - Augmentation logic for the semantic test — share a helper with
//     `python/appearance/build_pairs.py:augment_cluster()`.
//
// =============================================================================
//
// Usage:
//   Each test starts with GTEST_SKIP() so the build stays green during the
//   day-by-day fill-in. Activate by deleting GTEST_SKIP() and writing the
//   assertions.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <fstream>
#include <random>

#include "appearance_encoder.hpp"

using tracker::AppearanceEncoder;
using tracker::Embedding;
using tracker::FeatureVector;

namespace {

// Shared encoder per test — construction is cheap (small weights) but the
// test reads cleanly when each test owns its instance.
AppearanceEncoder make_encoder() { return AppearanceEncoder(); }

// Reproducible random feature vector — one helper for all the random-input
// tests so seeding is consistent across the file.
FeatureVector random_features(std::mt19937& rng) {
    std::normal_distribution<float> n(0.0f, 1.0f);
    FeatureVector f;
    for (int i = 0; i < 8; ++i) f(i) = n(rng);
    return f;
}

}  // namespace

// =============================================================================
// Structural — runnable against placeholder weights from day 1
// =============================================================================

TEST(AppearanceEncoder, EncoderProducesUnitVectors) {
    //GTEST_SKIP() << "Day 1 (Mon 05-04): implement encode() Steps 1-5 (especially Step 5: L2-norm).";

    // YOUR CODE:
    //
    // 1) AppearanceEncoder enc = make_encoder();
    //    std::mt19937 rng(42);
    //
    // 2) For 100 random feature vectors, encode and assert unit norm:
    //      for (int i = 0; i < 100; ++i) {
    //          FeatureVector f = random_features(rng);
    //          Embedding e = enc.encode(f);
    //          EXPECT_NEAR(e.norm(), 1.0f, 1e-5f) << "iter " << i;
    //      }
    //
    //    Failure mode if Step 5 (L2-norm) is missing: e.norm() will be
    //    whatever the unnormalized layer-3 output happens to be —
    //    typically 0 < n < 100 with the placeholder weights.

    AppearanceEncoder enc = make_encoder();
    std::mt19937 rng(42);
    for (int i = 0; i < 100; ++i) {
        FeatureVector f = random_features(rng);
        Embedding e = enc.encode(f);
        EXPECT_NEAR(e.norm(), 1.0f, 1e-5f) << "iter " << i;
    }   
}

TEST(AppearanceEncoder, IdenticalInputProducesIdenticalEmbedding) {
    //GTEST_SKIP() << "Day 1 (Mon 05-04): implement encode() forward pass first.";

    // YOUR CODE:
    //
    // 1) AppearanceEncoder a = make_encoder();
    //    AppearanceEncoder b = make_encoder();
    //    std::mt19937 rng(7);
    //    FeatureVector f = random_features(rng);
    //
    // 2) EXPECT_TRUE(a.encode(f).isApprox(b.encode(f), 1e-7f));
    //    EXPECT_TRUE(a.encode(f).isApprox(a.encode(f), 1e-7f));
    //
    //   Catches: any accidental randomness in encode() (dropout left
    //   enabled, batchnorm in training mode, uninitialized scratch
    //   memory). Float exact-match would also work here; isApprox at
    //   1e-7 leaves a tiny window for legitimate float reordering.
    AppearanceEncoder a = make_encoder();
    AppearanceEncoder b = make_encoder();
    std::mt19937 rng(7);
    FeatureVector f = random_features(rng);
    EXPECT_TRUE(a.encode(f).isApprox(b.encode(f), 1e-7f));
    EXPECT_TRUE(a.encode(f).isApprox(a.encode(f), 1e-7f));
}

// =============================================================================
// Reference — gated on trained weights
// =============================================================================

TEST(AppearanceEncoder, EncoderMatchesPyTorchReference) {
    //GTEST_SKIP() << "Gated on Fri 05-08: requires trained weights and tests/data/appearance_reference.csv.";

    // YOUR CODE:
    //
    // 1) Load tests/data/appearance_reference.csv. Format (per row):
    //      f0,f1,f2,f3,f4,f5,f6,f7, e0,e1,...,e31
    //    First 8 columns are the input features; next 32 are the
    //    PyTorch-computed expected embedding.
    //
    // 2) For each row:
    //      FeatureVector f = ...;        // first 8 cols
    //      Embedding expected = ...;     // last 32 cols
    //      Embedding actual = enc.encode(f);
    //      const float max_abs_diff = (actual - expected).cwiseAbs().maxCoeff();
    //      EXPECT_LT(max_abs_diff, 1e-5f)
    //          << "row " << row_idx << "  diff=" << max_abs_diff;
    //
    //    If max_abs_diff > 1e-5 systematically, the most likely culprits
    //    (in decreasing order):
    //      - Z-score skipped (Step 1 missing).
    //      - Wrong activation (tanh/sigmoid where ReLU was expected).
    //      - W matrix transposed (W.transpose() * x instead of W * x).
    //      - L2-norm applied AFTER ReLU instead of after layer 3.

    //load tests/data/appearance_reference.csv
    std::ifstream in("tests/data/appearance_reference.csv");
    ASSERT_TRUE(in.is_open()) << "failed to open tests/data/appearance_reference.csv";

    AppearanceEncoder enc = make_encoder();
    std::string line;
    std::getline(in, line);   // skip header row (f0,f1,...,e31)
    int row_idx = 0;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::vector<float> values;
        for (std::string cell; std::getline(ss, cell, ','); ) {
            values.push_back(std::stof(cell));
        }
        ASSERT_EQ(values.size(), 40u) << "expected 40 columns per row";

        FeatureVector f;
        for (int i = 0; i < 8; ++i) f(i) = values[i];
        Embedding expected;
        for (int i = 0; i < 32; ++i) expected(i) = values[8 + i];

        Embedding actual = enc.encode(f);
        const float max_abs_diff = (actual - expected).cwiseAbs().maxCoeff();
        EXPECT_LT(max_abs_diff, 1e-5f)
            << "row " << row_idx << "  diff=" << max_abs_diff;
        ++row_idx;
    }
    EXPECT_GT(row_idx, 0) << "no data rows in appearance_reference.csv";





}

// =============================================================================
// Semantic — gated on trained weights
// =============================================================================

TEST(AppearanceEncoder, EmbeddingDistanceMonotoneOnAugmentation) {
    //GTEST_SKIP() << "Gated on Fri 05-08: requires trained weights to test the semantic claim.";

    // YOUR CODE:
    //
    // 1) Load tests/data/appearance_triplets.csv. Format (per row):
    //      anchor f0..f7, positive (jittered-anchor) f0..f7, negative f0..f7
    //
    // 2) For each row, encode all three and compute:
    //      const float d_pos = (a - p).norm();
    //      const float d_neg = (a - n).norm();
    //      if (d_pos < d_neg) ++satisfied;
    //
    // 3) After the loop, assert:
    //      EXPECT_GE(static_cast<float>(satisfied) / total, 0.95f)
    //          << satisfied << " / " << total << " triplets satisfied "
    //          << "the d_pos < d_neg ordering";
    //
    //   95% (not 100%) tolerance because the augmentation is geometric
    //   jitter — at the tail of the noise distribution, the augmented
    //   version legitimately moves enough that a near-neighbor cluster
    //   can be closer in embedding space. Triplet loss training targets
    //   the typical case, not the worst case.

    //load tests/data/appearance_triplets.csv  (100 rows)

    std::ifstream in("tests/data/appearance_triplets.csv");
    ASSERT_TRUE(in.is_open()) << "failed to open tests/data/appearance_triplets.csv";

    AppearanceEncoder enc = make_encoder();
    int satisfied = 0;
    int total = 0;
    std::string line;
    std::getline(in, line);   // skip header row (a0,...,n7)
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::vector<float> values;
        for (std::string cell; std::getline(ss, cell, ','); ) {
            values.push_back(std::stof(cell));
        }
        ASSERT_EQ(values.size(), 24u) << "expected 24 columns per row";

        FeatureVector anchor, positive, negative;
        for (int i = 0; i < 8; ++i) {
            anchor(i)   = values[i];
            positive(i) = values[8 + i];
            negative(i) = values[16 + i];
        }

        Embedding a = enc.encode(anchor);
        Embedding p = enc.encode(positive);
        Embedding n = enc.encode(negative);

        const float d_pos = (a - p).norm();
        const float d_neg = (a - n).norm();
        if (d_pos < d_neg) ++satisfied;
        ++total;
    }
    ASSERT_GT(total, 0) << "no triplets in appearance_triplets.csv";
    EXPECT_GE(static_cast<float>(satisfied) / total, 0.95f)
        << satisfied << " / " << total << " triplets satisfied the "
        << "d_pos < d_neg ordering";
}
