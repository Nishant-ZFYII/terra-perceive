// test_sort_tracker.cpp — Tests for the SORT multi-object tracker.
//
// P2-M4.3 checkpoints:
//   M4.3.1: SingleObjectPersistentId
//   M4.3.2: SpuriousDetectionSuppressedByMinHits
//   M4.3.3: AppearDisappearPrunedAfterMaxMisses
//   M4.3.4: OccludedTrackResumedBeforeMaxMisses
//   M4.3.5: VelocityEstimationConverges
//   M4.3.6: TwoCrossingTracksNoIdSwap_WithMunkres
//   M4.3.7: TwoCrossingTracksIdSwap_WithGreedy   (asserts the swap HAPPENS,
//                                                 documenting greedy's
//                                                 weakness; this is the
//                                                 ablation, not a failure)
//
// References:
//   Bewley et al., "Simple Online and Realtime Tracking" (ICIP 2016).
//
// Author note: deterministic tests only — fixed seeds, hand-crafted scenes.
// Do not put TEST() blocks inside namespace tracker; gtest macros expand to
// global registrations. The `using` declarations below are sufficient.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <random>
#include <set>
#include <vector>

#include "sort_tracker.hpp"

using tracker::SORTTracker;
using tracker::Solver;
using tracker::Track;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

// Run `tracker.update(...)` for a single detection at position p.
// Wraps the Eigen::Vector2f construction so the test bodies stay readable.
static std::vector<Track> step_one_det(SORTTracker& tracker,
                                       float x, float y,
                                       int class_id = 0) {
    std::vector<Eigen::Vector2f> dets{Eigen::Vector2f(x, y)};
    std::vector<int> cls{class_id};
    return tracker.update(dets, cls);
}

// Step with two detections — used by the crossing-scenario tests.
static std::vector<Track> step_two_dets(SORTTracker& tracker,
                                        Eigen::Vector2f a,
                                        Eigen::Vector2f b) {
    return tracker.update({a, b}, {0, 0});
}

// Step with no detections (occlusion frame).
static std::vector<Track> step_no_dets(SORTTracker& tracker) {
    return tracker.update({}, {});
}

// Look up a Track by id in a vector. Returns nullptr if not found.
static const Track* find_by_id(const std::vector<Track>& v, int id) {
    for (const auto& t : v) if (t.id == id) return &t;
    return nullptr;
}

// =============================================================================
// M4.3.1 — SingleObjectPersistentId
// One detection per frame on a smooth trajectory. After 10 frames, the same
// track id should still be alive.
// =============================================================================
TEST(SORTTracker, SingleObjectPersistentId) {
    // YOUR CODE:
    //   1. SORTTracker tr(/*max_dist=*/2.0f, /*max_misses=*/3, /*min_hits=*/1);
    //   2. Walk a target along (1 + 0.5*i, 2 + 0.3*i) for i in [0, 10).
    //      At each step call step_one_det(tr, x, y) and stash the published
    //      track ids into a std::set.
    //   3. EXPECT_EQ(ids.size(), 1u);    // one and only one id ever appeared
    //   4. EXPECT_EQ(tr.tracks().size(), 1u);  // exactly one live track

    SORTTracker tr(2.0f, 3, 1);
    std::set<int> ids;
    for (int i = 0; i < 10; ++i) {
        float x = 1.0f + 0.5f * i;
        float y = 2.0f + 0.3f * i;
        auto pub = step_one_det(tr, x, y);
        ASSERT_EQ(pub.size(), 1u);   // always published
        ids.insert(pub[0].id);
    }
    EXPECT_EQ(ids.size(), 1u);
    EXPECT_EQ(tr.tracks().size(), 1u);
    const int original_id = tr.tracks()[0].id;
}

// =============================================================================
// M4.3.2 — SpuriousDetectionSuppressedByMinHits
// A single 1-frame false positive must NEVER appear in the publishable
// returns when min_hits >= 2. The flickering detection should die in the
// next frame's prune (because no follow-up detection arrives to confirm it).
// =============================================================================
TEST(SORTTracker, SpuriousDetectionSuppressedByMinHits) {
    // YOUR CODE:
    //   1. SORTTracker tr(/*max_dist=*/2.0f, /*max_misses=*/1,
    //                     /*min_hits=*/3);    // need 3 hits to publish
    //   2. step_one_det(tr, 5.0f, 5.0f);      // spurious frame
    //      Then 5 frames of no detections:
    //          for (int i = 0; i < 5; ++i) {
    //              auto pub = step_no_dets(tr);
    //              EXPECT_TRUE(pub.empty());
    //          }
    //   3. EXPECT_TRUE(tr.tracks().empty());
    //      // Internal state cleaned out too: misses exceeded max_misses.

    SORTTracker tr(2.0f, 1, 3);
    step_one_det(tr, 5.0f, 5.0f);
    for (int i = 0; i < 5; ++i) {
        auto pub = step_no_dets(tr);
        EXPECT_TRUE(pub.empty());
    }
    EXPECT_TRUE(tr.tracks().empty());
}

// =============================================================================
// M4.3.3 — AppearDisappearPrunedAfterMaxMisses
// A track confirmed for several frames, then disappears. It should be pruned
// from internal state exactly once misses > max_misses.
// =============================================================================
TEST(SORTTracker, AppearDisappearPrunedAfterMaxMisses) {
    // YOUR CODE:
    //   1. SORTTracker tr(/*max_dist=*/2.0f, /*max_misses=*/3, /*min_hits=*/1);
    //   2. Walk a track for 5 frames at (i * 0.1f, 0.0f). Capture its id.
    //   3. EXPECT_EQ(tr.tracks().size(), 1u);  // alive
    //   4. Now feed empty detections for max_misses+1 = 4 frames:
    //        for (int i = 0; i < 4; ++i) step_no_dets(tr);
    //   5. EXPECT_TRUE(tr.tracks().empty());   // pruned
    //   6. The next new detection should get a NEW id, not the old one:
    //        auto pub = step_one_det(tr, 100.0f, 100.0f);
    //        // pub may be empty if min_hits>1; check the internal vector.
    //        ASSERT_EQ(tr.tracks().size(), 1u);
    //        EXPECT_NE(tr.tracks()[0].id, original_id);

    SORTTracker tr(2.0f, 3, 1);
    int original_id = -1;
    for (int i = 0; i < 5; ++i) {
        float x = 0.1f * i;
        float y = 0.0f;
        auto pub = step_one_det(tr, x, y);
        ASSERT_EQ(pub.size(), 1u);
        original_id = pub[0].id;
    }
    EXPECT_EQ(tr.tracks().size(), 1u);
    for (int i = 0; i < 4; ++i) step_no_dets(tr);
    EXPECT_TRUE(tr.tracks().empty());
    auto pub = step_one_det(tr, 100.0f, 100.0f);
    ASSERT_EQ(tr.tracks().size(), 1u);
    EXPECT_NE(tr.tracks()[0].id, original_id);
}

// =============================================================================
// M4.3.4 — OccludedTrackResumedBeforeMaxMisses
// 3-frame occlusion gap with max_misses=5 — same id should persist when the
// detection returns. (Different from M4.3.3: occlusion is shorter than the
// pruning threshold.)
// =============================================================================
TEST(SORTTracker, OccludedTrackResumedBeforeMaxMisses) {
    // YOUR CODE:
    //   1. SORTTracker tr(/*max_dist=*/2.0f, /*max_misses=*/5, /*min_hits=*/1);
    //   2. Drive a track for 5 frames at (i*0.5f, 0.0f). Capture its id.
    //   3. Three empty frames (occlusion):
    //        for (int i = 0; i < 3; ++i) step_no_dets(tr);
    //      EXPECT_EQ(tr.tracks().size(), 1u);   // still alive, just missing
    //   4. Resume detections roughly where the predicted position would be.
    //      Track velocity is ~0.5 m/frame; after 8 frames total + 3 occluded
    //      → next predicted x ≈ 0.5 * 8 = 4.0. Provide a detection near there.
    //        auto pub = step_one_det(tr, 4.0f, 0.0f);
    //   5. EXPECT_EQ(tr.tracks().size(), 1u);
    //   6. EXPECT_EQ(tr.tracks()[0].id, original_id);
    //
    // Gotcha: max_dist must be wide enough that the predicted position after
    // 3 frames of pure prediction (no measurement) still gates the resumed
    // detection. With dt=0.1 and Q small, predicted position after 3 misses
    // is well-tracked; max_dist=2.0 is plenty.

    SORTTracker tr(2.0f, 5, 1);
    int original_id = -1;
    for (int i = 0; i < 5; ++i) {
        float x = 0.5f * i;
        float y = 0.0f;
        auto pub = step_one_det(tr, x, y);
        ASSERT_EQ(pub.size(), 1u);
        original_id = pub[0].id;
    }
    EXPECT_EQ(tr.tracks().size(), 1u);
    for (int i = 0; i < 3; ++i) step_no_dets(tr);
    EXPECT_EQ(tr.tracks().size(), 1u);
    auto pub = step_one_det(tr, 4.0f, 0.0f);
    EXPECT_EQ(tr.tracks().size(), 1u);
    EXPECT_EQ(tr.tracks()[0].id, original_id);
}

// =============================================================================
// M4.3.5 — VelocityEstimationConverges
// Drive a target at a known constant velocity. After ~10 frames, the track's
// internal Kalman velocity estimate should match truth to within tolerance.
// =============================================================================
TEST(SORTTracker, VelocityEstimationConverges) {
    // YOUR CODE:
    //   1. SORTTracker tr(/*max_dist=*/2.0f, /*max_misses=*/3, /*min_hits=*/1);
    //   2. Constant velocity (0.5, 0.3); dt = 0.1f (the SORTTracker default).
    //   3. Walk 30 frames; do NOT add measurement noise (deterministic test).
    //   4. ASSERT_EQ(tr.tracks().size(), 1u);
    //      const auto v = tr.tracks()[0].velocity();
    //      EXPECT_NEAR(v.x(), 0.5f, 0.05f);
    //      EXPECT_NEAR(v.y(), 0.3f, 0.05f);
    //
    // Why no noise? This test is about the FILTER's convergence under ideal
    // conditions. The Kalman convergence test in test_kalman.cpp covers the
    // noisy case — duplicating it here with the SORT wrapper adds nothing.

    SORTTracker tr(2.0f, 3, 1);
    for (int i = 0; i < 30; ++i) {
        float x = 0.5f * i * 0.1f;
        float y = 0.3f * i * 0.1f;  // dt=0.1f, so per-frame displacement is velocity * dt  
        auto pub = step_one_det(tr, x, y);
        ASSERT_EQ(pub.size(), 1u);
    }
    ASSERT_EQ(tr.tracks().size(), 1u);
    const auto v = tr.tracks()[0].velocity();
    EXPECT_NEAR(v.x(), 0.5f, 0.05f);
    EXPECT_NEAR(v.y(), 0.3f, 0.05f);
}

// =============================================================================
// M4.3.6 — TwoCrossingTracksNoIdSwap_WithMunkres
// Two targets cross paths at right angles. With Solver::Munkres, both ids
// should persist through the crossing without swapping.
// =============================================================================
TEST(SORTTracker, TwoCrossingTracksNoIdSwap_WithMunkres) {
    // YOUR CODE:
    //   1. SORTTracker tr(/*max_dist=*/3.0f, /*max_misses=*/3, /*min_hits=*/1,
    //                     Solver::Munkres);
    //   2. Define two trajectories that cross at the midpoint:
    //        A: (0, 5) → (10, 5)     moving +x at 1.0 m/frame
    //        B: (5, 0) → (5, 10)     moving +y at 1.0 m/frame
    //      They cross at (5, 5) on frame 5.
    //   3. Capture initial ids on frame 0 (post-confirmation if min_hits>1
    //      — easier to keep min_hits=1 for this test).
    //   4. Walk 10 frames feeding both detections each frame. After each,
    //      record which id is currently at A's position vs B's position.
    //   5. EXPECT that the (id_at_A, id_at_B) mapping stays the SAME for
    //      every frame after the crossing.
    //
    // Hint: id_at_A vs id_at_B is a pair of ints. Build a vector<std::pair>
    // across frames; assert all entries are equal to the pre-cross pair.
    
    SORTTracker tr(3.0f, 3, 1, Solver::Munkres);
    std::vector<std::pair<int, int>> mappings;
    for (int i = 0; i < 10; ++i) {
        Eigen::Vector2f pos_A(0.0f + i * 1.0f, 5.1f);
        Eigen::Vector2f pos_B(5.1f, 0.0f + i * 1.0f);
        auto pub = step_two_dets(tr, pos_A, pos_B);
        ASSERT_EQ(pub.size(), 2u);

        const float d0_to_A = (pub[0].position() - pos_A).norm();
        const float d0_to_B = (pub[1].position() - pos_B).norm();
        const int id_at_A = (d0_to_A < d0_to_B) ? pub[0].id : pub[1].id;
        const int id_at_B = (d0_to_A < d0_to_B) ? pub[1].id : pub[0].id;
        mappings.emplace_back(id_at_A, id_at_B);

    }
    for(size_t i = 1; i < mappings.size(); ++i) {
        EXPECT_EQ(mappings[0], mappings[i]) << "frame " << i << " mapping differs from frame 0 ";
    }
}

// =============================================================================
// M4.3.7 — TwoCrossingTracksIdSwap_WithGreedy   (debugging story #3 —
//                                                INTEGRATION level)
// Same scene as M4.3.6, but Solver::Greedy. The expected behavior is that
// the ids swap mid-crossing because greedy's column-wise nearest-pick lands
// on the wrong target when the two predicted positions cluster within
// max_dist of both detections simultaneously. ASSERT THAT THE SWAP HAPPENS.
// =============================================================================
TEST(SORTTracker, TwoCrossingTracksIdSwap_WithGreedy) {
    // YOUR CODE:
    //   1. Identical setup to M4.3.6 except Solver::Greedy.
    //   2. EXPECT_NE the post-cross (id_at_A, id_at_B) mapping vs the
    //      pre-cross mapping. If they ARE equal, greedy didn't actually
    //      swap on this scene — investigate, either tighten the crossing
    //      or pick more adversarial trajectories.
    //
    // Important framing: this test passing means greedy IS suboptimal in
    // exactly the way the blog claims. If it ever fails, the blog story
    // is wrong, not the code.

    SORTTracker tr(3.0f, 3, 1, Solver::Greedy);
    std::vector<std::pair<int, int>> mappings;
    for (int i = 0; i < 10; ++i) {
        Eigen::Vector2f pos_A(0.0f + i * 1.0f, 5.0f);
        Eigen::Vector2f pos_B(5.0f, 0.0f + i * 1.0f);
        auto pub = step_two_dets(tr, pos_A, pos_B);
        ASSERT_EQ(pub.size(), 2u);

        const float d0_to_A = (pub[0].position() - pos_A).norm();
        const float d1_to_A = (pub[1].position() - pos_A).norm();
        const int id_at_A = (d0_to_A < d1_to_A) ? pub[0].id : pub[1].id;
        const int id_at_B = (pub[0].id == id_at_A) ? pub[1].id : pub[0].id;
        mappings.emplace_back(id_at_A, id_at_B);
    }
    bool any_swap = false;
    for(size_t i = 1; i < mappings.size(); ++i) {
        if (mappings[0] != mappings[i]) {
            any_swap = true;
            break;
        }
    }
    EXPECT_TRUE(any_swap) << "Greedy did not swap ids on the crossing scenario; investigate trajectories and crossing tightness to ensure the test is valid.";

}
