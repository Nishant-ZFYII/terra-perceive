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
#include <Eigen/Geometry>
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

// =============================================================================
// P3.5 cascade matching — tests
// =============================================================================
//
// CascadeRevivesAfterLongOcclusion
//   max_misses=3, max_age=20. Object visible for 5 frames, then 10
//   occlusion frames (> max_misses, so the track transitions Live→Lost),
//   then re-detected at the same spot. With cascade matching enabled,
//   the SAME track_id should continue (revival via Lost-stage match).
//
// CascadeRespectsMaxAgeBudget
//   Same setup but the gap is > max_age. The Lost track must be erased
//   before the re-detection arrives, so a NEW track_id is assigned —
//   this guards against unbounded ghost-track accumulation.
//
// Both tests use position-only matching (use_appearance=false). With
// appearance off, cascade matching falls back to relaxed-position-gate
// re-association — still correct behavior on a stationary object.

TEST(SORTTrackerCascade, CascadeRevivesAfterLongOcclusion) {
    // max_misses=3 → Live→Lost transition after 4 missed frames.
    // max_age=20 → Lost track stays in the retired pool for 20 more frames.
    SORTTracker tr(/*max_dist=*/3.0f,
                   /*max_misses=*/3,
                   /*min_hits=*/1,
                   Solver::Munkres,
                   tracker::Order::PredictThenUpdate,
                   /*dt=*/0.1f,
                   /*process_noise=*/0.01f,
                   /*meas_noise=*/0.1f,
                   tracker::FilterKind::CV,
                   /*use_appearance=*/false,
                   /*appearance_weight=*/0.0f,
                   /*embedding_alpha=*/0.0f,
                   /*max_age=*/20);

    // 5 frames of detection at (10, 5) → 1 published track with id=0.
    int original_id = -1;
    for (int i = 0; i < 5; ++i) {
        auto pub = step_one_det(tr, 10.0f, 5.0f);
        ASSERT_EQ(pub.size(), 1u);
        if (i == 0) original_id = pub[0].id;
        else EXPECT_EQ(pub[0].id, original_id);
    }

    // 10 occlusion frames — all empty. After max_misses=3 missed frames,
    // the track transitions Live→Lost (lost_pos = (10, 5), lost_age = 0).
    // Subsequent miss frames bump lost_age; cascade keeps the track alive.
    for (int i = 0; i < 10; ++i) {
        auto pub = step_no_dets(tr);
        EXPECT_TRUE(pub.empty()) << "Lost tracks must not be published";
    }

    // Re-detection at the same spot. Cascade stage 2 should match the
    // detection to the Lost track and revive it with the original id.
    auto pub = step_one_det(tr, 10.0f, 5.0f);
    ASSERT_EQ(pub.size(), 1u) << "exactly one track should publish";
    EXPECT_EQ(pub[0].id, original_id)
        << "cascade matching failed to revive the Lost track — "
        << "got new id=" << pub[0].id << " instead of " << original_id;
}

TEST(SORTTrackerCascade, CascadeRespectsMaxAgeBudget) {
    SORTTracker tr(/*max_dist=*/3.0f,
                   /*max_misses=*/3,
                   /*min_hits=*/1,
                   Solver::Munkres,
                   tracker::Order::PredictThenUpdate,
                   /*dt=*/0.1f,
                   /*process_noise=*/0.01f,
                   /*meas_noise=*/0.1f,
                   tracker::FilterKind::CV,
                   /*use_appearance=*/false,
                   /*appearance_weight=*/0.0f,
                   /*embedding_alpha=*/0.0f,
                   /*max_age=*/10);

    // 3 frames of detection → established track at original_id.
    int original_id = -1;
    for (int i = 0; i < 3; ++i) {
        auto pub = step_one_det(tr, 10.0f, 5.0f);
        ASSERT_EQ(pub.size(), 1u);
        if (i == 0) original_id = pub[0].id;
    }

    // Long occlusion: max_misses(3) + max_age(10) + 5 buffer = 18 frames
    // with no detections. The Lost track should be FINAL-ERASED before
    // the re-detection arrives.
    for (int i = 0; i < 18; ++i) {
        auto pub = step_no_dets(tr);
        EXPECT_TRUE(pub.empty());
    }

    // Re-detection at the same spot — but the original track was erased.
    // A new track must spawn with a different id.
    auto pub = step_one_det(tr, 10.0f, 5.0f);
    ASSERT_EQ(pub.size(), 1u);
    EXPECT_NE(pub[0].id, original_id)
        << "cascade ignored max_age budget — track resurrected after "
        << "the retirement window expired";
}

// -----------------------------------------------------------------------------
// Fix B regression guard.
//
// Setup mirrors the RELLIS bug surfaced in docs/m10-debug-log.md "False
// revivals — cascade matching's ego-frame bug": a stationary world object
// gets occluded while the ego translates several meters; once the ego has
// driven away, a different physical cluster happens to land at the SAME
// ego-relative coordinate the original track was last seen at, and the
// pre-Fix-B cascade matched against this stale ego anchor — silently
// resurrecting the track on a totally different physical thing.
//
// Fix B stores the freeze position in world frame and projects it back into
// current-ego before gating, so the cascade can only revive on the actual
// physical cluster the track was previously associated with.
// -----------------------------------------------------------------------------
TEST(SORTTrackerCascade, CascadeRevivalSurvivesEgoMotion) {
    SORTTracker tr(/*max_dist=*/3.0f,
                   /*max_misses=*/3,
                   /*min_hits=*/1,
                   Solver::Munkres,
                   tracker::Order::PredictThenUpdate,
                   /*dt=*/0.1f,
                   /*process_noise=*/0.01f,
                   /*meas_noise=*/0.1f,
                   tracker::FilterKind::CV,
                   /*use_appearance=*/false,
                   /*appearance_weight=*/0.0f,
                   /*embedding_alpha=*/0.0f,
                   /*max_age=*/30);

    auto pose_at_world_x = [](float wx) {
        Eigen::Isometry2f T = Eigen::Isometry2f::Identity();
        T.translation() = Eigen::Vector2f(wx, 0.0f);
        return T;
    };

    // Phase 1 — establish a track for a stationary world tree at world (10, 0).
    // Ego is at world origin, so the tree appears in current-ego at (10, 0).
    int original_id = -1;
    for (int i = 0; i < 5; ++i) {
        std::vector<Eigen::Vector2f> dets{Eigen::Vector2f(10.0f, 0.0f)};
        auto pub = tr.update(dets, {0}, pose_at_world_x(0.0f));
        ASSERT_EQ(pub.size(), 1u);
        if (i == 0) original_id = pub[0].id;
    }

    // Phase 2 — 10 frames of occlusion while ego translates +1 m/frame
    // toward the tree. After miss #4 the track goes Lost. Fix B captures
    // lost_pos_world = T_world_ego_at_freeze * (10, 0)_ego ≈ (14, 0)_world.
    // Pre-Fix-B would have stored ego (10, 0) — which a few frames later
    // means a totally different world location.
    for (int k = 1; k <= 10; ++k) {
        const float wx = static_cast<float>(k);
        auto pub = tr.update({}, {}, pose_at_world_x(wx));
        EXPECT_TRUE(pub.empty());
    }

    // Phase 3 — ego is now at world (10, 0), so the same physical tree
    // (world (10, 0)) appears in CURRENT ego at (0, 0). Add a decoy
    // cluster (a different physical thing) at world (20, 0) — it appears
    // at current-ego (10, 0), exactly where the pre-Fix-B stale ego anchor
    // lived. This is the trap.
    //
    //   Fix B path (now): lost_pos_world (14, 0) → current-ego (4, 0).
    //                     Closest detection is the real tree at (0, 0)
    //                     (4 m away) → revived ID lands on the right thing.
    //   Pre-Fix-B path:   lost_pos (10, 0)_ego_stale → matches the decoy
    //                     at (10, 0) (0 m away) → revival migrates to the
    //                     wrong physical object.
    std::vector<Eigen::Vector2f> dets{
        Eigen::Vector2f(0.0f, 0.0f),    // real tree, current-ego
        Eigen::Vector2f(10.0f, 0.0f),   // decoy cluster, current-ego
    };
    auto pub = tr.update(dets, {0, 0}, pose_at_world_x(10.0f));
    const Track* revived = find_by_id(pub, original_id);
    ASSERT_NE(revived, nullptr) << "original track id was not revived";
    EXPECT_NEAR(revived->position().x(), 0.0f, 1.0f)
        << "revived track latched onto the decoy at the stale ego anchor — "
        << "Fix B (world-frame lost_pos) is not transforming back into "
        << "current ego on cascade match";
    EXPECT_NEAR(revived->position().y(), 0.0f, 1.0f);
}

// Pair to the test above: with T_world_ego left at Identity (the legacy
// path / synthetic-data callers / pre-Fix-B behavior), the same scenario
// silently revives onto the decoy cluster. This documents the bug Fix B
// solves and guards against accidentally regressing the world-frame
// projection back to ego-frame.
TEST(SORTTrackerCascade, CascadeRevivalWithoutEgoMotionAnchorsToDecoy) {
    SORTTracker tr(/*max_dist=*/3.0f,
                   /*max_misses=*/3,
                   /*min_hits=*/1,
                   Solver::Munkres,
                   tracker::Order::PredictThenUpdate,
                   /*dt=*/0.1f,
                   /*process_noise=*/0.01f,
                   /*meas_noise=*/0.1f,
                   tracker::FilterKind::CV,
                   /*use_appearance=*/false,
                   /*appearance_weight=*/0.0f,
                   /*embedding_alpha=*/0.0f,
                   /*max_age=*/30);

    // Same establishment phase as the Fix B test, but with NO ego pose
    // passed (Identity throughout).
    int original_id = -1;
    for (int i = 0; i < 5; ++i) {
        auto pub = step_one_det(tr, 10.0f, 0.0f);
        ASSERT_EQ(pub.size(), 1u);
        if (i == 0) original_id = pub[0].id;
    }
    for (int k = 0; k < 10; ++k) {
        auto pub = step_no_dets(tr);
        EXPECT_TRUE(pub.empty());
    }
    // Decoy at the stale ego anchor (10, 0) plus a tree at (0, 0).
    auto pub = tr.update({Eigen::Vector2f(0.0f, 0.0f),
                          Eigen::Vector2f(10.0f, 0.0f)},
                         {0, 0});
    const Track* revived = find_by_id(pub, original_id);
    ASSERT_NE(revived, nullptr);
    // Without ego-motion compensation the cascade revives onto the decoy
    // — exactly the bug Fix B targets. Asserting this nails the failure
    // mode in place so the next refactor can't quietly hide it.
    EXPECT_NEAR(revived->position().x(), 10.0f, 1.0f)
        << "without Fix B the cascade should anchor to the stale ego "
        << "coordinate; if this assertion is now failing, the cascade "
        << "behavior has changed and the Fix B test above no longer "
        << "exercises a meaningful difference";
}
