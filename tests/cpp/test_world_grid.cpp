// =============================================================================
// test_world_grid.cpp — unit tests for WorldGrid (P2-M3)
// =============================================================================
//
// Test plan (one TEST() per item, ordered from cheapest to most involved):
//
//   Construction & indexing
//     - ConstructorAllocatesCells
//     - WorldToGridRoundTrip
//     - WorldToGridOutOfBounds
//
//   Update rules (ablation B correctness)
//     - EMA_FirstObservationNoBlend
//     - EMA_SteadyStateConverges
//     - LogOdds_HitSaturatesAtClamp
//     - LogOdds_MissSaturatesAtClamp
//     - LogOdds_RiskSyncedToSigmoid
//     - Overwrite_HasNoMemory
//
//   Confidence & decay (ablations D and E)
//     - ConfidenceGrowsMonotonically
//     - ConfidenceScaledByPoseSigma
//     - DecayDisabledByZeroRate
//     - DecayOnlyTouchesConfidenceNotRisk
//
//   Coverage (AABB + hull)
//     - CoverageMonotonicOverFrames
//     - CoverageHullTighterThanAABB        // curvy trajectory — hull denominator smaller
//
//   Snapshot round-trip
//     - SaveLoadRoundTripCellEquality      // proves the column-header skip fix
//     - LoadMismatchedConfigFails          // grid config != snapshot metadata → false
//
// DO NOT fill in:
//   - Helper implementations that generate synthetic CellFeatures (encodes algorithm)
//   - Expected numerical values (compute once by hand, paste into EXPECT_NEAR)
//
// =============================================================================

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "world_grid.hpp"
#include "traversability.hpp"
#include "pose_graph_slam.hpp"

// -----------------------------------------------------------------------------
// Test-local sigmoid helper.
// WorldGrid::logOddsToProb is private — we can't call it from here. Duplicating
// the one-line formula in the test is intentional: tests should bind to the
// MATHEMATICAL invariant (risk == sigmoid(logodds)), not to a specific private
// implementation. If WorldGrid later replaces the sigmoid with a lookup table
// or clamped variant, this helper still expresses the right expectation and
// nothing in the test file needs to change.
// -----------------------------------------------------------------------------
static inline float testSigmoid(float l) {
    return 1.0f / (1.0f + std::exp(-l));
}

// -----------------------------------------------------------------------------
// Test fixture — tiny 10x10 grid, 0.5m resolution. Big enough for transform
// round-trips, small enough to iterate by hand in a debugger.
// -----------------------------------------------------------------------------
class WorldGridTest : public ::testing::Test {
 protected:
    void SetUp() override {
        cfg_.x_min      = -2.5;
        cfg_.x_max      =  2.5;
        cfg_.y_min      = -2.5;
        cfg_.y_max      =  2.5;
        cfg_.resolution =  0.5;   // → 10 × 10 cells
        cfg_.alpha      =  0.3;
        cfg_.update_rule = UpdateRule::EMA;
        cfg_.decay_rate = 0.0;    // disabled by default; individual tests override
    }

    // Helper: identity pose at origin.
    Pose identityPose() const {
        Pose p;
        p.R = Eigen::Matrix3d::Identity();
        p.t = Eigen::Vector3d::Zero();
        return p;
    }

    // Helper: build a single-frame TraversabilityGrid whose cell (ix, iy)
    // contains a dense cluster of ground points with mean elevation ≈ mean_z.
    // The resulting CellFeatures.risk and .confidence are NOT directly
    // controllable — they are derived by TraversabilityGrid::compute() from
    // the synthetic point geometry. This is intentional: we test against the
    // REAL producer, not a mock.
    //
    // If a test needs a specific risk value (e.g. exactly 0.8 for EMA
    // assertions), prefer to:
    //   (a) call updateCellEMA/updateCellLogOdds directly — but they are
    //       PRIVATE. So either...
    //   (b) add a test-only friend declaration to WorldGrid, or
    //   (c) test against whatever risk compute() actually produces — read it
    //       back from the local grid before calling g_.update().
    //
    // Path (c) is cleanest and what this helper enables.
    //
    // YOUR CODE:
    //   TraversabilityGrid makeSingleCellGrid(int ix, int iy, float mean_z) {
    //       GridParams gp;         // defaults
    //       VehicleKinematics vk;  // defaults
    //       TraversabilityGrid tg(gp, vk);
    //
    //       // Center of target cell in LiDAR frame:
    //       const float cx = gp.x_min + (ix + 0.5f) * gp.resolution;
    //       const float cy = gp.y_min + (iy + 0.5f) * gp.resolution;
    //
    //       // Pack, say, 10 points into that cell around (cx, cy, mean_z).
    //       // Scatter them slightly so compute() runs PCA without singularities.
    //       std::vector<Eigen::Vector3f> pts;
    //       for (int i = 0; i < 10; ++i) {
    //           float dx = (i % 5) * 0.01f;  // tiny scatter
    //           float dy = (i / 5) * 0.01f;
    //           pts.emplace_back(cx + dx, cy + dy, mean_z);
    //       }
    //       tg.compute(pts);
    //       return tg;
    //   }

    // If hazardous=false: flat dense points → risk ≈ 0 (safe). Exercises miss branch.
    // If hazardous=true:  large z-spread + steep pseudo-slope → risk ≈ 1 (hazard).
    //                     Exercises hit branch of log-odds updates.
    TraversabilityGrid makeSingleCellGrid(int ix, int iy, float mean_z,
                                          bool hazardous = false) {
        GridParams gp;         // defaults
        VehicleKinematics vk;  // defaults
        TraversabilityGrid tg(gp, vk);

        const float cx = gp.x_min + (ix + 0.5f) * gp.resolution;
        const float cy = gp.y_min + (iy + 0.5f) * gp.resolution;

        std::vector<Eigen::Vector3f> pts;
        if (!hazardous) {
            // Flat ground — tiny horizontal scatter at a single z.
            for (int i = 0; i < 10; ++i) {
                float dx = (i % 5) * 0.01f;
                float dy = (i / 5) * 0.01f;
                pts.emplace_back(cx + dx, cy + dy, mean_z);
            }
        } else {
            // Vertical-wall-like cluster: large z-spread → large step_height
            // AND a near-vertical normal → steep slope. Both push
            // compute_vehicle_aware_score well past vehicle tolerances → risk ≈ 1.
            for (int i = 0; i < 10; ++i) {
                float dx = (i % 5) * 0.01f;
                float dy = (i / 5) * 0.01f;
                float dz = static_cast<float>(i) * 0.5f;   // 0..4.5m vertical spread
                pts.emplace_back(cx + dx, cy + dy, mean_z + dz);
            }
        }
        tg.compute(pts);
        return tg;
    }

    // Helper: given a local (ix, iy) and identity-pose transform, return the
    // world coordinates of that cell's center. Use this to pick the right
    // getCell() argument in assertions.
    Eigen::Vector2d localCellCenterInWorld(int ix, int iy) const {
        GridParams gp;   // defaults from TraversabilityGrid
        const double x = gp.x_min + (ix + 0.5) * gp.resolution;
        const double y = gp.y_min + (iy + 0.5) * gp.resolution;
        return {x, y};
    }

    WorldGridConfig cfg_;
    WorldGrid g_{cfg_};
    



};

// -----------------------------------------------------------------------------
// Construction & indexing
// -----------------------------------------------------------------------------

TEST_F(WorldGridTest, ConstructorAllocatesCells) {
    WorldGrid g(cfg_);
    EXPECT_EQ(g.rows(), 10);
    EXPECT_EQ(g.cols(), 10);
    EXPECT_NEAR(g.resolution(), 0.5, 1e-9);
    // YOUR CODE: spot-check a cell is default-initialized:
    //   const WorldCell* c = g.getCell(0.0, 0.0);
    //   ASSERT_NE(c, nullptr);
    //   EXPECT_EQ(c->obs_count, 0u);
    //   EXPECT_FLOAT_EQ(c->risk, 0.0f);

    const WorldCell* c = g.getCell(0.0, 0.0);
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->obs_count, 0u);
    EXPECT_FLOAT_EQ(c->risk, 0.0f);
}

TEST_F(WorldGridTest, WorldToGridRoundTrip) {
    WorldGrid g(cfg_);
    // Walk every cell center; getCell(world_x, world_y) should return a
    // pointer that is ADDRESS-IDENTICAL to the pointer returned by at(r, c).
    // This verifies worldToGrid + indexing round-trip without needing to call
    // the private worldToGrid directly.
    //
    // Convention note: the class's internal mapping is row ↔ x-axis, col ↔
    // y-axis (see the design-note in the WorldGrid ctor comment). So a cell
    // at row `r`, col `c` has center world-coords
    //   x = x_min + (r + 0.5) * res
    //   y = y_min + (c + 0.5) * res
    for (int r = 0; r < g.rows(); ++r) {
        for (int c = 0; c < g.cols(); ++c) {
            double x = cfg_.x_min + (r + 0.5) * cfg_.resolution;
            double y = cfg_.y_min + (c + 0.5) * cfg_.resolution;
            const WorldCell* cell = g.getCell(x, y);
            ASSERT_NE(cell, nullptr);
            EXPECT_EQ(cell, &g.at(r, c))
                << "round-trip mismatch at (r=" << r << ", c=" << c << ")";
        }
    }
}

TEST_F(WorldGridTest, WorldToGridOutOfBounds) {
    WorldGrid g(cfg_);
    EXPECT_EQ(g.getCell(1e6, 0.0),  nullptr);
    EXPECT_EQ(g.getCell(0.0,  1e6), nullptr);
    EXPECT_EQ(g.getCell(-1e6, 0.0), nullptr);
}

// -----------------------------------------------------------------------------
// Update rules — ablation B correctness
// -----------------------------------------------------------------------------

TEST_F(WorldGridTest, EMA_FirstObservationNoBlend) {
    // Pick a local cell whose world coordinate (under identity pose) falls
    // INSIDE the 10×10 WorldGrid bounds x,y ∈ [-2.5, 2.5]. Default
    // TraversabilityGrid extents are x ∈ [-5, 30], y ∈ [-15, 15] at 0.5m.
    // Local ix=10 → x_local = -5 + 10.5*0.5 = 0.25m. Local iy=30 → y_local = 0.0m.
    // Both inside world bounds. Adjust to taste.
    // YOUR CODE:
    //   cfg_.update_rule = UpdateRule::EMA;
    //   WorldGrid g(cfg_);                         // rebuild with updated cfg
    //   const int ix = 10, iy = 30;
    //
    //   TraversabilityGrid local = makeSingleCellGrid(ix, iy, /*mean_z=*/0.0f);
    //
    //   // Read what risk compute() actually produced — we don't control it.
    //   const float expected_risk = local.at(ix, iy).risk;
    //   ASSERT_GT(local.at(ix, iy).point_count, 0);
    //
    //   g.update(local, identityPose(), /*pose_sigma=*/0.0, /*timestamp=*/0.0);
    //
    //   auto cc = localCellCenterInWorld(ix, iy);
    //   const WorldCell* cell = g.getCell(cc.x(), cc.y());
    //   ASSERT_NE(cell, nullptr);
    //   EXPECT_FLOAT_EQ(cell->risk, expected_risk);
    //
    // This proves the first-observation branch: cell.risk == observation,
    // NOT alpha * observation + (1-alpha) * 0.

    cfg_.update_rule = UpdateRule::EMA;
    WorldGrid g(cfg_);                         // rebuild with updated cfg
    const int ix = 10, iy = 30;
    TraversabilityGrid local = makeSingleCellGrid(ix, iy, /*mean_z=*/0.0f);
    const float expected_risk = local.at(ix, iy).risk;
    ASSERT_GT(local.at(ix, iy).point_count, 0);
    g.update(local, identityPose(), /*pose_sigma=*/0.0, /*timestamp=*/0.0);
    auto cc = localCellCenterInWorld(ix, iy);
    const WorldCell* cell = g.getCell(cc.x(), cc.y());
    ASSERT_NE(cell, nullptr);
    EXPECT_FLOAT_EQ(cell->risk, expected_risk);


}

TEST_F(WorldGridTest, EMA_SteadyStateConverges) {
    // YOUR CODE:
    //   Feed the same risk=0.8 observation 100 times with alpha=0.3.
    //   Assert final cell.risk is within 1e-3 of 0.8.
    //   (Geometric convergence: error halves every ~2 frames at alpha=0.3.)
    cfg_.update_rule = UpdateRule::EMA;
    WorldGrid g(cfg_);                         // rebuild with updated cfg
    const int ix = 10, iy = 30;
    TraversabilityGrid local = makeSingleCellGrid(ix, iy, /*mean_z=*/0.0f);
    const float expected_risk = local.at(ix, iy).risk;
    ASSERT_GT(local.at(ix, iy).point_count, 0);
    for (int i = 0; i < 100; ++i) {
        g.update(local, identityPose(), /*pose_sigma=*/0.0, /*timestamp=*/0.0);
    }
    auto cc = localCellCenterInWorld(ix, iy);
    const WorldCell* cell = g.getCell(cc.x(), cc.y());
    ASSERT_NE(cell, nullptr);
    EXPECT_NEAR(cell->risk, expected_risk, 1e-3);
}

TEST_F(WorldGridTest, LogOdds_HitSaturatesAtClamp) {
    // YOUR CODE:
    //   cfg_.update_rule = UpdateRule::LogOdds;
    //   Use default cfg_.logodds_hit=0.85, clamp_max=3.5.
    //   Feed risk=1.0 (hit) for 100 iterations.
    //   Assert cell.logodds ≈ 3.5 (clamped).
    //   Assert cell.risk ≈ testSigmoid(3.5f) ≈ 0.971.
    //   EXPECT_NEAR(cell.risk, testSigmoid(static_cast<float>(cfg_.logodds_clamp_max)), 1e-5);
    cfg_.update_rule = UpdateRule::LogOdds;
    WorldGrid g(cfg_);
    const int ix = 10, iy = 30;
    // hazardous=true → risk > 0.5 → every update fires the HIT branch (+logodds_hit).
    TraversabilityGrid local = makeSingleCellGrid(ix, iy, /*mean_z=*/0.0f, /*hazardous=*/true);
    ASSERT_GT(local.at(ix, iy).point_count, 0);
    ASSERT_GT(local.at(ix, iy).risk, 0.5f)
        << "makeSingleCellGrid(hazardous=true) did not produce risk > 0.5; "
           "adjust the synthetic geometry so compute_vehicle_aware_score pushes risk into hazard regime.";
    for (int i = 0; i < 100; ++i) {
        g.update(local, identityPose(), /*pose_sigma=*/0.0, /*timestamp=*/0.0);
    }
    auto cc = localCellCenterInWorld(ix, iy);
    const WorldCell* cell = g.getCell(cc.x(), cc.y());
    ASSERT_NE(cell, nullptr);
    EXPECT_NEAR(cell->logodds, cfg_.logodds_clamp_max, 1e-5);
    EXPECT_NEAR(cell->risk, testSigmoid(static_cast<float>(cfg_.logodds_clamp_max)), 1e-5);
}

TEST_F(WorldGridTest, LogOdds_MissSaturatesAtClamp) {
    // YOUR CODE:
    //   Symmetric to above. Feed risk=0.0 (miss) 100×.
    //   Assert cell.logodds ≈ -3.5.
    //   EXPECT_NEAR(cell.risk, testSigmoid(static_cast<float>(cfg_.logodds_clamp_min)), 1e-5);
    cfg_.update_rule = UpdateRule::LogOdds;
    WorldGrid g(cfg_);                         // rebuild with updated cfg
    const int ix = 10, iy = 30;
    TraversabilityGrid local = makeSingleCellGrid(ix, iy, /*mean_z=*/0.0f);
    const float expected_risk = local.at(ix, iy).risk;
    ASSERT_GT(local.at(ix, iy).point_count, 0);
    for (int i = 0; i < 100; ++i) {
        g.update(local, identityPose(), /*pose_sigma=*/0.0, /*timestamp=*/0.0);
    }
    auto cc = localCellCenterInWorld(ix, iy);
    const WorldCell* cell = g.getCell(cc.x(), cc.y());
    ASSERT_NE(cell, nullptr);
    EXPECT_NEAR(cell->logodds, cfg_.logodds_clamp_min, 1e-5);
    EXPECT_NEAR(cell->risk, testSigmoid(static_cast<float>(cfg_.logodds_clamp_min)), 1e-5);
}

TEST_F(WorldGridTest, LogOdds_RiskSyncedToSigmoid) {
    // YOUR CODE:
    //   Drive one or more log-odds updates into a cell. Regardless of the
    //   intermediate logodds value, the invariant must hold:
    //     EXPECT_NEAR(cell->risk, testSigmoid(cell->logodds), 1e-5);
    //   Invariant: risk is always in sync with logodds.
    cfg_.update_rule = UpdateRule::LogOdds;
    WorldGrid g(cfg_);                         // rebuild with updated cfg  
    const int ix = 10, iy = 30; 
    TraversabilityGrid local = makeSingleCellGrid(ix, iy, /*mean_z=*/0.0f);
    const float expected_risk = local.at(ix, iy).risk;
    ASSERT_GT(local.at(ix, iy).point_count, 0);
    for (int i = 0; i < 100; ++i) {
        g.update(local, identityPose(), /*pose_sigma=*/0.0, /*timestamp=*/0.0);
        auto cc = localCellCenterInWorld(ix, iy);
        const WorldCell* cell = g.getCell(cc.x(), cc.y());
        ASSERT_NE(cell, nullptr);
        EXPECT_NEAR(cell->risk, testSigmoid(cell->logodds), 1e-5);
    }   
}

TEST_F(WorldGridTest, Overwrite_HasNoMemory) {
    // YOUR CODE:
    //   cfg_.update_rule = UpdateRule::Overwrite;
    //   Feed risk=0.9, then risk=0.1.
    //   Assert cell.risk == 0.1 (no blending, last wins).
    cfg_.update_rule = UpdateRule::Overwrite;
    WorldGrid g(cfg_);                         // rebuild with updated cfg
    const int ix = 10, iy = 30;
    TraversabilityGrid local1 = makeSingleCellGrid(ix, iy, /*mean_z=*/0.0f);
    TraversabilityGrid local2 = makeSingleCellGrid(ix, iy, /*mean_z=*/0.0f);
    const float expected_risk1 = local1.at(ix, iy).risk;
    const float expected_risk2 = local2.at(ix, iy).risk;
    ASSERT_GT(local1.at(ix, iy).point_count, 0);
    ASSERT_GT(local2.at(ix, iy).point_count, 0);    
}

// -----------------------------------------------------------------------------
// Confidence & decay
// -----------------------------------------------------------------------------

TEST_F(WorldGridTest, ConfidenceGrowsMonotonically) {
    // YOUR CODE:
    //   Feed N observations, record cell.confidence each time.
    //   Assert confidence is non-decreasing and caps at cfg_.confidence_max.
    cfg_.update_rule = UpdateRule::EMA;
    WorldGrid g(cfg_);                         // rebuild with updated cfg
    const int ix = 10, iy = 30;
    TraversabilityGrid local = makeSingleCellGrid(ix, iy, /*mean_z=*/0.0f);
    const float expected_risk = local.at(ix, iy).risk;
    ASSERT_GT(local.at(ix, iy).point_count, 0);
    float last_conf = 0.0f;
    for (int i = 0; i < 100; ++i) {
        g.update(local, identityPose(), /*pose_sigma=*/0.0, /*timestamp=*/0.0);
        auto cc = localCellCenterInWorld(ix, iy);
        const WorldCell* cell = g.getCell(cc.x(), cc.y());
        ASSERT_NE(cell, nullptr);
        EXPECT_GE(cell->confidence, last_conf);
        last_conf = cell->confidence;
        EXPECT_LE(cell->confidence, static_cast<float>(cfg_.confidence_max));
    }
}

TEST_F(WorldGridTest, ConfidenceScaledByPoseSigma) {
    // YOUR CODE:
    //   Run two identical updates except pose_sigma: one at 0.0 (perfect),
    //   one at 2.0 (high uncertainty).
    //   Assert: confidence from low-sigma run > confidence from high-sigma run.
    //   This is the SLAM-uncertainty → map-confidence propagation (ablation E).
    cfg_.update_rule = UpdateRule::EMA;
    const int ix = 10, iy = 30;
    TraversabilityGrid local = makeSingleCellGrid(ix, iy, /*mean_z=*/0.0f);
    ASSERT_GT(local.at(ix, iy).point_count, 0);

    // Two INDEPENDENT grids, each receiving exactly one observation —
    // otherwise the "high-sigma" grid also has the "low-sigma" update baked in
    // and the comparison is confounded.
    WorldGrid g_lo(cfg_);
    WorldGrid g_hi(cfg_);
    g_lo.update(local, identityPose(), /*pose_sigma=*/0.0, /*timestamp=*/0.0);
    g_hi.update(local, identityPose(), /*pose_sigma=*/2.0, /*timestamp=*/0.0);

    auto cc = localCellCenterInWorld(ix, iy);
    const WorldCell* cell_lo = g_lo.getCell(cc.x(), cc.y());
    const WorldCell* cell_hi = g_hi.getCell(cc.x(), cc.y());
    ASSERT_NE(cell_lo, nullptr);
    ASSERT_NE(cell_hi, nullptr);

    EXPECT_GT(cell_lo->confidence, cell_hi->confidence)
        << "pose_sigma scaling is not reducing confidence for the high-sigma pose. "
           "Check cfg_.pose_uncertainty_k and adjustConfidence().";
}

TEST_F(WorldGridTest, DecayDisabledByZeroRate) {
    // YOUR CODE:
    //   cfg_.decay_rate = 0.0.
    //   Observe a cell, then call applyDecay(t_future).
    //   Assert cell.confidence is unchanged.
    cfg_.decay_rate = 0.0;
    WorldGrid g(cfg_);                         // rebuild with updated cfg
    const int ix = 10, iy = 30;
    TraversabilityGrid local = makeSingleCellGrid(ix, iy, /*mean_z=*/0.0f);
    const float expected_risk = local.at(ix, iy).risk;
    ASSERT_GT(local.at(ix, iy).point_count, 0);
    g.update(local, identityPose(), /*pose_sigma=*/0.0, /*timestamp=*/0.0);
    auto cc = localCellCenterInWorld(ix, iy);
    const WorldCell* cell_before_decay = g.getCell(cc.x(), cc.y());
    ASSERT_NE(cell_before_decay, nullptr);
    float confidence_before_decay = cell_before_decay->confidence;
    g.applyDecay(100.0);  // far future
    const WorldCell* cell_after_decay = g.getCell(cc.x(), cc.y());
    ASSERT_NE(cell_after_decay, nullptr);
    float confidence_after_decay = cell_after_decay->confidence;
    EXPECT_FLOAT_EQ(confidence_before_decay, confidence_after_decay);
}

TEST_F(WorldGridTest, DecayOnlyTouchesConfidenceNotRisk) {
    // YOUR CODE:
    //   cfg_.decay_rate = 0.5 (strong).
    //   Observe cell with risk=0.8, confidence > 0.
    //   applyDecay(t + 100 seconds).
    //   Assert cell.risk still ≈ 0.8 (UNCHANGED — safety invariant).
    //   Assert cell.confidence == 0.0 (fully decayed).
    cfg_.decay_rate = 0.5;
    WorldGrid g(cfg_);                         // rebuild with updated cfg
    const int ix = 10, iy = 30;
    TraversabilityGrid local = makeSingleCellGrid(ix, iy, /*mean_z=*/0.0f);
    const float expected_risk = local.at(ix, iy).risk;
    ASSERT_GT(local.at(ix, iy).point_count, 0);
    g.update(local, identityPose(), /*pose_sigma=*/0.0, /*timestamp=*/0.0);
    auto cc = localCellCenterInWorld(ix, iy);
    const WorldCell* cell_before_decay = g.getCell(cc.x(), cc.y());
    ASSERT_NE(cell_before_decay, nullptr);
    float risk_before_decay = cell_before_decay->risk;
    float confidence_before_decay = cell_before_decay->confidence;
    g.applyDecay(100.0);  // far future
    const WorldCell* cell_after_decay = g.getCell(cc.x(), cc.y());
    ASSERT_NE(cell_after_decay, nullptr);
    float risk_after_decay = cell_after_decay->risk;
    float confidence_after_decay = cell_after_decay->confidence;
    EXPECT_FLOAT_EQ(risk_before_decay, risk_after_decay);
    EXPECT_FLOAT_EQ(confidence_after_decay, 0.0f);
}

// -----------------------------------------------------------------------------
// Coverage metric
// -----------------------------------------------------------------------------

TEST_F(WorldGridTest, CoverageMonotonicOverFrames) {
    // YOUR CODE:
    //   Simulate a trajectory of 50 (x, y) points spaced 0.1m apart along x.
    //   After each "frame," call g.coveragePercent(traj_so_far).
    //   Assert coverage is non-decreasing across frames.
    WorldGrid g(cfg_);                         // rebuild with updated cfg
    std::vector<Eigen::Vector2d> traj;
    for (int i = 0; i < 50; ++i) {
        double x = -2.0 + i * 0.1;  // from -2.0 to +2.9
        double y = 0.0;
        traj.emplace_back(x, y);
        float coverage = g.coveragePercent(traj);
        if (i > 0) {
            EXPECT_GE(coverage, g.coveragePercent(std::vector<Eigen::Vector2d>(traj.begin(), traj.end() - 1)));
        }
    }
}

TEST_F(WorldGridTest, CoverageHullTighterThanAABB) {
    // YOUR CODE — the ablation-worthy test:
    //   Build a C-shaped trajectory: go right, curve up, come back left.
    //   The AABB over this fills a big square; the convex hull fills roughly
    //   half. If you observe only cells the vehicle actually passed over,
    //   coverage-via-hull-denominator > coverage-via-AABB-denominator.
    //
    //   This test currently has no way to call the AABB version explicitly
    //   (convexHull is file-local in world_grid.cpp). Options:
    //     (a) Compute a reference AABB coverage here using g.at(r,c) and your
    //         own trajectory bbox, then assert coveragePercent() > reference.
    //     (b) Expose convexHull/AABB as public static methods for testing.
    //   Prefer (a) — cleaner encapsulation, still exercises the real path.
    WorldGrid g(cfg_);

    // Triangular trajectory — hull area is exactly half the AABB area, and
    // the triangle interior (where we place the single observation) is
    // guaranteed inside BOTH the hull and the AABB.
    //
    //   vertex (-2,  0)   bottom-left
    //   vertex ( 2,  0)   bottom-right
    //   vertex ( 0,  2)   top
    //   centroid ( 0, 2/3)  ← where we observe
    //
    // AABB = [-2, 2] × [0, 2], area = 8.   (cells inside ≈ 32 at 0.5m res)
    // Triangle area = 0.5 * base * height = 0.5 * 4 * 2 = 4. (cells inside ≈ 16)
    // Hull denominator ≈ AABB / 2 ⇒ same numerator (1) ⇒ coverage_hull ≈ 2× coverage_aabb.
    std::vector<Eigen::Vector2d> traj = {
        {-2.0, 0.0},
        { 2.0, 0.0},
        { 0.0, 2.0}
    };

    // Pose at the triangle centroid. The one observed cell lands at
    // (centroid + local-cell-offset), which is still inside the triangle
    // since the centroid sits well away from any edge.
    Pose pose;
    pose.R = Eigen::Matrix3d::Identity();
    pose.t = Eigen::Vector3d(0.0, 2.0 / 3.0, 0.0);

    TraversabilityGrid local = makeSingleCellGrid(10, 30, /*mean_z=*/0.0f);
    g.update(local, pose, /*pose_sigma=*/0.0, /*timestamp=*/0.0);

    const double coverage_hull = g.coveragePercent(traj);

    // Reference AABB coverage: count cells whose centers are inside the AABB.
    double xmin = +std::numeric_limits<double>::infinity(),
           xmax = -std::numeric_limits<double>::infinity();
    double ymin = xmin, ymax = xmax;
    for (const auto& p : traj) {
        xmin = std::min(xmin, p.x()); xmax = std::max(xmax, p.x());
        ymin = std::min(ymin, p.y()); ymax = std::max(ymax, p.y());
    }
    int aabb_total = 0, aabb_observed = 0;
    for (int r = 0; r < g.rows(); ++r) {
        for (int c = 0; c < g.cols(); ++c) {
            const double cx = cfg_.x_min + (r + 0.5) * cfg_.resolution;
            const double cy = cfg_.y_min + (c + 0.5) * cfg_.resolution;
            if (cx < xmin || cx > xmax || cy < ymin || cy > ymax) continue;
            ++aabb_total;
            if (g.at(r, c).obs_count > 0) ++aabb_observed;
        }
    }
    ASSERT_GT(aabb_total, 0);
    ASSERT_GT(aabb_observed, 0) << "no observed cells landed in the AABB — trajectory or offset is wrong";
    const double coverage_aabb = static_cast<double>(aabb_observed) / aabb_total;

    // With a single interior observation, hull denominator (~triangle cells) is
    // ~half the AABB denominator, so coverage_hull should be ~2× coverage_aabb.
    EXPECT_GT(coverage_hull, coverage_aabb)
        << "hull=" << coverage_hull << " aabb=" << coverage_aabb;
}

// -----------------------------------------------------------------------------
// Snapshot round-trip
// -----------------------------------------------------------------------------

TEST_F(WorldGridTest, SaveLoadRoundTripCellEquality) {
    // YOUR CODE:
    //   WorldGrid g1(cfg_);
    //   Update a handful of cells with distinct risk/confidence values.
    //   Save to a temp path (use std::tmpnam or ::testing::TempDir()).
    //
    //   WorldGrid g2(cfg_);
    //   ASSERT_TRUE(g2.loadSnapshot(tmp_path));
    //
    //   For each cell touched in g1, assert g1.at(r,c).risk == g2.at(r,c).risk
    //   and confidence/obs_count/mean_z match within 1e-5.
    //
    //   This test proves the column-header-skip fix. Without it, loadSnapshot
    //   returns false and ASSERT_TRUE fires.
    //
    //   Cleanup: std::remove(tmp + ".csv"); std::remove(tmp + ".png");
    WorldGrid g1(cfg_);
    const int ix = 10, iy = 30;
    TraversabilityGrid local = makeSingleCellGrid(ix, iy, /*mean_z=*/0.0f);
    const float expected_risk = local.at(ix, iy).risk;
    ASSERT_GT(local.at(ix, iy).point_count, 0);
    g1.update(local, identityPose(), /*pose_sigma=*/0.0, /*timestamp=*/0.0);
    std::string tmp_path = ::testing::TempDir() + "/world_grid_test_snapshot";
    ASSERT_TRUE(g1.saveSnapshot(tmp_path));
    WorldGrid g2(cfg_);
    ASSERT_TRUE(g2.loadSnapshot(tmp_path));
    auto cc = localCellCenterInWorld(ix, iy);
    const WorldCell* cell1 = g1.getCell(cc.x(), cc.y());
    const WorldCell* cell2 = g2.getCell(cc.x(), cc.y());
    ASSERT_NE(cell1, nullptr);
    ASSERT_NE(cell2, nullptr);
    EXPECT_FLOAT_EQ(cell1->risk, cell2->risk);
    EXPECT_FLOAT_EQ(cell1->confidence, cell2->confidence);
    EXPECT_EQ(cell1->obs_count, cell2->obs_count);
    EXPECT_FLOAT_EQ(cell1->mean_z, cell2->mean_z);
    std::remove((tmp_path + ".csv").c_str());
    std::remove((tmp_path + ".png").c_str());
}

TEST_F(WorldGridTest, LoadMismatchedConfigFails) {
    // YOUR CODE:
    //   Save with cfg_ (10x10).
    //   Build WorldGrid with DIFFERENT resolution (say 1.0).
    //   Assert loadSnapshot returns false (mismatch detected by metadata parse).
    WorldGrid g1(cfg_);
    const int ix = 10, iy = 30;
    TraversabilityGrid local = makeSingleCellGrid(ix, iy, /*mean_z=*/0.0f);
    const float expected_risk = local.at(ix, iy).risk;
    ASSERT_GT(local.at(ix, iy).point_count, 0);
    g1.update(local, identityPose(), /*pose_sigma=*/0.0, /*timestamp=*/0.0);
    std::string tmp_path = ::testing::TempDir() + "/world_grid_test_snapshot";
    ASSERT_TRUE(g1.saveSnapshot(tmp_path));
    WorldGridConfig different_cfg = cfg_;
    different_cfg.resolution = 1.0;  // mismatch
    WorldGrid g2(different_cfg);
    ASSERT_FALSE(g2.loadSnapshot(tmp_path));
    std::remove((tmp_path + ".csv").c_str());
    std::remove((tmp_path + ".png").c_str());
}

// -----------------------------------------------------------------------------
// main — ament_cmake_gtest provides one by default; no need to define here.
// -----------------------------------------------------------------------------
