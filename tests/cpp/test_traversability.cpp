// test_traversability.cpp — Tests for traversability grid.
// Covers vehicle-aware scoring and confidence tracking.
//
// P1-M3 checkpoints:
//   M3.1: Grid binning assigns points to correct cells
//   M3.2: PCA normals correct for known surfaces
//   M3.3: Slope, roughness, step computed correctly
//   M3.4: Vehicle-aware scoring matches physical limits
//   M3.5: Confidence degrades with range and sparse data

#include <gtest/gtest.h>

#include "traversability.hpp"

TEST(Traversability, FlatGround) {
    // TODO: Create flat ground at z=0, slight noise
    // ASSERT: slope ~0 deg, roughness ~0, step ~0
    // ASSERT: traversability_score close to 1.0
}

TEST(Traversability, SlopedSurface) {
    // TODO: Create 30-degree slope
    // ASSERT: slope_deg approximately 30
    // With max_climbable_grade=20: this exceeds the limit
    // ASSERT: traversability_score < 0.3
}

TEST(Traversability, VehicleAwareScoring) {
    // TODO: slope=10 deg, max_climbable=20 deg
    // ASSERT: slope_penalty = 1 - (10/20)^2 = 0.75
    // slope=20 deg: penalty = 1 - (20/20)^2 = 0.0 (at limit)
    // slope=0 deg: penalty = 1.0 (perfect)
}

TEST(Traversability, UnknownCells) {
    // TODO: Cell with 0 points
    // ASSERT: score = 0.5 (unknown), confidence = 0.0
    // Cell with 1 point (below min_points_per_cell)
    // ASSERT: score = 0.5, confidence = 0.0
}

TEST(Traversability, ConfidenceNearFar) {
    // TODO: Cells at range 2m with 20 points -> confidence > 0.8
    // Cells at range 25m with 5 points -> confidence < 0.5
    // Empty cells -> confidence = 0.0
}

TEST(Traversability, RoughSurface) {
    // TODO: Create noisy surface with high z variance
    // ASSERT: roughness > max_roughness_tolerance
    // ASSERT: traversability_score penalized
}
