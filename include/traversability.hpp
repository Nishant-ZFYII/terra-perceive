// traversability.hpp
// BEV traversability grid — implement from scratch.
// Phase 1 upgrade: VEHICLE-AWARE scoring + CONFIDENCE tracking.
//
// PDF reference: Part 3 (pages 11-12) + Critique ("Traversability score is a magic number")
// Read: Wermelinger et al., "Navigation Planning for Legged Robots" (IROS 2016)
// Watch: Cyrill Stachniss, "Point Cloud Processing" (YouTube)
// Eigen ref: SelfAdjointEigenSolver for PCA
//
// YOUR TASK (P1-M3, Days 5-6):
//   P1-M3.1: Grid binning — single pass O(N)
//   P1-M3.2: PCA surface normals via covariance eigendecomposition
//   P1-M3.3: Feature computation — slope, roughness, step height
//   P1-M3.4: Vehicle-aware scoring — tie to physical machine limits, not magic weights
//   P1-M3.5: Confidence tracking — fewer points or farther range = lower confidence
//   P1-M3.6: BEV visualization (Python helper in scripts/visualize_bev.py)
//
// Key insight: score = f(slope/max_slope, roughness/max_roughness, step/max_step)
// where max_* comes from actual vehicle specs, not arbitrary tuning.

#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <vector>

struct VehicleKinematics {
    float max_climbable_grade_deg = 20.0f;   // drum roller limit
    float max_roughness_tolerance = 0.15f;   // meters RMS
    float max_step_height = 0.3f;            // meters
    float vehicle_mass_kg = 30000.0f;        // 30-ton drum roller
    float max_speed_mps = 2.0f;              // meters per second
};

struct GridParams {
    float resolution = 0.5f;
    float x_min = -5.0f, x_max = 30.0f;
    float y_min = -15.0f, y_max = 15.0f;
    int min_points_per_cell = 3;
};

struct CellFeatures {
    float slope_deg = 0.0f;
    float roughness = 0.0f;
    float step_height = 0.0f;
    int point_count = 0;
    float traversability_score = 0.5f;  // [0, 1]
    float confidence = 0.0f;            // [0, 1] — how much to trust this score
    float mean_z = 0.0f;               // mean elevation for visualization
    float range_from_sensor = 0.0f;    // mean distance from origin
};

class TraversabilityGrid {
   public:
    TraversabilityGrid(const GridParams& params, const VehicleKinematics& vehicle);
    void compute(const std::vector<Eigen::Vector3f>& ground_pts);
    const CellFeatures& at(int ix, int iy) const;
    int rows() const;
    int cols() const;
    std::vector<int8_t> to_occupancy_grid() const;

    const GridParams& grid_params() const { return params_; }
    const VehicleKinematics& vehicle_params() const { return vehicle_; }

   private:
    GridParams params_;
    VehicleKinematics vehicle_;
    std::vector<std::vector<CellFeatures>> grid_;

    Eigen::Vector3f compute_normal(const std::vector<Eigen::Vector3f>& pts);
    float compute_confidence(int point_count, float range_from_sensor);
    float compute_vehicle_aware_score(float slope_deg, float roughness, float step_height);
};
