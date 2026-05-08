// traversability.hpp
// BEV traversability grid — implement from scratch.
// Phase 1 upgrade: VEHICLE-AWARE scoring + CONFIDENCE tracking.
//
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
    float max_climbable_grade_deg = 30.0f;   // warthog limit
    float max_roughness_tolerance = 0.275f;   // meters RMS
    float max_step_height = 0.2f;            // meters
    float vehicle_mass_kg = 590.0f;        // weight from spec sheet
    float max_speed_mps = 5.0f;              // meters per second
};

// P2-M6: confidence-mode toggle for the traversability grid.
// Heuristic   = legacy formula (count_factor * range_factor) shipped in P1-M3.
// Probabilistic = noise-aware formula derived in notes/m6_math_sketch.md:
//                 propagates per-point range noise sigma(r) through the cell's
//                 PCA eigenvalues and combines planarity, sample size, and
//                 noise-vs-cell-extent factors.
enum class ConfidenceMode { Heuristic, Probabilistic };

// Ouster OS1-64 range noise model, sigma(r) = sigma_0 + k * r^2.
// Defaults are anchored to the OS1-64 datasheet (~1cm at 2m, ~7cm at 25m).
struct LidarNoiseModel {
    float sigma_0 = 0.01f;
    float k = 0.0001f;
};

// Range-dependent LiDAR range noise. Pure function; exposed so tests can hit it
// directly without constructing a full grid.
float lidar_sigma(float r, const LidarNoiseModel& m);

// Output of compute_normal_with_eigenvalues. Eigenvalues are sorted ascending:
// eigenvalues[0] = lambda_min (perpendicular-to-plane component for a planar cell),
// eigenvalues[2] = lambda_max (largest in-plane spread).
struct NormalEstimate {
    Eigen::Vector3f normal = Eigen::Vector3f::UnitZ();
    Eigen::Vector3f eigenvalues = Eigen::Vector3f::Zero();
    float mean_sigma_r = 0.0f;
    int point_count = 0;
};

//Lidar frame -> x-> forward, y-> left
struct GridParams {
    //0.5m x 0.5m cell ; if too small PCA might fail due to few points. if too large, we loose detail.
    //0.5m gives ~4-10 ground points per cell within 15m range on Ouster 0s1-64
    float resolution = 0.5f;

    //grid covers -5m rear to +30m ahead
    float x_min = -5.0f, x_max = 30.0f;

    //grid covers 15m to each side, total 30m width
    float y_min = -15.0f, y_max = 15.0f;

    //min points for PCA normal estimation.
    int min_points_per_cell = 3;

    // P2-M6: confidence-mode toggle and noise-model parameters.
    // Default keeps legacy heuristic behaviour; existing callers see no change.
    ConfidenceMode confidence_mode = ConfidenceMode::Heuristic;
    LidarNoiseModel noise_model;
};

struct CellFeatures {
    //roughness formuale from 'Semantic point cloud segmentation based on surface normal and curvature' paper.Weinmann et al. 2015
    //angle between cell's surface normal and the verical. 0 for ground 90 for a wall. 
    float slope_deg = 0.0f;

    //eigen value ration 
    float roughness = 0.0f;

    //max(z) - min(z) within the cell, gives an estimate of step height.
    //Catch discrete obstacles, a single tall rock can have low roughness but high step height.
    float step_height = 0.0f;

    //store the number of point clouds per bin
    //if < min_points_per_cell, skip PCA
    //used in compute_confidence() -> more points -> higher confidence.
    int point_count = 0;

    //risk -> 0 safe risk -> 1 unsafe. 
    float risk = 0.0f;  // [0, 1]

    // [0, 1] — how much to trust this score
    float confidence = 0.0f;            

    // mean elevation for visualization. BEV color coding
    float mean_z = 0.0f;
    
    // mean distance from origin. farther cell gets lower confidence due to LiDAR noise increase with range.
    float range_from_sensor = 0.0f;    
};

class TraversabilityGrid {
   public:
    TraversabilityGrid(const GridParams& params, const VehicleKinematics& vehicle);
    // Wermelinger et al. (IROS 2016) — Section II.A                                                                            
    // "elevation map... two-dimensional regular grid"   

    //called once per lidar frame
    //out of RANSAC segementer and fills in grid
    void compute(const std::vector<Eigen::Vector3f>& ground_pts);

    const CellFeatures& at(int ix, int iy) const;

    //grid dimensions
    int rows() const;
    int cols() const;

    //convert to nav2 fomrat. 
    std::vector<int8_t> to_occupancy_grid() const;

    const GridParams& grid_params() const { return params_; }
    const VehicleKinematics& vehicle_params() const { return vehicle_; }

   private:
    GridParams params_;
    VehicleKinematics vehicle_;
    
    // Wermelinger (IROS 2016) — Section II.A: "regular grid, each cell stores h and sigma(variance)"
    //BeV grid
    std::vector<std::vector<CellFeatures>> grid_;

    // Weinmann (2015) Eq(1),(3) + PCL normal estimation tutorial
    // covariance → SelfAdjointEigenSolver → col(0) = normal
    Eigen::Vector3f compute_normal(const std::vector<Eigen::Vector3f>& pts);
    float compute_confidence(int point_count, float range_from_sensor);

    // P2-M6: sibling of compute_normal that returns eigenvalues + mean range
    // alongside the normal. Public so tests can target the eigenvalue logic
    // directly. Defined per Fork 1 (sibling, not mutate) of the M6 plan.
   public:
    NormalEstimate compute_normal_with_eigenvalues(const std::vector<Eigen::Vector3f>& pts) const;

    // P2-M6: probabilistic confidence formula. See notes/m6_math_sketch.md §1.4.
    // Combines: (a) planarity above the noise floor, (b) sample-size confidence,
    // (c) range factor that penalises noise dominating cell-scale geometry.
    float compute_probabilistic_confidence(const NormalEstimate& est) const;

   private:
    // Wermelinger (IROS 2016) Eq(1) — concept (weighted penalties)
    float compute_vehicle_aware_score(float slope_deg, float roughness, float step_height);
};
