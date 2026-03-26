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

    // Wermelinger (IROS 2016) Eq(1) — concept (weighted penalties) 
    float compute_vehicle_aware_score(float slope_deg, float roughness, float step_height);
};
