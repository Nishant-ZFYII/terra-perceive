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
#include <cmath>
#include <vector>
#include "traversability.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Fixture helpers — you implement these
static GridParams test_grid_params();    // returns GridParams matching config                                                  
static VehicleKinematics test_vehicle(); // fixed params for deterministic tests, NOT Warthog defaults                          
                                                                                                                                
// Point cloud generators — you implement these                                                                                 
// make_flat_points(p, ix, iy, n)    — n points in cell (ix,iy) at z=0, slight noise                                            
// make_sloped_points(p, ix, iy, n, tilt_deg) — n points on a tilted plane                                                      
// make_rough_points(p, ix, iy, n)   — n points with high z variance  

//make_flat_points
//take GridPrams p, cell index ix and iy , count n
//reutn vector with n points inside cell (ix,iy) with z ~ 0 with tiny noise

static GridParams test_grid_params() {
    GridParams params;
    params.resolution = 0.5f;
    params.x_min = -5.0f;
    params.x_max = 30.0f;
    params.y_min = -15.0f;
    params.y_max = 15.0f;
    params.min_points_per_cell = 3;
    return params;
}

static VehicleKinematics test_vehicle() {
    VehicleKinematics vehicle;
    vehicle.max_climbable_grade_deg = 20.0f;
    vehicle.max_roughness_tolerance = 0.12f;
    vehicle.max_step_height = 0.3f;
    vehicle.vehicle_mass_kg = 590.0f;
    vehicle.max_speed_mps = 5.0f;
    return vehicle;
}

static std::vector<Eigen::Vector3f> make_flat_points(const GridParams& p, int ix, int iy, int n) {
    std::vector<Eigen::Vector3f> pts;
    //start with cell corner
    float x_start = p.x_min + ix * p.resolution;
    float y_start = p.y_min + iy * p.resolution;
    int cols_per_row = static_cast<int>(std::ceil(std::sqrt(n)));

    //deterministic noise
    for(int it = 0; it < n; ++it){
        float x = x_start + (static_cast<float>(it / cols_per_row) + 0.5f) * p.resolution / cols_per_row; // evenly spaced in cell
        float y = y_start + (static_cast<float>(it % cols_per_row) + 0.5f) * p.resolution / cols_per_row; // evenly spaced in cell
        float z = 0.01f * ((it % 3 == 0) ? 1.0f : -1.0f); // small noise in z
        pts.emplace_back(x, y, z);

    }
    return pts;
}

static std::vector<Eigen::Vector3f> make_sloped_points(const GridParams& p, int ix, int iy, int n, float tilt_deg) {
    std::vector<Eigen::Vector3f> pts;
    float tilt_rad = tilt_deg * M_PI / 180.0f;
    float z_slope = std::tan(tilt_rad);
    float x_start = p.x_min + ix * p.resolution;
    float y_start = p.y_min + iy * p.resolution;
    float x_center = x_start + 0.5f * p.resolution;
    int cols_per_row = static_cast<int>(std::ceil(std::sqrt(n)));

    for(int it = 0; it < n; ++it){
        float x = x_start + (static_cast<float>(it / cols_per_row) + 0.5f) * p.resolution / cols_per_row; // evenly spaced in cell
        float y = y_start + (static_cast<float>(it % cols_per_row) + 0.5f) * p.resolution / cols_per_row; // evenly spaced in cell
        float z = z_slope * (x - x_center); // slope in x direction, flat in y
        pts.emplace_back(x, y, z);
    }
    return pts;
}

static std::vector<Eigen::Vector3f> make_rough_points(const GridParams& p, int ix, int iy, int n) {
    std::vector<Eigen::Vector3f> pts;
    float x_start = p.x_min + ix * p.resolution;
    float y_start = p.y_min + iy * p.resolution;
    int cols_per_row = static_cast<int>(std::ceil(std::sqrt(n)));

    for(int it = 0; it < n; ++it){
        float x = x_start + (static_cast<float>(it / cols_per_row) + 0.5f) * p.resolution / cols_per_row; // evenly spaced in cell
        float y = y_start + (static_cast<float>(it % cols_per_row) + 0.5f) * p.resolution / cols_per_row; // evenly spaced in cell
        float z = (it % 3 == 0) ? +0.25f : -0.25f ; // high variance in z
        pts.emplace_back(x, y, z);
    }
    return pts;
}

//create point cloud where roughness ~ 0 and step ~0. 
//Used for TEST(Traversability, VehicleAwareScoring) 




TEST(Traversability, FlatGround) {
    // TODO: Create flat ground at z=0, slight noise
    // ASSERT: slope ~0 deg, roughness ~0, step ~0
    // ASSERT: risk close to 0.0
    GridParams params = test_grid_params();
    VehicleKinematics vehicle = test_vehicle();
    TraversabilityGrid grid(params, vehicle);
    auto flat_pts = make_flat_points(params, 10, 10, 20); // cell (10,10) with 20 points
    grid.compute(flat_pts);
    const CellFeatures& cell = grid.at(10, 10);
    EXPECT_NEAR(cell.slope_deg, 0.0f, 1.0f);
    EXPECT_NEAR(cell.roughness, 0.0f, 0.05f);
    EXPECT_NEAR(cell.step_height, 0.0f, 0.05f);
    EXPECT_NEAR(cell.risk, 0.0f, 0.1f);
}

TEST(Traversability, SlopedSurface) {
    // TODO: Create 30-degree slope
    // ASSERT: slope_deg approximately 30
    // With max_climbable_grade=20: this exceeds the limit
    // ASSERT: risk > 0.7
    GridParams params = test_grid_params();
    VehicleKinematics vehicle = test_vehicle();
    TraversabilityGrid grid(params, vehicle);
    auto sloped_pts = make_sloped_points(params, 10, 10, 20, 30.0f); // cell (10,10) with 30-degree slope
    grid.compute(sloped_pts);
    const CellFeatures& cell = grid.at(10, 10);
    EXPECT_NEAR(cell.slope_deg, 30.0f, 5.0f);
    EXPECT_GT(cell.risk, 0.7f);
}

TEST(Traversability, VehicleAwareScoring) {
    // TODO: slope=10 deg, max_climbable=20 deg
    // ASSERT: slope_penalty = 1 - (10/20)^2 = 0.75
    // slope=20 deg: penalty = 1 - (20/20)^2 = 0.0 (at limit)
    // slope=0 deg: penalty = 1.0 (perfect)
    GridParams params = test_grid_params();
    VehicleKinematics vehicle = test_vehicle();
    TraversabilityGrid grid(params, vehicle);
    //slope = 0
    auto flat_pts = make_sloped_points(params, 10, 10, 20, 0.0f);
    grid.compute(flat_pts);
    const CellFeatures& cell_flat = grid.at(10, 10);
    EXPECT_NEAR(cell_flat.risk, 0.0f, 0.1f);
    //slope = 10
    auto flat_pts_10 = make_sloped_points(params, 10, 10, 20, 10.0f);
    grid.compute(flat_pts_10);
    const CellFeatures& cell_10 = grid.at(10, 10);
    EXPECT_NEAR(cell_10.risk, 0.25f, 0.1f); // risk = 1 - slope_penalty = 1 - 0.75 = 0.25
    //slope = 20
    auto flat_pts_20 = make_sloped_points(params, 10, 10, 20, 20.0f);
    grid.compute(flat_pts_20);
    const CellFeatures& cell_20 = grid.at(10, 10);
    EXPECT_NEAR(cell_20.risk, 1.0f, 0.1f); // risk = 1 - slope_penalty = 1 - 0.0 = 1.0
}

TEST(Traversability, UnknownCells) {
    // TODO: Cell with 0 points
    // ASSERT: risk = 0.0, confidence = 0.0
    // Cell with 1 point (below min_points_per_cell)
    // ASSERT: risk = 0.0, confidence = 0.0
    GridParams params = test_grid_params();
    VehicleKinematics vehicle = test_vehicle();
    TraversabilityGrid grid(params, vehicle);
    // Cell with 0 points
    grid.compute({}); // empty point cloud
    const CellFeatures& cell_empty = grid.at(10, 10);
    EXPECT_NEAR(cell_empty.risk, 0.0f, 0.1f);
    EXPECT_NEAR(cell_empty.confidence, 0.0f, 0.1f);
    // Cell with 1 point
    auto one_pt = make_flat_points(params, 10, 10, 1);
    grid.compute(one_pt);
    const CellFeatures& cell_one_pt = grid.at(10, 10);
    EXPECT_NEAR(cell_one_pt.risk, 0.0f, 0.1f);
    EXPECT_NEAR(cell_one_pt.confidence, 0.0f, 0.1f);
}

TEST(Traversability, ConfidenceNearFar) {
    // TODO: Cells at range 2m with 20 points -> confidence > 0.8
    // Cells at range 25m with 5 points -> confidence < 0.5
    // Empty cells -> confidence = 0.0
    GridParams params = test_grid_params();
    VehicleKinematics vehicle = test_vehicle();
    TraversabilityGrid grid(params, vehicle);
    // Near cell (2m)
    auto near_pts = make_flat_points(params, 2, 30, 20); // cell (2,30) is at x= -5 + 2*0.5 = -4m
    grid.compute(near_pts);
    const CellFeatures& cell_near = grid.at(2, 30);
    EXPECT_GT(cell_near.confidence, 0.8f);
    // Far cell (25m)
    auto far_pts = make_flat_points(params, 59, 10, 5); // cell (59,10) is at x= -5 + 59*0.5 = 24.5m
    grid.compute(far_pts);
    const CellFeatures& cell_far = grid.at(59, 10);
    EXPECT_LT(cell_far.confidence, 0.5f);
    // Empty cell
    grid.compute({}); // empty point cloud
    const CellFeatures& cell_empty = grid.at(10, 10);
    EXPECT_NEAR(cell_empty.confidence, 0.0f, 0.1f);
}

TEST(Traversability, RoughSurface) {
    // TODO: Create noisy surface with high z variance
    // ASSERT: roughness > max_roughness_tolerance
    // ASSERT: risk penalized (> 0.0)
    GridParams params = test_grid_params();
    VehicleKinematics vehicle = test_vehicle();
    TraversabilityGrid grid(params, vehicle);
    auto rough_pts = make_rough_points(params, 10, 10, 20); // cell (10,10) with high roughness
    grid.compute(rough_pts);
    const CellFeatures& cell = grid.at(10, 10);
    EXPECT_GT(cell.roughness, vehicle.max_roughness_tolerance);
    EXPECT_GT(cell.risk, 0.0f);
}
