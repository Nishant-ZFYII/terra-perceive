// traversability.cpp
// TODO: Implement traversability grid from scratch.
// See include/traversability.hpp for interface and P1-M3 sub-goals.

#include "traversability.hpp"

TraversabilityGrid::TraversabilityGrid(const GridParams& params,
                                       const VehicleKinematics& vehicle)
    : params_(params), vehicle_(vehicle) {
    // TODO (P1-M3.1): Initialize grid dimensions from params
    // rows = (x_max - x_min) / resolution, cols = (y_max - y_min) / resolution
}

void TraversabilityGrid::compute(const std::vector<Eigen::Vector3f>& ground_pts) {
    // TODO (P1-M3.1): Grid binning — single pass O(N)
    //   For each point: ix = (x - x_min) / resolution, iy = (y - y_min) / resolution
    //   Push point into cell bucket
    //
    // TODO (P1-M3.2): PCA surface normals per cell
    //   For each cell with >= min_points_per_cell:
    //     Center points (subtract mean)
    //     Form 3xN matrix M, compute C = (1/N) * M * M^T
    //     SelfAdjointEigenSolver -> eigenvector of smallest eigenvalue = normal
    //
    // TODO (P1-M3.3): Feature computation
    //   slope_deg = acos(|n.z()|) * 180 / M_PI
    //   roughness = smallest_eigenvalue / sum_of_eigenvalues
    //   step_height = max(z) - min(z)
    //
    // TODO (P1-M3.4): Vehicle-aware scoring
    //   score = compute_vehicle_aware_score(slope_deg, roughness, step_height)
    //
    // TODO (P1-M3.5): Confidence
    //   confidence = compute_confidence(point_count, range_from_sensor)
    //   Unknown cells: score=0.5, confidence=0.0
}

float TraversabilityGrid::compute_vehicle_aware_score(float slope_deg, float roughness,
                                                      float step_height) {
    // TODO (P1-M3.4): Physics-grounded scoring
    //   Each term penalizes exceeding a physical vehicle limit:
    //   slope_penalty = max(0, 1 - (slope_deg / vehicle_.max_climbable_grade_deg)^2)
    //   roughness_penalty = max(0, 1 - (roughness / vehicle_.max_roughness_tolerance)^2)
    //   step_penalty = max(0, 1 - (step_height / vehicle_.max_step_height)^2)
    //   score = slope_penalty * roughness_penalty * step_penalty
    return 0.5f;
}

float TraversabilityGrid::compute_confidence(int point_count, float range_from_sensor) {
    // TODO (P1-M3.5): Confidence based on observation quality
    //   Point count factor: saturates at ~20 points (confidence from count = min(1, count/20))
    //   Range factor: degrades with distance (confidence from range = max(0, 1 - range/30))
    //   confidence = count_factor * range_factor
    return 0.0f;
}

const CellFeatures& TraversabilityGrid::at(int ix, int iy) const {
    return grid_[ix][iy];
}

int TraversabilityGrid::rows() const {
    return static_cast<int>((params_.x_max - params_.x_min) / params_.resolution);
}

int TraversabilityGrid::cols() const {
    return static_cast<int>((params_.y_max - params_.y_min) / params_.resolution);
}

std::vector<int8_t> TraversabilityGrid::to_occupancy_grid() const {
    // TODO: Map traversability [0,1] to OccupancyGrid int8 [0, 100]
    //   score 1.0 (safe) -> cost 0 (free), score 0.0 (hazard) -> cost 100 (lethal)
    //   unknown (confidence=0) -> -1 (unknown)
    return {};
}

Eigen::Vector3f TraversabilityGrid::compute_normal(
    const std::vector<Eigen::Vector3f>& pts) {
    // TODO (P1-M3.2): PCA via covariance eigendecomposition
    //   1. Compute mean of pts
    //   2. Center: subtract mean from each point
    //   3. Form 3x3 covariance matrix C
    //   4. Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(C)
    //   5. Return eigenvector corresponding to smallest eigenvalue
    return Eigen::Vector3f::UnitZ();
}
