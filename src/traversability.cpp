// traversability.cpp
// TODO: Implement traversability grid from scratch.
// See include/traversability.hpp for interface and P1-M3 sub-goals.

#include "traversability.hpp"
#include <iostream>
#include <cmath>
#include <algorithm> 
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846f  // define M_PI if not defined by cmath
#endif

TraversabilityGrid::TraversabilityGrid(const GridParams& params,
                                       const VehicleKinematics& vehicle)
    : params_(params), vehicle_(vehicle) {
    //init grid with default Cell Features
    //constructor body
    grid_.resize(rows(), std::vector<CellFeatures>(cols()));
}

void TraversabilityGrid::compute(const std::vector<Eigen::Vector3f>& ground_pts) {
    // TODO (P1-M3.1): Grid binning — single pass O(N)
    // Reset grid_ to default CellFeatures before binning
    grid_.assign(rows(), std::vector<CellFeatures>(cols()));
    
    //create a temp holder for binning
    std::vector<std::vector<std::vector<Eigen::Vector3f>>> buckets(rows(),
                                                            std::vector<std::vector<Eigen::Vector3f>>(cols()));
    //   For each point: ix = (x - x_min) / resolution, iy = (y - y_min) / resolution
    for (const auto& pt : ground_pts) {
        int ix = static_cast<int>((pt.x() - params_.x_min) / params_.resolution);
        int iy = static_cast<int>((pt.y() - params_.y_min) / params_.resolution);
        if(ix >=0 && ix < rows() && iy >=0 && iy < cols()){
            buckets[ix][iy].push_back(pt);
        }
    }

    for(int ix = 0; ix < rows(); ++ix){
        for(int iy = 0; iy < cols(); ++iy){
            auto& pts = buckets[ix][iy];
            int point_count = pts.size();
            auto& cell = grid_[ix][iy];
            cell.point_count = point_count;
            if (point_count < params_.min_points_per_cell){
                cell.risk = 0.0f;
                cell.confidence = 0.0f;
                continue;
            }

            //roughness = smallest_eigenvalue / sum_of_eigenvalues
            Eigen::Matrix3f C = Eigen::Matrix3f::Zero();
            Eigen::Vector3f mean = Eigen::Vector3f::Zero();
            for (const auto& pt : pts){
                mean += pt;
            }
            mean /= static_cast<float>(point_count);
            for (const auto& pt : pts){
                C += (pt - mean) * (pt - mean).transpose();  // accumulate outer product for covariance
            }
            C /= static_cast<float>(point_count);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(C);
            if (solver.info() != Eigen::Success) {
                std::cerr<<"Eigen decomposition failed for covariance matrix. Setting roughness to 0."<<std::endl;
                cell.roughness = 0.0f;
            } else {
                Eigen::Vector3f eigenvalues = solver.eigenvalues();
                float sum_eigenvalues = eigenvalues.sum();
                cell.roughness = sum_eigenvalues > 0 ? eigenvalues(0) / sum_eigenvalues : 0.0f;
                Eigen::Vector3f normal = solver.eigenvectors().col(0);
                if (normal.z() < 0){
                    normal = -normal;
                }
                cell.slope_deg = std::acos(std::abs(normal.z())) * 180.0f / static_cast<float>(M_PI);
            }
            

            //step_height = max(z) - min(z)
            float min_z = std::numeric_limits<float>::max();
            float max_z = std::numeric_limits<float>::lowest();
            float sum_z = 0.0f;
            float sum_range = 0.0f;
            for (const auto& pt : pts){
                float z = pt.z();
                min_z = std::min(min_z, z);
                max_z = std::max(max_z, z);
                sum_z += z;
                sum_range += std::sqrt(pt.x() * pt.x() + pt.y() * pt.y() + pt.z() * pt.z());
            }
            cell.step_height = max_z - min_z;
            cell.mean_z = sum_z / static_cast<float>(point_count);
            cell.range_from_sensor = sum_range / static_cast<float>(point_count);

            //risk = 1.0 - compute_vehicle_aware_score(slope_deg, roughness, step_height)
            cell.risk = 1.0f - compute_vehicle_aware_score(cell.slope_deg, cell.roughness, cell.step_height);
            cell.confidence = compute_confidence(point_count, cell.range_from_sensor);


        }
    }
}

float TraversabilityGrid::compute_vehicle_aware_score(float slope_deg, float roughness,
                                                      float step_height) {
    // 10.1109/IROS.2009.5354535 — Eq(1) — Vehicle-aware traversability score
    //   Each term penalizes exceeding a physical vehicle limit:
    //   slope_penalty = max(0, 1 - (slope_deg / vehicle_.max_climbable_grade_deg)^2)
    float slope_penalty = std::max<float>(0.0f, 1.0f - std::pow(slope_deg/vehicle_.max_climbable_grade_deg, 2));
    //   roughness_penalty = max(0, 1 - (roughness / vehicle_.max_roughness_tolerance)^2)
    float roughness_penalty = std::max<float>(0.0f, 1.0f - std::pow(roughness/vehicle_.max_roughness_tolerance, 2));
    //   step_penalty = max(0, 1 - (step_height / vehicle_.max_step_height)^2)
    float step_penalty = std::max<float>(0.0f, 1.0f - std::pow(step_height/vehicle_.max_step_height, 2));
    //   score = slope_penalty * roughness_penalty * step_penalty
    float score = slope_penalty * roughness_penalty * step_penalty;

    return score;
}

float TraversabilityGrid::compute_confidence(int point_count, float range_from_sensor) {
    // Confidence based on observation quality
    //   Point count factor: saturates at ~20 points (confidence from count = min(1, count/20))
    float count_factor = std::min<float>(1.0f, static_cast<float>(point_count) / 20.0f);
    //   Range factor: degrades with distance (confidence from range = max(0, 1 - range/30))
    float range_factor = std::max<float>(0.0f, 1.0f - range_from_sensor / 30.0f);
    //   confidence = count_factor * range_factor
    float confidence = count_factor * range_factor;

    return confidence;
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

    //unknown cell: confidence = 0 -> cost = -1
    //hazard: rish high -> cost approaches 100
    //safe: risk low -> cost approaches 0
    std::vector<int8_t> occupancy_grid(rows() * cols(), -1); 
    for (int ix = 0; ix < rows(); ++ix){
        for (int iy = 0; iy < cols(); ++iy){
            const CellFeatures& cell = grid_[ix][iy];
            int idx = ix * cols() + iy;
            if (cell.confidence < 1e-6f){
                occupancy_grid[idx] = -1; // unknown
            } else {
                occupancy_grid[idx] = static_cast<int8_t>(std::round(cell.risk * 100.0f)); // risk [0,1] -> cost [0,100]
            }
            
        }
    }
    return occupancy_grid;
}

Eigen::Vector3f TraversabilityGrid::compute_normal(
    const std::vector<Eigen::Vector3f>& pts) {
    // TODO (P1-M3.2): PCA via covariance eigendecomposition
    // Utility for tests/external callers only — main loop uses solver inline in compute()    
    //   1. Compute mean of pts
    int num_pts = pts.size();
    if (num_pts == 0){
        return Eigen::Vector3f::UnitZ();
    }
    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    for (const auto& pt : pts){
        mean += pt;
    }
    mean /= static_cast<float>(num_pts);
    //Center: subtract mean from each point and transpose to find the Covariance matrix
    Eigen::Matrix3f C = Eigen::Matrix3f::Zero();
    for (const auto& pt : pts){
        C += (pt - mean) * (pt - mean).transpose();  // accumulate outer product for covariance
    }

    // normalize by number of points
    C /= static_cast<float>(num_pts);  
    //Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(C)

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(C);
    if (solver.info() != Eigen::Success) {
        // Handle decomposition failure. We return default normal
        //throw an error
        std::cerr<<"Eigen decomposition failed for covariance matrix. Returning default normal."<<std::endl;
        return Eigen::Vector3f::UnitZ();
    }
    //Return eigenvectors().col(0) — smallest eigenvalue's eigenvector
    
    return solver.eigenvectors().col(0);
}
