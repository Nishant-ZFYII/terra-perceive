// test_ransac.cpp — Tests for RANSAC ground segmentation.
// Covers both global RANSAC and sector-based RANSAC.
//
// P1-M2 checkpoints:
//   M2.1: Synthetic flat plane — normal ~(0,0,1), inlier_ratio ~0.8
//   M2.2: SVD refinement improves normal accuracy
//   M2.4: Sector RANSAC handles multi-slope terrain
//   M2.5: Continuity check flags wildly different sectors

#include <algorithm>
#include <cmath>
#include <random>

#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>

#include "ransac_ground_seg.hpp"

namespace {

constexpr float kPi = 3.14159265358979323846f;

float degrees_to_radians(float degrees) {
    return degrees * kPi / 180.0f;
}

float angle_between(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {                                                      
      if (a.norm() == 0.0f || b.norm() == 0.0f) return 0.0f;                                                                     
      float cos_angle = a.dot(b) / (a.norm() * b.norm());
      cos_angle = std::clamp(cos_angle, -1.0f, 1.0f);                                                                            
      float angle = std::acos(cos_angle);
      return std::min(angle, kPi - angle);  // handle sign ambiguity                                                             
}      

Eigen::Vector3f expected_flat_normal() {
    return Eigen::Vector3f::UnitZ();
}

Eigen::Vector3f expected_tilted_normal(float slope_deg) {
    float slope_rad = degrees_to_radians(slope_deg);
    float m = std::tan(slope_rad);
    Eigen::Vector3f normal(-m, 0.0f, 1.0f);
    return normal.normalized();
}

std::vector<Eigen::Vector3f> sample_plane(float slope_deg, int points, float noise_sigma,
                                           std::mt19937& gen, float x_min = -10.0f,
                                           float x_max = 10.0f, float y_min = -5.0f,
                                           float y_max = 5.0f) {
    std::uniform_real_distribution<float> x_dist(x_min, x_max);
    std::uniform_real_distribution<float> y_dist(y_min, y_max);
    std::normal_distribution<float> noise_dist(0.0f, noise_sigma);
    float slope_rad = degrees_to_radians(slope_deg);
    float tan_slope = std::tan(slope_rad);
    std::vector<Eigen::Vector3f> out;
    out.reserve(points);

    for (int i = 0; i < points; ++i) {
        float x = x_dist(gen);
        float y = y_dist(gen);
        float z = tan_slope * x + noise_dist(gen);
        out.emplace_back(x, y, z);
    }

    return out;
}

void add_outliers(std::vector<Eigen::Vector3f>& cloud, int count, std::mt19937& gen) {
    std::uniform_real_distribution<float> xy_dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> z_dist(1.0f, 3.0f);
    for (int i = 0; i < count; ++i) {
        float x = xy_dist(gen);
        float y = xy_dist(gen);
        float z = z_dist(gen);
        cloud.emplace_back(x, y, z);
    }
}

Eigen::Vector3f compute_pca_normal(const std::vector<Eigen::Vector3f>& points) {
    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    for (const auto& pt : points) {
        mean += pt;
    }
    mean /= static_cast<float>(points.size());

    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    for (const auto& pt : points) {
        Eigen::Vector3f centered = pt - mean;
        cov += centered * centered.transpose();
    }
    cov /= static_cast<float>(points.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
    return solver.eigenvectors().col(0).normalized();
}

std::vector<Eigen::Vector3f> build_multi_slope_cloud(std::mt19937& gen, float slope_deg,
                                                     int per_sector) {
    auto left = sample_plane(0.0f, per_sector, 0.02f, gen, -10.0f, -0.1f);
    auto right = sample_plane(slope_deg, per_sector, 0.02f, gen, 0.1f, 10.0f);
    left.insert(left.end(), right.begin(), right.end());
    return left;
}

}  // namespace

// --- Global RANSAC tests (P1-M2.1, P1-M2.2) ---

TEST(RansacGroundSeg, SyntheticFlatPlane) {
    std::mt19937 gen(42);
    std::vector<Eigen::Vector3f> cloud = sample_plane(0.0f, 800, 0.02f, gen);
    add_outliers(cloud, 200, gen);

    RANSACParams params;
    params.distance_threshold = 0.05f;
    params.max_iterations = 200;
    params.min_inliers = 300;

    auto result = segment_ground(cloud, params);

    ASSERT_GT(result.ground_points.size(), 0u);
    EXPECT_GT(result.inlier_ratio, 0.7f);
    EXPECT_NEAR(0.0f, result.ground_plane.normal.x(), 0.05f);
    EXPECT_NEAR(0.0f, result.ground_plane.normal.y(), 0.05f);
    EXPECT_NEAR(1.0f, std::abs(result.ground_plane.normal.z()), 0.05f);

    for (const auto& obs : result.obstacle_points) {
        if (obs.z() > 0.3f) {
            EXPECT_GT(obs.z(), 0.5f);
        }
    }
}

TEST(RansacGroundSeg, TiltedPlane30Deg) {
    std::mt19937 gen(1);
    std::vector<Eigen::Vector3f> cloud = sample_plane(30.0f, 1000, 0.02f, gen);

    RANSACParams params;
    params.distance_threshold = 0.05f;
    params.max_iterations = 200;
    params.min_inliers = 400;

    auto result = segment_ground(cloud, params);
    Eigen::Vector3f expected = expected_tilted_normal(30.0f);

    float angle = angle_between(result.ground_plane.normal, expected);
    EXPECT_LT(angle, degrees_to_radians(5.0f));
    EXPECT_GT(result.ground_points.size(), 0u);
}

TEST(RansacGroundSeg, SVDRefinement) {
    std::mt19937 gen(24);
    std::vector<Eigen::Vector3f> cloud = sample_plane(0.0f, 1200, 0.05f, gen);
    add_outliers(cloud, 300, gen);

    RANSACParams params;
    params.distance_threshold = 0.06f;
    params.max_iterations = 250;
    params.min_inliers = 400;

    auto result = segment_ground(cloud, params);
    ASSERT_GT(result.ground_points.size(), 0u);

    Eigen::Vector3f pca_normal = compute_pca_normal(result.ground_points);
    float angle = angle_between(result.ground_plane.normal, pca_normal);

    EXPECT_LT(angle, degrees_to_radians(3.0f));
}

TEST(RansacGroundSeg, DegenerateInput) {
    std::vector<Eigen::Vector3f> empty_cloud;
    RANSACParams params;
    params.max_iterations = 10;

    auto result = segment_ground(empty_cloud, params);
    EXPECT_TRUE(result.ground_points.empty());
    EXPECT_TRUE(result.obstacle_points.empty());

    std::vector<Eigen::Vector3f> collinear = {
        Eigen::Vector3f(0.0f, 0.0f, 0.0f),
        Eigen::Vector3f(1.0f, 0.0f, 0.0f),
        Eigen::Vector3f(2.0f, 0.0f, 0.0f),
        Eigen::Vector3f(3.0f, 0.0f, 0.0f)
    };
    auto collinear_result = segment_ground(collinear, params);
    EXPECT_TRUE(collinear_result.ground_points.empty());
}

// --- Sector RANSAC tests (P1-M2.4, P1-M2.5) ---

TEST(SectorRansac, MultiSlopeTerrain) {
    std::mt19937 gen(12);
    auto cloud = build_multi_slope_cloud(gen, 15.0f, 1500);

    RANSACParams ransac_params;
    ransac_params.distance_threshold = 0.05f;
    ransac_params.max_iterations = 150;

    SectorParams sector_params;

    auto result = sector_segment_ground(cloud, ransac_params, sector_params);
    EXPECT_GE(result.num_reliable_sectors, 2);

    bool saw_flat = false;
    bool saw_tilted = false;
    Eigen::Vector3f flat = expected_flat_normal();
    Eigen::Vector3f tilted = expected_tilted_normal(15.0f);

    for (const auto& plane : result.sector_planes) {
        if (!plane.reliable) {
            continue;
        }
        float angle_to_flat = angle_between(plane.plane.normal, flat);
        float angle_to_tilted = angle_between(plane.plane.normal, tilted);
        if (angle_to_flat < degrees_to_radians(10.0f)) {
            saw_flat = true;
        }
        if (angle_to_tilted < degrees_to_radians(10.0f)) {
            saw_tilted = true;
        }
    }

    EXPECT_TRUE(saw_flat);
    EXPECT_TRUE(saw_tilted);
}

TEST(SectorRansac, ContinuityCheckPass) {
    std::mt19937 gen(99);
    auto cloud = build_multi_slope_cloud(gen, 10.0f, 1200);
    RANSACParams ransac_params;
    ransac_params.distance_threshold = 0.05f;
    ransac_params.max_iterations = 150;
    SectorParams sector_params;
    sector_params.continuity_angle_thresh_deg = 30.0f;

    auto result = sector_segment_ground(cloud, ransac_params, sector_params);
    EXPECT_GE(result.num_reliable_sectors, 2);
}

TEST(SectorRansac, ContinuityCheckFail) {
    std::mt19937 gen(123);
    auto cloud = build_multi_slope_cloud(gen, 60.0f, 1200);
    RANSACParams ransac_params;
    ransac_params.distance_threshold = 0.05f;
    ransac_params.max_iterations = 150;
    SectorParams sector_params;
    sector_params.continuity_angle_thresh_deg = 30.0f;

    auto result = sector_segment_ground(cloud, ransac_params, sector_params);
    bool has_unreliable = false;
    for (const auto& plane : result.sector_planes) {
        if (!plane.reliable) {
            has_unreliable = true;
            break;
        }
    }
    EXPECT_TRUE(has_unreliable);
}

TEST(SectorRansac, EmptySector) {
    std::mt19937 gen(77);
    auto cloud = build_multi_slope_cloud(gen, 0.0f, 800);
    // constrain x-range so some sectors stay empty
    for (auto& pt : cloud) {
        pt.x() = std::clamp(pt.x(), -4.0f, 4.0f);
    }

    RANSACParams ransac_params;
    ransac_params.distance_threshold = 0.05f;
    ransac_params.max_iterations = 150;
    SectorParams sector_params;
    sector_params.sector_size_x = 2.0f;                                                                                            
    sector_params.sector_size_y = 2.0f;  

    auto result = sector_segment_ground(cloud, ransac_params, sector_params);
    bool saw_empty = false;
    for (const auto& plane : result.sector_planes) {
        if (!plane.reliable) {
            saw_empty = true;
            break;
        }
    }
    EXPECT_TRUE(saw_empty);
}
