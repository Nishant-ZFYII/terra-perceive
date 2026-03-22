// ransac_ground_seg.hpp
// Ground segmentation using RANSAC — implement from scratch.
// Phase 1 upgrade: SECTOR-BASED RANSAC for off-road terrain with slopes.
//
// PDF reference: Part 2 (pages 9-10) + Critique ("RANSAC assumes flat ground")
// Read: Fischler & Bolles, "Random Sample Consensus" (1981)
// Watch: Cyrill Stachniss, "RANSAC" (YouTube, Photogrammetry II)
// Eigen ref: eigen.tuxfamily.org/dox/group__QuickRefPage.html
//
// YOUR TASK (P1-M2, Days 3-4):
//   P1-M2.1: Implement global segment_ground() first — make it work on flat ground
//   P1-M2.2: Add SVD refinement via SelfAdjointEigenSolver on inliers
//   P1-M2.3: Test on real RELLIS-3D data — observe failure on slopes
//   P1-M2.4: Implement sector_segment_ground() — RANSAC per spatial sector
//   P1-M2.5: Add continuity checks between adjacent sectors
//   P1-M2.6: Test on 5+ real frames, tune parameters
//
// Performance target: < 5ms for 30K points per sector.

#pragma once
#include <Eigen/Dense>
#include <random>
#include <vector>

struct PlaneModel {
    Eigen::Vector3f normal;
    float d;  // ax + by + cz + d = 0
    float distance(const Eigen::Vector3f& pt) const;
};

struct RANSACParams {
    float distance_threshold = 0.15f;
    int max_iterations = 1000;
    float confidence = 0.99f;
    int min_inliers = 100;
};

struct SegmentationResult {
    PlaneModel ground_plane;
    std::vector<Eigen::Vector3f> ground_points;
    std::vector<Eigen::Vector3f> obstacle_points;
    int iterations_used;
    float inlier_ratio;
};

// --- Sector-based RANSAC (Phase 1 upgrade) ---

struct SectorParams {
    float sector_size_x = 5.0f;  // meters per sector in x
    float sector_size_y = 5.0f;  // meters per sector in y
    float continuity_angle_thresh_deg = 30.0f;  // max normal disagreement between neighbors
};

struct SectorPlane {
    PlaneModel plane;
    int sector_ix;
    int sector_iy;
    int num_inliers;
    float inlier_ratio;
    bool reliable;  // false if too few points or failed continuity check
};

struct SectorSegmentationResult {
    std::vector<SectorPlane> sector_planes;
    std::vector<Eigen::Vector3f> ground_points;
    std::vector<Eigen::Vector3f> obstacle_points;
    int num_sectors;
    int num_reliable_sectors;
};

// Global RANSAC — implement first, then observe it fail on slopes
SegmentationResult segment_ground(const std::vector<Eigen::Vector3f>& cloud,
                                  const RANSACParams& params);

// Sector-based RANSAC — the upgrade that handles graded terrain
SectorSegmentationResult sector_segment_ground(const std::vector<Eigen::Vector3f>& cloud,
                                               const RANSACParams& ransac_params,
                                               const SectorParams& sector_params);
