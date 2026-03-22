// ransac_ground_seg.cpp
// TODO: Implement RANSAC ground segmentation from scratch.
// See include/ransac_ground_seg.hpp for interface and P1-M2 sub-goals.

#include "ransac_ground_seg.hpp"

float PlaneModel::distance(const Eigen::Vector3f& pt) const {
    // TODO (P1-M2.1): |normal.dot(pt) + d|
    return 0.0f;
}

SegmentationResult segment_ground(const std::vector<Eigen::Vector3f>& cloud,
                                  const RANSACParams& params) {
    SegmentationResult result;
    // TODO (P1-M2.1): Global RANSAC — implement first, then observe it fail on slopes
    //
    // Step 1: Random 3-point sampling with std::mt19937
    // Step 2: Plane fitting: n = (p2-p1).cross(p3-p1), normalize, d = -n.dot(p1)
    // Step 3: Degenerate check: if cross product norm < 1e-6, skip
    // Step 4: Inlier counting: |n.dot(pt) + d| < distance_threshold
    // Step 5: Track best model (most inliers)
    // Step 6: Early termination if inlier_ratio > 0.8
    // Step 7: Iteration count: N = log(1-p) / log(1-(1-e)^s)
    //
    // TODO (P1-M2.2): SVD refinement on final inlier set
    //   Center inliers (subtract mean), form 3xN matrix M
    //   Covariance C = (1/N) * M * M^T
    //   SelfAdjointEigenSolver -> eigenvector of smallest eigenvalue = refined normal
    return result;
}

SectorSegmentationResult sector_segment_ground(const std::vector<Eigen::Vector3f>& cloud,
                                               const RANSACParams& ransac_params,
                                               const SectorParams& sector_params) {
    SectorSegmentationResult result;
    // TODO (P1-M2.4): Sector-based RANSAC
    //
    // Step 1: Compute point cloud x/y bounds
    // Step 2: Divide into sectors of sector_size_x * sector_size_y
    // Step 3: Assign each point to its sector (single pass O(N))
    // Step 4: For each sector with enough points, run segment_ground()
    // Step 5: Store SectorPlane for each sector
    //
    // TODO (P1-M2.5): Continuity check
    //   For each pair of adjacent sectors, compute angle between normals
    //   If angle > continuity_angle_thresh_deg, flag the sector with fewer inliers as unreliable
    //
    // Step 6: Classify all points: ground if within local sector plane, else obstacle
    return result;
}
