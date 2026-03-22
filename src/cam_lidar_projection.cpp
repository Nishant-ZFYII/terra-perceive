// cam_lidar_projection.cpp
// TODO: Implement camera-LiDAR projection from scratch.
// See include/cam_lidar_projection.hpp for interface and instructions.

#include "cam_lidar_projection.hpp"

CamLidarProjector::CamLidarProjector(const Eigen::Matrix4f& T_cam_lidar,
                                     const Eigen::Matrix3f& K, int img_width,
                                     int img_height)
    : T_cam_lidar_(T_cam_lidar), K_(K), img_width_(img_width), img_height_(img_height) {}

std::vector<int> CamLidarProjector::project_and_label(
    const std::vector<Eigen::Vector3f>& lidar_pts, const uint8_t* semantic_img) {
    // TODO: implement
    // For each point:
    //   1. P_cam = T_cam_lidar * [x, y, z, 1]^T
    //   2. if P_cam.z <= 0: label = -1, continue
    //   3. p = K * P_cam[:3]
    //   4. u = p[0]/p[2], v = p[1]/p[2]
    //   5. bounds check, lookup semantic_img[v * width + u]
    return {};
}

bool CamLidarProjector::project_point(const Eigen::Vector3f& pt_lidar,
                                      Eigen::Vector2f& pixel_out) const {
    // TODO: implement single-point projection
    return false;
}
