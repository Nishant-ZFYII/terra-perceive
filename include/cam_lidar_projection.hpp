// cam_lidar_projection.hpp
// Camera-LiDAR projection using extrinsic + intrinsic matrices — from scratch.
//

//
// YOUR TASK:
//   1. Transform LiDAR point to camera frame via T_cam_lidar
//   2. Depth check (z > 0)
//   3. Project to pixel via K (intrinsics)
//   4. Bounds check against image dimensions
//   5. Semantic label lookup from SegFormer output
//
// No OpenCV projectPoints — direct Eigen matrix math only.

#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <vector>

class CamLidarProjector {
   public:
    CamLidarProjector(const Eigen::Matrix4f& T_cam_lidar, const Eigen::Matrix3f& K,
                      int img_width, int img_height);

    std::vector<int> project_and_label(const std::vector<Eigen::Vector3f>& lidar_pts,
                                       const uint8_t* semantic_img);

    bool project_point(const Eigen::Vector3f& pt_lidar, Eigen::Vector2f& pixel_out) const;

   private:
    Eigen::Matrix4f T_cam_lidar_;
    Eigen::Matrix3f K_;
    int img_width_;
    int img_height_;
};
