// cam_lidar_projection.cpp
// See include/cam_lidar_projection.hpp for interface and instructions.

#include "cam_lidar_projection.hpp"

CamLidarProjector::CamLidarProjector(const Eigen::Matrix4f& T_cam_lidar,
                                     const Eigen::Matrix3f& K, int img_width,
                                     int img_height)
    : T_cam_lidar_(T_cam_lidar), K_(K), img_width_(img_width), img_height_(img_height) {}

std::vector<int> CamLidarProjector::project_and_label(
    const std::vector<Eigen::Vector3f>& lidar_pts, const uint8_t* semantic_img) {
    //lookup semantic_img
    std::vector<int> labels;
    labels.reserve(lidar_pts.size());
    //go through each lidar point
    for(const auto& pt : lidar_pts){
        Eigen::Vector2f pixel = Eigen::Vector2f::Zero();
        if(project_point(pt, pixel)){
            int u = static_cast<int>(pixel[0]);
            int v = static_cast<int>(pixel[1]);

            //for evey pixel get the label
            int label = semantic_img[v * img_width_ + u];
            labels.push_back(label);
        } else {
            labels.push_back(-1);

        }
    }
    return labels;
}

bool CamLidarProjector::project_point(const Eigen::Vector3f& pt_lidar,
                                      Eigen::Vector2f& pixel_out) const {
    // TODO: implement single-point projection
    // P_cam = T_cam_lidar * [x, y, z, 1]^T
    // p = K * P_cam[:3]

    //pt_lidar -> 4x1 [x,y,z,1] and T_cam_lidar is 4x4 -> P_cam is 4x1
    Eigen::Vector4f P_cam = T_cam_lidar_ * Eigen::Vector4f(pt_lidar[0], pt_lidar[1], pt_lidar[2], 1.0f);
    
    //depth check if P_cam.z <=0 return false

    if(P_cam[2] <= 0){
        return false;
    }

    //flatten P_cam from 3d to 2d using K -> calibration matrix. 
    //K is 3x3 and P_cam 4x1. 
    Eigen::Vector3f p = K_ * P_cam.head<3>();

    //get pixel coordinates

    float u = p[0] / p[2];
    float v = p[1] /p[2];

    //dimension check, so that we capture only points that are visible to both cam and lidar
    //check against the width and height of the image. 

    if(u < 0 || u >= img_width_ || v < 0 || v >= img_height_){
        return false;
    }
    pixel_out[0] = u;
    pixel_out[1] = v;
    
    return true;
    
}
