// test_projection.cpp — Tests for camera-LiDAR projection.
//
// YOUR TASK:
//   1. Known T and K -> project a point at (10, 0, 0) in LiDAR frame
//   2. Verify pixel coordinates match hand-calculated expected values
//   3. Test behind-camera rejection (negative depth)
//   4. Test out-of-bounds rejection
//
// PDF reference: Part 4

#include <gtest/gtest.h>

#include "cam_lidar_projection.hpp"
#include <Eigen/Dense>

//identity T and K, projecting a point directly on the optical axis.
const Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
const Eigen::Matrix3f K = (Eigen::Matrix3f() << 100.0f, 0.0f, 320.0f,
                                             0.0f, 100.0f, 240.0f,
                                             0.0f,   0.0f,   1.0f).finished();

const int image_width = 640;
const int image_height = 480;



TEST(CamLidarProjection, KnownProjection) {
    //test point directly ahead
    CamLidarProjector proj(T,K,image_width,image_height);
    Eigen::Vector3f pt(0.0f, 0.0f, 10.0f); 

    Eigen::Vector2f pixel;
    bool visible = proj.project_point(pt, pixel);

    EXPECT_TRUE(visible);
    EXPECT_NEAR(pixel[0], 320.0f, 1.0f);
    EXPECT_NEAR(pixel[1], 240.0f, 1.0f);
}

TEST(CamLidarProjection, BehindCamera) {
    CamLidarProjector proj(T,K,image_width,image_height);
    Eigen::Vector3f pt(0.0f, 0.0f, -10.0f); 

    Eigen::Vector2f pixel;
    bool visible = proj.project_point(pt, pixel);

    EXPECT_FALSE(visible);
}

TEST(CamLidarProjection, OutOfBounds) {
    CamLidarProjector proj(T,K,image_width,image_height);
    Eigen::Vector3f pt(1000.0f, 1000.0f, 10.0f); 

    Eigen::Vector2f pixel;
    bool visible = proj.project_point(pt, pixel);

    EXPECT_FALSE(visible);
    
}
