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

TEST(CamLidarProjection, KnownProjection) {
    // TODO: implement
}

TEST(CamLidarProjection, BehindCamera) {
    // TODO: implement
}

TEST(CamLidarProjection, OutOfBounds) {
    // TODO: implement
}
