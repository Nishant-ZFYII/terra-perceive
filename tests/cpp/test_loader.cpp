//test_loader.cpp — Tests for point cloud loading and basic properties.
#include <gtest/gtest.h>
#include "point_cloud_loader.hpp"
#include <limits>
#include <algorithm>

TEST(PointCloudLoader, BasicSanityCheck){
    const std::string test_file = "data/RELLIS-3D/Rellis_3D_lidar_example/os1_cloud_node_kitti_bin/000000.bin";
    auto points = load_bin(test_file);

    //from the python run from visulaize_pointcloud.py, we know that this file has 131072 points and x/y/z ranges.
    
    //assert_eq(a,b) checks if a==b 
    ASSERT_EQ(points.size(), 131072);

    /*From the ranges we get from visualize_pointcloud.py, we know the expected ranges for x, y, and z.
      We can add assertions to check if the loaded points fall within these ranges.
      This ensures that the loader isn't corrupting or misreading the binary data. 
      Give some slack so that the numbers are not stric*/
    
    float x_min = std::numeric_limits<float>::max();
    float y_min = std::numeric_limits<float>::max();
    float x_max = std::numeric_limits<float>::lowest();
    float y_max = std::numeric_limits<float>::lowest();
    float z_max = std::numeric_limits<float>::lowest();
    float z_min = std::numeric_limits<float>::max();

    for (const auto& pt: points){
        x_min = std::min(x_min, pt.x());
        y_min = std::min(y_min, pt.y());
        z_min = std::min(z_min, pt.z());
        x_max = std::max(x_max, pt.x());
        y_max = std::max(y_max, pt.y());
        z_max = std::max(z_max, pt.z());
    }

    //the ranges have to be within a limit
    EXPECT_GE(x_min, -51.0f);
    EXPECT_LE(x_max, 95.0f);
    EXPECT_GE(y_min, -60.0f);
    EXPECT_LE(y_max, 60.0f);
    EXPECT_GE(z_min, -4.0f);
    EXPECT_LE(z_max, 8.0f);


}