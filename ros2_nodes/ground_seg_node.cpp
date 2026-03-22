// ground_seg_node.cpp
// ROS2 wrapper for RANSAC ground segmentation.
//
// YOUR TASK:
//   - Subscribe to /lidar/points (sensor_msgs::msg::PointCloud2)
//   - Convert PointCloud2 -> vector<Eigen::Vector3f>
//   - Call segment_ground()
//   - Publish ground + obstacle clouds as separate PointCloud2 topics
//   - Publish PlaneModel as custom message or diagnostic
//
// PDF reference: Part 2 + Part 9 (ROS2 nodes)

#include "ransac_ground_seg.hpp"
// TODO: implement ROS2 node
