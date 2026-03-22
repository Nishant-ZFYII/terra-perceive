// tracker_node.cpp
// ROS2 wrapper for SORT multi-object tracker.
//
// YOUR TASK:
//   - Subscribe to /detections (from YOLO node)
//   - Call SORTTracker::update() each frame
//   - Publish /tracked_objects with IDs, positions, velocities
//   - Publish visualization markers (bounding boxes + ID labels + velocity arrows)
//
// PDF reference: Part 5

#include "sort_tracker.hpp"
// TODO: implement ROS2 node
