// safety_node.cpp
// ROS2 safety supervisor node — independent cmd_vel filter.
//
// YOUR TASK:
//   - Subscribe to /cmd_vel, /tracked_objects, /traversability_grid
//   - 50 Hz WallTimer
//   - Apply priority-ordered rules (P0 > P1 > P2)
//   - Publish /cmd_vel_safe
//   - Publish /safety_events (custom message or std_msgs::String JSON)
//   - Self-monitor loop latency, warn if > 50ms
//
// PDF reference: Part 6
// Consider: make this a lifecycle node for extra professionalism.

#include "safety_supervisor.hpp"
// TODO: implement ROS2 node
