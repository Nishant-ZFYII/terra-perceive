"""
ros2_bridge.py — Bridge NATS subjects to/from ROS2 topics.

YOUR TASK:
  1. Subscribe to NATS subjects
  2. Republish as ROS2 messages for RViz/Nav2 visualization
  3. Subscribe to ROS2 topics (e.g., /cmd_vel from Nav2)
  4. Republish on NATS for safety supervisor

This keeps ROS2 as an adapter, not the core transport.

PDF reference: Addendum, Section 4
"""
