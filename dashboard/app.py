"""
app.py — Streamlit ops dashboard for construction perception stack.

PDF reference: Part 9 (page 20)

YOUR TASK — Build 6 panels:
  1. Camera View: Annotated RGB with boxes, IDs, distances
  2. BEV Traversability: Green/yellow/red grid + robot + tracked objects
  3. Safety Event Timeline: Scrolling log of stop/slow events
  4. Session Metrics: Detections, stops, uptime, area covered, idle %
  5. Sensor Health: LiDAR Hz, camera Hz, per-stage latency
  6. Operator Escalation: Red banner + Acknowledge button on P0 events

Data source: NATS (telemetry.events + perception.traversability.grid)
             or ROS2 topics via rosbridge (adapter mode)
"""
