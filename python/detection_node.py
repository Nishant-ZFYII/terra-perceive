"""
detection_node.py — YOLOv8 object detection ROS2 node.

This is a LIBRARY CALL — you are NOT reimplementing YOLO.
Your value: fine-tuning on domain data + pipeline integration.

YOUR TASK:
  1. Fine-tune YOLOv8n on Roboflow construction safety dataset
  2. Subscribe to /camera/image_raw
  3. Run inference via ultralytics API
  4. Publish /detections (Detection2DArray or custom msg)
  5. Include 3D localization: for each bbox, find LiDAR points
     that project within it, take median depth

PDF reference: Part 7.2 + Part 6 (Day 6)
Train ref: docs.ultralytics.com/modes/train/
"""
