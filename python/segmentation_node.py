"""
segmentation_node.py — SegFormer semantic segmentation ROS2 node.

This is a LIBRARY CALL — you are NOT reimplementing SegFormer.
Your value: fine-tuning on RELLIS-3D + choosing the right architecture.

YOUR TASK:
  1. Fine-tune SegFormer MIT-B0 on RELLIS-3D (20 terrain classes)
  2. Subscribe to /camera/image_raw
  3. Run inference via HuggingFace transformers
  4. Publish /semantic_img (uint8 label image)
  5. Export to ONNX for Jetson deployment

PDF reference: Part 7.1
Train ref: huggingface.co/docs/transformers/model_doc/segformer
Dataset: RELLIS-3D (Jiang et al., RA-L 2021)
"""
