#!/bin/bash
# run_pipeline.sh — End-to-end Phase 1 pipeline runner.
# P1-M6.1: Chains all components on a single RELLIS-3D frame.
#
# Usage: bash scripts/run_pipeline.sh [path_to_bin] [path_to_image]
#
# YOUR TASK:
#   1. Load point cloud
#   2. Run sector RANSAC -> ground/obstacle split
#   3. Run traversability grid -> BEV scores
#   4. Run cam-LiDAR projection + SegFormer -> fused BEV
#   5. Run safety supervisor on mock scenarios -> intervention log
#   6. Output: BEV images + safety CSV + timing report

set -euo pipefail

BIN_FILE="${1:-data/RELLIS-3D/sample_frame.bin}"
IMG_FILE="${2:-data/RELLIS-3D/sample_frame.png}"
OUTPUT_DIR="output/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

echo "=== Terra Perceive — Phase 1 Pipeline ==="
echo "Input LiDAR: $BIN_FILE"
echo "Input Image: $IMG_FILE"
echo "Output dir:  $OUTPUT_DIR"
echo ""

# TODO: Call each component and save outputs
# Step 1: Sector RANSAC
# Step 2: Traversability grid + BEV image
# Step 3: Camera-LiDAR projection + fusion
# Step 4: Safety supervisor mock scenarios
# Step 5: Timing report

echo "=== Pipeline complete ==="
echo "Results in: $OUTPUT_DIR"
