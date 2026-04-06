#!/usr/bin/env bash
# run_cartographer_offline.sh — Run Cartographer 3D SLAM on RELLIS-3D rosbags.
#
# Uses the unmannedlab/cartographer fork which has pre-tuned configs:
#   - warthog_3d_ouster.lua (Ouster OS1-64 + VectorNav VN-300)
#   - offline_warthog_3d_ouster.launch
#   - warthog.urdf.xacro (correct sensor TFs from paper Fig. 1)
#
# Usage:
#   bash scripts/run_cartographer_offline.sh [bag_dir] [output_dir]
#
# Defaults:
#   bag_dir   = data/RELLIS-3D
#   output_dir = data

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BAG_DIR="${1:-$REPO_ROOT/data/RELLIS-3D}"
OUTPUT_DIR="${2:-$REPO_ROOT/data}"

# Verify bags exist
for i in 00 01 02 03 04; do
  BAG="$BAG_DIR/00000_${i}.bag"
  if [ ! -f "$BAG" ]; then
    echo "ERROR: Bag not found: $BAG"
    exit 1
  fi
done

# Build comma-separated bag list for Cartographer (Docker paths)
BAG_FILES="/data/00000_00.bag,/data/00000_01.bag,/data/00000_02.bag,/data/00000_03.bag,/data/00000_04.bag"

echo "=== Cartographer Offline SLAM (unmannedlab fork) ==="
echo "Bags: $BAG_DIR/00000_0{0..4}.bag"
echo "Output: $OUTPUT_DIR"
echo ""

# Build the Cartographer Docker image (if not already built)
echo "[1/3] Building Cartographer Docker image (20-40 min first time)..."
docker compose -f "$REPO_ROOT/docker/docker-compose.yml" \
  --profile cartographer build cartographer

# Run offline SLAM
echo "[2/3] Running offline SLAM (10-30 minutes)..."
START_SEC=$(date +%s)

docker compose -f "$REPO_ROOT/docker/docker-compose.yml" \
  --profile cartographer run --rm \
  -v "$BAG_DIR:/data" \
  -v "$OUTPUT_DIR:/output" \
  -v "$REPO_ROOT/docker/offline_headless.launch:/catkin_ws/offline_headless.launch:ro" \
  cartographer \
  roslaunch /catkin_ws/offline_headless.launch \
    "config_path:=/catkin_ws/cart_config/config" \
    "bag_filename:=/data/00000_00.bag,/data/00000_01.bag,/data/00000_02.bag,/data/00000_03.bag,/data/00000_04.bag"

END_SEC=$(date +%s)
ELAPSED=$((END_SEC - START_SEC))
echo "SLAM completed in ${ELAPSED}s"

# Find the .pbstream file
PBSTREAM=$(find "$OUTPUT_DIR" "$BAG_DIR" -name "*.pbstream" -newer "$REPO_ROOT/scripts/run_cartographer_offline.sh" -print -quit 2>/dev/null || true)

if [ -n "$PBSTREAM" ]; then
  echo "[3/3] .pbstream produced: $PBSTREAM ($(du -h "$PBSTREAM" | cut -f1))"
  # Copy to output directory if not already there
  if [ "$(dirname "$PBSTREAM")" != "$OUTPUT_DIR" ]; then
    cp "$PBSTREAM" "$OUTPUT_DIR/"
    echo "Copied to $OUTPUT_DIR/"
  fi
else
  echo "WARNING: No .pbstream file found. Cartographer may have failed."
  echo "Check Docker logs above for errors."
  exit 1
fi

echo ""
echo "Next step: extract poses with:"
echo "  python3 scripts/extract_poses_cartographer.py --pbstream $OUTPUT_DIR/$(basename "${PBSTREAM:-output.pbstream}") --out data/poses_carto.csv"
