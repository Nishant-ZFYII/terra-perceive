#!/bin/bash
# download_data.sh — Download RELLIS-3D sequence 00 for development.
#
# YOUR TASK:
#   1. Download LiDAR point clouds (.bin format)
#   2. Download camera images
#   3. Download calibration files
#   4. Download semantic annotations
#   5. Place in data/ directory (gitignored)
#
# Source: github.com/unmannedlab/RELLIS-3D
# Paper: Jiang et al., "RELLIS-3D" (RA-L 2021)

set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"
mkdir -p "$DATA_DIR"

echo "=== RELLIS-3D Data Download ==="
echo "Target directory: $DATA_DIR"
echo ""
echo "TODO: Add download commands after reviewing RELLIS-3D repo structure."
echo "See: https://github.com/unmannedlab/RELLIS-3D"
echo ""
echo "Expected structure:"
echo "  data/"
echo "    RELLIS-3D/"
echo "      Rellis_3D_pcd_raw/         # LiDAR point clouds"
echo "      Rellis-3D/                  # Camera images"
echo "      Rellis_3D_os1_cloud_node_color_ply/  # Colored point clouds"
echo "      calibration/                # Extrinsic + intrinsic matrices"
