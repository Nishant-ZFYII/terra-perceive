#!/bin/bash
# setup_env.sh — One-command environment setup for terra-perceive.
# Run this on a fresh machine to get everything working.

set -euo pipefail

echo "=== Terra Perceive Environment Setup ==="

# 1. Conda environment
if conda env list | grep -q "terra-perceive"; then
    echo "[OK] Conda env 'terra-perceive' already exists."
else
    echo "[SETUP] Creating conda environment..."
    conda env create -f environment.yml
fi

echo "[INFO] Activate with: conda activate terra-perceive"

# 2. ROS2 sourcing
if [ -f /opt/ros/humble/setup.bash ]; then
    echo "[OK] ROS2 Humble found."
else
    echo "[WARN] ROS2 Humble not found at /opt/ros/humble. Install it."
fi

# 3. Pre-commit hooks
echo "[SETUP] Installing pre-commit hooks..."
pre-commit install

# 4. Proto compilation
echo "[SETUP] Compiling protobuf schemas..."
protoc --python_out=transport/ transport/proto/*.proto 2>/dev/null || \
    echo "[WARN] protoc failed. Install protobuf compiler."

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. conda activate terra-perceive"
echo "  2. source /opt/ros/humble/setup.bash"
echo "  3. colcon build"
echo "  4. bash scripts/download_data.sh"
