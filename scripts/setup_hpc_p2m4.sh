#!/bin/bash
# ============================================================================
# One-time HPC setup for Terra Perceive M4 on NYU Torch cluster.
# Mirrors the conda-prefix-on-$SCRATCH pattern from setup_hpc.sh in
# Nishant-ZFYII/ml_inference.
#
# Run this interactively on a login node AFTER scripts/sync_to_hpc.sh has
# pushed the repo to $HOME/terra-perceive-p2m4:
#
#   ssh <NetID>@login.torch.hpc.nyu.edu
#   cd $HOME/terra-perceive-p2m4
#   bash scripts/setup_hpc_p2m4.sh
#
# What it does:
#   1. Creates conda prefix env at $SCRATCH/conda_envs/terra_perceive_m4.
#   2. Installs ROS Humble (via robostack-style conda-forge), Eigen, fmt,
#      gtest, cmake, ffmpeg, plus python deps (rosbags, numpy, matplotlib).
#   3. Clones tinycolormap + stb header-only deps into $SCRATCH/third_party
#      and symlinks them into the repo as third_party/.
#
# Build time: ~10-15 min on a login node (mostly conda solver work).
# ============================================================================

set -euo pipefail

ENV_NAME=terra_perceive_m4
ENV_PATH=$SCRATCH/conda_envs/${ENV_NAME}
REPO_DIR=$HOME/terra-perceive-p2m4
THIRD_PARTY=$SCRATCH/third_party

echo "=== NYU Torch HPC setup for terra-perceive-p2m4 ==="

# ── Step 1: Set up conda directories in $SCRATCH ───────────────────────────
mkdir -p $SCRATCH/conda_envs $SCRATCH/conda_pkgs $THIRD_PARTY $SCRATCH/m4_perframe

# ── Step 2: Load anaconda module ───────────────────────────────────────────
module purge
module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh

# ── Step 3: Create prefix environment (idempotent) ─────────────────────────
if [ -d "$ENV_PATH" ]; then
    echo "Environment already exists at $ENV_PATH"
    echo "  (to recreate: rm -rf $ENV_PATH && bash $0)"
else
    echo "Creating conda environment at $ENV_PATH ..."
    conda create -p $ENV_PATH python=3.11 -y
fi

# ── Step 4: Activate ───────────────────────────────────────────────────────
source activate $ENV_PATH
export PATH=$ENV_PATH/bin:$PATH
export PYTHONNOUSERSITE=True

# ── Step 5: Install build toolchain + ROS Humble + C++ libs ────────────────
# Robostack ships ROS Humble on conda-forge. Channel order matters: put
# robostack-staging first so its ROS packages are picked over conda-forge's
# (sometimes-older) variants.
echo "Configuring conda channels (robostack-staging + conda-forge)..."
conda config --env --prepend channels conda-forge
conda config --env --prepend channels robostack-staging
conda config --env --set channel_priority strict

echo "Installing ROS Humble and C++ build tools (this is the slow step) ..."
conda install -y \
    ros-humble-ament-cmake \
    ros-humble-ament-cmake-gtest \
    ros-humble-ament-cmake-core \
    ros-humble-rclcpp \
    ros-humble-sensor-msgs \
    ros-humble-nav-msgs \
    ros-humble-geometry-msgs \
    ros-humble-visualization-msgs \
    colcon-common-extensions \
    cmake make compilers \
    eigen=3.4 \
    fmt \
    gtest \
    ffmpeg

# ── Step 6: Python deps for the pipeline ───────────────────────────────────
echo "Installing python deps (rosbags, scipy, Pillow) ..."
pip install --no-cache-dir \
    rosbags \
    scipy \
    Pillow

# numpy and matplotlib ship with conda above — but install via pip if missing
python -c "import numpy, matplotlib" 2>/dev/null || pip install numpy matplotlib

# ── Step 7: Header-only third-party libs ──────────────────────────────────
echo "Cloning header-only deps (tinycolormap, stb) into $THIRD_PARTY ..."
cd $THIRD_PARTY
[ -d tinycolormap ] || git clone --depth 1 https://github.com/yuki-koyama/tinycolormap.git
[ -d stb ]          || git clone --depth 1 https://github.com/nothings/stb.git

# Symlink into the repo so existing CMakeLists paths resolve.
cd "$REPO_DIR"
if [ ! -e third_party ]; then
    ln -s "$THIRD_PARTY" third_party
    echo "Linked: $REPO_DIR/third_party -> $THIRD_PARTY"
fi

echo
echo "=== Setup complete ==="
echo "Environment : $ENV_PATH"
echo "Third-party : $THIRD_PARTY"
echo "M4 outputs  : $SCRATCH/m4_perframe"
echo
echo "To use interactively:"
echo "  module purge && module load anaconda3/2025.06"
echo "  source \$(conda info --base)/etc/profile.d/conda.sh"
echo "  source activate $ENV_PATH"
echo "  export PATH=$ENV_PATH/bin:\$PATH"
echo "  export PYTHONNOUSERSITE=True"
echo
echo "To submit Ablation G:"
echo "  cd $REPO_DIR"
echo "  sbatch slurm/run_ablation_g.slurm"
echo
echo "IMPORTANT: before first sbatch, verify your account in train.slurm:"
echo "  groups | tr ' ' '\\n' | grep torch_   # find your account name"
echo "  sinfo                                 # find an available CPU partition"
echo "Then update --account and --partition in slurm/run_ablation_g.slurm."
