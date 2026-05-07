#!/bin/bash
#SBATCH --job-name=tp_open3d_chase
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --account=torch_pr_742_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
# Open3D EGL headless rendering needs a real GPU; this is the partition we have.

# =============================================================================
# 4K Open3D chase-camera 3D render of the tracker output, stride=1.
#
# Pick which tracker config to render via the TP_TRACKS env var:
#   TP_TRACKS=tracks_k3_eps07          (default; combined knee, 272 distinct)
#   TP_TRACKS=tracks_k3_eps05          (detector knee, 307 distinct)
#   TP_TRACKS=tracks_mahal_v2_per_mode (tracker knee, 242 distinct)
#   TP_TRACKS=tracks                   (sort_m4_baseline/tracks.csv — M4 baseline)
#
# Submit from $REPO with:
#   sbatch --export=ALL,TP_M4_EXT_ROOT=/scratch/np3129/m4_perframe,TP_TRACKS=tracks_k3_eps07 \
#       slurm/render_open3d_chase.slurm
#
# Rerunnable: per-frame PNGs are cached. If the job times out at 12 hours,
# resubmitting picks up where it left off.
#
# FIRST-RUN ONLY: install Open3D into terra_perceive_m13 if not already there.
# Run interactively before submitting this batch job. ALL THREE redirects below
# are required — pip's TMPDIR fallback to $HOME has filled the home quota twice
# (2026-04-29, 2026-04-30) on this exact install.
#
#   ssh torch
#   module purge && module load anaconda3/2025.06
#   source $(conda info --base)/etc/profile.d/conda.sh
#   source activate /scratch/np3129/conda_envs/terra_perceive_m13
#   export CONDA_PKGS_DIRS=/scratch/np3129/conda_pkgs
#   export PIP_CACHE_DIR=/scratch/np3129/pip_cache
#   export TMPDIR=/scratch/np3129/tmp
#   mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $TMPDIR
#   rm -rf $HOME/.cache/pip   # clear any partial download from a prior failure
#   pip install --no-cache-dir open3d==0.19.0
#   python -c "import open3d as o3d; print(o3d.__version__)"
#
# If Open3D fails on EGL (runtime error mentions OSMesa or libGL):
#   mamba install -y -c conda-forge libegl libgl
# =============================================================================
set -euo pipefail
module purge
module load anaconda3/2025.06

ENV_PATH=$SCRATCH/conda_envs/terra_perceive_m13
source $(conda info --base)/etc/profile.d/conda.sh
source activate $ENV_PATH
export PATH=$ENV_PATH/bin:$PATH

export OPEN3D_HEADLESS=1
export DISPLAY=
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

REPO=$SCRATCH/terra-perceive-p2m4
EXT_ROOT=${TP_M4_EXT_ROOT:-$SCRATCH/m4_perframe}
TRACKS_NAME=${TP_TRACKS:-tracks_k3_eps07}
RENDER_OUT=$REPO/results_m4/ablation_g/blog_renders
FRAMES_OUT=$SCRATCH/render_frames_open3d_${TRACKS_NAME}
mkdir -p $RENDER_OUT $FRAMES_OUT

if [[ "$TRACKS_NAME" == "tracks" ]]; then
    TRACKS_CSV=$REPO/results_m4/ablation_g/sort_m4_baseline/tracks.csv
    TRACKS_LABEL=m4_baseline_979
    CLUSTERS_DIR=$EXT_ROOT/clusters_sweetspot
else
    TRACKS_CSV=$REPO/results_m4/ablation_g/sort_on_rellis/${TRACKS_NAME}.csv
    TRACKS_LABEL=$TRACKS_NAME
    case "$TRACKS_NAME" in
        tracks_k3_eps05)         CLUSTERS_DIR=$EXT_ROOT/clusters_k3 ;;
        tracks_k3_eps07)         CLUSTERS_DIR=$EXT_ROOT/clusters_k3_eps07 ;;
        tracks_mahal_v2_per_mode|tracks_mahal_v1_combined|tracks_mahal_v3_chi228|tracks_fixB_only|tracks_fixC_only)
                                 CLUSTERS_DIR=$EXT_ROOT/clusters_sweetspot ;;
        *)                       CLUSTERS_DIR=$EXT_ROOT/clusters_sweetspot ;;
    esac
fi

if [[ ! -f "$TRACKS_CSV" ]]; then
    echo "ERROR: $TRACKS_CSV missing." >&2
    exit 1
fi
if [[ ! -d "$CLUSTERS_DIR" ]]; then
    echo "ERROR: $CLUSTERS_DIR missing." >&2
    exit 1
fi

cd $REPO

LIDAR_DIR=$REPO/data/extracted_frames_full
LAST_FRAME=$(find "$LIDAR_DIR" -maxdepth 1 -name '*.bin' \
    | sed -E 's|.*/([0-9]+)\.bin|\1|' | sort -n | tail -1 | sed 's/^0*//')
LAST_FRAME=${LAST_FRAME:-2848}
echo "[open3d-chase] config=$TRACKS_LABEL  last frame=$LAST_FRAME"
echo "               tracks=$TRACKS_CSV"
echo "               clusters=$CLUSTERS_DIR"
echo "               frames-out=$FRAMES_OUT"

# Diagnostic: confirm EGL/GPU visibility from inside the batch job before launch
echo "[diag] node=$(hostname)"
echo "[diag] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi 2>&1 | head -8 || echo "[diag] nvidia-smi missing"
echo "[diag] __EGL_VENDOR_LIBRARY_FILENAMES=$__EGL_VENDOR_LIBRARY_FILENAMES"
echo "[diag] __GLX_VENDOR_LIBRARY_NAME=$__GLX_VENDOR_LIBRARY_NAME"
ls /usr/share/glvnd/egl_vendor.d/ 2>&1
ls -la /usr/lib64/libEGL_nvidia* 2>&1 | head
echo "[diag] running OffscreenRenderer with env forced inline:"
env __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    python3 -c "import open3d as o3d; r=o3d.visualization.rendering.OffscreenRenderer(64,64); print('[diag] EGL OK', r)" \
    || { echo "[diag] EGL init failed in batch context — aborting render"; exit 2; }

env __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    python3 scripts/animate_tracker_3d_chase.py \
    --lidar-dir       "$LIDAR_DIR" \
    --clusters-dir    "$CLUSTERS_DIR" \
    --tracks-csv      "$TRACKS_CSV" \
    --frame-start     0 \
    --frame-end       "$LAST_FRAME" \
    --stride          1 \
    --width           3840 \
    --height          2160 \
    --fps             10 \
    --out-mp4         "$RENDER_OUT/open3d_chase_${TRACKS_LABEL}_4k.mp4" \
    --out-frames-dir  "$FRAMES_OUT"

echo
ls -la $RENDER_OUT/open3d_chase_${TRACKS_LABEL}_4k.mp4
echo "[open3d-chase] DONE"
