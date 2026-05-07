#!/bin/bash
#SBATCH --job-name=tp_m4_baseline_s1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --account=torch_pr_742_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
# (l40s_public is the only partition that worked on this cluster;
#  GPU is unused for this 2D matplotlib animation but the partition demands one.)

# =============================================================================
# Stride-1 render of the M4 baseline 3-panel animation.
#
# Inputs assumed already on $SCRATCH (rsync'd from laptop before submit):
#   $REPO/results_m4/ablation_g/sort_m4_baseline/tracks.csv  (~979 distinct)
#   $EXT_ROOT/clusters_sweetspot/clusters_*.csv              (K=1 detections)
#   $EXT_ROOT/extracted_frames_camera/                       (RGB)
#   $REPO/data/extracted_frames_full/                        (LiDAR PCDs)
#
# Submit from $REPO with:
#   sbatch --export=ALL,TP_M4_EXT_ROOT=/scratch/np3129/m4_perframe \
#       slurm/render_m4_baseline_stride1.slurm
# =============================================================================
set -euo pipefail
module purge
module load anaconda3/2025.06

ENV_PATH=$SCRATCH/conda_envs/terra_perceive_m13
source $(conda info --base)/etc/profile.d/conda.sh
source activate $ENV_PATH
export PATH=$ENV_PATH/bin:$PATH

REPO=$SCRATCH/terra-perceive-p2m4
EXT_ROOT=${TP_M4_EXT_ROOT:-$SCRATCH/m4_perframe}
RENDER_OUT=$REPO/results_m4/ablation_g/blog_renders
mkdir -p $RENDER_OUT

cd $REPO

LIDAR_DIR=$REPO/data/extracted_frames_full
CAM_DIR=$EXT_ROOT/extracted_frames_camera
TRACKS_CSV=$REPO/results_m4/ablation_g/sort_m4_baseline/tracks.csv

if [[ ! -f "$TRACKS_CSV" ]]; then
    echo "ERROR: $TRACKS_CSV missing." >&2
    echo "       Generate locally first via tracker_runner and rsync." >&2
    exit 1
fi

LAST_FRAME=$(find "$EXT_ROOT/clusters_sweetspot" -maxdepth 1 -name 'clusters_*.csv' \
    | sed -E 's/.*clusters_([0-9]+)\.csv/\1/' | sort -n | tail -1 | sed 's/^0*//')
LAST_FRAME=${LAST_FRAME:-2848}
echo "[m4-baseline-render] last frame: $LAST_FRAME, stride=1"

python3 scripts/animate_tracker_vs_dbscan.py \
    --lidar-dir      "$LIDAR_DIR" \
    --camera-dir     "$CAM_DIR" \
    --clusters-dir   "$EXT_ROOT/clusters_sweetspot" \
    --tracks-csv     "$TRACKS_CSV" \
    --frame-start    0 \
    --frame-end      "$LAST_FRAME" \
    --fps            10 \
    --stride         1 \
    --out-mp4        "$RENDER_OUT/01_m4_baseline_979_stride1.mp4" \
    --out-gif        "$RENDER_OUT/01_m4_baseline_979_stride1.gif"

echo
ls -la $RENDER_OUT/01_m4_baseline_979_stride1.*
echo "[m4-baseline-render] DONE"
