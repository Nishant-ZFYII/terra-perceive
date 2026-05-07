#!/bin/bash
#SBATCH --job-name=tp_m4_render
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --account=torch_pr_742_general
#SBATCH --partition=cpu_normal
# ↑ Verify with: sacctmgr list user $USER format=Account%30 -p
#   and:        sinfo -o "%P %D %c %m" | head

# =============================================================================
# slurm/render_blog_animations.slurm
#
# Renders the four-config Phase-4 comparison animation suite:
#   1. M4 baseline (CV, max_misses=10, no cascade) — regenerated on HPC
#   2. M13.5 Mahal-v2 K=1 (per-mode IMM cov + χ²=5.99, ego anchor, max_age=300)
#   3. M13.5 + Phase-4 K=3 eps=0.5 (detector-knee, zero >20m drift)
#   4. M13.5 + Phase-4 K=3 eps=0.7 (combined-knee, zero >20m drift, recovered distinct)
#
# Each invocation calls animate_tracker_vs_dbscan.py once. Outputs land in
# $RENDER_OUT/<config_name>.{mp4,gif}.
#
# Inputs assumed to exist on $SCRATCH (push from laptop via dtn before submit):
#   $EXT_ROOT/clusters_sweetspot/        (K=1 DBSCAN clusters)
#   $EXT_ROOT/clusters_k3/               (K=3 eps=0.5 clusters)
#   $EXT_ROOT/clusters_k3_eps07/         (K=3 eps=0.7 clusters)
#   $EXT_ROOT/extracted_frames_camera/   (RGB camera frames)
#   $REPO/data/extracted_frames_full/    (LiDAR PCDs/bins)
#   $REPO/data/poses_slam_full.csv       (SLAM ego poses)
#   $REPO/results_m4/ablation_g/sort_on_rellis/tracks_mahal_v2_per_mode.csv
#   $REPO/results_m4/ablation_g/sort_on_rellis/tracks_k3_eps05.csv
#   $REPO/results_m4/ablation_g/sort_on_rellis/tracks_k3_eps07.csv
#   $REPO/results_m4/ablation_g/rellis_detections.csv         (K=1 detections, for M4 baseline run)
#   $REPO/build/construction_perception/tracker_runner        (built on HPC via colcon)
#
# Submit from $REPO with:
#   sbatch slurm/render_blog_animations.slurm
#
# Wall-clock budget: 4 hours. Expected actual ~2 hr at 30 min/config.
# =============================================================================
set -euo pipefail

module purge
module load anaconda3/2025.06

ENV_PATH=$SCRATCH/conda_envs/terra_perceive_m4
source $(conda info --base)/etc/profile.d/conda.sh
source activate $ENV_PATH
export PATH=$ENV_PATH/bin:$PATH

REPO=$SCRATCH/terra-perceive-p2m4
EXT_ROOT=${TP_M4_EXT_ROOT:-$SCRATCH/m4_perframe}
RENDER_OUT=$REPO/results_m4/ablation_g/blog_renders
mkdir -p $RENDER_OUT

cd $REPO

# Determine frame range from the existing K=1 cluster directory.
LAST_FRAME=$(find "$EXT_ROOT/clusters_sweetspot" -maxdepth 1 -name 'clusters_*.csv' \
    | sed -E 's/.*clusters_([0-9]+)\.csv/\1/' | sort -n | tail -1 | sed 's/^0*//')
LAST_FRAME=${LAST_FRAME:-2848}
echo "[render] last frame: $LAST_FRAME"

LIDAR_DIR=$REPO/data/extracted_frames_full
CAM_DIR=$EXT_ROOT/extracted_frames_camera
PYTHON=python3

# -----------------------------------------------------------------------------
# Stage 0 — regenerate M4 baseline tracks (CV filter, no cascade) for the
#           "before" panel of the comparison. ~5 sec.
# -----------------------------------------------------------------------------
M4_BASELINE_DIR=$REPO/results_m4/ablation_g/sort_m4_baseline
M4_BASELINE_TRACKS=$M4_BASELINE_DIR/tracks.csv
if [[ ! -f "$M4_BASELINE_TRACKS" ]]; then
    echo "[render] generating M4 baseline tracks"
    mkdir -p $M4_BASELINE_DIR
    ./build/construction_perception/tracker_runner \
        --detections     $REPO/results_m4/ablation_g/rellis_detections.csv \
        --solver         munkres \
        --filter         cv \
        --max-dist       5.0 \
        --max-misses     10 \
        --max-age        0 \
        --min-hits       1 \
        --dt             0.1 \
        --process-noise  2.0 \
        --meas-noise     0.3 \
        --snapshot-every 0 \
        --out            $M4_BASELINE_DIR \
        --verbose
fi

# -----------------------------------------------------------------------------
# Helper: render one (clusters_dir, tracks_csv, out_basename) tuple.
# -----------------------------------------------------------------------------
render_one() {
    local clusters_dir="$1"
    local tracks_csv="$2"
    local label="$3"
    echo
    echo "=========================================================="
    echo "[render] config: $label"
    echo "         clusters: $clusters_dir"
    echo "         tracks:   $tracks_csv"
    echo "         out:      $RENDER_OUT/$label.{mp4,gif}"
    echo "=========================================================="
    "${PYTHON}" scripts/animate_tracker_vs_dbscan.py \
        --lidar-dir      "$LIDAR_DIR" \
        --camera-dir     "$CAM_DIR" \
        --clusters-dir   "$clusters_dir" \
        --tracks-csv     "$tracks_csv" \
        --frame-start    0 \
        --frame-end      "$LAST_FRAME" \
        --fps            10 \
        --stride         5 \
        --out-mp4        "$RENDER_OUT/$label.mp4" \
        --out-gif        "$RENDER_OUT/$label.gif"
}

# -----------------------------------------------------------------------------
# Render all four configs.
# -----------------------------------------------------------------------------
render_one \
    "$EXT_ROOT/clusters_sweetspot" \
    "$M4_BASELINE_TRACKS" \
    "01_m4_baseline_979"

render_one \
    "$EXT_ROOT/clusters_sweetspot" \
    "$REPO/results_m4/ablation_g/sort_on_rellis/tracks_mahal_v2_per_mode.csv" \
    "02_mahalv2_k1_242"

render_one \
    "$EXT_ROOT/clusters_k3" \
    "$REPO/results_m4/ablation_g/sort_on_rellis/tracks_k3_eps05.csv" \
    "03_k3_eps05_307"

render_one \
    "$EXT_ROOT/clusters_k3_eps07" \
    "$REPO/results_m4/ablation_g/sort_on_rellis/tracks_k3_eps07.csv" \
    "04_k3_eps07_272"

echo
echo "================ blog renders summary ================"
ls -la $RENDER_OUT/*.mp4 $RENDER_OUT/*.gif
echo "======================================================="
