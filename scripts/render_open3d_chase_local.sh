#!/bin/bash
# =============================================================================
# Local Open3D 4K chase-camera render. Drop-in replacement for the HPC slurm
# job that keeps failing on EGL init in batch context.
#
# Renders 2849 frames @ 3840x2160 stride=1. Expected wall time on laptop:
# 5-8 hours depending on GPU/CPU. Per-frame PNGs are cached, so if you have
# to kill it and resume, just re-run this script — frames already written
# are skipped.
#
# Usage:
#   bash scripts/render_open3d_chase_local.sh                 # default config: tracks_k3_eps07
#   TP_TRACKS=tracks_k3_eps05 bash scripts/render_open3d_chase_local.sh
#   TP_TRACKS=tracks_mahal_v2_per_mode bash scripts/render_open3d_chase_local.sh
#
# Before running an overnight batch, follow feedback_overnight_batch_safety.md:
#   sudo systemctl stop unattended-upgrades.timer
#   sudo systemctl mask unattended-upgrades.timer
#   echo -1000 | sudo tee /proc/$$/oom_score_adj
# =============================================================================
set -euo pipefail

REPO=/home/nishant/MS_Project/terra-perceive-p2m4
EXT_ROOT=/media/nishant/SeeGayt2/terra_perceive/m4_perframe
PYTHON=/home/nishant/anaconda3/envs/foundation_stereo/bin/python

TRACKS_NAME=${TP_TRACKS:-tracks_k3_eps07}

# Resolve clusters dir + tracks csv from config name
case "$TRACKS_NAME" in
    tracks_k3_eps05)
        CLUSTERS_DIR=$EXT_ROOT/clusters_k3
        TRACKS_CSV=$REPO/results_m4/ablation_g/sort_on_rellis/tracks_k3_eps05.csv
        ;;
    tracks_k3_eps07)
        CLUSTERS_DIR=$EXT_ROOT/clusters_k3_eps07
        TRACKS_CSV=$REPO/results_m4/ablation_g/sort_on_rellis/tracks_k3_eps07.csv
        ;;
    tracks_mahal_v2_per_mode|tracks_mahal_v1_combined|tracks_mahal_v3_chi228|tracks_fixB_only|tracks_fixC_only)
        CLUSTERS_DIR=$EXT_ROOT/clusters_sweetspot
        TRACKS_CSV=$REPO/results_m4/ablation_g/sort_on_rellis/${TRACKS_NAME}.csv
        ;;
    tracks)
        CLUSTERS_DIR=$EXT_ROOT/clusters_sweetspot
        TRACKS_CSV=$REPO/results_m4/ablation_g/sort_m4_baseline/tracks.csv
        TRACKS_NAME=m4_baseline_979
        ;;
    *)
        echo "ERROR: unknown TP_TRACKS=$TRACKS_NAME" >&2
        exit 1
        ;;
esac

LIDAR_DIR=$REPO/data/extracted_frames_full
RENDER_OUT=$REPO/results_m4/ablation_g/blog_renders
FRAMES_OUT=/tmp/render_frames_open3d_${TRACKS_NAME}
OUT_MP4=$RENDER_OUT/open3d_chase_${TRACKS_NAME}_4k.mp4

# Sanity checks before kicking off a multi-hour run
for p in "$REPO" "$EXT_ROOT" "$LIDAR_DIR" "$CLUSTERS_DIR" "$TRACKS_CSV"; do
    if [[ ! -e "$p" ]]; then
        echo "ERROR: missing path: $p" >&2
        exit 1
    fi
done
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: $PYTHON not found / not executable" >&2
    exit 1
fi

mkdir -p "$RENDER_OUT" "$FRAMES_OUT"

LAST_FRAME=$(find "$LIDAR_DIR" -maxdepth 1 -name '*.bin' \
    | sed -E 's|.*/([0-9]+)\.bin|\1|' | sort -n | tail -1 | sed 's/^0*//')
LAST_FRAME=${LAST_FRAME:-2848}

echo "============================================================"
echo "[local-render] config       = $TRACKS_NAME"
echo "[local-render] tracks csv   = $TRACKS_CSV"
echo "[local-render] clusters dir = $CLUSTERS_DIR"
echo "[local-render] lidar dir    = $LIDAR_DIR"
echo "[local-render] frames cache = $FRAMES_OUT"
echo "[local-render] output mp4   = $OUT_MP4"
echo "[local-render] last frame   = $LAST_FRAME (stride=1)"
echo "[local-render] resolution   = 3840x2160 @ 10 fps"
echo "[local-render] python       = $PYTHON"
echo "============================================================"
echo "Starting at $(date)"
START=$SECONDS

cd "$REPO"

"$PYTHON" scripts/animate_tracker_3d_chase.py \
    --lidar-dir       "$LIDAR_DIR" \
    --clusters-dir    "$CLUSTERS_DIR" \
    --tracks-csv      "$TRACKS_CSV" \
    --frame-start     0 \
    --frame-end       "$LAST_FRAME" \
    --stride          1 \
    --width           3840 \
    --height          2160 \
    --fps             10 \
    --cam-x   50  --cam-y  -2  --cam-z  12 \
    --look-x -25  --look-y   0  --look-z   0 \
    --up-x     0  --up-y     0  --up-z    1 \
    --fov-deg 50 \
    --out-mp4         "$OUT_MP4" \
    --out-frames-dir  "$FRAMES_OUT"

ELAPSED=$((SECONDS - START))
echo
echo "[local-render] DONE at $(date)  (elapsed: ${ELAPSED}s = $((ELAPSED/3600))h$((ELAPSED%3600/60))m)"
ls -la "$OUT_MP4"
