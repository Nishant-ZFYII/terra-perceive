#!/usr/bin/env bash
# run_ablation_g.sh — DBSCAN sweep on RELLIS obstacle clouds, full sequence.
#
# Question (rule #2 — predict before running):
#   "Where's the sweet spot for clustering RELLIS obstacle clouds, and how
#    does the chosen parameter set look across the full driving sequence?"
#
# Pipeline:
#   Stage 0: extract_all_bags.sh extracts all 5 RELLIS bags into one
#            contiguous-numbered directory on the external drive (one-time).
#   Stage 1: obstacle_extractor runs sector RANSAC on every frame to dump
#            obstacles_NNNNNN.csv per frame. Writes to external drive.
#   Stage 2: 3 × 3 sweep over (eps, min_points) on a SINGLE demo frame for
#            the parameter-grid figure.
#   Stage 3: full-sequence DBSCAN at the sweet-spot params for the animation.
#   Stage 4: render the 3×3 thumbnail grid (single PNG).
#   Stage 5: render the full-sequence animation MP4 + GIF.
#
# Predicted outcomes:
#   eps=0.3, mp=20  → over-tight: real objects fragment, many noise points
#   eps=1.0, mp=5   → over-loose: distinct objects merge into mega-clusters
#   eps=0.5, mp=10  → likely sweet spot for the M4 blog story
#
# Per ablation pre-flight rules:
#   #3 overnight safety: full pipeline ~30-60 min. Disable
#                        unattended-upgrades BEFORE launching:
#                          sudo systemctl stop unattended-upgrades.service \
#                                              apt-daily-upgrade.timer apt-daily.timer
#   #5 heavy data on external drive (results_m4 holds JSON + final renders only)
#   #6 resumable per-cell AND per-frame
#   #10 qual pair: 3×3 grid PNG + full-sequence MP4 + GIF
#   #11 calibrate: stage 1 prints first/last frame timing — extrapolate before
#                  letting stage 3 run on all frames

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

# Heavy outputs live off the SSD on laptop, on /scratch on HPC. Override
# via env var (TP_M4_EXT_ROOT=/scratch/$USER/m4_perframe in the Slurm job).
EXT_ROOT="${TP_M4_EXT_ROOT:-/media/nishant/SeeGayt2/terra_perceive/m4_perframe}"
OBSTACLES_DIR="${EXT_ROOT}/obstacles"
SWEETSPOT_CLUSTERS_DIR="${EXT_ROOT}/clusters_sweetspot"

# Lightweight outputs (PNG, MP4, JSON) stay in the repo for git tracking.
OUT_ROOT="results_m4/ablation_g"
LIDAR_DIR="data/extracted_frames_full"   # symlink set up by extract_all_bags.sh
EXTRACTOR="./build/construction_perception/obstacle_extractor"
CLI="./build/construction_perception/dbscan_cli"
PYTHON="${PYTHON:-python3}"

# Sweet-spot params for the full-sequence animation. Tuneable based on grid result.
SWEET_EPS=0.5
SWEET_MP=10

# Demo frame for the 3×3 parameter grid figure. Pick a mid-sequence frame
# with visible clutter; tune after Stage 1 prints frame counts.
DEMO_FRAME=500

mkdir -p "${OUT_ROOT}" "${OBSTACLES_DIR}" "${SWEETSPOT_CLUSTERS_DIR}"

# -----------------------------------------------------------------------------
# Stage 0 — extract all 5 bags into one contiguous-numbered dir
# -----------------------------------------------------------------------------
if [[ ! -L "${LIDAR_DIR}" ]]; then
    echo "[ablation_g] running extract_all_bags.sh (all 5 bags)"
    bash scripts/extract_all_bags.sh
else
    echo "[ablation_g] reusing existing ${LIDAR_DIR}"
fi
TOTAL_FRAMES=$(ls "${LIDAR_DIR}"/*.bin 2>/dev/null | wc -l)
LAST_FRAME=$(( TOTAL_FRAMES - 1 ))
echo "[ablation_g] total frames available: ${TOTAL_FRAMES} (ids 0..${LAST_FRAME})"

# -----------------------------------------------------------------------------
# Stage 1 — extract obstacle clouds (resumable: skip if metrics.json exists)
# -----------------------------------------------------------------------------
if [[ ! -f "${OBSTACLES_DIR}/metrics.json" ]]; then
    echo "[ablation_g] running obstacle_extractor on all ${TOTAL_FRAMES} frames"
    "${EXTRACTOR}" \
        --lidar       "${LIDAR_DIR}" \
        --frame-start 0 \
        --frame-end   "${LAST_FRAME}" \
        --ransac-dist 0.15 \
        --ransac-iter 200 \
        --ransac-min-inliers 50 \
        --sector-size 5.0 \
        --out         "${OBSTACLES_DIR}" \
        --verbose
else
    echo "[ablation_g] reusing cached obstacle CSVs in ${OBSTACLES_DIR}"
fi

# Sanity: confirm the demo frame's CSV exists.
DEMO_OBSTACLES="${OBSTACLES_DIR}/obstacles_$(printf '%06d' "${DEMO_FRAME}").csv"
if [[ ! -f "${DEMO_OBSTACLES}" ]]; then
    echo "[ablation_g] ERROR: demo frame obstacle CSV not produced: ${DEMO_OBSTACLES}"
    echo "[ablation_g] check obstacle_extractor --verbose log for failures"
    exit 1
fi

# -----------------------------------------------------------------------------
# Stage 2 — DBSCAN sweep on the SINGLE demo frame (3 × 3 = 9 cells)
# -----------------------------------------------------------------------------
EPS_VALUES=(0.3 0.5 1.0)
MP_VALUES=(5 10 20)

for eps in "${EPS_VALUES[@]}"; do
    for mp in "${MP_VALUES[@]}"; do
        cell_dir="${OUT_ROOT}/eps_${eps}_mp_${mp}"
        cell_csv="${cell_dir}/clusters_$(printf '%06d' "${DEMO_FRAME}").csv"
        if [[ -f "${cell_csv}" ]]; then
            echo "[ablation_g] ✅ grid eps=${eps} mp=${mp}: already done"
            continue
        fi
        mkdir -p "${cell_dir}"
        echo "[ablation_g] running grid eps=${eps} min_points=${mp}"
        "${CLI}" \
            --in         "${DEMO_OBSTACLES}" \
            --eps        "${eps}" \
            --min-points "${mp}" \
            --out        "${cell_csv}"
    done
done

# -----------------------------------------------------------------------------
# Stage 3 — full-sequence DBSCAN at sweet-spot params (resumable per frame)
# -----------------------------------------------------------------------------
echo "[ablation_g] running full-sequence DBSCAN at eps=${SWEET_EPS} mp=${SWEET_MP}"
processed=0
skipped=0
for fid in $(seq 0 "${LAST_FRAME}"); do
    pad=$(printf '%06d' "${fid}")
    obs_csv="${OBSTACLES_DIR}/obstacles_${pad}.csv"
    out_csv="${SWEETSPOT_CLUSTERS_DIR}/clusters_${pad}.csv"
    if [[ ! -f "${obs_csv}" ]]; then
        # Frame may be missing if obstacle_extractor skipped it.
        continue
    fi
    if [[ -f "${out_csv}" ]]; then
        skipped=$(( skipped + 1 ))
        continue
    fi
    "${CLI}" \
        --in         "${obs_csv}" \
        --eps        "${SWEET_EPS}" \
        --min-points "${SWEET_MP}" \
        --out        "${out_csv}" >/dev/null
    processed=$(( processed + 1 ))
    if (( processed % 200 == 0 )); then
        echo "[ablation_g]   ... processed ${processed} new frames"
    fi
done
echo "[ablation_g] full-sequence DBSCAN: ${processed} new, ${skipped} cached"

# -----------------------------------------------------------------------------
# Stage 4 — render 3×3 parameter grid (single PNG, the headline blog asset)
# -----------------------------------------------------------------------------
echo "[ablation_g] rendering 3×3 parameter grid"
"${PYTHON}" scripts/plot_dbscan_grid.py \
    --root         "${OUT_ROOT}" \
    --eps          "${EPS_VALUES[@]}" \
    --min-points   "${MP_VALUES[@]}" \
    --frame        "${DEMO_FRAME}" \
    --out          "${OUT_ROOT}/dbscan_grid.png"

# -----------------------------------------------------------------------------
# Stage 5 — render full-sequence DBSCAN animation MP4 + GIF
# -----------------------------------------------------------------------------
echo "[ablation_g] rendering full-sequence DBSCAN animation"
"${PYTHON}" scripts/dbscan_animate.py \
    --clusters-dir "${SWEETSPOT_CLUSTERS_DIR}" \
    --frame-start  0 \
    --frame-end    "${LAST_FRAME}" \
    --eps          "${SWEET_EPS}" \
    --min-points   "${SWEET_MP}" \
    --fps          10 \
    --out-mp4      "${OUT_ROOT}/dbscan_animation.mp4" \
    --out-gif      "${OUT_ROOT}/dbscan_animation.gif" \
    --gif-stride   5     # subsample to keep GIF size manageable

# -----------------------------------------------------------------------------
# Stage 6 — extract Pylon camera frames synced to LiDAR (resumable per bag)
# -----------------------------------------------------------------------------
CAM_DIR="${EXT_ROOT}/extracted_frames_camera"
if [[ -z "$(ls -A "${CAM_DIR}" 2>/dev/null)" ]]; then
    echo "[ablation_g] running extract_all_cameras.sh"
    bash scripts/extract_all_cameras.sh
else
    echo "[ablation_g] reusing existing camera frames in ${CAM_DIR}"
fi

# -----------------------------------------------------------------------------
# Stage 7 — render the perception-pipeline triptych (camera | LiDAR | DBSCAN)
# -----------------------------------------------------------------------------
echo "[ablation_g] rendering 3-panel perception triptych"
"${PYTHON}" scripts/dbscan_animate_triptych.py \
    --lidar-dir       "${LIDAR_DIR}" \
    --camera-dir      "${CAM_DIR}" \
    --obstacles-dir   "${OBSTACLES_DIR}" \
    --clusters-dir    "${SWEETSPOT_CLUSTERS_DIR}" \
    --frame-start     0 \
    --frame-end       "${LAST_FRAME}" \
    --eps             "${SWEET_EPS}" \
    --min-points      "${SWEET_MP}" \
    --fps             10 \
    --stride          5 \
    --out-mp4         "${OUT_ROOT}/triptych.mp4" \
    --out-gif         "${OUT_ROOT}/triptych.gif"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo
echo "================ Ablation G summary ================"
printf "  %-8s  %-12s  %-10s  %-10s\n" "eps" "min_points" "K_clusters" "noise"
for eps in "${EPS_VALUES[@]}"; do
    for mp in "${MP_VALUES[@]}"; do
        cell_csv="${OUT_ROOT}/eps_${eps}_mp_${mp}/clusters_$(printf '%06d' "${DEMO_FRAME}").csv"
        if [[ -f "${cell_csv}" ]]; then
            stats=$(awk -F, 'NR>1 {
                if ($4 == -1) noise++;
                else clusters[$4]=1;
            } END {
                k = 0; for (c in clusters) k++;
                printf "%d %d", k, (noise+0);
            }' "${cell_csv}")
            k=$(echo "${stats}" | awk '{print $1}')
            n=$(echo "${stats}" | awk '{print $2}')
            printf "  %-8s  %-12s  %-10s  %-10s\n" "${eps}" "${mp}" "${k}" "${n}"
        fi
    done
done
echo
echo "  Grid PNG  : ${OUT_ROOT}/dbscan_grid.png"
echo "  Animation : ${OUT_ROOT}/dbscan_animation.{mp4,gif}  (single-panel)"
echo "  Triptych  : ${OUT_ROOT}/triptych.{mp4,gif}          (camera | LiDAR | DBSCAN)"
echo "===================================================="
