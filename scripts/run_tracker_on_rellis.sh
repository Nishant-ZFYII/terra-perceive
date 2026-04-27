#!/usr/bin/env bash
# run_tracker_on_rellis.sh — final M4 closing-hero pipeline.
#
# Inputs:  per-frame DBSCAN cluster CSVs (cached from run_ablation_g.sh stage 3)
# Outputs: side-by-side animation showing the same scene with FLICKERING
#          DBSCAN cluster IDs vs STABLE SORT track IDs.
#
# Pipeline:
#   1. clusters_to_detections.py  → rellis_detections.csv
#                                  (one row per cluster centroid per frame)
#   2. tracker_runner             → sort_on_rellis/tracks.csv
#                                  (one row per published track per frame)
#   3. animate_tracker_vs_dbscan  → sort_vs_dbscan.{mp4,gif}
#                                  (3-panel: camera | DBSCAN | SORT)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

EXT_ROOT="${TP_M4_EXT_ROOT:-/media/nishant/SeeGayt2/terra_perceive/m4_perframe}"
CLUSTERS_DIR="${EXT_ROOT}/clusters_sweetspot"
LIDAR_DIR="data/extracted_frames_full"
CAM_DIR="${EXT_ROOT}/extracted_frames_camera"
OBSTACLES_DIR="${EXT_ROOT}/obstacles"

OUT_ROOT="results_m4/ablation_g"
DET_CSV="${OUT_ROOT}/rellis_detections.csv"
TRACK_OUT="${OUT_ROOT}/sort_on_rellis"
TRACKS_CSV="${TRACK_OUT}/tracks.csv"

RUNNER="./build/construction_perception/tracker_runner"
PYTHON="${PYTHON:-python3}"

# Sanity checks on inputs.
if [[ ! -d "${CLUSTERS_DIR}" ]] || [[ -z "$(ls -A "${CLUSTERS_DIR}" 2>/dev/null)" ]]; then
    echo "ERROR: cluster CSVs missing at ${CLUSTERS_DIR}"
    echo "       run scripts/run_ablation_g.sh first."
    exit 1
fi

# Determine the frame range from cluster CSVs that exist.
LAST_FRAME=$(find "${CLUSTERS_DIR}" -maxdepth 1 -name 'clusters_*.csv' \
    | sed -E 's/.*clusters_([0-9]+)\.csv/\1/' | sort -n | tail -1 | sed 's/^0*//')
LAST_FRAME=${LAST_FRAME:-0}
echo "[rellis-tracker] using frames 0..${LAST_FRAME}"

# -----------------------------------------------------------------------------
# Stage 1 — clusters → detections CSV
# -----------------------------------------------------------------------------
if [[ ! -f "${DET_CSV}" ]]; then
    echo "[rellis-tracker] converting cluster centroids → detections CSV"
    "${PYTHON}" scripts/clusters_to_detections.py \
        --clusters-dir "${CLUSTERS_DIR}" \
        --frame-start  0 \
        --frame-end    "${LAST_FRAME}" \
        --out          "${DET_CSV}"
else
    echo "[rellis-tracker] reusing cached ${DET_CSV}"
fi

# -----------------------------------------------------------------------------
# Stage 2 — tracker_runner on RELLIS-derived detections
# -----------------------------------------------------------------------------
mkdir -p "${TRACK_OUT}"
if [[ ! -f "${TRACKS_CSV}" ]]; then
    echo "[rellis-tracker] running SORT (Munkres) on RELLIS detections"
    rm -rf "${TRACK_OUT}"/snapshots
    # NOTE: process_noise bumped 0.5 → 2.0 and min_hits dropped 3 → 1 after
    # the stationary-segment flicker analysis on the first run. Higher Q lets
    # the KF re-adapt to ego stops in ~2 frames instead of ~10; min_hits=1
    # publishes a re-acquired track immediately instead of forcing a 3-frame
    # warmup. See docs/m10-debug-log.md "stationary segment ID flicker" entry.
    # P3-M12: --filter selects CV (M4 baseline) or IMM (CV+CP). Defaults to
    # imm so this script's "happy path" exercises the new milestone. Override
    # for an A/B comparison run:
    #     TP_M4_FILTER=cv  bash scripts/run_tracker_on_rellis.sh   # M4 baseline
    #     TP_M4_FILTER=imm bash scripts/run_tracker_on_rellis.sh   # P3-M12
    "${RUNNER}" \
        --detections     "${DET_CSV}" \
        --solver         munkres \
        --filter         "${TP_M4_FILTER:-imm}" \
        --max-dist       5.0 \
        --max-misses     10 \
        --min-hits       1 \
        --dt             0.1 \
        --process-noise  2.0 \
        --meas-noise     0.3 \
        --snapshot-every 0 \
        --out            "${TRACK_OUT}" \
        --verbose
else
    echo "[rellis-tracker] reusing cached ${TRACKS_CSV}"
fi

# Quick stat: how many distinct track ids did SORT produce vs how many
# cluster_ids exist (the noise floor of "if you treated each cluster as its
# own track")?
N_TRACKS=$(awk -F, 'NR>1 {ids[$2]=1} END {print length(ids)}' "${TRACKS_CSV}")
N_CLUSTERS=$(awk -F, 'NR>1 {if ($4 >= 0) ids[$2"_"$4]=1} END {print length(ids)}' \
    "${EXT_ROOT}/clusters_sweetspot/clusters_000500.csv" 2>/dev/null || echo "?")
echo "[rellis-tracker] SORT produced ${N_TRACKS} distinct track ids "
echo "                across the whole sequence."

# -----------------------------------------------------------------------------
# Stage 3 — render closing-hero animation
# -----------------------------------------------------------------------------
echo "[rellis-tracker] rendering 3-panel closing-hero animation"
"${PYTHON}" scripts/animate_tracker_vs_dbscan.py \
    --lidar-dir       "${LIDAR_DIR}" \
    --camera-dir      "${CAM_DIR}" \
    --clusters-dir    "${CLUSTERS_DIR}" \
    --tracks-csv      "${TRACKS_CSV}" \
    --frame-start     0 \
    --frame-end       "${LAST_FRAME}" \
    --fps             10 \
    --stride          5 \
    --out-mp4         "${OUT_ROOT}/sort_vs_dbscan.mp4" \
    --out-gif         "${OUT_ROOT}/sort_vs_dbscan.gif"

echo
echo "================ M4 closing-hero summary ================"
echo "  Detections CSV : ${DET_CSV}"
echo "  Tracks CSV     : ${TRACKS_CSV}"
echo "  Distinct tracks: ${N_TRACKS}"
echo "  Animation MP4  : ${OUT_ROOT}/sort_vs_dbscan.mp4"
echo "  Animation GIF  : ${OUT_ROOT}/sort_vs_dbscan.gif"
echo "==========================================================="
