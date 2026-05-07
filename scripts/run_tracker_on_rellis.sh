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
FEATURES_DIR="${EXT_ROOT}/appearance"
LIDAR_DIR="data/extracted_frames_full"
CAM_DIR="${EXT_ROOT}/extracted_frames_camera"
OBSTACLES_DIR="${EXT_ROOT}/obstacles"

OUT_ROOT="results_m4/ablation_g"
DET_CSV="${OUT_ROOT}/rellis_detections.csv"
FEAT_CSV="${OUT_ROOT}/rellis_detections.features.csv"
TRACK_OUT="${OUT_ROOT}/sort_on_rellis"
TRACKS_CSV="${TRACK_OUT}/tracks.csv"

# P3-M13: enable appearance via env var; defaults to OFF (M12 path).
#     TP_M4_USE_APPEARANCE=1 bash scripts/run_tracker_on_rellis.sh
# Optionally tune:
#     TP_M4_APPEARANCE_WEIGHT=0.6 ...
#
# λ default lowered from 0.4 → 0.2 after the M13 RELLIS λ sweep
# (see docs/m10-debug-log.md "λ sweep at max_misses=300"). 0.2 was the
# minimum-distinct-IDs operating point at the recommended max_misses=300
# config; 0.4 over-weighted the encoder and regressed.
USE_APPEARANCE="${TP_M4_USE_APPEARANCE:-0}"
APPEARANCE_WEIGHT="${TP_M4_APPEARANCE_WEIGHT:-0.2}"
EMBEDDING_ALPHA="${TP_M4_EMBEDDING_ALPHA:-0.1}"
# Phase-3.5 cascade matching: keep dying tracks alive in a retired pool
# for max_age frames after they exceed max_misses.
#
# Fix B (shipped): cascade's frozen position is now stored in WORLD
# frame using the P2-M2 SLAM ego pose, and the per-frame match step
# transforms it back into current-ego coordinates before gating. With
# ego motion accounted for, max_age=300 (30 sec) is honest again — Lost
# tracks revive only when a re-detection lands at the actual physical
# position they were last seen, not at any cluster that happens to land
# at similar ego-relative coordinates after ego has driven away.
#
# History:
#   max_age=300, ego-frame anchor (pre-Fix-B):  99 distinct  — ARTIFACT,
#       18/35 stationary tracks were false-merge revivals up to 15.7 m apart
#   max_age=30,  ego-frame anchor (defensive Fix A):       202 distinct
#       3-second ego validity ≈ ~5 m drift, false revivals minimized
#   max_age=300, world-frame anchor (Fix B):              <honest_number>
#       cascade reaches across long gaps but only at the right physical place
#
# Override:
#   TP_M4_MAX_MISSES=10 TP_M4_MAX_AGE=300 bash scripts/run_tracker_on_rellis.sh
#   TP_M4_MAX_AGE=0  bash ...    # disable cascade entirely
#   TP_M4_EGO_POSES= bash ...    # disable Fix B (ego-frame anchor, legacy)
MAX_MISSES="${TP_M4_MAX_MISSES:-10}"
MAX_AGE="${TP_M4_MAX_AGE:-300}"
EGO_POSES_CSV="${TP_M4_EGO_POSES:-data/poses_slam_full.csv}"

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
# Stage 1 — clusters → detections CSV (+ features CSV if appearance is on)
# -----------------------------------------------------------------------------
NEED_FEATURES_REGEN=0
if [[ "${USE_APPEARANCE}" == "1" && ! -f "${FEAT_CSV}" ]]; then
    NEED_FEATURES_REGEN=1
fi

if [[ ! -f "${DET_CSV}" || "${NEED_FEATURES_REGEN}" == "1" ]]; then
    echo "[rellis-tracker] converting cluster centroids → detections CSV"
    if [[ "${USE_APPEARANCE}" == "1" ]]; then
        if [[ ! -d "${FEATURES_DIR}" ]] || [[ ! -f "${FEATURES_DIR}/corpus_stats.json" ]]; then
            echo "ERROR: --use-appearance needs ${FEATURES_DIR}/corpus_stats.json"
            echo "       run: python python/appearance/extract_features.py \\"
            echo "           --clusters-dir ${CLUSTERS_DIR} \\"
            echo "           --out-dir ${FEATURES_DIR} --workers 8"
            exit 1
        fi
        "${PYTHON}" scripts/clusters_to_detections.py \
            --clusters-dir "${CLUSTERS_DIR}" \
            --features-dir "${FEATURES_DIR}" \
            --frame-start  0 \
            --frame-end    "${LAST_FRAME}" \
            --out          "${DET_CSV}" \
            --features-out "${FEAT_CSV}"
    else
        "${PYTHON}" scripts/clusters_to_detections.py \
            --clusters-dir "${CLUSTERS_DIR}" \
            --frame-start  0 \
            --frame-end    "${LAST_FRAME}" \
            --out          "${DET_CSV}"
    fi
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
    APPEARANCE_FLAGS=()
    if [[ "${USE_APPEARANCE}" == "1" ]]; then
        APPEARANCE_FLAGS=(
            --use-appearance
            --features-csv      "${FEAT_CSV}"
            --appearance-weight "${APPEARANCE_WEIGHT}"
            --embedding-alpha   "${EMBEDDING_ALPHA}"
        )
        echo "[rellis-tracker] appearance ON  λ=${APPEARANCE_WEIGHT}  α=${EMBEDDING_ALPHA}"
    fi
    EGO_POSE_FLAGS=()
    if [[ -n "${EGO_POSES_CSV}" && -f "${EGO_POSES_CSV}" ]]; then
        EGO_POSE_FLAGS=(--ego-poses "${EGO_POSES_CSV}")
        echo "[rellis-tracker] Fix B ON  ego-poses=${EGO_POSES_CSV}"
    elif [[ -n "${EGO_POSES_CSV}" ]]; then
        echo "[rellis-tracker] WARN ego-poses CSV not found at ${EGO_POSES_CSV} — falling back to Identity (no Fix B)"
    fi
    "${RUNNER}" \
        --detections     "${DET_CSV}" \
        --solver         munkres \
        --filter         "${TP_M4_FILTER:-imm}" \
        --max-dist       5.0 \
        --max-misses     "${MAX_MISSES}" \
        --max-age        "${MAX_AGE}" \
        --min-hits       1 \
        --dt             0.1 \
        --process-noise  2.0 \
        --meas-noise     0.3 \
        --snapshot-every 0 \
        --out            "${TRACK_OUT}" \
        --verbose \
        "${EGO_POSE_FLAGS[@]}" \
        "${APPEARANCE_FLAGS[@]}"
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
# Stage 3 — render closing-hero animation (OPT-IN, ~30 minutes on laptop)
# -----------------------------------------------------------------------------
# The 3-panel render is the blog asset, not the iteration loop. Default OFF
# so a tracker tweak can be measured in seconds without burning CPU on a
# rendering pass that's not part of the metric. Enable for the final
# blog-quality run only:
#     TP_M4_RENDER_ANIM=1 bash scripts/run_tracker_on_rellis.sh
RENDER_ANIM="${TP_M4_RENDER_ANIM:-0}"
if [[ "${RENDER_ANIM}" == "1" ]]; then
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
    ANIM_MP4="${OUT_ROOT}/sort_vs_dbscan.mp4"
    ANIM_GIF="${OUT_ROOT}/sort_vs_dbscan.gif"
else
    echo "[rellis-tracker] animation skipped (TP_M4_RENDER_ANIM=0)."
    echo "                 Re-run with TP_M4_RENDER_ANIM=1 once metrics ship."
    ANIM_MP4="(skipped)"
    ANIM_GIF="(skipped)"
fi

echo
echo "================ M4 closing-hero summary ================"
echo "  Detections CSV : ${DET_CSV}"
echo "  Tracks CSV     : ${TRACKS_CSV}"
echo "  Distinct tracks: ${N_TRACKS}"
echo "  Animation MP4  : ${ANIM_MP4}"
echo "  Animation GIF  : ${ANIM_GIF}"
echo "==========================================================="
