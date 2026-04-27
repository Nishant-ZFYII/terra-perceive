#!/usr/bin/env bash
# extract_all_cameras.sh — extract Pylon camera frames from all 5 RELLIS bags,
# numbered in lockstep with extracted_frames/ (LiDAR).
#
# After this script:
#   .../m4_perframe/extracted_frames/000000.bin     (LiDAR cloud)
#   .../m4_perframe/extracted_frames_camera/000000.jpg  (RGB image, time-synced)
#   ... and so on for all NNNN frames.
#
# Resumable per bag via marker files (same pattern as extract_all_bags.sh).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

EXT_ROOT="${TP_M4_EXT_ROOT:-/media/nishant/SeeGayt2/terra_perceive/m4_perframe}"
LIDAR_OUT="${EXT_ROOT}/extracted_frames"
CAM_OUT="${EXT_ROOT}/extracted_frames_camera"
BAG_DIR="data/RELLIS-3D"
PYTHON="${PYTHON:-python3}"

mkdir -p "${CAM_OUT}"

BAGS=(
    "00000_00.bag"
    "00000_01.bag"
    "00000_02.bag"
    "00000_03.bag"
    "00000_04.bag"
)

if [[ ! -d "${LIDAR_OUT}" ]] || [[ -z "$(ls -A "${LIDAR_OUT}" 2>/dev/null)" ]]; then
    echo "[extract_all_cameras] ERROR: LiDAR dir empty: ${LIDAR_OUT}"
    echo "  Run scripts/extract_all_bags.sh first."
    exit 1
fi

start_id=0
for bag in "${BAGS[@]}"; do
    bag_path="${BAG_DIR}/${bag}"
    marker="${CAM_OUT}/.${bag}.done"
    count_file="${CAM_OUT}/.${bag}.count"
    lidar_count_file="${LIDAR_OUT}/.${bag}.count"

    if [[ ! -f "${bag_path}" ]]; then
        echo "[extract_all_cameras] WARN: missing ${bag_path}, skipping"
        continue
    fi

    if [[ ! -f "${lidar_count_file}" ]]; then
        echo "[extract_all_cameras] ERROR: ${lidar_count_file} missing — extract_all_bags.sh hasn't run for ${bag}?"
        exit 1
    fi
    lidar_count=$(cat "${lidar_count_file}")

    if [[ -f "${marker}" ]]; then
        prev_count=$(cat "${count_file}")
        echo "[extract_all_cameras] ✅ ${bag}: cached (${prev_count} frames at offset ${start_id})"
        start_id=$(( start_id + lidar_count ))
        continue
    fi

    echo "[extract_all_cameras] extracting camera from ${bag} (offset=${start_id}, expecting ${lidar_count} frames)"
    before=$(find "${CAM_OUT}" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)
    "${PYTHON}" -u scripts/extract_camera.py \
        "${bag_path}" \
        "${CAM_OUT}" \
        --start-id    "${start_id}" \
        --lidar-count "${lidar_count}"
    after=$(find "${CAM_OUT}" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)
    bag_count=$(( after - before ))

    echo "${bag_count}" > "${count_file}"
    touch "${marker}"
    echo "[extract_all_cameras] ${bag}: ${bag_count} JPEGs (cumulative ${after})"

    start_id=$(( start_id + lidar_count ))
done

total=$(find "${CAM_OUT}" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)
echo
echo "================ Camera extraction summary ================"
echo "  Output dir   : ${CAM_OUT}"
echo "  Total frames : ${total}"
echo "==========================================================="
