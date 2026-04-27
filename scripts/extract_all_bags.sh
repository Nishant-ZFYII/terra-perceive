#!/usr/bin/env bash
# extract_all_bags.sh — full RELLIS-3D LiDAR extraction across 5 bags.
#
# Output goes to the external drive per ablation rule #5 (heavy snapshot
# data lives off the SSD). A symlink data/extracted_frames_full points
# at the external dir for shell-script convenience.
#
# Frame numbering is contiguous across all 5 bags (00000_00 → 00000_04),
# achieved via extract_bag.py's --start-id flag. After this script
# completes, the directory holds 000000.bin .. NNNNNN.bin where N is the
# total frame count across the recording.
#
# Resumable: each bag's frames are written under a temporary marker, and
# the script skips bags whose extraction completed previously. To force a
# full re-extract, delete the marker file (or the whole output dir).
#
# Per ablation pre-flight rules:
#   #3 overnight safety: the extraction itself is ~15 min on a laptop
#                        SSD-to-HDD copy, so unattended-upgrades is fine
#                        to leave running. Disable it later if you chain
#                        obstacle_extractor + animation in one batch >3h.
#   #5 heavy data lives on /media/nishant/SeeGayt2 not /
#   #6 resumable via marker file per bag

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

EXT_ROOT="/media/nishant/SeeGayt2/terra_perceive/m4_perframe"
EXT_OUT="${EXT_ROOT}/extracted_frames"
BAG_DIR="data/RELLIS-3D"
PYTHON="${PYTHON:-python3}"

mkdir -p "${EXT_OUT}"

# Bags in chronological order. The names are 00000_00 .. 00000_04 in the
# RELLIS-3D download; preserve that order so frame_id corresponds to the
# real-world recording sequence.
BAGS=(
    "00000_00.bag"
    "00000_01.bag"
    "00000_02.bag"
    "00000_03.bag"
    "00000_04.bag"
)

# -----------------------------------------------------------------------------
# Per-bag extraction with continuous numbering and resume guard.
# -----------------------------------------------------------------------------
start_id=0
for bag in "${BAGS[@]}"; do
    bag_path="${BAG_DIR}/${bag}"
    marker="${EXT_OUT}/.${bag}.done"
    count_file="${EXT_OUT}/.${bag}.count"

    if [[ ! -f "${bag_path}" ]]; then
        echo "[extract_all_bags] WARN: missing ${bag_path}, skipping"
        continue
    fi

    if [[ -f "${marker}" ]]; then
        # Resume: read the count we recorded last time, advance start_id.
        prev_count=$(cat "${count_file}")
        echo "[extract_all_bags] ✅ ${bag}: already done (${prev_count} frames at offset ${start_id})"
        start_id=$(( start_id + prev_count ))
        continue
    fi

    echo "[extract_all_bags] extracting ${bag} starting at frame_id=${start_id}"
    before=$(find "${EXT_OUT}" -maxdepth 1 -name '*.bin' 2>/dev/null | wc -l)
    "${PYTHON}" scripts/extract_bag.py \
        "${bag_path}" \
        "${EXT_OUT}" \
        --every    1 \
        --start-id "${start_id}"
    after=$(find "${EXT_OUT}" -maxdepth 1 -name '*.bin' 2>/dev/null | wc -l)
    bag_count=$(( after - before ))

    # Persist the count so we can resume. Touch the marker LAST so a kill
    # mid-extract leaves the marker absent and we re-extract on retry.
    echo "${bag_count}" > "${count_file}"
    touch "${marker}"
    echo "[extract_all_bags] ${bag}: extracted ${bag_count} frames (total now ${after})"

    start_id=$(( start_id + bag_count ))
done

# -----------------------------------------------------------------------------
# Symlink the extracted dir into the repo for shell-script convenience.
# Existing symlinks are replaced; existing real files/dirs are left alone.
# -----------------------------------------------------------------------------
LINK="${REPO_ROOT}/data/extracted_frames_full"
if [[ -L "${LINK}" ]]; then
    rm "${LINK}"
fi
if [[ ! -e "${LINK}" ]]; then
    ln -s "${EXT_OUT}" "${LINK}"
    echo "[extract_all_bags] symlink: ${LINK} → ${EXT_OUT}"
elif [[ -e "${LINK}" && ! -L "${LINK}" ]]; then
    echo "[extract_all_bags] WARN: ${LINK} exists and is NOT a symlink — leaving alone."
fi

# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------
total=$(find "${EXT_OUT}" -maxdepth 1 -name '*.bin' 2>/dev/null | wc -l)
last_id=$(( start_id - 1 ))
echo
echo "================ Extraction summary ================"
echo "  Output directory : ${EXT_OUT}"
echo "  Repo symlink     : ${LINK}"
echo "  Total frames     : ${total}"
echo "  Frame ID range   : 0 .. ${last_id}"
echo "===================================================="
