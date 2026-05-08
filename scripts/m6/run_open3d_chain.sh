#!/bin/bash
# scripts/m6/run_open3d_chain.sh
#
# Runs the three M6 Open3D chase-cam animations sequentially.
# Outputs to /media/nishant/SeeGayt2/terra_perceive/m6_animations/.
# A sentinel file `m6_open3d_chain.done` is touched when the third one lands.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/nishant/miniconda3/envs/terra-perceive/bin/python}"
LIDAR_DIR="${LIDAR_DIR:-/media/nishant/SeeGayt2/terra_perceive/m4_perframe/extracted_frames}"
SNAP_DIR="${SNAP_DIR:-/media/nishant/SeeGayt2/terra_perceive/m6_perframe/trav_probabilistic_perframe/snapshots}"
OUT_DIR="${OUT_DIR:-/media/nishant/SeeGayt2/terra_perceive/m6_animations}"
LOG_DIR="${LOG_DIR:-/tmp}"
mkdir -p "$OUT_DIR"
SENTINEL="$OUT_DIR/m6_open3d_chain.done"
rm -f "$SENTINEL"

run_mode() {
    local mode="$1"
    local out="$2"
    local extra=()
    if [[ "$mode" == "confidence" ]]; then
        extra=(--conf-snapshots "$SNAP_DIR")
    fi
    if [[ -f "$out" ]]; then
        echo "[skip] $out already exists"
        return 0
    fi
    echo "[run ] mode=$mode out=$out"
    "$PYTHON_BIN" "$REPO_ROOT/scripts/m6/open3d_chase.py" \
        --mode "$mode" \
        --lidar-dir "$LIDAR_DIR" \
        --out "$out" \
        --fps 30 \
        --width 1280 --height 720 \
        --point-decim 2 \
        "${extra[@]}" \
        > "$LOG_DIR/m6_open3d_${mode}.log" 2>&1
    echo "[done] $mode -> $out"
}

run_mode raw        "$OUT_DIR/open3d_raw.mp4"
run_mode ransac     "$OUT_DIR/open3d_ransac.mp4"
run_mode confidence "$OUT_DIR/open3d_confidence.mp4"

touch "$SENTINEL"
echo "[chain] all three Open3D renders complete; sentinel at $SENTINEL"
