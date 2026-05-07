#!/usr/bin/env bash
# run_ablation_a.sh — Ablation A (greedy vs Munkres on crossing scenario).
#
# Per ablation pre-flight rules:
#   #6  resumable: skips runs whose metrics.json already exists.
#   #2  metric:    ID-switch count printed live and dumped to id_switches.csv.
#   #1  cadence:   --snapshot-every 1 (animation-grade).
#   #10 qualitative: also produces side-by-side MP4 + cumulative line plot.
#
# Run from repo root (assumes build/construction_perception/ exists):
#   colcon build --packages-select construction_perception
#   bash scripts/run_ablation_a.sh
#
# Outputs land under results_m4/ablation_a/:
#   greedy/   (tracks.csv, id_switches.csv, snapshots/, metrics.json)
#   munkres/  (same shape)
#   crossing.csv   (synthetic input, regenerated only if missing)
#   greedy_vs_munkres.mp4
#   id_switches.png

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

OUT_ROOT="results_m4/ablation_a"
DET_CSV="${OUT_ROOT}/crossing.csv"
RUNNER="./build/construction_perception/tracker_runner"
PYTHON="${PYTHON:-python3}"

mkdir -p "${OUT_ROOT}"

# -----------------------------------------------------------------------------
# Step 1 — synthetic crossing CSV (regenerate only if missing)
# -----------------------------------------------------------------------------
if [[ ! -f "${DET_CSV}" ]]; then
    echo "[ablation_a] generating synthetic crossing CSV"
    "${PYTHON}" scripts/synth_detections.py crossing \
        --frames 30 \
        --seed 42 \
        --out "${DET_CSV}"
else
    echo "[ablation_a] reusing existing ${DET_CSV}"
fi

# -----------------------------------------------------------------------------
# Step 2 — run both solvers (skip if metrics.json already exists, rule #6)
# -----------------------------------------------------------------------------
run_one() {
    local solver="$1"
    local out_dir="${OUT_ROOT}/${solver}"
    if [[ -f "${out_dir}/metrics.json" ]]; then
        echo "[ablation_a] ✅ ${solver}: already done (metrics.json present)"
        return
    fi
    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"
    echo "[ablation_a] running ${solver}"
    "${RUNNER}" \
        --detections     "${DET_CSV}" \
        --solver         "${solver}" \
        --max-dist       3.0 \
        --max-misses     3 \
        --min-hits       1 \
        --dt             0.1 \
        --process-noise  0.01 \
        --meas-noise     0.1 \
        --snapshot-every 1 \
        --out            "${out_dir}" \
        --verbose
}

run_one greedy
run_one munkres

# -----------------------------------------------------------------------------
# Step 3 — qualitative pair (rule #10): side-by-side MP4 + cumulative line plot
# -----------------------------------------------------------------------------
echo "[ablation_a] rendering side-by-side MP4"
"${PYTHON}" scripts/animate_tracker.py compare \
    --tracks-left  "${OUT_ROOT}/greedy/tracks.csv" \
    --tracks-right "${OUT_ROOT}/munkres/tracks.csv" \
    --label-left   "Greedy" \
    --label-right  "Munkres" \
    --detections   "${DET_CSV}" \
    --out          "${OUT_ROOT}/greedy_vs_munkres.mp4" \
    --fps          10

echo "[ablation_a] plotting ID-switch lines"
"${PYTHON}" scripts/plot_id_switches.py \
    --run   "${OUT_ROOT}/greedy"  --label "Greedy" \
    --run   "${OUT_ROOT}/munkres" --label "Munkres" \
    --out   "${OUT_ROOT}/id_switches.png"

# -----------------------------------------------------------------------------
# Step 4 — print headline metric (rule #2: predict greedy>=1, munkres=0)
# -----------------------------------------------------------------------------
echo
echo "================ Ablation A summary ================"
for solver in greedy munkres; do
    f="${OUT_ROOT}/${solver}/metrics.json"
    if [[ -f "${f}" ]]; then
        sw=$(grep -E '"id_switches"' "${f}" | grep -oE '[0-9]+' | head -1)
        rt=$(grep -E '"runtime_ms"'  "${f}" | grep -oE '[0-9.]+' | head -1)
        printf "  %-8s  id_switches=%-3s  runtime_ms=%s\n" "${solver}" "${sw:-?}" "${rt:-?}"
    fi
done
echo "===================================================="
