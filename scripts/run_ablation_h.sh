#!/usr/bin/env bash
# run_ablation_h.sh — predict/update ordering on the linear scenario.
#
# Question (rule #2 — predict before running):
#   "Does the order of predict/update inside the tracker's per-frame pipeline
#    matter, on data that has no velocity change?"
#
# Predicted outcome:
#   PredictThenUpdate (canonical SORT) — clean residual, cov trace shrinks
#                                        to a low steady state.
#   UpdateThenPredict (the bug pattern) — residual one-frame-lagged; cov
#                                         trace still shrinks (the bug is
#                                         most visible during maneuvers,
#                                         but a small transient asymmetry
#                                         is still observable on linear data).
#
# This integration-level ablation visualizes the consequence of the same
# bug exposed by the unit test KalmanFilter.UpdateOrderMatters, which uses
# a velocity reversal at frame 25 to amplify the divergence. Here on a pure
# linear scene the difference is more subtle — the unit test carries the
# strong claim, this ablation provides the visualization.
#
# Per ablation pre-flight rules:
#   #1 cadence:    --snapshot-every 1
#   #6 resumable:  metrics.json check per cell
#   #10 qual pair: 3-panel residual + cov-trace, both orderings overlaid

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

OUT_ROOT="results_m4/ablation_h"
DET_CSV="${OUT_ROOT}/maneuver.csv"
RUNNER="./build/construction_perception/tracker_runner"
PYTHON="${PYTHON:-python3}"

mkdir -p "${OUT_ROOT}"

# -----------------------------------------------------------------------------
# Step 1 — synthesize linear scenario (deterministic seed)
# -----------------------------------------------------------------------------
if [[ ! -f "${DET_CSV}" ]]; then
    echo "[ablation_h] generating maneuver CSV (linear with velocity reversal at frame 25)"
    "${PYTHON}" scripts/synth_detections.py maneuver \
        --frames 50 \
        --seed   42 \
        --out    "${DET_CSV}"
else
    echo "[ablation_h] reusing existing ${DET_CSV}"
fi

# -----------------------------------------------------------------------------
# Step 2 — two cells: correct vs swapped order
# -----------------------------------------------------------------------------
run_correct() {
    local out_dir="${OUT_ROOT}/correct"
    if [[ -f "${out_dir}/metrics.json" ]]; then
        echo "[ablation_h] ✅ correct (PredictThenUpdate): already done"
        return
    fi
    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"
    echo "[ablation_h] running correct (PredictThenUpdate)"
    "${RUNNER}" \
        --detections     "${DET_CSV}" \
        --solver         greedy \
        --max-dist       10.0 \
        --max-misses     5 \
        --min-hits       1 \
        --dt             0.1 \
        --process-noise  0.1 \
        --meas-noise     0.1 \
        --snapshot-every 1 \
        --out            "${out_dir}" \
        --verbose
}

run_swapped() {
    local out_dir="${OUT_ROOT}/swapped"
    if [[ -f "${out_dir}/metrics.json" ]]; then
        echo "[ablation_h] ✅ swapped (UpdateThenPredict): already done"
        return
    fi
    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"
    echo "[ablation_h] running swapped (UpdateThenPredict)"
    "${RUNNER}" \
        --detections     "${DET_CSV}" \
        --solver         greedy \
        --max-dist       10.0 \
        --max-misses     5 \
        --min-hits       1 \
        --dt             0.1 \
        --process-noise  0.1 \
        --meas-noise     0.1 \
        --snapshot-every 1 \
        --swap-order \
        --out            "${out_dir}" \
        --verbose
}

run_correct
run_swapped

# -----------------------------------------------------------------------------
# Step 3 — qualitative pair (rule #10): reuse Q-sweep plotter with new title
# -----------------------------------------------------------------------------
echo "[ablation_h] rendering ordering-sweep plot"
"${PYTHON}" scripts/plot_kalman_convergence.py \
    --runs       "${OUT_ROOT}/correct" "${OUT_ROOT}/swapped" \
    --labels     "predict→update" "update→predict" \
    --detections "${DET_CSV}" \
    --title      "Predict/update ordering — visual consequence of the bug pattern" \
    --out        "${OUT_ROOT}/order_sweep.png"

# -----------------------------------------------------------------------------
# Step 4 — summary
# -----------------------------------------------------------------------------
echo
echo "================ Ablation H summary ================"
printf "  %-20s  %-10s\n" "ordering" "runtime_ms"
for cell in correct swapped; do
    f="${OUT_ROOT}/${cell}/metrics.json"
    if [[ -f "${f}" ]]; then
        rt=$(grep -E '"runtime_ms"' "${f}" | grep -oE '[0-9.]+' | head -1)
        printf "  %-20s  %-10s\n" "${cell}" "${rt:-?}"
    fi
done
echo
echo "  Unit-level proof: KalmanFilter.UpdateOrderMatters in test_kalman.cpp"
echo "  Visual consequence: ${OUT_ROOT}/order_sweep.png"
echo "===================================================="
