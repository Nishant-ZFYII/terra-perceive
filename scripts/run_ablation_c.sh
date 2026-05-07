#!/usr/bin/env bash
# run_ablation_c.sh — R (measurement-noise) sweep on the linear scenario.
#
# Question (rule #2 — predict before running):
#   "How much should the Kalman filter trust each incoming measurement
#    relative to its motion-model prior?"
#
# This is the symmetric question to Ablation B's Q sweep. The mathematical
# duality: the Kalman gain depends on the ratio P/(HPH^T + R). Increasing R
# at fixed Q has the same QUALITATIVE effect on the gain as decreasing Q at
# fixed R — both make the filter trust the model more. So we expect this
# sweep's residual panel to look like the Q sweep's, but with the order
# reversed (low R = follows noise, high R = smooths through noise).
#
# Predicted outcome:
#   R=0.01 → gain near 1.0 → estimate follows every measurement (jittery).
#   R=1.0  → gain near 0   → estimate barely updates (smooth, but biased
#                            if the model isn't perfect).
#   R=0.1  → balanced; the natural pick when measurement noise really IS 0.1.
#
# Per ablation pre-flight rules:
#   #1 cadence:    --snapshot-every 1
#   #2 metric:     residual amplitude (small=smooth, large=tracks noise)
#                  + cov trace
#   #6 resumable:  metrics.json check per cell
#   #10 qual pair: same 3-panel PNG as Q sweep, retitled

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

OUT_ROOT="results_m4/ablation_c"
DET_CSV="${OUT_ROOT}/linear.csv"
RUNNER="./build/construction_perception/tracker_runner"
PYTHON="${PYTHON:-python3}"

mkdir -p "${OUT_ROOT}"

# -----------------------------------------------------------------------------
# Step 1 — synthesize linear scenario (same generator as B; deterministic seed)
# -----------------------------------------------------------------------------
if [[ ! -f "${DET_CSV}" ]]; then
    echo "[ablation_c] generating linear scenario CSV"
    "${PYTHON}" scripts/synth_detections.py linear \
        --frames 50 \
        --seed   42 \
        --out    "${DET_CSV}"
else
    echo "[ablation_c] reusing existing ${DET_CSV}"
fi

# -----------------------------------------------------------------------------
# Step 2 — R sweep (3 cells; rule #6 skips done runs)
# -----------------------------------------------------------------------------
R_VALUES=(0.01 0.1 1.0)

run_one() {
    local r="$1"
    local out_dir="${OUT_ROOT}/r_${r}"
    if [[ -f "${out_dir}/metrics.json" ]]; then
        echo "[ablation_c] ✅ R=${r}: already done"
        return
    fi
    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"
    echo "[ablation_c] running R=${r}"
    "${RUNNER}" \
        --detections     "${DET_CSV}" \
        --solver         greedy \
        --max-dist       3.0 \
        --max-misses     5 \
        --min-hits       1 \
        --dt             0.1 \
        --process-noise  0.1 \
        --meas-noise     "${r}" \
        --snapshot-every 1 \
        --out            "${out_dir}" \
        --verbose
}
for r in "${R_VALUES[@]}"; do run_one "${r}"; done

# -----------------------------------------------------------------------------
# Step 3 — qualitative pair (rule #10): reuse Q-sweep plotter with new title
# -----------------------------------------------------------------------------
run_dirs=()
labels=()
for r in "${R_VALUES[@]}"; do
    run_dirs+=("${OUT_ROOT}/r_${r}")
    labels+=("R=${r}")
done

echo "[ablation_c] rendering R-sweep plot"
"${PYTHON}" scripts/plot_kalman_convergence.py \
    --runs       "${run_dirs[@]}" \
    --labels     "${labels[@]}" \
    --detections "${DET_CSV}" \
    --title      "Measurement-noise R sweep — measurement trust vs smoothing" \
    --out        "${OUT_ROOT}/r_sweep.png"

# -----------------------------------------------------------------------------
# Step 4 — summary table
# -----------------------------------------------------------------------------
echo
echo "================ Ablation C summary ================"
printf "  %-8s  %-10s  %s\n" "R" "runtime_ms" "metric_dump"
for r in "${R_VALUES[@]}"; do
    f="${OUT_ROOT}/r_${r}/metrics.json"
    if [[ -f "${f}" ]]; then
        rt=$(grep -E '"runtime_ms"' "${f}" | grep -oE '[0-9.]+' | head -1)
        printf "  %-8s  %-10s  %s\n" "R=${r}" "${rt:-?}" "${f}"
    fi
done
echo "===================================================="
