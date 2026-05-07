#!/usr/bin/env bash
# run_ablation_b.sh — Q (process-noise) sweep on a noisy linear scenario.
#
# Question (rule #2 — predict before running):
#   "How much should the Kalman filter trust its motion model vs incoming
#    noisy measurements?"
#
# Predicted outcome:
#   Q=0.01  → filter trusts the constant-velocity model; estimate is smooth
#             and lags slightly through any small velocity drift; cov trace
#             converges to a small steady state.
#   Q=10.0  → filter trusts every measurement; estimate visibly tracks the
#             noise; cov trace stays large.
#   Q=0.1, 1.0 → in-between; one of these is the "right" pick for this Q/R
#                regime — the plot lets the reader see the trade-off.
#
# Per ablation pre-flight rules:
#   #1 cadence:    --snapshot-every 1 (need per-frame data for the plot)
#   #2 metric:     position-residual + cov-trace, both per-frame
#   #6 resumable:  per-cell metrics.json check below
#   #10 qualitative pair: position-vs-frame plot + cov-trace plot in one PNG

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

OUT_ROOT="results_m4/ablation_b"
DET_CSV="${OUT_ROOT}/linear.csv"
RUNNER="./build/construction_perception/tracker_runner"
PYTHON="${PYTHON:-python3}"

mkdir -p "${OUT_ROOT}"

# -----------------------------------------------------------------------------
# Step 1 — synthesize linear scenario (50 frames, deterministic seed)
# -----------------------------------------------------------------------------
if [[ ! -f "${DET_CSV}" ]]; then
    echo "[ablation_b] generating linear scenario CSV"
    "${PYTHON}" scripts/synth_detections.py linear \
        --frames 50 \
        --seed   42 \
        --out    "${DET_CSV}"
else
    echo "[ablation_b] reusing existing ${DET_CSV}"
fi

# -----------------------------------------------------------------------------
# Step 2 — Q sweep (4 cells; rule #6 skips done runs)
# -----------------------------------------------------------------------------
Q_VALUES=(0.01 0.1 1.0 10.0)

run_one() {
    local q="$1"
    local out_dir="${OUT_ROOT}/q_${q}"
    if [[ -f "${out_dir}/metrics.json" ]]; then
        echo "[ablation_b] ✅ Q=${q}: already done"
        return
    fi
    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"
    echo "[ablation_b] running Q=${q}"
    "${RUNNER}" \
        --detections     "${DET_CSV}" \
        --solver         greedy \
        --max-dist       3.0 \
        --max-misses     5 \
        --min-hits       1 \
        --dt             0.1 \
        --process-noise  "${q}" \
        --meas-noise     0.1 \
        --snapshot-every 1 \
        --out            "${out_dir}" \
        --verbose
}
for q in "${Q_VALUES[@]}"; do run_one "${q}"; done

# -----------------------------------------------------------------------------
# Step 3 — qualitative pair (rule #10): the headline plot for the Kalman
# section of the blog. Two-panel PNG: x-position-vs-frame and cov-trace-vs-frame.
# -----------------------------------------------------------------------------
run_dirs=()
labels=()
for q in "${Q_VALUES[@]}"; do
    run_dirs+=("${OUT_ROOT}/q_${q}")
    labels+=("Q=${q}")
done

echo "[ablation_b] rendering Kalman convergence plot"
"${PYTHON}" scripts/plot_kalman_convergence.py \
    --runs       "${run_dirs[@]}" \
    --labels     "${labels[@]}" \
    --detections "${DET_CSV}" \
    --out        "${OUT_ROOT}/q_sweep.png"

# -----------------------------------------------------------------------------
# Step 4 — summary table
# -----------------------------------------------------------------------------
echo
echo "================ Ablation B summary ================"
printf "  %-8s  %-10s  %s\n" "Q" "runtime_ms" "metric_dump"
for q in "${Q_VALUES[@]}"; do
    f="${OUT_ROOT}/q_${q}/metrics.json"
    if [[ -f "${f}" ]]; then
        rt=$(grep -E '"runtime_ms"' "${f}" | grep -oE '[0-9.]+' | head -1)
        printf "  %-8s  %-10s  %s\n" "Q=${q}" "${rt:-?}" "${f}"
    fi
done
echo "===================================================="
