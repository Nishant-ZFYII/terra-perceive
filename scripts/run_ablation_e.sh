#!/usr/bin/env bash
# run_ablation_e.sh — min_hits confirmation-threshold sweep on the spurious scenario.
#
# Question (rule #2 — predict before running):
#   "How many consecutive hits should we require before publishing a new track?"
#
# Predicted outcome:
#   min_hits=1 → publishes immediately. Spurious 1-frame false positives
#                briefly appear in the output. FP rate high. Init latency = 0.
#   min_hits=3 → spurious detections die before reaching threshold. Init
#                latency on real tracks = 2 frames.
#   min_hits=5 → even cleaner output but real tracks take 4 frames to confirm.
#
# Metrics:
#   fp_count        — number of unique track ids that ever appeared in
#                     publishable output but were associated with gt=-1
#                     detections in their first publishable frame.
#   init_latency    — number of frames between gt=0 first detection and
#                     gt=0 first publishable appearance.
#
# Per ablation pre-flight rules:
#   #1 cadence:    --snapshot-every 1
#   #6 resumable:  metrics.json check per cell
#   #10 qual pair: bar chart of (fp_count, init_latency) per cell

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

OUT_ROOT="results_m4/ablation_e"
DET_CSV="${OUT_ROOT}/spurious.csv"
RUNNER="./build/construction_perception/tracker_runner"
PYTHON="${PYTHON:-python3}"

mkdir -p "${OUT_ROOT}"

# -----------------------------------------------------------------------------
# Step 1 — synthesize spurious scenario
# -----------------------------------------------------------------------------
if [[ ! -f "${DET_CSV}" ]]; then
    echo "[ablation_e] generating spurious CSV"
    "${PYTHON}" scripts/synth_detections.py spurious \
        --frames 30 \
        --seed   42 \
        --out    "${DET_CSV}"
else
    echo "[ablation_e] reusing existing ${DET_CSV}"
fi

# -----------------------------------------------------------------------------
# Step 2 — min_hits sweep
# -----------------------------------------------------------------------------
MIN_HITS_VALUES=(1 3 5)

run_one() {
    local mh="$1"
    local out_dir="${OUT_ROOT}/mh_${mh}"
    if [[ -f "${out_dir}/metrics.json" ]]; then
        echo "[ablation_e] ✅ min_hits=${mh}: already done"
        return
    fi
    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"
    echo "[ablation_e] running min_hits=${mh}"
    "${RUNNER}" \
        --detections     "${DET_CSV}" \
        --solver         greedy \
        --max-dist       3.0 \
        --max-misses     1 \
        --min-hits       "${mh}" \
        --dt             0.1 \
        --process-noise  0.01 \
        --meas-noise     0.1 \
        --snapshot-every 1 \
        --out            "${out_dir}" \
        --verbose
}
for mh in "${MIN_HITS_VALUES[@]}"; do run_one "${mh}"; done

# -----------------------------------------------------------------------------
# Step 3 — bar chart of (fp_count, init_latency) per cell
# -----------------------------------------------------------------------------
run_dirs=()
labels=()
for mh in "${MIN_HITS_VALUES[@]}"; do
    run_dirs+=("${OUT_ROOT}/mh_${mh}")
    labels+=("min_hits=${mh}")
done

echo "[ablation_e] rendering min_hits sweep bar chart"
"${PYTHON}" scripts/plot_min_hits_sweep.py \
    --runs       "${run_dirs[@]}" \
    --labels     "${labels[@]}" \
    --detections "${DET_CSV}" \
    --out        "${OUT_ROOT}/min_hits_sweep.png"

echo
echo "================ Ablation E summary ================"
echo "  See ${OUT_ROOT}/min_hits_sweep.png for fp_count + init_latency."
echo "===================================================="
