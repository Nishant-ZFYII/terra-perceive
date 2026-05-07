#!/usr/bin/env bash
# run_ablation_d.sh — max_dist gating sweep on the crossing scenario.
#
# Question (rule #2 — predict before running):
#   "How tight should the gating threshold be?"
#
# Predicted outcome:
#   max_dist=1   → tight; small detection noise (sigma=0.05) is fine,
#                  but any genuine drift in predicted vs actual position will
#                  drop the match and fragment the track.
#   max_dist=3   → default; comfortable for our scenarios.
#   max_dist=10  → loose; almost no gating; in dense scenes ID-switch risk
#                  rises because impossible matches become possible.
#
# Metrics (rule #2):
#   id_switches      from metrics.json (computed by tracker_runner)
#   distinct_track_ids  from metrics.json (proxy for fragmentation)
#
# Per ablation pre-flight rules:
#   #1 cadence:    --snapshot-every 1
#   #6 resumable:  metrics.json check per cell
#   #10 qual pair: bar chart per metric

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

OUT_ROOT="results_m4/ablation_d"
DET_CSV="${OUT_ROOT}/crossing.csv"
RUNNER="./build/construction_perception/tracker_runner"
PYTHON="${PYTHON:-python3}"

mkdir -p "${OUT_ROOT}"

# -----------------------------------------------------------------------------
# Step 1 — synthesize crossing scenario (same generator as Ablation A)
# -----------------------------------------------------------------------------
if [[ ! -f "${DET_CSV}" ]]; then
    echo "[ablation_d] generating crossing CSV"
    "${PYTHON}" scripts/synth_detections.py crossing \
        --frames 30 \
        --seed   42 \
        --out    "${DET_CSV}"
else
    echo "[ablation_d] reusing existing ${DET_CSV}"
fi

# -----------------------------------------------------------------------------
# Step 2 — max_dist sweep
# -----------------------------------------------------------------------------
MAX_DIST_VALUES=(1 3 10)

run_one() {
    local d="$1"
    local out_dir="${OUT_ROOT}/d_${d}"
    if [[ -f "${out_dir}/metrics.json" ]]; then
        echo "[ablation_d] ✅ max_dist=${d}: already done"
        return
    fi
    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"
    echo "[ablation_d] running max_dist=${d}"
    "${RUNNER}" \
        --detections     "${DET_CSV}" \
        --solver         greedy \
        --max-dist       "${d}" \
        --max-misses     3 \
        --min-hits       1 \
        --dt             0.1 \
        --process-noise  0.01 \
        --meas-noise     0.1 \
        --snapshot-every 1 \
        --out            "${out_dir}" \
        --verbose
}
for d in "${MAX_DIST_VALUES[@]}"; do run_one "${d}"; done

# -----------------------------------------------------------------------------
# Step 3 — bar chart of metrics
# -----------------------------------------------------------------------------
run_dirs=()
labels=()
for d in "${MAX_DIST_VALUES[@]}"; do
    run_dirs+=("${OUT_ROOT}/d_${d}")
    labels+=("${d}m")
done

echo "[ablation_d] rendering max_dist sweep bar chart"
"${PYTHON}" scripts/plot_max_dist_sweep.py \
    --runs   "${run_dirs[@]}" \
    --labels "${labels[@]}" \
    --out    "${OUT_ROOT}/max_dist_sweep.png"

# -----------------------------------------------------------------------------
# Step 4 — summary
# -----------------------------------------------------------------------------
echo
echo "================ Ablation D summary ================"
printf "  %-10s  %-12s  %-18s\n" "max_dist" "id_switches" "distinct_track_ids"
for d in "${MAX_DIST_VALUES[@]}"; do
    f="${OUT_ROOT}/d_${d}/metrics.json"
    if [[ -f "${f}" ]]; then
        sw=$(grep -E '"id_switches"' "${f}" | grep -oE '[0-9]+' | head -1)
        dt=$(grep -E '"distinct_track_ids"' "${f}" | grep -oE '[0-9]+' | head -1)
        printf "  %-10s  %-12s  %-18s\n" "${d}m" "${sw:-?}" "${dt:-?}"
    fi
done
echo "===================================================="
