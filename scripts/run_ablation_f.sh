#!/usr/bin/env bash
# run_ablation_f.sh — max_misses persistence sweep on the occluded scenario.
#
# Question (rule #2 — predict before running):
#   "How many missed frames should a track survive before pruning?"
#
# Predicted outcome (gen_occluded has an 8-frame contiguous detection gap):
#   max_misses=1  → track pruned after 2 missed frames. New track id on resume.
#                   "ID swap on every blink."
#   max_misses=3  → pruned after 4 misses. Still gets pruned during the gap.
#                   New id on resume.
#   max_misses=10 → tolerates the gap; SAME id resumes. "Track persists
#                   through occlusion."
#
# Metrics:
#   recovered_same_id  — boolean: did the same track id appear before AND
#                        after the occlusion gap?
#   distinct_track_ids — total ids ever spawned (1 = clean; >1 = pruned + restart)
#
# Per ablation pre-flight rules:
#   #1 cadence:    --snapshot-every 1
#   #6 resumable:  metrics.json check per cell
#   #10 qual pair: track-id-vs-frame timeline per cell, gap shaded

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

OUT_ROOT="results_m4/ablation_f"
DET_CSV="${OUT_ROOT}/occluded.csv"
RUNNER="./build/construction_perception/tracker_runner"
PYTHON="${PYTHON:-python3}"

mkdir -p "${OUT_ROOT}"

# -----------------------------------------------------------------------------
# Step 1 — synthesize occluded scenario
# -----------------------------------------------------------------------------
if [[ ! -f "${DET_CSV}" ]]; then
    echo "[ablation_f] generating occluded CSV"
    "${PYTHON}" scripts/synth_detections.py occluded \
        --frames 30 \
        --seed   42 \
        --out    "${DET_CSV}"
else
    echo "[ablation_f] reusing existing ${DET_CSV}"
fi

# -----------------------------------------------------------------------------
# Step 2 — max_misses sweep
# -----------------------------------------------------------------------------
MAX_MISS_VALUES=(1 3 10)

run_one() {
    local m="$1"
    local out_dir="${OUT_ROOT}/m_${m}"
    if [[ -f "${out_dir}/metrics.json" ]]; then
        echo "[ablation_f] ✅ max_misses=${m}: already done"
        return
    fi
    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"
    echo "[ablation_f] running max_misses=${m}"
    "${RUNNER}" \
        --detections     "${DET_CSV}" \
        --solver         greedy \
        --max-dist       3.0 \
        --max-misses     "${m}" \
        --min-hits       1 \
        --dt             0.1 \
        --process-noise  0.01 \
        --meas-noise     0.1 \
        --snapshot-every 1 \
        --out            "${out_dir}" \
        --verbose
}
for m in "${MAX_MISS_VALUES[@]}"; do run_one "${m}"; done

# -----------------------------------------------------------------------------
# Step 3 — track-id timeline per cell
# -----------------------------------------------------------------------------
run_dirs=()
labels=()
for m in "${MAX_MISS_VALUES[@]}"; do
    run_dirs+=("${OUT_ROOT}/m_${m}")
    labels+=("max_misses=${m}")
done

echo "[ablation_f] rendering max_misses sweep timeline"
"${PYTHON}" scripts/plot_max_misses_sweep.py \
    --runs       "${run_dirs[@]}" \
    --labels     "${labels[@]}" \
    --detections "${DET_CSV}" \
    --out        "${OUT_ROOT}/max_misses_sweep.png"

echo
echo "================ Ablation F summary ================"
printf "  %-14s  %-18s\n" "max_misses" "distinct_track_ids"
for m in "${MAX_MISS_VALUES[@]}"; do
    f="${OUT_ROOT}/m_${m}/metrics.json"
    if [[ -f "${f}" ]]; then
        dt=$(grep -E '"distinct_track_ids"' "${f}" | grep -oE '[0-9]+' | head -1)
        printf "  %-14s  %-18s\n" "${m}" "${dt:-?}"
    fi
done
echo "===================================================="
