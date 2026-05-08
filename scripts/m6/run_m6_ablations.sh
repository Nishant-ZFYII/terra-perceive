#!/bin/bash
# scripts/m6/run_m6_ablations.sh
#
# Resumable orchestrator for the P2-M6 ablations.
# Skips runs whose results_m6/<run>/metrics.json already exists.
#
# Usage:
#   scripts/m6/run_m6_ablations.sh safety        # m13 (local, fast)
#   scripts/m6/run_m6_ablations.sh traversability   # m12 (needs RELLIS lidar dir)
#   scripts/m6/run_m6_ablations.sh all
#
# Environment:
#   LIDAR_DIR  — path to a directory of RELLIS *.bin frames (m12 only).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SAFETY_BIN="$REPO_ROOT/build/construction_perception/safety_runner"
TRAV_BIN="$REPO_ROOT/build/construction_perception/traversability_runner"
RESULTS_ROOT="$REPO_ROOT/results_m6"
SCEN_DIR="$SCRIPT_DIR/scenarios"

PYTHON_BIN="${PYTHON_BIN:-/home/nishant/miniconda3/envs/terra-perceive/bin/python}"

mkdir -p "$RESULTS_ROOT"

run_safety() {
    "$PYTHON_BIN" "$SCRIPT_DIR/synth_safety_scenarios.py" --out "$SCEN_DIR"

    local scenarios=(head_on angled_20 occluded multi_worker far_pass edge_of_arc)
    for mode in kinematic cbf; do
        for scen in "${scenarios[@]}"; do
            local out_dir="$RESULTS_ROOT/cbf_${mode}/${scen}"
            local metrics="$out_dir/metrics.json"
            if [[ -f "$metrics" ]]; then
                echo "[skip] $metrics exists"
                continue
            fi
            mkdir -p "$out_dir"
            local extra=()
            if [[ "$mode" == "cbf" ]]; then
                extra=(--cbf-gamma "${CBF_GAMMA:-1.0}" --cbf-d-safe-min 0.5 --cbf-dt 0.1)
            fi
            echo "[run ] mode=$mode scen=$scen"
            "$SAFETY_BIN" --scenario "$SCEN_DIR/${scen}.csv" \
                          --safety-mode "$mode" \
                          --frames 100 \
                          --out "$out_dir" "${extra[@]}"
        done
    done

    "$PYTHON_BIN" "$SCRIPT_DIR/plot_velocity_profiles.py" \
        --kinematic-root "$RESULTS_ROOT/cbf_kinematic" \
        --cbf-root       "$RESULTS_ROOT/cbf_cbf" \
        --out            "$RESULTS_ROOT/figures/velocity_profiles.png"
    "$PYTHON_BIN" "$SCRIPT_DIR/plot_stopping_margin.py" \
        --kinematic-root "$RESULTS_ROOT/cbf_kinematic" \
        --cbf-root       "$RESULTS_ROOT/cbf_cbf" \
        --out            "$RESULTS_ROOT/figures/stopping_margin.png"
    "$PYTHON_BIN" "$SCRIPT_DIR/plot_intervention_timing.py" \
        --kinematic-root "$RESULTS_ROOT/cbf_kinematic" \
        --cbf-root       "$RESULTS_ROOT/cbf_cbf" \
        --out            "$RESULTS_ROOT/figures/intervention_timing.png"
}

run_traversability() {
    if [[ -z "${LIDAR_DIR:-}" ]]; then
        echo "[error] LIDAR_DIR env var not set; skipping m12 ablation."
        return 1
    fi
    if [[ ! -d "$LIDAR_DIR" ]]; then
        echo "[error] LIDAR_DIR=$LIDAR_DIR is not a directory."
        return 1
    fi

    local frames="${FRAMES:-2847}"
    for mode in heuristic probabilistic; do
        local out_dir="$RESULTS_ROOT/trav_${mode}"
        local metrics="$out_dir/metrics.json"
        if [[ -f "$metrics" ]]; then
            echo "[skip] $metrics exists"
            continue
        fi
        mkdir -p "$out_dir"
        echo "[run ] mode=$mode frames=$frames"
        "$TRAV_BIN" --lidar "$LIDAR_DIR" \
                    --frames "$frames" \
                    --confidence-mode "$mode" \
                    --sigma-0 0.01 --sigma-k 0.0001 \
                    --snapshot-every 50 \
                    --out "$out_dir"
    done

    "$PYTHON_BIN" "$SCRIPT_DIR/plot_sigma_r.py" \
        --out "$RESULTS_ROOT/figures/sigma_r.png"
    "$PYTHON_BIN" "$SCRIPT_DIR/plot_confidence_compare.py" \
        --heuristic-root     "$RESULTS_ROOT/trav_heuristic" \
        --probabilistic-root "$RESULTS_ROOT/trav_probabilistic" \
        --out-dir            "$RESULTS_ROOT/figures"
}

run_animations() {
    # Animations destination defaults to seagate; override with ANIM_OUT.
    local anim_out="${ANIM_OUT:-/media/nishant/SeeGayt2/terra_perceive/m6_animations}"
    mkdir -p "$anim_out" "$anim_out/scenarios"

    "$PYTHON_BIN" "$SCRIPT_DIR/animate_velocity_profiles.py" \
        --kinematic-root "$RESULTS_ROOT/cbf_kinematic" \
        --cbf-root       "$RESULTS_ROOT/cbf_cbf" \
        --out            "$anim_out/velocity_profiles.mp4" --fps 10

    "$PYTHON_BIN" "$SCRIPT_DIR/animate_scenario_bev.py" \
        --scenarios-dir  "$SCRIPT_DIR/scenarios" \
        --kinematic-root "$RESULTS_ROOT/cbf_kinematic" \
        --cbf-root       "$RESULTS_ROOT/cbf_cbf" \
        --out-dir        "$anim_out/scenarios" --fps 10

    # m12 BEV side-by-side requires per-frame snapshots. Skip if not present.
    local heur_pf="${HEUR_PERFRAME:-/media/nishant/SeeGayt2/terra_perceive/m6_perframe/trav_heuristic_perframe}"
    local prob_pf="${PROB_PERFRAME:-/media/nishant/SeeGayt2/terra_perceive/m6_perframe/trav_probabilistic_perframe}"
    if [[ -f "$heur_pf/metrics.json" && -f "$prob_pf/metrics.json" ]]; then
        "$PYTHON_BIN" "$SCRIPT_DIR/animate_confidence_bev.py" \
            --heuristic-root     "$heur_pf" \
            --probabilistic-root "$prob_pf" \
            --out                "$anim_out/confidence_bev.mp4" \
            --fps 30
    else
        echo "[skip] confidence_bev.mp4 — perframe runs not present at"
        echo "       $heur_pf and $prob_pf"
        echo "       Re-run traversability_runner with --snapshot-every 1 to those dirs."
    fi

    # Raw LiDAR BEV across the full sequence. Skip if the lidar dir is unset.
    local lidar_dir_anim="${LIDAR_DIR:-/media/nishant/SeeGayt2/terra_perceive/m4_perframe/extracted_frames}"
    if [[ -d "$lidar_dir_anim" ]]; then
        "$PYTHON_BIN" "$SCRIPT_DIR/animate_rellis_lidar.py" \
            --lidar-dir "$lidar_dir_anim" \
            --out       "$anim_out/rellis_lidar_bev.mp4" \
            --fps 30
    else
        echo "[skip] rellis_lidar_bev.mp4 — LIDAR_DIR not set or not a directory."
    fi
}

case "${1:-all}" in
    safety) run_safety ;;
    traversability) run_traversability ;;
    animations) run_animations ;;
    all) run_safety; run_traversability || true; run_animations || true ;;
    *) echo "Usage: $0 {safety|traversability|animations|all}"; exit 1 ;;
esac
